import inspect
from typing import Callable, List, Optional, Union
from dataclasses import dataclass

import numpy as np
import torch

from diffusers.utils import is_accelerate_available
from diffusers.utils.torch_utils import is_compiled_module
from packaging import version

from transformers import CLIPVisionModelWithProjection

from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL, AutoencoderKLTemporalDecoder
from diffusers.pipelines import DiffusionPipeline
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.utils import deprecate, logging, BaseOutput

from einops import rearrange

from src.models.rd_unet import RealisDanceUnet
from src.pipelines.context import get_context_scheduler
from src.utils.util import color_restore


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class RealisDancePipelineOutput(BaseOutput):
    videos: Union[torch.Tensor, np.ndarray]


class RealisDancePipeline(DiffusionPipeline):
    _optional_components = []

    def __init__(
        self,
        vae: Union[AutoencoderKL, AutoencoderKLTemporalDecoder],
        image_encoder: CLIPVisionModelWithProjection,
        unet: RealisDanceUnet,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
    ):
        super().__init__()

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        is_unet_sample_size_less_64 = hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely. If your checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)

        self.register_modules(
            vae=vae,
            image_encoder=image_encoder,
            unet=unet,
            scheduler=scheduler,
        )

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_finetune = None

    def enable_vae_slicing(self):
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        self.vae.disable_slicing()

    def enable_sequential_cpu_offload(self, gpu_id=0):
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")

        for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae]:
            if cpu_offloaded_model is not None:
                cpu_offload(cpu_offloaded_model, device)

    @property
    def _execution_device(self):
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def _encode_prompt(self, ref_image, ref_image_clip):
        with torch.no_grad():
            ref_latents = self.vae.encode(ref_image).latent_dist
            ref_latents = ref_latents.mode()
            ref_latents = ref_latents * self.vae.config.scaling_factor

            clip_latents = self.image_encoder(ref_image_clip).last_hidden_state

        return ref_latents, clip_latents

    def decode_latents(self, latents, decode_chunk_size=14):
        latents = 1 / self.vae.config.scaling_factor * latents
        if not self.image_finetune:
            video_length = latents.shape[2]
            latents = rearrange(latents, "b c f h w -> (b f) c h w")
            decode_chunk_size = video_length

        forward_vae_fn = self.vae._orig_mod.forward if is_compiled_module(self.vae) else self.vae.forward
        accepts_num_frames = "num_frames" in set(inspect.signature(forward_vae_fn).parameters.keys())

        # decode decode_chunk_size frames at a time to avoid OOM
        video = []
        for frame_idx in range(0, latents.shape[0], decode_chunk_size):
            num_frames_in = latents[frame_idx:frame_idx+decode_chunk_size].shape[0]
            decode_kwargs = {}
            if accepts_num_frames:
                # we only pass num_frames_in if it's expected
                decode_kwargs["num_frames"] = num_frames_in

            video.append(self.vae.decode(latents[frame_idx:frame_idx+decode_chunk_size], **decode_kwargs).sample)
        video = torch.cat(video)
        if not self.image_finetune:
            video = rearrange(video, "(b f) c h w -> b c f h w", f=video_length)
        video = (video / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        video = video.cpu().float().numpy()
        return video

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(self, pose,  hamer, smpl, ref_image, ref_image_clip, height, width, callback_steps):
        if not isinstance(pose, torch.Tensor):
            raise ValueError(f"`pose` has to be of type `torch.Tensor` but is {type(pose)}")
        if not isinstance(hamer, torch.Tensor):
            raise ValueError(f"`hamer` has to be of type `torch.Tensor` but is {type(hamer)}")
        if not isinstance(smpl, torch.Tensor):
            raise ValueError(f"`smpl` has to be of type `torch.Tensor` but is {type(smpl)}")
        if not isinstance(ref_image, torch.Tensor):
            raise ValueError(f"`ref_image` has to be of type `torch.Tensor` but is {type(ref_image)}")
        if not isinstance(ref_image_clip, torch.Tensor):
            raise ValueError(f"`ref_image_clip` has to be of type `torch.Tensor` but is {type(ref_image_clip)}")

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

    def prepare_latents(self, batch_size, num_channels_latents, video_length, height, width, dtype, device,
                        generator, latents=None):
        if self.image_finetune:
            shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width //
                     self.vae_scale_factor)
        else:
            shape = (batch_size, num_channels_latents, video_length,
                     height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        if latents is None:
            rand_device = "cpu" if device.type == "mps" else device

            if isinstance(generator, list):
                shape = (1,) + shape[1:]
                latents = [
                    torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype)
                    for i in range(batch_size)
                ]
                latents = torch.cat(latents, dim=0).contiguous().to(device)
            else:
                latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype).to(device)
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    @torch.no_grad()
    def __call__(
        self,
        pose: torch.FloatTensor,
        hamer: torch.FloatTensor,
        smpl: torch.FloatTensor,
        ref_image: torch.FloatTensor,
        ref_image_clip: torch.FloatTensor,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "tensor",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        context_frames: int = 16,
        context_stride: int = 1,
        context_overlap: int = 4,
        context_batch_size: int = 1,
        context_schedule: str = "uniform",
        fake_uncond: bool = True,
        do_color_restore: bool = True,
        **kwargs,
    ):
        # TODO: support multiple images per prompt
        assert num_images_per_prompt == 1, "not support multiple images per prompt yet"

        # Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # Check inputs. Raise error if not correct
        self.check_inputs(pose, hamer, smpl, ref_image, ref_image_clip, height, width, callback_steps)

        # Define call parameters
        batch_size = ref_image_clip.shape[0]
        if not self.image_finetune:
            video_length = pose.shape[2]
        else:
            video_length = -1

        device = self._execution_device

        # Encode input prompt
        ref_latents, clip_latents = self._encode_prompt(ref_image, ref_image_clip)
        if guidance_scale > 1.0:
            if fake_uncond:
                uncond_clip_latents = torch.zeros_like(clip_latents)
            else:
                uncond_clip_latents = self.image_encoder(torch.zeros_like(ref_image_clip)).last_hidden_state

        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # Prepare latent variables
        num_channels_latents = self.unet.in_channels
        latents = self.prepare_latents(
            batch_size,
            num_channels_latents,
            video_length,
            height,
            width,
            clip_latents.dtype,
            device,
            generator,
            latents,
        )
        latents_dtype = latents.dtype

        # Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        if not self.image_finetune:
            context_scheduler = get_context_scheduler(context_schedule)

        # Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = self.scheduler.scale_model_input(latents, t)
                if self.image_finetune:
                    noise_pred = self.unet(
                        sample=latent_model_input, ref_sample=ref_latents,
                        pose=pose, hamer=hamer, smpl=smpl, timestep=t,
                        encoder_hidden_states=clip_latents, drop_reference=False,
                    ).sample.to(dtype=latents_dtype)
                    if guidance_scale > 1.0:
                        noise_uncond = self.unet(
                            sample=latent_model_input, ref_sample=None,
                            pose=pose, hamer=hamer, smpl=smpl, timestep=t,
                            encoder_hidden_states=uncond_clip_latents, drop_reference=True,
                        ).sample.to(dtype=latents_dtype)
                        noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)
                else:
                    context_queue = list(context_scheduler(
                        i, num_inference_steps, latents.shape[2], context_frames, context_stride, context_overlap,
                    ))
                    counter = torch.zeros_like(latent_model_input)
                    noise_pred = torch.zeros_like(latent_model_input)
                    if guidance_scale > 1.0:
                        noise_uncond = torch.zeros_like(latent_model_input)
                    for c in context_queue:
                        partial_latent_model_input = latent_model_input[:, :, c]
                        partial_pose = pose[:, :, c]
                        partial_hamer = hamer[:, :, c]
                        partial_smpl = smpl[:, :, c]
                        # predict the noise residual
                        noise_pred[:, :, c] += self.unet(
                            sample=partial_latent_model_input, ref_sample=ref_latents,
                            pose=partial_pose, hamer=partial_hamer, smpl=partial_smpl, timestep=t,
                            encoder_hidden_states=clip_latents, drop_reference=False,
                        ).sample.to(dtype=latents_dtype)
                        counter[:, :, c] += 1

                        if guidance_scale > 1.0:
                            noise_uncond[:, :, c] += self.unet(
                                sample=partial_latent_model_input, ref_sample=None,
                                pose=partial_pose,  hamer=partial_hamer, smpl=partial_smpl, timestep=t,
                                encoder_hidden_states=uncond_clip_latents, drop_reference=True,
                            ).sample.to(dtype=latents_dtype)

                    noise_pred /= counter
                    if guidance_scale > 1.0:
                        noise_uncond /= counter
                        noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        # Post-processing
        video = self.decode_latents(latents)
        if do_color_restore:
            ref_image = (ref_image / 2 + 0.5).clamp(0, 1)
            ref_image = ref_image.cpu().float().numpy()  # b c h w
            mean_ref = ref_image.mean(axis=(-1, -2), keepdims=True)
            std_ref = ref_image.std(axis=(-1, -2), keepdims=True)

            if len(video.shape) == 4:
                mean_video = video.mean(axis=(-1, -2), keepdims=True)
                std_video = video.std(axis=(-1, -2), keepdims=True)
            else:
                mean_video = video.mean(axis=(-1, -2, -3), keepdims=True)
                std_video = video.std(axis=(-1, -2, -3), keepdims=True)
                mean_ref = mean_ref[:, :, None, :, :]
                std_ref = std_ref[:, :, None, :, :]

            std_video[std_video < 1e-10] = 1e-10
            video = (video - mean_video) * std_ref / std_video + mean_ref
            video = np.clip(video, 0, 1)

        # Convert to tensor
        if output_type == "tensor":
            video = torch.from_numpy(video)

        if not return_dict:
            return video

        return RealisDancePipelineOutput(videos=video)
