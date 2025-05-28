# Copyright 2025 The RealisDance-DiT Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import copy
import html
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import regex as re
import torch
import torch.nn.functional as F
from einops import rearrange
from transformers import AutoTokenizer, CLIPVisionModel, UMT5EncoderModel
from transformers.image_utils import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD

from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.loaders import WanLoraLoaderMixin
from diffusers.models import AutoencoderKLWan
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.wan.pipeline_output import WanPipelineOutput
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import is_ftfy_available, is_torch_xla_available, logging, replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor

from ..models.rd_dit import RealisDanceDiT

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

if is_ftfy_available():
    import ftfy

EXAMPLE_DOC_STRING = """
    Examples:
        ```python
        >>> import torch
        >>> from diffusers import AutoencoderKLWan
        >>> from transformers import CLIPVisionModel
        >>> from src.pipelines.rd_dit_pipeline import RealisDanceDiTPipeline

        >>> model_id = "theFoxofSky/RealisDance-DiT"
        >>> image_encoder = CLIPVisionModel.from_pretrained(
        ...     model_id, subfolder="image_encoder", torch_dtype=torch.float32
        ... )
        >>> vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
        >>> pipe = RealisDanceDiTPipeline.from_pretrained(
        ...     model_id, vae=vae, image_encoder=image_encoder, torch_dtype=torch.bfloat16
        ... )
        >>> pipe.enable_model_cpu_offload()
        
        >>> # prepare ref_image, smpl, hamer, all of them are in shape B C F H W, in range [-1, 1]
        
        >>> max_res = 768*768
        >>> output = pipe(
        ...     image=ref_image,
        ...     smpl=smpl,
        ...     hamer=hamer,
        ...     prompt=prompt,
        ...     max_resolution=max_res,
        ... ).frames[0]
        >>> export_to_video(output, "output.mp4", fps=16)
        ```
"""


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def prompt_clean(text):
    text = whitespace_clean(basic_clean(text))
    return text


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents
def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


class RealisDanceDiTPipeline(DiffusionPipeline, WanLoraLoaderMixin):
    r"""
    Pipeline for RealisDance-DiT built upon Wan I2V.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        tokenizer ([`T5Tokenizer`]):
            Tokenizer from [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5Tokenizer),
            specifically the [google/umt5-xxl](https://huggingface.co/google/umt5-xxl) variant.
        text_encoder ([`T5EncoderModel`]):
            [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5EncoderModel), specifically
            the [google/umt5-xxl](https://huggingface.co/google/umt5-xxl) variant.
        image_encoder ([`CLIPVisionModel`]):
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPVisionModel), specifically
            the
            [clip-vit-huge-patch14](https://github.com/mlfoundations/open_clip/blob/main/docs/PRETRAINED.md#vit-h14-xlm-roberta-large)
            variant.
        transformer ([`RealisDanceDiT`]):
            Conditional Transformer to denoise the input latents.
        scheduler ([`UniPCMultistepScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
        vae ([`AutoencoderKLWan`]):
            Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.
    """

    model_cpu_offload_seq = "text_encoder->image_encoder->transformer->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        text_encoder: UMT5EncoderModel,
        image_encoder: CLIPVisionModel,
        transformer: RealisDanceDiT,
        vae: AutoencoderKLWan,
        scheduler: FlowMatchEulerDiscreteScheduler,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            image_encoder=image_encoder,
            transformer=transformer,
            scheduler=scheduler,
        )

        self.vae_scale_factor_temporal = 2 ** sum(self.vae.temperal_downsample) if getattr(self, "vae", None) else 4
        self.vae_scale_factor_spatial = 2 ** len(self.vae.temperal_downsample) if getattr(self, "vae", None) else 8
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

    def process_shape(self, video: torch.Tensor, tgt_h: int, tgt_w: int, resize_type: str) -> torch.Tensor:
        num_frame, ori_h, ori_w = video.shape[-3:]
        if resize_type == "max_resolution":
            ratio = ((tgt_h * tgt_w) / (ori_h * ori_w)) ** 0.5
            scale_factor_h = self.vae_scale_factor_spatial * self.transformer.config.patch_size[1]
            h = round(ori_h * ratio / scale_factor_h) * scale_factor_h
            scale_factor_w = self.vae_scale_factor_spatial * self.transformer.config.patch_size[2]
            w = round(ori_w * ratio / scale_factor_w) * scale_factor_w
            i = j = 0
            tgt_h = h
            tgt_w = w
        elif resize_type == "resize_crop":
            ratio_h, ratio_w = tgt_h / ori_h, tgt_w / ori_w
            if ratio_h > ratio_w:
                h, w = tgt_h, round(ori_w * ratio_h)
                i = 0
                j = int(round(w - tgt_w) / 2.0)
            else:
                h, w = round(ori_h * ratio_w), tgt_w
                i = int(round(h - tgt_h) / 2.0)
                j = 0
            assert i + tgt_h <= h and j + tgt_w <= w
        else:
            raise ValueError(f"Unsupported `resize_type`: {resize_type}")

        video = rearrange(video, "b c f h w -> (b f) c h w")
        video = F.interpolate(video, size=(h, w), mode='bicubic', align_corners=False)
        video = rearrange(video, "(b f) c h w -> b c f h w", f=num_frame)
        video = video[..., i:i + tgt_h, j:j + tgt_w]
        return video

    def _image_trans_for_clip(self, image: torch.Tensor) -> torch.Tensor:
        clip_image = image.squeeze(2) * 0.5 + 0.5  # B C 1 H W -> B C H W; [-1, 1] -> [0, 1]
        clip_image = F.interpolate(clip_image, size=(224, 224), mode='bicubic', align_corners=False)
        clip_mean = (
            torch.tensor(OPENAI_CLIP_MEAN)
            .reshape(1, clip_image.shape[1], 1, 1)
            .to(device=clip_image.device, dtype=clip_image.dtype)
        )
        clip_std = (
            torch.tensor(OPENAI_CLIP_STD)
            .reshape(1, clip_image.shape[1], 1, 1)
            .to(device=clip_image.device, dtype=clip_image.dtype)
        )
        clip_image = (clip_image - clip_mean) / clip_std
        return clip_image

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt = [prompt_clean(u) for u in prompt]
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()

        prompt_embeds = self.text_encoder(text_input_ids.to(device), mask.to(device)).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in prompt_embeds], dim=0
        )

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        return prompt_embeds

    def encode_image(
        self,
        image: torch.Tensor,
        device: Optional[torch.device] = None,
    ):
        device = device or self._execution_device
        image = self._image_trans_for_clip(image).to(device=device)
        image_embeds = self.image_encoder(pixel_values=image, output_hidden_states=True)
        return image_embeds.hidden_states[-2]

    # Copied from diffusers.pipelines.wan.pipeline_wan.WanPipeline.encode_prompt
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            do_classifier_free_guidance (`bool`, *optional*, defaults to `True`):
                Whether to use classifier free guidance or not.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                Number of videos that should be generated per prompt. torch device to place the resulting embeddings on
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            device: (`torch.device`, *optional*):
                torch device
            dtype: (`torch.dtype`, *optional*):
                torch dtype
        """
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_prompt_embeds = self._get_t5_prompt_embeds(
                prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        return prompt_embeds, negative_prompt_embeds

    def check_inputs(
        self,
        prompt,
        negative_prompt,
        image,
        smpl,
        hamer,
        height,
        width,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
    ):
        if image is not None and not isinstance(image, torch.Tensor):
            raise ValueError(f"`image` has to be of type `torch.Tensor` but is {type(image)}")
        if smpl is not None and not isinstance(smpl, torch.Tensor):
            raise ValueError(f"`smpl` has to be of type `torch.Tensor` but is {type(smpl)}")
        if hamer is not None and not isinstance(hamer, torch.Tensor):
            raise ValueError(f"`hamer` has to be of type `torch.Tensor` but is {type(hamer)}")
        if height is not None and width is not None and (height % 16 != 0 or width % 16 != 0):
            raise ValueError(f"`height` and `width` have to be divisible by 16 but are {height} and {width}.")

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`: {negative_prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        elif negative_prompt is not None and (
            not isinstance(negative_prompt, str) and not isinstance(negative_prompt, list)
        ):
            raise ValueError(f"`negative_prompt` has to be of type `str` or `list` but is {type(negative_prompt)}")

    def prepare_latents(
        self,
        image: torch.Tensor,
        smpl: torch.Tensor,
        hamer: torch.Tensor,
        batch_size: int,
        num_channels_latents: int = 16,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        latent_height = height // self.vae_scale_factor_spatial
        latent_width = width // self.vae_scale_factor_spatial

        shape = (batch_size, num_channels_latents, num_latent_frames, latent_height, latent_width)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device=device, dtype=dtype)

        image = image.to(device=device, dtype=dtype)
        video_condition = torch.zeros(
            image.shape[0], image.shape[1], num_frames, height, width
        )
        video_condition = video_condition.to(device=device, dtype=dtype)
        smpl = smpl.to(device=device, dtype=dtype)
        hamer = hamer.to(device=device, dtype=dtype)

        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )

        if isinstance(generator, list):
            def vae_encode_1(x):
                latent_x = [
                    retrieve_latents(self.vae.encode(x), sample_mode="argmax") for _ in generator
                ]
                return (torch.cat(latent_x) - latents_mean) * latents_std

            latent_condition = vae_encode_1(video_condition)
            latent_smpl = vae_encode_1(smpl)
            latent_hamer = vae_encode_1(hamer)
            latent_ref = vae_encode_1(image)
            if self.do_classifier_free_guidance:
                latent_null_ref = vae_encode_1(torch.zeros_like(image))
            else:
                latent_null_ref = None
        else:
            def vae_encode_2(x):
                latent_x = retrieve_latents(self.vae.encode(x), sample_mode="argmax")
                return (latent_x.repeat(batch_size, 1, 1, 1, 1) - latents_mean) * latents_std

            latent_condition = vae_encode_2(video_condition)
            latent_smpl = vae_encode_2(smpl)
            latent_hamer = vae_encode_2(hamer)
            latent_ref = vae_encode_2(image)
            if self.do_classifier_free_guidance:
                latent_null_ref = vae_encode_2(torch.zeros_like(image))
            else:
                latent_null_ref = None

        mask_lat_size = torch.zeros(batch_size, 4, num_latent_frames, latent_height, latent_width)
        mask_lat_size = mask_lat_size.to(latent_condition.device)
        latent_i2v_condition = torch.cat(
            [mask_lat_size, latent_condition], dim=1
        )

        latent_pose = torch.cat((latent_smpl, latent_hamer), dim=1)

        return latents, latent_i2v_condition, latent_pose, latent_ref, latent_null_ref

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def interrupt(self):
        return self._interrupt

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        image: torch.Tensor,
        smpl: torch.Tensor,
        hamer: torch.Tensor,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        max_resolution: int = 768 * 768,
        num_frames: int = 81,
        num_inference_steps: int = 40,
        guidance_scale: float = 2.0,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "np",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        enable_teacache: bool = False,
        teacache_thresh: float = 0.2,
        use_timestep_proj: bool = True,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            image (`torch.Tensor`):
                The input image to condition the generation on. Must be a `torch.Tensor`
                with shape B C 1 H W and range [-1, 1].
            smpl (`torch.Tensor`):
                The input smpl video to condition the generation on. Must be a `torch.Tensor`
                with shape B C 1 H W and range [-1, 1].
            hamer (`torch.Tensor`):
                The input hamer video to condition the generation on. Must be a `torch.Tensor`
                with shape B C 1 H W and range [-1, 1].
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            height (`int`, defaults to `480`):
                The height of the generated video.
            width (`int`, defaults to `832`):
                The width of the generated video.
            num_frames (`int`, defaults to `81`):
                The number of frames in the generated video.
            num_inference_steps (`int`, defaults to `50`):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, defaults to `5.0`):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `negative_prompt` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`WanPipelineOutput`] instead of a plain tuple.
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
                A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
                each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
                DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
                list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int`, *optional*, defaults to `512`):
                The maximum sequence length of the prompt.
            enable_teacache (`bool`, *optional*, defaults to False):
                Whether to use teacache to accelerate inference. Note that enabling teacache will hurt generation
                quality.
            teacache_thresh (`float`, *optional*, defaults to 0.2):
                Threshold for teacache. Higher speedup will cause to worse quality.
            use_timestep_proj (`bool`, *optional*, defaults to True):
                Whether to use timestep_proj or temb.
        Examples:

        Returns:
            [`~WanPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`WanPipelineOutput`] is returned, otherwise a `tuple` is returned where
                the first element is a list with the generated images and the second element is a list of `bool`s
                indicating whether the corresponding generated image contains "not-safe-for-work" (nsfw) content.
        """

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            negative_prompt,
            image,
            smpl,
            hamer,
            height,
            width,
            prompt_embeds,
            negative_prompt_embeds,
            callback_on_step_end_tensor_inputs,
        )

        if height is None or width is None:
            smpl_height, smpl_width = smpl.shape[-2:]
            ratio = (max_resolution / (smpl_height * smpl_width)) ** 0.5
            scale_factor_h = self.vae_scale_factor_spatial * self.transformer.config.patch_size[1]
            height = round(smpl_height * ratio / scale_factor_h) * scale_factor_h
            scale_factor_w = self.vae_scale_factor_spatial * self.transformer.config.patch_size[2]
            width = round(smpl_width * ratio / scale_factor_w) * scale_factor_w

        if num_frames % self.vae_scale_factor_temporal != 1:
            logger.warning(
                f"`num_frames - 1` has to be divisible by {self.vae_scale_factor_temporal}. Rounding to the nearest number."
            )
            num_frames = num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
        num_frames = max(num_frames, 1)

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        device = self._execution_device

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )

        # Encode image embedding
        transformer_dtype = self.transformer.dtype
        prompt_embeds = prompt_embeds.to(transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

        image_embeds = self.encode_image(image, device)
        image_embeds = image_embeds.repeat(batch_size, 1, 1)
        image_embeds = image_embeds.to(transformer_dtype)
        if self.do_classifier_free_guidance:
            null_image_embeds = self.encode_image(torch.zeros_like(image), device)
            null_image_embeds = null_image_embeds.repeat(batch_size, 1, 1)
            null_image_embeds = null_image_embeds.to(transformer_dtype)
        else:
            null_image_embeds = None

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.vae.config.z_dim
        image = self.process_shape(image, height, width, resize_type="max_resolution").to(device, dtype=torch.float32)
        smpl = self.process_shape(smpl, height, width, resize_type="resize_crop").to(device, dtype=torch.float32)
        hamer = self.process_shape(hamer, height, width, resize_type="resize_crop").to(device, dtype=torch.float32)
        latents, i2v_condition, pose_condition, ref_condition, null_ref_condition = self.prepare_latents(
            image,
            smpl,
            hamer,
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            height,
            width,
            num_frames,
            torch.float32,
            device,
            generator,
            latents,
        )
        pose_condition = pose_condition.to(transformer_dtype)
        ref_condition = ref_condition.to(transformer_dtype)
        null_ref_condition = null_ref_condition.to(transformer_dtype)

        # 6. TeaCache settings
        if enable_teacache:
            teacache_kwargs = {
                "teacache_thresh": teacache_thresh,
                "accumulated_rel_l1_distance": 0,
                "previous_e0": None,
                "previous_residual": None,
                "use_timestep_proj": use_timestep_proj,
                "coefficients": [
                    8.10705460e+03, 2.13393892e+03, -3.72934672e+02, 1.66203073e+01, -4.17769401e-02
                ] if use_timestep_proj else[
                    -114.36346466, 65.26524496, -18.82220707, 4.91518089, -0.23412683
                ],
                "ret_steps": 5 if use_timestep_proj else 1,
                "cutoff_steps": num_inference_steps
            }
            if self.do_classifier_free_guidance:
                teacache_kwargs_uncond = copy.deepcopy(teacache_kwargs)
        else:
            teacache_kwargs = teacache_kwargs_uncond = None

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t
                latent_model_input = torch.cat([latents, i2v_condition], dim=1).to(transformer_dtype)
                timestep = t.expand(latents.shape[0])

                noise_pred, teacache_kwargs = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    encoder_hidden_states_image=image_embeds,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                    add_cond=pose_condition,
                    attn_cond=ref_condition,
                    enable_teacache=enable_teacache,
                    current_step=i,
                    teacache_kwargs=teacache_kwargs,
                )

                if self.do_classifier_free_guidance:
                    noise_uncond, teacache_kwargs_uncond = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=negative_prompt_embeds,
                        encoder_hidden_states_image=null_image_embeds,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                        add_cond=pose_condition,
                        attn_cond=null_ref_condition,
                        enable_teacache=enable_teacache,
                        current_step=i,
                        teacache_kwargs=teacache_kwargs_uncond,
                    )
                    noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        self._current_timestep = None

        if not output_type == "latent":
            latents = latents.to(self.vae.dtype)
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = (
                1.0 / torch.tensor(self.vae.config.latents_std)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents = latents / latents_std + latents_mean
            video = self.vae.decode(latents, return_dict=False)[0]
            video = self.video_processor.postprocess_video(video, output_type=output_type)
        else:
            video = latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return WanPipelineOutput(frames=video)
