import torch
import torch.nn as nn

from einops import rearrange
from omegaconf import OmegaConf
from typing import Union, Optional

from diffusers.utils.import_utils import is_xformers_available
from diffusers.models.controlnet import zero_module
from diffusers.models import UNet2DConditionModel

from src.models.hack_unet2d import HackUNet2DConditionModel
from src.models.hack_unet3d import HackUNet3DConditionModel
from src.models.reference_net_attention import ReferenceNetAttention
from src.models.orig_attention import FeedForward
from src.models.pose_guider import PoseGuider


class SimpleResFF(nn.Module):
    def __init__(
            self,
            dim: int,
            dim_out: Optional[int] = None,
            mult: int = 2,
            dropout: float = 0.0,
            activation_fn: str = "geglu",):
        super(SimpleResFF, self).__init__()

        self.identity = nn.Linear(dim, dim_out)
        self.ff = FeedForward(dim, dim_out, mult, dropout, activation_fn)
        self.post_norm = nn.LayerNorm(dim_out)
        self.post_proj = zero_module(nn.Linear(dim_out, dim_out))

    def forward(self, x):
        return self.post_proj(self.post_norm(self.identity(x) + self.ff(x)))


class RealisDanceUnet(nn.Module):
    def __init__(
        self,
        pretrained_model_path,
        unet_additional_kwargs=None,
        pose_guider_kwargs=None,
        clip_projector_kwargs=None,
        fix_ref_t=False,
        image_finetune=False,
        fusion_blocks="full",
    ):
        super(RealisDanceUnet, self).__init__()

        self.image_finetune = image_finetune
        self.fix_ref_t = fix_ref_t

        self.unet_ref = UNet2DConditionModel.from_pretrained(pretrained_model_path, subfolder="unet")
        pose_guider_kwargs_dict = OmegaConf.to_container(pose_guider_kwargs.get("args"))

        if self.image_finetune:
            self.unet_main = HackUNet2DConditionModel.from_pretrained(pretrained_model_path, subfolder="unet")
        else:
            unet_additional_kwargs_dict = OmegaConf.to_container(unet_additional_kwargs)
            pose_guider_kwargs_dict["motion_module_type"] = unet_additional_kwargs_dict["motion_module_type"]
            pose_guider_kwargs_dict["motion_module_kwargs"] = unet_additional_kwargs_dict["motion_module_kwargs"]
            self.unet_main = HackUNet3DConditionModel.from_pretrained_2d(
                pretrained_model_path, subfolder="unet",
                unet_additional_kwargs=unet_additional_kwargs_dict)

        self.pose_guider = PoseGuider(
            image_finetune=image_finetune,
            num_conds=3,
            **pose_guider_kwargs_dict)

        self.clip_projector = SimpleResFF(
            clip_projector_kwargs.get("in_features"),
            clip_projector_kwargs.get("out_features"),
        )

        self.reference_writer = ReferenceNetAttention(
            self.unet_ref, mode='write', fusion_blocks=fusion_blocks, is_image=image_finetune)
        self.reference_reader = ReferenceNetAttention(
            self.unet_main, mode='read', fusion_blocks=fusion_blocks, is_image=image_finetune)

    def enable_xformers_memory_efficient_attention(self):
        if is_xformers_available():
            self.unet_ref.enable_xformers_memory_efficient_attention()
            self.unet_main.enable_xformers_memory_efficient_attention()
        else:
            print("xformers is not available, therefore not enabled")

    def enable_gradient_checkpointing(self):
        self.unet_ref.enable_gradient_checkpointing()
        self.unet_main.enable_gradient_checkpointing()

    @property
    def in_channels(self):
        return self.unet_main.config.in_channels

    @property
    def config(self):
        return self.unet_main.config

    @property
    def dtype(self):
        return self.unet_main.dtype

    @property
    def device(self):
        return self.unet_main.device

    def forward(
            self,
            sample: torch.FloatTensor,
            ref_sample: torch.FloatTensor,
            pose: torch.FloatTensor,
            hamer: torch.FloatTensor,
            smpl: torch.FloatTensor,
            timestep: Union[torch.Tensor, float, int],
            encoder_hidden_states: torch.Tensor,
            drop_reference: bool = False,
            return_dict: bool = True,
    ):

        self.reference_reader.clear()
        self.reference_writer.clear()

        encoder_hidden_states = self.clip_projector(encoder_hidden_states)

        if not drop_reference:
            ref_timestep = torch.zeros_like(timestep) if self.fix_ref_t else timestep
            self.unet_ref(
                ref_sample,
                ref_timestep,
                encoder_hidden_states=encoder_hidden_states,  # clip_latents
            )
            self.reference_reader.update(self.reference_writer)

        if self.image_finetune:
            pose_emb = self.pose_guider(dwpose=pose, hamer=hamer, smpl=smpl)
        else:
            video_length = sample.shape[2]
            pose = rearrange(pose, "b c f h w -> (b f) c h w")
            hamer = rearrange(hamer, "b c f h w -> (b f) c h w")
            smpl = rearrange(smpl, "b c f h w -> (b f) c h w")
            pose_emb = self.pose_guider(dwpose=pose, hamer=hamer, smpl=smpl, video_length=video_length,
                                        temb=timestep, encoder_hidden_states=encoder_hidden_states)
            pose_emb = [rearrange(pe, "(b f) c h w -> b c f h w", f=video_length) for pe in pose_emb]

        model_pred = self.unet_main(
            sample,
            timestep,
            latent_pose=pose_emb,
            encoder_hidden_states=encoder_hidden_states,  # clip_latents
            return_dict=return_dict
        )

        self.reference_reader.clear()
        self.reference_writer.clear()

        return model_pred

    def set_trainable_parameters(self, trainable_modules):
        self.requires_grad_(False)
        for param_name, param in self.named_parameters():
            for trainable_module_name in trainable_modules:
                if trainable_module_name in param_name:
                    param.requires_grad = True
                    break

