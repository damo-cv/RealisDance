import torch
import torch.nn as nn

from typing import Tuple, Optional

from einops import rearrange
from torch.nn import functional as F

from diffusers.models.controlnet import zero_module
from src.models.motion_module import get_motion_module


class MultiCondBackbone(nn.Module):
    def __init__(
        self,
        conditioning_channels: int = 3,
        block_out_channels: Tuple[int, ...] = (16, 32, 96, 256),
        num_conds: int = 3,
    ):
        super().__init__()

        self.conv_in = nn.Conv2d(
            conditioning_channels * num_conds,
            block_out_channels[0] * num_conds,
            kernel_size=(3, 3),
            groups=num_conds,
            padding=1)

        self.blocks = nn.ModuleList([])

        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            self.blocks.append(
                nn.Conv2d(
                    channel_in * num_conds,
                    channel_in * num_conds,
                    kernel_size=(3, 3),
                    groups=num_conds,
                    padding=1))
            self.blocks.append(
                nn.Conv2d(
                    channel_in * num_conds,
                    channel_out * num_conds,
                    kernel_size=(3, 3),
                    groups=num_conds,
                    padding=1,
                    stride=(2, 2)))

    def forward(self, conditioning):
        embedding = self.conv_in(conditioning)
        embedding = F.silu(embedding)

        for block in self.blocks:
            embedding = block(embedding)
            embedding = F.silu(embedding)

        return embedding


class GateModule(nn.Module):
    def __init__(
            self,
            channels: int,
            num_conds: int = 3
    ):
        super(GateModule, self).__init__()
        self.channels = channels
        self.num_conds = num_conds
        self.gate_layer = nn.Sequential(
            nn.Conv2d(self.channels * self.num_conds, self.channels//2, kernel_size=(3, 3), padding=1),
            nn.SiLU(),
            nn.Conv2d(self.channels//2, self.num_conds, kernel_size=(7, 7), padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, H, W = x.shape
        gate_weight = self.gate_layer(x).reshape(B, self.num_conds, 1, H, W)
        x = x.reshape(B, self.num_conds, -1, H, W)
        x = x * gate_weight
        x = x.reshape(B, C, H, W)
        return x


class PoseGuider(nn.Module):
    def __init__(
            self,
            conditioning_channels: int = 3,
            backbone_channels: Tuple[int, ...] = (16, 32, 96, 256),
            out_channels: Tuple[int, ...] = (320, 320, 640, 1280, 1280),
            image_finetune: bool = False,
            motion_module_type: Optional = None,
            motion_module_kwargs: Optional = None,
            num_conds: int = 3,
            ):
        super(PoseGuider, self).__init__()
        self.conditioning_channels = conditioning_channels
        self.backbone_channels = backbone_channels
        self.out_channels = out_channels
        self.image_finetune = image_finetune
        self.num_conds = num_conds

        self.backbone = MultiCondBackbone(
            conditioning_channels=self.conditioning_channels,
            block_out_channels=self.backbone_channels,
            num_conds=self.num_conds)

        self.gate_module = GateModule(channels=backbone_channels[-1], num_conds=num_conds)

        self.blocks_0 = nn.Sequential(
            nn.Conv2d(backbone_channels[-1]*self.num_conds, out_channels[0], kernel_size=(1, 1)),
            nn.SiLU()
        )
        self.block_0_out_proj = zero_module(nn.Conv2d(out_channels[0], out_channels[0], kernel_size=(1, 1)))

        motion_modules = []
        if not self.image_finetune:
            motion_modules.append(
                get_motion_module(
                    in_channels=out_channels[0],
                    motion_module_type=motion_module_type,
                    motion_module_kwargs=motion_module_kwargs,
                )
            )

        for i in range(1, len(out_channels)):
            self.register_module(f'blocks_{i}', nn.Sequential(
                nn.Conv2d(out_channels[i - 1], out_channels[i], kernel_size=(3, 3), padding=1,
                          stride=(2, 2) if i < len(out_channels) - 1 else (1, 1)),
                nn.SiLU(),
                nn.Conv2d(out_channels[i], out_channels[i], kernel_size=(3, 3), padding=1, stride=(1, 1)),
                nn.SiLU(),
            ))
            self.register_module(f'block_{i}_out_proj',
                                 zero_module(nn.Conv2d(out_channels[i], out_channels[i], kernel_size=(1, 1))))
            if not self.image_finetune:
                motion_modules.append(
                    get_motion_module(
                        in_channels=out_channels[i],
                        motion_module_type=motion_module_type,
                        motion_module_kwargs=motion_module_kwargs,
                    )
                )

        if not self.image_finetune:
            self.motion_modules = nn.ModuleList(motion_modules)

    def forward(
            self, dwpose, hamer, smpl,
            temb=None, encoder_hidden_states=None, video_length=-1
    ):
        cond = torch.cat((dwpose, hamer, smpl), dim=1)
        cond = self.backbone(cond)
        cond = self.gate_module(cond)
        outs = []

        for i in range(len(self.out_channels)):
            cond = self.get_submodule(f'blocks_{i}')(cond)
            if not self.image_finetune and video_length > 1:
                cond = rearrange(cond, "(b f) c h w -> b c f h w", f=video_length)
                cond = self.motion_modules[i](
                    cond, temb=temb, encoder_hidden_states=encoder_hidden_states)
                cond = rearrange(cond, "b c f h w -> (b f) c h w")
            outs.append(self.get_submodule(f'block_{i}_out_proj')(cond))

        return tuple(outs)


class ControlNetConditioningEmbedding(nn.Module):
    """
    Quoting from https://arxiv.org/abs/2302.05543: "Stable Diffusion uses a pre-processing method similar to VQ-GAN
    [11] to convert the entire dataset of 512 × 512 images into smaller 64 × 64 “latent images” for stabilized
    training. This requires ControlNets to convert image-based conditions to 64 × 64 feature space to match the
    convolution size. We use a tiny network E(·) of four convolution layers with 4 × 4 kernels and 2 × 2 strides
    (activated by ReLU, channels are 16, 32, 64, 128, initialized with Gaussian weights, trained jointly with the full
    model) to encode image-space conditions ... into feature maps ..."
    """

    def __init__(
        self,
        conditioning_embedding_channels: int,
        conditioning_channels: int = 3,
        block_out_channels: Tuple[int, ...] = (16, 32, 96, 256),
    ):
        super().__init__()

        self.conv_in = nn.Conv2d(conditioning_channels, block_out_channels[0], kernel_size=(3, 3), padding=1)

        self.blocks = nn.ModuleList([])

        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            self.blocks.append(nn.Conv2d(channel_in, channel_in, kernel_size=(3, 3), padding=1))
            self.blocks.append(nn.Conv2d(channel_in, channel_out, kernel_size=(3, 3), padding=1, stride=(2, 2)))

        self.conv_out = zero_module(
            nn.Conv2d(block_out_channels[-1], conditioning_embedding_channels, kernel_size=(3, 3), padding=1)
        )

    def forward(self, conditioning):
        embedding = self.conv_in(conditioning)
        embedding = F.silu(embedding)

        for block in self.blocks:
            embedding = block(embedding)
            embedding = F.silu(embedding)

        embedding = self.conv_out(embedding)

        return (embedding,)
