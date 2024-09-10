# Adapted from https://github.com/magic-research/magic-animate/blob/main/magicanimate/models/mutual_self_attention.py

import torch

from einops import rearrange
from typing import Any, Dict, Optional

from diffusers.models.attention import BasicTransformerBlock
from .attention import BasicTransformerBlock as _BasicTransformerBlock


def torch_dfs(model: torch.nn.Module):
    result = [model]
    for child in model.children():
        result += torch_dfs(child)
    return result


class ReferenceNetAttention():

    def __init__(self,
                 unet,
                 mode="write",
                 attention_auto_machine_weight=float('inf'),
                 gn_auto_machine_weight=1.0,
                 style_fidelity=1.0,
                 fusion_blocks="full",
                 is_image=False,
                 ) -> None:
        # 10. Modify self attention and group norm
        self.unet = unet
        assert mode in ["read", "write"]
        assert fusion_blocks in ["midup", "full"]
        self.fusion_blocks = fusion_blocks
        self.register_reference_hooks(
            mode,
            attention_auto_machine_weight,
            gn_auto_machine_weight,
            style_fidelity,
            fusion_blocks=fusion_blocks,
            is_image=is_image,
        )

    def register_reference_hooks(
            self,
            mode,
            attention_auto_machine_weight,
            gn_auto_machine_weight,
            style_fidelity,
            fusion_blocks='full',
            is_image=False,
    ):
        MODE = mode
        attention_auto_machine_weight = attention_auto_machine_weight
        gn_auto_machine_weight = gn_auto_machine_weight
        style_fidelity = style_fidelity
        fusion_blocks = fusion_blocks
        is_image = is_image

        def hacked_basic_transformer_inner_forward(
                self,
                hidden_states: torch.FloatTensor,
                attention_mask: Optional[torch.FloatTensor] = None,
                encoder_hidden_states: Optional[torch.FloatTensor] = None,
                encoder_attention_mask: Optional[torch.FloatTensor] = None,
                timestep: Optional[torch.LongTensor] = None,
                cross_attention_kwargs: Dict[str, Any] = None,
                class_labels: Optional[torch.LongTensor] = None,
                video_length=None,
        ):
            if self.use_ada_layer_norm:
                norm_hidden_states = self.norm1(hidden_states, timestep)
            elif self.use_ada_layer_norm_zero:
                norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                    hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
                )
            else:
                norm_hidden_states = self.norm1(hidden_states)

            # 1. Self-Attention
            cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}
            if self.only_cross_attention:
                attn_output = self.attn1(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                    attention_mask=attention_mask,
                    **cross_attention_kwargs,
                )
            else:
                if MODE == "write":
                    self.bank.append(norm_hidden_states.clone())
                    attn_output = self.attn1(
                        norm_hidden_states,
                        encoder_hidden_states=None,
                        attention_mask=attention_mask,
                        **cross_attention_kwargs,
                    )
                if MODE == "read":
                    if not is_image:
                        self.bank = [rearrange(d.unsqueeze(1).repeat(1, video_length, 1, 1), "b t l c -> (b t) l c")[
                                     :hidden_states.shape[0]] for d in self.bank]
                    # modify Reference Sec 3.2.2
                    modify_norm_hidden_states = torch.cat([norm_hidden_states] + self.bank, dim=-2)

                    hidden_states = self.attn1(norm_hidden_states,
                                               encoder_hidden_states=modify_norm_hidden_states,
                                               attention_mask=attention_mask) + hidden_states

                    if self.attn2 is not None:
                        # Cross-Attention
                        norm_hidden_states = (
                            self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(
                                hidden_states)
                        )
                        hidden_states = self.attn2(
                            norm_hidden_states, encoder_hidden_states=encoder_hidden_states,
                            attention_mask=attention_mask) + hidden_states

                    # Feed-forward
                    hidden_states = self.ff(self.norm3(hidden_states)) + hidden_states

                    # Temporal-Attention
                    if not is_image:
                        if self.unet_use_temporal_attention:
                            d = hidden_states.shape[1]
                            hidden_states = rearrange(hidden_states, "(b f) d c -> (b d) f c", f=video_length)
                            norm_hidden_states = (
                                self.norm_temp(hidden_states, timestep) if self.use_ada_layer_norm else self.norm_temp(
                                    hidden_states)
                            )
                            hidden_states = self.attn_temp(norm_hidden_states) + hidden_states
                            hidden_states = rearrange(hidden_states, "(b d) f c -> (b f) d c", d=d)

                    return hidden_states

            if self.use_ada_layer_norm_zero:
                attn_output = gate_msa.unsqueeze(1) * attn_output
            hidden_states = attn_output + hidden_states

            if self.attn2 is not None:
                norm_hidden_states = (
                    self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
                )

                # 2. Cross-Attention
                attn_output = self.attn2(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=encoder_attention_mask,
                    **cross_attention_kwargs,
                )
                hidden_states = attn_output + hidden_states

            # 3. Feed-forward
            norm_hidden_states = self.norm3(hidden_states)

            if self.use_ada_layer_norm_zero:
                norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

            ff_output = self.ff(norm_hidden_states)

            if self.use_ada_layer_norm_zero:
                ff_output = gate_mlp.unsqueeze(1) * ff_output

            hidden_states = ff_output + hidden_states

            return hidden_states

        if self.fusion_blocks == "midup":
            attn_modules = [module for module in (torch_dfs(self.unet.mid_block) + torch_dfs(self.unet.up_blocks))
                            if
                            isinstance(module, BasicTransformerBlock) or isinstance(module, _BasicTransformerBlock)]
        elif self.fusion_blocks == "full":
            attn_modules = [module for module in torch_dfs(self.unet) if
                            isinstance(module, BasicTransformerBlock) or isinstance(module, _BasicTransformerBlock)]
        attn_modules = sorted(attn_modules, key=lambda x: -x.norm1.normalized_shape[0])

        for i, module in enumerate(attn_modules):
            module._original_inner_forward = module.forward
            module.forward = hacked_basic_transformer_inner_forward.__get__(module, BasicTransformerBlock)
            module.bank = []
            module.attn_weight = float(i) / float(len(attn_modules))

    def update(self, writer):
        if self.fusion_blocks == "midup":
            reader_attn_modules = [module for module in
                                   (torch_dfs(self.unet.mid_block) + torch_dfs(self.unet.up_blocks)) if
                                   isinstance(module, _BasicTransformerBlock)]
            writer_attn_modules = [module for module in
                                   (torch_dfs(writer.unet.mid_block) + torch_dfs(writer.unet.up_blocks)) if
                                   isinstance(module, BasicTransformerBlock)]
        elif self.fusion_blocks == "full":
            reader_attn_modules = [module for module in torch_dfs(self.unet) if
                                   isinstance(module, _BasicTransformerBlock) or isinstance(module,
                                                                                            BasicTransformerBlock)]
            writer_attn_modules = [module for module in torch_dfs(writer.unet) if
                                   isinstance(module, _BasicTransformerBlock) or isinstance(module,
                                                                                            BasicTransformerBlock)]
        reader_attn_modules = sorted(reader_attn_modules, key=lambda x: -x.norm1.normalized_shape[0])
        writer_attn_modules = sorted(writer_attn_modules, key=lambda x: -x.norm1.normalized_shape[0])

        if len(reader_attn_modules) == 0:
            print('reader_attn_modules is null')
            assert False
        if len(writer_attn_modules) == 0:
            print('writer_attn_modules is null')
            assert False

        for r, w in zip(reader_attn_modules, writer_attn_modules):
            r.bank = [v.clone() for v in w.bank]

    def clear(self):
        if self.fusion_blocks == "midup":
            reader_attn_modules = [module for module in
                                   (torch_dfs(self.unet.mid_block) + torch_dfs(self.unet.up_blocks)) if
                                   isinstance(module, BasicTransformerBlock) or isinstance(module,
                                                                                           _BasicTransformerBlock)]
        elif self.fusion_blocks == "full":
            reader_attn_modules = [module for module in torch_dfs(self.unet) if
                                   isinstance(module, BasicTransformerBlock) or isinstance(module,
                                                                                           _BasicTransformerBlock)]
        reader_attn_modules = sorted(reader_attn_modules, key=lambda x: -x.norm1.normalized_shape[0])
        for r in reader_attn_modules:
            r.bank.clear()
