# This source code is licensed under the Apache License found in the
# LICENSE file in the current directory.
"""
The code has been adopted from Kandinsky-4
(https://github.com/ai-forever/Kandinsky-4/blob/main/kandinsky/model/dit.py)
"""

import math

import torch
from torch import nn
import torch.nn.functional as F

from diffusers import CogVideoXDDIMScheduler

from models.model_40.model_kd40_env import Model_KD40_Environment

from .nn import (
    TimeEmbeddings,
    TextEmbeddings,
    VisualEmbeddings,
    RoPE3D,
    Modulation,
    MultiheadSelfAttention,
    MultiheadSelfAttentionTP,
    FeedForward,
    OutLayer,
)
from .utils import exist


from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    PrepareModuleOutput,
    RowwiseParallel,
    SequenceParallel,
    parallelize_module,
)

from torch.distributed._tensor import Replicate, Shard


def parallelize(model, tp_mesh):
    if tp_mesh.size() > 1:

        plan = {
            "in_layer": ColwiseParallel(),
            "out_layer": RowwiseParallel(
                output_layouts=Replicate(),
            ),
        }
        parallelize_module(model.time_embeddings, tp_mesh, plan)

        plan = {
            "in_layer": ColwiseParallel(
                output_layouts=Replicate(),
            )
        }
        parallelize_module(model.text_embeddings, tp_mesh, plan)
        parallelize_module(model.visual_embeddings, tp_mesh, plan)

        for i, doubled_transformer_block in enumerate(model.transformer_blocks):
            for j, transformer_block in enumerate(doubled_transformer_block):
                transformer_block.self_attention = MultiheadSelfAttentionTP(
                    transformer_block.self_attention
                )
                plan = {
                    # text modulation
                    "text_modulation": PrepareModuleInput(
                        input_layouts=(None, None),
                        desired_input_layouts=(Replicate(), None),
                    ),
                    "text_modulation.out_layer": ColwiseParallel(
                        output_layouts=Replicate(),
                    ),
                    # visual modulation
                    "visual_modulation": PrepareModuleInput(
                        input_layouts=(None, None),
                        desired_input_layouts=(Replicate(), None),
                    ),
                    "visual_modulation.out_layer": ColwiseParallel(
                        output_layouts=Replicate(), use_local_output=True
                    ),
                    # self_attention_norm
                    "self_attention_norm": SequenceParallel(
                        sequence_dim=0, use_local_output=True
                    ),  # TODO надо ли вообще это??? если у нас смешанный ввод нескольких видосом может быть
                    # self_attention
                    "self_attention.to_query": ColwiseParallel(
                        input_layouts=Replicate(),
                    ),
                    "self_attention.to_key": ColwiseParallel(
                        input_layouts=Replicate(),
                    ),
                    "self_attention.to_value": ColwiseParallel(
                        input_layouts=Replicate(),
                    ),
                    "self_attention.query_norm": SequenceParallel(
                        sequence_dim=0, use_local_output=True
                    ),
                    "self_attention.key_norm": SequenceParallel(
                        sequence_dim=0, use_local_output=True
                    ),
                    "self_attention.output_layer": RowwiseParallel(
                        # input_layouts=(Shard(0), ),
                        output_layouts=Replicate(),
                    ),
                    # feed_forward_norm
                    "feed_forward_norm": SequenceParallel(
                        sequence_dim=0, use_local_output=True
                    ),
                    # feed_forward
                    "feed_forward.in_layer": ColwiseParallel(),
                    "feed_forward.out_layer": RowwiseParallel(),
                }
                self_attn = transformer_block.self_attention
                self_attn.num_heads = self_attn.num_heads // tp_mesh.size()
                parallelize_module(transformer_block, tp_mesh, plan)

        plan = {
            "modulation_out": ColwiseParallel(
                output_layouts=Replicate(),
            ),
            "out_layer": ColwiseParallel(
                output_layouts=Replicate(),
            ),
        }
        parallelize_module(model.out_layer, tp_mesh, plan)

        plan = {
            "time_embeddings": PrepareModuleInput(
                desired_input_layouts=Replicate(),
            ),
            "text_embeddings": PrepareModuleInput(
                desired_input_layouts=Replicate(),
            ),
            "visual_embeddings": PrepareModuleInput(
                desired_input_layouts=Replicate(),
            ),
            "out_layer": PrepareModuleInput(
                input_layouts=(None, None, None, None),
                desired_input_layouts=(Replicate(), Replicate(), Replicate(), None),
            ),
        }
        parallelize_module(model, tp_mesh, {})
    return model


class TransformerBlock(nn.Module):

    def __init__(self, model_dim, time_dim, ff_dim, attention_type, head_dim=64):
        super().__init__()
        self.visual_modulation = Modulation(time_dim, model_dim)
        self.text_modulation = Modulation(time_dim, model_dim)

        self.self_attention_norm = nn.LayerNorm(model_dim, elementwise_affine=True)
        self.self_attention = MultiheadSelfAttention(
            model_dim, head_dim, attention_type
        )

        self.feed_forward_norm = nn.LayerNorm(model_dim, elementwise_affine=True)
        self.feed_forward = FeedForward(model_dim, ff_dim)

    def forward(
        self,
        visual_embed,
        text_embed,
        time_embed,
        rope,
        visual_cu_seqlens,
        text_cu_seqlens,
        num_groups,
        attention_type,
    ):
        visual_shape = visual_embed.shape[:-1]
        visual_self_attn_params, visual_ff_params = self.visual_modulation(
            time_embed, visual_cu_seqlens
        )
        text_self_attn_params, text_ff_params = self.text_modulation(
            time_embed, text_cu_seqlens
        )

        visual_shift, visual_scale, visual_gate = torch.chunk(
            visual_self_attn_params, 3, dim=-1
        )
        text_shift, text_scale, text_gate = torch.chunk(
            text_self_attn_params, 3, dim=-1
        )
        visual_out = (
            self.self_attention_norm(visual_embed) * (visual_scale[:, None, None] + 1.0)
            + visual_shift[:, None, None]
        )
        text_out = (
            self.self_attention_norm(text_embed) * (text_scale + 1.0) + text_shift
        )
        visual_out, text_out = self.self_attention(
            visual_out,
            text_out,
            rope,
            visual_cu_seqlens,
            text_cu_seqlens,
            num_groups,
            attention_type,
        )

        visual_embed = visual_embed + visual_gate[:, None, None] * visual_out
        text_embed = text_embed + text_gate * text_out

        visual_shift, visual_scale, visual_gate = torch.chunk(
            visual_ff_params, 3, dim=-1
        )
        visual_out = (
            self.feed_forward_norm(visual_embed) * (visual_scale[:, None, None] + 1.0)
            + visual_shift[:, None, None]
        )
        visual_embed = visual_embed + visual_gate[:, None, None] * self.feed_forward(
            visual_out
        )

        text_shift, text_scale, text_gate = torch.chunk(text_ff_params, 3, dim=-1)
        text_out = self.feed_forward_norm(text_embed) * (text_scale + 1.0) + text_shift
        text_embed = text_embed + text_gate * self.feed_forward(text_out)
        return visual_embed, text_embed


class DiffusionTransformer3D(nn.Module):
    def __init__(
        self,
        k_attention_type="flash",
        in_visual_dim=4,
        in_text_dim=2048,
        time_dim=512,
        out_visual_dim=4,
        patch_size=(1, 2, 2),
        model_dim=2048,
        ff_dim=5120,
        num_blocks=8,
        axes_dims=(16, 24, 24),
    ):
        super().__init__()
        head_dim = sum(axes_dims)
        self.k_attention_type = k_attention_type
        self.in_visual_dim = in_visual_dim
        self.model_dim = model_dim
        self.num_blocks = num_blocks

        self.time_embeddings = TimeEmbeddings(model_dim, time_dim)
        self.text_embeddings = TextEmbeddings(in_text_dim, model_dim)
        self.visual_embeddings = VisualEmbeddings(in_visual_dim, model_dim, patch_size)
        self.rope_embeddings = RoPE3D(axes_dims)

        self.transformer_blocks = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        TransformerBlock(
                            model_dim,
                            time_dim,
                            ff_dim,
                            k_attention_type,
                            head_dim,
                        ),
                        TransformerBlock(
                            model_dim,
                            time_dim,
                            ff_dim,
                            k_attention_type,
                            head_dim,
                        ),
                    ]
                )
                for _ in range(num_blocks)
            ]
        )

        self.out_layer = OutLayer(model_dim, time_dim, out_visual_dim, patch_size)

    def forward(
        self,
        x,
        text_embed,
        time,
        visual_cu_seqlens,
        text_cu_seqlens,
        num_groups=(1, 1, 1),
        scale_factor=(1.0, 1.0, 1.0),
    ):
        time_embed = self.time_embeddings(time)
        text_embed = self.text_embeddings(text_embed)
        visual_embed = self.visual_embeddings(x)
        rope = self.rope_embeddings(visual_embed, visual_cu_seqlens, scale_factor)

        for i, (local_attention, global_attention) in enumerate(
            self.transformer_blocks
        ):
            visual_embed, text_embed = local_attention(
                visual_embed,
                text_embed,
                time_embed,
                rope,
                visual_cu_seqlens,
                text_cu_seqlens,
                num_groups,
                "local",
            )
            visual_embed, text_embed = global_attention(
                visual_embed,
                text_embed,
                time_embed,
                rope,
                visual_cu_seqlens,
                text_cu_seqlens,
                num_groups,
                "global",
            )

        return self.out_layer(visual_embed, text_embed, time_embed, visual_cu_seqlens)


def get_dit(conf):
    dit = DiffusionTransformer3D(**conf.params)
    if conf.checkpoint_path.endswith((".safetensors", ".sft")):
        from safetensors.torch import load_file

        state_dict = load_file(conf.checkpoint_path, device="cpu")
    else:
        state_dict = torch.load(
            conf.checkpoint_path, weights_only=True, map_location=torch.device("cpu")
        )
    dit.load_state_dict(state_dict, strict=False)
    return dit
