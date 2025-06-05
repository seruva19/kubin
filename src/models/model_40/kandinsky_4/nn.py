# This source code is licensed under the Apache License found in the
# LICENSE file in the current directory.
"""
The code has been adopted from Kandinsky-4
(https://github.com/ai-forever/Kandinsky-4/blob/main/kandinsky/model/nn.py)
"""


import time
import math

import torch
from torch import nn
import torch.nn.functional as F

from utils.logging import k_log
from .attention import standard_flash_attn_varlen_qkvpacked_func_replacement

flash_attn_not_available = False

try:
    from flash_attn import flash_attn_varlen_qkvpacked_func
except:
    flash_attn_not_available = True

from .utils import (
    exist,
    get_freqs,
    cat_interleave,
    split_interleave,
    to_1dimension,
    to_3dimension,
)


def apply_rotary(x, rope):
    x_ = x.reshape(*x.shape[:-1], -1, 1, 2).to(torch.float32)
    x_out = rope[..., 0] * x_[..., 0] + rope[..., 1] * x_[..., 1]
    return x_out.reshape(*x.shape)


class TimeEmbeddings(nn.Module):

    def __init__(self, model_dim, time_dim, max_period=10000.0):
        super().__init__()
        assert model_dim % 2 == 0
        self.freqs = get_freqs(model_dim // 2, max_period)

        self.in_layer = nn.Linear(model_dim, time_dim, bias=True)
        self.activation = nn.SiLU()
        self.out_layer = nn.Linear(time_dim, time_dim, bias=True)

    def forward(self, time):
        args = torch.outer(time, self.freqs.to(device=time.device))
        time_embed = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.out_layer(self.activation(self.in_layer(time_embed)))


class TextEmbeddings(nn.Module):

    def __init__(self, text_dim, model_dim):
        super().__init__()
        self.in_layer = nn.Linear(text_dim, model_dim, bias=True)

    def forward(self, text_embed):
        return self.in_layer(text_embed)


class VisualEmbeddings(nn.Module):

    def __init__(self, visual_dim, model_dim, patch_size):
        super().__init__()
        self.patch_size = patch_size
        self.in_layer = nn.Linear(math.prod(patch_size) * visual_dim, model_dim)

    def forward(self, x):
        duration, height, width, dim = x.shape
        x = (
            x.view(
                duration // self.patch_size[0],
                self.patch_size[0],
                height // self.patch_size[1],
                self.patch_size[1],
                width // self.patch_size[2],
                self.patch_size[2],
                dim,
            )
            .permute(0, 2, 4, 1, 3, 5, 6)
            .flatten(3, 6)
        )
        return self.in_layer(x)


class RoPE3D(nn.Module):

    def __init__(self, axes_dims, max_pos=(128, 128, 128), max_period=10000.0):
        super().__init__()
        for i, (axes_dim, ax_max_pos) in enumerate(zip(axes_dims, max_pos)):
            freq = get_freqs(axes_dim // 2, max_period)
            pos = torch.arange(ax_max_pos, dtype=freq.dtype)
            self.register_buffer(f"args_{i}", torch.outer(pos, freq))

    def args(self, i, cu_seqlens):
        args = self.__getattr__(f"args_{i}")
        if torch.is_tensor(cu_seqlens):
            args = torch.cat([args[:end] for end in torch.diff(cu_seqlens)])
        else:
            args = args[:cu_seqlens]
        return args

    def forward(self, x, cu_seqlens, scale_factor=(1.0, 1.0, 1.0)):
        duration, height, width = x.shape[:-1]
        args = [
            self.args(i, ax_cu_seqlens) / ax_scale_factor
            for i, (ax_cu_seqlens, ax_scale_factor) in enumerate(
                zip([cu_seqlens, height, width], scale_factor)
            )
        ]
        args = torch.cat(
            [
                args[0].view(duration, 1, 1, -1).repeat(1, height, width, 1),
                args[1].view(1, height, 1, -1).repeat(duration, 1, width, 1),
                args[2].view(1, 1, width, -1).repeat(duration, height, 1, 1),
            ],
            dim=-1,
        )
        rope = torch.stack(
            [torch.cos(args), -torch.sin(args), torch.sin(args), torch.cos(args)],
            dim=-1,
        )
        rope = rope.view(*rope.shape[:-1], 2, 2)
        return rope.unsqueeze(-4)


class Modulation(nn.Module):

    def __init__(self, time_dim, model_dim):
        super().__init__()
        self.activation = nn.SiLU()
        self.out_layer = nn.Linear(time_dim, 6 * model_dim)
        self.out_layer.weight.data.zero_()
        self.out_layer.bias.data.zero_()

    def forward(self, x, cu_seqlens):
        modulation_params = self.out_layer(self.activation(x))
        modulation_params = modulation_params.repeat_interleave(
            torch.diff(cu_seqlens), dim=0
        )
        self_attn_params, ff_params = torch.chunk(modulation_params, 2, dim=-1)
        return self_attn_params, ff_params


class MultiheadSelfAttention(nn.Module):

    def __init__(self, num_channels, head_dim=64, attention_type="flash"):
        super().__init__()
        assert num_channels % head_dim == 0
        self.attention_type = attention_type
        self.num_heads = num_channels // head_dim

        self.to_query_key_value = nn.Linear(num_channels, 3 * num_channels, bias=True)
        self.query_norm = nn.LayerNorm(head_dim)
        self.key_norm = nn.LayerNorm(head_dim)

        self.output_layer = nn.Linear(num_channels, num_channels, bias=True)

    def scaled_dot_product_attention(
        self,
        visual_query_key_value,
        text_query_key_value,
        visual_cu_seqlens,
        text_cu_seqlens,
        num_groups,
        attention_type,
        return_attn_probs=False,
    ):
        if self.attention_type == "sage":
            raise NotImplementedError(
                "scaled_dot_product_attention is not implemented for attention_type=sage"
            )

        visual_shape, text_len = (
            visual_query_key_value.shape[:3],
            text_cu_seqlens[1],
        )
        visual_query_key_value, visual_cu_seqlens = to_1dimension(
            visual_query_key_value,
            visual_cu_seqlens,
            visual_shape,
            num_groups,
            attention_type,
        )
        text_query_key_value = text_query_key_value.unsqueeze(0).expand(
            math.prod(num_groups), *text_query_key_value.size()
        )
        query_key_value = cat_interleave(
            visual_query_key_value,
            text_query_key_value,
            visual_cu_seqlens,
            text_cu_seqlens,
        )
        cu_seqlens = visual_cu_seqlens + text_cu_seqlens

        max_seqlen = torch.diff(cu_seqlens).max()
        query_key_value = query_key_value.flatten(0, 1)
        large_cu_seqlens = torch.cat(
            [cu_seqlens + i * cu_seqlens[-1] for i in range(math.prod(num_groups))]
        )

        if self.attention_type == "flash":
            out, softmax_lse, _ = flash_attn_varlen_qkvpacked_func(
                query_key_value,
                large_cu_seqlens,
                max_seqlen,
                return_attn_probs=True,
            )
        elif self.attention_type == "sdpa":
            out, softmax_lse, _ = standard_flash_attn_varlen_qkvpacked_func_replacement(
                query_key_value, large_cu_seqlens, max_seqlen, return_attn_probs=True
            )

        out = out.reshape(math.prod(num_groups), -1, *out.shape[1:]).flatten(-2, -1)

        visual_out, text_out = split_interleave(out, cu_seqlens, text_len)
        visual_out = to_3dimension(visual_out, visual_shape, num_groups, attention_type)
        if return_attn_probs:
            return (visual_out, text_out), softmax_lse, None
        return visual_out, text_out

    def forward(
        self,
        visual_embed,
        text_embed,
        rope,
        visual_cu_seqlens,
        text_cu_seqlens,
        num_groups,
        attention_type,
    ):
        visual_shape = visual_embed.shape[:-1]
        visual_query_key_value = self.to_query_key_value(visual_embed)

        visual_query, visual_key, visual_value = torch.chunk(
            visual_query_key_value, 3, dim=-1
        )
        visual_query = self.query_norm(
            visual_query.reshape(*visual_shape, self.num_heads, -1)
        ).type_as(visual_query)
        visual_key = self.key_norm(
            visual_key.reshape(*visual_shape, self.num_heads, -1)
        ).type_as(visual_key)
        visual_value = visual_value.reshape(*visual_shape, self.num_heads, -1)
        visual_query = apply_rotary(visual_query, rope).type_as(visual_query)
        visual_key = apply_rotary(visual_key, rope).type_as(visual_key)
        visual_query_key_value = torch.stack(
            [visual_query, visual_key, visual_value], dim=3
        )

        text_len = text_embed.shape[0]
        text_query_key_value = self.to_query_key_value(text_embed)
        text_query, text_key, text_value = torch.chunk(text_query_key_value, 3, dim=-1)
        text_query = self.query_norm(
            text_query.reshape(text_len, self.num_heads, -1)
        ).type_as(text_query)
        text_key = self.key_norm(
            text_key.reshape(text_len, self.num_heads, -1)
        ).type_as(text_key)
        text_value = text_value.reshape(text_len, self.num_heads, -1)
        text_query_key_value = torch.stack([text_query, text_key, text_value], dim=1)

        visual_out, text_out = self.scaled_dot_product_attention(
            visual_query_key_value,
            text_query_key_value,
            visual_cu_seqlens,
            text_cu_seqlens,
            num_groups,
            attention_type,
        )
        visual_out = self.output_layer(visual_out)
        text_out = self.output_layer(text_out)

        return visual_out, text_out


class MultiheadSelfAttentionTP(nn.Module):

    def __init__(self, initial_multihead_self_attention):
        super().__init__()
        num_channels = initial_multihead_self_attention.to_query_key_value.weight.shape[
            1
        ]
        self.num_heads = initial_multihead_self_attention.num_heads
        head_dim = num_channels // self.num_heads
        self.attention_type = initial_multihead_self_attention.attention_type

        self.to_query = nn.Linear(num_channels, num_channels, bias=True)
        self.to_key = nn.Linear(num_channels, num_channels, bias=True)
        self.to_value = nn.Linear(num_channels, num_channels, bias=True)

        weight = initial_multihead_self_attention.to_query_key_value.weight
        bias = initial_multihead_self_attention.to_query_key_value.bias
        self.to_query.weight = torch.nn.Parameter(weight[:num_channels])
        self.to_key.weight = torch.nn.Parameter(weight[num_channels : 2 * num_channels])
        self.to_value.weight = torch.nn.Parameter(weight[2 * num_channels :])
        self.to_query.bias = torch.nn.Parameter(bias[:num_channels])
        self.to_key.bias = torch.nn.Parameter(bias[num_channels : 2 * num_channels])
        self.to_value.bias = torch.nn.Parameter(bias[2 * num_channels :])

        self.query_norm = initial_multihead_self_attention.query_norm
        self.key_norm = initial_multihead_self_attention.key_norm
        self.output_layer = initial_multihead_self_attention.output_layer

    def scaled_dot_product_attention(
        self,
        visual_query_key_value,
        text_query_key_value,
        visual_cu_seqlens,
        text_cu_seqlens,
        num_groups,
        attention_type,
        return_attn_probs=False,
    ):
        if self.attention_type == "sage":
            raise NotImplementedError(
                "scaled_dot_product_attention is not implemented for attention_type=sage"
            )

        visual_shape, text_len = (
            visual_query_key_value.shape[:3],
            text_cu_seqlens[1],
        )
        visual_query_key_value, visual_cu_seqlens = to_1dimension(
            visual_query_key_value,
            visual_cu_seqlens,
            visual_shape,
            num_groups,
            attention_type,
        )
        text_query_key_value = text_query_key_value.unsqueeze(0).expand(
            math.prod(num_groups), *text_query_key_value.size()
        )
        query_key_value = cat_interleave(
            visual_query_key_value,
            text_query_key_value,
            visual_cu_seqlens,
            text_cu_seqlens,
        )
        cu_seqlens = visual_cu_seqlens + text_cu_seqlens

        max_seqlen = torch.diff(cu_seqlens).max()
        query_key_value = query_key_value.flatten(0, 1)
        large_cu_seqlens = torch.cat(
            [cu_seqlens + i * cu_seqlens[-1] for i in range(math.prod(num_groups))]
        )

        if self.attention_type == "flash":
            out, softmax_lse, _ = flash_attn_varlen_qkvpacked_func(
                query_key_value, large_cu_seqlens, max_seqlen, return_attn_probs=True
            )

        elif self.attention_type == "sdpa":
            out, softmax_lse, _ = standard_flash_attn_varlen_qkvpacked_func_replacement(
                query_key_value, large_cu_seqlens, max_seqlen, return_attn_probs=True
            )

        out = out.reshape(math.prod(num_groups), -1, *out.shape[1:]).flatten(-2, -1)

        visual_out, text_out = split_interleave(out, cu_seqlens, text_len)
        visual_out = to_3dimension(visual_out, visual_shape, num_groups, attention_type)
        if return_attn_probs:
            return (visual_out, text_out), softmax_lse, None
        return visual_out, text_out

    def forward(
        self,
        visual_embed,
        text_embed,
        rope,
        visual_cu_seqlens,
        text_cu_seqlens,
        num_groups,
        attention_type,
    ):
        visual_shape = visual_embed.shape[:-1]
        visual_query, visual_key, visual_value = (
            self.to_query(visual_embed),
            self.to_key(visual_embed),
            self.to_value(visual_embed),
        )
        visual_query = self.query_norm(
            visual_query.reshape(*visual_shape, self.num_heads, -1)
        ).type_as(visual_query)
        visual_key = self.key_norm(
            visual_key.reshape(*visual_shape, self.num_heads, -1)
        ).type_as(visual_key)
        visual_value = visual_value.reshape(*visual_shape, self.num_heads, -1)
        visual_query = apply_rotary(visual_query, rope).type_as(visual_query)
        visual_key = apply_rotary(visual_key, rope).type_as(visual_key)
        visual_query_key_value = torch.stack(
            [visual_query, visual_key, visual_value], dim=3
        )

        text_len = text_embed.shape[0]
        text_query, text_key, text_value = (
            self.to_query(text_embed),
            self.to_key(text_embed),
            self.to_value(text_embed),
        )
        text_query = self.query_norm(
            text_query.reshape(text_len, self.num_heads, -1)
        ).type_as(text_query)
        text_key = self.key_norm(
            text_key.reshape(text_len, self.num_heads, -1)
        ).type_as(text_key)
        text_value = text_value.reshape(text_len, self.num_heads, -1)
        text_query_key_value = torch.stack([text_query, text_key, text_value], dim=1)

        visual_out, text_out = self.scaled_dot_product_attention(
            visual_query_key_value,
            text_query_key_value,
            visual_cu_seqlens,
            text_cu_seqlens,
            num_groups,
            attention_type,
        )
        visual_out = self.output_layer(visual_out)
        text_out = self.output_layer(text_out)

        return visual_out, text_out


class FeedForward(nn.Module):
    def __init__(self, dim, ff_dim):
        super().__init__()
        self.in_layer = nn.Linear(dim, ff_dim, bias=True)
        self.activation = nn.GELU()
        self.out_layer = nn.Linear(ff_dim, dim, bias=True)

    def forward(self, x):
        return self.out_layer(self.activation(self.in_layer(x)))


class OutLayer(nn.Module):
    def __init__(self, model_dim, time_dim, visual_dim, patch_size):
        super().__init__()
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(model_dim, elementwise_affine=True)
        self.out_layer = nn.Linear(
            model_dim, math.prod(patch_size) * visual_dim, bias=True
        )

        self.modulation_activation = nn.SiLU()
        self.modulation_out = nn.Linear(time_dim, 2 * model_dim, bias=True)
        self.modulation_out.weight.data.zero_()
        self.modulation_out.bias.data.zero_()

    def forward(self, visual_embed, text_embed, time_embed, visual_cu_seqlens):
        modulation_params = self.modulation_out(self.modulation_activation(time_embed))
        modulation_params = modulation_params.repeat_interleave(
            torch.diff(visual_cu_seqlens), dim=0
        )
        shift, scale = torch.chunk(modulation_params, 2, dim=-1)
        visual_embed = (
            self.norm(visual_embed) * (scale[:, None, None, :] + 1)
            + shift[:, None, None, :]
        )
        x = self.out_layer(visual_embed)

        duration, height, width, dim = x.shape
        x = (
            x.view(
                duration,
                height,
                width,
                -1,
                self.patch_size[0],
                self.patch_size[1],
                self.patch_size[2],
            )
            .permute(0, 4, 1, 5, 2, 6, 3)
            .flatten(0, 1)
            .flatten(1, 2)
            .flatten(2, 3)
        )
        return x
