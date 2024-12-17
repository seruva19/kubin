# This source code is licensed under the Apache License found in the
# LICENSE file in the current directory.
"""
The code has been adopted from Kandinsky-4
(https://github.com/ai-forever/Kandinsky-4/blob/main/kandinsky/model/utils.py)
"""


import math

import torch


def exist(item):
    return item is not None


def freeze(model):
    for p in model.parameters():
        p.requires_grad = False
    return model


def get_freqs(dim, max_period=10000.0):
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=dim, dtype=torch.float32)
        / dim
    )
    return freqs


def get_group_sizes(shape, num_groups):
    return [*map(lambda x: x[0] // x[1], zip(shape, num_groups))]


def rescale_group_rope(num_groups, scale_factor, rescale_factor):
    num_groups = [*map(lambda x: int(x[0] / x[1]), zip(num_groups, rescale_factor))]
    scale_factor = [*map(lambda x: x[0] / x[1], zip(scale_factor, rescale_factor))]
    return num_groups, scale_factor


def cat_interleave(
    visual_query_key_value, text_query_key_value, visual_cu_seqlens, text_cu_seqlens
):
    query_key_value = []
    for local_visual_query_key_value, local_text_query_key_value in zip(
        torch.split(
            visual_query_key_value, torch.diff(visual_cu_seqlens).tolist(), dim=1
        ),
        torch.split(text_query_key_value, torch.diff(text_cu_seqlens).tolist(), dim=1),
    ):
        query_key_value += [local_visual_query_key_value, local_text_query_key_value]
    query_key_value = torch.cat(query_key_value, dim=1)
    return query_key_value


def split_interleave(out, cu_seqlens, split_len):
    visual_out, text_out = [], []
    for local_out in torch.split(out, torch.diff(cu_seqlens).tolist(), dim=1):
        visual_out.append(local_out[:, :-split_len])
        text_out.append(local_out[0, -split_len:])
    visual_out, text_out = torch.cat(visual_out, dim=1), torch.cat(text_out, dim=0)
    return visual_out, text_out


def local_patching(x, shape, group_size, dim=0):
    duration, height, width = shape
    g1, g2, g3 = group_size
    x = x.reshape(
        *x.shape[:dim],
        duration // g1,
        g1,
        height // g2,
        g2,
        width // g3,
        g3,
        *x.shape[dim + 3 :]
    )
    x = x.permute(
        *range(len(x.shape[:dim])),
        dim,
        dim + 2,
        dim + 4,
        dim + 1,
        dim + 3,
        dim + 5,
        *range(dim + 6, len(x.shape))
    )
    x = x.flatten(dim, dim + 2).flatten(dim + 1, dim + 3)
    return x


def local_merge(x, shape, group_size, dim=0):
    duration, height, width = shape
    g1, g2, g3 = group_size
    x = x.reshape(
        *x.shape[:dim],
        duration // g1,
        height // g2,
        width // g3,
        g1,
        g2,
        g3,
        *x.shape[dim + 2 :]
    )
    x = x.permute(
        *range(len(x.shape[:dim])),
        dim,
        dim + 3,
        dim + 1,
        dim + 4,
        dim + 2,
        dim + 5,
        *range(dim + 6, len(x.shape))
    )
    x = x.flatten(dim, dim + 1).flatten(dim + 1, dim + 2).flatten(dim + 2, dim + 3)
    return x


def global_patching(x, shape, group_size, dim=0):
    latent_group_size = [
        axis // axis_group_size for axis, axis_group_size in zip(shape, group_size)
    ]
    x = local_patching(x, shape, latent_group_size, dim)
    x = x.transpose(dim, dim + 1)
    return x


def global_merge(x, shape, group_size, dim=0):
    latent_group_size = [
        axis // axis_group_size for axis, axis_group_size in zip(shape, group_size)
    ]
    x = x.transpose(dim, dim + 1)
    x = local_merge(x, shape, latent_group_size, dim)
    return x


def to_1dimension(
    visual_embed, visual_cu_seqlens, visual_shape, num_groups, attention_type
):
    group_size = get_group_sizes(visual_shape, num_groups)
    if attention_type == "local":
        visual_embed = local_patching(visual_embed, visual_shape, group_size, dim=0)
    if attention_type == "global":
        visual_embed = global_patching(visual_embed, visual_shape, group_size, dim=0)
    visual_cu_seqlens = visual_cu_seqlens * math.prod(group_size[1:])
    return visual_embed, visual_cu_seqlens


def to_3dimension(visual_embed, visual_shape, num_groups, attention_type):
    group_size = get_group_sizes(visual_shape, num_groups)
    if attention_type == "local":
        x = local_merge(visual_embed, visual_shape, group_size, dim=0)
    if attention_type == "global":
        x = global_merge(visual_embed, visual_shape, group_size, dim=0)
    return x
