# This source code is licensed under the Apache License found in the
# LICENSE file in the current directory.
"""
The code has been adopted from Kandinsky-4
(https://github.com/ai-forever/Kandinsky-4/blob/main/kandinsky4_video2audio/model/attention_processors.py)
"""


import math

import torch
import torch.nn.functional as F

from diffusers.models.attention_processor import AttnProcessor2_0
from einops import rearrange
from torch import nn


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def get_posemb(x, dim):
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
    emb = rearrange(x, "i -> i 1") * rearrange(emb, "j -> 1 j")
    return torch.cat((emb.sin(), emb.cos()), dim=-1)


class VideoAttnProcessor2_0(nn.Module):

    def __init__(self, hidden_size):
        super().__init__()

        self.processor = AttnProcessor2_0()

        self.video_q = nn.Linear(hidden_size, hidden_size)
        self.video_k = nn.Linear(4096, hidden_size)
        self.video_v = nn.Linear(4096, hidden_size)
        self.video_out = zero_module(nn.Linear(hidden_size, hidden_size))

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        image_embeds=None,
    ):
        hidden_states = self.processor(
            attn, hidden_states, encoder_hidden_states, attention_mask, temb
        )

        input_ndim = hidden_states.ndim

        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        query = self.video_q(hidden_states)
        key = self.video_k(image_embeds)
        value = self.video_v(image_embeds)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        pos_emb = get_posemb(
            torch.arange(key.shape[2], device=key.device), key.shape[-1]
        )
        key += pos_emb
        value += pos_emb

        hidden_states_video = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states_video = (
            hidden_states_video.transpose(1, 2)
            .reshape(batch_size, -1, attn.heads * head_dim)
            .to(query.dtype)
        )

        if input_ndim == 4:
            hidden_states_video = hidden_states_video.transpose(-1, -2).reshape(batch_size, channel, height, width)  # type: ignore

        hidden_states = hidden_states + self.video_out(hidden_states_video)

        return hidden_states


class PromptAttnProcessor2_0(nn.Module):
    def __init__(self, hidden_size=768):
        super().__init__()

        self.processor = AttnProcessor2_0()
        self.text_projection = nn.Linear(4096, hidden_size)

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        image_embeds=None,
    ):
        encoder_hidden_states = self.text_projection(encoder_hidden_states)
        return self.processor(
            attn, hidden_states, encoder_hidden_states, attention_mask, temb
        )
