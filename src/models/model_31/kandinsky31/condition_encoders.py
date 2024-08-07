# This source code is licensed under the Apache License found in the
# LICENSE file in the current directory.
"""
The code has been adopted from Kandinsky-3
(https://github.com/ai-forever/Kandinsky-3/blob/main/kandinsky3/condition_encoders.py)
"""

from models.model_31.model_kd31_env import Model_KD31_Environment
import torch
from torch import nn
from transformers import T5EncoderModel, BitsAndBytesConfig
from typing import Optional, Union


class T5TextConditionEncoder(nn.Module):

    def __init__(
        self,
        model_path,
        environment: Model_KD31_Environment,
        context_dim,
        cache_dir,
        low_cpu_mem_usage: bool = True,
        device: Optional[str] = None,
        dtype: Union[str, torch.dtype] = torch.float32,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
    ):
        super().__init__()
        if environment.kd31_low_vram:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_type=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=False,
            )
        else:
            quantization_config = None

        self.encoder = T5EncoderModel.from_pretrained(
            model_path,
            low_cpu_mem_usage=low_cpu_mem_usage,
            device_map="auto" if environment.kd31_low_vram else device,
            torch_dtype=dtype,
            # load_in_8bit=load_in_8bit,
            # load_in_4bit=environment.kd31_low_vram,
            cache_dir=cache_dir,
            quantization_config=quantization_config,
        ).encoder
        self.projection = nn.Sequential(
            nn.Linear(self.encoder.config.d_model, context_dim, bias=False),
            nn.LayerNorm(context_dim),
        )

    def forward(self, model_input):
        embeddings = self.encoder(**model_input).last_hidden_state
        context = self.projection(embeddings)
        if "attention_mask" in model_input:
            context_mask = model_input["attention_mask"]
            context[context_mask == 0] = torch.zeros_like(context[context_mask == 0])
            max_seq_length = context_mask.sum(-1).max() + 1
            context = context[:, :max_seq_length]
            context_mask = context_mask[:, :max_seq_length]
        else:
            context_mask = torch.ones(
                *embeddings.shape[:-1], dtype=torch.long, device=embeddings.device
            )
        return context, context_mask


def get_condition_encoder(conf):
    return T5TextConditionEncoder(**conf)
