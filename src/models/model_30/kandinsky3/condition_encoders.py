# This source code is licensed under the Apache License found in the
# LICENSE file in the current directory.
"""
The code has been adopted from Kandinsky-3
(https://github.com/ai-forever/Kandinsky-3/blob/main/kandinsky3/condition_encoders.py)
"""

from abc import abstractmethod
from typing import Optional
import torch
from torch import nn
from einops import repeat
from transformers import T5Model, T5EncoderModel, CLIPModel, BitsAndBytesConfig
from typing import Optional

from models.model_30.kandinsky3.utils import freeze
from models.model_30.model_kd30_env import Model_KD3_Environment


class ConditionEncoder(nn.Module):
    def __init__(self, context_dim, model_dims):
        super().__init__()
        self.model_idx = {key: i for i, key in enumerate(model_dims.keys())}
        self.projections = nn.ModuleDict(
            {
                model_name: nn.Sequential(
                    nn.Linear(model_dim, context_dim, bias=False),
                    nn.LayerNorm(context_dim),
                )
                for model_name, model_dim in model_dims.items()
            }
        )

    @abstractmethod
    def encode(self, model_input, model_name):
        pass

    def forward(self, model_inputs):
        context = []
        context_mask = []
        for model_name, model_idx in self.model_idx.items():
            model_input = model_inputs[model_name]
            embeddings = self.encode(model_input, model_name)
            if "attention_mask" in model_input:
                bad_embeddings = (embeddings == 0).all(-1).all(-1)
                model_input["attention_mask"][bad_embeddings] = torch.zeros_like(
                    model_input["attention_mask"][bad_embeddings]
                )
            embeddings = self.projections[model_name](embeddings)
            if "attention_mask" in model_input:
                attention_mask = model_input["attention_mask"]
                embeddings[attention_mask == 0] = torch.zeros_like(
                    embeddings[attention_mask == 0]
                )
                max_seq_length = attention_mask.sum(-1).max() + 1
                embeddings = embeddings[:, :max_seq_length]
                attention_mask = attention_mask[:, :max_seq_length]
            else:
                attention_mask = torch.ones(
                    *embeddings.shape[:-1], dtype=torch.long, device=embeddings.device
                )
            context.append(embeddings)
            context_mask.append((model_idx + 1) * attention_mask)
        context = torch.cat(context, dim=1)
        context_mask = torch.cat(context_mask, dim=1)
        return context, context_mask


class T5TextConditionEncoder(ConditionEncoder):
    def __init__(
        self,
        environment: Model_KD3_Environment,
        model_names,
        context_dim,
        model_dims,
        cache_dir,
        low_cpu_mem_usage: bool = True,
        device_map: Optional[str] = None,
    ):
        super().__init__(context_dim, model_dims)
        if environment.kd30_low_vram:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_type=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=False,
            )
        else:
            quantization_config = None

        t5_model = T5EncoderModel.from_pretrained(
            model_names["t5"],
            low_cpu_mem_usage=low_cpu_mem_usage,
            device_map=device_map,
            # load_in_4bit=environment.kd30_low_vram,
            torch_dtype=torch.bfloat16,
            cache_dir=cache_dir,
            quantization_config=quantization_config,
        )
        self.encoders = nn.ModuleDict(
            {
                "t5": (
                    t5_model.encoder
                    if environment.kd30_low_vram
                    else t5_model.encoder.half()
                )
            }
        )
        self.encoders = freeze(self.encoders)

    @torch.no_grad()
    def encode(self, model_input, model_name):
        embeddings = self.encoders[model_name](**model_input).last_hidden_state
        is_inf_embeddings = torch.isinf(embeddings).any(-1).any(-1)
        is_nan_embeddings = torch.isnan(embeddings).any(-1).any(-1)
        bad_embeddings = is_inf_embeddings + is_nan_embeddings
        embeddings[bad_embeddings] = torch.zeros_like(embeddings[bad_embeddings])
        embeddings = embeddings.type(torch.float32)
        return embeddings


def get_condition_encoder(conf):
    if hasattr(conf, "model_names"):
        model_names = conf.model_names.keys()
        if "t5" in model_names:
            return T5TextConditionEncoder(**conf)
        else:
            raise NotImplementedError("Condition Encoder not implemented")
    else:
        return ConditionEncoder(**conf)
