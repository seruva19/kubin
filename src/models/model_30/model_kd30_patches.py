from transformers import T5Tokenizer, CLIPImageProcessor, T5Model, CLIPModel
from typing import Optional
from torch import nn
from models.model_30.kandinsky3.utils import freeze


def patch_kd30_pipelines(k_params):
    from models.model_30.kandinsky3.condition_processors import T5TextConditionProcessor
    from models.model_30.kandinsky3.condition_encoders import T5TextConditionEncoder

    cache_dir = k_params("general", "cache_dir")

    def t5_cond_proc_ctor(self, tokens_length, processor_names):
        self.tokens_length = tokens_length["t5"]
        self.processor = T5Tokenizer.from_pretrained(
            processor_names["t5"], cache_dir=cache_dir
        )

    def t5_cond_enc_ctor(
        self,
        model_names,
        context_dim,
        model_dims,
        low_cpu_mem_usage: bool = True,
        device_map: Optional[str] = None,
    ):
        T5TextConditionEncoder.__base__.__init__(self, context_dim, model_dims)
        t5_model = T5Model.from_pretrained(
            model_names["t5"],
            low_cpu_mem_usage=low_cpu_mem_usage,
            device_map=device_map,
            cache_dir=cache_dir,
        )
        self.encoders = nn.ModuleDict(
            {
                "t5": t5_model.encoder.half(),
            }
        )
        self.encoders = freeze(self.encoders)

    T5TextConditionProcessor.__init__ = t5_cond_proc_ctor
    T5TextConditionEncoder.__init__ = t5_cond_enc_ctor
