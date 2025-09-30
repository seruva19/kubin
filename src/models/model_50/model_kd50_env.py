from dataclasses import dataclass
from typing import Literal
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig

import torch
from torchao.quantization import (
    quantize_ as tao_quantize,
    int8_weight_only as tao_int8_weight_only,
)

from utils.logging import k_log

tao_quantization = tao_int8_weight_only


@dataclass
class Model_KD50_Environment:
    use_model_offload: bool = True
    use_magcache: bool = False
    use_dit_int8_ao_quantization: bool = False
    use_save_quantized_weights: bool = False

    kd50_conf: DictConfig = None

    def set_conf(self, conf):
        self.kd50_conf = OmegaConf.create(conf)

    def from_config(self, params):
        optimization_flags = [
            value.strip() for value in params("native", "optimization_flags").split(";")
        ]

        self.use_model_offload = "kd50_model_offload" in optimization_flags
        self.use_magcache = "kd50_magcache" in optimization_flags
        self.use_dit_int8_ao_quantization = (
            "kd50_dit_int8_ao_quantization" in optimization_flags
        )
        self.use_save_quantized_weights = (
            "kd50_save_quantized_weights" in optimization_flags
        )

        return self


def quantize_with_torch_ao(model, freeze=False, save_name=None):
    tao_quantize(model, tao_quantization())
    if freeze:
        pass
    if save_name is not None:
        k_log("cannot (yet) save with torch_ao")
    return model
