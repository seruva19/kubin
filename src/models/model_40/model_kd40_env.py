from dataclasses import dataclass
from typing import Literal
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig

from optimum.quanto import (
    freeze as optq_freeze,
    qfloat8 as optq_qfloat8,
    quantize as optq_quantize,
)


from torchao.quantization import (
    quantize_ as tao_quantize,
    int8_weight_only as tao_int8_weight_only,
    int4_weight_only as tao_int4_weight_only,
    int8_dynamic_activation_int8_weight as tao_int8_dynamic_activation_int8_weight,
)

from utils.logging import k_log

tao_quantization = tao_int8_weight_only
optq_quantization = optq_qfloat8


@dataclass
class Model_KD40_Environment:
    use_textencoder_fp8_quantization: bool = False
    use_vae_fp8_quantization: bool = False
    use_dit_fp8_quantization: bool = False
    use_vae_tiling: bool = False
    use_vae_slicing: bool = False
    use_model_offload: bool = False
    use_torchao_quantization: bool = False
    use_save_quantized_weights: bool = False

    kd40_conf: DictConfig = None

    def set_conf(self, conf):
        self.kd40_conf = OmegaConf.create(conf)

    def from_config(self, params):
        optimization_flags = [
            value.strip() for value in params("native", "optimization_flags").split(";")
        ]

        self.use_textencoder_fp8_quantization = (
            "kd40_fp8_tenc_ao_quantization" in optimization_flags
        )
        self.use_vae_fp8_quantization = (
            "kd40_fp8_vae_ao_quantization" in optimization_flags
        )
        self.use_dit_fp8_quantization = (
            "kd40_fp8_dit_ao_quantization" in optimization_flags
        )
        self.use_vae_tiling = "kd40_vae_tiling" in optimization_flags
        self.use_vae_slicing = "kd40_vae_slicing" in optimization_flags
        self.use_model_offload = "kd40_model_offload" in optimization_flags
        self.use_torchao_quantization = (
            "kd40_torchao_quantization" in optimization_flags
        )
        self.use_save_quantized_weights = (
            "kd40_save_quantized_weights" in optimization_flags
        )

        return self


def quantize_with_torch_ao(model, freeze=False, save_name=None):
    tao_quantize(model, tao_quantization())
    if freeze:
        # k_log("cannot (yet) freeze with torch_ao")
        pass
    if save_name is not None:
        k_log("cannot (yet) save with torch_ao")
    return model


def quantize_with_optimum_quanto(model, freeze=False, save_name=None):
    optq_quantize(model, weights=optq_qfloat8)
    if freeze:
        k_log("freezing...")
        optq_freeze(model)
    if save_name is not None:
        k_log("saving to " + save_name)
        model.save_pretrained(save_name)
    return model
