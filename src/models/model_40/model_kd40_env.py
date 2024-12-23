from dataclasses import dataclass
from typing import Literal
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig

from optimum.quanto import (
    freeze as optq_freeze,
    qfloat8 as optq_qfloat8,
    quantize as optq_quantize,
)
from transformers import BitsAndBytesConfig

import torch
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
    # available_optimization_flags:kd21_flash_attention;kd30_low_vram;kd31_low_vram;kd40_flash_attention;kd40_sage_attention;kd40_t2v_tenc_int8_ao_quantization;kd40_t2v_vae_int8_ao_quantization;kd40_t2v_dit_int8_ao_quantization;kd40_v2a_mm_int8_bnb_quantization;kd40_v2a_mm_int4_bnb_quantization;kd40_v2a_vae_int8_bnb_quantization;kd40_v2a_vae_int4_bnb_quantization;kd40_v2a_unet_int8_bnb_quantization;kd40_v2a_unet_int4_bnb_quantization;kd40_vae_tiling;kd40_vae_slicing;kd40_model_offload;kd40_save_quantized_weights
    use_t2v_tenc_int8_ao_quantization: bool = False
    use_t2v_vae_int8_ao_quantization: bool = False
    use_t2v_dit_int8_ao_quantization: bool = False
    use_t2v_tenc_int8_oq_quantization: bool = False
    use_t2v_vae_int8_oq_quantization: bool = False
    use_t2v_dit_int8_oq_quantization: bool = False

    use_v2a_mm_int8_bnb_quantization: bool = False
    use_v2a_mm_nf4_bnb_quantization: bool = False
    use_v2a_vae_int8_bnb_quantization: bool = False
    use_v2a_vae_nf4_bnb_quantization: bool = False
    use_v2a_unet_int8_bnb_quantization: bool = False
    use_v2a_unet_nf4_bnb_quantization: bool = False

    use_vae_tiling: bool = False
    use_vae_slicing: bool = False
    use_model_offload: bool = False
    use_save_quantized_weights: bool = False

    kd40_conf: DictConfig = None

    def set_conf(self, conf):
        self.kd40_conf = OmegaConf.create(conf)

    def from_config(self, params):
        optimization_flags = [
            value.strip() for value in params("native", "optimization_flags").split(";")
        ]

        self.use_t2v_vae_int8_ao_quantization = (
            "kd40_t2v_vae_int8_ao_quantization" in optimization_flags
        )
        self.use_t2v_tenc_int8_ao_quantization = (
            "kd40_t2v_tenc_int8_ao_quantization" in optimization_flags
        )
        self.use_t2v_dit_int8_ao_quantization = (
            "kd40_t2v_dit_int8_ao_quantization" in optimization_flags
        )

        self.use_t2v_vae_int8_oq_quantization = (
            "kd40_t2v_vae_int8_oq_quantization" in optimization_flags
        )
        self.use_t2v_tenc_int8_oq_quantization = (
            "kd40_t2v_tenc_int8_oq_quantization" in optimization_flags
        )
        self.use_t2v_dit_int8_oq_quantization = (
            "kd40_t2v_dit_int8_oq_quantization" in optimization_flags
        )

        self.use_v2a_mm_int8_bnb_quantization = (
            "kd40_v2a_mm_int8_bnb_quantization" in optimization_flags
        )
        self.use_v2a_mm_nf4_bnb_quantization = (
            "kd40_v2a_mm_nf4_bnb_quantization" in optimization_flags
        )
        self.use_v2a_vae_int8_bnb_quantization = (
            "kd40_v2a_vae_int8_bnb_quantization" in optimization_flags
        )
        self.use_v2a_vae_nf4_bnb_quantization = (
            "kd40_v2a_vae_nf4_bnb_quantization" in optimization_flags
        )
        self.use_v2a_unet_int8_bnb_quantization = (
            "kd40_v2a_unet_int8_bnb_quantization" in optimization_flags
        )
        self.use_v2a_unet_nf4_bnb_quantization = (
            "kd40_v2a_unet_nf4_bnb_quantization" in optimization_flags
        )

        self.use_vae_tiling = "kd40_vae_tiling" in optimization_flags
        self.use_vae_slicing = "kd40_vae_slicing" in optimization_flags
        self.use_model_offload = "kd40_model_offload" in optimization_flags
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


def bnb_config(nf4=False, int8=False):
    if nf4:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            llm_int8_enable_fp32_cpu_offload=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=False,
        )
    elif int8:
        return BitsAndBytesConfig(load_in_8bit=True)
