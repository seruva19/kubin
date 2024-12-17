from dataclasses import dataclass
from typing import Literal
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig


@dataclass
class Model_KD40_Environment:
    use_textencoder_fp8_quantization: bool = False
    use_vae_fp8_quantization: bool = False
    use_dit_fp8_quantization: bool = False
    use_vae_tiling: bool = False
    use_vae_slicing: bool = False
    use_model_offload: bool = False

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

        return self
