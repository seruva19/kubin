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
    use_text_embedder_int8_ao_quantization: bool = False

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
        self.use_text_embedder_int8_ao_quantization = (
            "kd50_text_embedder_int8_ao_quantization" in optimization_flags
        )

        return self


def quantize_with_torch_ao(model, freeze=False, save_name=None):
    print("   Applying torchao int8_weight_only quantization...")

    initial_params = list(model.parameters())
    initial_dtype = initial_params[0].dtype if initial_params else None
    initial_size = sum(p.numel() * p.element_size() for p in initial_params) / (
        1024**2
    )  # MB

    tao_quantize(model, tao_quantization())

    quantized_params = list(model.parameters())
    quantized_dtype = quantized_params[0].dtype if quantized_params else None

    actual_storage_size = 0
    int8_params_found = 0
    for p in quantized_params:
        if hasattr(p, "tensor_impl") and hasattr(p.tensor_impl, "data"):
            impl_data = p.tensor_impl.data
            actual_storage_size += impl_data.numel() * impl_data.element_size()
            if impl_data.dtype == torch.int8:
                int8_params_found += 1
        else:
            actual_storage_size += p.numel() * p.element_size()
    actual_storage_size /= 1024**2  # Convert to MB

    has_quantized_tensors = any(
        "LinearActivationQuantizedTensor" in str(type(p))
        or "AffineQuantizedTensor" in str(type(p))
        for p in model.parameters()
    )

    print(f"   → Before: {initial_dtype}, {initial_size:.1f} MB")
    print(
        f"   → After:  {quantized_dtype} (wrapper), actual storage: {actual_storage_size:.1f} MB"
    )

    if has_quantized_tensors and int8_params_found > 0:
        reduction_pct = (1 - actual_storage_size / initial_size) * 100
        print(
            f"   ✓ Quantization verified: Found {int8_params_found} int8 quantized parameters"
        )
        print(
            f"   ✓ Memory savings: {initial_size - actual_storage_size:.1f} MB ({reduction_pct:.1f}% reduction)"
        )
    elif has_quantized_tensors:
        print(
            f"   ⚠️  WARNING: Quantized tensor structure found but no int8 parameters detected"
        )
        print(f"   → Quantization may not have worked correctly")
    else:
        print(f"   ⚠️  WARNING: Quantization failed - no quantized tensor types found")
        print(f"   → Model is still using full precision weights")

    if freeze:
        print("   Freezing quantized weights...")
        pass
    if save_name is not None:
        k_log(
            "⚠️  Cannot (yet) save quantized weights with torch_ao - feature not implemented"
        )
    return model
