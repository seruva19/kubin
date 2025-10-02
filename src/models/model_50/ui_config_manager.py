import os
import json
from typing import Dict, Any, Optional
from omegaconf import OmegaConf
from utils.logging import k_log


class UIConfigManager:
    def __init__(self, base_config_dir: str = None):
        if base_config_dir is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            base_config_dir = os.path.join(current_dir, "configs")

        self.base_config_dir = base_config_dir
        self.ui_config_dir = os.path.join(base_config_dir, "ui")
        self._ensure_directories()

    def _ensure_directories(self):
        os.makedirs(self.ui_config_dir, exist_ok=True)

    def get_ui_config_path(self, variant: str) -> str:
        return os.path.join(self.ui_config_dir, f"config_{variant}_ui.yaml")

    def get_default_config_path(self, variant: str) -> str:
        return os.path.join(self.base_config_dir, f"config_{variant}.yaml")

    def ui_config_exists(self, variant: str) -> bool:
        return os.path.exists(self.get_ui_config_path(variant))

    def load_config(self, variant: str) -> Any:
        ui_config_path = self.get_ui_config_path(variant)
        default_config_path = self.get_default_config_path(variant)

        if self.ui_config_exists(variant):
            k_log(f"Loading UI config for {variant} from {ui_config_path}")
            ui_conf = OmegaConf.load(ui_config_path)

            if "sft" in variant.lower() and not hasattr(ui_conf, "magcache"):
                k_log(
                    f"UI config missing magcache section, loading from default config"
                )
                default_conf = OmegaConf.load(default_config_path)
                if hasattr(default_conf, "magcache"):
                    ui_conf.magcache = default_conf.magcache
                    k_log(f"Added magcache section from default config")

            return ui_conf
        else:
            k_log(f"Loading default config for {variant} from {default_config_path}")
            return OmegaConf.load(default_config_path)

    def save_ui_config(self, variant: str, config_data: Dict[str, Any]):
        ui_config_path = self.get_ui_config_path(variant)

        # Convert to OmegaConf and save
        conf = OmegaConf.create(config_data)
        OmegaConf.save(conf, ui_config_path)

        k_log(f"Saved UI config for {variant} to {ui_config_path}")

    def reset_ui_config(self, variant: str) -> bool:
        ui_config_path = self.get_ui_config_path(variant)

        if os.path.exists(ui_config_path):
            os.remove(ui_config_path)
            k_log(f"Reset UI config for {variant} (deleted {ui_config_path})")
            return True
        else:
            k_log(f"No UI config to reset for {variant}")
            return False

    def build_config_from_ui_params(
        self,
        variant: str,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        duration: int,
        seed: int,
        num_steps: int,
        guidance_weight: float,
        expand_prompts: bool,
        use_offload: bool,
        use_magcache: bool,
        use_dit_int8_ao_quantization: bool,
        use_save_quantized_weights: bool,
        use_text_embedder_int8_ao_quantization: bool,
        use_torch_compile: bool,
        use_flash_attention: bool,
        in_visual_dim: int,
        out_visual_dim: int,
        time_dim: int,
        model_dim: int,
        ff_dim: int,
        num_text_blocks: int,
        num_visual_blocks: int,
        patch_size: str,
        axes_dims: str,
        visual_cond: bool,
        in_text_dim: int,
        in_text_dim2: int,
        attention_type: str,
        attention_causal: bool,
        attention_local: bool,
        attention_glob: bool,
        attention_window: int,
        nabla_P: float,
        nabla_wT: int,
        nabla_wW: int,
        nabla_wH: int,
        nabla_add_sta: bool,
        nabla_method: str,
        qwen_emb_size: int,
        qwen_max_length: int,
        qwen_checkpoint: str,
        clip_emb_size: int,
        clip_max_length: int,
        clip_checkpoint: str,
        vae_checkpoint: str,
        vae_name: str,
        vae_tile_threshold: int,
        model_checkpoint: str,
    ) -> Dict[str, Any]:
        patch_size_list = (
            eval(patch_size) if isinstance(patch_size, str) else patch_size
        )
        axes_dims_list = eval(axes_dims) if isinstance(axes_dims, str) else axes_dims

        config_dict = {
            "metrics": {
                "scale_factor": [1.0, 2.0, 2.0],
                "resolution": 512,
            },
            "ui_settings": {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "width": width,
                "height": height,
                "duration": duration,
                "seed": seed,
                "expand_prompts": expand_prompts,
                "use_offload": use_offload,
                "use_magcache": use_magcache,
                "use_dit_int8_ao_quantization": use_dit_int8_ao_quantization,
                "use_save_quantized_weights": use_save_quantized_weights,
                "use_text_embedder_int8_ao_quantization": use_text_embedder_int8_ao_quantization,
            },
            "optimizations": {
                "use_torch_compile": use_torch_compile,
                "use_flash_attention": use_flash_attention,
            },
            "model": {
                "checkpoint_path": model_checkpoint,
                "num_steps": num_steps,
                "guidance_weight": guidance_weight,
                "duration": duration,
                "dit_params": {
                    "in_visual_dim": in_visual_dim,
                    "out_visual_dim": out_visual_dim,
                    "time_dim": time_dim,
                    "model_dim": model_dim,
                    "ff_dim": ff_dim,
                    "num_text_blocks": num_text_blocks,
                    "num_visual_blocks": num_visual_blocks,
                    "patch_size": patch_size_list,
                    "axes_dims": axes_dims_list,
                    "visual_cond": visual_cond,
                    "in_text_dim": in_text_dim,
                    "in_text_dim2": in_text_dim2,
                },
                "attention": {
                    "type": attention_type,
                    "causal": attention_causal,
                    "local": attention_local,
                    "glob": attention_glob,
                    "window": attention_window,
                    "P": nabla_P,
                    "wT": nabla_wT,
                    "wW": nabla_wW,
                    "wH": nabla_wH,
                    "add_sta": nabla_add_sta,
                    "method": nabla_method,
                },
                "vae": {
                    "checkpoint_path": vae_checkpoint,
                    "name": vae_name,
                    "tile_threshold": vae_tile_threshold,
                },
                "text_embedder": {
                    "qwen": {
                        "emb_size": qwen_emb_size,
                        "max_length": qwen_max_length,
                        "checkpoint_path": qwen_checkpoint,
                    },
                    "clip": {
                        "emb_size": clip_emb_size,
                        "max_length": clip_max_length,
                        "checkpoint_path": clip_checkpoint,
                    },
                },
            },
        }

        if "sft" in variant.lower():
            try:
                default_config_path = self.get_default_config_path(variant)
                if os.path.exists(default_config_path):
                    default_conf = OmegaConf.load(default_config_path)
                    if hasattr(default_conf, "magcache"):
                        config_dict["magcache"] = OmegaConf.to_container(
                            default_conf.magcache
                        )
            except Exception as e:
                k_log(f"Warning: Could not load magcache config: {e}")

        return config_dict
