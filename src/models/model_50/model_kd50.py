import gc
import os
import random
import re
import json
from datetime import datetime

import numpy as np
import torch
import torchvision
from omegaconf import OmegaConf

from params import KubinParams
from utils.file_system import save_output
from utils.logging import k_log
from utils.env_data import load_env_value


def set_kd50_cache_env():
    shared_cache_dir = load_env_value("KD50_CACHE_DIR", "./weights")
    os.environ["HF_HOME"] = shared_cache_dir
    os.environ["HUGGINGFACE_HUB_CACHE"] = shared_cache_dir
    os.environ["TRANSFORMERS_CACHE"] = shared_cache_dir


set_kd50_cache_env()

if os.environ.get("KD5_DISABLE_COMPILE") == "1":
    torch._dynamo.config.disable = True
    torch._dynamo.config.suppress_errors = True

from models.model_50.utils import get_T2V_pipeline
from models.model_50.t2v_pipeline import Kandinsky5T2VPipeline
from models.model_50.model_kd50_env import Model_KD50_Environment
from models.model_50.ui_config_manager import UIConfigManager


class Model_KD50:
    def __init__(self, params: KubinParams):
        k_log("using pipeline: native (5.0)")

        self.kparams = params
        self.t2v_pipe: Kandinsky5T2VPipeline | None = None
        self.current_config_name = None
        self.current_torch_compile_state = None
        self.config_manager = UIConfigManager()

    def prepare_model(self, task, config_name, kd50_conf, use_custom_config=False):
        k_log(f"task queued: {task}")
        assert task in ["text2video"]

        shared_cache_dir = self.kparams("general", "cache_dir")
        shared_cache_dir = load_env_value("KD50_CACHE_DIR", shared_cache_dir)
        cache_dir = shared_cache_dir

        os.environ["HF_HOME"] = cache_dir
        os.environ["HUGGINGFACE_HUB_CACHE"] = cache_dir
        os.environ["TRANSFORMERS_CACHE"] = cache_dir

        device = self.kparams("general", "device")

        print(f"Requested device from config: {device}")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA version: {torch.version.cuda}")

        if torch.cuda.is_available():
            print(f"CUDA device count: {torch.cuda.device_count()}")
            print(f"Current CUDA device: {torch.cuda.current_device()}")
            print(f"CUDA device name: {torch.cuda.get_device_name()}")

            try:
                test_tensor = torch.randn(10, 10).cuda()
                result = test_tensor @ test_tensor
                cuda_working = True
            except Exception as e:
                cuda_working = False
        else:
            cuda_working = False

        if device == "cuda":
            if not torch.cuda.is_available():
                print("ERROR: CUDA requested but not available!")
                print("This likely means you're not using the virtual environment.")
                print(
                    "Please run the application using start.bat or activate the venv first."
                )
                device = "cpu"
            elif not cuda_working:
                print("ERROR: CUDA available but not working!")
                print("PyTorch CUDA installation may be corrupted.")
                device = "cpu"
            else:
                device = "cuda:0"

        environment = Model_KD50_Environment().from_config(self.kparams)
        environment.set_conf(kd50_conf)

        device_map = {
            "dit": torch.device(device),
            "vae": torch.device(device),
            "text_embedder": torch.device(device),
        }

        if device_map["dit"].type == "cuda" and torch.cuda.is_available():
            print("✓ GPU acceleration enabled for KD50 models")
        else:
            print("⚠ Running on CPU - GPU performance not available")

        if task == "text2video":
            # First, determine the torch_compile settings to check if they changed
            check_torch_compile_dit = None
            check_torch_compile_vae = None
            if use_custom_config and config_name in kd50_conf:
                config_data = kd50_conf[config_name]
                # Get torch_compile from UI config
                check_torch_compile_dit = config_data.get("use_torch_compile_dit", True)
                check_torch_compile_vae = config_data.get("use_torch_compile_vae", True)
            else:
                # Load config to check torch_compile setting
                temp_conf = self.config_manager.load_config(config_name)
                check_torch_compile_dit = (
                    getattr(temp_conf.optimizations, "use_torch_compile_dit", True)
                    if hasattr(temp_conf, "optimizations")
                    else True
                )
                check_torch_compile_vae = (
                    getattr(temp_conf.optimizations, "use_torch_compile_vae", True)
                    if hasattr(temp_conf, "optimizations")
                    else True
                )

            # Store as tuple for comparison
            check_torch_compile = (check_torch_compile_dit, check_torch_compile_vae)

            # Force reload if torch_compile state changed
            torch_compile_changed = (
                self.current_torch_compile_state is not None
                and self.current_torch_compile_state != check_torch_compile
            )

            if torch_compile_changed:
                k_log(
                    f"⚠️  torch.compile setting changed from {self.current_torch_compile_state} to {check_torch_compile} - forcing model reload"
                )

            if (
                self.t2v_pipe is None
                or self.current_config_name != config_name
                or torch_compile_changed
            ):
                self.flush(task)

                k_log(f"preparing K5.0-T2V pipeline with config: {config_name}")

                use_offload = environment.use_model_offload
                use_magcache = environment.use_magcache

                if use_custom_config and config_name in kd50_conf:
                    config_data = kd50_conf[config_name]
                    conf = self._build_config_from_ui(
                        config_data, cache_dir, config_name
                    )

                    use_offload = config_data.get(
                        "use_offload", environment.use_model_offload
                    )
                    use_magcache = config_data.get(
                        "use_magcache", environment.use_magcache
                    )
                    use_dit_int8_ao_quantization = config_data.get(
                        "use_dit_int8_ao_quantization",
                        environment.use_dit_int8_ao_quantization,
                    )
                    use_save_quantized_weights = config_data.get(
                        "use_save_quantized_weights",
                        environment.use_save_quantized_weights,
                    )
                    use_text_embedder_int8_ao_quantization = config_data.get(
                        "use_text_embedder_int8_ao_quantization",
                        environment.use_text_embedder_int8_ao_quantization,
                    )
                    k_log(f"using custom config from UI for {config_name}")
                else:
                    conf = self.config_manager.load_config(config_name)
                    k_log(f"loaded config for {config_name} via UIConfigManager")

                    if hasattr(conf, "ui_settings"):
                        use_offload = getattr(
                            conf.ui_settings,
                            "use_offload",
                            environment.use_model_offload,
                        )
                        use_magcache = getattr(
                            conf.ui_settings, "use_magcache", environment.use_magcache
                        )
                        use_dit_int8_ao_quantization = getattr(
                            conf.ui_settings,
                            "use_dit_int8_ao_quantization",
                            environment.use_dit_int8_ao_quantization,
                        )
                        use_save_quantized_weights = getattr(
                            conf.ui_settings,
                            "use_save_quantized_weights",
                            environment.use_save_quantized_weights,
                        )
                        use_text_embedder_int8_ao_quantization = getattr(
                            conf.ui_settings,
                            "use_text_embedder_int8_ao_quantization",
                            environment.use_text_embedder_int8_ao_quantization,
                        )
                        k_log(f"using UI settings from saved config for {config_name}")
                    else:
                        use_dit_int8_ao_quantization = (
                            environment.use_dit_int8_ao_quantization
                        )
                        use_save_quantized_weights = (
                            environment.use_save_quantized_weights
                        )
                        use_text_embedder_int8_ao_quantization = (
                            environment.use_text_embedder_int8_ao_quantization
                        )

                k_log(
                    f"offload={use_offload}, magcache={use_magcache}, dit_int8_quantization={use_dit_int8_ao_quantization}, save_quantized={use_save_quantized_weights}, text_embedder_int8_quantization={use_text_embedder_int8_ao_quantization}"
                )

                if use_offload and (
                    use_dit_int8_ao_quantization
                    or use_text_embedder_int8_ao_quantization
                ):
                    k_log("ℹ️  Quantization + Offloading enabled")

                use_torch_compile_dit = (
                    getattr(conf.optimizations, "use_torch_compile_dit", True)
                    if hasattr(conf, "optimizations")
                    else True
                )
                use_torch_compile_vae = (
                    getattr(conf.optimizations, "use_torch_compile_vae", True)
                    if hasattr(conf, "optimizations")
                    else True
                )
                use_flash_attention = (
                    getattr(conf.optimizations, "use_flash_attention", True)
                    if hasattr(conf, "optimizations")
                    else True
                )

                if not use_torch_compile_dit:
                    k_log("⚙️  torch.compile (DiT) disabled")
                if not use_torch_compile_vae:
                    k_log("⚙️  torch.compile (VAE) disabled")
                if not use_flash_attention:
                    k_log("⚙️  Flash Attention disabled, using PyTorch native SDPA")

                self.t2v_pipe = get_T2V_pipeline(
                    device_map=device_map,
                    resolution=512,
                    cache_dir=cache_dir,
                    dit_path=(
                        conf.model.checkpoint_path
                        if hasattr(conf.model, "checkpoint_path")
                        else None
                    ),
                    text_encoder_path=(
                        conf.model.text_embedder.qwen.checkpoint_path
                        if hasattr(conf.model, "text_embedder")
                        else None
                    ),
                    text_encoder2_path=(
                        conf.model.text_embedder.clip.checkpoint_path
                        if hasattr(conf.model, "text_embedder")
                        else None
                    ),
                    vae_path=(
                        conf.model.vae.checkpoint_path
                        if hasattr(conf.model, "vae")
                        else None
                    ),
                    conf=conf,  # Pass the conf object directly so magcache settings are preserved
                    offload=use_offload,
                    magcache=use_magcache,
                    quantize_dit=use_dit_int8_ao_quantization,
                    quantize_text_embedder=use_text_embedder_int8_ao_quantization,
                    use_torch_compile_dit=use_torch_compile_dit,
                    use_torch_compile_vae=use_torch_compile_vae,
                    use_flash_attention=use_flash_attention,
                )

                self.t2v_pipe.conf = conf
                self.t2v_pipe.num_steps = conf.model.num_steps
                self.t2v_pipe.guidance_weight = conf.model.guidance_weight

                self.current_config_name = config_name
                self.current_torch_compile_state = (
                    use_torch_compile_dit,
                    use_torch_compile_vae,
                )

    def _build_config_from_ui(self, config_data, cache_dir, config_name):
        conf_dict = {
            "model": {
                "checkpoint_path": config_data.get(
                    "checkpoint_path",
                    f"{cache_dir}/model/kandinsky5lite_t2v_{self.current_config_name}.safetensors",
                ),
                "num_steps": int(config_data.get("num_steps", 50)),
                "guidance_weight": float(config_data.get("guidance_weight", 5.0)),
                "dit_params": config_data.get("dit_params", {}),
                "attention": config_data.get("attention", {}),
                "vae": config_data.get("vae", {}),
                "text_embedder": config_data.get("text_embedder", {}),
            },
            "metrics": {
                "scale_factor": [1.0, 2.0, 2.0],
                "resolution": 512,
            },
        }

        if "sft" in config_name.lower():
            base_config_path = self._get_config_path(config_name)
            if os.path.exists(base_config_path):
                base_conf = OmegaConf.load(base_config_path)
                if hasattr(base_conf, "magcache"):
                    conf_dict["magcache"] = OmegaConf.to_container(
                        base_conf.magcache, resolve=True
                    )

        return OmegaConf.create(conf_dict)

    def _get_config_path(self, config_name):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        configs_dir = os.path.join(current_dir, "configs")

        config_map = {
            "5s_sft": "config_5s_sft.yaml",
            "5s_pretrain": "config_5s_pretrain.yaml",
            "5s_nocfg": "config_5s_nocfg.yaml",
            "5s_distil": "config_5s_distil.yaml",
            "10s_sft": "config_10s_sft.yaml",
            "10s_pretrain": "config_10s_pretrain.yaml",
            "10s_nocfg": "config_10s_nocfg.yaml",
            "10s_distil": "config_10s_distil.txt",
        }

        config_file = config_map.get(config_name, "config_5s_sft.yaml")
        return os.path.join(configs_dir, config_file)

    def t2v(self, params):
        task = "text2video"

        config_name = params["pipeline_args"].get("config_name", "5s_sft")
        kd50_conf = params["pipeline_args"].get("kd50_conf", {})

        prompt = params["prompt"]
        negative_prompt = params.get(
            "negative_prompt",
            "Static, 2D cartoon, cartoon, 2d animation, paintings, images, worst quality, low quality, ugly, deformed, walking backwards",
        )
        time_length = params["time_length"]

        width = params.get("width", 512)
        height = params.get("height", 512)

        seed = params["seed"]
        generate_image = params.get("generate_image", False)

        num_steps = params.get("num_steps", None)
        guidance_weight = params.get("guidance_weight", None)
        expand_prompts = params.get("expand_prompts", False)

        enhance_enable = params.get("enhance_enable", False)
        enhance_weight = params.get("enhance_weight", 3.4)
        enhance_max_tokens = params.get("enhance_max_tokens", 0)
        try:
            enhance_weight = float(enhance_weight)
        except (TypeError, ValueError):
            enhance_weight = 3.4
        try:
            enhance_max_tokens_int = int(enhance_max_tokens)
        except (TypeError, ValueError):
            enhance_max_tokens_int = 0
        if enhance_max_tokens_int < 0:
            enhance_max_tokens_int = 0
        enhance_options = {
            "enabled": bool(enhance_enable),
            "weight": enhance_weight,
            "max_tokens": (
                enhance_max_tokens_int if enhance_max_tokens_int > 0 else None
            ),
        }
        if not enhance_options["enabled"]:
            enhance_options = None
        elif enhance_options:
            max_tokens_desc = (
                enhance_options["max_tokens"]
                if enhance_options["max_tokens"] is not None
                else "auto"
            )
            k_log(
                f"🪄  Enhance-A-Video enabled (weight={enhance_options['weight']:.2f}, max_tokens={max_tokens_desc})"
            )

        if generate_image:
            time_length = 0
            enhance_options = None

        use_custom_config = False  # Old system disabled, use YAML configs only
        self.prepare_model(task, config_name, kd50_conf, use_custom_config)

        save_image_path = None
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_video_path = os.path.join(
            params.get(
                ".output_dir",
                os.path.join(self.kparams("general", "output_dir"), task),
            ),
            f"k5v-{config_name}-{timestamp}-{'_'.join(prompt.split()[:5])}.mp4",
        )

        os.makedirs(os.path.dirname(save_video_path), exist_ok=True)

        result = self.t2v_pipe(
            text=prompt,
            negative_caption=negative_prompt,
            save_path=save_video_path if not generate_image else None,
            time_length=time_length,
            width=width,
            height=height,
            seed=None if seed == -1 else seed,
            num_steps=num_steps,
            guidance_weight=guidance_weight,
            expand_prompts=expand_prompts,
            progress=True,
            magcache=params.get("magcache", None),
            enhance_options=enhance_options,
        )

        # Extract expanded prompt and actual result from dict
        expanded_prompt = None
        actual_result = result
        if isinstance(result, dict):
            expanded_prompt = result.get("expanded_prompt")
            actual_result = result.get("path") or result.get("video")
            if expanded_prompt:
                k_log(f"Expanded prompt: {expanded_prompt}")

        result = actual_result

        if generate_image:
            save_image_path = save_images(self.kparams, params, task, result)
            save_video_path = None
        else:
            if save_video_path and os.path.exists(save_video_path):
                try:
                    from mutagen.mp4 import MP4, MP4Tags

                    video = MP4(save_video_path)

                    metadata_str = f"Prompt: {prompt}\n"
                    if expanded_prompt:
                        metadata_str += f"Expanded Prompt: {expanded_prompt}\n"
                    metadata_str += f"Negative Prompt: {negative_prompt}\n"
                    metadata_str += f"Model: {config_name}\n"
                    metadata_str += f"Resolution: {width}x{height}\n"
                    metadata_str += f"Duration: {time_length}s\n"
                    metadata_str += f"Seed: {seed if seed != -1 else 'random'}\n"
                    metadata_str += f"Steps: {num_steps if num_steps else 'default'}\n"
                    metadata_str += f"Guidance Weight: {guidance_weight if guidance_weight else 'default'}\n"
                    metadata_str += f"Expand Prompts: {expand_prompts}\n"
                    metadata_str += f"Generated: {timestamp}"

                    video["\xa9cmt"] = metadata_str  # Comment field
                    video["\xa9des"] = (
                        expanded_prompt if expanded_prompt else prompt
                    )  # Description field
                    video.save()

                    k_log(f"Embedded metadata into {save_video_path}")
                except ImportError:
                    k_log(
                        "mutagen not installed - saving metadata to JSON file instead"
                    )
                    metadata_path = save_video_path.replace(".mp4", ".json")
                    metadata = {
                        "prompt": prompt,
                        "expanded_prompt": expanded_prompt,
                        "negative_prompt": negative_prompt,
                        "config_variant": config_name,
                        "width": width,
                        "height": height,
                        "duration": time_length,
                        "seed": seed if seed != -1 else "random",
                        "num_steps": num_steps if num_steps else "default",
                        "guidance_weight": (
                            guidance_weight if guidance_weight else "default"
                        ),
                        "expand_prompts": expand_prompts,
                        "timestamp": timestamp,
                    }
                    with open(metadata_path, "w", encoding="utf-8") as f:
                        json.dump(metadata, f, indent=2, ensure_ascii=False)
                    k_log(f"Saved video metadata to {metadata_path}")
                except Exception as e:
                    k_log(f"Failed to save video metadata: {e}")

        k_log("text2video task: done")

        if save_video_path and not save_image_path:
            return save_video_path
        elif save_image_path and not save_video_path:
            return save_image_path
        else:
            return save_video_path, save_image_path

    def flush(self, task=None):
        cleared = False

        if self.t2v_pipe is not None:
            k_log("flushing K5.0 pipeline")

            for component in ["text_embedder", "dit", "vae"]:
                comp = getattr(self.t2v_pipe, component, None)
                if comp is not None:
                    try:
                        comp.to("cpu")
                        torch.cuda.empty_cache()
                    except Exception as e:
                        k_log(f"Could not move {component} to CPU: {e}")

                    try:
                        setattr(self.t2v_pipe, component, None)
                    except Exception as e:
                        k_log(f"Could not clear {component} reference: {e}")

            self.t2v_pipe = None
            self.current_config_name = None
            cleared = True

        if cleared:
            # Aggressive garbage collection
            gc.collect()
            gc.collect()  # Run twice to clear circular references

            device = self.kparams("general", "device")
            if device.startswith("cuda") and torch.cuda.is_available():
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            k_log("K5.0 pipeline flushed")


def save_images(kparams, params, task, batch):
    if params.get("task", None) is None:
        params["task"] = task

    output_dir = params.get(
        ".output_dir",
        os.path.join(kparams("general", "output_dir"), task),
    )

    saved_batch = save_output(output_dir, batch, params)
    return saved_batch
