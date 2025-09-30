import gc
import os
import random
import re

import numpy as np
import torch
import torchvision
from omegaconf import OmegaConf

from params import KubinParams
from utils.file_system import save_output
from utils.logging import k_log
from utils.env_data import load_env_value

# Set HF environment variables early to ensure consistent cache directory
def set_kd50_cache_env():
    shared_cache_dir = load_env_value("KD50_CACHE_DIR", "./weights")
    cache_dir = os.path.join(shared_cache_dir, "kandinsky-5")
    os.environ['HF_HOME'] = cache_dir
    os.environ['HUGGINGFACE_HUB_CACHE'] = cache_dir
    os.environ['TRANSFORMERS_CACHE'] = cache_dir

# Set environment variables on module import
set_kd50_cache_env()

from models.model_50.utils import get_T2V_pipeline
from models.model_50.t2v_pipeline import Kandinsky5T2VPipeline
from models.model_50.model_kd50_env import Model_KD50_Environment


class Model_KD50:
    def __init__(self, params: KubinParams):
        k_log("using pipeline: native (5.0)")

        self.kparams = params
        self.t2v_pipe: Kandinsky5T2VPipeline | None = None
        self.current_config_name = None

    def prepare_model(self, task, config_name, kd50_conf, use_custom_config=False):
        k_log(f"task queued: {task}")
        assert task in ["text2video"]

        shared_cache_dir = self.kparams("general", "cache_dir")
        shared_cache_dir = load_env_value("KD50_CACHE_DIR", shared_cache_dir)
        cache_dir = os.path.join(shared_cache_dir, "kandinsky-5")

        # Set HF environment variables to ensure consistent cache directory
        os.environ['HF_HOME'] = cache_dir
        os.environ['HUGGINGFACE_HUB_CACHE'] = cache_dir
        os.environ['TRANSFORMERS_CACHE'] = cache_dir

        device = self.kparams("general", "device")

        # Enhanced device verification
        print(f"=== KD50 Device Verification ===")
        print(f"Requested device from config: {device}")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA version: {torch.version.cuda}")

        if torch.cuda.is_available():
            print(f"CUDA device count: {torch.cuda.device_count()}")
            print(f"Current CUDA device: {torch.cuda.current_device()}")
            print(f"CUDA device name: {torch.cuda.get_device_name()}")

            # Test CUDA functionality
            try:
                test_tensor = torch.randn(10, 10).cuda()
                result = test_tensor @ test_tensor
                print(f"CUDA test successful: {result.device}")
                cuda_working = True
            except Exception as e:
                print(f"CUDA test failed: {e}")
                cuda_working = False
        else:
            cuda_working = False

        # Check CUDA availability and adjust device accordingly
        if device == "cuda":
            if not torch.cuda.is_available():
                print("ERROR: CUDA requested but not available!")
                print("This likely means you're not using the virtual environment.")
                print("Please run the application using start.bat or activate the venv first.")
                device = "cpu"
            elif not cuda_working:
                print("ERROR: CUDA available but not working!")
                print("PyTorch CUDA installation may be corrupted.")
                device = "cpu"
            else:
                device = "cuda:0"  # Ensure we use a specific CUDA device
                print("GPU verified and working!")

        environment = Model_KD50_Environment().from_config(self.kparams)
        environment.set_conf(kd50_conf)

        device_map = {
            "dit": torch.device(device),
            "vae": torch.device(device),
            "text_embedder": torch.device(device),
        }

        print(f"Final device_map: {device_map}")
        print(f"=== End Device Verification ===")

        # Final verification - ensure all models will be on correct device
        if device_map["dit"].type == "cuda" and torch.cuda.is_available():
            print("✓ GPU acceleration enabled for KD50 models")
        else:
            print("⚠ Running on CPU - GPU performance not available")

        if task == "text2video":
            if self.t2v_pipe is None or self.current_config_name != config_name:
                self.flush(task)

                k_log(f"preparing K5.0-T2V pipeline with config: {config_name}")

                use_offload = environment.use_model_offload
                use_magcache = environment.use_magcache

                if use_custom_config and config_name in kd50_conf:
                    config_data = kd50_conf[config_name]
                    conf = self._build_config_from_ui(config_data, cache_dir)

                    use_offload = config_data.get(
                        "use_offload", environment.use_model_offload
                    )
                    use_magcache = config_data.get(
                        "use_magcache", environment.use_magcache
                    )
                    k_log(f"using custom config from UI for {config_name}")
                else:
                    config_path = self._get_config_path(config_name)
                    conf = OmegaConf.load(config_path)
                    k_log(f"loaded config from {config_path}")

                k_log(f"offload={use_offload}, magcache={use_magcache}")

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
                    conf_path=None,  # We're passing the conf object directly through the pipeline
                    offload=use_offload,
                    magcache=use_magcache,
                    quantize_dit=environment.use_dit_int8_ao_quantization,
                )

                # Override the pipeline's conf with our custom one
                self.t2v_pipe.conf = conf
                self.t2v_pipe.num_steps = conf.model.num_steps
                self.t2v_pipe.guidance_weight = conf.model.guidance_weight

                self.current_config_name = config_name

    def _build_config_from_ui(self, config_data, cache_dir):
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
        time_length = params["time_length"]

        width = params.get("width", 512)
        height = params.get("height", 512)

        seed = params["seed"]
        generate_image = params.get("generate_image", False)

        num_steps = params.get("num_steps", None)
        guidance_weight = params.get("guidance_weight", None)
        expand_prompts = params.get("expand_prompts", True)

        if generate_image:
            time_length = 0

        # Check if we have custom config data from UI
        use_custom_config = isinstance(kd50_conf, dict) and config_name in kd50_conf
        self.prepare_model(task, config_name, kd50_conf, use_custom_config)

        save_image_path = None
        save_video_path = os.path.join(
            params.get(
                ".output_dir",
                os.path.join(self.kparams("general", "output_dir"), task),
            ),
            f"k5v-{config_name}-{'_'.join(prompt.split()[:5])}.mp4",
        )

        os.makedirs(os.path.dirname(save_video_path), exist_ok=True)

        result = self.t2v_pipe(
            text=prompt,
            save_path=save_video_path if not generate_image else None,
            time_length=time_length,
            width=width,
            height=height,
            seed=None if seed == -1 else seed,
            num_steps=num_steps,
            guidance_weight=guidance_weight,
            expand_prompts=expand_prompts,
            progress=True,
        )

        if generate_image:
            save_image_path = save_images(self.kparams, params, task, result)
            save_video_path = None

        k_log("text2video task: done")

        # Return appropriate format for Gradio
        # If only video, return just the path (not a tuple)
        # If only image, return just the path
        if save_video_path and not save_image_path:
            return save_video_path
        elif save_image_path and not save_video_path:
            return save_image_path
        else:
            # Both or neither - return tuple
            return save_video_path, save_image_path

    def flush(self, task=None):
        cleared = False

        if self.t2v_pipe is not None:
            k_log("flushing K5.0 pipeline")

            # Move components to CPU
            for component in ["text_embedder", "dit", "vae"]:
                comp = getattr(self.t2v_pipe, component, None)
                if comp is not None:
                    try:
                        comp.to("cpu")
                    except Exception as e:
                        k_log(f"Could not move {component} to CPU: {e}")

            self.t2v_pipe = None
            self.current_config_name = None
            cleared = True

        if cleared:
            gc.collect()
            device = self.kparams("general", "device")
            if device.startswith("cuda") and torch.cuda.is_available():
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()
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
