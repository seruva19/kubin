# This source code is licensed under the Apache License found in the
# LICENSE file in the current directory.
"""
The code has been adopted from Kandinsky-5
(https://github.com/ai-forever/Kandinsky-5/blob/main/kandinsky/utils.py)
"""


import os
import logging
from typing import Optional, Union

import torch
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from huggingface_hub import hf_hub_download, snapshot_download
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig

from safetensors.torch import load_file


def get_T2V_pipeline(
    device_map: Union[str, torch.device, dict],
    resolution: int = 512,
    cache_dir: str = "./weights/",
    dit_path: str = None,
    text_encoder_path: str = None,
    text_encoder2_path: str = None,
    vae_path: str = None,
    conf_path: str = None,
    conf: DictConfig = None,
    offload: bool = False,
    magcache: bool = False,
    quantize_dit: bool = False,
    quantize_text_embedder: bool = False,
    use_torch_compile: bool = True,
    use_flash_attention: bool = True,
) -> "Kandinsky5T2VPipeline":  # type: ignore
    assert resolution in [512]

    # Set environment variable to disable torch.compile for KD5 only
    # MUST be set BEFORE importing model modules so @kd5_compile decorators see it
    if not use_torch_compile:
        os.environ["KD5_DISABLE_COMPILE"] = "1"
        torch._dynamo.config.disable = True
    else:
        os.environ.pop("KD5_DISABLE_COMPILE", None)
        torch._dynamo.config.suppress_errors = True
        # Disable verbose logging for Triton autotuning and inductor
        os.environ["TRITON_PRINT_AUTOTUNING"] = "0"
        logging.getLogger("torch._dynamo").setLevel(logging.WARNING)
        logging.getLogger("torch._inductor").setLevel(logging.WARNING)
        logging.getLogger("torch.fx").setLevel(logging.WARNING)

    from .models.dit import get_dit
    from .models.text_embedders import get_text_embedder
    from .models.vae import build_vae
    from .models.parallelize import parallelize_dit
    from .t2v_pipeline import Kandinsky5T2VPipeline
    from .magcache_utils import set_magcache_params

    # Set environment variable to control Flash Attention usage
    if use_flash_attention:
        os.environ["KD5_USE_FLASH_ATTENTION"] = "1"
    else:
        os.environ["KD5_USE_FLASH_ATTENTION"] = "0"

    if not isinstance(device_map, dict):
        device_map = {"dit": device_map, "vae": device_map, "text_embedder": device_map}

    try:
        local_rank, world_size = int(os.environ["LOCAL_RANK"]), int(
            os.environ["WORLD_SIZE"]
        )
    except:
        local_rank, world_size = 0, 1

    assert not (
        world_size > 1 and offload
    ), "Offloading available only with not parallel inference"

    if world_size > 1:
        device_mesh = init_device_mesh(
            "cuda", (world_size,), mesh_dim_names=("tensor_parallel",)
        )
        device_map["dit"] = torch.device(f"cuda:{local_rank}")
        device_map["vae"] = torch.device(f"cuda:{local_rank}")
        device_map["text_embedder"] = torch.device(f"cuda:{local_rank}")

    os.makedirs(cache_dir, exist_ok=True)

    # Load config: prioritize passed conf, then conf_path, then default
    if conf is None:
        if conf_path is not None:
            conf = OmegaConf.load(conf_path)
        else:
            conf = None

    # Auto-download VAE if not provided
    if vae_path is None:
        vae_dir = os.path.join(cache_dir, "vae")
        if not os.path.exists(vae_dir) or not os.listdir(vae_dir):
            print("Downloading HunyuanVideo VAE...")
            snapshot_download(
                repo_id="hunyuanvideo-community/HunyuanVideo",
                allow_patterns="vae/*",
                local_dir=cache_dir,
            )
        vae_path = vae_dir + "/"

    # Auto-download text encoders if not provided
    if text_encoder_path is None:
        te_dir = os.path.join(cache_dir, "text_encoder")
        if not os.path.exists(te_dir) or not os.listdir(te_dir):
            print("Downloading Qwen2.5-VL text encoder...")
            snapshot_download(
                repo_id="Qwen/Qwen2.5-VL-7B-Instruct",
                local_dir=te_dir,
            )
        text_encoder_path = te_dir + "/"

    if text_encoder2_path is None:
        te2_dir = os.path.join(cache_dir, "text_encoder2")
        if not os.path.exists(te2_dir) or not os.listdir(te2_dir):
            print("Downloading CLIP text encoder...")
            snapshot_download(
                repo_id="openai/clip-vit-large-patch14",
                local_dir=te2_dir,
            )
        text_encoder2_path = te2_dir + "/"

    # Build or update config
    if conf is None:
        conf = get_default_conf(
            dit_path, vae_path, text_encoder_path, text_encoder2_path
        )

    # When offloading, only load text embedder first
    # VAE and DIT will be loaded later when needed
    if offload:
        print(
            "🔄 DEFERRED LOADING: Offload enabled - loading only text embedder initially"
        )
        print("   → VAE and DIT will be loaded when needed during generation")

    text_embedder = get_text_embedder(
        conf.model.text_embedder, use_torch_compile=use_torch_compile
    )

    # Apply torchao quantization to text embedder if requested
    if quantize_text_embedder:
        from .model_kd50_env import quantize_with_torch_ao

        print("🔧 QUANTIZATION: Applying torchao int8 quantization to Text Embedder...")
        # Get the actual PyTorch models from the wrapper
        qwen_model = text_embedder.embedder.model
        clip_model = text_embedder.clip_embedder.model
        print(
            f"   Qwen model device before quantization: {next(qwen_model.parameters()).device}"
        )
        print(
            f"   CLIP model device before quantization: {next(clip_model.parameters()).device}"
        )

        # Quantize both models on CPU
        text_embedder.embedder.model = quantize_with_torch_ao(qwen_model)
        text_embedder.clip_embedder.model = quantize_with_torch_ao(clip_model)

        print(
            f"   Qwen model device after quantization: {next(text_embedder.embedder.model.parameters()).device}"
        )
        print(
            f"   CLIP model device after quantization: {next(text_embedder.clip_embedder.model.parameters()).device}"
        )
        print("✅ QUANTIZATION: Text Embedder quantization process completed")

        if not offload:
            print("   → Moving quantized text embedders to GPU (offload disabled)")
            target_device = device_map["text_embedder"]
            # Use .to() method to move the entire quantized models at once
            # This preserves torchao's internal quantization state
            text_embedder.embedder.model = text_embedder.embedder.model.to(
                target_device
            )
            text_embedder.clip_embedder.model = text_embedder.clip_embedder.model.to(
                target_device
            )
            print(
                f"   → Quantized Qwen model now on: {next(text_embedder.embedder.model.parameters()).device}"
            )
            print(
                f"   → Quantized CLIP model now on: {next(text_embedder.clip_embedder.model.parameters()).device}"
            )
        else:
            print(
                "   → Quantized text embedders will be moved to GPU during generation using .to() method"
            )
    else:
        print(
            "ℹ️  QUANTIZATION: Text Embedder will use fp16/bf16 (int8 quantization disabled)"
        )

        if not offload:
            text_embedder = text_embedder.to(device=device_map["text_embedder"])

    # Only build VAE and DIT if not offloading
    # When offloading, these will be created lazily during generation
    if not offload:
        vae_low_vram_mode = conf.model.vae.get("low_vram_mode", False)
        vae = build_vae(conf.model.vae, low_vram_mode=vae_low_vram_mode)
        vae = vae.eval()
        vae = vae.to(device=device_map["vae"])
        dit = get_dit(conf.model.dit_params)
    else:
        vae = None
        dit = None

    # Skip magcache and DIT loading/quantization if offload is enabled
    # These will be done during generation
    if not offload:
        if (
            magcache
            and hasattr(conf, "magcache")
            and hasattr(conf.magcache, "mag_ratios")
        ):
            mag_ratios = conf.magcache.mag_ratios
            num_steps = conf.model.num_steps
            guidance_weight = conf.model.guidance_weight
            # Detect no_cfg models: guidance_weight == 1.0
            no_cfg = guidance_weight == 1.0

            # Check if calibration mode is enabled via environment variable
            calibrate_mode = os.environ.get("MAGCACHE_CALIBRATE", "0") == "1"

            set_magcache_params(
                dit, mag_ratios, num_steps, no_cfg, calibrate=calibrate_mode
            )

            if not calibrate_mode:
                mag_ratio_count = (
                    len(conf.magcache.mag_ratios)
                    if hasattr(conf.magcache, "mag_ratios")
                    else 0
                )
                print(f"✓ Magcache configuration complete")
                print(f"   → Guidance weight: {guidance_weight}")
                print(f"   → Mag ratios loaded: {mag_ratio_count}")
        elif magcache:
            print(
                "⚠️  Magcache requested but config missing 'magcache.mag_ratios' section"
            )
            print(f"   → Config type: {type(conf)}")
            print(f"   → Has magcache attr: {hasattr(conf, 'magcache')}")
            if hasattr(conf, "magcache"):
                print(f"   → Has mag_ratios: {hasattr(conf.magcache, 'mag_ratios')}")
            print(f"   → Magcache is only available for SFT models with proper config")
            print(
                f"   → Try using Reset button to reload config with magcache parameters"
            )
        else:
            # Ensure magcache is disabled if it was previously enabled
            from .magcache_utils import disable_magcache

            disable_magcache(dit)

        # Download DIT model if it's a Hugging Face repository
        # Check if it's a valid HF repo ID (contains "/" but doesn't look like a Windows path)
        checkpoint_path = conf.model.checkpoint_path
        is_windows_path = (
            len(checkpoint_path) > 1 and checkpoint_path[1] == ":"
        )  # Check for drive letter like "C:"
        is_unix_path = checkpoint_path.startswith("/") or checkpoint_path.startswith(
            "./"
        )
        is_hf_repo = "/" in checkpoint_path and not is_windows_path and not is_unix_path

        if is_hf_repo:
            cache_dir = os.environ.get("KD50_CACHE_DIR", "./weights/")
            cache_dir = os.path.abspath(
                os.path.normpath(cache_dir)
            )  # Ensure absolute normalized path
            print(f"Downloading DIT model to: {cache_dir}")

            # Set HF_HOME to ensure consistent cache directory
            os.environ["HF_HOME"] = cache_dir
            os.environ["HUGGINGFACE_HUB_CACHE"] = cache_dir

            model_path = snapshot_download(
                repo_id=conf.model.checkpoint_path,
                cache_dir=cache_dir,
            )
            checkpoint_path = model_path
            print(f"Model downloaded to: {checkpoint_path}")
        else:
            checkpoint_path = conf.model.checkpoint_path
            checkpoint_path = os.path.abspath(checkpoint_path)  # Ensure absolute path
            print(f"Using local checkpoint path: {checkpoint_path}")

        possible_filenames = [
            "model.safetensors",
            "kandinsky5lite_t2v_distilled16steps_5s.safetensors",
            "kandinsky5lite_t2v_distilled16steps_10s.safetensors",
            "kandinsky5lite_t2v_sft_5s.safetensors",
            "kandinsky5lite_t2v_pretrain_5s.safetensors",
            "kandinsky5lite_t2v_distil_5s.safetensors",
            "kandinsky5lite_t2v_nocfg_5s.safetensors",
            "kandinsky5lite_t2v_sft_10s.safetensors",
            "kandinsky5lite_t2v_pretrain_10s.safetensors",
            "kandinsky5lite_t2v_distil_10s.safetensors",
            "kandinsky5lite_t2v_nocfg_10s.safetensors",
        ]

        full_model_path = None
        for filename in possible_filenames:
            test_path = os.path.join(checkpoint_path, filename)
            if os.path.exists(test_path):
                full_model_path = test_path
                print(f"Found model file: {full_model_path}")
                break

            test_path_in_model = os.path.join(checkpoint_path, "model", filename)
            if os.path.exists(test_path_in_model):
                full_model_path = test_path_in_model
                print(f"Found model file in model subdirectory: {full_model_path}")
                break

        if full_model_path is None:
            try:
                if os.path.exists(checkpoint_path):
                    files = os.listdir(checkpoint_path)
                    print(f"Files available in {checkpoint_path}: {files}")
                    model_subdir = os.path.join(checkpoint_path, "model")
                    if os.path.exists(model_subdir):
                        model_files = os.listdir(model_subdir)
                        print(f"Files available in {model_subdir}: {model_files}")
            except Exception as e:
                print(f"Could not list files: {e}")
            raise FileNotFoundError(
                f"No model file found in {checkpoint_path}. Tried: {possible_filenames}"
            )

        # Load weights - for CUDA, load directly to GPU to save one copy operation
        target_device = device_map["dit"]
        if target_device.type == "cuda":
            print(f"   → Loading DIT weights directly to {target_device}")
            state_dict = load_file(full_model_path, device=str(target_device))
            dit.load_state_dict(state_dict, assign=True)
            # Parameters are now on GPU, but buffers are still on CPU
            # .to() is smart - it only moves what's not already there
            dit = dit.to(target_device)
        else:
            state_dict = load_file(full_model_path)
            dit.load_state_dict(state_dict, assign=True)

        if quantize_dit:
            from .model_kd50_env import quantize_with_torch_ao

            print("🔧 QUANTIZATION: Applying torchao int8 quantization to DIT model...")
            print(f"   DIT device before quantization: {next(dit.parameters()).device}")

            dit = quantize_with_torch_ao(dit)

            print(f"   DIT device after quantization: {next(dit.parameters()).device}")
            print("✅ QUANTIZATION: DIT quantization process completed")
            print(f"   → Quantized DIT ready on: {next(dit.parameters()).device}")
        else:
            print(
                "ℹ️  QUANTIZATION: DIT model will use fp16 (int8 quantization disabled)"
            )
            print(f"   → DIT ready on: {next(dit.parameters()).device}")

    if world_size > 1 and not offload:
        dit = parallelize_dit(dit, device_mesh["tensor_parallel"])

    pipeline = Kandinsky5T2VPipeline(
        device_map=device_map,
        dit=dit,
        text_embedder=text_embedder,
        vae=vae,
        resolution=resolution,
        local_dit_rank=local_rank,
        world_size=world_size,
        conf=conf,
        offload=offload,
    )

    # Store deferred loading parameters on the pipeline for use during generation
    if offload:
        pipeline._deferred_loading_params = {
            "magcache": magcache,
            "quantize_dit": quantize_dit,
            "use_torch_compile": use_torch_compile,
        }

    if not use_torch_compile:
        print("   → DIT model: torch.compile disabled")
        print("   → Text embedder: torch.compile disabled")
        print("   → Magcache: torch.compile disabled")
        print("   → All NN layers: torch.compile disabled")

    return pipeline


def get_default_conf(
    dit_path,
    vae_path,
    text_encoder_path,
    text_encoder2_path,
) -> DictConfig:
    dit_params = {
        "in_visual_dim": 16,
        "out_visual_dim": 16,
        "time_dim": 512,
        "patch_size": [1, 2, 2],
        "model_dim": 1792,
        "ff_dim": 7168,
        "num_text_blocks": 2,
        "num_visual_blocks": 32,
        "axes_dims": [16, 24, 24],
        "visual_cond": True,
        "in_text_dim": 3584,
        "in_text_dim2": 768,
    }

    attention = {
        "type": "flash",
        "causal": False,
        "local": False,
        "glob": False,
        "window": 3,
    }

    vae = {
        "checkpoint_path": vae_path,
        "name": "hunyuan",
    }

    text_embedder = {
        "qwen": {
            "emb_size": 3584,
            "checkpoint_path": text_encoder_path,
            "max_length": 256,
        },
        "clip": {
            "checkpoint_path": text_encoder2_path,
            "emb_size": 768,
            "max_length": 77,
        },
    }

    conf = {
        "model": {
            "checkpoint_path": dit_path,
            "vae": vae,
            "text_embedder": text_embedder,
            "dit_params": dit_params,
            "attention": attention,
            "num_steps": 50,
            "guidance_weight": 5.0,
        },
        "metrics": {"scale_factor": (1, 2, 2)},
        "resolution": 512,
    }

    return DictConfig(conf)
