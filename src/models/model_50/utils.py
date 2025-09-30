# This source code is licensed under the Apache License found in the
# LICENSE file in the current directory.
"""
The code has been adopted from Kandinsky-5
(https://github.com/ai-forever/Kandinsky-5/blob/main/kandinsky/utils.py)
"""


import os
from typing import Optional, Union

import torch
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from huggingface_hub import hf_hub_download, snapshot_download
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig

from .models.dit import get_dit
from .models.text_embedders import get_text_embedder
from .models.vae import build_vae
from .models.parallelize import parallelize_dit
from .t2v_pipeline import Kandinsky5T2VPipeline
from .magcache_utils import set_magcache_params

from safetensors.torch import load_file

torch._dynamo.config.suppress_errors = True


def get_T2V_pipeline(
    device_map: Union[str, torch.device, dict],
    resolution: int = 512,
    cache_dir: str = "./weights/",
    dit_path: str = None,
    text_encoder_path: str = None,
    text_encoder2_path: str = None,
    vae_path: str = None,
    conf_path: str = None,
    offload: bool = False,
    magcache: bool = False,
    quantize_dit: bool = False,
) -> Kandinsky5T2VPipeline:
    assert resolution in [512]

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

    # Load config first to determine which model to download
    if conf_path is not None:
        conf = OmegaConf.load(conf_path)
    else:
        conf = None

    # Auto-download DiT model if not provided
    if dit_path is None:
        if conf is not None and hasattr(conf, 'model') and hasattr(conf.model, 'checkpoint_path'):
            # Extract model info from config checkpoint path
            checkpoint_filename = os.path.basename(conf.model.checkpoint_path)
            # Map checkpoint filename to HuggingFace repo ID
            model_map = {
                "kandinsky5lite_t2v_sft_5s.safetensors": "ai-forever/Kandinsky-5.0-T2V-Lite-sft-5s",
                "kandinsky5lite_t2v_sft_10s.safetensors": "ai-forever/Kandinsky-5.0-T2V-Lite-sft-10s",
                "kandinsky5lite_t2v_pretrain_5s.safetensors": "ai-forever/Kandinsky-5.0-T2V-Lite-pretrain-5s",
                "kandinsky5lite_t2v_pretrain_10s.safetensors": "ai-forever/Kandinsky-5.0-T2V-Lite-pretrain-10s",
                "kandinsky5lite_t2v_nocfg_5s.safetensors": "ai-forever/Kandinsky-5.0-T2V-Lite-nocfg-5s",
                "kandinsky5lite_t2v_nocfg_10s.safetensors": "ai-forever/Kandinsky-5.0-T2V-Lite-nocfg-10s",
                "kandinsky5lite_t2v_distil_5s.safetensors": "ai-forever/Kandinsky-5.0-T2V-Lite-distilled16steps-5s",
                "kandinsky5lite_t2v_distil_10s.safetensors": "ai-forever/Kandinsky-5.0-T2V-Lite-distilled16steps-10s",
            }

            repo_id = model_map.get(checkpoint_filename)
            if repo_id:
                model_path = os.path.join(cache_dir, "model", checkpoint_filename)
                if not os.path.exists(model_path):
                    print(f"Downloading {checkpoint_filename} from {repo_id}...")
                    snapshot_download(
                        repo_id=repo_id,
                        allow_patterns="model/*",
                        local_dir=cache_dir,
                    )
                dit_path = model_path
            else:
                print(f"Unknown model: {checkpoint_filename}, will try to use existing file")
                dit_path = conf.model.checkpoint_path
        else:
            # Default to sft_5s if no config provided
            print("Downloading Kandinsky 5.0 T2V Lite SFT 5s model (default)...")
            model_path = os.path.join(cache_dir, "model", "kandinsky5lite_t2v_sft_5s.safetensors")
            if not os.path.exists(model_path):
                snapshot_download(
                    repo_id="ai-forever/Kandinsky-5.0-T2V-Lite-sft-5s",
                    allow_patterns="model/*",
                    local_dir=cache_dir,
                )
            dit_path = model_path

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

    text_embedder = get_text_embedder(conf.model.text_embedder)
    if not offload:
        text_embedder = text_embedder.to(device=device_map["text_embedder"])

    vae = build_vae(conf.model.vae)
    vae = vae.eval()
    if not offload:
        vae = vae.to(device=device_map["vae"])

    dit = get_dit(conf.model.dit_params)

    if magcache:
        set_magcache_params(dit, conf.magcache.mag_ratios)

    # Download DIT model if it's a Hugging Face repository
    if "/" in conf.model.checkpoint_path and not conf.model.checkpoint_path.startswith("./"):
        cache_dir = os.environ.get('KD50_CACHE_DIR', './weights/')
        cache_dir = os.path.abspath(os.path.normpath(cache_dir))  # Ensure absolute normalized path
        print(f"Downloading DIT model to: {cache_dir}")

        # Set HF_HOME to ensure consistent cache directory
        os.environ['HF_HOME'] = cache_dir
        os.environ['HUGGINGFACE_HUB_CACHE'] = cache_dir

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

    # Try different possible model filenames
    possible_filenames = [
        "model.safetensors",
        "kandinsky5lite_t2v_sft_5s.safetensors",
        "kandinsky5lite_t2v_pretrain_5s.safetensors",
        "kandinsky5lite_t2v_distil_5s.safetensors",
        "kandinsky5lite_t2v_nocfg_5s.safetensors",
        "kandinsky5lite_t2v_sft_10s.safetensors",
        "kandinsky5lite_t2v_pretrain_10s.safetensors",
        "kandinsky5lite_t2v_distil_10s.safetensors",
        "kandinsky5lite_t2v_nocfg_10s.safetensors"
    ]

    full_model_path = None
    for filename in possible_filenames:
        test_path = os.path.join(checkpoint_path, filename)
        if os.path.exists(test_path):
            full_model_path = test_path
            print(f"Found model file: {full_model_path}")
            break
        # Also check in a 'model' subdirectory
        test_path_in_model = os.path.join(checkpoint_path, "model", filename)
        if os.path.exists(test_path_in_model):
            full_model_path = test_path_in_model
            print(f"Found model file in model subdirectory: {full_model_path}")
            break

    if full_model_path is None:
        # List what files are actually available
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
        raise FileNotFoundError(f"No model file found in {checkpoint_path}. Tried: {possible_filenames}")

    state_dict = load_file(full_model_path)
    dit.load_state_dict(state_dict, assign=True)

    # Apply torchao quantization to DIT if requested
    if quantize_dit:
        from .model_kd50_env import quantize_with_torch_ao
        print("Quantizing DIT model with torchao int8...")
        dit = quantize_with_torch_ao(dit)

    # DIT should stay on CPU initially when offload is enabled
    # Only move to GPU when text processing is complete
    if offload:
        print("Offload enabled - DIT will stay on CPU until text processing is complete")
        print("DIT initially loaded on CPU - will move to GPU for latent generation phase")
        # DIT stays on CPU
    else:
        # If offload is disabled, move DIT to GPU immediately
        target_device = device_map["dit"]
        print(f"Offload disabled - Moving DIT to: {target_device}")
        dit = dit.to(target_device)
        print(f"DIT successfully moved to: {next(dit.parameters()).device}")

    if world_size > 1:
        dit = parallelize_dit(dit, device_mesh["tensor_parallel"])

    return Kandinsky5T2VPipeline(
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
