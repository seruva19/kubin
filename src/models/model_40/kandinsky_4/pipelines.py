# This source code is licensed under the Apache License found in the
# LICENSE file in the current directory.
"""
The code has been adopted from Kandinsky-4
(https://github.com/ai-forever/Kandinsky-4/blob/main/kandinsky/__init__.py)
"""


import os
from typing import Optional, Union

import torch
from omegaconf import OmegaConf

from models.model_40.model_kd40_env import (
    Model_KD40_Environment,
    quantize_with_optimum_quanto,
    quantize_with_torch_ao,
)
from utils.logging import k_log
from .dit import get_dit, parallelize
from .text_embedders import get_text_embedder
from diffusers import AutoencoderKLCogVideoX, CogVideoXDDIMScheduler
from omegaconf.dictconfig import DictConfig
from huggingface_hub import hf_hub_download, snapshot_download

from .t2v_pipeline import Kandinsky4T2VPipeline

from torch.distributed.device_mesh import DeviceMesh, init_device_mesh


def get_T2V_pipeline(
    device_map: Union[str, torch.device, dict],
    resolution: int = 512,
    cache_dir: str = "./weights/",
    dit_path: str = None,
    text_encoder_path: str = None,
    tokenizer_path: str = None,
    vae_path: str = None,
    scheduler_path: str = None,
    conf_path: str = None,
    environment: Model_KD40_Environment = Model_KD40_Environment(),
) -> Kandinsky4T2VPipeline:
    assert resolution in [512]

    if not isinstance(device_map, dict):
        device_map = {"dit": device_map, "vae": device_map, "text_embedder": device_map}

    try:
        local_rank, world_size = int(os.environ["LOCAL_RANK"]), int(
            os.environ["WORLD_SIZE"]
        )
    except:
        local_rank, world_size = 0, 1

    if world_size > 1:
        device_mesh = init_device_mesh(
            "cuda", (world_size,), mesh_dim_names=("tensor_parallel",)
        )
        device_map["dit"] = torch.device(f"cuda:{local_rank}")

    os.makedirs(cache_dir, exist_ok=True)

    if dit_path is None:
        dit_path = hf_hub_download(
            repo_id="ai-forever/kandinsky-4-t2v-flash",
            filename=f"kandinsky4_distil_{resolution}.pt",
            local_dir=cache_dir,
        )

    if vae_path is None:
        vae_path = snapshot_download(
            repo_id="THUDM/CogVideoX-5b", allow_patterns="vae/*", local_dir=cache_dir
        )
        vae_path = os.path.join(cache_dir, f"vae/")

    if scheduler_path is None:
        scheduler_path = snapshot_download(
            repo_id="THUDM/CogVideoX-5b",
            allow_patterns="scheduler/*",
            local_dir=cache_dir,
        )
        scheduler_path = os.path.join(cache_dir, f"scheduler/")

    if text_encoder_path is None:
        text_encoder_path = snapshot_download(
            repo_id="THUDM/CogVideoX-5b",
            allow_patterns="text_encoder/*",
            local_dir=cache_dir,
        )
        text_encoder_path = os.path.join(cache_dir, f"text_encoder/")

    if tokenizer_path is None:
        tokenizer_path = snapshot_download(
            repo_id="THUDM/CogVideoX-5b",
            allow_patterns="tokenizer/*",
            local_dir=cache_dir,
        )
        tokenizer_path = os.path.join(cache_dir, f"tokenizer/")

    if environment.kd40_conf is not None:
        conf = environment.kd40_conf
        conf.vae.checkpoint_path = vae_path
        conf.text_embedder.params.checkpoint_path = text_encoder_path
        conf.text_embedder.params.tokenizer_path = tokenizer_path
        conf.dit.scheduler = scheduler_path
        conf.dit.checkpoint_path = (
            dit_path if conf.dit.checkpoint_path == "" else conf.dit.checkpoint_path
        )

    elif conf_path is None:
        conf = get_default_conf(
            vae_path, text_encoder_path, tokenizer_path, scheduler_path, dit_path
        )
    else:
        conf = OmegaConf.load(conf_path)
    print(f"loaded configuration: {conf}")

    k_log("loading DiT...")
    dit = get_dit(conf.dit)
    dit = dit.to(dtype=torch.bfloat16, device=device_map["dit"])
    # dit = dit.to(dtype=torch.float8_e4m3fn, device=device_map["dit"])

    path, ext = os.path.splitext(conf.dit.checkpoint_path)
    if environment.use_t2v_tenc_int8_ao_quantization:
        k_log("quantizing DiT [torchao-int8]...")
        dit = quantize_with_torch_ao(
            dit,
            True,
            (f"{path}.q8_tao{ext}" if environment.use_save_quantized_weights else None),
        )
    elif environment.use_t2v_dit_int8_oq_quantization:
        k_log("quantizing DiT [optimum-quanto]...")
        dit = quantize_with_optimum_quanto(
            dit,
            True,
            (f"{path}.q8_oq{ext}" if environment.use_save_quantized_weights else None),
        )

    noise_scheduler = CogVideoXDDIMScheduler.from_pretrained(conf.dit.scheduler)

    if world_size > 1:
        dit = parallelize(dit, device_mesh["tensor_parallel"])

    k_log("loading text embedder...")
    text_embedder = get_text_embedder(conf)

    path, ext = os.path.splitext(conf.text_embedder.params.checkpoint_path)
    if environment.use_t2v_tenc_int8_ao_quantization:
        k_log("quantizing embedder [torchao-int8]...")
        text_embedder.llm = quantize_with_torch_ao(
            text_embedder.llm,
            True,
            (f"{path}.q8_tao{ext}" if environment.use_save_quantized_weights else None),
        )
    elif environment.use_t2v_tenc_int8_oq_quantization:
        k_log("quantizing embedder [optimum-quanto]...")
        text_embedder.llm = quantize_with_optimum_quanto(
            text_embedder.llm,
            True,
            (f"{path}.q8_oq{ext}" if environment.use_save_quantized_weights else None),
        )
    else:
        text_embedder = text_embedder.freeze()

    if local_rank == 0:
        text_embedder = text_embedder.to(
            device=device_map["text_embedder"], dtype=torch.bfloat16
        )

    k_log("loading vae...")
    vae = AutoencoderKLCogVideoX.from_pretrained(conf.vae.checkpoint_path)

    if environment.use_vae_tiling:
        k_log("vae tiling enabled")
        vae.enable_tiling()
    else:
        vae.disable_tiling()

    if environment.use_vae_slicing:
        k_log("vae slicing enabled")
        vae.enable_slicing()
    else:
        vae.disable_slicing()

    vae = vae.eval()

    if local_rank == 0:
        vae = vae.to(device_map["vae"], dtype=torch.bfloat16)

    path, ext = os.path.splitext(conf.vae.checkpoint_path)
    if environment.use_t2v_vae_int8_ao_quantization:
        k_log("quantizing vae [torchao-int8]...")
        vae = quantize_with_torch_ao(
            vae,
            True,
            (f"{path}.q8_tao{ext}" if environment.use_save_quantized_weights else None),
        )
    elif environment.use_t2v_vae_int8_oq_quantization:
        k_log("quantizing vae [optimum-quanto]...")
        vae = quantize_with_optimum_quanto(
            vae,
            True,
            (f"{path}.q8_oq{ext}" if environment.use_save_quantized_weights else None),
        )

    return Kandinsky4T2VPipeline(
        environment=environment,
        device_map=device_map,
        dit=dit,
        text_embedder=text_embedder,
        vae=vae,
        noise_scheduler=noise_scheduler,
        resolution=resolution,
        local_dit_rank=local_rank,
        world_size=world_size,
    )


def get_default_conf(
    vae_path,
    text_encoder_path,
    tokenizer_path,
    scheduler_path,
    dit_path,
) -> DictConfig:
    dit_params = {
        "in_visual_dim": 16,
        "in_text_dim": 4096,
        "out_visual_dim": 16,
        "time_dim": 512,
        "patch_size": [1, 2, 2],
        "model_dim": 3072,
        "ff_dim": 12288,
        "num_blocks": 21,
        "axes_dims": [16, 24, 24],
    }

    conf = {
        "vae": {"checkpoint_path": vae_path},
        "text_embedder": {
            "emb_size": 4096,
            "tokens_length": 224,
            "params": {
                "checkpoint_path": text_encoder_path,
                "tokenizer_path": tokenizer_path,
            },
        },
        "dit": {
            "scheduler": scheduler_path,
            "checkpoint_path": dit_path,
            "params": dit_params,
        },
        "resolution": 512,
    }

    return DictConfig(conf)
