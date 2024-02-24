# This source code is licensed under the Apache License found in the
# LICENSE file in the current directory.
"""
The code has been adopted from Kandinsky-3
(https://github.com/ai-forever/Kandinsky-3/blob/main/kandinsky3/__init__.py)
"""

from typing import Optional, Union, cast, no_type_check
import numpy as np
from huggingface_hub import hf_hub_download

import torch

from models.model_30.kandinsky3.inpainting_optimized_pipeline import (
    Kandinsky3InpaintingOptimizedPipeline,
)
from models.model_30.kandinsky3.model.unet import UNet
from models.model_30.kandinsky3.movq import MoVQ
from models.model_30.kandinsky3.condition_encoders import T5TextConditionEncoder
from models.model_30.kandinsky3.condition_processors import T5TextConditionProcessor
from models.model_30.kandinsky3.t2i_optimized_pipeline import (
    Kandinsky3T2IOptimizedPipeline,
)
from models.model_30.kandinsky3.t2i_pipeline import Kandinsky3T2IPipeline
from models.model_30.kandinsky3.inpainting_pipeline import Kandinsky3InpaintingPipeline
from models.model_30.kandinsky3.utils import release_vram
from models.model_30.model_kd30_env import Model_KD3_Environment


@no_type_check
def get_T2I_unet(
    device: Union[str, torch.device],
    environment: Model_KD3_Environment,
    weights_path: Optional[str] = None,
    fp16: bool = False,
) -> (UNet, Optional[torch.Tensor], Optional[dict]):
    unet = UNet(
        model_channels=384,
        num_channels=4,
        init_channels=192,
        time_embed_dim=1536,
        context_dim=4096,
        groups=32,
        head_dim=64,
        expansion_ratio=4,
        compression_ratio=2,
        dim_mult=(1, 2, 4, 8),
        num_blocks=(3, 3, 3, 3),
        add_cross_attention=(False, True, True, True),
        add_self_attention=(False, True, True, True),
    )

    if environment.kd30_low_vram:
        state_dict = torch.load(weights_path, map_location=torch.device("cpu"))
        unet.load_state_dict(state_dict["unet"])
        unet.eval().to(cast(torch.device, device), torch.float8_e4m3fn)
        return unet, None, None
    else:
        null_embedding = None
        projections_state_dict = None
        if weights_path:
            state_dict = torch.load(weights_path, map_location=torch.device("cpu"))
            projections_state_dict = state_dict["projections"]
            null_embedding = state_dict["null_embedding"]
            unet.load_state_dict(state_dict["unet"])

        unet.eval().to(device)
        if fp16:
            unet = unet.half()

        return unet, null_embedding, projections_state_dict


@no_type_check
def get_inpainting_unet(
    device: Union[str, torch.device],
    environment: Model_KD3_Environment,
    weights_path: Optional[str] = None,
    fp16: bool = False,
) -> (UNet, Optional[torch.Tensor], Optional[dict]):
    unet = UNet(
        model_channels=384,
        num_channels=9,
        init_channels=192,
        time_embed_dim=1536,
        context_dim=4096,
        groups=32,
        head_dim=64,
        expansion_ratio=4,
        compression_ratio=2,
        dim_mult=(1, 2, 4, 8),
        num_blocks=(3, 3, 3, 3),
        add_cross_attention=(False, True, True, True),
        add_self_attention=(False, True, True, True),
    )

    if environment.kd30_low_vram:
        state_dict = torch.load(weights_path, map_location=torch.device("cpu"))
        unet.load_state_dict(state_dict["unet"])
        unet.eval().to(cast(torch.device, device), torch.float8_e4m3fn)
        return unet, None, None
    else:
        null_embedding = None
        projections_state_dict = None
        if weights_path:
            state_dict = torch.load(weights_path, map_location=torch.device("cpu"))
            projections_state_dict = state_dict["projections"]
            null_embedding = state_dict["null_embedding"]
            unet.load_state_dict(state_dict["unet"])

        unet.eval().to(device)
        if fp16:
            unet = unet.half()

        return unet, null_embedding, projections_state_dict


@no_type_check
def get_T2I_nullemb_projections(
    weights_path: Optional[str] = None,
) -> (torch.Tensor, dict):
    state_dict = torch.load(weights_path, map_location=torch.device("cpu"))

    projections_state_dict = state_dict["projections"]
    null_embedding = state_dict["null_embedding"]

    return null_embedding, projections_state_dict


@no_type_check
def get_T5encoder(
    device: Union[str, torch.device],
    environment: Model_KD3_Environment,
    weights_path: str,
    cache_dir: str,
    projections_state_dict: Optional[dict] = None,
    fp16: bool = True,
    low_cpu_mem_usage: bool = True,
    device_map: Optional[str] = None,
) -> (T5TextConditionProcessor, T5TextConditionEncoder):
    model_names = {"t5": weights_path}
    tokens_length = {"t5": 128}
    context_dim = 4096
    model_dims = {"t5": 4096}
    processor = T5TextConditionProcessor(
        tokens_length=tokens_length, processor_names=model_names, cache_dir=cache_dir
    )
    condition_encoders = T5TextConditionEncoder(
        environment=environment,
        model_names=model_names,
        context_dim=context_dim,
        model_dims=model_dims,
        cache_dir=cache_dir,
        low_cpu_mem_usage=low_cpu_mem_usage,
        device_map=device_map,
    )

    if projections_state_dict:
        condition_encoders.projections.load_state_dict(projections_state_dict)

    condition_encoders = condition_encoders.eval().to(device)
    return processor, condition_encoders


def get_T5processor(weights_path: str, cache_dir: str) -> T5TextConditionProcessor:
    model_names = {"t5": weights_path}
    tokens_length = {"t5": 128}

    processor = T5TextConditionProcessor(tokens_length, model_names, cache_dir)

    return processor


def get_movq(
    device: Union[str, torch.device],
    environment: Model_KD3_Environment,
    weights_path: str,
    fp16: bool = False,
) -> MoVQ:
    generator_config = {
        "double_z": False,
        "z_channels": 4,
        "resolution": 256,
        "in_channels": 3,
        "out_ch": 3,
        "ch": 256,
        "ch_mult": [1, 2, 2, 4],
        "num_res_blocks": 2,
        "attn_resolutions": [32],
        "dropout": 0.0,
    }
    movq = MoVQ(generator_config)
    movq.load_state_dict(torch.load(weights_path))

    if environment.kd30_low_vram:
        movq.eval().to(cast(torch.device, device), torch.float8_e4m3fn)
    else:
        movq = movq.eval().to(device)
        if fp16:
            movq = movq.half()

    return movq


def get_T2I_pipeline(
    device: Union[str, torch.device],
    environment: Model_KD3_Environment,
    fp16: bool = False,
    cache_dir: str = "/tmp/kandinsky3/",
    unet_path: str = None,
    text_encoder_path: str = None,
    movq_path: str = None,
) -> Kandinsky3T2IPipeline:
    if unet_path is None:
        unet_path = hf_hub_download(
            repo_id="ai-forever/Kandinsky3.0",
            filename="weights/kandinsky3.pt",
            cache_dir=cache_dir,
        )
    if text_encoder_path is None:
        text_encoder_path = "google/flan-ul2"
    if movq_path is None:
        movq_path = hf_hub_download(
            repo_id="ai-forever/Kandinsky3.0",
            filename="weights/movq.pt",
            cache_dir=cache_dir,
        )

    if environment.kd30_low_vram:
        null_embedding, projections_state_dict = get_T2I_nullemb_projections(unet_path)
        processor = get_T5processor(text_encoder_path, cache_dir)

        encoder_loader = lambda: get_T5encoder(
            device=device,
            environment=environment,
            weights_path=text_encoder_path,
            projections_state_dict=projections_state_dict,
            device_map=None,
            low_cpu_mem_usage=None,
            fp16=fp16,
            cache_dir=cache_dir,
        )[1]
        unet_loader = lambda: get_T2I_unet(device, environment, unet_path, fp16=fp16)[0]
        movq_loader = lambda: get_movq(
            device=device, environment=environment, weights_path=movq_path, fp16=fp16
        )

        release_vram()

        return Kandinsky3T2IOptimizedPipeline(
            device,
            unet_loader,
            null_embedding,
            processor,
            encoder_loader,
            movq_loader,
            fp16=fp16,
        )
    else:
        unet, null_embedding, projections_state_dict = get_T2I_unet(
            device, unet_path, fp16=fp16
        )
        processor, condition_encoders = get_T5encoder(
            device,
            weights_path=text_encoder_path,
            cache_dir=cache_dir,
            projections_state_dict=projections_state_dict,
            low_cpu_mem_usage=True,
            fp16=True,
            device_map=None,
        )
        movq = get_movq(device, movq_path, fp16=fp16)
        return Kandinsky3T2IPipeline(
            device, unet, null_embedding, processor, condition_encoders, movq, fp16=fp16
        )


def get_inpainting_pipeline(
    device: Union[str, torch.device],
    environment: Model_KD3_Environment,
    fp16: bool = False,
    cache_dir: str = "/tmp/kandinsky3/",
    unet_path: str = None,
    text_encoder_path: str = None,
    movq_path: str = None,
) -> Kandinsky3InpaintingPipeline:
    if unet_path is None:
        unet_path = hf_hub_download(
            repo_id="ai-forever/Kandinsky3.0",
            filename="weights/kandinsky3_inpainting.pt",
            cache_dir=cache_dir,
        )
    if text_encoder_path is None:
        text_encoder_path = "google/flan-ul2"
    if movq_path is None:
        movq_path = hf_hub_download(
            repo_id="ai-forever/Kandinsky3.0",
            filename="weights/movq.pt",
            cache_dir=cache_dir,
        )

    if environment.kd30_low_vram:
        null_embedding, projections_state_dict = get_T2I_nullemb_projections(unet_path)
        processor = get_T5processor(text_encoder_path, cache_dir)

        encoder_loader = lambda: get_T5encoder(
            device=device,
            environment=environment,
            weights_path=text_encoder_path,
            projections_state_dict=projections_state_dict,
            device_map=None,
            low_cpu_mem_usage=None,
            fp16=fp16,
            cache_dir=cache_dir,
        )[1]
        unet_loader = lambda: get_inpainting_unet(
            device, environment, unet_path, fp16=fp16
        )[0]

        movq_env = Model_KD3_Environment(kd30_low_vram=False)
        movq = get_movq(device, movq_env, movq_path, fp16=fp16)

        release_vram()

        return Kandinsky3InpaintingOptimizedPipeline(
            device,
            unet_loader,
            null_embedding,
            processor,
            encoder_loader,
            movq,
            fp16=fp16,
        )
    else:
        unet, null_embedding, projections_state_dict = get_inpainting_unet(
            device, unet_path, fp16=fp16
        )
        processor, condition_encoders = get_T5encoder(
            device, text_encoder_path, projections_state_dict, fp16=fp16
        )
        movq = get_movq(device, movq_path, fp16=False)
        return Kandinsky3InpaintingPipeline(
            device, unet, null_embedding, processor, condition_encoders, movq, fp16=fp16
        )
