import gc
import torch
from diffusers import (
    Kandinsky3Pipeline,
    Kandinsky3Img2ImgPipeline,
    AutoPipelineForText2Image,
    AutoPipelineForImage2Image,
)

from utils.logging import k_log


def prepare_model_for_task(model, task):
    cache_dir = model.params("general", "cache_dir")
    device = model.params("general", "device")

    if task == "text2img" or task == "img2img":
        if model.t2i_pipe is None:
            flush_if_required(model, task)

            model.t2i_pipe = Kandinsky3Pipeline.from_pretrained(
                "kandinsky-community/kandinsky-3",
                torch_dtype=type_of_weights(model.params),
                cache_dir=cache_dir,
                resume_download=True,
            )

        if task == "text2img":
            model.t2i_pipe = model.t2i_pipe.to(device)
        else:
            model.i2i_pipe = Kandinsky3Img2ImgPipeline(**model.t2i_pipe.components)
            model.i2i_pipe = model.t2i_pipe.to(device)


def prepare_autopipeline_for_task(model, task):
    cache_dir = model.params("general", "cache_dir")
    device = model.params("general", "device")
    model_cpu_offload = model.params("diffusers", "full_model_offload")
    sequential_cpu_offload = model.params("diffusers", "sequential_cpu_offload")

    pipe = None

    if task == "text2img" or task == "img2img":
        if model.auto_t2i_pipe is None:
            flush_if_required(model, task)

            model.auto_t2i_pipe = AutoPipelineForText2Image.from_pretrained(
                "kandinsky-community/kandinsky-3",
                variant="fp16",
                torch_dtype=type_of_weights(model.params),
                cache_dir=cache_dir,
            )

        if task == "text2img":
            pipe = model.auto_t2i_pipe
        else:
            model.auto_i2i_pipe = AutoPipelineForImage2Image(
                **model.auto_t2i_pipe.components
            )
            pipe = model.auto_i2i_pipe

        if model_cpu_offload:
            if not model.optimizations.model_offload:
                pipe.enable_model_cpu_offload()
                model.optimizations.model_offload = True
        elif sequential_cpu_offload:
            if model.optimizations.sequential_offload:
                pipe.enable_sequential_cpu_offload()
                model.optimizations.sequential_offload = True
        else:
            pipe.to(device)

    return pipe


def flush_if_required(model, target):
    k_log("flush: not implemented")
    None


def type_of_weights(k_params):
    return torch.float16 if k_params("diffusers", "half_precision_weights") else "auto"
