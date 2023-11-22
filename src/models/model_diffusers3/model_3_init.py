import gc
import torch
from diffusers import KandinskyV3Pipeline, KandinskyV3Img2ImgPipeline


def prepare_weights_for_task_3(model, task):
    cache_dir = model.params("general", "cache_dir")
    device = model.params("general", "device")

    if task == "text2img" or task == "img2img":
        if model.t2i_pipe is None:
            flush_if_required_3(model, task)

            model.t2i_pipe = KandinskyV3Pipeline.from_pretrained(
                "kandinsky-community/kandinsky-3",
                torch_dtype=type_of_weights(model.params),
                cache_dir=cache_dir,
                resume_download=True,
            )

        if task == "text2img":
            model.t2i_pipe = model.t2i_pipe.to(device)
        else:
            model.i2i_pipe = KandinskyV3Img2ImgPipeline(**model.t2i_pipe.components)
            model.i2i_pipe = model.t2i_pipe.to(device)


def flush_if_required_3(model, target):
    None


def type_of_weights(k_params):
    return torch.float16 if k_params("diffusers", "half_precision_weights") else "auto"
