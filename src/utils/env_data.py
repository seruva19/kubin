import os
import torch
from collections import defaultdict
from safetensors.torch import load_file, save_file


def map_target_to_task(target):
    return (
        "text2img"
        if target == "t2i"
        else "img2img"
        if target == "i2i"
        else "inpainting"
        if target == "inpaint"
        else "outpainting"
        if target == "outpaint"
        else target
    )
