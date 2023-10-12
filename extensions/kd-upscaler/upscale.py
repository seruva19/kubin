from PIL import Image
import numpy as np
from real_esrgan.model import RealESRGAN
import os
import torch


def upscale_with(
    kubin,
    upscaler,
    device,
    cache_dir,
    scale,
    output_dir,
    input_image,
    clear_before_upscale,
):
    if clear_before_upscale:
        kubin.model.flush()

    if upscaler == "Real-ESRGAN":
        upscaled_image = upscale_esrgan(device, cache_dir, input_image, scale)
        upscaled_image_path = kubin.fs_utils.save_output(
            os.path.join(output_dir, "upscale"), [upscaled_image]
        )

        return upscaled_image_path

    else:
        kubin.log(f"upscale method {upscaler} not implemented")
        return []


def upscale_esrgan(device, cache_dir, input_image, scale):
    # implementation taken from https://github.com/ai-forever/Real-ESRGAN
    esrgan = RealESRGAN(device, scale=int(scale))
    esrgan.load_weights(f"{cache_dir}/esrgan/RealESRGAN_x{scale}.pth", download=True)

    image = input_image.convert("RGB")
    upscaled_image = esrgan.predict(image)

    return upscaled_image
