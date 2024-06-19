import numpy as np
import requests
from gradio.processing_utils import decode_base64_to_image
import base64
from PIL import Image, ImageOps
from io import BytesIO
import cv2


def image_path_to_pil(image_url):
    response = requests.get(image_url)
    pil_img = Image.open(BytesIO(response.content))

    return pil_img


def round_to_nearest(x, base):
    return round(x / base) * base


def create_inpaint_targets(
    pil_img, image_mask, output_size, inpaint_region, inpaint_target
):
    pil_img = pil_img.resize(output_size, resample=Image.LANCZOS)
    pil_img = pil_img.convert("RGB")

    image_mask = ImageOps.invert(image_mask)

    image_mask = image_mask.resize(output_size)
    image_mask = image_mask.convert("L")

    image_mask = np.array(image_mask).astype(np.float32) / 255.0

    if inpaint_target == "only mask":
        image_mask = 1.0 - image_mask

    return pil_img, image_mask


def create_outpaint_targets(image, offset, infer_size, width, height):
    image_w, image_h = image.size

    if offset is not None:
        top, right, bottom, left = offset
        inferred_mask_size = tuple(
            a + b for a, b in zip(image.size, (left + right, top + bottom))
        )[::-1]
        mask = np.zeros(inferred_mask_size, dtype=np.float32)
        mask[top : image_h + top, left : image_w + left] = 1
        image = ImageOps.expand(image, border=(left, top, right, bottom), fill=0)

    else:
        x1, y1, x2, y2 = image.getbbox()
        mask = np.ones((image_h, image_w), dtype=np.float32)
        mask[0:y1, :] = 0
        mask[:, 0:x1] = 0
        mask[y2:image_h, :] = 0
        mask[:, x2:image_w] = 0

    if infer_size:
        height, width = mask.shape[:2]

    mask = 1.0 - mask
    return image, mask, width, height


def composite_images(original_img, second_img, mask_array):
    composited_img = original_img.copy()

    mask_img = Image.fromarray(mask_array).convert("L")
    transparent = Image.new("RGBA", second_img.size, (0, 0, 0, 0))

    cut_out_mask = Image.eval(mask_img, lambda p: 0 if p == 0 else 255)
    cut_out_region = Image.composite(second_img, transparent, cut_out_mask)
    composited_img.paste(cut_out_region, (0, 0), cut_out_region)

    return composited_img
