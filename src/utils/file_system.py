import os
import uuid
from datetime import datetime
import json
from PIL import PngImagePlugin, Image
import re


def create_png_info(metadata):
    png_info = PngImagePlugin.PngInfo()
    if metadata:
        png_info.add_text("kubin_image_metadata", metadata)
    return png_info


def create_filename(path, params):
    current_datetime = datetime.now()
    format_string = "%Y%m%d%H%M%S"
    formatted_datetime = current_datetime.strftime(format_string)

    if params is not None and params.get("prompt", None) is not None:
        invalid_symbols_pattern = r'[\/:*?"<>|]'
        prompt_string = params.get("prompt").strip()
        prompt_string = re.sub(invalid_symbols_pattern, "", prompt_string)

        prompt_words = "_".join(prompt_string.split()[:5])
        postfix = f"_{prompt_words}"
    else:
        postfix = ""

    filename = f"{path}/{formatted_datetime}{postfix}.png"
    while os.path.exists(filename):
        unique_id = current_datetime.microsecond
        filename = f"{formatted_datetime}_{unique_id}.png"

    return filename


def save_output(output_dir, task_type, images, params=None):
    output = []
    params_as_json = None

    if params:
        params = {k: v for k, v in params.items() if not isinstance(v, Image.Image)}
        params_as_json = json.dumps(
            params, skipkeys=True, default=lambda _: "<parameter cannot be serialized>"
        )

    for img in images:
        path = f"{output_dir}/{task_type}"
        if not os.path.exists(path):
            os.makedirs(path)

        filename = create_filename(path, params)
        img.save(filename, "PNG", pnginfo=create_png_info(params_as_json))
        output.append(filename)

    return output
