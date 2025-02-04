import torch
import gc
from typing import Union, List, Dict, Optional
from collections import defaultdict
from safetensors.torch import load_file, save_file
from utils.logging import k_log
import os
import yaml

models: Dict[str, torch.nn.Module] = {}


def reg(model_id, weights):
    if model_id in models:
        k_log(f"model with name '{model_id}' already exists")
    models[model_id] = weights


def clear(model_names: Optional[Union[str, List[str]]] = None):
    names_to_clear = []

    if model_names is None:
        names_to_clear = list(models.keys())
    elif isinstance(model_names, str):
        names_to_clear = [model_names]
    else:
        names_to_clear = model_names

    for name in names_to_clear:
        if name not in models:
            k_log(f"model '{name}' not registered, cannot release")

        try:
            models[name].to("cpu")
        except:
            k_log(f"failed to release model '{name}'")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        del models[name]
    gc.collect()


def load_env_value(key, default_value):
    value = os.environ.get(key, default_value)
    k_log(f"env key: {key}, value: {value}")
    return value


def load_custom_env(file_path):
    try:
        if os.path.exists(file_path):
            k_log(f"loading custom env values from {file_path}")
            with open(file_path, "r") as yaml_file:
                config = yaml.safe_load(yaml_file)

            if config is None:
                return

            for key, value in config.items():
                os.environ[key] = str(value)
                k_log(f"custom environment variable set: {key} = {value}")

    except Exception as e:
        k_log(f"error loading custom env values from {file_path}: {e}")


def map_target_to_task(target):
    return (
        "text2img"
        if target == "t2i"
        else (
            "img2img"
            if target == "i2i"
            else (
                "inpainting"
                if target == "inpaint"
                else "outpainting" if target == "outpaint" else target
            )
        )
    )
