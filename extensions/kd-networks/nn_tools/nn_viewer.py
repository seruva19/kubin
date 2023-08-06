import gradio as gr
import gc
import os
import torch
from torchinfo import summary


def read_model_info(model):
    return str(summary(model, None, verbose=0))


def get_path_by_name_and_hash(networks_list, name_with_hash):
    all_models = [item for array in networks_list for item in array]

    for model in all_models:
        if f'{model["name"]} [{model["hash"]}]' == name_with_hash:
            return model["path"]
