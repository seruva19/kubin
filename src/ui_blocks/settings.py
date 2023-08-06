import sys
import gradio as gr
import torch
import psutil
import platform
from kandinsky2 import CONFIG_2_1
from env import Kubin
from utils.yaml import flatten_yaml
from .settings_options import options_ui
from .settings_ckpt import ckpt_selector


def settings_ui(kubin: Kubin, start_fn, ui):
    with gr.Column() as settings_block:
        settings_block.elem_classes = ["settings-tabs"]

        with gr.TabItem("Options"):
            options_ui(kubin, start_fn, ui)

        with gr.TabItem("Checkpoints", elem_id="checkpoint-switcher"):
            ckpt_selector(kubin)

    return settings_block
