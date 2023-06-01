import sys
import gradio as gr
import torch
import psutil
import platform
from kandinsky2 import CONFIG_2_1
import pandas as pd
from env import Kubin
from models.model_kd2 import Model_KD2
from models.model_mock import Model_Mock
from .settings_ckpt import ckpt_selector


def update_info():
    print("scanning system information")
    torch_version = torch.__version__
    cuda_version = torch.version.cuda

    if torch.cuda.is_available():
        torch_free, torch_total = torch.cuda.mem_get_info()
    else:
        torch_free, torch_total = 0, 0

    vmem = psutil.virtual_memory()
    ram_total = vmem.total
    ram_available = vmem.available

    torch_total_mb = round(torch_total / (1024 * 1024))
    torch_free_mb = round(torch_free / (1024 * 1024))
    ram_total_mb = round(ram_total / (1024 * 1024))
    ram_available_mb = round(ram_available / (1024 * 1024))

    return (
        f"python: {sys.version}\n"
        f"torch: {torch_version}\n"
        f"CUDA: {cuda_version}\n"
        f"processor: {platform.processor()}\n"
        f"RAM (total): {ram_total_mb}\n"
        f"RAM (free): {ram_available_mb}\n"
        f"VRAM (total): {torch_total_mb}\n"
        f"VRAM (free): {torch_free_mb}"
    )


def flatten_model_config(config):
    normalized = pd.json_normalize(config, sep=".")
    values = normalized.to_dict(orient="records")[0]

    return values


def settings_ui(kubin: Kubin):
    model_config = flatten_model_config(CONFIG_2_1)

    with gr.Column() as settings_block:
        with gr.TabItem("Checkpoints"):
            ckpt_selector(kubin)

        with gr.TabItem("System"):
            with gr.Row():
                system_info = gr.TextArea(
                    update_info, lines=10, label="System info", interactive=False
                ).style(show_copy_button=True)
                textbox_log = gr.TextArea(
                    label="System  log", lines=10, interactive=False
                ).style(show_copy_button=True)

            with gr.Row():
                update_btn = gr.Button(value="Update system info").style(
                    full_width=False, size="sm"
                )
                update_btn.click(update_info, outputs=system_info)

                unload_model = gr.Button(value="Free memory").style(
                    full_width=False, size="sm"
                )
                unload_model.click(lambda: kubin.model.flush(), queue=False).then(
                    fn=None, _js='_ => kubin.notify.success("Model unloaded")'
                )

            with gr.Accordion("Model params", open=False):
                values = []
                for key in model_config:
                    value = gr.Textbox(label=str(key), value=model_config[key])
                    values.append(value)

    return settings_block
