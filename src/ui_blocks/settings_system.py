import sys
import gradio as gr
import torch
import psutil
import platform
from env import Kubin
from utils.logging import get_log, k_log


def in_mb(bytes: float):
    return round(bytes / (1024**2))


def update_info():
    k_log("scanning system information")
    torch_version = torch.__version__
    cuda_version = torch.version.cuda

    if torch.cuda.is_available():
        torch_free, torch_total = torch.cuda.mem_get_info()
    else:
        torch_free, torch_total = 0, 0

    xformers_info = ""
    try:
        import xformers

        xformers_info = f"xformers: {xformers.__version__}\n"
    except:
        xformers_info = f"xformers: not installed\n"

    flash_attn_info = ""
    try:
        import flash_attn

        flash_attn_info = f"flash_attn: {flash_attn.__version__}\n"
    except:
        flash_attn_info = f"flash_attn: not installed\n"

    diffusers_info = ""
    try:
        import diffusers

        diffusers_info = f"diffusers: {diffusers.__version__}\n"
    except:
        diffusers_info = f"diffusers: not installed\n"

    transformers_info = ""
    try:
        import transformers

        transformers_info = f"transformers: {transformers.__version__}\n"
    except:
        transformers_info = f"transformers: not installed\n"

    accelerate_info = ""
    try:
        import accelerate

        accelerate_info = f"accelerate: {accelerate.__version__}\n"
    except:
        accelerate_info = f"accelerate: not installed\n"

    vmem = psutil.virtual_memory()
    ram_total = vmem.total
    ram_available = vmem.available

    torch_total_mb = in_mb(torch_total)
    torch_free_mb = in_mb(torch_free)
    ram_total_mb = in_mb(ram_total)
    ram_available_mb = in_mb(ram_available)

    return (
        f"python: {sys.version}\n"
        f"torch: {torch_version}\n"
        f"CUDA: {cuda_version}\n"
        f"processor: {platform.processor()}\n"
        f"RAM (total): {ram_total_mb}\n"
        f"RAM (free): {ram_available_mb}\n"
        f"VRAM (total): {torch_total_mb}\n"
        f"VRAM (free): {torch_free_mb}\n"
        f"gradio: {gr.__version__}\n"
        f"{xformers_info}"
        f"{flash_attn_info}"
        f"{diffusers_info}"
        f"{transformers_info}"
        f"{accelerate_info}"
    )


def system_ui(kubin: Kubin):
    with gr.Row() as system_block:
        system_log = gr.Textbox(
            update_info,
            lines=15,
            max_lines=15,
            label="System info",
            interactive=False,
            show_copy_button=True,
            elem_classes=["system-info"],
        )

    with gr.Row(equal_height=False):
        update_btn = gr.Button(value="💻 Output system info", scale=0, size="sm")
        update_btn.click(update_info, outputs=system_log)

        show_log = gr.Button(value="📅 Output log", scale=0, size="sm")
        show_log.click(
            fn=lambda: "\n".join(get_log(["INFO"])),
            outputs=system_log,
        )

        show_errors = gr.Button(value="⚠️ Output only errors", scale=0, size="sm")
        show_errors.click(
            fn=lambda: "\n".join(get_log(["ERROR"])),
            outputs=system_log,
        )

        unload_model = gr.Button(value="📉 Free memory", scale=0, size="sm")
        unload_model.click(lambda: kubin.model.flush(), queue=False).then(
            fn=None, _js='_ => kubin.notify.success("Model unloaded")'
        )

        wake_ui = gr.Button(value="⏰ Wake UI", scale=0, size="sm")
        wake_ui.click(fn=None, _js="_ => kubin.UI.wakeAll()")

    return system_block
