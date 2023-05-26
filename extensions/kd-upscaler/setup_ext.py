import gradio as gr
import torch
from PIL import Image
import numpy as np
from RealESRGAN import RealESRGAN


def setup(kubin):
    source_image = gr.Image(
        type="pil", label="Image to upscale", elem_classes=["full-height"]
    )

    def upscaler_ui(ui_shared, ui_tabs):
        with gr.Row() as upscaler_block:
            with gr.Column(scale=1):
                with gr.Row():
                    source_image.render()

                    with gr.Column() as upscale_selector:
                        upscaler = gr.Radio(
                            ["Real-ESRGAN"], value="Real-ESRGAN", label="Upscaler"
                        )
                        scale = gr.Radio(
                            ["2", "4", "8"],
                            value="2",
                            label="Upscale by",
                            interactive=True,
                        )

                with gr.Row():
                    clear_memory = gr.Checkbox(False, label="Clear VRAM before upscale")

            with gr.Column(scale=1):
                upscale_btn = gr.Button("Upscale", variant="primary")
                upscale_output = gr.Gallery(label="Upscaled Image").style(preview=True)

                ui_shared.create_base_send_targets(upscale_output, gr.State(0), ui_tabs)  # type: ignore

            upscale_btn.click(
                fn=lambda *p: upscale_with(kubin, *p),
                inputs=[
                    upscaler,
                    gr.Textbox(value=kubin.options.device, visible=False),
                    gr.Textbox(value=kubin.options.cache_dir, visible=False),
                    scale,
                    gr.Textbox(value=kubin.options.output_dir, visible=False),
                    source_image,
                    clear_memory,
                ],
                outputs=upscale_output,
            )

        return upscaler_block

    return {
        "title": "Upscaler",
        "tab_ui": lambda ui_s, ts: upscaler_ui(ui_s, ts),
        "send_target": source_image,
    }


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
            output_dir, "upscale", [upscaled_image]
        )

        return upscaled_image_path

    else:
        print(f"upscale method {upscaler} not implemented")
        return []


def upscale_esrgan(device, cache_dir, input_image, scale):
    esrgan = RealESRGAN(device, scale=int(scale))
    esrgan.load_weights(f"{cache_dir}/esrgan/RealESRGAN_x{scale}.pth", download=True)

    image = input_image.convert("RGB")
    upscaled_image = esrgan.predict(image)

    return upscaled_image
