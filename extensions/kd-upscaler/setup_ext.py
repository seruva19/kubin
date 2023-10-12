from upscale import upscale_with
import gradio as gr
from pathlib import Path


dir = Path(__file__).parent.absolute()
default_upscalers_path = f"{dir}/upscalers.default.yaml"

title = "Upscaler"


def setup(kubin):
    source_image = gr.Image(
        type="pil", label="Image to upscale", elem_classes=["full-height"]
    )

    def upscaler_ui(ui_shared, ui_tabs):
        with gr.Row() as upscaler_block:
            with gr.Column(scale=1) as upscaler_params_block:
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
                upscale_output = gr.Gallery(label="Upscaled Image", preview=True)

                upscale_output.select(
                    fn=None,
                    _js=f"() => kubin.UI.setImageIndex('upscale-output')",
                    show_progress=False,
                    outputs=gr.State(None),
                )

                ui_shared.create_base_send_targets(
                    upscale_output, "upscale-output", ui_tabs
                )

            kubin.ui_utils.click_and_disable(
                upscale_btn,
                fn=lambda *p: upscale_with(kubin, *p),
                inputs=[
                    upscaler,
                    gr.Textbox(value=kubin.params("general", "device"), visible=False),
                    gr.Textbox(
                        value=kubin.params("general", "cache_dir"), visible=False
                    ),
                    scale,
                    gr.Textbox(
                        value=kubin.params("general", "output_dir"), visible=False
                    ),
                    source_image,
                    clear_memory,
                ],
                outputs=upscale_output,
                js=[
                    f"args => kubin.UI.taskStarted('{title}')",
                    f"args => kubin.UI.taskFinished('{title}')",
                ],
            )

            upscaler_params_block.elem_classes = ["block-params"]
        return upscaler_block

    def upscaler_select_ui(target):
        None

    def upscale_after_inference(target, params, upscale_params):
        None

    return {
        "send_to": f"📐 Send to {title}",
        "title": title,
        "targets": ["t2i", "i2i", "mix", "inpaint", "outpaint"],
        # "inject_ui": lambda target: upscaler_select_ui(target),
        "inject_fn": lambda target, params, augmentations: upscale_after_inference(
            target, params, augmentations[0]
        ),
        "tab_ui": lambda ui_s, ts: upscaler_ui(ui_s, ts),
        "send_target": source_image,
    }
