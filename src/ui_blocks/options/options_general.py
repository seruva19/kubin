import gradio as gr
from env import Kubin


def options_tab_general(kubin: Kubin):
    updated_config = kubin.params._updated

    with gr.Column(
        elem_classes=["options-block", "options-general", "active"]
    ) as general:
        pipeline = gr.Radio(
            value=kubin.params("general", "pipeline"),
            choices=["native", "diffusers"],
            label="Pipeline",
        )
        use_flash_attention = gr.Checkbox(
            value=kubin.params("general", "flash_attention"),
            label="Use Flash Attention",
        )

        def change_value(key, value):
            updated_config["general"][key] = value

        pipeline.change(change_value, inputs=[gr.State("pipeline"), pipeline])
        use_flash_attention.change(
            change_value, inputs=[gr.State("flash_attention"), use_flash_attention]
        )

    return general
