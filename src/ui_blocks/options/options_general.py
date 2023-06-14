import gradio as gr
from env import Kubin


def options_tab_general(kubin: Kubin):
    updated_config = kubin.params._updated
    current_config = kubin.params.conf

    with gr.Column(
        elem_classes=["options-block", "options-general", "active"]
    ) as general_options:
        pipeline = gr.Radio(
            value=kubin.params("general", "pipeline"),
            choices=["native", "diffusers"],
            label="Pipeline",
        )
        with gr.Row():
            options_log = gr.HTML(
                "No changes", elem_classes=["block-info", "options-info"]
            )

        def change_value(key, value):
            updated_config["general"][key] = value
            return f'Config key "general.{key}" changed to "{value}" (old value: "{current_config["general"][key]}"). Press "Apply changes" for them to take effect.'

        pipeline.change(
            change_value, inputs=[gr.State("pipeline"), pipeline], outputs=options_log
        )

    return general_options
