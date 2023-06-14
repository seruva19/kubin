import gradio as gr
from env import Kubin


def options_tab_general(kubin: Kubin):
    updated_config = kubin.params._updated

    with gr.Column(
        elem_classes=["options-block", "options-general", "active"]
    ) as general_options:
        pipeline = gr.Radio(
            value=kubin.params("general", "pipeline"),
            choices=["native", "diffusers"],
            label="Pipeline",
        )

        def change_value(key, value):
            updated_config["general"][key] = value

        pipeline.change(change_value, inputs=[gr.State("pipeline"), pipeline])

    return general_options
