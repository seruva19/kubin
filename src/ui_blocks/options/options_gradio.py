from dataclasses import dataclass
import gradio as gr
from env import Kubin


def options_tab_gradio(kubin: Kubin):
    updated_config = kubin.params._updated

    with gr.Column(elem_classes=["options-block", "options-gradio"]) as gradio:
        theme = gr.Dropdown(
            value=kubin.params("gradio", "theme"),
            choices=["base", "default", "glass", "monochrome", "soft"],
            label="Gradio theme",
        )

    def change_value(key, value):
        updated_config["gradio"][key] = value

    theme.change(change_value, inputs=[gr.State("theme"), theme])

    return gradio
