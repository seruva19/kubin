from dataclasses import dataclass
import gradio as gr
from env import Kubin


def options_tab_gradio(kubin: Kubin):
    updated_config = kubin.params._updated
    current_config = kubin.params.conf

    with gr.Column(elem_classes=["options-block", "options-gradio"]) as gradio_options:
        theme = gr.Dropdown(
            value=kubin.params("gradio", "theme"),
            choices=["base", "default", "glass", "monochrome", "soft"],
            label="Gradio theme",
        )
        with gr.Row():
            options_log = gr.HTML("", elem_classes=["block-info", "options-info"])

        def change_value(key, value):
            updated_config["gradio"][key] = value
            return f'Config key "gradio.{key}" changed to "{value}" (old value: "{current_config["gradio"][key]}"). Press "Apply changes" for them to take effect.'

    theme.change(
        change_value,
        inputs=[gr.State("theme"), theme],
        outputs=options_log,
        show_progress=False,
    )

    return gradio_options
