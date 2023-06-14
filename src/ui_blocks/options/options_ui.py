from dataclasses import dataclass
import gradio as gr
from env import Kubin


def options_tab_ui(kubin: Kubin):
    updated_config = kubin.params._updated
    current_config = kubin.params.conf

    with gr.Column(elem_classes=["options-block", "options-ui"]) as ui_options:
        with gr.Row():
            options_log = gr.HTML(
                "No changes", elem_classes=["block-info", "options-info"]
            )

    def change_value(key, value):
        updated_config["ui"][key] = value
        return f'Config key "ui.{key}" changed to "{value}" (old value: "{current_config["ui"][key]}"). Press "Apply changes" for them to take effect.'

    return ui_options
