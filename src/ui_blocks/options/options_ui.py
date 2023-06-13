from dataclasses import dataclass
import gradio as gr
from env import Kubin


def options_tab_ui(kubin: Kubin):
    updated_config = kubin.params._updated

    with gr.Column(elem_classes=["options-block", "options-ui"]) as ui:
        None

    def change_value(key, value):
        updated_config["ui"][key] = value

    return ui
