import gradio as gr
from env import Kubin


def options_tab_native(kubin: Kubin):
    updated_config = kubin.params._updated

    with gr.Column(elem_classes=["options-block", "options-native"]) as native_options:
        native = gr.Checkbox(
            value=kubin.params("native", "flash_attention"),
            label="Use Flash Attention",
        )

        def change_value(key, value):
            updated_config["native"][key] = value

        native.change(change_value, inputs=[gr.State("flash_attention"), native])

    return native_options
