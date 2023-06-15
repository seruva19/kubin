import gradio as gr
from env import Kubin


def options_tab_native(kubin: Kubin):
    updated_config = kubin.params._updated
    current_config = kubin.params.conf

    with gr.Column(elem_classes=["options-block", "options-native"]) as native_options:
        native = gr.Checkbox(
            value=kubin.params("native", "flash_attention"),
            label="Use Flash Attention",
        )
        with gr.Row():
            options_log = gr.HTML(
                "No changes", elem_classes=["block-info", "options-info"]
            )

        def change_value(key, value):
            updated_config["native"][key] = value
            return f'Config key "native.{key}" changed to "{value}" (old value: "{current_config["native"][key]}"). Press "Apply changes" for them to take effect.'

        native.change(
            change_value,
            inputs=[gr.State("flash_attention"), native],
            outputs=options_log,
            show_progress=False,
        )

    return native_options
