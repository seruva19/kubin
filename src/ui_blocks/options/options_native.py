import gradio as gr
from env import Kubin


def options_tab_native(kubin: Kubin):
    on_change = "(key, value, requiresRestart) => kubin.utils.optionsChanged(key, value, requiresRestart)"

    with gr.Column(elem_classes=["options-block", "options-native"]) as native_options:
        native = gr.Checkbox(
            value=lambda: kubin.params("native", "flash_attention"),
            label="Use Flash Attention",
        )

        native.change(
            fn=None,
            _js=on_change,
            inputs=[
                gr.Text("native.flash_attention", visible=False),
                native,
                gr.Checkbox(False, visible=False),
            ],
            show_progress=False,
        )

    return native_options
