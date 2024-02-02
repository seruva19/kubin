import gradio as gr
from env import Kubin


def options_tab_native(kubin: Kubin):
    on_change = "(key, value, requiresRestart) => kubin.utils.optionsChanged(key, value, requiresRestart)"

    with gr.Column(elem_classes=["options-block", "options-native"]) as native_options:
        text_encoder = gr.Dropdown(
            value=lambda: kubin.params("native", "text_encoder"),
            choices=["google/flan-ul2", "pszemraj/flan-ul2-text-encoder"],
            label="Text encoder",
            allow_custom_value=True,
            elem_classes=["options-medium"],
        )

        native = gr.Checkbox(
            value=lambda: kubin.params("native", "flash_attention"),
            label="Use Flash Attention",
        )

        text_encoder.change(
            fn=None,
            _js=on_change,
            inputs=[
                gr.Text("native.text_encoder", visible=False),
                text_encoder,
                gr.Checkbox(False, visible=False),
            ],
            show_progress=False,
        )

        native.change(
            fn=None,
            _js=on_change,
            inputs=[
                gr.Text("native.flash_attention", visible=False),
                native,
                gr.Checkbox(True, visible=False),
            ],
            show_progress=False,
        )

    return native_options
