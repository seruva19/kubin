import gradio as gr
from env import Kubin


def options_tab_native(kubin: Kubin):
    on_change = "(key, value, requiresRestart) => kubin.utils.optionsChanged(key, value, requiresRestart)"

    with gr.Column(elem_classes=["options-block", "options-native"]) as native_options:
        use_kandinsky_flash = gr.Checkbox(
            value=lambda: kubin.params("native", "use_kandinsky_flash"),
            label="Use Kandinsky 3 Flash pipeline",
        )

        text_encoder = gr.Dropdown(
            value=lambda: kubin.params("native", "text_encoder"),
            choices=["default", "google/flan-ul2", "pszemraj/flan-ul2-text-encoder"],
            label="Text encoder",
            allow_custom_value=True,
            elem_classes=["options-medium"],
        )

        optimization_flags = gr.TextArea(
            lines=4,
            value=lambda: kubin.params("native", "optimization_flags"),
            label="Optimization flags",
        )

        flash_attention = gr.Checkbox(
            value=lambda: kubin.params("native", "flash_attention"),
            label="Use Flash Attention",
        )

        use_kandinsky_flash.change(
            fn=None,
            _js=on_change,
            inputs=[
                gr.Text("native.use_kandinsky_flash", visible=False),
                use_kandinsky_flash,
                gr.Checkbox(False, visible=False),
            ],
            show_progress=False,
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

        optimization_flags.change(
            fn=None,
            _js=on_change,
            inputs=[
                gr.Text("native.optimization_flags", visible=False),
                optimization_flags,
                gr.Checkbox(False, visible=False),
            ],
            show_progress=False,
        )

        flash_attention.change(
            fn=None,
            _js=on_change,
            inputs=[
                gr.Text("native.flash_attention", visible=False),
                flash_attention,
                gr.Checkbox(True, visible=False),
            ],
            show_progress=False,
        )

    return native_options
