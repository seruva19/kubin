import gradio as gr
from env import Kubin


def options_tab_native(kubin: Kubin):
    on_change = "(key, value, requiresRestart) => kubin.utils.optionsChanged(key, value, requiresRestart)"

    with gr.Column(elem_classes=["options-block", "options-native"]) as native_options:
        use_kandinsky31_flash = gr.Checkbox(
            value=lambda: kubin.params("native", "use_kandinsky31_flash"),
            label="Use Kandinsky 3.1 Flash pipeline",
        )

        text_encoder = gr.Dropdown(
            value=lambda: kubin.params("native", "text_encoder"),
            choices=kubin.params.default_config_value(
                "native", "available_text_encoders"
            ).split(";"),
            label="Custom text encoder",
            allow_custom_value=True,
            elem_classes=["options-medium"],
        )

        optimizations = gr.CheckboxGroup(
            elem_classes=["options-optimizations"],
            label="Optimizations",
            choices=kubin.params.default_config_value(
                "native", "available_optimization_flags"
            ).split(";"),
            value=lambda: kubin.params("native", "optimization_flags").split(";"),
        )

        optimization_flags = gr.TextArea(
            elem_id="options-optimizations-text",
            lines=2,
            value=lambda: kubin.params("native", "optimization_flags"),
            label="",
            interactive=False,
        )

        optimizations.change(
            fn=lambda flags: ";".join(flags),
            inputs=optimizations,
            outputs=optimization_flags,
            show_progress=True,
        )

        use_kandinsky31_flash.change(
            fn=None,
            _js=on_change,
            inputs=[
                gr.Text("native.use_kandinsky31_flash", visible=False),
                use_kandinsky31_flash,
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
                gr.Checkbox(True, visible=False),
            ],
            show_progress=False,
        )

    return native_options
