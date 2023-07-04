from dataclasses import dataclass
import gradio as gr
from env import Kubin


def options_tab_gradio(kubin: Kubin):
    on_change = "(key, value, requiresRestart) => kubin.utils.optionsChanged(key, value, requiresRestart)"

    with gr.Column(elem_classes=["options-block", "options-gradio"]) as gradio_options:
        theme = gr.Dropdown(
            value=kubin.params("gradio", "theme"),
            choices=["base", "default", "glass", "monochrome", "soft"],
            label="Gradio theme",
        )

    theme.change(
        fn=None,
        _js=on_change,
        inputs=[
            gr.Text("gradio.theme", visible=False),
            theme,
            gr.Checkbox(True, visible=False),
        ],
        show_progress=False,
    )

    return gradio_options
