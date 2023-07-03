import gradio as gr
from env import Kubin


def options_tab_general(kubin: Kubin):
    on_change = "(key, value, requiresRestart) => kubin.utils.optionsChanged(key, value, requiresRestart)"

    with gr.Column(
        elem_classes=["options-block", "options-general", "active"]
    ) as general_options:
        pipeline = gr.Radio(
            value=kubin.params("general", "pipeline"),
            choices=["native", "diffusers"],
            label="Pipeline",
        )
        device = gr.Textbox(value=kubin.params("general", "device"), label="Device")

        pipeline.change(
            fn=None,
            _js=on_change,
            inputs=[
                gr.Text("general.pipeline", visible=False),
                pipeline,
                gr.Checkbox(False, visible=False),
            ],
            show_progress=False,
        )
        device.change(
            fn=None,
            _js=on_change,
            inputs=[
                gr.Text("general.device", visible=False),
                device,
                gr.Checkbox(False, visible=False),
            ],
            show_progress=False,
        )
    return general_options
