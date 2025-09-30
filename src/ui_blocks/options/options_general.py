import gradio as gr
from env import Kubin


def options_tab_general(kubin: Kubin):
    on_change = "(key, value, requiresRestart) => kubin.utils.optionsChanged(key, value, requiresRestart)"

    with gr.Column(
        elem_classes=["options-block", "options-general", "active"]
    ) as general_options:
        model_name = gr.Radio(
            value=lambda: kubin.params("general", "model_name"),
            choices=["kd20", "kd21", "kd22", "kd30", "kd31", "kd40", "kd50"],
            info=kubin.ui.info(
                "kd20: Kandinsky 2.0, kd21: Kandinsky 2.1, kd22: Kandinsky 2.2, kd30: Kandinsky 3.0, kd31: Kandinsky 3.1, kd40: Kandinsky 4.0, kd50: Kandinsky 5.0 T2V Lite"
            ),
            label="Base model",
        )
        pipeline = gr.Radio(
            value=lambda: kubin.params("general", "pipeline"),
            choices=["native", "diffusers"],
            label="Pipeline",
        )

        device = gr.Textbox(
            value=lambda: kubin.params("general", "device"),
            label="Device",
            lines=1,
            max_lines=1,
            elem_classes=["options-small"],
        )

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

        model_name.change(
            fn=None,
            _js=on_change,
            inputs=[
                gr.Text("general.model_name", visible=False),
                model_name,
                gr.Checkbox(False, visible=False),
            ],
            show_progress=False,
        )

    return general_options
