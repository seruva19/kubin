import gradio as gr
import os
import torch


def networks_converter_ui(kubin):
    with gr.Row() as nn_converters_block:
        with gr.Accordion("Network conversion", open=True):
            with gr.Row():
                conversion_from = gr.Dropdown(
                    choices=["bin"],
                    type="value",
                    value="bin",
                    label="Source format",
                )
                conversion_to = gr.Dropdown(
                    choices=["safetensors"],
                    type="value",
                    value="safetensors",
                    label="Target format",
                )

            with gr.Column():
                with gr.Row():
                    source_path = gr.Text("", label="Source path")
                    target_path = gr.Text(
                        "",
                        label="Target path",
                        info="Leaving empty will save to the same folder and the same name",
                    )

                with gr.Row():
                    convert_nn_btn = gr.Button("🔀 Convert", scale=0)

            def convert_network(convert_from, convert_to, source_path, target_path):
                if convert_from == "bin" and convert_to == "safetensors":
                    if target_path == "":
                        target_folder = os.path.dirname(source_path)
                        filename = os.path.basename(source_path)
                        filename_without_extension = os.path.splitext(filename)[0]
                        target_path = os.path.join(
                            target_folder, filename_without_extension + ".safetensors"
                        )

                    kubin.nn_utils.convert_pt_to_safetensors(source_path, target_path)

            kubin.ui_utils.click_and_disable(
                convert_nn_btn,
                convert_network,
                [conversion_from, conversion_to, source_path, target_path],
                None,
            ).then(
                fn=None,
                _js="_ => kubin.notify.success('Conversion completed')",
                inputs=None,
                outputs=None,
            )

    return nn_converters_block
