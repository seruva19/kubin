from nn_tools.nn_viewer import get_path_by_name_and_hash, read_model_info
import gradio as gr
import gc
import os
import torch
from file_tools import (
    calculate_file_hash,
    filenames_with_hash,
    scan_for_models,
    load_model_from_path,
)


def networks_browser_ui(kubin, config, prior_loras, decoder_loras, networks_list):
    lora_paths = config["lora_path"]
    lora_prior_pattern = config["lora_prior_pattern"]
    lora_decoder_pattern = config["lora_decoder_pattern"]

    with gr.Accordion("Base", open=True):
        model_components = gr.State({})

        scan_model = gr.Button(value="👓 Scan model components", scale=0, size="sm")
        model_components_list = gr.Radio(
            scale=0,
            label="List of model components",
            choices=[],
            visible=False,
            interactive=True,
        )

    with gr.Accordion("LoRA", open=True):
        with gr.Row():
            lora_prior_list = gr.Dropdown(
                label="List of prior LoRA models",
                choices=prior_loras,
            )

            lora_decoder_list = gr.Dropdown(
                label="List of decoder LoRA models",
                choices=decoder_loras,
            )

            rescan_loras_btn = gr.Button(
                value="🔄 Rescan LoRA models", scale=0, size="sm"
            )

    network_metadata = gr.HTML(
        "",
        label="Network info",
        interactive=False,
        visible=False,
        elem_classes=["network-info"],
    )

    def scan_lora_models():
        networks_list["loras"] = scan_for_models(
            lora_paths, [lora_prior_pattern, lora_decoder_pattern]
        )

        prior_loras, decoder_loras = filenames_with_hash(networks_list["loras"])
        return [gr.update(choices=prior_loras), gr.update(choices=decoder_loras)]

    def read_model_components():
        model = kubin.params("general", "model_name")
        pipeline = kubin.params("general", "pipeline")

        if model == "kd21" and pipeline == "native":
            return [
                None,
                {
                    "model": kubin.model,
                },
                gr.update(choices=["model"], visible=True),
            ]

        elif model == "kd22" and pipeline == "diffusers":
            try:
                components = {
                    "pipe_prior": kubin.model.pipe_prior,
                    "pipe_prior_e2e": kubin.model.pipe_prior_e2e,
                    "t2i_pipe": kubin.model.t2i_pipe,
                    "i2i_pipe": kubin.model.i2i_pipe,
                    "inpaint_pipe": kubin.model.inpaint_pipe,
                    "cnet_t2i_pipe": kubin.model.cnet_t2i_pipe,
                    "cnet_i2i_pipe": kubin.model.cnet_i2i_pipe,
                }

                return [
                    None,
                    components,
                    gr.update(choices=list(components.keys()), visible=True),
                ]
            except Exception as exception:
                return [
                    gr.update(
                        visible=True,
                        value=f"Failed to read model metadata with error: {exception}",
                    ),
                    gr.update(visible=False),
                    gr.update(visible=False),
                ]
        else:
            return [
                gr.update(
                    visible=True,
                    value=f"Scanning ({model}, {pipeline}) model metadata is not implemented yet",
                ),
                gr.update(visible=False),
                gr.update(visible=False),
            ]

    def read_component_metadata(pipe_name, components):
        model = kubin.params("general", "model_name")
        pipeline = kubin.params("general", "pipeline")

        if model == "kd21" and pipeline == "native":
            data = []
            system_conf = kubin.model.system_config
            for key in system_conf:
                data.append(f"{str(key)}: {system_conf[key]}")
            return gr.update(visible=True, value="\n".join(data))

        elif model == "kd22" and pipeline == "diffusers":
            pipe = components[pipe_name]
            if pipe is None:
                return gr.update(
                    visible=True,
                    value="Component is not loaded, activate required pipeline to analyze",
                )

            if "prior" in pipe_name:
                return gr.update(visible=True, value=read_model_info(pipe.prior))
            else:
                return gr.update(visible=True, value=read_model_info(pipe.unet))

    def read_network_metadata(name_hash):
        path = get_path_by_name_and_hash(networks_list["loras"], name_hash)
        lora_model = load_model_from_path(path)

        output = []
        output.append(f"path: {path}")
        output.append(f"hash: {calculate_file_hash(path)}\n")
        for index, key in enumerate(lora_model.keys()):
            tensor = lora_model[key]
            output.append(f"({index+1}) {key}: {tensor.shape} [{tensor.dtype}]")

        return gr.update(visible=True, value="\n".join(output))

    scan_model.click(
        fn=read_model_components,
        inputs=None,
        outputs=[network_metadata, model_components, model_components_list],
    )

    model_components_list.change(
        read_component_metadata,
        inputs=[model_components_list, model_components],
        outputs=network_metadata,
    )

    lora_prior_list.select(
        read_network_metadata, inputs=[lora_prior_list], outputs=network_metadata
    )
    lora_decoder_list.select(
        read_network_metadata, inputs=[lora_decoder_list], outputs=network_metadata
    )

    rescan_loras_btn.click(
        fn=scan_lora_models,
        inputs=None,
        outputs=[lora_prior_list, lora_decoder_list],
    )
