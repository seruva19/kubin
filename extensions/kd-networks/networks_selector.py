from file_tools import filenames_with_hash, scan_for_models
from nn_tools.nn_viewer import get_path_by_name_and_hash
import gradio as gr


def networks_selector_ui(
    kubin, target, config, prior_loras, decoder_loras, networks_list, networks_info
):
    lora_paths = config["lora_path"]
    lora_prior_pattern = config["lora_prior_pattern"]
    lora_decoder_pattern = config["lora_decoder_pattern"]

    with gr.Column() as lora_selector:
        lora_selector.elem_classes = ["k-form"]
        enable_lora = gr.Checkbox(False, label="Enable LoRA")

        with gr.Row():
            lora_prior_selector = gr.Dropdown(
                label="Prior LoRA",
                choices=prior_loras,
                interactive=True,
                scale=4,
            )
            lora_decoder_selector = gr.Dropdown(
                label="Decoder LoRA",
                choices=decoder_loras,
                interactive=True,
                scale=4,
            )

            rescan_btn = gr.Button(
                value="🔄 Rescan",
                scale=1,
                size="sm",
            )

            def scan_lora_models():
                networks_list["loras"] = scan_for_models(
                    lora_paths, [lora_prior_pattern, lora_decoder_pattern]
                )

                prior_loras, decoder_loras = filenames_with_hash(networks_list["loras"])

                return [
                    gr.update(choices=prior_loras),
                    gr.update(choices=decoder_loras),
                ]

            rescan_btn.click(
                fn=scan_lora_models,
                inputs=None,
                outputs=[lora_prior_selector, lora_decoder_selector],
            )

        def enable_lora_binding(session, enable_lora, prior, decoder):
            prior_path = get_path_by_name_and_hash(networks_list["loras"], prior)
            decoder_path = get_path_by_name_and_hash(networks_list["loras"], decoder)

            lora_state = {
                "enabled": enable_lora,
                "prior": prior_path,
                "decoder": decoder_path,
            }
            networks_info["lora"][session] = lora_state

        session = gr.Textbox(visible=False)
        enable_lora.change(
            _js="(...args) => [window._kubinSession, ...args.slice(1)]",
            fn=enable_lora_binding,
            inputs=[session, enable_lora, lora_prior_selector, lora_decoder_selector],
            outputs=[],
        )
        lora_prior_selector.select(
            _js="(...args) => [window._kubinSession, ...args.slice(1)]",
            fn=enable_lora_binding,
            inputs=[session, enable_lora, lora_prior_selector, lora_decoder_selector],
            outputs=[],
        )
        lora_decoder_selector.select(
            _js="(...args) => [window._kubinSession, ...args.slice(1)]",
            fn=enable_lora_binding,
            inputs=[session, enable_lora, lora_prior_selector, lora_decoder_selector],
            outputs=[],
        )
    return lora_selector
