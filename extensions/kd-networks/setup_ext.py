from nn_tools.nn_attach import bind_networks
from networks_selector import networks_selector_ui
from networks_browser import networks_browser_ui
from networks_converter import networks_converter_ui
from networks_downloader import networks_downloader_ui
import gradio as gr
from pathlib import Path
from file_tools import scan_for_models, filenames_with_hash

dir = Path(__file__).parent.absolute()

ignored = False

networks_list = {"loras": []}
networks_info = {"lora": {}}


def lora_state(lora_enabled, prior_path, decoder_path):
    return {
        "enabled": lora_enabled,
        "prior": prior_path,
        "decoder": decoder_path,
        "binded": False,
    }


def setup(kubin):
    yaml_config = kubin.yaml_utils.YamlConfig(dir)
    config = yaml_config.read()

    lora_paths = config["lora_path"]
    lora_prior_pattern = config["lora_prior_pattern"]
    lora_decoder_pattern = config["lora_decoder_pattern"]

    networks_list["loras"] = scan_for_models(
        lora_paths, [lora_prior_pattern, lora_decoder_pattern]
    )

    prior_loras, decoder_loras = filenames_with_hash(networks_list["loras"])

    def networks_ui(ui_shared):
        with gr.Column() as networks_block:
            with gr.Tabs() as networks_tabs:
                with gr.TabItem("Browser") as networks_browser_block:
                    networks_browser_ui(
                        kubin, config, prior_loras, decoder_loras, networks_list
                    )

                with gr.TabItem("Converter") as networks_converter_block:
                    networks_converter_ui(kubin)

                # with gr.TabItem("Downloader") as networks_downloader_block:
                #     networks_downloader_ui(kubin)

        return networks_block

    def on_hook(hook, **kwargs):
        if hook == kubin.params.HOOK.BEFORE_PREPARE_PARAMS:
            model = kwargs["model"]
            if hasattr(model, "config"):
                bind_networks(
                    kubin,
                    model.config,
                    kwargs["prior"],
                    kwargs["decoder"],
                    kwargs["params"],
                    kwargs["task"],
                    networks_info,
                )
            else:
                if not ignored:
                    ignored = True
                    kubin.log(
                        "Current model does not support additional networks, so 'BEFORE_PREPARE_PARAMS' hook is ignored."
                    )

    def settings_ui():
        def save_changes(inputs):
            config["lora_path"] = inputs[lora_path]
            config["lora_prior_pattern"] = inputs[lora_prior_pattern]
            config["lora_decoder_pattern"] = inputs[lora_decoder_pattern]
            config["autopair_lora_models"] = inputs[autopair_lora_models]
            yaml_config.write(config)

        with gr.Column() as settings_block:
            lora_path = gr.Textbox(
                lambda: config["lora_path"], label="Path to LORA models", scale=0
            )
            lora_prior_pattern = gr.Textbox(
                lambda: config["lora_prior_pattern"],
                label="LoRA prior model files mask",
                scale=0,
            )
            lora_decoder_pattern = gr.Textbox(
                lambda: config["lora_decoder_pattern"],
                label="LoRA decoder model files mask",
                scale=0,
            )
            autopair_lora_models = gr.Checkbox(
                lambda: config["autopair_lora_models"],
                label="Autopair prior and decoder on selection",
                scale=0,
            )

            save_btn = gr.Button("Save settings", size="sm", scale=0)
            save_btn.click(
                save_changes,
                inputs={
                    lora_path,
                    lora_prior_pattern,
                    lora_decoder_pattern,
                    autopair_lora_models,
                },
                queue=False,
            ).then(fn=None, _js=("(x) => kubin.notify.success('Settings saved')"))

        settings_block.elem_classes = ["k-form"]
        return settings_block

    return {
        "targets": ["t2i", "i2i", "mix", "inpaint", "outpaint"],
        "title": "Networks",
        "tab_ui": lambda ui_s, ts: networks_ui(ui_s),
        "inject_ui": lambda target, kubin=kubin: networks_selector_ui(
            kubin,
            target,
            config,
            prior_loras,
            decoder_loras,
            networks_list,
            networks_info,
        ),
        "hook_fn": on_hook,
        "settings_ui": settings_ui,
        "supports": ["diffusers-kd22"],
    }
