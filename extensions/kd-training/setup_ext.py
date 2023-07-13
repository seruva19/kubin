import gradio as gr

from train_modules.train_prior_ui import train_prior_ui
from train_modules.train_unclip_ui import train_unclip_ui
from train_modules.train_tools import train_tools_ui


def setup(kubin):
    def training_ui(ui_shared, ui_tabs):
        with gr.Column() as training_block:
            with gr.Tabs() as training_tabs:
                with gr.TabItem("2.1") as training_selector_kd21:
                    training_selector_kd21.elem_classes = ["training-selector-kd21"]

                    with gr.TabItem("Prior", id="training-prior"):
                        prior_ui = train_prior_ui(kubin, training_tabs)

                    with gr.TabItem("UnCLIP", id="training-unclip"):
                        unclip_ui = train_unclip_ui(kubin, training_tabs)

                with gr.TabItem("Dataset", id="training-tools"):
                    utils_ui = train_tools_ui(kubin)

        return training_block

    def inference_ui(target):
        with gr.Column() as model_block:
            with gr.Tab("Checkpoint"):
                model = gr.Dropdown(choices=[], label="Prior")
                model = gr.Dropdown(choices=[], label="Unclip")

        return model_block, model

    def apply_model(params, model):
        print(f"applying model {model}")

        return params

    def settings_ui():
        with gr.Column() as settings_block:
            None

        return settings_block

    return {
        "title": "Training",
        "targets": ["t2i", "i2i", "mix", "inpaint", "outpaint"],
        "tab_ui": lambda ui_s, ts: training_ui(ui_s, ts),
        "inject_title": "Model",
        # 'inject_ui': lambda target: inference_ui(target),
        "inject_fn": lambda target, params, augmentations: apply_model(
            params, augmentations[0]
        ),
        "hook_fn": lambda h: None,
        "settings_ui": lambda target: settings_ui(),
    }
