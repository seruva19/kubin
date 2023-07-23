import gradio as gr

from train_modules.train_21.train_prior_ui import train_prior_ui
from train_modules.train_21.train_unclip_ui import train_unclip_ui
from train_modules.train_dataset import train_dataset_ui
from train_modules.train_tools import train_tools_ui

from train_modules.train_22.train_22_decoder_ui import train_22_decoder_ui
from train_modules.train_22.train_22_prior_ui import train_22_prior_ui
from train_modules.lora_22.train_lora_prior_ui import train_lora_prior_ui
from train_modules.lora_22.train_lora_decoder_ui import train_lora_decoder_ui
from train_modules.dreambooth_22.train_22_dreambooth_prior_ui import (
    train_22_dreambooth_prior_ui,
)
from train_modules.dreambooth_22.train_22_dreambooth_decoder_ui import (
    train_22_dreambooth_decoder_ui,
)


def setup(kubin):
    def training_ui(ui_shared, ui_tabs):
        with gr.Column() as training_block:
            with gr.Tabs() as training_tabs:
                with gr.TabItem("2.2 LoRA") as training_selector_kd22_lora:
                    training_selector_kd22_lora.elem_classes = [
                        "training-selector-kd22-lora"
                    ]
                    with gr.TabItem("Prior", id="training-lora-prior"):
                        lora_prior_ui = train_lora_prior_ui(kubin, training_tabs)

                    with gr.TabItem("Decoder", id="training-lora-decoder"):
                        lora_decoder_ui = train_lora_decoder_ui(kubin, training_tabs)

                # with gr.TabItem("2.2 Fine-tuning") as training_selector_kd22:
                #     training_selector_kd22.elem_classes = ["training-selector-kd22"]

                #     with gr.TabItem("Prior", id="training-22-prior"):
                #         training_22_prior_ui = train_22_prior_ui(kubin, training_tabs)

                #     with gr.TabItem("Decoder", id="training-22-decoder"):
                #         training_22_decoder_ui = train_22_decoder_ui(
                #             kubin, training_tabs
                #         )

                # with gr.TabItem("2.2 Dreambooth") as training_selector_kd22_dreambooth:
                #     training_selector_kd22_dreambooth.elem_classes = [
                #         "training-selector-kd22-dreambooth"
                #     ]
                #     training_22_dreambooth_prior_block = train_22_dreambooth_prior_ui(
                #         kubin, training_tabs
                #     )
                #     training_22_dreambooth_decoder_block = (
                #         train_22_dreambooth_decoder_ui(kubin, training_tabs)
                #     )

                with gr.TabItem("2.1 Fine-tuning") as training_selector_kd21:
                    training_selector_kd21.elem_classes = ["training-selector-kd21"]

                    with gr.TabItem("Prior", id="training-prior"):
                        prior_ui = train_prior_ui(kubin, training_tabs)

                    with gr.TabItem("UnCLIP", id="training-unclip"):
                        unclip_ui = train_unclip_ui(kubin, training_tabs)

                with gr.TabItem("Dataset", id="training-dataset"):
                    tools_ui = train_dataset_ui(kubin, training_tabs)

                with gr.TabItem("Tools", id="training-tools"):
                    tools_ui = train_tools_ui(kubin, training_tabs)

        return training_block

    return {"title": "Training", "tab_ui": lambda ui_s, ts: training_ui(ui_s, ts)}
