import gradio as gr
import gc
import torch

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

extension_title = "Training"


def clear_models(loaded_models):
    for model in loaded_models:
        model.to("cpu")
    gc.collect()
    if torch.cuda.is_available():
        with torch.cuda.device("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


def setup(kubin):
    utils = {
        "loaded_models": [],
        "clear_models": lambda: clear_models(utils.models),
    }

    def training_ui(ui_shared, ui_tabs, clear_models):
        with gr.Column() as training_block:
            with gr.Tabs() as training_tabs:
                with gr.TabItem(
                    "2.2 LoRA", id="training-kd22-lora"
                ) as training_kd22_lora:
                    training_kd22_lora.elem_classes = ["training-selector-kd22-lora"]
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

                # with gr.TabItem("2.2 Dreambooth") as training_kd22_dreambooth:
                #     training_kd22_dreambooth.elem_classes = [
                #         "training-selector-kd22-dreambooth"
                #     ]
                #     training_22_dreambooth_prior_block = train_22_dreambooth_prior_ui(
                #         kubin, training_tabs
                #     )
                #     training_22_dreambooth_decoder_block = (
                #         train_22_dreambooth_decoder_ui(kubin, training_tabs)
                #     )

                with gr.TabItem("2.1 Fine-tuning", id="training-kd21") as training_kd21:
                    training_kd21.elem_classes = ["training-selector-kd21"]

                    with gr.TabItem("Prior", id="training-prior"):
                        training_kd21_prior_ui = train_prior_ui(kubin, training_tabs)

                    with gr.TabItem("UnCLIP", id="training-unclip"):
                        training_kd21_unclip_ui = train_unclip_ui(kubin, training_tabs)

                # with gr.TabItem("2.1 Textual Inversion") as training_kd21_ti:
                #     training_kd21_ti.elem_classes = ["training_kd21_ti"]

                # with gr.TabItem("2.1 Dreambooth") as training_kd21_dreambooth:
                #     training_kd21_dreambooth.elem_classes = ["training_kd21_dreambooth"]

                with gr.TabItem("Dataset", id="training-dataset"):
                    tools_ui = train_dataset_ui(kubin, training_tabs)

                with gr.TabItem("Tools", id="training-tools"):
                    tools_ui = train_tools_ui(kubin, training_tabs)

        training_block.elem_classes = ["kd-training-block"]
        return training_block

    return {
        "title": extension_title,
        "tab_ui": lambda ui_s, ts: training_ui(ui_s, ts, utils["clear_models"]),
    }
