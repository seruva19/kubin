import gradio as gr
from PIL import Image
from clip_interrogator import Config, Interrogator
import pandas as pd
import torch
import os

title = "Interrogator"
use_monkey_patch = False


def patched_prepare_inputs_for_generation(
    self,
    input_ids,
    past=None,
    attention_mask=None,
    encoder_hidden_states=None,
    encoder_attention_mask=None,
    **model_kwargs,
):
    input_shape = input_ids.shape

    if attention_mask is None:
        attention_mask = input_ids.new_ones(input_shape)

    if past is not None:
        input_ids = input_ids[:, -1:]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "past_key_values": past,
        "encoder_hidden_states": encoder_hidden_states,
        "encoder_attention_mask": encoder_attention_mask,
        "is_decoder": True,
    }


# 1) monkey patch to prevent https://github.com/huggingface/transformers/issues/19290
# 2) force download of CLIP/BLIP models into app models folder
def use_patch(kubin):
    from blip.models.med import BertLMHeadModel

    old_method = BertLMHeadModel.prepare_inputs_for_generation
    old_torch_dir = torch.hub.get_dir()

    BertLMHeadModel.prepare_inputs_for_generation = (
        patched_prepare_inputs_for_generation
    )
    torch.hub.set_dir(kubin.params("general", "cache_dir"))
    return old_method, old_torch_dir


def cancel_patch(patch):
    from blip.models.med import BertLMHeadModel

    BertLMHeadModel.prepare_inputs_for_generation = patch[0]
    torch.hub.set_dir(patch[1])


def setup(kubin):
    ci = None
    ci_config = None
    source_image = gr.Image(type="pil", label="Input image", elem_classes=[])

    def get_interrogator(clip_model, blip_type, cache_path, chunk_size):
        nonlocal ci
        nonlocal ci_config

        if ci is None or [clip_model, blip_type] != ci_config:
            ci_config = [clip_model, blip_type]
            ci = Interrogator(
                Config(
                    clip_model_name=clip_model,
                    caption_model_name=blip_type,
                    clip_model_path=cache_path,
                    cache_path=cache_path,
                    download_cache=True,
                    chunk_size=chunk_size,
                )
            )

        return ci

    def interrogate(image, mode, clip_model, blip_type, chunk_size):
        if use_monkey_patch:
            patch = use_patch(kubin)

        image = image.convert("RGB")
        interrogated_text = ""

        interrogator = get_interrogator(
            clip_model=clip_model,
            blip_type=blip_type,
            cache_path=f"{kubin.params('general','cache_dir')}/clip_cache",
            chunk_size=chunk_size,
        )
        if mode == "best":
            interrogated_text = interrogator.interrogate(image)
        elif mode == "classic":
            interrogated_text = interrogator.interrogate_classic(image)
        elif mode == "fast":
            interrogated_text = interrogator.interrogate_fast(image)
        elif mode == "negative":
            interrogated_text = interrogator.interrogate_negative(image)

        if use_monkey_patch:
            cancel_patch(patch)

        return interrogated_text

    def batch_interrogate(
        image_dir,
        batch_mode,
        image_extensions,
        output_dir,
        caption_extension,
        output_csv,
        mode,
        clip_model,
        blip_type,
        chunk_size,
        progress=gr.Progress(),
    ):
        if output_dir == "":
            output_dir = image_dir
        os.makedirs(output_dir, exist_ok=True)

        relevant_images = []

        progress(0, desc="Starting batch interrogation...")
        if not os.path.exists(image_dir):
            return f"Error: image folder {image_dir} does not exists"

        for filename in os.listdir(image_dir):
            if filename.endswith(tuple(image_extensions)):
                relevant_images.append([filename, f"{image_dir}/{filename}", ""])

        print(f"found {len(relevant_images)} images to interrogate")
        image_count = 0
        for _ in progress.tqdm(relevant_images, unit="images"):
            filename = relevant_images[image_count][0]
            filepath = relevant_images[image_count][1]

            image = Image.open(filepath)
            caption = interrogate(image, mode, clip_model, blip_type, chunk_size)

            if batch_mode == 0:
                caption_filename = os.path.splitext(filename)[0]
                with open(
                    f"{output_dir}/{caption_filename}{caption_extension}",
                    "w",
                    encoding="utf-8",
                ) as file:
                    file.write(caption)
            elif batch_mode == 1:
                relevant_images[image_count][2] = caption

            image_count += 1

        if batch_mode == 1:
            captions_df = pd.DataFrame(
                [i[1:] for i in relevant_images], columns=["image_name", "caption"]
            )
            csv_path = f"{output_dir}/{output_csv}"
            captions_df.to_csv(csv_path, index=False)
            print(f"CSV file with captions saved to {csv_path}")

        return f"Captions for {len(relevant_images)} images created"

    def interrogator_ui(ui_shared, ui_tabs):
        with gr.Row() as interrogator_block:
            with gr.Column(scale=1) as interrogator_params_block:
                with gr.Row():
                    clip_model = gr.Dropdown(
                        choices=["ViT-L-14/openai", "ViT-H-14/laion2b_s32b_b79k"],
                        value="ViT-L-14/openai",
                        label="CLIP model",
                    )
                with gr.Row():
                    mode = gr.Radio(
                        ["best", "classic", "fast", "negative"],
                        value="fast",
                        label="Mode",
                    )
                with gr.Row():
                    blip_model_type = gr.Radio(
                        ["blip-base", "blip-large", "git-large-coco"],
                        value="blip-large",
                        label="Caption model name",
                    )
                with gr.Row():
                    chunk_size = gr.Slider(
                        512, 2048, 2048, step=512, label="Chunk size"
                    )

            with gr.Column(scale=1):
                with gr.Tabs():
                    with gr.TabItem("Single image"):
                        with gr.Column(scale=1):
                            with gr.Row():
                                source_image.render()

                        with gr.Column(scale=1):
                            interrogate_btn = gr.Button(
                                "Interrogate", variant="primary"
                            )
                            target_text = gr.Textbox(
                                lines=5,
                                label="Interrogated text",
                                show_copy_button=True,
                            )

                            kubin.ui_utils.click_and_disable(
                                interrogate_btn,
                                fn=interrogate,
                                inputs=[
                                    source_image,
                                    mode,
                                    clip_model,
                                    blip_model_type,
                                    chunk_size,
                                ],
                                outputs=[target_text],
                                js=[
                                    f"args => kubin.UI.taskStarted('{title}')",
                                    f"args => kubin.UI.taskFinished('{title}')",
                                ],
                            )
                    with gr.TabItem("Batch"):
                        image_dir = gr.Textbox(label="Directory with images")

                        with gr.Row():
                            image_types = gr.CheckboxGroup(
                                [".jpg", ".jpeg", ".png", ".bmp"],
                                value=[".jpg", ".jpeg", ".png", ".bmp"],
                                label="Files to interrogate",
                            )

                        caption_mode = gr.Radio(
                            choices=["text files", "csv dataset"],
                            info="Save captions to separate text files or to a single csv file",
                            value="text files",
                            label="Caption save mode",
                            type="index",
                        )
                        output_dir = gr.Textbox(
                            label="Output folder",
                            info="If empty, the same folder will be used",
                        )

                        caption_extension = gr.Textbox(
                            ".txt", label="Caption files extension", visible=True
                        )
                        output_csv = gr.Textbox(
                            value="captions.csv",
                            label="Name of csv file",
                            visible=False,
                        )

                        caption_mode.select(
                            fn=lambda m: [
                                gr.update(visible=m != 0),
                                gr.update(visible=m == 0),
                            ],
                            inputs=[caption_mode],
                            outputs=[caption_extension, output_csv],
                        )

                        batch_interrogate_btn = gr.Button(
                            "Interrogate", variant="primary"
                        )
                        progress = gr.HTML(label="Interrogation progress")

                        kubin.ui_utils.click_and_disable(
                            batch_interrogate_btn,
                            fn=batch_interrogate,
                            inputs=[
                                image_dir,
                                caption_mode,
                                image_types,
                                output_dir,
                                caption_extension,
                                output_csv,
                                mode,
                                clip_model,
                                blip_model_type,
                                chunk_size,
                            ],
                            outputs=[progress],
                            js=[
                                f"args => kubin.UI.taskStarted('{title}')",
                                f"args => kubin.UI.taskFinished('{title}')",
                            ],
                        )

            interrogator_params_block.elem_classes = ["block-params"]
        return interrogator_block

    return {
        "title": title,
        "send_to": f"📄 Send to{title}",
        "tab_ui": lambda ui_s, ts: interrogator_ui(ui_s, ts),
        "send_target": source_image,
        "api": {
            interrogate: lambda image, mode="fast", clip_model="ViT-L-14/openai", blip_type="large", chunks=2048: interrogate(
                image, mode, clip_model, blip_type, chunks
            )
        },
    }
