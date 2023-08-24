import gradio as gr
from PIL import Image
from clip_interrogator import Config, Interrogator
from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    BlipForConditionalGeneration,
    Blip2ForConditionalGeneration,
)
import pandas as pd
import torch
import os

title = "Interrogator"


def setup(kubin):
    cache_dir = kubin.params("general", "cache_dir")

    CAPTION_MODELS = {
        "blip-base": "Salesforce/blip-image-captioning-base",  # 990MB
        "blip-large": "Salesforce/blip-image-captioning-large",  # 1.9GB
        "blip2-2.7b": "Salesforce/blip2-opt-2.7b",  # 15.5GB
        "blip2-flan-t5-xl": "Salesforce/blip2-flan-t5-xl",  # 15.77GB
        "git-large-coco": "microsoft/git-large-coco",  # 1.58GB
    }

    def patched_load_caption_model(self):
        if self.config.caption_model is None and self.config.caption_model_name:
            if not self.config.quiet:
                print(f"Loading caption model {self.config.caption_model_name}...")

            model_path = CAPTION_MODELS[self.config.caption_model_name]
            if self.config.caption_model_name.startswith("git-"):
                caption_model = AutoModelForCausalLM.from_pretrained(
                    model_path, torch_dtype=torch.float32, cache_dir=cache_dir
                )
            elif self.config.caption_model_name.startswith("blip2-"):
                caption_model = Blip2ForConditionalGeneration.from_pretrained(
                    model_path, torch_dtype=self.dtype, cache_dir=cache_dir
                )
            else:
                caption_model = BlipForConditionalGeneration.from_pretrained(
                    model_path, torch_dtype=self.dtype, cache_dir=cache_dir
                )
            self.caption_processor = AutoProcessor.from_pretrained(
                model_path, cache_dir=cache_dir
            )

            caption_model.eval()
            if not self.config.caption_offload:
                caption_model = caption_model.to(self.config.device)
            self.caption_model = caption_model
        else:
            self.caption_model = self.config.caption_model
            self.caption_processor = self.config.caption_processor

    Interrogator.load_caption_model = patched_load_caption_model

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
        image = image.convert("RGB")
        interrogated_text = ""

        interrogator = get_interrogator(
            clip_model=clip_model,
            blip_type=blip_type,
            cache_path=f"{cache_dir}/clip_cache",
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
        "send_to": f"📄 Send to {title}",
        "tab_ui": lambda ui_s, ts: interrogator_ui(ui_s, ts),
        "send_target": source_image,
        "api": {
            interrogate: lambda image, mode="fast", clip_model="ViT-L-14/openai", blip_type="large", chunks=2048: interrogate(
                image, mode, clip_model, blip_type, chunks
            )
        },
    }
