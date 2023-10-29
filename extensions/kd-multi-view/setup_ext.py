import torch
import random
import gradio as gr
import numpy
import rembg

from PIL import Image
from diffusers import (
    DiffusionPipeline,
    EulerAncestralDiscreteScheduler,
    DDPMParallelScheduler,
)

title = "Multi View"


def setup(kubin):
    pipeline = None
    source_image = gr.Image(type="pil", label="Source image")

    def create_multiview(cache_dir, device, source_image, seed):
        nonlocal pipeline

        if pipeline is None:
            pipeline = DiffusionPipeline.from_pretrained(
                "sudo-ai/zero123plus-v1.1",
                custom_pipeline="sudo-ai/zero123plus-pipeline",
                torch_dtype=torch.float16,
                cache_dir=cache_dir,
            )

            pipeline.scheduler = DDPMParallelScheduler.from_config(
                pipeline.scheduler.config, timestep_spacing="trailing"
            )

            pipeline.to(device)

        if seed == -1:
            seed = random.randint(1, 1000000)

        source_image = rembg.remove(source_image)
        result = pipeline(
            source_image,
            num_inference_steps=75,
            generator=torch.Generator(pipeline.device).manual_seed(int(seed)),
        ).images[0]
        return result

    def multiview_ui(ui_shared, ui_tabs):
        with gr.Row() as multiview_block:
            with gr.Column(scale=1) as multiview_params_block:
                with gr.Row():
                    source_image.render()

            with gr.Column(scale=1):
                create_btn = gr.Button(
                    "Generate multiview", label="Generate multiview", variant="primary"
                )
                multiview_output = gr.Image(label="Generated multiview")

            kubin.ui_utils.click_and_disable(
                create_btn,
                fn=create_multiview,
                inputs=[
                    gr.State(kubin.params("general", "cache_dir")),
                    gr.State(kubin.params("general", "device")),
                    source_image,
                    gr.State(-1),
                ],
                outputs=multiview_output,
                js=[
                    f"args => kubin.UI.taskStarted('{title}')",
                    f"args => kubin.UI.taskFinished('{title}')",
                ],
            )

            multiview_params_block.elem_classes = ["block-params"]
        return multiview_block

    return {
        "send_to": f"📇 Send to {title}",
        "title": title,
        "tab_ui": lambda ui_s, ts: multiview_ui(ui_s, ts),
        "send_target": source_image,
    }
