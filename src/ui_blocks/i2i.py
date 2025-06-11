import asyncio
import gradio as gr
from ui_blocks.shared.compatibility import batch_size_classes, prior_block_classes
from ui_blocks.shared.samplers import samplers_controls
from ui_blocks.shared.ui_shared import SharedUI
from utils.gradio_ui import click_and_disable
from utils.logging import k_log
import os
from PIL import Image

from utils.storage import get_value
from utils.text import generate_prompt_from_wildcard

block = "i2i"


def i2i_ui(generate_fn, shared: SharedUI, tabs, session):
    augmentations = shared.create_ext_augment_blocks("i2i")
    value = lambda name, def_value: get_value(shared.storage, block, name, def_value)

    with gr.Row() as i2i_block:
        i2i_block.elem_classes = ["i2i_block"]
        with gr.Column(scale=2) as i2i_params:
            with gr.Accordion("PRESETS", open=False, visible=False):
                pass

            augmentations["ui_before_prompt"]()

            with gr.Tabs():
                with gr.TabItem("Single image"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            shared.input_i2i_image.render()
                        with gr.Column(scale=1):
                            prompt = gr.TextArea(
                                value=lambda: value("prompt", ""),
                                placeholder="",
                                label="Prompt",
                                lines=2,
                            )
                            strength = gr.Slider(
                                minimum=0,
                                maximum=1,
                                value=lambda: value("strength", 0.3),
                                step=0.05,
                                label="Transformation strength",
                                info=shared.info(
                                    "Reference image transformation strength"
                                ),
                            )

                    augmentations["ui_before_cnet"]()

                    with gr.Accordion("ControlNet", open=False) as i2i_cnet:
                        cnet_enable = gr.Checkbox(
                            value=lambda: value("cnet_enable", False),
                            label="Enable",
                            elem_classes=["cnet-enable"],
                        )

                        with gr.Row():
                            with gr.Column():
                                cnet_img_reuse = gr.Checkbox(
                                    value=lambda: value("cnet_img_reuse", True),
                                    label="Reuse input image for ControlNet condition",
                                )
                                shared.input_cnet_i2i_image.render()
                                with gr.Row():
                                    cnet_condition = gr.Radio(
                                        choices=["depth-map"],
                                        value=lambda: value(
                                            "cnet_condition", "depth-map"
                                        ),
                                        label="Condition",
                                    )
                                    cnet_depth_estimator = gr.Dropdown(
                                        choices=[
                                            "Intel/dpt-hybrid-midas",
                                            "Intel/dpt-large",
                                        ],
                                        value=lambda: value(
                                            "cnet_depth_estimator", "Intel/dpt-large"
                                        ),
                                        label="Depth estimator",
                                    )

                            cnet_img_reuse.change(
                                lambda x: gr.update(visible=not x),
                                inputs=[cnet_img_reuse],
                                outputs=[shared.input_cnet_i2i_image],
                            )

                            with gr.Column():
                                cnet_emb_transform_strength = gr.Slider(
                                    minimum=0,
                                    maximum=1,
                                    value=lambda: value(
                                        "cnet_emb_transform_strength", 0.85
                                    ),
                                    step=0.05,
                                    label="Embedding strength",
                                )

                                cnet_neg_emb_transform_strength = gr.Slider(
                                    minimum=0,
                                    maximum=1,
                                    value=lambda: value(
                                        "cnet_neg_emb_transform_strength", 1
                                    ),
                                    step=0.05,
                                    label="Negative prior embedding strength",
                                )

                                cnet_img_strength = gr.Slider(
                                    minimum=0,
                                    maximum=1,
                                    value=lambda: value("cnet_img_strength", 0.5),
                                    step=0.05,
                                    label="Image strength",
                                )

                    i2i_cnet.elem_classes = ["control-net", "kubin-accordion"]

                with gr.TabItem("Batch"):
                    with gr.Row():
                        input_folder = gr.Textbox(
                            label="Folder with input images",
                            info=shared.info("Folder to read images from"),
                        )
                        output_folder = gr.Textbox(
                            label="Folder with output images",
                            info=shared.info(
                                "If empty, the default img2img folder will be used"
                            ),
                        )
                    batch_prompt = gr.TextArea(
                        "", placeholder="", label="Prompt", lines=2
                    )
                    with gr.Row():
                        img_extension = gr.Textbox(
                            ".jpg;.jpeg;.png;.bmp",
                            label="File extension filter",
                            info=shared.info(
                                "Only use images with the following extensions"
                            ),
                        )
                        batch_strength = gr.Slider(
                            0,
                            1,
                            0.3,
                            step=0.05,
                            label="Input image strength",
                            info=shared.info("Reference image transformation strength"),
                        )
                    with gr.Row():
                        generate_batch_i2i = gr.Button(
                            "🖼️ Execute batch processing", variant="secondary"
                        )
                        show_processed_i2i = gr.Button(
                            "🔍 Show images from output folder", variant="secondary"
                        )
                    batch_progress = gr.HTML(label="Batch progress")

            augmentations["ui_before_params"]()

            with gr.Accordion(
                "Advanced params", open=not shared.ui_params("collapse_advanced_params")
            ) as i2i_advanced_params:
                with gr.Row():
                    steps = gr.Slider(
                        minimum=1,
                        maximum=200,
                        value=lambda: value(
                            "num_steps", shared.ui_params("decoder_steps_default")
                        ),
                        step=1,
                        label="Steps",
                        elem_classes=["inline-flex"],
                    )
                    guidance_scale = gr.Slider(
                        minimum=1,
                        maximum=30,
                        value=lambda: value("guidance_scale", 4),
                        step=1,
                        label="Guidance scale",
                        elem_classes=["inline-flex"],
                    )
                    batch_count = gr.Slider(
                        minimum=1,
                        maximum=shared.ui_params("max_batch_count"),
                        value=lambda: value("batch_count", 4),
                        step=1,
                        label="Batch count",
                    )

                with gr.Row():
                    width = gr.Slider(
                        minimum=shared.ui_params("image_width_min"),
                        maximum=shared.ui_params("image_width_max"),
                        value=lambda: value(
                            "w", shared.ui_params("image_width_default")
                        ),
                        step=shared.ui_params("image_width_step"),
                        label="Width",
                        elem_id="i2i-width",
                        elem_classes=["prompt-size"],
                    )
                    width.elem_classes = ["inline-flex"]
                    height = gr.Slider(
                        minimum=shared.ui_params("image_height_min"),
                        maximum=shared.ui_params("image_height_max"),
                        value=lambda: value(
                            "h", shared.ui_params("image_height_default")
                        ),
                        step=shared.ui_params("image_height_step"),
                        label="Height",
                        elem_id="i2i-height",
                        elem_classes=["prompt-size"],
                    )
                    height.elem_classes = ["inline-flex"]
                    aspect_ratio = gr.Dropdown(
                        choices=["none"]
                        + shared.ui_params("aspect_ratio_list").split(";"),
                        value="none",
                        label="Aspect ratio",
                        elem_id="i2i-aspect",
                    )
                    width.change(
                        fn=None,
                        _js=f"(width, aspect_ratio) => kubin.UI.aspectRatio.sizeChanged('i2i-width', 'i2i-height', 'width', width, aspect_ratio, {shared.ui_params('image_width_step')})",
                        show_progress=False,
                        inputs=[width, aspect_ratio],
                        outputs=gr.State(None),
                    )
                    height.change(
                        fn=None,
                        _js=f"(height, aspect_ratio) => kubin.UI.aspectRatio.sizeChanged('i2i-width', 'i2i-height', 'height', height, aspect_ratio, {shared.ui_params('image_height_step')})",
                        show_progress=False,
                        inputs=[height, aspect_ratio],
                        outputs=gr.State(None),
                    )
                    aspect_ratio.change(
                        fn=None,
                        _js=f"(width, aspect_ratio) => kubin.UI.aspectRatio.sizeChanged('i2i-width', 'i2i-height', 'width', width, aspect_ratio, {shared.ui_params('image_width_step')})",
                        show_progress=False,
                        inputs=[width, aspect_ratio],
                        outputs=gr.State(None),
                    )

                with gr.Row(equal_height=True):
                    (
                        sampler_20,
                        sampler_21_native,
                        sampler_diffusers,
                    ) = samplers_controls(
                        [
                            value("_sampler20", "p_sampler"),
                            value("_sampler21", "p_sampler"),
                            value("_sampler_diffusers", "DDPM"),
                        ]
                    )

                    seed = gr.Number(
                        value=lambda: value("input_seed", -1), label="Seed", precision=0
                    )
                    batch_size = gr.Slider(
                        minimum=1,
                        maximum=16,
                        value=lambda: value("batch_size", 1),
                        step=1,
                        label="Batch size",
                    )
                    batch_size.elem_classes = batch_size_classes() + ["inline-flex"]

                with gr.Row() as prior_block:
                    prior_scale = gr.Slider(
                        minimum=1,
                        maximum=30,
                        value=lambda: value("prior_cf_scale", 4),
                        step=1,
                        label="Prior guidance scale",
                        elem_classes=["inline-flex"],
                    )
                    prior_steps = gr.Slider(
                        minimum=2,
                        maximum=100,
                        value=lambda: value("prior_steps", 25),
                        step=1,
                        label="Prior steps",
                        elem_classes=["inline-flex"],
                    )
                    negative_prior_prompt = gr.TextArea(
                        value=lambda: value("negative_prior_prompt", ""),
                        label="Negative prior prompt",
                        elem_classes=["inline-flex"],
                        lines=2,
                    )
                prior_block.elem_classes = prior_block_classes()

            augmentations["ui"]()

        with gr.Column(scale=1):
            augmentations["ui_before_generate"]()

            generate_i2i = gr.Button("Generate", variant="primary")
            i2i_output = gr.Gallery(
                label="Generated Images",
                columns=2,
                preview=True,
                elem_classes=["i2i-output"],
            )

            i2i_output.select(
                fn=None,
                _js=f"() => kubin.UI.setImageIndex('i2i-output')",
                show_progress=False,
                outputs=gr.State(None),
            )

            shared.create_base_send_targets(i2i_output, "i2i-output", tabs)
            shared.create_ext_send_targets(i2i_output, "i2i-output", tabs)

            augmentations["ui_after_generate"]()

            async def generate(
                session,
                image,
                prompt,
                negative_prior_prompt,
                strength,
                steps,
                batch_count,
                batch_size,
                guidance_scale,
                width,
                height,
                sampler_20,
                sampler_21_native,
                sampler_diffusers,
                prior_scale,
                prior_steps,
                seed,
                cnet_enable,
                cnet_img_reuse,
                cnet_image,
                cnet_condition,
                cnet_depth_estimator,
                cnet_emb_transform_strength,
                cnet_neg_emb_transform_strength,
                cnet_img_strength,
                *injections,
            ):
                while True:
                    sampler = shared.select_sampler(
                        sampler_20, sampler_21_native, sampler_diffusers
                    )

                    cnet_target_image = image
                    if cnet_enable:
                        if not cnet_img_reuse and cnet_image is None:
                            k_log(
                                "No image selected for ControlNet input, using original image instead"
                            )
                        elif not cnet_img_reuse:
                            cnet_target_image = cnet_image

                    prompt = generate_prompt_from_wildcard(prompt)

                    params = {
                        ".session": session,
                        "init_image": image,
                        "prompt": prompt,
                        "negative_prior_prompt": negative_prior_prompt,
                        "strength": strength,
                        "num_steps": steps,
                        "batch_count": batch_count,
                        "batch_size": batch_size,
                        "guidance_scale": guidance_scale,
                        "w": width,
                        "h": height,
                        "sampler": sampler,
                        "_sampler20": sampler_20,
                        "_sampler21": sampler_21_native,
                        "_sampler_diffusers": sampler_diffusers,
                        "prior_cf_scale": prior_scale,
                        "prior_steps": prior_steps,
                        "input_seed": seed,
                        "cnet_enable": cnet_enable,
                        "cnet_image": cnet_target_image,
                        "cnet_condition": cnet_condition,
                        "cnet_depth_estimator": cnet_depth_estimator,
                        "cnet_emb_transform_strength": cnet_emb_transform_strength,
                        "cnet_neg_emb_transform_strength": cnet_neg_emb_transform_strength,
                        "cnet_img_strength": cnet_img_strength,
                        "negative_prompt": "",
                    }

                    saved_params = {
                        k: v
                        for k, v in params.items()
                        if k not in ["init_image", "cnet_image"]
                    }

                    shared.storage.save(block, saved_params)
                    params = augmentations["exec"](params, injections)

                    yield generate_fn(params)

                    if not shared.check("LOOP_I2I", False):
                        break

            click_and_disable(
                element=generate_i2i,
                fn=generate,
                inputs=[
                    session,
                    shared.input_i2i_image,
                    prompt,
                    negative_prior_prompt,
                    strength,
                    steps,
                    batch_count,
                    batch_size,
                    guidance_scale,
                    width,
                    height,
                    sampler_20,
                    sampler_21_native,
                    sampler_diffusers,
                    prior_scale,
                    prior_steps,
                    seed,
                    cnet_enable,
                    cnet_img_reuse,
                    shared.input_cnet_i2i_image,
                    cnet_condition,
                    cnet_depth_estimator,
                    cnet_emb_transform_strength,
                    cnet_neg_emb_transform_strength,
                    cnet_img_strength,
                ]
                + augmentations["injections"],
                outputs=i2i_output,
                js=[
                    "args => kubin.UI.taskStarted('Image To Image')",
                    "args => kubin.UI.taskFinished('Image To Image')",
                ],
            )

            def generate_batch(
                session,
                input_folder,
                output_folder,
                extensions,
                batch_prompt,
                batch_strength,
                steps,
                batch_count,
                batch_size,
                guidance_scale,
                width,
                height,
                sampler_20,
                sampler_21_native,
                sampler_diffusers,
                prior_scale,
                prior_steps,
                seed,
                *injections,
            ):
                sampler = shared.select_sampler(
                    sampler_20, sampler_21_native, sampler_diffusers
                )

                if not os.path.exists(input_folder):
                    return "Input folder does not exist"

                if input_folder == output_folder:
                    return "Input and output folder cannot be the same"

                i2i_source = []
                for filename in os.listdir(input_folder):
                    if filename.endswith(tuple(extensions.split(";"))):
                        i2i_source.append(filename)

                k_log(f"found {len(i2i_source)} images for i2i processing")

                for index, imagename in enumerate(i2i_source):
                    imagepath = f"{input_folder}/{imagename}"
                    image = Image.open(imagepath)

                    params = {
                        ".session": session,
                        "init_image": image,
                        "prompt": batch_prompt,
                        "strength": batch_strength,
                        "num_steps": steps,
                        "batch_count": batch_count,
                        "batch_size": batch_size,
                        "guidance_scale": guidance_scale,
                        "w": width,
                        "h": height,
                        "sampler": sampler,
                        "prior_cf_scale": prior_scale,
                        "prior_steps": prior_steps,
                        "input_seed": seed,
                    }

                    params = augmentations["exec"](params, injections)
                    if output_folder != "":
                        params[".output_dir"] = output_folder

                    k_log(f"{index+1}/{len(i2i_source)}: processing {imagepath}")
                    _ = generate_fn(params)
                return f"{len(i2i_source)} images successfully processed"

            click_and_disable(
                generate_batch_i2i,
                generate_batch,
                [
                    session,
                    input_folder,
                    output_folder,
                    img_extension,
                    batch_prompt,
                    batch_strength,
                    steps,
                    batch_count,
                    batch_size,
                    guidance_scale,
                    width,
                    height,
                    sampler_20,
                    sampler_21_native,
                    sampler_diffusers,
                    prior_scale,
                    prior_steps,
                    seed,
                ]
                + augmentations["injections"],
                [batch_progress],
            )

            def show_processed_images(output_folder, extensions):
                if not os.path.exists(output_folder):
                    return [
                        gr.Gallery.update(visible=True),
                        "Output folder does not exist",
                    ]

                output_images = []
                for filename in os.listdir(output_folder):
                    if filename.endswith(tuple(extensions.split(";"))):
                        output_images.append(f"{output_folder}/{filename}")

                return [output_images, ""]

            click_and_disable(
                show_processed_i2i,
                fn=show_processed_images,
                inputs=[output_folder, img_extension],
                outputs=[i2i_output, batch_progress],
            )

        i2i_params.elem_classes = ["block-params i2i_params"]
        i2i_advanced_params.elem_classes = [
            "block-advanced-params",
            "i2i_advanced_params",
            "kubin-accordion",
        ]
    return i2i_block
