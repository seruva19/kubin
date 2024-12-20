import asyncio
import gradio as gr
from ui_blocks.shared.compatibility import (
    batch_size_classes,
    negative_prompt_classes,
    prior_block_classes,
)
from ui_blocks.shared.samplers import samplers_controls
from ui_blocks.shared.ui_shared import SharedUI
from utils.gradio_ui import click_and_disable
from utils.storage import get_value
from utils.text import generate_prompt_from_wildcard

block = "inpaint"


# TODO: implement region of inpainting
def inpaint_ui(generate_fn, shared: SharedUI, tabs, session):
    augmentations = shared.create_ext_augment_blocks("inpaint")
    value = lambda name, def_value: get_value(shared.storage, block, name, def_value)

    with gr.Row() as inpaint_block:
        inpaint_block.elem_classes = ["inpaint_block"]
        with gr.Column(scale=2) as inpaint_params:
            with gr.Accordion("PRESETS", open=False, visible=False):
                pass

            augmentations["ui_before_prompt"]()

            with gr.Row():
                shared.input_inpaint_image.render()
            with gr.Column():
                prompt = gr.TextArea(
                    value("prompt", ""), placeholder="", label="Prompt", lines=2
                )
                negative_prompt = gr.TextArea(
                    value("negative_prompt", ""),
                    placeholder="",
                    label="Negative prompt",
                    lines=2,
                )
                negative_prompt.elem_classes = negative_prompt_classes()

            augmentations["ui_before_cnet"]()
            augmentations["ui_before_params"]()

            with gr.Accordion(
                "Advanced params", open=not shared.ui_params("collapse_advanced_params")
            ) as inpaint_advanced_params:
                with gr.Row():
                    inpainting_target = gr.Radio(
                        ["only mask", "all but mask"],
                        value=lambda: value("target", "only mask"),
                        label="Inpainting target",
                    )
                    inpainting_region = gr.Radio(
                        ["whole", "mask"],
                        value=lambda: value("region", "whole"),
                        label="Inpainting composition",
                    )
                with gr.Row():
                    steps = gr.Slider(
                        minimum=1,
                        maximum=200,
                        value=lambda: value(
                            "num_steps", shared.ui_params("decoder_steps_default")
                        ),
                        step=1,
                        label="Steps",
                    )
                    guidance_scale = gr.Slider(
                        minimum=1,
                        maximum=30,
                        value=lambda: value("guidance_scale", 4),
                        step=1,
                        label="Guidance scale",
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
                        elem_id="inpaint-width",
                        interactive=False,
                        elem_classes=["prompt-size", "inline-flex"],
                    )
                    height = gr.Slider(
                        minimum=shared.ui_params("image_height_min"),
                        maximum=shared.ui_params("image_height_max"),
                        value=lambda: value(
                            "h", shared.ui_params("image_height_default")
                        ),
                        step=shared.ui_params("image_height_step"),
                        label="Height",
                        elem_id="inpaint-height",
                        interactive=False,
                        elem_classes=["prompt-size", "inline-flex"],
                    )
                    with gr.Column():
                        infer_size = gr.Checkbox(
                            value=lambda: value("infer_size", True),
                            label="Infer image size from input image",
                            elem_classes=["inline-flex"],
                        )
                        aspect_ratio = gr.Dropdown(
                            choices=["none"]
                            + shared.ui_params("aspect_ratio_list").split(";"),
                            value="none",
                            allow_custom_value=True,
                            interactive=False,
                            label="Aspect ratio",
                            elem_id="inpaint-aspect",
                        )
                    width.change(
                        fn=None,
                        _js=f"(width, aspect_ratio) => kubin.UI.aspectRatio.sizeChanged('inpaint-width', 'inpaint-height', 'width', width, aspect_ratio, {shared.ui_params('image_width_step')})",
                        show_progress=False,
                        inputs=[width, aspect_ratio],
                        outputs=gr.State(None),
                    )
                    height.change(
                        fn=None,
                        _js=f"(height, aspect_ratio) => kubin.UI.aspectRatio.sizeChanged('inpaint-width', 'inpaint-height', 'height', height, aspect_ratio, {shared.ui_params('image_height_step')})",
                        show_progress=False,
                        inputs=[height, aspect_ratio],
                        outputs=gr.State(None),
                    )
                    aspect_ratio.change(
                        fn=None,
                        _js=f"(width, aspect_ratio) => kubin.UI.aspectRatio.sizeChanged('inpaint-width', 'inpaint-height', 'width', width, aspect_ratio, {shared.ui_params('image_width_step')})",
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
                    # TODO: fix https://github.com/ai-forever/Kandinsky-2/issues/53
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
                        value=lambda: value("prior_steps", 5),
                        step=1,
                        label="Prior steps",
                        elem_classes=["inline-flex"],
                    )
                    negative_prior_prompt = gr.TextArea(
                        value=lambda: value("negative_prior_prompt", ""),
                        label="Negative prior prompt",
                        lines=2,
                    )
                prior_block.elem_classes = prior_block_classes()

            augmentations["ui"]()

        with gr.Column(scale=1):
            augmentations["ui_before_generate"]()

            generate_inpaint = gr.Button("Generate", variant="primary")
            inpaint_output = gr.Gallery(
                label="Generated Images",
                columns=2,
                preview=True,
                elem_classes=["inpaint-output"],
            )

            inpaint_output.select(
                fn=None,
                _js=f"() => kubin.UI.setImageIndex('inpaint-output')",
                show_progress=False,
                outputs=gr.State(None),
            )

            shared.create_base_send_targets(inpaint_output, "inpaint-output", tabs)
            shared.create_ext_send_targets(inpaint_output, "inpaint-output", tabs)

            augmentations["ui_after_generate"]()

            infer_size.change(
                fn=lambda x: [
                    gr.update(interactive=not x),
                    gr.update(interactive=not x),
                    gr.update(interactive=not x),
                ],
                inputs=[infer_size],
                outputs=[width, height, aspect_ratio],
            )

            async def generate(
                session,
                image_mask,
                prompt,
                negative_prompt,
                inpainting_target,
                inpainting_region,
                steps,
                batch_count,
                batch_size,
                guidance_scale,
                w,
                h,
                sampler_20,
                sampler_21_native,
                sampler_diffusers,
                prior_cf_scale,
                prior_steps,
                negative_prior_prompt,
                input_seed,
                infer_size,
                *injections,
            ):
                while True:
                    sampler = shared.select_sampler(
                        sampler_20, sampler_21_native, sampler_diffusers
                    )

                    prompt = generate_prompt_from_wildcard(prompt)

                    params = {
                        ".session": session,
                        "image_mask": image_mask,
                        "prompt": prompt,
                        "negative_prompt": negative_prompt,
                        "target": inpainting_target,
                        "region": inpainting_region,
                        "num_steps": steps,
                        "batch_count": batch_count,
                        "batch_size": batch_size,
                        "guidance_scale": guidance_scale,
                        "w": w,
                        "h": h,
                        "sampler": sampler,
                        "_sampler20": sampler_20,
                        "_sampler21": sampler_21_native,
                        "_sampler_diffusers": sampler_diffusers,
                        "prior_cf_scale": prior_cf_scale,
                        "prior_steps": prior_steps,
                        "negative_prior_prompt": negative_prior_prompt,
                        "input_seed": input_seed,
                        "infer_size": infer_size,
                    }

                    shared.storage.save(block, params)
                    params = augmentations["exec"](params, injections)

                    yield generate_fn(params)
                    await asyncio.sleep(1)

                    if not shared.check("LOOP_INPAINT", False):
                        break

        click_and_disable(
            element=generate_inpaint,
            fn=generate,
            inputs=[
                session,
                shared.input_inpaint_image,
                prompt,
                negative_prompt,
                inpainting_target,
                inpainting_region,
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
                negative_prior_prompt,
                seed,
                infer_size,
            ]
            + augmentations["injections"],
            outputs=inpaint_output,
            js=[
                "args => kubin.UI.taskStarted('Inpainting')",
                "args => kubin.UI.taskFinished('Inpainting')",
            ],
        )

        inpaint_params.elem_classes = ["block-params inpaint_params"]
        inpaint_advanced_params.elem_classes = [
            "block-advanced-params inpaint_advanced_params"
        ]
    return inpaint_block
