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

block = "t2i"


def t2i_ui(generate_fn, shared: SharedUI, tabs, session):
    augmentations = shared.create_ext_augment_blocks(block)
    value = lambda name, def_value: get_value(shared.storage, block, name, def_value)

    with gr.Row() as t2i_block:
        t2i_block.elem_classes = ["t2i_block"]
        with gr.Column(scale=2) as t2i_params:
            with gr.Accordion("PRESETS", open=False, visible=False):
                pass

            augmentations["ui_before_prompt"]()

            prompt = gr.TextArea(
                value=lambda: value("prompt", ""),
                label="Prompt",
                placeholder="",
                lines=2,
            )
            negative_prompt = gr.TextArea(
                value=lambda: value("negative_prompt", ""),
                placeholder="",
                label="Negative prompt",
                lines=2,
            )
            negative_prompt.elem_classes = negative_prompt_classes()

            augmentations["ui_before_cnet"]()

            with gr.Accordion("ControlNet", open=False) as t2i_cnet:
                cnet_enable = gr.Checkbox(
                    value=lambda: value("cnet_enable", False),
                    label="Enable",
                    elem_classes=["cnet-enable"],
                )

                with gr.Row():
                    shared.input_cnet_t2i_image.render()
                    with gr.Column():
                        cnet_pipeline = gr.Dropdown(
                            value=lambda: value(
                                "cnet_pipeline", "ControlNetImg2ImgPipeline"
                            ),
                            choices=["ControlNetPipeline", "ControlNetImg2ImgPipeline"],
                            type="value",
                            label="Processing pipeline",
                            allow_custom_value=False,
                        )
                        with gr.Row():
                            cnet_condition = gr.Radio(
                                value=lambda: value("cnet_condition", "depth-map"),
                                choices=["depth-map"],
                                label="Condition",
                            )
                            cnet_depth_estimator = gr.Dropdown(
                                value=lambda: value(
                                    "cnet_depth_estimator", "Intel/dpt-large"
                                ),
                                choices=["Intel/dpt-hybrid-midas", "Intel/dpt-large"],
                                label="Depth estimator",
                            )
                        with gr.Column(visible=True) as cnet_i2i_params:
                            cnet_emb_transform_strength = gr.Slider(
                                value=lambda: value(
                                    "cnet_emb_transform_strength", 0.85
                                ),
                                minimum=0,
                                maximum=1,
                                step=0.05,
                                label="Embedding strength",
                                info=shared.info("Strength of reference embedding"),
                            )

                            cnet_neg_emb_transform_strength = gr.Slider(
                                value=lambda: value(
                                    "cnet_neg_emb_transform_strength", 1
                                ),
                                minimum=0,
                                maximum=1,
                                step=0.05,
                                label="Negative embedding strength",
                                info=shared.info(
                                    "Strength of reference negative embedding"
                                ),
                            )

                            cnet_img_strength = gr.Slider(
                                value=lambda: value("cnet_img_strength", 0.5),
                                minimum=0,
                                maximum=1,
                                step=0.05,
                                label="Image strength",
                                info=shared.info("Strength of reference image"),
                            )

            def pipeline_changed(pipeline):
                return gr.update(
                    visible=True if pipeline == "ControlNetImg2ImgPipeline" else False
                )

            cnet_pipeline.change(
                pipeline_changed,
                inputs=[cnet_pipeline],
                outputs=[cnet_i2i_params],
            )

            t2i_cnet.elem_classes = ["control-net", "kubin-accordion"]

            augmentations["ui_before_params"]()

            with gr.Accordion(
                "Advanced params", open=not shared.ui_params("collapse_advanced_params")
            ) as t2i_advanced_params:
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
                        value=lambda: value("batch_count", 1),
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
                        elem_id="t2i-width",
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
                        elem_id="t2i-height",
                        elem_classes=["prompt-size"],
                    )
                    height.elem_classes = ["inline-flex"]
                    aspect_ratio = gr.Dropdown(
                        choices=["none"]
                        + shared.ui_params("aspect_ratio_list").split(";"),
                        value="none",
                        allow_custom_value=True,
                        label="Aspect ratio",
                        elem_classes=["t2i-aspect"],
                    )
                    width.change(
                        fn=None,
                        _js=f"(width, aspect_ratio) => kubin.UI.aspectRatio.sizeChanged('t2i-width', 't2i-height', 'width', width, aspect_ratio, {shared.ui_params('image_width_step')})",
                        show_progress=False,
                        inputs=[width, aspect_ratio],
                        outputs=gr.State(None),
                    )
                    height.change(
                        fn=None,
                        _js=f"(height, aspect_ratio) => kubin.UI.aspectRatio.sizeChanged('t2i-width', 't2i-height', 'height', height, aspect_ratio, {shared.ui_params('image_height_step')})",
                        show_progress=False,
                        inputs=[height, aspect_ratio],
                        outputs=gr.State(None),
                    )
                    aspect_ratio.change(
                        fn=None,
                        _js=f"(width, aspect_ratio) => kubin.UI.aspectRatio.sizeChanged('t2i-width', 't2i-height', 'width', width, aspect_ratio, {shared.ui_params('image_width_step')})",
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
            t2i_advanced_params.elem_classes = [
                "block-advanced-params",
                "t2i_advanced_params",
                "kubin-accordion",
            ]

            augmentations["ui"]()

        t2i_params.elem_classes = ["block-params", "t2i_params"]

        with gr.Column(scale=1):
            augmentations["ui_before_generate"]()
            generate_t2i = gr.Button("Generate", variant="primary")
            t2i_output = gr.Gallery(
                label="Generated Images",
                columns=2,
                preview=True,
                elem_classes=["t2i-output"],
            )

            t2i_output.select(
                fn=None,
                _js=f"() => kubin.UI.setImageIndex('t2i-output')",
                show_progress=False,
                outputs=gr.State(None),
            )

            shared.create_base_send_targets(t2i_output, "t2i-output", tabs)
            shared.create_ext_send_targets(t2i_output, "t2i-output", tabs)

            augmentations["ui_after_generate"]()

            async def generate(
                session,
                prompt,
                negative_prompt,
                num_steps,
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
                cnet_enable,
                cnet_image,
                cnet_pipeline,
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

                    prompt = generate_prompt_from_wildcard(prompt)

                    params = {
                        ".session": session,
                        "prompt": prompt,
                        "negative_prompt": negative_prompt,
                        "num_steps": num_steps,
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
                        "cnet_enable": cnet_enable,
                        "cnet_image": cnet_image,
                        "cnet_pipeline": cnet_pipeline,
                        "cnet_condition": cnet_condition,
                        "cnet_depth_estimator": cnet_depth_estimator,
                        "cnet_emb_transform_strength": cnet_emb_transform_strength,
                        "cnet_neg_emb_transform_strength": cnet_neg_emb_transform_strength,
                        "cnet_img_strength": cnet_img_strength,
                        "init_image": None,
                    }

                    shared.storage.save(block, params)
                    params = augmentations["exec"](params, injections)

                    yield generate_fn(params)
                    await asyncio.sleep(1)

                    if not shared.check("LOOP_T2I", False):
                        break

            click_and_disable(
                element=generate_t2i,
                fn=generate,
                inputs=[
                    session,
                    prompt,
                    negative_prompt,
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
                    cnet_enable,
                    shared.input_cnet_t2i_image,
                    cnet_pipeline,
                    cnet_condition,
                    cnet_depth_estimator,
                    cnet_emb_transform_strength,
                    cnet_neg_emb_transform_strength,
                    cnet_img_strength,
                ]
                + augmentations["injections"],
                outputs=t2i_output,
                js=[
                    "args => kubin.UI.taskStarted('Text To Image')",
                    "args => kubin.UI.taskFinished('Text To Image')",
                ],
            )

    return t2i_block
