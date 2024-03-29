import gradio as gr
from ui_blocks.shared.compatibility import (
    batch_size_classes,
    negative_prompt_classes,
    prior_block_classes,
)
from ui_blocks.shared.samplers import samplers_controls
from ui_blocks.shared.ui_shared import SharedUI
from utils.gradio_ui import click_and_disable


def t2i_ui(generate_fn, shared: SharedUI, tabs, session):
    augmentations = shared.create_ext_augment_blocks("t2i")

    with gr.Row() as t2i_block:
        t2i_block.elem_classes = ["t2i_block"]
        with gr.Column(scale=2) as t2i_params:
            augmentations["ui_before_prompt"]()

            prompt = gr.TextArea("", label="Prompt", placeholder="", lines=2)
            negative_prompt = gr.TextArea(
                "",
                placeholder="",
                label="Negative prompt",
                lines=2,
            )
            negative_prompt.elem_classes = negative_prompt_classes()

            augmentations["ui_before_cnet"]()

            with gr.Accordion("ControlNet", open=False) as t2i_cnet:
                cnet_enable = gr.Checkbox(
                    False, label="Enable", elem_classes=["cnet-enable"]
                )

                with gr.Row():
                    shared.input_cnet_t2i_image.render()
                    with gr.Column():
                        cnet_pipeline = gr.Dropdown(
                            choices=["ControlNetPipeline", "ControlNetImg2ImgPipeline"],
                            value="ControlNetImg2ImgPipeline",
                            type="value",
                            label="Processing pipeline",
                            allow_custom_value=False,
                        )
                        with gr.Row():
                            cnet_condition = gr.Radio(
                                choices=["depth-map"],
                                value="depth-map",
                                label="Condition",
                            )
                            cnet_depth_estimator = gr.Dropdown(
                                choices=["Intel/dpt-hybrid-midas", "Intel/dpt-large"],
                                value="Intel/dpt-large",
                                label="Depth estimator",
                            )
                        with gr.Column(visible=True) as cnet_i2i_params:
                            cnet_emb_transform_strength = gr.Slider(
                                0,
                                1,
                                0.85,
                                step=0.05,
                                label="Embedding strength",
                                info=shared.info("Strength of reference embedding"),
                            )

                            cnet_neg_emb_transform_strength = gr.Slider(
                                0,
                                1,
                                1,
                                step=0.05,
                                label="Negative embedding strength",
                                info=shared.info(
                                    "Strength of reference negative embedding"
                                ),
                            )

                            cnet_img_strength = gr.Slider(
                                0,
                                1,
                                0.5,
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
                        1,
                        200,
                        shared.ui_params("decoder_steps_default"),
                        step=1,
                        label="Steps",
                    )
                    guidance_scale = gr.Slider(1, 30, 4, step=1, label="Guidance scale")
                    batch_count = gr.Slider(
                        1,
                        shared.ui_params("max_batch_count"),
                        4,
                        step=1,
                        label="Batch count",
                    )
                with gr.Row():
                    width = gr.Slider(
                        shared.ui_params("image_width_min"),
                        shared.ui_params("image_width_max"),
                        shared.ui_params("image_width_default"),
                        step=shared.ui_params("image_width_step"),
                        label="Width",
                        elem_id="t2i-width",
                        elem_classes=["prompt-size"],
                    )
                    width.elem_classes = ["inline-flex"]
                    height = gr.Slider(
                        shared.ui_params("image_height_min"),
                        shared.ui_params("image_height_max"),
                        shared.ui_params("image_height_default"),
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
                    ) = samplers_controls()
                    seed = gr.Number(-1, label="Seed", precision=0)
                    batch_size = gr.Slider(1, 16, 1, step=1, label="Batch size")
                    batch_size.elem_classes = batch_size_classes() + ["inline-flex"]

                with gr.Row() as prior_block:
                    prior_scale = gr.Slider(
                        1,
                        30,
                        4,
                        step=1,
                        label="Prior guidance scale",
                        elem_classes=["inline-flex"],
                    )
                    prior_steps = gr.Slider(
                        2,
                        100,
                        25,
                        step=1,
                        label="Prior steps",
                        elem_classes=["inline-flex"],
                    )
                    negative_prior_prompt = gr.TextArea(
                        "",
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

            def generate(
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
                sampler = shared.select_sampler(
                    sampler_20, sampler_21_native, sampler_diffusers
                )

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

                params = augmentations["exec"](params, injections)
                return generate_fn(params)

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
