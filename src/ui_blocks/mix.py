import asyncio
import gradio as gr
from ui_blocks.shared.compatibility import batch_size_classes, prior_block_classes
from ui_blocks.shared.samplers import samplers_controls
from ui_blocks.shared.ui_shared import SharedUI
from utils.gradio_ui import click_and_disable
from utils.storage import get_value
from utils.text import generate_prompt_from_wildcard

block = "mix"


def update(image):
    no_image = image == None
    return gr.update(
        label="Prompt" if no_image else "Prompt (ignored, using image instead)",
        visible=no_image,
        interactive=no_image,
    )


def mix_ui(generate_fn, shared: SharedUI, tabs, session):
    augmentations = shared.create_ext_augment_blocks("mix")
    value = lambda name, def_value: get_value(shared.storage, block, name, def_value)

    with gr.Row() as mix_block:
        mix_block.elem_classes = ["mix_block"]
        with gr.Column(scale=2) as mix_params:
            with gr.Accordion("PRESETS", open=False, visible=False):
                pass

            augmentations["ui_before_prompt"]()

            with gr.Row(visible=False):
                mix_image_count = gr.Slider(
                    minimum=2,
                    maximum=6,
                    value=lambda: value(
                        "mix_image_count", shared.ui_params("mix_image_count")
                    ),
                    step=1,
                    label="Mix image count",
                )

            with gr.Row():
                with gr.Column(scale=1):
                    shared.input_mix_images[0].render()
                    text_1 = gr.TextArea(
                        value=lambda: value("text_1", ""),
                        placeholder="",
                        label="Prompt 1",
                        lines=2,
                    )
                    shared.input_mix_images[0].change(
                        fn=update, inputs=shared.input_mix_images[0], outputs=text_1
                    )
                    weight_1 = gr.Slider(
                        minimum=0,
                        maximum=1,
                        value=lambda: value("weight_1", 0.5),
                        step=0.05,
                        label="Weight 1",
                    )
                with gr.Column(scale=1):
                    shared.input_mix_images[1].render()
                    text_2 = gr.TextArea(
                        value=lambda: value("text_2", ""),
                        placeholder="",
                        label="Prompt 2",
                        lines=2,
                    )
                    shared.input_mix_images[1].change(
                        fn=update, inputs=shared.input_mix_images[1], outputs=text_2
                    )
                    weight_2 = gr.Slider(
                        minimum=0,
                        maximum=1,
                        value=lambda: value("weight_2", 0.5),
                        step=0.05,
                        label="Weight 2",
                    )

            with gr.Row(visible=False):
                with gr.Column(scale=1):
                    shared.input_mix_images[2].render()
                    text_3 = gr.TextArea(
                        value=lambda: value("text_3", ""),
                        placeholder="",
                        label="Prompt 3",
                        lines=2,
                    )
                    shared.input_mix_images[2].change(
                        fn=update, inputs=shared.input_mix_images[2], outputs=text_3
                    )
                    weight_3 = gr.Slider(
                        minimum=0,
                        maximum=1,
                        value=lambda: value("weight_3", 0.5),
                        step=0.05,
                        label="Weight 3",
                    )
                with gr.Column(scale=1):
                    shared.input_mix_images[3].render()
                    text_4 = gr.TextArea(
                        value=lambda: value("text_4", ""),
                        placeholder="",
                        label="Prompt 4",
                        lines=2,
                    )
                    shared.input_mix_images[3].change(
                        fn=update, inputs=shared.input_mix_images[3], outputs=text_2
                    )
                    weight_4 = gr.Slider(
                        minimum=0,
                        maximum=1,
                        value=lambda: value("weight_4", 0.5),
                        step=0.05,
                        label="Weight 4",
                    )

            with gr.Row(visible=False):
                with gr.Column(scale=1):
                    shared.input_mix_images[4].render()
                    text_5 = gr.TextArea(
                        value=lambda: value("text_5", ""),
                        placeholder="",
                        label="Prompt 5",
                        lines=2,
                    )
                    shared.input_mix_images[4].change(
                        fn=update, inputs=shared.input_mix_images[4], outputs=text_5
                    )
                    weight_5 = gr.Slider(
                        minimum=0,
                        maximum=1,
                        value=lambda: value("weight_5", 0.5),
                        step=0.05,
                        label="Weight 5",
                    )

                with gr.Column(scale=1):
                    shared.input_mix_images[5].render()
                    text_6 = gr.TextArea(
                        value=lambda: value("text_6", ""),
                        placeholder="",
                        label="Prompt 6",
                        lines=2,
                    )
                    shared.input_mix_images[5].change(
                        fn=update, inputs=shared.input_mix_images[5], outputs=text_6
                    )
                    weight_6 = gr.Slider(
                        minimum=0,
                        maximum=1,
                        value=lambda: value("weight_6", 0.5),
                        step=0.05,
                        label="Weight 6",
                    )

            negative_prompt = gr.TextArea(
                value=lambda: value("negative_prompt", ""),
                label="Negative prompt",
                lines=2,
            )

            augmentations["ui_before_cnet"]()

            with gr.Accordion("ControlNet", open=False) as mix_cnet:
                cnet_enable = gr.Checkbox(
                    value=lambda: value("cnet_enable", False),
                    label="Enable",
                    elem_classes=["cnet-enable"],
                )

                with gr.Row():
                    shared.input_cnet_mix_image.render()
                    with gr.Column():
                        with gr.Row():
                            cnet_condition = gr.Radio(
                                choices=["depth-map"],
                                value=lambda: value("cnet_condition", "depth-map"),
                                label="Condition",
                            )
                            cnet_depth_estimator = gr.Dropdown(
                                choices=["Intel/dpt-hybrid-midas", "Intel/dpt-large"],
                                value=lambda: value(
                                    "cnet_depth_estimator", "Intel/dpt-large"
                                ),
                                label="Depth estimator",
                            )

                        cnet_img_strength = gr.Slider(
                            minimum=0,
                            maximum=1,
                            value=lambda: value("cnet_img_strength", 1),
                            step=0.05,
                            label="Image strength",
                        )

            mix_cnet.elem_classes = ["control-net", "kubin-accordion"]

            augmentations["ui_before_params"]()

            with gr.Accordion(
                "Advanced params", open=not shared.ui_params("collapse_advanced_params")
            ) as mix_advanced_params:
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
                        elem_id="mix-width",
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
                        elem_id="mix-height",
                        elem_classes=["prompt-size"],
                    )
                    height.elem_classes = ["inline-flex"]
                    aspect_ratio = gr.Dropdown(
                        choices=["none"]
                        + shared.ui_params("aspect_ratio_list").split(";"),
                        value="none",
                        label="Aspect ratio",
                        elem_id="mix-aspect",
                    )
                    width.change(
                        fn=None,
                        _js=f"(width, aspect_ratio) => kubin.UI.aspectRatio.sizeChanged('mix-width', 'mix-height', 'width', width, aspect_ratio, {shared.ui_params('image_width_step')})",
                        show_progress=False,
                        inputs=[width, aspect_ratio],
                        outputs=gr.State(None),
                    )
                    height.change(
                        fn=None,
                        _js=f"(height, aspect_ratio) => kubin.UI.aspectRatio.sizeChanged('mix-width', 'mix-height', 'height', height, aspect_ratio, {shared.ui_params('image_height_step')})",
                        show_progress=False,
                        inputs=[height, aspect_ratio],
                        outputs=gr.State(None),
                    )
                    aspect_ratio.change(
                        fn=None,
                        _js=f"(width, aspect_ratio) => kubin.UI.aspectRatio.sizeChanged('mix-width', 'mix-height', 'width', width, aspect_ratio, {shared.ui_params('image_width_step')})",
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
                        lines=2,
                    )
                prior_block.elem_classes = prior_block_classes()

            augmentations["ui"]()

        with gr.Column(scale=1):
            augmentations["ui_before_generate"]()

            generate_mix = gr.Button("Generate", variant="primary")
            mix_output = gr.Gallery(
                label="Generated Images",
                columns=2,
                preview=True,
                elem_classes=["mix-output"],
            )

            mix_output.select(
                fn=None,
                _js=f"() => kubin.UI.setImageIndex('mix-output')",
                show_progress=False,
                outputs=gr.State(None),
            )

            shared.create_base_send_targets(mix_output, "mix-output", tabs)
            shared.create_ext_send_targets(mix_output, "mix-output", tabs)

            augmentations["ui_after_generate"]()

            async def generate(
                session,
                mix_image_count,
                image_1,
                image_2,
                image_3,
                image_4,
                image_5,
                image_6,
                text_1,
                text_2,
                text_3,
                text_4,
                text_5,
                text_6,
                weight_1,
                weight_2,
                weight_3,
                weight_4,
                weight_5,
                weight_6,
                negative_prompt,
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
                cnet_enable,
                cnet_image,
                cnet_condition,
                cnet_depth_estimator,
                cnet_img_strength,
                *injections,
            ):
                while True:
                    sampler = shared.select_sampler(
                        sampler_20, sampler_21_native, sampler_diffusers
                    )

                    params = {
                        ".session": session,
                        "mix_image_count": mix_image_count,
                        "image_1": image_1,
                        "image_2": image_2,
                        "image_3": image_3,
                        "image_4": image_4,
                        "image_5": image_5,
                        "image_6": image_6,
                        "text_1": text_1,
                        "text_2": text_2,
                        "text_2": text_2,
                        "text_3": text_3,
                        "text_4": text_4,
                        "text_5": text_5,
                        "text_6": text_6,
                        "weight_1": weight_1,
                        "weight_2": weight_2,
                        "weight_3": weight_3,
                        "weight_4": weight_4,
                        "weight_5": weight_5,
                        "weight_6": weight_6,
                        "negative_prompt": negative_prompt,
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
                        "cnet_enable": cnet_enable,
                        "cnet_image": cnet_image,
                        "cnet_condition": cnet_condition,
                        "cnet_depth_estimator": cnet_depth_estimator,
                        "cnet_img_strength": cnet_img_strength,
                    }

                    shared.storage.save(block, params)
                    params = augmentations["exec"](params, injections)

                    yield generate_fn(params)

                    if not shared.check("LOOP_MIX", False):
                        break

        click_and_disable(
            element=generate_mix,
            fn=generate,
            inputs=[
                session,
                mix_image_count,
                shared.input_mix_images[0],
                shared.input_mix_images[1],
                shared.input_mix_images[2],
                shared.input_mix_images[3],
                shared.input_mix_images[4],
                shared.input_mix_images[5],
                text_1,
                text_2,
                text_3,
                text_4,
                text_5,
                text_6,
                weight_1,
                weight_2,
                weight_3,
                weight_4,
                weight_5,
                weight_6,
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
                shared.input_cnet_mix_image,
                cnet_condition,
                cnet_depth_estimator,
                cnet_img_strength,
            ]
            + augmentations["injections"],
            outputs=mix_output,
            js=[
                "args => kubin.UI.taskStarted('Mix Images')",
                "args => kubin.UI.taskFinished('Mix Images')",
            ],
        )

        mix_params.elem_classes = ["block-params mix_params"]
        mix_advanced_params.elem_classes = [
            "block-advanced-params",
            "mix_advanced_params",
            "kubin-accordion",
        ]
    return mix_block
