import gradio as gr
from ui_blocks.shared.samplers import samplers_controls
from ui_blocks.shared.ui_shared import SharedUI
from utils.gradio_ui import click_and_disable


def update(image):
    no_image = image == None
    return gr.update(
        label="Prompt" if no_image else "Prompt (ignored, using image instead)",
        visible=no_image,
        interactive=no_image,
    )


# TODO: add mixing for images > 2
# gradio does not directly support dynamic number of elements https://github.com/gradio-app/gradio/issues/2680
def mix_ui(generate_fn, shared: SharedUI, tabs, session):
    augmentations = shared.create_ext_augment_blocks("mix")

    with gr.Row() as mix_block:
        with gr.Column(scale=2) as mix_params:
            augmentations["ui_before_prompt"]()

            with gr.Row():
                with gr.Column(scale=1):
                    shared.input_mix_image_1.render()
                    text_1 = gr.TextArea("", placeholder="", label="Prompt", lines=2)
                    shared.input_mix_image_1.change(
                        fn=update, inputs=shared.input_mix_image_1, outputs=text_1
                    )
                    weight_1 = gr.Slider(0, 1, 0.5, step=0.05, label="Weight")
                with gr.Column(scale=1):
                    shared.input_mix_image_2.render()
                    text_2 = gr.TextArea("", placeholder="", label="Prompt", lines=2)
                    shared.input_mix_image_2.change(
                        fn=update, inputs=shared.input_mix_image_2, outputs=text_2
                    )
                    weight_2 = gr.Slider(0, 1, 0.5, step=0.05, label="Weight")

            negative_prompt = gr.TextArea("", label="Negative prompt", lines=2)

            augmentations["ui_before_cnet"]()

            with gr.Accordion("ControlNet", open=False) as mix_cnet:
                cnet_enable = gr.Checkbox(
                    False, label="Enable", elem_classes=["cnet-enable"]
                )

                with gr.Row():
                    shared.input_cnet_mix_image.render()
                    with gr.Column():
                        cnet_condition = gr.Radio(
                            choices=["depth-map"],
                            value="depth-map",
                            label="Condition",
                        )

                        cnet_img_strength = gr.Slider(
                            0, 1, 1, step=0.05, label="Image strength"
                        )

            mix_cnet.elem_classes = ["control-net"]

            augmentations["ui_before_params"]()

            with gr.Accordion(
                "Advanced params", open=not shared.ui_params("collapse_advanced_params")
            ) as mix_advanced_params:
                with gr.Row():
                    steps = gr.Slider(
                        1,
                        200,
                        shared.ui_params("decoder_steps_default"),
                        step=1,
                        label="Steps",
                    )
                    guidance_scale = gr.Slider(1, 30, 4, step=1, label="Guidance scale")
                with gr.Row():
                    batch_count = gr.Slider(
                        1,
                        shared.ui_params("max_batch_count"),
                        4,
                        step=1,
                        label="Batch count",
                    )
                    batch_size = gr.Slider(1, 16, 1, step=1, label="Batch size")
                with gr.Row():
                    width = gr.Slider(
                        shared.ui_params("image_width_min"),
                        shared.ui_params("image_width_max"),
                        shared.ui_params("image_width_default"),
                        step=shared.ui_params("image_width_step"),
                        label="Width",
                    )
                    height = gr.Slider(
                        shared.ui_params("image_height_min"),
                        shared.ui_params("image_height_max"),
                        shared.ui_params("image_height_default"),
                        step=shared.ui_params("image_height_step"),
                        label="Height",
                    )
                with gr.Row():
                    (
                        sampler_20,
                        sampler_21_native,
                        sampler_diffusers,
                    ) = samplers_controls()

                    seed = gr.Number(-1, label="Seed", precision=0)
                with gr.Row():
                    prior_scale = gr.Slider(
                        1,
                        100,
                        4,
                        step=1,
                        label="Prior scale",
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
                        lines=2,
                    )

            augmentations["ui"]()

        with gr.Column(scale=1):
            generate_mix = gr.Button("Generate", variant="primary")
            mix_output = gr.Gallery(
                label="Generated Images",
                columns=2,
                preview=True,
                elem_classes="mix-output",
            )

            mix_output.select(
                fn=None,
                _js=f"() => kubin.UI.setImageIndex('mix-output')",
                show_progress=False,
                outputs=gr.State(None),
            )

            shared.create_base_send_targets(mix_output, "mix-output", tabs)
            shared.create_ext_send_targets(mix_output, "mix-output", tabs)

            def generate(
                session,
                image_1,
                image_2,
                text_1,
                text_2,
                weight_1,
                weight_2,
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
                cnet_img_strength,
                *injections,
            ):
                sampler = shared.select_sampler(
                    sampler_20, sampler_21_native, sampler_diffusers
                )

                params = {
                    ".session": session,
                    "image_1": image_1,
                    "image_2": image_2,
                    "text_1": text_1,
                    "text_2": text_2,
                    "weight_1": weight_1,
                    "weight_2": weight_2,
                    "negative_prompt": negative_prompt,
                    "num_steps": steps,
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
                    "cnet_condition": cnet_condition,
                    "cnet_img_strength": cnet_img_strength,
                }

                params = augmentations["exec"](params, injections)
                return generate_fn(params)

        click_and_disable(
            element=generate_mix,
            fn=generate,
            inputs=[
                session,
                shared.input_mix_image_1,
                shared.input_mix_image_2,
                text_1,
                text_2,
                weight_1,
                weight_2,
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
        mix_advanced_params.elem_classes = ["block-advanced-params mix_advanced_params"]
    return mix_block
