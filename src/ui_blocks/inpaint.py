import gradio as gr
from ui_blocks.shared.samplers import samplers_controls
from ui_blocks.shared.ui_shared import SharedUI
from utils.gradio_ui import click_and_disable


# TODO: implement region of inpainting
def inpaint_ui(generate_fn, shared: SharedUI, tabs):
    augmentations = shared.create_ext_augment_blocks("inpaint")

    with gr.Row() as inpaint_block:
        with gr.Column(scale=2) as inpaint_params:
            with gr.Row():
                with gr.Column(scale=1):
                    shared.input_inpaint_image.render()

                with gr.Column():
                    prompt = gr.TextArea("", placeholder="", label="Prompt", lines=2)
                    negative_prompt = gr.TextArea(
                        "", placeholder="", label="Negative prompt", lines=2
                    )

            with gr.Accordion(
                "Advanced params", open=not shared.ui_params("collapse_advanced_params")
            ) as inpaint_advanced_params:
                with gr.Row():
                    inpainting_target = gr.Radio(
                        ["only mask", "all but mask"],
                        value="only mask",
                        label="Inpainting target",
                    )
                    inpainting_region = gr.Radio(
                        ["whole", "mask"],
                        value="whole",
                        label="Inpainting region",
                        interactive=False,
                    )
                with gr.Row():
                    steps = gr.Slider(
                        1,
                        200,
                        shared.ui_params("decoder_steps_default"),
                        step=1,
                        label="Steps",
                    )
                    guidance_scale = gr.Slider(
                        1, 30, 10, step=1, label="Guidance scale"
                    )
                with gr.Row():
                    batch_count = gr.Slider(
                        1,
                        shared.ui_params("max_batch_count"),
                        4,
                        step=1,
                        label="Batch count",
                    )
                    batch_size = gr.Slider(1, 16, 1, step=1, label="Batch size")
                    # TODO: fix https://github.com/ai-forever/Kandinsky-2/issues/53
                with gr.Row():
                    infer_size = gr.Checkbox(
                        True,
                        label="Infer image size from input image",
                        elem_classes=["inline-flex"],
                    )
                    width = gr.Slider(
                        shared.ui_params("image_width_min"),
                        shared.ui_params("image_width_max"),
                        shared.ui_params("image_width_default"),
                        step=shared.ui_params("image_width_step"),
                        label="Width",
                        interactive=False,
                    )
                    height = gr.Slider(
                        shared.ui_params("image_height_min"),
                        shared.ui_params("image_height_max"),
                        shared.ui_params("image_height_default"),
                        step=shared.ui_params("image_height_step"),
                        label="Height",
                        interactive=False,
                    )
                with gr.Row():
                    (
                        sampler_20,
                        sampler_21_native,
                        sampler_diffusers,
                    ) = samplers_controls()

                    seed = gr.Number(-1, label="Seed", precision=0)
                with gr.Row() as prior_block:
                    prior_scale = gr.Slider(
                        1,
                        100,
                        4,
                        step=1,
                        label="Prior scale",
                        elem_classes=["inline-flex"],
                    )
                    prior_steps = gr.Slider(
                        1,
                        100,
                        5,
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
            generate_inpaint = gr.Button("Generate", variant="primary")
            inpaint_output = gr.Gallery(
                label="Generated Images",
                columns=2,
                preview=True,
                elem_classes="inpaint-output",
            )

            inpaint_output.select(
                fn=None,
                _js=f"() => kubin.UI.setImageIndex('inpaint-output')",
                show_progress=False,
                outputs=gr.State(None),
            )

            shared.create_base_send_targets(inpaint_output, "inpaint-output", tabs)
            shared.create_ext_send_targets(inpaint_output, "inpaint-output", tabs)

            infer_size.change(
                fn=lambda x: [
                    gr.update(interactive=not x),
                    gr.update(interactive=not x),
                ],
                inputs=[infer_size],
                outputs=[width, height],
            )

            def generate(
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
                sampler = shared.select_sampler(
                    sampler_20, sampler_21_native, sampler_diffusers
                )

                params = {
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
                    "prior_cf_scale": prior_cf_scale,
                    "prior_steps": prior_steps,
                    "negative_prior_prompt": negative_prior_prompt,
                    "input_seed": input_seed,
                    "infer_size": infer_size,
                }

                params = augmentations["exec"](params, injections)
                return generate_fn(params)

        click_and_disable(
            element=generate_inpaint,
            fn=generate,
            inputs=[
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
        )

        batch_size.elem_classes = (
            negative_prompt.elem_classes
        ) = prior_block.elem_classes = ["unsupported_20"]
        inpaint_params.elem_classes = ["block-params inpaint_params"]
        inpaint_advanced_params.elem_classes = [
            "block-advanced-params inpaint_advanced_params"
        ]
    return inpaint_block
