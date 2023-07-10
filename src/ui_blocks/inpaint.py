import gradio as gr
from ui_blocks.shared.ui_shared import SharedUI


def inpaint_gallery_select(evt: gr.SelectData):
    return [evt.index, f"Selected image index: {evt.index}"]


# TODO: implement region of inpainting
def inpaint_ui(generate_fn, shared: SharedUI, tabs):
    selected_inpaint_image_index = gr.State(None)  # type: ignore
    augmentations = shared.create_ext_augment_blocks("inpaint")

    with gr.Row() as inpaint_block:
        with gr.Column(scale=2) as inpaint_params:
            with gr.Row():
                with gr.Column(scale=1):
                    shared.input_inpaint_image.render()

                with gr.Column():
                    prompt = gr.TextArea("", placeholder="", label="Prompt", lines=2)
                    negative_decoder_prompt = gr.TextArea(
                        "", placeholder="", label="Negative decoder prompt", lines=2
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
                    steps = gr.Slider(1, 200, 100, step=1, label="Steps")
                    guidance_scale = gr.Slider(
                        1, 30, 10, step=1, label="Guidance scale"
                    )
                with gr.Row():
                    batch_count = gr.Slider(1, 16, 4, step=1, label="Batch count")
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
                    sampler = gr.Radio(
                        ["ddim_sampler", "p_sampler", "plms_sampler"],
                        value="p_sampler",
                        label="Sampler",
                    )
                    sampler_diffusers = gr.Radio(
                        ["ddim_sampler"], value="ddim_sampler", label="Sampler"
                    )
                    sampler.elem_classes = ["t2i_sampler", "native-sampler"]
                    sampler_diffusers.elem_classes = [
                        "t2i_sampler",
                        "diffusers-sampler",
                    ]
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
                    negative_prior_prompt = gr.Textbox(
                        "", label="Negative prior prompt"
                    )

            augmentations["ui"]()

        with gr.Column(scale=1):
            generate_inpaint = gr.Button("Generate", variant="primary")
            inpaint_output = gr.Gallery(label="Generated Images").style(
                grid=2, preview=True
            )
            selected_image_info = gr.HTML(value="", elem_classes=["block-info"])
            inpaint_output.select(
                fn=inpaint_gallery_select,
                outputs=[selected_inpaint_image_index, selected_image_info],
                show_progress=False,
            )

            shared.create_base_send_targets(
                inpaint_output, selected_inpaint_image_index, tabs
            )
            shared.create_ext_send_targets(
                inpaint_output, selected_inpaint_image_index, tabs
            )

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
                negative_decoder_prompt,
                inpainting_target,
                inpainting_region,
                steps,
                batch_count,
                batch_size,
                guidance_scale,
                w,
                h,
                sampler,
                prior_cf_scale,
                prior_steps,
                negative_prior_prompt,
                input_seed,
                infer_size,
                *injections,
            ):
                params = {
                    "image_mask": image_mask,
                    "prompt": prompt,
                    "negative_decoder_prompt": negative_decoder_prompt,
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

        generate_inpaint.click(
            generate,
            inputs=[
                shared.input_inpaint_image,
                prompt,
                negative_decoder_prompt,
                inpainting_target,
                inpainting_region,
                steps,
                batch_count,
                batch_size,
                guidance_scale,
                width,
                height,
                sampler,
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
            negative_decoder_prompt.elem_classes
        ) = prior_block.elem_classes = ["unsupported2_0"]
        inpaint_params.elem_classes = ["block-params inpaint_params"]
        inpaint_advanced_params.elem_classes = [
            "block-advanced-params inpaint_advanced_params"
        ]
    return inpaint_block
