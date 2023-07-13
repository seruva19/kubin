import gradio as gr
from ui_blocks.shared.ui_shared import SharedUI


def mix_gallery_select(evt: gr.SelectData):
    return [evt.index, f"Selected image index: {evt.index}"]


def update(image):
    no_image = image == None
    return gr.update(
        label="Prompt" if no_image else "Prompt (ignored, using image instead)",
        visible=no_image,
        interactive=no_image,
    )


# TODO: add mixing for images > 2
# gradio does not directly support dynamic number of elements https://github.com/gradio-app/gradio/issues/2680
def mix_ui(generate_fn, shared: SharedUI, tabs):
    selected_mix_image_index = gr.State(None)  # type: ignore
    augmentations = shared.create_ext_augment_blocks("mix")

    with gr.Row() as mix_block:
        with gr.Column(scale=2) as mix_params:
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
            with gr.Accordion(
                "Advanced params", open=not shared.ui_params("collapse_advanced_params")
            ) as mix_advanced_params:
                with gr.Row():
                    steps = gr.Slider(1, 200, 100, step=1, label="Steps")
                    guidance_scale = gr.Slider(1, 30, 4, step=1, label="Guidance scale")
                with gr.Row():
                    batch_count = gr.Slider(1, 16, 4, step=1, label="Batch count")
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
                    sampler = gr.Radio(
                        ["ddim_sampler", "p_sampler", "plms_sampler"],
                        value="p_sampler",
                        label="Sampler",
                    )
                    sampler_diffusers = gr.Radio(
                        ["ddim_sampler"], value="ddim_sampler", label="Sampler"
                    )
                    sampler.elem_classes = ["t2i_sampler", "native-control"]
                    sampler_diffusers.elem_classes = [
                        "t2i_sampler",
                        "diffusers-control",
                    ]
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
                        1,
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
            mix_output = gr.Gallery(label="Generated Images").style(
                grid=2, preview=True
            )
            selected_image_info = gr.HTML(value="", elem_classes=["block-info"])
            mix_output.select(
                fn=mix_gallery_select,
                outputs=[selected_mix_image_index, selected_image_info],
                show_progress=False,
            )

            shared.create_base_send_targets(mix_output, selected_mix_image_index, tabs)
            shared.create_ext_send_targets(mix_output, selected_mix_image_index, tabs)

            def generate(
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
                sampler,
                prior_cf_scale,
                prior_steps,
                negative_prior_prompt,
                input_seed,
                *injections,
            ):
                params = {
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
                }

                params = augmentations["exec"](params, injections)
                return generate_fn(params)

        generate_mix.click(
            generate,
            inputs=[
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
                sampler,
                prior_scale,
                prior_steps,
                negative_prior_prompt,
                seed,
            ]
            + augmentations["injections"],
            outputs=mix_output,
        )

        mix_params.elem_classes = ["block-params mix_params"]
        mix_advanced_params.elem_classes = ["block-advanced-params mix_advanced_params"]
    return mix_block
