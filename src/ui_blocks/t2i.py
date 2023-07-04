import gradio as gr
from ui_blocks.shared.ui_shared import SharedUI


def t2i_gallery_select(evt: gr.SelectData):
    return [evt.index, f"Selected image index: {evt.index}"]


def t2i_ui(generate_fn, shared: SharedUI, tabs):
    selected_t2i_image_index = gr.State(None)  # type: ignore
    augmentations = shared.create_ext_augment_blocks("t2i")

    with gr.Row() as t2i_block:
        with gr.Column(scale=2) as t2i_params:
            prompt = gr.TextArea("", label="Prompt", placeholder="", lines=2)
            negative_decoder_prompt = gr.TextArea(
                "",
                placeholder="",
                label="Negative decoder prompt",
                lines=2,
            )
            with gr.Accordion(
                "Advanced params", open=not shared.ui_params("collapse_advanced_params")
            ) as t2i_advanced_params:
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
                    sampler.elem_classes = ["t2i_sampler", "native-sampler"]
                    sampler_diffusers.elem_classes = [
                        "t2i_sampler",
                        "diffusers-sampler",
                    ]
                    seed = gr.Number(-1, label="Seed", precision=0)
                with gr.Row() as prior_block:
                    prior_scale = gr.Slider(1, 100, 4, step=1, label="Prior scale")
                    prior_steps = gr.Slider(1, 100, 5, step=1, label="Prior steps")
                    negative_prior_prompt = gr.Textbox(
                        "", label="Negative prior prompt"
                    )

            augmentations["ui"]()

        with gr.Column(scale=1):
            generate_t2i = gr.Button("Generate", variant="primary")
            t2i_output = gr.Gallery(label="Generated Images").style(
                grid=2,
                preview=True,
            )
            selected_image_info = gr.HTML(value="", elem_classes=["block-info"])
            t2i_output.select(
                fn=t2i_gallery_select,
                outputs=[selected_t2i_image_index, selected_image_info],
                show_progress=False,
            )

            shared.create_base_send_targets(t2i_output, selected_t2i_image_index, tabs)
            shared.create_ext_send_targets(t2i_output, selected_t2i_image_index, tabs)

            def generate(
                prompt,
                negative_decoder_prompt,
                num_steps,
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
                    "prompt": prompt,
                    "negative_decoder_prompt": negative_decoder_prompt,
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
                }

                params = augmentations["exec"](params, injections)
                return generate_fn(params)

            generate_t2i.click(
                generate,
                inputs=[
                    prompt,
                    negative_decoder_prompt,
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
                outputs=t2i_output,
            )

        prior_block.elem_classes = (
            batch_size.elem_classes
        ) = negative_decoder_prompt.elem_classes = ["unsupported2_0"]
        t2i_params.elem_classes = ["block-params t2i_params"]
        t2i_advanced_params.elem_classes = ["block-advanced-params t2i_advanced_params"]
    return t2i_block
