import gradio as gr
from ui_blocks.shared.ui_shared import SharedUI
from utils.gradio_ui import click_and_disable
import os
from PIL import Image


def i2i_gallery_select(evt: gr.SelectData):
    return [evt.index, f"Selected image index: {evt.index}"]


def i2i_ui(generate_fn, shared: SharedUI, tabs):
    selected_i2i_image_index = gr.State(None)  # type: ignore
    augmentations = shared.create_ext_augment_blocks("i2i")

    with gr.Row() as i2i_block:
        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.TabItem("Single image"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            shared.input_i2i_image.render()
                        with gr.Column(scale=1):
                            prompt = gr.Textbox("", placeholder="", label="Prompt")

                with gr.TabItem("Batch"):
                    with gr.Row():
                        input_folder = gr.Textbox(
                            label="Folder with input images",
                            info="Folder to read images from",
                        )
                        output_folder = gr.Textbox(
                            label="Folder with output images",
                            info="If empty, the default img2img folder will be used",
                        )
                    batch_prompt = gr.Textbox("", placeholder="", label="Prompt")
                    img_extension = gr.Textbox(
                        ".jpg;.jpeg;.png;.bmp", label="File extension filter"
                    )
                    with gr.Row():
                        generate_batch_i2i = gr.Button(
                            "🖼️ Execute batch processing", variant="secondary"
                        )
                        show_processed_i2i = gr.Button(
                            "🔍 Show images from output folder", variant="secondary"
                        )
                    batch_progress = gr.HTML(label="Batch progress")

            with gr.Accordion("Advanced params", open=True):
                with gr.Row():
                    steps = gr.Slider(1, 200, 100, step=1, label="Steps")
                    guidance_scale = gr.Slider(1, 30, 7, step=1, label="Guidance scale")
                    strength = gr.Slider(
                        0,
                        1,
                        0.7,
                        step=0.05,
                        label="Strength",
                        info="Input image strength",
                    )
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
                        value="ddim_sampler",
                        label="Sampler",
                    )
                    seed = gr.Number(-1, label="Seed", precision=0)
                with gr.Row():
                    prior_scale = gr.Slider(1, 100, 4, step=1, label="Prior scale")
                    prior_steps = gr.Slider(1, 100, 5, step=1, label="Prior steps")

            augmentations["ui"]()

        with gr.Column(scale=1):
            generate_i2i = gr.Button("Generate", variant="primary")
            i2i_output = gr.Gallery(label="Generated Images").style(
                grid=2, preview=True
            )
            selected_image_info = gr.HTML(value="", elem_classes=["block-info"])
            i2i_output.select(
                fn=i2i_gallery_select,
                outputs=[selected_i2i_image_index, selected_image_info],
                show_progress=False,
            )

            shared.create_base_send_targets(i2i_output, selected_i2i_image_index, tabs)
            shared.create_ext_send_targets(i2i_output, selected_i2i_image_index, tabs)

            def generate(
                image,
                prompt,
                strength,
                steps,
                batch_count,
                batch_size,
                guidance_scale,
                width,
                height,
                sampler,
                prior_scale,
                prior_steps,
                seed,
                *injections,
            ):
                params = {
                    "init_image": image,
                    "prompt": prompt,
                    "strength": strength,
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
                return generate_fn(params)

            generate_i2i.click(
                generate,
                inputs=[
                    shared.input_i2i_image,
                    prompt,
                    strength,
                    steps,
                    batch_count,
                    batch_size,
                    guidance_scale,
                    width,
                    height,
                    sampler,
                    prior_scale,
                    prior_steps,
                    seed,
                ]
                + augmentations["injections"],
                outputs=i2i_output,
            )

            def generate_batch(
                input_folder,
                output_folder,
                extensions,
                batch_prompt,
                strength,
                steps,
                batch_count,
                batch_size,
                guidance_scale,
                width,
                height,
                sampler,
                prior_scale,
                prior_steps,
                seed,
                *injections,
            ):
                if not os.path.exists(input_folder):
                    return "Input folder does not exist"

                if input_folder == output_folder:
                    return "Input and output folder cannot be the same"

                i2i_source = []
                for filename in os.listdir(input_folder):
                    if filename.endswith(tuple(extensions.split(";"))):
                        i2i_source.append(filename)

                print(f"found {len(i2i_source)} images for i2i processing")

                for index, imagename in enumerate(i2i_source):
                    imagepath = f"{input_folder}/{imagename}"
                    image = Image.open(imagepath)

                    params = {
                        "init_image": image,
                        "prompt": batch_prompt,
                        "strength": strength,
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

                    print(f"{index+1}/{len(i2i_source)}: processing {imagepath}")
                    _ = generate_fn(params)
                return f"{len(i2i_source)} images successfully processed"

            click_and_disable(
                generate_batch_i2i,
                generate_batch,
                [
                    input_folder,
                    output_folder,
                    img_extension,
                    batch_prompt,
                    strength,
                    steps,
                    batch_count,
                    batch_size,
                    guidance_scale,
                    width,
                    height,
                    sampler,
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
    return i2i_block
