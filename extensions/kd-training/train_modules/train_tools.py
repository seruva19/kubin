import gradio as gr
import os
import pandas as pd
from PIL import Image
from omegaconf import OmegaConf


def load_config_from_path(path):
    config = OmegaConf.load(path)
    return config


def save_config_to_path(config, path):
    OmegaConf.save(config, path, resolve=True)


def train_tools_ui(kubin):
    with gr.Accordion(open=True, label="Create dataset"):
        with gr.Row():
            with gr.Column():
                image_folder_path = gr.Textbox(
                    "train/images", label="Path to image folder"
                )
                image_extensions = gr.CheckboxGroup(
                    [".jpg", ".jpeg", ".png", ".bmp"],
                    value=[".jpg", ".jpeg", ".png", ".bmp"],
                    label="Image files",
                )
            with gr.Column():
                resize_images = gr.Checkbox(label="Resize images", default=False)
                resized_images_path = gr.Textbox(
                    "train/images_resized", label="Path to folder with resize images"
                )
        with gr.Row():
            with gr.Column():
                caption_extension = gr.Textbox(".txt", label="Caption files extension")
                extract_caption_from_files = gr.Checkbox(
                    label="Extract captions from image filenames", default=False
                )
            output_csv_path = gr.Textbox(
                "train/dataset.csv", label="Path to output dataset file"
            )
        with gr.Row():
            create_df = gr.Button("Create dataset", variant="primary")
            view_df = gr.Button("View dataset")
            clear_existing = gr.Button(
                "Clear existing dataset and resized images"
            ).style(size="sm")
        dataframe_result = gr.HTML("", elem_id="training-tools-df-result")

        with gr.Row():
            images_dataframe = gr.Dataframe(
                max_rows=20,
                overflow_row_behaviour="paginate",
                headers=["image_name", "caption"],
                datatype=["str", "str"],
                visible=False,
                interactive=False,
                elem_id="training-tools-df",
            )

            with gr.Column(visible=False) as image_info:
                df_caption = gr.Textbox(interactive=False, label="Caption")
                df_image = gr.Image(interactive=False, show_label=False)

        def show_image_and_caption(df, evt: gr.SelectData):
            index = evt.index[0]  # type: ignore
            image = df["image_name"][index]
            caption = df["caption"][index]
            return [gr.update(visible=True), image, caption]

        images_dataframe.select(fn=show_image_and_caption, inputs=[images_dataframe], outputs=[image_info, df_image, df_caption])  # type: ignore

        dataframe_error = gr.Checkbox(False, visible=False)

        create_df.click(
            lambda: "Preparing dataset...",
            outputs=[dataframe_result],
            queue=False,
            show_progress=False,
        ).then(
            fn=prepare_dataset,
            inputs=[
                image_folder_path,
                image_extensions,
                caption_extension,
                output_csv_path,
                resize_images,
                resized_images_path,
                extract_caption_from_files,
            ],
            outputs=[dataframe_result, dataframe_error],
            queue=False,
            show_progress=False,  # type: ignore
        ).then(
            fn=None,
            inputs=[dataframe_error],
            outputs=[dataframe_error],
            show_progress=False,
            _js='(e) => !e ? kubin.notify.success("Dataset created") : kubin.notify.error("Error creating dataset")',
        )

        clear_existing.click(
            fn=clear_existing_data,
            inputs=[
                output_csv_path,
                resize_images,
                resized_images_path,
            ],
            outputs=[images_dataframe, image_info, dataframe_result],  # type: ignore
        )

        dataframe_not_exists = gr.Checkbox(False, visible=False)

        view_df.click(
            fn=load_dataframe,
            inputs=[output_csv_path],
            outputs=[images_dataframe, dataframe_result, dataframe_not_exists],
        ).then(
            fn=None,
            inputs=[dataframe_not_exists, dataframe_result],
            outputs=[dataframe_not_exists],
            _js="(err, res) => err && kubin.notify.error(res)",
        )


def load_dataframe(csv_path):
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        return gr.update(value=df, visible=True), "", False

    return gr.update(visible=False), "Dataset does not exist", True


def clear_existing_data(csv_path, resize_enabled, resized_path):
    if os.path.exists(csv_path):
        os.remove(csv_path)
        print(f"{csv_path} removed")
    else:
        print(f"{csv_path} does not exist")

    if os.path.exists(resized_path):
        existing_resized_files = os.listdir(resized_path)

        if len(existing_resized_files) > 0:
            for filename in existing_resized_files:
                file_path = os.path.join(resized_path, filename)
                os.remove(file_path)
            print(f"{len(existing_resized_files)} files removed from {resized_path}")
        else:
            print(f"no files found in {resized_path}")

    return (
        gr.update(visible=False),
        gr.update(visible=False),
        "Existing output data was cleared",
    )


def prepare_dataset(
    image_dir,
    image_extensions,
    caption_extension,
    csv_path,
    resize_enabled,
    resized_path,
    captions_from_filenames,
):
    data = []

    if os.path.exists(csv_path):
        return f"Error: file {csv_path} already exists", True

    if not os.path.exists(image_dir):
        return f"Error: image folder {image_dir} does not exists", True

    for filename in os.listdir(image_dir):
        if filename.endswith(tuple(image_extensions)):
            image_path = os.path.join(image_dir, filename)
            image_filename = os.path.splitext(filename)[0]

            if captions_from_filenames:
                caption_text = image_filename
            else:
                caption_file = image_filename + caption_extension
                caption_path = os.path.join(image_dir, caption_file)

                with open(caption_path) as f:
                    caption_text = f.read()

            data.append([image_path, caption_text])
    print(f"{len(data)} images with captions found and added to dataset")

    processed_data = []
    if resize_enabled:
        print("resizing source images")
        os.makedirs(resized_path, exist_ok=True)

        if len(os.listdir(resized_path)) == 0:
            for image_path, caption_text in data:
                image = Image.open(image_path).convert("RGB")
                resized_image = image.resize((768, 768))
                new_image_path = os.path.join(
                    resized_path, os.path.basename(image_path)
                )
                resized_image.save(new_image_path)
                processed_data.append([new_image_path, caption_text])
        else:
            return f"Error: directory {resized_path} is not empty", True
    else:
        processed_data = data

    df = pd.DataFrame(processed_data, columns=["image_name", "caption"])
    csv_dir = os.path.dirname(csv_path)
    os.makedirs(csv_dir, exist_ok=True)

    df.to_csv(csv_path, index=False)
    print(f"Dataset saved to {csv_path}")

    return f"Dataset with {len(processed_data)} images created", False
