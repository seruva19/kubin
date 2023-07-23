import gradio as gr
import os
from omegaconf import OmegaConf
import pandas as pd
from PIL import Image
import io
import os
import requests
import re
import uuid
import tempfile

show_text_tips = False


def text_tip(text):
    return text if show_text_tips else None


def load_config_from_path(path):
    config = OmegaConf.load(path)
    return config


def save_config_to_path(config, path):
    OmegaConf.save(config, path, resolve=True)


def train_tools_ui(kubin, tabs):
    with gr.Row() as train_tools_block:
        with gr.Accordion("Conversion", open=True) as conversion_tools:
            with gr.Row():
                conversion_from = gr.Dropdown(
                    choices=["parquet"],
                    type="value",
                    value="parquet",
                    label="Source format",
                )
                conversion_to = gr.Dropdown(
                    choices=["folder with images"],
                    type="value",
                    value="folder with images",
                    label="Target format",
                )

            with gr.Column() as conversion_sources:
                with gr.Row():
                    source_path = gr.Text("", label="Source path or URL")
                    target_path = gr.Text("", label="Target path")
                    caption_name = gr.Text("text", label="Caption field name")
                    imagedata_name = gr.Text(
                        "image.bytes", label="Image data field name"
                    )
                with gr.Row():
                    convert_btn = gr.Button("Convert", scale=0)
                    peek_btn = gr.Button("Peek", scale=0)

            conversion_info = gr.HTML("", elem_id="training-tools-conversion-info")

            kubin.ui_utils.click_and_disable(
                convert_btn,
                convert,
                [
                    conversion_from,
                    conversion_to,
                    source_path,
                    target_path,
                    caption_name,
                    imagedata_name,
                ],
                [conversion_info],
            )
            peek_btn.click(
                peek, [conversion_from, conversion_to, source_path], [conversion_info]
            )

    return train_tools_block


def convert(
    convert_from,
    convert_to,
    source_path,
    target_path,
    caption_field,
    imagedata_field,
    progress=gr.Progress(),
):
    if convert_from == "parquet" and convert_to == "folder with images":
        if not os.path.exists(target_path):
            os.makedirs(target_path, exist_ok=True)

        remove_temp_file = False
        if source_path.startswith("http"):
            filename = str(uuid.uuid4())
            temp_dir = tempfile.gettempdir()
            res = requests.get(source_path, allow_redirects=True)
            cd = res.headers.get("content-disposition")
            if cd is not None:
                fname = re.findall(r"filename=(.+?)(?:;|$)", cd)
                if len(fname) > 0:
                    filename = fname[0].strip("\"'")

            filepath = os.path.join(temp_dir, filename)
            open(filepath, "wb").write(res.content)
            source_path = filepath
            remove_temp_file = True

        df_dataset = pd.read_parquet(source_path, engine="fastparquet")

        for i in progress.tqdm(range(len(df_dataset)), unit="samples"):
            image = Image.open(io.BytesIO(df_dataset[imagedata_field].iloc[i]))
            img_path = os.path.join(target_path, f"{i}.png")
            image.save(img_path)

            caption_text = df_dataset[caption_field].iloc[i]
            with open(os.path.join(target_path, f"{i}.txt"), "w") as caption_file:
                caption_file.write(caption_text)

        if remove_temp_file:
            os.remove(source_path)

        return [f"Images with captions saved to {target_path}"]

    else:
        return [f"Conversion of '{convert_from}' to '{convert_to}' is not supported"]


def peek(convert_from, convert_to, source_path):
    if convert_from == "parquet" and convert_to == "folder with images":
        if source_path.startswith("http"):
            return [f"Peeking URL is not supported"]

        df_dataset = pd.read_parquet(source_path, engine="fastparquet")
        num_records = len(df_dataset)
        fields = df_dataset.columns.tolist()
        return [
            f"<span>Number of records: {num_records}</span><br/><span>Fields: {';'.join(fields)}</span>"
        ]

    else:
        return [f"Peeking of '{convert_from}' is not supported"]
