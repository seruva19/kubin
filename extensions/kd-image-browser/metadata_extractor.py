import gradio as gr
import os
from PIL import Image
import json

def metadata_to_html(path_to_image):
  image = Image.open(path_to_image)
  metadata = image.text.get('kubin_image_metadata', None) # type: ignore
  html = ''

  if metadata is not None:
    params = json.loads(metadata)
    html += f'<div style="font-size: large;">Image metadata:</div><br />'

    for key in params:
      html += f'<b>{key}</b>: {params[key]}<br />'

  return html