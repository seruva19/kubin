import sys
import gradio as gr
import torch
import psutil
import platform
from kandinsky2 import CONFIG_2_1
import pandas as pd
from env import Kubin
from models.model_kd2 import Model_KD2
from models.model_mock import Model_Mock

def update_info():
  torch_version = torch.__version__
  cuda_version = torch.version.cuda
  torch_free, torch_total = torch.cuda.mem_get_info()
  vmem = psutil.virtual_memory()
  ram_total = vmem.total
  ram_available = vmem.available

  torch_total_mb = round(torch_total / (1024 * 1024))
  torch_free_mb = round(torch_free / (1024 * 1024))
  ram_total_mb = round(ram_total / (1024 * 1024))
  ram_available_mb = round(ram_available / (1024 * 1024))

  return (
    f'python: {sys.version}\n'
    f'torch: {torch_version}\n'
    f'CUDA: {cuda_version}\n'
    f'processor: {platform.processor()}\n'
    f'RAM (total): {ram_total_mb}\n'
    f'RAM (free): {ram_available_mb}\n'
    f'VRAM (total): {torch_total_mb}\n'
    f'VRAM (free): {torch_free_mb}'
  )

def flatten_model_config(config):
  normalized = pd.json_normalize(config, sep='.')
  values = normalized.to_dict(orient='records')[0]

  return values

def settings_ui(kubin: Kubin):
  model_config = flatten_model_config(CONFIG_2_1)

  with gr.Column() as settings_block:
    system_info = gr.Textbox(update_info, label='System info', lines=10).style(show_copy_button=True)
    with gr.Row():
      update_btn = gr.Button(value='Update system info')
      update_btn.click(update_info, outputs=system_info)
      unload_model = gr.Button(value='Unload model')
      unload_model.click(lambda: kubin.model.flush())

    with gr.Accordion('Model params', open=False):
      values = []
      for key in model_config:
        value = gr.Textbox(label=str(key), value=model_config[key])
        values.append(value)

  return settings_block
