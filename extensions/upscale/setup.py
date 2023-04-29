import gradio as gr
import torch
from PIL import Image
import numpy as np
from RealESRGAN import RealESRGAN
from transformers import CLIPImageProcessor
from diffusers import StableDiffusionUpscalePipeline

def setup(kubin):
  source_image = gr.Image(type='pil')

  def upscaler_ui(ui_shared, ui_tabs):
    with gr.Row() as upscaler_block:
      with gr.Column(scale=1):
        with gr.Row():
          source_image.render()
          
          with gr.Column() as upscale_selector:
            upscaler = gr.Radio(['Real-ESRGAN', 'SD-X4'], value='Real-ESRGAN', label='Upscaler')

        with gr.Row(visible=True) as esrgan_ui:
          scale = gr.Radio(['2', '4', '8'], value='2', label='Upscale by', interactive=True)

        with gr.Row(visible=False) as sdx4_ui:
          scale = gr.Radio(['4'], value='4', label='Upscale by')
          prompt = gr.Textbox('', label='Prompt', placeholder='')
        
        with gr.Row():
          clear_memory = gr.Checkbox(False, label='Clear VRAM before upscale')

        def upscale_params_ui(upscaler):
          return [gr.Box.update(visible=upscaler == 'SD-X4'), gr.update(visible=upscaler == 'Real-ESRGAN')]

        upscaler.select(upscale_params_ui, [upscaler], [esrgan_ui, sdx4_ui]) # type: ignore
                                    
      with gr.Column(scale=1):
        upscale_btn = gr.Button('Upscale', variant='primary')
        upscale_output = gr.Gallery(label='Upscaled Image').style(preview=True)
        
        ui_shared.create_base_send_targets(upscale_output, gr.State(0), ui_tabs) # type: ignore

      upscale_btn.click(fn=lambda *p: upscale_with(kubin, *p), inputs=[
        upscaler,
        prompt,
        gr.Textbox(value=kubin.args.device, visible=False),
        gr.Textbox(value=kubin.args.cache_dir, visible=False),
        scale,
        gr.Textbox(value=kubin.args.output_dir, visible=False),
        source_image,
        clear_memory
      ], outputs=upscale_output)

    return upscaler_block
  
  return {
    'type': 'standalone', 
    'title': 'Upscale',
    'ui_fn': lambda ui_s, ts: upscaler_ui(ui_s, ts),
    'send_target': source_image
  } 

def upscale_with(kubin, upscaler, prompt, device, cache_dir, scale, output_dir, input_image, clear_before_upscale):
  if clear_before_upscale:
    kubin.model.flush()

  if upscaler == 'Real-ESRGAN':
    upscaled_image = upscale_esrgan(device, cache_dir, input_image, scale)
    upscaled_image_path = kubin.fs_utils.save_output(output_dir, 'upscale', [upscaled_image], 'esrgan')

    return upscaled_image_path

  elif upscaler == 'SD-X4':
    upscaled_image = upscale_sdx4(device, prompt, cache_dir, input_image)
    upscaled_image_path = kubin.fs_utils.save_output(output_dir, 'upscale', [upscaled_image], 'sd-x4')

    return upscaled_image_path

  else:
    print(f'upscale method {upscaler} not implemented')
    return []

def upscale_esrgan(device, cache_dir, input_image, scale):
  esrgan = RealESRGAN(device, scale=int(scale))
  esrgan.load_weights(f'{cache_dir}/esrgan/RealESRGAN_x{scale}.pth', download=True)

  image = input_image.convert('RGB')
  upscaled_image = esrgan.predict(image)

  return upscaled_image  
  
def upscale_sdx4(device, prompt, cache_dir, input_image):
  model_id = "stabilityai/stable-diffusion-x4-upscaler"
  pipeline = StableDiffusionUpscalePipeline.from_pretrained(model_id, torch_dtype=torch.float16, cache_dir=cache_dir)
  pipeline = pipeline.to(device) # type: ignore

  upscaled_image = pipeline(prompt=prompt, image=input_image).images[0] # type: ignore
  return upscaled_image