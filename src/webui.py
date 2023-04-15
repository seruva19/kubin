import gradio as gr
from env import Kubin
from ui_blocks.extensions import extensions_ui
from ui_blocks.i2i import i2i_ui
from ui_blocks.inpaint import inpaint_ui
from ui_blocks.mix import mix_ui
from ui_blocks.outpaint import outpaint_ui
from ui_blocks.settings import settings_ui
from ui_blocks.t2i import t2i_ui

def gradio_ui(kubin: Kubin):
  input_i2i_image = gr.Image(type='pil') 
  input_mix_image_1 = gr.Image(type='pil') 
  input_mix_image_2 = gr.Image(type='pil') 
  input_inpaint_image = gr.ImageMask(type='pil')

  with gr.Blocks(title='Kubin: Kandinsky 2.1 WebGUI') as ui:
    with gr.Tabs() as ui_tabs:
      with gr.TabItem('Text To Image', id=0):
        t2i_ui(generate_fn=lambda *p: kubin.model.t2i(*p),
          input_i2i_image=input_i2i_image, 
          input_mix_image_1=input_mix_image_1, input_mix_image_2=input_mix_image_2,
          input_inpaint_image=input_inpaint_image,
          tabs=ui_tabs
        )
      
      with gr.TabItem('Image To Image', id=1):
        i2i_ui(generate_fn=lambda *p: kubin.model.i2i(*p),
          input_i2i_image=input_i2i_image,
          input_mix_image_1=input_mix_image_1, input_mix_image_2=input_mix_image_2,
          input_inpaint_image=input_inpaint_image,
          tabs=ui_tabs
        ) 
       
      with gr.TabItem('Mix Images', id=2):
        mix_ui(generate_fn=lambda *p: kubin.model.mix(*p),
          input_mix_image_1=input_mix_image_1, input_mix_image_2=input_mix_image_2,
          input_i2i_image=input_i2i_image, 
          input_inpaint_image=input_inpaint_image,
          tabs=ui_tabs
        )
       
      with gr.TabItem('Inpaint Image', id=3):
        inpaint_ui(generate_fn=lambda *p: kubin.model.inpaint(*p),
          input_i2i_image=input_i2i_image, 
          input_mix_image_1=input_mix_image_1, input_mix_image_2=input_mix_image_2,
          input_inpaint_image=input_inpaint_image,
          tabs=ui_tabs
        )

      # with gr.TabItem('Outpaint Image', id=4):
      #   outpaint_ui()

      # with gr.TabItem('Extensions', id=5):
      #   extensions_ui()

      # with gr.TabItem('Settings', id=6):
      #   settings_ui()

  return ui