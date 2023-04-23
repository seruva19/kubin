import gradio as gr

class SharedUI():
  def __init__(self):
    self.input_i2i_image = gr.Image(type='pil') 
    self.input_mix_image_1 = gr.Image(type='pil') 
    self.input_mix_image_2 = gr.Image(type='pil') 
    self.input_inpaint_image = gr.ImageMask(type='pil')