import gradio as gr
from env import Kubin
from utils.image import image_path_to_pil
import gradio as gr

class SharedUI():
  def __init__(self, kubin: Kubin, extension_targets, augment_exts):
    self.input_i2i_image = gr.Image(type='pil', elem_classes=['i2i_image', 'full-height'])
    self.input_mix_image_1 = gr.Image(type='pil', elem_classes=['mix_1_image', 'full-height'])
    self.input_mix_image_2 = gr.Image(type='pil', elem_classes=['mix_2_image', 'full-height'])
    self.input_inpaint_image = gr.ImageMask(type='pil', elem_classes=['inpaint_image'])
    self.input_outpaint_image = gr.ImageMask(type='pil', tool='editor', elem_classes=['outpaint_image'])

    self.extensions_images_targets = extension_targets
    self.extensions_augment = augment_exts
    
  def select_theme(self, theme):
    themes = {
      'base': lambda: gr.themes.Base(),
      'default': lambda: gr.themes.Default(),
      'glass': lambda: gr.themes.Glass(),
      'monochrome': lambda: gr.themes.Monochrome(),
      'soft':  lambda: gr.themes.Soft()
    }

    return themes.get(theme, themes['default'])()

  def open_another_tab(self, tab_index):
    return gr.Tabs.update(selected=tab_index)

  def send_gallery_image_to_another_tab(self, gallery, gallery_selected_index):
    image_url = gallery[gallery_selected_index]['data']
    img = image_path_to_pil(image_url) #for some reason just passing url does not work
    
    return gr.update(value=img)
  
  def create_base_send_targets(self, output, selected_image_index, tabs):
    send_i2i_btn = gr.Button('Send to Img2img', variant='secondary')
    send_i2i_btn.click(fn=self.open_another_tab, inputs=[gr.State(1)], outputs=tabs, # type: ignore
      queue=False).then(
        self.send_gallery_image_to_another_tab, inputs=[output, selected_image_index], outputs=[self.input_i2i_image] 
      )

    with gr.Row():
      send_mix_1_btn = gr.Button('Send to Mix (1)', variant='secondary')
      send_mix_1_btn.click(fn=self.open_another_tab, inputs=[gr.State(2)], outputs=tabs, # type: ignore
        queue=False).then( 
          self.send_gallery_image_to_another_tab, inputs=[output, selected_image_index], outputs=[self.input_mix_image_1] 
        )

      send_mix_2_btn = gr.Button('Send to Mix (2)', variant='secondary')
      send_mix_2_btn.click(fn=self.open_another_tab, inputs=[gr.State(2)], outputs=tabs, # type: ignore
        queue=False).then( 
          self.send_gallery_image_to_another_tab, inputs=[output, selected_image_index], outputs=[self.input_mix_image_2] 
        )

    send_inpaint_btn = gr.Button('Send to Inpaint', variant='secondary')
    send_inpaint_btn.click(fn=self.open_another_tab, inputs=[gr.State(3)], outputs=tabs, # type: ignore
      queue=False).then( 
        self.send_gallery_image_to_another_tab, inputs=[output, selected_image_index], outputs=[self.input_inpaint_image] 
      )
    
    send_outpaint_btn = gr.Button('Send to Outpaint', variant='secondary')
    send_outpaint_btn.click(fn=self.open_another_tab, inputs=[gr.State(4)], outputs=tabs, # type: ignore
      queue=False).then( 
        self.send_gallery_image_to_another_tab, inputs=[output, selected_image_index], outputs=[self.input_outpaint_image] 
      )
    
  def create_ext_send_targets(self, output, selected_image_index, tabs):
    ext_image_targets = []
    for ext in self.extensions_images_targets:
      send_toext_btn = gr.Button(f'Send to {ext[0]}', variant='secondary')
      send_toext_btn.click(fn=self.open_another_tab, inputs=[gr.State(ext[2])], outputs=tabs, # type: ignore
        queue=False).then( 
          self.send_gallery_image_to_another_tab, inputs=[output, selected_image_index], outputs=[ext[1]] 
        )

      ext_image_targets.append(send_toext_btn)
      
  def create_ext_augment_blocks(self, target):
    ext_blocks = []
    ext_exec = []
    ext_injections = []
    
    def create_block():
      for ext_augment in self.extensions_augment:
        if target in ext_augment['targets']:
          with gr.Row() as row:
            with gr.Accordion(ext_augment['title'], open=ext_augment.get('opened', lambda o: False)(target)):
              ext_info = ext_augment['augment_fn'](target)

          ext_blocks.append(row)
          ext_exec.append(ext_augment['exec_fn'])

          for ext_injection in ext_info[1:]:
            ext_injections.append(ext_injection)

    def augment_params(target, params, *injections):
      for index, ext_fn in enumerate(ext_exec):
        params = ext_fn(target, params, injections[index])
      return params


    return {
      'ui': lambda: create_block(),
      'exec': lambda p, *a: augment_params(target, p, *a),
      'injections': ext_injections
    }
