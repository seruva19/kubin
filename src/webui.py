import gradio as gr
from env import Kubin
from ui_blocks.extensions import create_extensions_info, extensions_ui
from ui_blocks.i2i import i2i_ui
from ui_blocks.inpaint import inpaint_ui
from ui_blocks.mix import mix_ui
from ui_blocks.outpaint import outpaint_ui
from ui_blocks.settings import settings_ui
from ui_blocks.shared.ui_shared import SharedUI
from ui_blocks.t2i import t2i_ui
from shared import client

def gradio_ui(kubin: Kubin):
  ext_standalone = kubin.ext_registry.standalone() 

  ext_start_tab_index = 5 
  ext_target_images = create_ext_targets(ext_standalone, ext_start_tab_index) 
  ext_client_folders, ext_client_resources = kubin.ext_registry.locate_resources()

  ui_shared = SharedUI(kubin, ext_target_images, kubin.ext_registry.injectable())

  with gr.Blocks(title='Kubin: Web-GUI for Kandinsky 2.1', theme=ui_shared.select_theme(kubin.options.theme), css=client.css_styles) as ui:
    ui.load(fn=None, _js=client.js_loader(ext_client_resources))
    
    with gr.Tabs() as ui_tabs:
      with gr.TabItem('Text To Image', id=0):
        t2i_ui(generate_fn=lambda params: kubin.model.t2i(params), shared=ui_shared, tabs=ui_tabs)
      
      with gr.TabItem('Image To Image', id=1):
        i2i_ui(generate_fn=lambda params: kubin.model.i2i(params), shared=ui_shared, tabs=ui_tabs)
       
      with gr.TabItem('Mix Images', id=2):
        mix_ui(generate_fn=lambda params: kubin.model.mix(params), shared=ui_shared, tabs=ui_tabs)
       
      with gr.TabItem('Inpaint Image', id=3):
        inpaint_ui(generate_fn=lambda params: kubin.model.inpaint(params), shared=ui_shared, tabs=ui_tabs)
      
      with gr.TabItem('Outpaint Image', id=4):
        outpaint_ui(generate_fn=lambda params: kubin.model.outpaint(params), shared=ui_shared, tabs=ui_tabs)

      create_ext_tabs(ext_standalone, ext_start_tab_index, ui_shared, ui_tabs)

      next_id = len(ext_standalone) + ext_start_tab_index
      with gr.TabItem('Extensions', id=next_id+1):
        extensions_ui(kubin, create_extensions_info(kubin))

      with gr.TabItem('Settings', id=next_id+2):
        settings_ui(kubin)

  return ui, ext_client_folders

def create_ext_targets(exts, ext_start_tab_index):
  ext_targets = []
  for tab_index, ext in enumerate(exts):
    target = ext.get('send_target', None)
    if target is not None:
      ext_targets.append((ext['title'], target, ext_start_tab_index + tab_index))

  return ext_targets

def create_ext_tabs(exts, ext_start_tab_index, ui_shared, tabs):
  ext_ui = []
  for tab_index, ext in enumerate(exts):
    title = ext.get('tab_title', ext['title'])
    with gr.TabItem(title, id = ext_start_tab_index + tab_index):  
      ext_ui.append(ext['tab_ui'](ui_shared, tabs))