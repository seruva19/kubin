import gradio as gr
import os
from PIL import Image

def setup(kubin):
  image_root = kubin.args.output_dir
  
  def get_folders():
    return [entry.name for entry in os.scandir(image_root) if entry.is_dir()]
  
  def view_folder(folder):
    image_files = [entry.path for entry in os.scandir(f'{image_root}/{folder}') if entry.is_file() and entry.name.endswith(('png'))]
    return image_files
  
  def folder_contents_gallery_select(evt: gr.SelectData):
    return evt.index

  def image_browser_ui(ui_shared, ui_tabs):
    selected_folder_contents_index = gr.State(None) # type: ignore

    with gr.Row() as image_browser_block:
      with gr.Column(scale=2):
        image_folder = gr.Radio([folder for folder in get_folders()], label='Folders', interactive=True)
        refresh_btn = gr.Button('Refresh', variant='secondary')

      with gr.Column(scale=5):
        folder_contents = gr.Gallery(label='Images in folder').style(preview=False, grid=5)
        folder_contents.select(fn=folder_contents_gallery_select, outputs=[selected_folder_contents_index], show_progress=False)
        
        ui_shared.create_base_send_targets(folder_contents, selected_folder_contents_index, ui_tabs) # type: ignore
        image_folder.change(fn=view_folder, inputs=image_folder, outputs=folder_contents)
        refresh_btn.click(view_folder, inputs=image_folder, outputs=folder_contents)
        
    return image_browser_block
  
  return {
    'type': 'standalone', 
    'title': 'Image Browser',
    'ui_fn': lambda ui_s, ts: image_browser_ui(ui_s, ts)
  }