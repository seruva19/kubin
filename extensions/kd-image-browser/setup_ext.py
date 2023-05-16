import gradio as gr
import os
from PIL import Image

# TODO: add paging
def setup(kubin):
  image_root = kubin.args.output_dir
  
  def get_folders():
    return [entry.name for entry in os.scandir(image_root) if entry.is_dir()] if os.path.exists(image_root) else []
  
  def check_folders(folder):
    existing_folders = get_folders()
    exist = len(existing_folders) > 0
    choice = None if not exist else folder
    return [gr.update(visible=not exist), gr.update(visible=exist, value=choice, choices=[folder for folder in existing_folders])]
    
  def refresh(folder):
    return [] if folder is None else view_folder(folder)
    # TODO: fix strange bug where first refresh after 'outputs' folder is removed causes error in gallery output
  
  def view_folder(folder):
    image_files = [entry.path for entry in os.scandir(f'{image_root}/{folder}') if entry.is_file() and entry.name.endswith(('png'))]
    image_files.sort(key=lambda f: os.path.getmtime(f), reverse=True)

    return image_files
  
  def folder_contents_gallery_select(evt: gr.SelectData):
    return evt.index

  def image_browser_ui(ui_shared, ui_tabs):
    folders = get_folders() 
    selected_folder_contents_index = gr.State(None) # type: ignore
    
    with gr.Row() as image_browser_block:
      with gr.Column(scale=2):
        no_folders_message = gr.HTML('No image folders found', visible=len(folders) == 0)
        image_folders = gr.Radio([folder for folder in folders], label='Folders', interactive=True, visible=len(folders) > 0)
        refresh_btn = gr.Button('Refresh', variant='secondary')

      with gr.Column(scale=5):
        folder_contents = gr.Gallery(label='Images in folder').style(preview=False, grid=5)
        folder_contents.select(fn=folder_contents_gallery_select, outputs=[selected_folder_contents_index], show_progress=False)
        
        ui_shared.create_base_send_targets(folder_contents, selected_folder_contents_index, ui_tabs)
        ui_shared.create_ext_send_targets(folder_contents, selected_folder_contents_index, ui_tabs) 
        image_folders.change(fn=view_folder, inputs=image_folders, outputs=folder_contents)

        refresh_btn.click(fn=check_folders, inputs=[image_folders], outputs=[no_folders_message, image_folders], 
        queue=False).then( 
          fn=refresh, inputs=[image_folders], outputs=[folder_contents]
        )

    return image_browser_block
  
  return {
    'title': 'Image Browser',
    'tab_ui': lambda ui_s, ts: image_browser_ui(ui_s, ts)
  }