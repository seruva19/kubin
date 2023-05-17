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
    return [gr.update(visible=not exist), gr.update(visible=exist), gr.update(visible=exist, value=choice, choices=[folder for folder in existing_folders])]
    
  def refresh(folder, sort_by, order_by):
    return [] if folder is None else view_folder(folder, sort_by, order_by)
    # TODO: fix strange bug where first refresh after 'outputs' folder is removed causes error in gallery output
  
  def view_folder(folder, sort_by, order_by):
    image_files = [entry.path for entry in os.scandir(f'{image_root}/{folder}') if entry.is_file() and entry.name.endswith(('png'))]
    if sort_by == "date":
      image_files = sorted(image_files, key=lambda f: os.path.getctime(f), reverse=order_by == "descending")
    elif sort_by == "name":
      image_files = sorted(image_files, key=lambda f: str(os.path.splitext(os.path.basename(f))[0]).lower(), reverse=order_by == "descending")

    return image_files
  
  def folder_contents_gallery_select(evt: gr.SelectData):
    return evt.index

  def image_browser_ui(ui_shared, ui_tabs):
    folders = get_folders() 
    selected_folder_contents_index = gr.State(None) # type: ignore

    with gr.Row() as image_browser_block:
      with gr.Column(scale=2):
        no_folders_message = gr.HTML('No image folders found', visible=len(folders) == 0)
        with gr.Box(visible=len(folders) > 0) as image_sources:
          image_folders = gr.Radio([folder for folder in folders], label='Folder', interactive=True)
          image_sort = gr.Radio(["date", "name"], value="date", label='Sort images by', interactive=True)
          image_order = gr.Radio(["ascending", "descending"], value="descending", label='Sort order', interactive=True)
        refresh_btn = gr.Button('Refresh', variant='secondary')
          
      with gr.Column(scale=5):
        folder_contents = gr.Gallery(label='Images in folder').style(preview=False, grid=5)
        folder_contents.select(fn=folder_contents_gallery_select, outputs=[selected_folder_contents_index], show_progress=False)
        
        ui_shared.create_base_send_targets(folder_contents, selected_folder_contents_index, ui_tabs)
        ui_shared.create_ext_send_targets(folder_contents, selected_folder_contents_index, ui_tabs) 
        image_folders.change(fn=view_folder, inputs=[image_folders, image_sort, image_order], outputs=folder_contents)
        image_sort.change(fn=view_folder, inputs=[image_folders, image_sort, image_order], outputs=folder_contents)
        image_order.change(fn=view_folder, inputs=[image_folders, image_sort, image_order], outputs=folder_contents)

        refresh_btn.click(fn=check_folders, inputs=[image_folders], outputs=[no_folders_message, image_sources, image_folders],  # type: ignore
        queue=False).then( 
          fn=refresh, inputs=[image_folders, image_sort, image_order], outputs=[folder_contents]
        )

    return image_browser_block
  
  return {
    'title': 'Image Browser',
    'tab_ui': lambda ui_s, ts: image_browser_ui(ui_s, ts)
  }