from utils.image import imagePathToPil
import gradio as gr

def open_another_tab(tab_index):
  return gr.Tabs.update(selected=tab_index)

def send_gallery_image_to_another_tab(gallery, gallery_selected_index):
  image_url = gallery[gallery_selected_index]['data']
  img = imagePathToPil(image_url) #for some reason just passing url does not work
  
  return gr.update(value=img)