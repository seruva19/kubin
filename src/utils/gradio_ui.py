
from utils.image import imagePathToPil
import gradio as gr

def send_gallery_image_to_another_tab(gallery, gallery_selected_index, tab_index):
  image_url = gallery[gallery_selected_index]['data']
  img = imagePathToPil(image_url) #for some reason just passing url does not work
  
  return [gr.Tabs.update(selected=tab_index), gr.update(value=img)]
