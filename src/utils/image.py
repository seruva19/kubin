import numpy
import requests
from gradio.processing_utils import decode_base64_to_image
import base64
from PIL import Image
from io import BytesIO
import cv2

def image_path_to_pil(image_url):
  response = requests.get(image_url)
  pil_img = Image.open(BytesIO(response.content))

  return pil_img

def resize_pil_img(pil_img, size):
  numpy_image = numpy.array(pil_img)  
  cv_img = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)

  resized_cv_img = cv2.resize(cv_img, size)
  color_converted_cv = cv2.cvtColor(resized_cv_img, cv2.COLOR_BGR2RGB)
  pil_img = Image.fromarray(color_converted_cv)
  
  return pil_img