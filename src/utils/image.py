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
