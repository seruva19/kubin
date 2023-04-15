import base64
from PIL import Image
from io import BytesIO

class DummyModel:
  def __init__(self):
    print(f'setting dummy model params')

  def prepare(self, task):
    print(f'preparing dummy for {task}')
    return self 
  
  def memGC(self): 
    print(f'dummy memory freed')

  def withSeed(self, seed):
    print('dummy seed generated')

  def t2i(self, *args, **kwargs):
    print('dummy t2i executed')
    return self.dummyImages()
  
  def i2i(self, *args, **kwargs):
    print('dummy i2i executed')
    return self.dummyImages()
  
  def mix(self, *args, **kwargs):
    print('dummy mix executed')
    return self.dummyImages()
  
  def inpaint(self, *args, **kwargs):
    print('dummy inpaint executed')
    return self.dummyImages()
  
  def dummyImages(self):
    white_image = Image.open(BytesIO(base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAIAAAACACAQAAABpN6lAAAAAqklEQVR42u3QMQEAAAgDINc/9IzhIUQg7bwWAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAg4N4C6z7/gQlsSUcAAAAASUVORK5CYII=')))
    black_image = Image.open(BytesIO(base64.b64decode('R0lGODlhAQABAIAAAAUEBAAAACwAAAAAAQABAAACAkQBADs=')))

    return [white_image, black_image, white_image, black_image]
    