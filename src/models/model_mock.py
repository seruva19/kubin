import base64
import os
from PIL import Image
from io import BytesIO
import numpy as np

# intended for testing purposes only
class Model_Mock:
  def __init__(self):
    print(f'setting mock model params')

  def prepare(self, task):
    print(f'preparing mock for {task}')
    return self 
  
  def flush(self): 
    print(f'mock memory freed')

  def withSeed(self, seed):
    print('mock seed generated')

  def t2i(self, params):
    print('mock t2i executed')
    return self.dummyImages()
  
  def i2i(self, params):
    print('mock i2i executed')
    return self.dummyImages()
  
  def mix(self, params):
    print('mock mix executed')
    return self.dummyImages()
  
  def inpaint(self, params):
    print('mock inpaint executed')
    
    output_size = (params['w'], params['h'])
    image_with_mask = params['image_mask']

    image = image_with_mask['image']
    image = image.convert('RGB')
    image = image.resize(output_size, resample=Image.LANCZOS)

    mask = image_with_mask['mask']
    mask = mask.convert('L')
    mask = mask.resize(output_size, resample=Image.LANCZOS)

    return [image, mask]
  
  def outpaint(self, params):
    print('mock outpaint executed')
    return self.dummyImages()
  
  def dummyImages(self):
    screenshots_as_dummies = [entry.name for entry in os.scandir('sshots') if entry.is_file()]
    return [Image.open(f'sshots/{screenshot}') for screenshot in screenshots_as_dummies]
    