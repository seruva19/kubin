import base64
import os
from PIL import Image, ImageOps
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
    image = params['image'] 
    old_w, old_h = image.size
    
    offset = params['offset']

    if offset is not None:
      top, right, bottom, left = offset
      inferred_mask_size = tuple(a + b for a, b in zip(image.size, (left + right, top + bottom)))[::-1]
      mask = np.zeros(inferred_mask_size, dtype=np.float32)
      mask[top:old_h+top, left:old_w+left] = 1
      image = ImageOps.expand(image, border=(left, top, right, bottom), fill=0)

    else:
      output_size = (params['w'], params['h'])
      image = image.resize(output_size)
      x1, y1, x2, y2 = image.getbbox()
      w_factor = params['w'] / old_w
      h_factor = params['h'] / old_h

      x1 *= w_factor
      x2 *= w_factor
      y1 *= h_factor
      y2 *= h_factor

      mask = np.zeros(image.size, dtype=np.float32)
      mask[int(y1):int(y2), int(x1):int(x2)] = 1
    
    infer_size = params['infer_size']
    if infer_size:
      height, width = mask.shape[:2]
    else:
      width=params['w']
      height=params['h']

    return [image, Image.fromarray(np.uint8(mask * 255))]
  
  def dummyImages(self):
    screenshots_as_dummies = [entry.name for entry in os.scandir('sshots') if entry.is_file()]
    return [Image.open(f'sshots/{screenshot}') for screenshot in screenshots_as_dummies]
    