import base64
import os
from PIL import Image
from io import BytesIO

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
    return self.dummyImages()
  
  def dummyImages(self):
    screenshots_as_dummies = [entry.name for entry in os.scandir('sshots') if entry.is_file()]
    return [Image.open(f'sshots/{screenshot}') for screenshot in screenshots_as_dummies]
    