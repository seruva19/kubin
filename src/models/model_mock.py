import base64
from PIL import Image
from io import BytesIO

class Model_Mock:
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
    return [Image.open("sshots/t2i.png"), Image.open("sshots/i2i.png"), Image.open("sshots/mix.png"), Image.open("sshots/inpaint.png")]
    