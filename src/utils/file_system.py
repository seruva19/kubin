import os
import uuid

from model import Model

def save(model: Model, images):
  output = []
  for img in images:
    path = f'{model.output_dir}/{model.task_type}'
    if not os.path.exists(path): os.makedirs(path)
  
    name = f'{path}/{uuid.uuid4()}.png'
    img.save(name, 'PNG')
    output.append(name)

  return output