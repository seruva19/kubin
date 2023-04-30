import os
import uuid
from datetime import datetime

def save_output(output_dir, task_type, images, seed):
  output = []
  for img in images:
    path = f'{output_dir}/{task_type}'
    if not os.path.exists(path): os.makedirs(path)
  
    name = f'{path}/{seed}-{uuid.uuid4()}.png'
    img.save(name, 'PNG')
    output.append(name)

  return output