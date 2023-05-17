import os
import uuid
from datetime import datetime

def save_output(output_dir, task_type, images, seed):
  output = []
  for img in images:
    path = f'{output_dir}/{task_type}'
    if not os.path.exists(path): os.makedirs(path)

    current_datetime = datetime.now()
    format_string = "%Y%m%d%H%M%S"
    formatted_datetime = current_datetime.strftime(format_string)

    # name = f'{path}/{seed}-{uuid.uuid4()}.png'
    filename = f'{path}/{formatted_datetime}.png'
    while os.path.exists(filename):
      unique_id = current_datetime.microsecond
      filename = f'{formatted_datetime}_{unique_id}.jpg'

    img.save(filename, 'PNG')
    output.append(filename)

  return output