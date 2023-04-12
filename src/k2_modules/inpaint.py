import itertools
import numpy as np
from model import Model
from utils.file_system import save

def inpaint(model: Model, image_mask, prompt, negative_decoder_prompt, target, region, num_steps, batch_count, batch_size, guidance_scale, w, h, sampler, prior_cf_scale, prior_steps, seed):
  model.prepare('inpainting').withSeed(seed)

  output_size = (w, h)
  pil_img = image_mask['image'].resize(output_size)

  mask = image_mask['mask'].resize(output_size)
 
  mask =  np.array(mask.convert('L')).astype(np.float32) / 255.0
  if target == 'only mask':
    mask = 1.0 - mask
    
  images = []
  for _ in itertools.repeat(None, batch_count):
    images = images + model.kandinsky.generate_inpainting(
    prompt=prompt,
    pil_img=pil_img,
    img_mask=mask,
    num_steps=num_steps,
    batch_size=batch_size, 
    guidance_scale=guidance_scale,
    h=h,
    w=w,
    sampler=sampler, 
    prior_cf_scale=prior_cf_scale,
    prior_steps=str(prior_steps),
    negative_prior_prompt='',
    negative_decoder_prompt=negative_decoder_prompt
  )

  return save(model, images)