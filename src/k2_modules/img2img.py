import itertools
from model import Model
from utils.file_system import save

def i2i(model: Model, init_image, prompt, strength, num_steps, batch_count, batch_size, guidance_scale, w, h, sampler, prior_cf_scale, prior_steps, seed):
  model.prepare('img2img').withSeed(seed)
  
  output_size = (w, h)
  pil_img = init_image.resize(output_size)

  images = []
  for _ in itertools.repeat(None, batch_count):
    images = images + model.kandinsky.generate_img2img(
    prompt=prompt,
    pil_img=pil_img,
    strength=strength,
    num_steps=num_steps,
    batch_size=batch_size, 
    guidance_scale=guidance_scale,
    h=h,
    w=w,
    sampler=sampler, 
    prior_cf_scale=prior_cf_scale,
    prior_steps=str(prior_steps)
  )

  return save(model, images)

  