import itertools
from model import Model
from utils.file_system import save

def fuse(model: Model, image_1, image_2, text_1, text_2, weight_1, weight_2, negative_decoder_prompt, num_steps, batch_count, batch_size, guidance_scale, w, h, sampler, prior_cf_scale, prior_steps, seed):
  model.prepare('fuse').withSeed(seed)

  images = []
  for _ in itertools.repeat(None, batch_count):
    images = images + model.kandinsky.mix_images(
      images_texts=[text_1 if image_1 is None else image_1, text_2 if image_2 is None else image_2],
      weights=[weight_1, weight_2], 
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
  