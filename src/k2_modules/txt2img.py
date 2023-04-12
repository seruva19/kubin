from model import Model
from utils.file_system import save
import itertools

def t2i(model: Model, prompt, negative_decoder_prompt, num_steps, batch_count, batch_size, guidance_scale, w, h, sampler, prior_cf_scale, prior_steps, seed):
  model.prepare('text2img').withSeed(seed)
  
  images = []
  for _ in itertools.repeat(None, batch_count):
    images = images + model.kandinsky.generate_text2img(
      prompt=prompt,
      num_steps=num_steps,
      batch_size=batch_size, 
      guidance_scale=guidance_scale,
      # progress=True,
      # dynamic_threshold_v=99.5
      # denoised_type='dynamic_threshold',
      h=h,
      w=w,
      sampler=sampler, 
      # ddim_eta=0.05,
      prior_cf_scale=prior_cf_scale,
      prior_steps=str(prior_steps),
      negative_prior_prompt='',
      negative_decoder_prompt=negative_decoder_prompt
    )

  return save(model, images)