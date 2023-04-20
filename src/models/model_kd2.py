import gc
import itertools
import secrets
import numpy as np
import torch
from kandinsky2 import get_kandinsky2, Kandinsky2, Kandinsky2_1

from utils.file_system import saveOutput
from utils.image import resizePilImg

class Model_KD2:
  def __init__(self, device, task_type, cache_dir, model_version, use_flash_attention, output_dir):
    print(f'setting model params')
    self.device = device
    self.model_version = model_version  
    self.cache_dir = cache_dir  
    self.task_type = task_type  
    self.use_flash_attention = use_flash_attention  
    self.output_dir = output_dir  
    self.kandinsky: Kandinsky2 | Kandinsky2_1 | None = None

  def prepare(self, task):
    print(f'preparing model for {task}')
    ready = True
    
    if task == 'img2img':
      task = 'text2img' # https://github.com/ai-forever/Kandinsky-2/issues/22
    
    if task == 'mix':
      task = 'text2img' 

    if self.task_type != task:
      self.task_type = task
      ready = False

    if self.kandinsky is None:
      ready = False

    if not ready:
      self.memGC()

      self.kandinsky = get_kandinsky2(
        self.device,
        use_auth_token=None,
        task_type=self.task_type,
        cache_dir=self.cache_dir,
        model_version=self.model_version,
        use_flash_attention=self.use_flash_attention
      )
      
    return self
  
  def memGC(self): 
    print(f'trying to free memory')
    
    self.kandinsky = None
    gc.collect()

    if self.device == 'cuda':
      with torch.cuda.device('cuda'):
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

  def withSeed(self, seed):
    seed = secrets.randbelow(99999999999) if seed == -1 else seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    print(f'seed generated: {seed}')
    return seed
    
  def t2i(self, prompt, negative_decoder_prompt, num_steps, batch_count, batch_size, guidance_scale, w, h, sampler, prior_cf_scale, prior_steps, negative_prior_prompt, input_seed):
    seed = self.prepare('text2img').withSeed(input_seed)
    assert self.kandinsky is not None
    
    images = []
    for _ in itertools.repeat(None, batch_count):
      current_batch = self.kandinsky.generate_text2img(
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
        prior_cf_scale=prior_cf_scale, # type: ignore
        prior_steps=str(prior_steps), # type: ignore
        negative_prior_prompt=negative_prior_prompt, # type: ignore
        negative_decoder_prompt=negative_decoder_prompt # type: ignore
      )

      saved_batch = saveOutput(self.output_dir, 'text2img', current_batch, seed)
      images = images + saved_batch
    return images
  
  def i2i(self, init_image, prompt, strength, num_steps, batch_count, batch_size, guidance_scale, w, h, sampler, prior_cf_scale, prior_steps, input_seed):
    seed = self.prepare('img2img').withSeed(input_seed)
    assert self.kandinsky is not None    

    output_size = (w, h)
    pil_img = resizePilImg(init_image, output_size)

    images = []    
    for _ in itertools.repeat(None, batch_count):
      current_batch = self.kandinsky.generate_img2img(
        prompt=prompt,
        pil_img=pil_img,
        strength=strength,
        num_steps=num_steps,
        batch_size=batch_size,  # type: ignore
        guidance_scale=guidance_scale,
        h=h,  # type: ignore
        w=w,  # type: ignore
        sampler=sampler, 
        prior_cf_scale=prior_cf_scale,  # type: ignore
        prior_steps=str(prior_steps)  # type: ignore
      )

      saved_batch = saveOutput(self.output_dir, 'img2img', current_batch, seed)
      images = images + saved_batch
    return images
    
  def mix(self, image_1, image_2, text_1, text_2, weight_1, weight_2, negative_decoder_prompt, num_steps, batch_count, batch_size, guidance_scale, w, h, sampler, prior_cf_scale, prior_steps, negative_prior_prompt, input_seed):
    seed = self.prepare('mix').withSeed(input_seed)
    assert self.kandinsky is not None    

    images = []
    for _ in itertools.repeat(None, batch_count):
      current_batch = self.kandinsky.mix_images( # type: ignore
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
        negative_prior_prompt=negative_prior_prompt,
        negative_decoder_prompt=negative_decoder_prompt
      )

      saved_batch = saveOutput(self.output_dir, 'mix', current_batch, seed)
      images = images + saved_batch
    return images
  
  def inpaint(self, image_mask, prompt, negative_decoder_prompt, target, region, num_steps, batch_count, batch_size, guidance_scale, w, h, sampler, prior_cf_scale, prior_steps, negative_prior_prompt, input_seed):
    seed = self.prepare('inpainting').withSeed(input_seed)
    assert self.kandinsky is not None

    output_size = (w, h)
    pil_img = resizePilImg(image_mask['image'], output_size)
    
    mask_img = resizePilImg(image_mask['mask'], output_size)
    mask = np.array(mask_img.convert('L')).astype(np.float32) / 255.0
    
    if target == 'only mask':
      mask = 1.0 - mask
      
    images = []
    for _ in itertools.repeat(None, batch_count):
      current_batch = self.kandinsky.generate_inpainting(
        prompt=prompt,
        pil_img=pil_img,
        img_mask=mask,
        num_steps=num_steps,
        batch_size=batch_size,  # type: ignore
        guidance_scale=guidance_scale,
        h=h, # type: ignore
        w=w, # type: ignore
        sampler=sampler, 
        prior_cf_scale=prior_cf_scale,  # type: ignore
        prior_steps=str(prior_steps),  # type: ignore
        negative_prior_prompt=negative_prior_prompt,  # type: ignore
        negative_decoder_prompt=negative_decoder_prompt  # type: ignore
      )

      saved_batch = saveOutput(self.output_dir, 'inpainting', current_batch, seed)
      images = images + saved_batch
    return images
  
