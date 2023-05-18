import gc
import itertools
import secrets
from PIL import Image, ImageOps, ImageChops
import cv2
import numpy as np
import torch
import torch.backends

from kandinsky2 import get_kandinsky2, Kandinsky2, Kandinsky2_1

from utils.file_system import save_output

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
    assert task in ['text2img', 'img2img', 'mix', 'inpainting', 'outpainting']

    ready = True
    
    if task == 'img2img':
      task = 'text2img' # https://github.com/ai-forever/Kandinsky-2/issues/22
    
    if task == 'mix':
      task = 'text2img' 
    
    if task == 'outpainting':
      task = 'inpainting' 

    if self.task_type != task:
      self.task_type = task
      ready = False

    if self.kandinsky is None:
      ready = False

    if not ready:
      self.flush()

      self.kandinsky = get_kandinsky2(
        self.device,
        use_auth_token=None,
        task_type=self.task_type,
        cache_dir=self.cache_dir,
        model_version=self.model_version,
        use_flash_attention=self.use_flash_attention
      )
      
      self.kandinsky.model.to(self.device)
      self.kandinsky.prior.to(self.device) # type: ignore

    return self
  
  def flush(self): 
    print(f'releasing vram')
    
    if (self.kandinsky is not None):
      self.kandinsky.model.to('cpu')
      self.kandinsky.prior.to('cpu') # type: ignore
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
    
  def t2i(self, params):
    seed = self.prepare('text2img').withSeed(params['input_seed'])
    assert self.kandinsky is not None
    
    images = []
    for _ in itertools.repeat(None, params['batch_count']):
      current_batch = self.kandinsky.generate_text2img(
        prompt=params['prompt'],
        num_steps=params['num_steps'],
        batch_size=params['batch_size'], 
        guidance_scale=params['guidance_scale'],
        # progress=True,
        # dynamic_threshold_v=99.5
        # denoised_type='dynamic_threshold',
        h=params['h'],
        w=params['w'],
        sampler=params['sampler'], 
        # ddim_eta=0.05,
        prior_cf_scale=params['prior_cf_scale'], # type: ignore
        prior_steps=str(params['prior_steps']), # type: ignore
        negative_prior_prompt=params['negative_prior_prompt'], # type: ignore
        negative_decoder_prompt=params['negative_decoder_prompt'] # type: ignore
      )

      saved_batch = save_output(self.output_dir, 'text2img', current_batch, seed)
      images = images + saved_batch
    return images
  
  def i2i(self, params):
    seed = self.prepare('img2img').withSeed(params['input_seed'])
    assert self.kandinsky is not None    

    output_size = (params['w'], params['h'])
    image = params['init_image']

    images = []    
    for _ in itertools.repeat(None, params['batch_count']):
      current_batch = self.kandinsky.generate_img2img(
        prompt=params['prompt'],
        pil_img=image,
        strength=params['strength'],
        num_steps=params['num_steps'],
        batch_size=params['batch_size'], # type: ignore
        guidance_scale=params['guidance_scale'],
        h=params['h'], # type: ignore
        w=params['w'], # type: ignore
        sampler=params['sampler'], # type: ignore
        prior_cf_scale=params['prior_cf_scale'], # type: ignore
        prior_steps=str(params['prior_steps']) # type: ignore
      )

      saved_batch = save_output(self.output_dir, 'img2img', current_batch, seed)
      images = images + saved_batch
    return images
    
  def mix(self, params):
    seed = self.prepare('mix').withSeed(params['input_seed'])
    assert self.kandinsky is not None    

    def images_or_texts(images, texts):
      images_texts = []
      for i in range(len(images)):
        images_texts.append(texts[i] if images[i] is None else images[i])

      return images_texts

    images = []
    for _ in itertools.repeat(None, params['batch_count']):
      current_batch = self.kandinsky.mix_images( # type: ignore
        images_texts=images_or_texts([params['image_1'], params['image_2']], [params['text_1'], params['text_2']]),
        weights=[params['weight_1'], params['weight_2']],
        num_steps=params['num_steps'],
        batch_size=params['batch_size'], 
        guidance_scale=params['guidance_scale'], 
        h=params['h'], 
        w=params['w'],
        sampler=params['sampler'], 
        prior_cf_scale=params['prior_cf_scale'], 
        prior_steps=str(params['prior_steps']),
        negative_prior_prompt=params['negative_prior_prompt'],
        negative_decoder_prompt=params['negative_decoder_prompt']
      )

      saved_batch = save_output(self.output_dir, 'mix', current_batch, seed)
      images = images + saved_batch
    return images
  
  def inpaint(self, params):
    seed = self.prepare('inpainting').withSeed(params['input_seed'])
    assert self.kandinsky is not None

    output_size = (params['w'], params['h'])
    image_mask = params['image_mask']
    pil_img = image_mask['image'].resize(output_size, resample=Image.LANCZOS)
    
    mask_img = image_mask['mask'].resize(output_size)
    mask_arr = np.array(mask_img.convert('L')).astype(np.float32) / 255.0
  
    if params['target'] == 'only mask':
      mask_arr = 1.0 - mask_arr

    images = []
    for _ in itertools.repeat(None, params['batch_count']):
      current_batch = self.kandinsky.generate_inpainting(
        prompt=params['prompt'],
        pil_img=pil_img,
        img_mask=mask_arr,
        num_steps=params['num_steps'],
        batch_size=params['batch_size'],  # type: ignore
        guidance_scale=params['guidance_scale'],
        h=params['h'], # type: ignore
        w=params['w'], # type: ignore
        sampler=params['sampler'], 
        prior_cf_scale=params['prior_cf_scale'],  # type: ignore
        prior_steps=str(params['prior_steps']),  # type: ignore
        negative_prior_prompt=params['negative_prior_prompt'], # type: ignore
        negative_decoder_prompt=params['negative_decoder_prompt'] # type: ignore
      )

      saved_batch = save_output(self.output_dir, 'inpainting', current_batch, seed)
      images = images + saved_batch
    return images
  
  def outpaint(self, params):
    seed = self.prepare('outpainting').withSeed(params['input_seed'])
    assert self.kandinsky is not None 

    image = params['image'] 
    image_w, image_h = image.size
    
    offset = params['offset']

    if offset is not None:
      top, right, bottom, left = offset
      inferred_mask_size = tuple(a + b for a, b in zip(image.size, (left + right, top + bottom)))[::-1]
      mask = np.zeros(inferred_mask_size, dtype=np.float32)
      mask[top:image_h+top, left:image_w+left] = 1
      image = ImageOps.expand(image, border=(left, top, right, bottom), fill=0)

    else:
      x1, y1, x2, y2 = image.getbbox()
      mask = np.ones((image_h, image_w), dtype=np.float32)
      mask[0:y1,:] = 0
      mask[:,0:x1] = 0
      mask[y2:image_h,:] = 0
      mask[:,x2:image_w] = 0
    
    infer_size = params['infer_size']
    if infer_size:
      height, width = mask.shape[:2]
    else:
      width=params['w']
      height=params['h']

    images = []
    for _ in itertools.repeat(None, params['batch_count']):
      current_batch = self.kandinsky.generate_inpainting(
        prompt=params['prompt'],
        pil_img=image,
        img_mask=mask,
        num_steps=params['num_steps'],
        batch_size=params['batch_size'],  # type: ignore
        guidance_scale=params['guidance_scale'],
        h=width, # type: ignore
        w=height, # type: ignore
        sampler=params['sampler'], 
        prior_cf_scale=params['prior_cf_scale'],  # type: ignore
        prior_steps=str(params['prior_steps']),  # type: ignore
        negative_prior_prompt=params['negative_prior_prompt'], # type: ignore
        negative_decoder_prompt=params['negative_decoder_prompt'] # type: ignore
      )

      saved_batch = save_output(self.output_dir, 'outpainting', current_batch, seed)
      images = images + saved_batch
    return images  
