import gc
import secrets
import torch
from kandinsky2 import get_kandinsky2

class Model:
  def __init__(self, device, task_type, cache_dir, model_version, use_flash_attention, output_dir):
    self.device = device
    self.model_version = model_version  
    self.cache_dir = cache_dir  
    self.task_type = task_type  
    self.use_flash_attention = use_flash_attention  
    self.output_dir = output_dir  
    self.kandinsky = None

  def prepare(self, task):
    print(f'preparing model for {task}')
    ready = True
    
    if task == 'img2img':
      task = 'text2img' # https://github.com/ai-forever/Kandinsky-2/issues/22
    
    if task == 'fuse':
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
    self.kandinsky = None

    if self.device == 'cuda':
      with torch.cuda.device('cuda'):
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    gc.collect()

  def withSeed(self, seed):
    seed = secrets.randbelow(99999999999) if seed == -1 else seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)