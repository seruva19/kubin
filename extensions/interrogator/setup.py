import gradio as gr
from PIL import Image
from clip_interrogator import Config, Interrogator
from blip.models.med import BertLMHeadModel
import open_clip
import torch

def patched_prepare_inputs_for_generation(self, input_ids, past=None, attention_mask=None,encoder_hidden_states=None, encoder_attention_mask=None, **model_kwargs):
  input_shape = input_ids.shape

  if attention_mask is None:
    attention_mask = input_ids.new_ones(input_shape)

  if past is not None:
    input_ids = input_ids[:, -1:]

  return {
    "input_ids": input_ids, 
    "attention_mask": attention_mask, 
    "past_key_values": past,
    "encoder_hidden_states": encoder_hidden_states,
    "encoder_attention_mask": encoder_attention_mask,
    "is_decoder": True,
  }


# monkey patching to prevent https://github.com/huggingface/transformers/issues/19290
# and force download of CLIP/BLIP models into app models folder
def use_patch(kubin):
  old_method = BertLMHeadModel.prepare_inputs_for_generation
  old_torch_dir = torch.hub.get_dir()

  BertLMHeadModel.prepare_inputs_for_generation = patched_prepare_inputs_for_generation
  torch.hub.set_dir(kubin.args.cache_dir)
  return old_method, old_torch_dir

def cancel_patch(patch):
  BertLMHeadModel.prepare_inputs_for_generation = patch[0]
  torch.hub.set_dir(patch[1])

def setup(kubin):
  ci = None
  ci_config = None

  def get_interrogator(clip_model, blip_type, cache_path, chunk_size):
    nonlocal ci 
    nonlocal ci_config

    if ci is None or [clip_model, blip_type] != ci_config:
      ci_config = [clip_model, blip_type]
      ci = Interrogator(Config(clip_model_name=clip_model, blip_model_type=blip_type, clip_model_path=cache_path, cache_path=cache_path, download_cache=True, chunk_size=chunk_size))
    
    return ci

  def interrogate(image, mode, clip_model, blip_type, chunk_size):
    patch = use_patch(kubin)

    image = image.convert('RGB') 
    interrogated_text = ''

    interrogator = get_interrogator(clip_model=clip_model, blip_type=blip_type, cache_path=f'{kubin.args.cache_dir}/clip_cache', chunk_size=chunk_size)
    if mode == 'best':
      interrogated_text = interrogator.interrogate(image)
    elif mode == 'classic':
      interrogated_text = interrogator.interrogate_classic(image)
    elif mode == 'fast':
      interrogated_text = interrogator.interrogate_fast(image)
    elif mode == 'negative':
      interrogated_text = interrogator.interrogate_negative(image)

    cancel_patch(patch)
    return interrogated_text

  def interrogator_ui(ui_shared, ui_tabs):
    with gr.Row() as interrogator_block:
      with gr.Column(scale=1):
        with gr.Row():
          source_image = gr.Image(type='pil', label='Input image')

        with gr.Row():
          clip_model = gr.Dropdown(choices=['ViT-L-14/openai', 'ViT-H-14/laion2b_s32b_b79k'], value='ViT-L-14/openai', label='CLIP model')
        with gr.Row():
          mode = gr.Radio(['best', 'classic', 'fast', 'negative'], value='fast', label='Mode')
        with gr.Row():
          blip_model_type = gr.Radio(['base', 'large'], value='large', label='BLIP model type')
        with gr.Row():
          chunk_size = gr.Slider(512, 2048, 2048, step=512, label='Chunk size')
 
      with gr.Column(scale=1):
        interrogate_btn = gr.Button('Interrogate', variant='primary')
        target_text = gr.Textbox(lines=5, label='Interrogated text').style(show_copy_button=True)
        interrogate_btn.click(fn=interrogate, inputs=[source_image, mode, clip_model, blip_model_type, chunk_size], outputs=[target_text])

    return interrogator_block
  
  return {
    'type': 'standalone', 
    'title': 'Interrogator',
    'ui_fn': lambda ui_s, ts: interrogator_ui(ui_s, ts)
  }
