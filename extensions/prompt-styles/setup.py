import gradio as gr
from pathlib import Path
import yaml
import uuid

dir = Path(__file__).parent.absolute()

def read_styles():
  with open(f'{dir}/styles.yaml', 'r') as stream:
    data = yaml.safe_load(stream)
    return data['styles']

def get_styles():
  return [{'name': 'none', 'prompt': None, 'negative': None}] + read_styles()

def write_styles(styles):
  with open(f'{dir}/styles.yaml', 'w') as stream:
    data = {'styles': styles}
    yaml.safe_dump(data, stream, default_flow_style=False, indent=2, allow_unicode=True)
    
def setup(kubin):
  targets = ['t2i', 'mix', 'inpaint', 'outpaint']

  def load_styles():
    initial_styles = get_styles()

    return (
      initial_styles,
      initial_styles[0],
      initial_styles[0],
      gr.update(choices=[style['name'] for style in initial_styles], value=initial_styles[0]['name']),
      gr.update(value=''),
      gr.update(value='')
    )

  def append_style(target, params, current_style, default_style):
    if 'prompt' in params:
      params['prompt'] += '' if current_style['name'] == default_style['name'] else f', {current_style["prompt"]}'

    if 'negative_decoder_prompt' in params:
      params['negative_decoder_prompt'] += '' if current_style['name'] == default_style['name'] else f', {current_style["negative"]}'

    return params

  def select_style(target, selected_style_name, available):
    selected_style = next(filter(lambda x: x['name'] == selected_style_name, available))

    selected_modifier = selected_style['prompt']
    selected_negative_modifier = selected_style['negative']

    return (
      '' if selected_modifier is None else f'adds to prompt: {selected_modifier}',
      '' if selected_negative_modifier is None else f'adds to negative prompt: {selected_negative_modifier}',
       gr.update(visible=selected_style['name'] != 'none'),
      selected_style
    )

  def add_style(chosen_style):
    return (
      gr.update(visible=True),
      f'User style {uuid.uuid4()}' if chosen_style is None else chosen_style['name'],
      '' if chosen_style is None else chosen_style['prompt'],
      '' if chosen_style is None else chosen_style['negative'],
      gr.update(visible=False)
    )
    
  def save_style(name, prompt, negative_prompt):
    initial_styles = read_styles()
    exists = False

    for style in initial_styles:
      if style['name'] == name:
        style['prompt'] = prompt
        style['negative'] = negative_prompt
        exists = True
        break 

    if not exists:
      initial_styles = initial_styles + [{'name': name, 'prompt': prompt, 'negative': negative_prompt}]

    write_styles(initial_styles)

    return (
      gr.update(visible=True),
      gr.update(visible=False)
    )
  
  def remove_style(name):
    initial_styles = read_styles()
    found_style = None

    for style in initial_styles:
      if style['name'] == name:
        found_style = style
        break 

    if found_style is not None:
      initial_styles.remove(found_style)

    write_styles(initial_styles)

    return (
      gr.update(visible=True),
      gr.update(visible=False),
    )

  def style_select_ui(target):
    target = gr.State(value=target) # type: ignore

    initial_styles = get_styles()
    available_styles = gr.State(value=initial_styles) # type: ignore
    default_style = gr.State(value=initial_styles[0]) # type: ignore
    current_style = gr.State(value=initial_styles[0]) # type: ignore

    with gr.Column() as style_selector_block:
      style_variant = gr.Dropdown([style['name'] for style in initial_styles], value=initial_styles[0]['name'], show_label=False, interactive=True) 
      style_info = gr.HTML(value='', elem_classes='block-info')
      style_negative_info = gr.HTML(value='', elem_classes='block-info')

      with gr.Row() as style_edit_elements:
        add_style_btn = gr.Button('Add style')
        edit_style_btn = gr.Button('Edit style', visible=False)
        refresh_styles_btn = gr.Button('Reload all styles')
        
      with gr.Column(visible=False) as edit_prompt_elements:
        style_name = gr.Textbox(label='Style name', value='', lines=1, interactive=True) 
        style_prompt = gr.Textbox(label='Style prompt', value='', lines=4, interactive=True)  
        style_negative_prompt = gr.Textbox(label='Style negative prompt', value='', lines=4, interactive=True)  

        with gr.Row():
          save_style_btn = gr.Button('Save style')
          cancel_style_btn = gr.Button('Cancel editing')
          remove_style_btn = gr.Button('Remove style')

      style_variant.change(fn=select_style,
        inputs=[target, style_variant, available_styles],
        outputs=[style_info, style_negative_info, edit_style_btn, current_style],
        show_progress=False
      )
      
      refresh_styles_btn.click(fn=load_styles, inputs=[], outputs=[
        available_styles,
        default_style,
        current_style,
        style_variant,
        style_info,
        style_negative_info
      ]) 

      add_style_btn.click(fn=add_style, inputs=[gr.State(None)], outputs=[ # type: ignore
        edit_prompt_elements,
        style_name,
        style_prompt,
        style_negative_prompt,
        style_edit_elements
      ]) 

      edit_style_btn.click(fn=add_style, inputs=[current_style], outputs=[ # type: ignore
        edit_prompt_elements,
        style_name,
        style_prompt,
        style_negative_prompt,
        style_edit_elements
      ]) 

      save_style_btn.click(fn=save_style, inputs=[style_name, style_prompt, style_negative_prompt], outputs=[ # type: ignore
        style_edit_elements,
        edit_prompt_elements
      ]) 

      cancel_style_btn.click(fn=lambda: [gr.update(visible=True), gr.update(visible=False)], inputs=[], outputs=[ # type: ignore
        style_edit_elements,
        edit_prompt_elements
      ]) 
      
      remove_style_btn.click(fn=remove_style, inputs=[style_name], outputs=[ # type: ignore
        style_edit_elements,
        edit_prompt_elements
      ]) 

    return style_selector_block, current_style, default_style
       
  return {
    'type': 'augment', 
    'title': 'Style',
    'augment_fn': lambda target: style_select_ui(target),
    'exec_fn': lambda target, params, augmentations: append_style(target, params, augmentations[0], augmentations[1]),
    'targets': targets
  } 
