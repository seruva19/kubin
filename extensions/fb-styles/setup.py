import gradio as gr
 
def setup(kubin):
  available_styles = get_styles()
  default_style = list(available_styles.keys())[0]
  targets = ['t2i', 'i2i', 'inpaint']

  def append_style(target, params, style):
    params['prompt'] += '' if style == default_style else f', {style}'
    return params

  def style_select(style, target, current_modifier):
    selected_style = available_styles[style]

    return '' if selected_style == '' else f'added to prompt: {selected_style}', selected_style
    
  def style_selector_ui(target):
    with gr.Column() as style_selector_block:
      current_target = gr.State(value=target) # type: ignore
      modifier = gr.State(default_style) # type: ignore
      
      style_variant = gr.Dropdown([style for style in available_styles], value=default_style, show_label=False, interactive=True)
      style_info = gr.HTML(value='', elem_classes='block-info')
      style_variant.change(fn=style_select, inputs=[style_variant, current_target, modifier], outputs=[style_info, modifier], show_progress=False)
          
    return style_selector_block, modifier
       
  return {
    'type': 'augment', 
    'title': 'Style',
    'ui_fn': lambda target: style_selector_ui(target),
    'exec_fn': lambda target, params, modifier: append_style(target, params, modifier),
    'targets': targets
  } 

# these prompt modifiers were taken from fusionbrain.ai
def get_styles():
  return {
    'none': '',
    'anime': 'in anime style',
    'detailed photo':	'4k, ultra HD, detailed phot',
    'cyberpunk': 'in cyberpunk style, futuristic cyberpunk',
    'kandinsky': 'painted by Vasily Kandinsky, abstractionis',
    'aivazovsky': 'painted by Aivazovsky',
    'malevich': 'Malevich, suprematism, avant-garde art, 20th century, geometric shapes , colorful, Russian avant-garde',
    'picasso': 'Cubist painting by Pablo Picasso, 1934, colorful',
    'goncharova': 'painted by Goncharova, Russian avant-garde, futurism, cubism, suprematism',
    'classicism':	'classicism painting, 17th century, trending on artstation, baroque painting',
    'renaissance': 'painting, renaissance old master royal collection, artstation',
    'oil painting':	'like oil painting',
    'pencil art':	'pencil art, pencil drawing, highly detailed',
    'digital painting':	'high quality, highly detailed, concept art, digital painting, by greg rutkowski trending on artstation',
    'medieval':	'medieval painting, 15th century, trending on artstation',
    'soviet cartoon':	'picture from soviet cartoons',
    '3d render': 'Unreal Engine rendering, 3d render, photorealistic, digital concept art, octane render, 4k HD',
    'cartoon': 'as cartoon, picture from cartoon',
    'studio photo':	'glamorous, emotional ,shot in the photo studio, professional studio lighting, backlit, rim lighting, 8k',
    'portrait photo':	'50mm portrait photography, hard rim lighting photography',
    'mosaic': 'as tile mosaic',
    'icon painting': 'in the style of a wooden christian medieval icon in the church',
    'khokhloma': 'in Russian style, Khokhloma, 16th century, marble, decorative, realistic',
    'new year':	'christmas, winter, x-mas, decorations, new year eve, snowflakes, 4k',
  }