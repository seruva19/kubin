import gradio as gr
from ui_blocks.shared.ui_shared import SharedUI
from utils.gradio_ui import send_gallery_image_to_another_tab, open_another_tab

def mix_gallery_select(evt: gr.SelectData):
  return [evt.index, f'Selected image index: {evt.index}']

def update(image):
  no_image = image == None
  return gr.update(label='Prompt' if no_image else 'Prompt (ignored, using image instead)', interactive=no_image)

# TODO: add mixing for images > 2
def mix_ui(generate_fn, shared: SharedUI, tabs):
  selected_mix_image_index = gr.State(None) # type: ignore

  with gr.Row() as mix_block:
    with gr.Column(scale=2):
      with gr.Row():
        with gr.Column(scale=1):
          shared.input_mix_image_1.render()
          text_1 = gr.Textbox('hare', label='Prompt')
          shared.input_mix_image_1.change(fn=update, inputs=shared.input_mix_image_1, outputs=text_1)
          weight_1 = gr.Slider(0, 1, 0.5, step=0.05, label='Weight')
        with gr.Column(scale=1):
          shared.input_mix_image_2.render()
          text_2 = gr.Textbox('hare', label='Prompt')
          shared.input_mix_image_2.change(fn=update, inputs=shared.input_mix_image_2, outputs=text_2)
          weight_2 = gr.Slider(0, 1, 0.5, step=0.05, label='Weight')
      add_btn = gr.Button('Add another image', interactive=False)
      negative_prompt = gr.Textbox('', label='Negative prompt')
      with gr.Row():
        steps = gr.Slider(0, 200, 100, step=1, label='Steps')
        guidance_scale = gr.Slider(0, 30, 4, step=1, label='Guidance scale')
      with gr.Row():
        batch_count = gr.Slider(0, 16, 4, step=1, label='Batch count')
        batch_size = gr.Slider(0, 16, 1, step=1, label='Batch size')
      with gr.Row():
        width = gr.Slider(0, 1024, 768, step=1, label='Width')
        height = gr.Slider(0, 1024, 768, step=1, label='Height')
      with gr.Row():
        sampler = gr.Radio(['ddim_sampler', 'p_sampler', 'plms_sampler'], value='p_sampler', label='Sampler')
        seed = gr.Number(-1, label='Seed', precision=0)
      with gr.Row():
        prior_scale = gr.Slider(0, 100, 4, step=1, label='Prior scale')
        prior_steps = gr.Slider(0, 100, 5, step=1, label='Prior steps')
        negative_prior_prompt = gr.Textbox('', label='Negative prior prompt')
    with gr.Column(scale=1):
      generate_mix = gr.Button('Generate', variant='primary')
      mix_output = gr.Gallery(label='Generated Images').style(grid=2, preview=True)
      selected_image_info = gr.HTML(value='')
      mix_output.select(fn=mix_gallery_select, outputs=[selected_mix_image_index, selected_image_info])

      send_i2i_btn = gr.Button('Send to img2img', variant='secondary')
      send_i2i_btn.click(fn=open_another_tab, inputs=[gr.State(1)], outputs=tabs, # type: ignore
        queue=False).then( 
          send_gallery_image_to_another_tab, inputs=[mix_output, selected_mix_image_index], outputs=[shared.input_i2i_image] 
        )
      
      with gr.Row():    
        send_mix_1_btn = gr.Button('Send to mix (1)', variant='secondary')
        send_mix_1_btn.click(fn=open_another_tab, inputs=[gr.State(2)], outputs=tabs, # type: ignore
          queue=False).then( 
            send_gallery_image_to_another_tab, inputs=[mix_output, selected_mix_image_index], outputs=[shared.input_mix_image_1]
          )
        
        send_mix_2_btn = gr.Button('Send to mix (2)', variant='secondary')
        send_mix_2_btn.click(fn=open_another_tab, inputs=[gr.State(2)], outputs=tabs, # type: ignore
          queue=False).then( 
            send_gallery_image_to_another_tab, inputs=[mix_output, selected_mix_image_index], outputs=[shared.input_mix_image_2] 
          )
      
      send_inpaint_btn = gr.Button('Send to inpaint', variant='secondary')
      send_inpaint_btn.click(fn=open_another_tab, inputs=[gr.State(3)], outputs=tabs, # type: ignore
        queue=False).then( 
          send_gallery_image_to_another_tab, inputs=[mix_output, selected_mix_image_index], outputs=[shared.input_inpaint_image] 
        )
      
    generate_mix.click(generate_fn, inputs=[
      shared.input_mix_image_1,
      shared.input_mix_image_2,
      text_1,
      text_2,
      weight_1,
      weight_2,
      negative_prompt,
      steps,
      batch_count,
      batch_size,
      guidance_scale,
      width,
      height,
      sampler,
      prior_scale,
      prior_steps,
      negative_prior_prompt,
      seed
    ], outputs=mix_output)

  return mix_block
