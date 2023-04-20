import gradio as gr
from utils.gradio_ui import send_gallery_image_to_another_tab, open_another_tab

def i2i_gallery_select(evt: gr.SelectData):
  return [evt.index, f'Selected image index: {evt.index}']

def i2i_ui(generate_fn, input_i2i_image, input_mix_image_1, input_mix_image_2, input_inpaint_image, tabs):
  selected_i2i_image_index = gr.State(None) # type: ignore

  with gr.Row() as i2i_block:
    with gr.Column(scale=2):
      with gr.Row():
        input_i2i_image.render()
        prompt = gr.Textbox('hare', label='Prompt')
      with gr.Row():
        steps = gr.Slider(0, 200, 100, step=1, label='Steps')
        guidance_scale = gr.Slider(0, 30, 7, step=1, label='Guidance scale')
        strength = gr.Slider(0, 1, 0.7, step=0.05, label='Strength')
      with gr.Row():
        batch_count = gr.Slider(0, 16, 4, step=1, label='Batch count')
        batch_size = gr.Slider(0, 16, 1, step=1, label='Batch size')
      with gr.Row():
        width = gr.Slider(0, 1024, 768, step=1, label='Width')
        height = gr.Slider(0, 1024, 768, step=1, label='Height')
      with gr.Row():
        sampler = gr.Radio(['ddim_sampler', 'p_sampler', 'plms_sampler'], value='ddim_sampler', label='Sampler')
        seed = gr.Number(-1, label='Seed', precision=0)
      with gr.Row():
        prior_scale = gr.Slider(0, 100, 4, step=1, label='Prior scale')
        prior_steps = gr.Slider(0, 100, 5, step=1, label='Prior steps')
    with gr.Column(scale=1):
      generate_i2i = gr.Button('Generate', variant='primary')
      i2i_output = gr.Gallery(label='Generated Images').style(grid=2, preview=True)
      selected_image_info = gr.HTML(value='')
      i2i_output.select(fn=i2i_gallery_select, outputs=[selected_i2i_image_index, selected_image_info])

      send_i2i_btn = gr.Button('Send to img2img', variant='secondary')
      send_i2i_btn.click(fn=open_another_tab, inputs=[gr.State(1)], outputs=tabs, # type: ignore
        queue=False).then(
          send_gallery_image_to_another_tab, inputs=[i2i_output, selected_i2i_image_index], outputs=[input_i2i_image] 
        )

      with gr.Row():    
        send_mix_1_btn = gr.Button('Send to mix (1)', variant='secondary')
        send_mix_1_btn.click(fn=open_another_tab, inputs=[gr.State(2)], outputs=tabs, # type: ignore
          queue=False).then( 
            send_gallery_image_to_another_tab, inputs=[i2i_output, selected_i2i_image_index], outputs=[input_mix_image_1] 
          )
        
        send_mix_2_btn = gr.Button('Send to mix (2)', variant='secondary')
        send_mix_2_btn.click(fn=open_another_tab, inputs=[gr.State(2)], outputs=tabs, # type: ignore
          queue=False).then( 
            send_gallery_image_to_another_tab, inputs=[i2i_output, selected_i2i_image_index], outputs=[input_mix_image_2] 
          )
        
      send_inpaint_btn = gr.Button('Send to inpaint', variant='secondary')
      send_inpaint_btn.click(fn=open_another_tab, inputs=[gr.State(3)], outputs=tabs, # type: ignore
        queue=False).then( 
          send_gallery_image_to_another_tab, inputs=[i2i_output, selected_i2i_image_index], outputs=[input_inpaint_image] 
        )
      
    generate_i2i.click(generate_fn, inputs=[
      input_i2i_image,
      prompt,
      strength,
      steps,
      batch_count,
      batch_size,
      guidance_scale,
      width,
      height,
      sampler,
      prior_scale,
      prior_steps,
      seed
    ], outputs=i2i_output)

  return i2i_block
