import gradio as gr
from utils.gradio_ui import send_gallery_image_to_another_tab

def i2i_gallery_select(evt: gr.SelectData):
  return evt.index

def i2i_ui(generate_fn, input_i2i_image, input_mix_image_1, input_mix_image_2, input_inpaint_image, tabs):
  selected_image_index = gr.State(-1) # type: ignore

  with gr.Row() as i2i_block:
    with gr.Column(scale=2):
      with gr.Row():
        input_i2i_image.render()
        prompt = gr.Textbox('bunny', label='Prompt')
      with gr.Row():
        steps = gr.Slider(0, 200, 100, step=1, label='Steps')
        guidance_scale = gr.Slider(0, 30, 7, step=1, label='Guidance scale')
        strength = gr.Slider(0, 1, 0.7, step=0.05, label='Strength')
      with gr.Row():
        batch_count = gr.Slider(0, 16, 4, step=1, label='Batch count')
        batch_size = gr.Slider(0, 16, 1, step=1, label='Batch size')
      with gr.Row():
        width = gr.Slider(0, 1024, 512, step=1, label='Width')
        height = gr.Slider(0, 1024, 512, step=1, label='Height')
      with gr.Row():
        sampler = gr.Radio(['ddim_sampler', 'p_sampler', 'plms_sampler'], value='ddim_sampler', label='Sampler')
        seed = gr.Number(-1, label='Seed')
      with gr.Row():
        prior_scale = gr.Slider(0, 100, 4, step=1, label='Prior scale')
        prior_steps = gr.Slider(0, 100, 5, step=1, label='Prior steps')
    with gr.Column(scale=1):
      generate_i2i = gr.Button('Generate', variant='primary')
      i2i_output = gr.Gallery(label='Generated Images').style(grid=2, preview=True)
      i2i_output.select(fn=i2i_gallery_select, outputs=[selected_image_index])

      with gr.Row():
        send_mix_1_btn = gr.Button('Send to mix (1)', variant='secondary')
        send_mix_1_btn.click(fn=send_gallery_image_to_another_tab, inputs=[i2i_output, selected_image_index, gr.State(2)], outputs=[tabs, input_mix_image_1]) # type: ignore

        send_mix_2_btn = gr.Button('Send to mix (2)', variant='secondary')
        send_mix_2_btn.click(fn=send_gallery_image_to_another_tab, inputs=[i2i_output, selected_image_index, gr.State(2)], outputs=[tabs, input_mix_image_2]) # type: ignore

      send_inpaint_btn = gr.Button('Send to inpaint', variant='secondary')
      send_inpaint_btn.click(fn=send_gallery_image_to_another_tab, inputs=[i2i_output, selected_image_index, gr.State(3)], outputs=[tabs, input_inpaint_image]) # type: ignore

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
