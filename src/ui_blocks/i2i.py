import gradio as gr
from ui_blocks.shared.ui_shared import SharedUI
from shared import params

def i2i_gallery_select(evt: gr.SelectData):
  return [evt.index, f'Selected image index: {evt.index}']

def i2i_ui(generate_fn, shared: SharedUI, tabs):
  selected_i2i_image_index = gr.State(None) # type: ignore
  augmentations = shared.create_ext_augment_blocks('i2i')

  with gr.Box() as i2i_block:
    with gr.Column():
      with gr.Row():
        shared.input_i2i_image.render()
      with gr.Row():
        prompt = gr.Textbox('', placeholder='', label='Prompt')
      with gr.Accordion("Advanced Image Settings", open=False):
        with gr.Row():
          strength = gr.Slider(0.1, 1, 0.7, step=0.1, label='Strength')

        with gr.Row():
          steps = gr.Slider(20, 200, 100, step=1, label='Steps')
          guidance_scale = gr.Slider(1, 30, 7, step=0.5, label='Guidance scale')

        with gr.Row():
          batch_count = gr.Slider(1, 16, 4, step=1, label='Batch count')
          batch_size = gr.Slider(1, 16, 1, step=1, label='Batch size')

        with gr.Row():
          width = gr.Slider(params.image_width_min, params.image_width_max, 768, step=params.image_width_step, label='Width')
          height = gr.Slider(params.image_height_min, params.image_height_max, 768, step=params.image_height_step, label='Height')

        with gr.Row():
          sampler = gr.Radio(['ddim_sampler', 'p_sampler', 'plms_sampler'], value='ddim_sampler', label='Sampler')
          seed = gr.Number(-1, label='Seed', precision=0)

        with gr.Row():
          prior_scale = gr.Slider(1, 100, 4, step=0.5, label='Prior Config scale')
          prior_steps = gr.Slider(1, 100, 5, step=0.5, label='Prior steps')

        augmentations['ui']()

    with gr.Box():
      with gr.Row():
        generate_i2i = gr.Button('Generate', variant='primary')

      i2i_output = gr.Gallery(label='Generated Images').style(grid=2, preview=True)
      selected_image_info = gr.HTML(value='', elem_classes=['block-info'])
      i2i_output.select(fn=i2i_gallery_select, outputs=[selected_i2i_image_index, selected_image_info], show_progress=False)

      with gr.Row():
        with gr.Accordion("Send image(s) to : Image 2 Image, Mix Images, Inpaint, Upscale", open=False):
          shared.create_base_send_targets(i2i_output, selected_i2i_image_index, tabs)
          shared.create_ext_send_targets(i2i_output, selected_i2i_image_index, tabs)
      
      def generate(image, prompt, strength, steps, batch_count, batch_size, guidance_scale, width, height, sampler, prior_scale, prior_steps, seed, *injections):
        params = {
          'init_image': image,
          'prompt': prompt,
          'strength': strength,
          'num_steps': steps,
          'batch_count': batch_count,
          'batch_size': batch_size,
          'guidance_scale': guidance_scale,
          'w': width,
          'h': height,
          'sampler': sampler,
          'prior_cf_scale': prior_scale,
          'prior_steps': prior_steps,
          'input_seed': seed
        }

        params = augmentations['exec'](params, *injections)
        return generate_fn(params)
      
      generate_i2i.click(generate,
        inputs=[
          shared.input_i2i_image, prompt, strength, steps, batch_count, batch_size, guidance_scale, width, height, sampler, prior_scale, prior_steps, seed
         ] + augmentations['injections'],
        outputs=i2i_output
      )

  return i2i_block
