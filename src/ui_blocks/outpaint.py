from io import BytesIO
import gradio as gr
from ui_blocks.shared.ui_shared import SharedUI
from shared import params

def outpaint_gallery_select(evt: gr.SelectData):
  return [evt.index, f'Selected image index: {evt.index}']

def outpaint_ui(generate_fn, shared: SharedUI, tabs):
  selected_outpaint_image_index = gr.State(None) # type: ignore
  augmentations = shared.create_ext_augment_blocks('outpaint')

  with gr.Row() as outpaint_block:
    with gr.Column(scale=2):
      with gr.Row():
        with gr.Column(scale=1):
          shared.input_outpaint_image.render()

        with gr.Column(scale=1):
          manual_control = gr.Checkbox(False, label='Manual control')
          offset_top = gr.Slider(0, 1024, 0, step=16, label='Top')
          with gr.Row():
            offset_left = gr.Slider(0, 1024, 0, step=16, label='Left')
            offset_right = gr.Slider(0, 1024, 0, step=16, label='Right')
          offset_bottom = gr.Slider(0, 1024, 0, step=16, label='Bottom')

          manual_control.change(
            fn=lambda x: [gr.update(interactive=x), gr.update(interactive=x), gr.update(interactive=x), gr.update(interactive=x)],
            inputs=[manual_control], outputs=[offset_top, offset_left, offset_right, offset_bottom]
          )

      with gr.Column():
        prompt = gr.Textbox('', placeholder='', label='Prompt')
        negative_prompt = gr.Textbox('', placeholder='', label='Negative prompt')
      with gr.Row():
        steps = gr.Slider(0, 200, 100, step=1, label='Steps')
        guidance_scale = gr.Slider(0, 30, 10, step=1, label='Guidance scale')
      with gr.Row():
        batch_count = gr.Slider(0, 16, 4, step=1, label='Batch count')
        batch_size = gr.Slider(0, 16, 1, step=1, label='Batch size')
      with gr.Row():
        width = gr.Slider(params.image_width_min, params.image_width_max, 768, step=params.image_width_step, label='Width')
        height = gr.Slider(params.image_height_min, params.image_height_max, 768, step=params.image_height_step, label='Height')
      with gr.Row():
        sampler = gr.Radio(['ddim_sampler', 'p_sampler', 'plms_sampler'], value='p_sampler', label='Sampler')
        seed = gr.Number(-1, label='Seed', precision=0)
      with gr.Row():
        prior_scale = gr.Slider(0, 100, 4, step=1, label='Prior scale')
        prior_steps = gr.Slider(0, 100, 5, step=1, label='Prior steps')
        negative_prior_prompt = gr.Textbox('', label='Negative prior prompt')

      augmentations['ui']()

    with gr.Column(scale=1):
      generate_outpaint= gr.Button('Generate', variant='primary')
      outpaint_output = gr.Gallery(label='Generated Images').style(grid=2, preview=True)
      selected_image_info = gr.HTML(value='', elem_classes=['block-info'])
      outpaint_output.select(fn=outpaint_gallery_select, outputs=[selected_outpaint_image_index, selected_image_info], show_progress=False)

      shared.create_base_send_targets(outpaint_output, selected_outpaint_image_index, tabs)
      shared.create_ext_send_targets(outpaint_output, selected_outpaint_image_index, tabs)
       
      def generate(image, prompt, negative_prompt, steps, batch_count, batch_size, guidance_scale, w, h, sampler, prior_cf_scale, prior_steps, negative_prior_prompt, input_seed, *injections):
        params = {
          'image': image,
          'prompt': prompt,
          'negative_decoder_prompt': negative_prompt,
          'num_steps': steps,
          'batch_count': batch_count,
          'batch_size': batch_size,
          'guidance_scale': guidance_scale,
          'w': w,
          'h': h,
          'sampler': sampler,
          'prior_cf_scale': prior_cf_scale,
          'prior_steps': prior_steps,
          'negative_prior_prompt': negative_prior_prompt,
          'input_seed': input_seed
        }

        params = augmentations['exec'](params, *injections)
        return generate_fn(params)
    
    generate_outpaint.click(generate, inputs=[
      shared.input_outpaint_image, prompt, negative_prompt, steps, batch_count, batch_size, guidance_scale, width, height, sampler, prior_scale, prior_steps, negative_prior_prompt, seed
    ] + augmentations['injections'],
    outputs=outpaint_output)

  return outpaint_block
