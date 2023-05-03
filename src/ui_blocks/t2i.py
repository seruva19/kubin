import gradio as gr
from ui_blocks.shared.ui_shared import SharedUI
from shared import params

def t2i_gallery_select(evt: gr.SelectData):
  return [evt.index, f'Selected image index: {evt.index}']

def t2i_ui(generate_fn, shared: SharedUI, tabs):
  selected_t2i_image_index = gr.State(None) # type: ignore
  augmentations = shared.create_ext_augment_blocks('t2i')

  with gr.Box():
    with gr.Row() as t2i_block:
      with gr.Box():
        prompt = gr.Textbox('', label='Prompt', placeholder='')
        negative_decoder_prompt = gr.Textbox('', placeholder='', label='Negative prompt')

        with gr.Accordion("Advanced Image Settings", open=False):
          with gr.Row():
            steps = gr.Slider(25, 200, 100, step=1, label='Steps')
            guidance_scale = gr.Slider(1, 30, 4, step=0.5, label='Guidance scale')

          with gr.Row():
            batch_count = gr.Slider(1, 16, 4, step=1, label='Batch count')
            batch_size = gr.Slider(1, 16, 1, step=1, label='Batch size')

          with gr.Row():
            width = gr.Slider(params.image_width_min, params.image_width_max, 768, step=params.image_width_step,
                              label='Width')
            height = gr.Slider(params.image_height_min, params.image_height_max, 768, step=params.image_height_step,
                               label='Height')

          with gr.Row():
            sampler = gr.Radio(['ddim_sampler', 'p_sampler', 'plms_sampler'], value='p_sampler', label='Sampler')

          with gr.Row():
            seed = gr.Number(-1, label='Seed', precision=0)
            prior_scale = gr.Slider(1, 100, 4, step=1, label='Prior Config scale')
            prior_steps = gr.Slider(1, 100, 5, step=1, label='Prior steps')

          with gr.Row():
            negative_prior_prompt = gr.Textbox('', label='Negative prior prompt')

          with gr.Row():
            augmentations['ui']()

        t2i_output = gr.Gallery(label='Generated Images').style(columns=[4], rows=[4], object_fit="contain",
                                                                height="auto", preview=True)
        selected_image_info = gr.HTML(value='', elem_classes=['block-info'])
        t2i_output.select(fn=t2i_gallery_select, outputs=[selected_t2i_image_index, selected_image_info],
                          show_progress=False)

  with gr.Box():
    with gr.Row():
      generate_t2i = gr.Button('Generate', variant='primary')

      def generate(prompt, negative_decoder_prompt, num_steps, batch_count, batch_size, guidance_scale, w, h, sampler,
                   prior_cf_scale, prior_steps, negative_prior_prompt, input_seed, *injections):
        params = {
          'prompt': prompt,
          'negative_decoder_prompt': negative_decoder_prompt,
          'num_steps': num_steps,
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

      generate_t2i.click(generate, inputs=[
                                            prompt, negative_decoder_prompt, steps, batch_count, batch_size,
                                            guidance_scale, width, height, sampler, prior_scale, prior_steps,
                                            negative_prior_prompt, seed
                                          ] + augmentations['injections'],
                         outputs=t2i_output
                         )
    with gr.Accordion("Send image(s) to : Image 2 Image, Mix Images, Inpaint, Upscale,", open=True):
      shared.create_base_send_targets(t2i_output, selected_t2i_image_index, tabs)
      shared.create_ext_send_targets(t2i_output, selected_t2i_image_index, tabs)
    
      def generate(prompt, negative_decoder_prompt, num_steps, batch_count, batch_size, guidance_scale, w, h, sampler, prior_cf_scale, prior_steps, negative_prior_prompt, input_seed, *injections):
        params = {
          'prompt': prompt,
          'negative_decoder_prompt': negative_decoder_prompt,
          'num_steps': num_steps,
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

      generate_t2i.click(generate, inputs=[
          prompt, negative_decoder_prompt, steps, batch_count, batch_size, guidance_scale, width, height, sampler, prior_scale, prior_steps, negative_prior_prompt, seed
        ] + augmentations['injections'],
        outputs=t2i_output
      )

  return t2i_block
