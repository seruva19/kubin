import gradio as gr
from k2_modules.fuse import fuse
from k2_modules.img2img import i2i
from k2_modules.inpaint import inpaint
from k2_modules.txt2img import t2i

def launch(model):
  # text2image UI
  with gr.Blocks() as t2i_ui:
    with gr.Row():
      with gr.Column(scale=2):
        prompt = gr.Textbox('bunny, 4K photo', label='Prompt')
        negative_prompt = gr.Textbox('bad anatomy, deformed, blurry, depth of field', label='Negative prompt')
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
          seed = gr.Number(-1, label='Seed')
        with gr.Row():
          prior_scale = gr.Slider(0, 100, 4, step=1, label='Prior scale')
          prior_steps = gr.Slider(0, 100, 5, step=1, label='Prior steps')
      with gr.Column(scale=1):
        generate_t2i = gr.Button('Generate', variant='primary')
        t2i_output = gr.Gallery(label='Generated Images').style(grid=[2], preview=True)

        generate_t2i.click(fn=lambda *p: t2i(model, *p), inputs=[
          prompt,
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
          seed
        ], outputs=t2i_output)

  # image2image UI
  with gr.Blocks() as i2i_ui:
    with gr.Row():
      with gr.Column(scale=2):
        input_image = gr.Image(type='pil')
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
        i2i_output = gr.Gallery(label='Generated Images').style(grid=[2], preview=True)

        generate_i2i.click(fn=lambda *p: i2i(model, *p), inputs=[
          input_image,
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

  # TODO: add mixing for images > 2
  # image fuse UI
  def update(image):
    no_image = image == None
    return gr.update(label='Prompt' if no_image else 'Prompt (ignored, using image instead)', interactive=no_image)

  with gr.Blocks() as fuse_ui:
    with gr.Row():
      with gr.Column(scale=2):
        with gr.Row():
          with gr.Column(scale=1):
            image_1 = gr.Image(type='pil', label='Image')
            text_1 = gr.Textbox('bunny', label='Prompt')
            image_1.change(fn=update, inputs=image_1, outputs=text_1)
            weight_1 = gr.Slider(0, 1, 0.5, step=0.05, label='Weight')
          with gr.Column(scale=1):
            image_2 = gr.Image(type='pil', label='Image')
            text_2 = gr.Textbox('bunny', label='Prompt')
            image_2.change(fn=update, inputs=image_2, outputs=text_2)
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
          seed = gr.Number(-1, label='Seed')
        with gr.Row():
          prior_scale = gr.Slider(0, 100, 4, step=1, label='Prior scale')
          prior_steps = gr.Slider(0, 100, 5, step=1, label='Prior steps')
      with gr.Column(scale=1):
        generate_fuse = gr.Button('Generate', variant='primary')
        fuse_output = gr.Gallery(label='Generated Images').style(grid=[2], preview=True)

        generate_fuse.click(fn=lambda *p: fuse(model, *p), inputs=[
          image_1,
          image_2,
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
          seed
        ], outputs=fuse_output)

  # TODO: implement region of inpainting
  # inpaint UI
  with gr.Blocks() as inpaint_ui:
    with gr.Row():
      with gr.Column(scale=2):
        image_with_mask = gr.ImageMask(type='pil')
        prompt = gr.Textbox('bunny, 4K photo', label='Prompt')
        negative_prompt = gr.Textbox('bad anatomy, deformed, blurry, depth of field', label='Negative prompt')
        with gr.Row():
          inpainting_target = gr.Radio(['only mask', 'all but mask'], value='only mask', label='Inpainting target')
          inpainting_region = gr.Radio(['whole', 'mask'], value='whole', label='Inpainting region', interactive=False)
        with gr.Row():
          steps = gr.Slider(0, 200, 100, step=1, label='Steps')
          guidance_scale = gr.Slider(0, 30, 10, step=1, label='Guidance scale')
        with gr.Row():
          batch_count = gr.Slider(0, 16, 4, step=1, label='Batch count')
          batch_size = gr.Slider(0, 16, 1, step=1, label='Batch size')
        with gr.Row():
          width = gr.Slider(0, 1024, 768, step=1, label='Width')
          height = gr.Slider(0, 1024, 768, step=1, label='Height')
        with gr.Row():
          sampler = gr.Radio(['ddim_sampler', 'p_sampler', 'plms_sampler'], value='ddim_sampler', label='Sampler')
          seed = gr.Number(-1, label='Seed')
        with gr.Row():
          prior_scale = gr.Slider(0, 100, 4, step=1, label='Prior scale')
          prior_steps = gr.Slider(0, 100, 5, step=1, label='Prior steps')
      with gr.Column(scale=1):
        generate_inpaint = gr.Button('Generate', variant='primary')
        inpaint_output = gr.Gallery(label='Generated Images').style(grid=[2], preview=True)

        generate_inpaint.click(fn=lambda *p: inpaint(model, *p), inputs=[
          image_with_mask,
          prompt,
          negative_prompt,
          inpainting_target,
          inpainting_region,
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
        ], outputs=inpaint_output)

  ui = gr.TabbedInterface(
    [t2i_ui, i2i_ui, fuse_ui, inpaint_ui],
    ['Text To Image', 'Image To Image', 'Mix Images', 'Inpaint Image'],
    title='Kubin: Kandinsky 2.1 WebGUI',
    css='html {min-height: 101%} .prose {display: none}'
  )
  
  return ui