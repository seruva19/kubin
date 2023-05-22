import gradio as gr

from env import Kubin

def create_extensions_info(kubin: Kubin):
  extensions = [(key, value) for key, value in kubin.ext_registry.extensions.items()]
  get_path = lambda x: f'{kubin.options.extensions_path}/{x}'

  extensions_info = []
  if len(extensions) > 0:
    for index, extension in enumerate(extensions):
      extensions_info.append({
        'name': extension[0],
        'title': extension[1]['title'],
        'is_tab': extension[1].get('tab_fn', None) is not None,
        'is_augment': extension[1].get('augment_fn', None) is not None,
        'path': get_path(extension[0]),
        'url': '',
        'enabled': True
      })
    
  disabled_exts = kubin.ext_registry.get_disabled_extensions()
  if len(disabled_exts) > 0:
    disabled_exts = [ext.strip() for ext in disabled_exts]
    for index, extension_name in list(enumerate(disabled_exts)):
      extensions_info.append({
        'name': extension_name,
        'title': 'unknown (not loaded)',
        'is_tab': 'unknown (not loaded)',
        'is_augment': 'unknown (not loaded)',
        'path': get_path(extension_name),
        'url': '',
        'enabled': False
      })

  return extensions_info
  
def extensions_ui(kubin: Kubin, extensions_data):
  with gr.Column() as extensions_block:
    gr.Markdown(f'## Local extensions found: {len(extensions_data)}')
    for index, extension_info in enumerate(extensions_data):
      with gr.Column():
        gr.Markdown(f'### {str(index+1)}: {extension_info["name"]}')
        with gr.Row():
          gr.Textbox(f'{extension_info["title"]}', label='Title', lines=1, interactive=False)
          gr.Textbox(f'{extension_info["path"]}', label='Path', lines=1, interactive=False)
        with gr.Row():
          gr.Textbox(f'{extension_info["is_tab"]}', label='Exposes tab', lines=1, interactive=False)
          gr.Textbox(f'{extension_info["is_augment"]}', label='Exposes augmentation', lines=1, interactive=False)
        with gr.Row():
          gr.Textbox(f'{extension_info["url"]}', label='URL', lines=1, interactive=False)
          gr.Textbox(f'{"enabled" if extension_info["enabled"] else "disabled"}', label='Enabled', lines=1, interactive=False)

        with gr.Row():
          with gr.Column(scale=1):
            clear_ext_install_btn = gr.Button(value='Force reinstall on next launch', label='Force reinstall', interactive=True).style(size='sm')
            clear_ext_install_btn.click(lambda: kubin.ext_registry.force_reinstall(extension_info["name"]))
          with gr.Column(scale=3):
            pass

    clear_ext_install_all_btn = gr.Button(value='Force reinstall of all extensions on next launch', label='Force reinstall', interactive=True)
    clear_ext_install_all_btn.click(lambda: kubin.ext_registry.force_reinstall())

  return extensions_block
