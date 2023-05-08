import gradio as gr

from env import Kubin

def create_extensions_info(kubin: Kubin):
  extensions = [(key, value) for key, value in kubin.ext_registry.extensions.items()]
  get_path = lambda x: f'{kubin.args.extensions_path}/{x}'

  extensions_info = []
  if len(extensions) > 0:
    for index, extension in enumerate(extensions):
      extensions_info.append({
        'name': extension[0],
        'title': extension[1]['title'],
        'is_tab': extension[1].get('tab_fn', None) is not None,
        'is_augment': extension[1].get('augment_fn', None) is not None,
        'path': get_path(extension[0]),
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
        'enabled': False
      })

  return extensions_info
  
def extensions_ui(kubin: Kubin, extensions_data):
  with gr.Column() as extensions_block:
    gr.HTML(f'Total extensions: {len(extensions_data)}')
    for index, extension_info in enumerate(extensions_data):
      with gr.Row():
        gr.Textbox(value=(
          f'title: {extension_info["title"]}\n'
          f'is tab: {extension_info["is_tab"]}\n'
          f'is augmentation: {extension_info["is_augment"]}\n'
          f'path: {extension_info["path"]}\n'
          f'status: {"enabled" if extension_info["enabled"] else "disabled"}'
        ), lines=5, label=f'{str(index+1)}: {extension_info["name"]}', interactive=False).style(show_copy_button=True)

    clear_ext_install_btn = gr.Button(value='Force extension reinstall on next run', label='Force reinstall', interactive=True)
    clear_ext_install_btn.click(lambda: kubin.ext_registry.force_reinstall())

  return extensions_block
