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
        'type': extension[1]['type'],
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
        'type': 'unknown (not loaded)',
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
          f'Title: {extension_info["title"]}\n'
          f'Type: {extension_info["type"]}\n'
          f'Path: {extension_info["path"]}\n'
          f'Status: {"enabled" if extension_info["enabled"] else "disabled"}'
        ), lines=5, label=f'{str(index+1)}: {extension_info["name"]}', interactive=False).style(show_copy_button=True)

    clear_ext_install_btn = gr.Button(value='Force extension reinstall on next run', label='Force reinstall', interactive=True)
    clear_ext_install_btn.click(lambda: kubin.ext_registry.force_reinstall())

  return extensions_block
