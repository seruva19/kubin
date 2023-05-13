
import subprocess
import sys
import os
import importlib.util
import sys

class ExtensionRegistry:
  def __init__(self, ext_path, enabled_exts, disabled_exts, skip_install):
    self.enabled = enabled_exts
    self.disabled = disabled_exts
    self.skip_install = skip_install
    self.root = ext_path

    self.extensions = {}

  def get_ext_folders(self):
    return [entry.name for entry in os.scandir(self.root) if entry.is_dir()]

  def get_enabled_extensions(self):
    return [] if self.enabled is None else [x.strip() for x in self.enabled.split(',')]
  
  def get_disabled_extensions(self):
    return [] if self.disabled is None else [x.strip() for x in self.disabled.split(',')]

  def register(self, kubin):
    ext_folders = self.get_ext_folders()
    print(f'found {len(ext_folders)} extensions')

    enabled_exts = self.get_enabled_extensions()
    if len(enabled_exts) > 0:
      ext_folders = filter(lambda ext: ext in enabled_exts, ext_folders)
      print(f'only following extensions are enabled: {self.enabled}')

    disabled_exts = self.get_disabled_extensions()

    for i, extension in enumerate(ext_folders):
      if (extension in disabled_exts):
        print(f'{i+1}: extension \'{extension}\' disabled, skipping')
      else:
        print(f'{i+1}: extension \'{extension}\' found')
        extension_reqs_path = f'{self.root}/{extension}/requirements.txt'
        extension_installed = f'{self.root}/{extension}/.installed'

        if not self.skip_install and os.path.isfile(extension_reqs_path):
          if os.path.exists(extension_installed):
            print(f'{i+1}: extension \'{extension}\' has requirements.txt, but was already installed, skipping')
          else:
            print(f'{i+1}: extension \'{extension}\' has requirements.txt, installing')
            self.install_ext_reqs(extension_reqs_path)
            open(extension_installed, 'a').close()

        extension_py_path = f'{self.root}/{extension}/setup.py'
        if os.path.exists(extension_py_path):
          spec = importlib.util.spec_from_file_location(f'{self.root}/{extension}', extension_py_path)
          if spec is not None:
            module = importlib.util.module_from_spec(spec)
            sys.modules[extension] = module
            if spec.loader is not None:
              spec.loader.exec_module(module)
              self.extensions[extension] = module.setup(kubin)
          
          print(f'{i+1}: extension \'{extension}\' successfully registered')
        else:
          print(f'{i+1}: setup.py not found for \'{extension}\', extension will not be registered')

  def install_ext_reqs(self, reqs_path):
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', f'{reqs_path}'])

  def standalone(self): # extensions with dedicated tab
    return list({key: value for key, value in self.extensions.items() if value.get('tab_fn', None) is not None}.values())

  def augment(self): # extensions for augmentation generation params
    return list({key: value for key, value in self.extensions.items() if value.get('augment_fn', None) is not None}.values())
  
  def force_reinstall(self, ext = None):
    ext_folders = self.get_ext_folders()
    for i, extension in enumerate(ext_folders):
      if ext is None or extension == ext:
        extension_installed = f'{self.root}/{extension}/.installed'
        if os.path.exists(extension_installed):
          os.remove(extension_installed)
          print(f'{i+1}: extension \'{extension}\' will be reinstalled on next run')