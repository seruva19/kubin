import os
from extension.ext_registry import ExtensionRegistry
from models.model_mock import Model_Mock
from models.model_kd2 import Model_KD2
from options import KubinOptions

class Kubin:
  def __init__(self, args):
    self.options = KubinOptions(args)

    self.model = Model_Mock() if self.options.mock else Model_KD2(
      self.options.device,
      self.options.task_type,
      self.options.cache_dir,
      self.options.model_version,
      self.options.use_flash_attention,
      self.options.output_dir
    )

    self.ext_registry = ExtensionRegistry(self.options.extensions_path, self.options.enabled_extensions, self.options.disabled_extensions, self.options.skip_install)

  def with_utils(self):
    import utils.file_system as fs_utils
    import utils.image as img_utils

    self.fs_utils = fs_utils
    self.img_utils = img_utils   
    self.root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

  def with_extensions(self):
    if not self.options.safe_mode:
      self.ext_registry.register(self)
    else:
      print('safe mode was initiated, skipping extension init phase')
