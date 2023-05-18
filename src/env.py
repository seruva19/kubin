from extension.ext_registry import ExtensionRegistry
from l10n.localizer import Localizer
from models.model_mock import Model_Mock
from models.model_kd2 import Model_KD2
from options import KubinOptions

class Kubin:
  def __init__(self, args):
    self.args = args
    self.options = KubinOptions(args)

    self.model = Model_Mock() if self.args.mock else Model_KD2(
      self.args.device,
      self.args.task_type,
      self.args.cache_dir,
      self.args.model_version,
      self.args.use_flash_attention,
      self.args.output_dir
    )

    self.ext_registry = ExtensionRegistry(self.args.extensions_path, self.args.enabled_extensions, self.args.disabled_extensions, self.args.skip_install)
    self.localizer = Localizer(self.args.locale)

  def with_utils(self):
    import utils.file_system as fs_utils
    import utils.image as img_utils

    self.fs_utils = fs_utils
    self.img_utils = img_utils   

  def init_extensions(self):
    if not self.args.safe_mode:
      self.ext_registry.register(self)
    else:
      print('safe mode was initiated, skipping extension init phase')
