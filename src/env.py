from l10n.localizer import Localizer
from models.model_mock import Model_Mock
from models.model_kd2 import Model_KD2

class Kubin:
  def __init__(self, args):
    self.args = args
  
    self.model = Model_Mock() if self.args.model_version == 'none' else Model_KD2(
      self.args.device, self.args.task_type, self.args.cache_dir, self.args.model_version, self.args.use_flash_attention, self.args.output_dir)

    self.localizer = Localizer(self.args.locale)