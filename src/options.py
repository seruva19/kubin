from extension.ext_registry import ExtensionRegistry
from l10n.localizer import Localizer
from models.model_mock import Model_Mock
from models.model_kd2 import Model_KD2

class KubinOptions:
  def __init__(self, args):
    self.args = args