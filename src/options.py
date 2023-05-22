from extension.ext_registry import ExtensionRegistry
from models.model_mock import Model_Mock
from models.model_kd2 import Model_KD2

class KubinOptions:
  def __init__(self, args):
    self.args = args

  def __getattr__(self, key):
    return getattr(self.args, key)