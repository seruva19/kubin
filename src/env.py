import os
from extension.ext_registry import ExtensionRegistry
from models.model_mock import Model_Mock
from models.model_kd2 import Model_KD2
from params import KubinParams


class Kubin:
    def __init__(self, args):
        self.params = KubinParams(args)

        self.root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        print(f"root dir: {self.root}")

        self.model = (
            Model_Mock(self.params) if self.params.mock else Model_KD2(self.params)
        )

        self.ext_registry = ExtensionRegistry(
            self.params.extensions_path,
            self.params.enabled_extensions,
            self.params.disabled_extensions,
            self.params.extensions_order,
            self.params.skip_install,
        )

    def with_utils(self):
        import utils.file_system as fs_utils
        import utils.image as img_utils

        self.fs_utils = fs_utils
        self.img_utils = img_utils

    def with_extensions(self):
        if not self.params.safe_mode:
            self.ext_registry.register(self)
        else:
            print("safe mode was initiated, skipping extension init phase")
