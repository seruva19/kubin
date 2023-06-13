import os
from extension.ext_registry import ExtensionRegistry
from params import KubinParams
from models.model_mock import Model_Mock
from models.model_kd2 import Model_KD2
from models.model_diffusers import Model_Diffusers


class Kubin:
    def __init__(self, args):
        self.model = None

        self.root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        print(f"root dir: {self.root}")

        self.params = KubinParams(args)
        self.params.load_config()

        self.ext_registry = ExtensionRegistry(
            self.params("general", "extensions_path"),
            self.params("general", "enabled_extensions"),
            self.params("general", "disabled_extensions"),
            self.params("general", "extensions_order"),
            self.params("general", "skip_install"),
        )

    def with_pipeline(self):
        use_mock = self.params("general", "mock")
        pipeline = self.params("general", "pipeline")

        if self.model is not None:
            self.model.flush()

        self.model = (
            Model_Mock(self.params)
            if use_mock
            else Model_Diffusers(self.params)
            if pipeline == "diffusers"
            else Model_KD2(self.params)
        )

    def with_utils(self):
        import utils.file_system as fs_utils
        import utils.image as img_utils
        import utils.gradio_ui as ui_utils

        self.fs_utils = fs_utils
        self.img_utils = img_utils
        self.ui_utils = ui_utils

    def with_extensions(self):
        if not self.params("general", "safe_mode"):
            self.ext_registry.register(self)
        else:
            print("safe mode was initiated, skipping extension init phase")
