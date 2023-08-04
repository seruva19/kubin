import os

import torch
from extension.ext_registry import ExtensionRegistry
from hooks.hooks import HookStore
from params import KubinParams
from utils.logging import k_log


class Kubin:
    def __init__(self):
        pass

    def with_args(self, args):
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
        model_name = self.params("general", "model_name")
        pipeline = self.params("general", "pipeline")

        if self.model is not None:
            self.model.flush()

        if use_mock:
            from models.model_mock import Model_Mock

            self.model = Model_Mock(self.params)

        elif pipeline == "diffusers" and model_name == "kd22":
            from models.model_diffusers22.model_22 import Model_Diffusers22

            self.model = Model_Diffusers22(self.params)

        elif pipeline == "diffusers" and model_name == "kd21":
            from models.model_diffusers import Model_Diffusers

            self.model = Model_Diffusers(self.params)

        elif pipeline == "native" and model_name == "kd21":
            from models.model_kd21 import Model_KD21

            self.model = Model_KD21(self.params)

        elif pipeline == "native" and model_name == "kd20":
            from models.model_kd20 import Model_KD20

            self.model = Model_KD20(self.params)

        else:
            raise Exception("No suitable model found!")

        if not torch.cuda.is_available():
            k_log(
                "torch not compiled with CUDA enabled, ignore this message if it is intentional; otherwise, use 'install-torch' script to fix this"
            )

    def with_utils(self):
        import utils.file_system as fs_utils
        import utils.image as img_utils
        import utils.gradio_ui as ui_utils
        import utils.yaml as yaml_utils
        import utils.nn as nn_utils

        self.fs_utils = fs_utils
        self.img_utils = img_utils
        self.ui_utils = ui_utils
        self.yaml_utils = yaml_utils
        self.nn_utils = nn_utils

    def with_extensions(self):
        if not self.params("general", "safe_mode"):
            self.ext_registry.register(self)
        else:
            print("safe mode was initiated, skipping extension init phase")

    def with_hooks(self):
        self.ext_registry.bind_hooks(self)
