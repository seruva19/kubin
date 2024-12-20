import os

import torch
from extension.ext_registry import ExtensionRegistry
from params import KubinParams
from utils.logging import k_error, k_log


class Kubin:
    def __init__(self):
        pass

    def with_args(self, args):
        self.log = k_log
        self.elog = k_error

        self.model = None

        self.root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        k_log(f"root dir: {self.root}")

        self.params = KubinParams(args)
        self.params.load_config()

        self.ext_registry = ExtensionRegistry(
            self.params("general", "extensions_path"),
            self.params("general", "enabled_extensions"),
            self.params("general", "disabled_extensions"),
            self.params("general", "extensions_order"),
            self.params("general", "skip_install"),
        )

        self.params.register_change_callback(self.ext_registry.propagate_params_changes)

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
            from models.model_diffusers21.model_diffusers import Model_Diffusers

            self.model = Model_Diffusers(self.params)

        elif pipeline == "diffusers" and model_name == "kd30":
            from models.model_diffusers30.model_30 import Model_Diffusers3

            self.model = Model_Diffusers3(self.params)

        elif pipeline == "native" and model_name == "kd30":
            from models.model_30.model_kd30 import Model_KD3

            self.model = Model_KD3(self.params)

        elif pipeline == "native" and model_name == "kd31":
            from models.model_31.model_kd31 import Model_KD31

            self.model = Model_KD31(self.params)

        elif pipeline == "native" and model_name == "kd40":
            from models.model_40.model_kd40 import Model_KD40

            self.model = Model_KD40(self.params)

        elif pipeline == "native" and model_name == "kd21":
            from models.model_21.model_kd21 import Model_KD21

            self.model = Model_KD21(self.params)

        elif pipeline == "native" and model_name == "kd20":
            from models.model_20.model_kd20 import Model_KD20

            self.model = Model_KD20(self.params)

        else:
            k_log("no suitable model found! please select another option")

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
        import utils.env_data as env_utils

        self.fs_utils = fs_utils
        self.img_utils = img_utils
        self.ui_utils = ui_utils
        self.yaml_utils = yaml_utils
        self.nn_utils = nn_utils
        self.env_utils = env_utils

    def with_extensions(self):
        if not self.params("general", "safe_mode"):
            self.ext_registry.register(self)
        else:
            k_log("safe mode was initiated, skipping extension init phase")

    def with_hooks(self):
        self.ext_registry.bind_hooks(self)
