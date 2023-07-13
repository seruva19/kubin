import os
import json
from omegaconf import OmegaConf
from model_utils.kandinsky_utils import KandinskyCheckpoint
from utils.yaml import flatten_yaml

default_value = "__default__"


class KubinParams:
    def __init__(self, args):
        self.args = args
        self.checkpoint = KandinskyCheckpoint()

    def get_conf_item(self, conf, keys):
        if isinstance(keys, tuple):
            value = conf
            for key in keys:
                value = value.get(key, default_value)
            return value
        else:
            return conf[keys]

    def __call__(self, *keys):
        value = self.get_conf_item(self.conf, keys)
        if value == default_value:
            value = self.get_conf_item(self._default, keys)
            print(
                f"key {('.').join(keys)} not found in user config, using default value ({value})"
            )

        return value

    def to_json(self):
        return json.dumps(flatten_yaml(OmegaConf.to_container(self.conf)))

    def load_config(self):
        default_config = "configs/kubin.default.yaml"
        user_config = "configs/kubin.yaml"
        custom_config_exists = False

        self._default = OmegaConf.load(default_config)

        if self.args.from_config:
            if os.path.exists(self.args.from_config):
                self.conf = OmegaConf.load(self.args.from_config)
                custom_config_exists = True
            else:
                print(f"Custom config file {self.args.from_config} not found, ignoring")

        if not custom_config_exists:
            if os.path.exists(user_config):
                self.conf = OmegaConf.load(user_config)
            else:
                self.conf = OmegaConf.load(default_config)

        self.merge_with_cli()
        self._updated = self.conf.copy()

    def apply_config_changes(self):
        reload_model = False
        if (
            self.conf["general"]["pipeline"] != self._updated["general"]["pipeline"]
            or self.conf["general"]["device"] != self._updated["general"]["device"]
            or self.conf["diffusers"]["half_precision_weights"]
            != self._updated["diffusers"]["half_precision_weights"]
            or self.conf["general"]["model_name"]
            != self._updated["general"]["model_name"]
        ):
            print("model will be reloaded")
            reload_model = True

        self.conf = self._updated.copy()
        return reload_model

    def save_user_config(self):
        user_config = "configs/kubin.yaml"
        OmegaConf.save(self.conf, user_config, resolve=True)

    def reset_config(self):
        user_config = "configs/kubin.yaml"
        user_config_bak = "configs/kubin.yaml.bak"
        if os.path.exists(user_config):
            if os.path.isfile(user_config_bak):
                os.remove(user_config_bak)
            os.rename(user_config, user_config_bak)

    def merge_with_cli(self):
        if self.args.model_name is not None:
            self.conf["general"]["model_name"] = self.args.model_name

        if self.args.device is not None:
            self.conf["general"]["device"] = self.args.device

        if self.args.cache_dir is not None:
            self.conf["general"]["cache_dir"] = self.args.cache_dir

        if self.args.output_dir is not None:
            self.conf["general"]["output_dir"] = self.args.output_dir

        if self.args.share is not None:
            self.conf["general"]["share"] = self.args.share

        if self.args.extensions_path is not None:
            self.conf["general"]["extensions_path"] = self.args.extensions_path

        if self.args.enabled_extensions is not None:
            self.conf["general"]["enabled_extensions"] = self.args.enabled_extensions

        if self.args.disabled_extensions is not None:
            self.conf["general"]["disabled_extensions"] = self.args.disabled_extensions

        if self.args.extensions_order is not None:
            self.conf["general"]["extensions_order"] = self.args.extensions_order

        if self.args.pipeline is not None:
            self.conf["general"]["pipeline"] = self.args.pipeline

        if self.args.server_name is not None:
            self.conf["gradio"]["server_name"] = self.args.server_name

        if self.args.server_port is not None:
            self.conf["gradio"]["server_port"] = self.args.server_port

        if self.args.concurrency_count is not None:
            self.conf["gradio"]["concurrency_count"] = self.args.concurrency_count

        if self.args.theme is not None:
            self.conf["gradio"]["theme"] = self.args.theme

        if self.args.debug is not None:
            self.conf["gradio"]["debug"] = self.args.debug == "use"

        if self.args.mock is not None:
            self.conf["general"]["mock"] = self.args.mock == "use"

        if self.args.flash_attention is not None:
            self.conf["native"]["flash_attention"] = self.args.flash_attention == "use"

        if self.args.skip_install is not None:
            self.conf["general"]["skip_install"] = self.args.skip_install == "use"

        if self.args.safe_mode is not None:
            self.conf["general"]["safe_mode"] = self.args.safe_mode == "use"

        if self.args.optimize is not None:
            for optimize_param in [x.strip() for x in self.args.optimize.split(",")]:
                if optimize_param == "half_weights":
                    self.conf["diffusers"]["half_precision_weights"] = True
                if optimize_param == "xformers":
                    self.conf["diffusers"]["enable_xformers"] = True
                if optimize_param == "sliced_attention":
                    self.conf["diffusers"]["enable_sliced_attention"] = True
                if optimize_param == "sequential_offload":
                    self.conf["diffusers"]["sequential_cpu_offload"] = True
                if optimize_param == "channels_last":
                    self.conf["diffusers"]["channels_last_memory"] = True

        if self.args.side_tabs is not None:
            self.conf["ui"]["side_tabs"] = self.args.side_tabs == "use"
