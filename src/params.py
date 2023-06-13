import os
from omegaconf import OmegaConf
from engine.kandinsky import KandinskyCheckpoint


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
        if self.conf["general"]["pipeline"] != self._updated["general"]["pipeline"]:
            reload_model = True

        self.conf = self._updated.copy()
        return reload_model

    def save_user_config(self):
        user_config = "configs/kubin.yaml"
        OmegaConf.save(self.conf, user_config, resolve=True)

    def reset_config(self):
        user_config = "configs/kubin.yaml"
        if os.path.exists(user_config):
            os.remove(user_config)

    def merge_with_cli(self):
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
            self.conf["general"]["flash_attention"] = self.args.flash_attention == "use"

        if self.args.skip_install is not None:
            self.conf["general"]["skip_install"] = self.args.skip_install == "use"

        if self.args.safe_mode is not None:
            self.conf["general"]["safe_mode"] = self.args.safe_mode == "use"
