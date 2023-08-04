import pandas as pd
import yaml
import os


def flatten_yaml(config):
    normalized = pd.json_normalize(config, sep=".")
    values = normalized.to_dict(orient="records")[0]

    return values


class YamlConfig:
    def __init__(self, dir):
        self.dir = dir

        self.default_config = f"{self.dir}/config.default.yaml"
        self.user_config = f"{self.dir}/config.yaml"

    def read(self):
        user_conf_exists = os.path.exists(self.user_config)
        with open(
            self.user_config if user_conf_exists else self.default_config, "r"
        ) as stream:
            data = yaml.safe_load(stream)
            return data

    def reset(self, data):
        if os.path.exists(self.user_config):
            os.remove(self.user_config)

    def write(self, data):
        with open(f"{self.dir}/config.yaml", "w") as stream:
            yaml.safe_dump(data, stream)
