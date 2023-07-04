import pandas as pd


def flatten_yaml(config):
    normalized = pd.json_normalize(config, sep=".")
    values = normalized.to_dict(orient="records")[0]

    return values
