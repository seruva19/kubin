import fnmatch
import os
import torch
from safetensors.torch import load_file
import hashlib


def calculate_file_hash(
    file_path, hash_algorithm="sha256", buffer_size=8192, hash_length=8
):
    hash_obj = hashlib.new(hash_algorithm)
    with open(file_path, "rb") as f:
        while chunk := f.read(buffer_size):
            hash_obj.update(chunk)
    full_hash = hash_obj.hexdigest()
    return full_hash[:hash_length]


def load_model_from_path(path):
    file_extension = os.path.splitext(path)[1].lstrip(".")
    if file_extension == "bin":
        lora_model = torch.load(path)
    else:
        lora_model = load_file(path)
    return lora_model


def scan_for_models(dirs, patterns):
    matching = [[] for _ in range(len(patterns))]

    for directory in dirs.split(";"):
        for root, _, filenames in os.walk(directory):
            for index, pattern in enumerate(patterns):
                for pattern_item in pattern.split(";"):
                    for filename in fnmatch.filter(filenames, pattern_item):
                        found_file = os.path.normpath(os.path.join(root, filename))
                        matching[index].append(
                            {
                                "name": filename,
                                "path": found_file,
                                "hash": calculate_file_hash(found_file),
                            }
                        )

    return matching


def filenames_with_hash(network_list):
    names_and_hashes = [
        [f'{d["name"]} [{d["hash"]}]' for d in sublist] for sublist in network_list
    ]
    return names_and_hashes
