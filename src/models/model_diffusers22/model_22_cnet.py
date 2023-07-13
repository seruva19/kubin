import numpy as np
import torch
from transformers import pipeline


def generate_depth_map(image, k_params):
    depth_estimator = pipeline("depth-estimation")
    image = depth_estimator(image)["depth"]
    image = np.array(image)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    detected_map = torch.from_numpy(image).float() / 255.0
    depth_map = detected_map.permute(2, 0, 1)
    return depth_map


def generate_hint(image, cnet_condition, k_params):
    device = k_params("general", "device")

    if cnet_condition == "depth-map":
        return generate_depth_map(image, k_params).unsqueeze(0).half().to(device)

    return None
