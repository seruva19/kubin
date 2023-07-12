import numpy as np
import torch
from transformers import pipeline


def generate_depth_map(image):
    depth_estimator = pipeline("depth-estimation")
    image = depth_estimator(image)["depth"]
    image = np.array(image)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    detected_map = torch.from_numpy(image).float() / 255.0
    depth_map = detected_map.permute(2, 0, 1)
    return depth_map


def generate_hint(image, cnet_condition):
    if cnet_condition == "depth-map":
        return generate_depth_map(image).unsqueeze(0).half().to("cuda")

    return None
