import numpy as np
import torch
from transformers import pipeline
from utils.logging import k_log


def generate_depth_map(depth_estimator, image, k_params):
    image = depth_estimator(image)["depth"]
    image = np.array(image)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    detected_map = torch.from_numpy(image).float() / 255.0
    depth_map = detected_map.permute(2, 0, 1)
    return depth_map


def generate_hint(model, image, cnet_condition, model_type, k_params):
    device = k_params("general", "device")
    cache_dir = k_params("general", "cache_dir")

    if cnet_condition == "depth-map":
        depth_estimator = model.pipe_info["cnet_depth_estimator"]
        dmap_type = model.pipe_info["cnet_dmap_type"]

        if depth_estimator is None or dmap_type != model_type:
            k_log(f"loading depth estimation model: {model_type}")
            depth_estimator = pipeline(
                "depth-estimation",
                model=model_type,
                model_kwargs={"cache_dir": cache_dir},
            )
            model.pipe_info["cnet_depth_estimator"] = depth_estimator
            model.pipe_info["dmap_type"] = model_type

        return (
            generate_depth_map(depth_estimator, image, k_params)
            .unsqueeze(0)
            .half()
            .to(device)
        )
