from uuid import uuid4
from deforum_kandinsky import KandinskyV22Img2ImgPipeline, DeforumKandinsky
from diffusers import KandinskyV22PriorPipeline
import imageio.v2 as iio
from PIL import Image
import numpy as np
import torch
from tqdm.notebook import tqdm
import os


def load_models(prior, decoder, device):
    prior = KandinskyV22PriorPipeline(**prior.components).to(device)
    decoder = KandinskyV22Img2ImgPipeline(**decoder.components).to(device)
    return prior, decoder


def create_deforum(prior, decoder, device):
    deforum = DeforumKandinsky(prior=prior, decoder_img2img=decoder, device=device)
    return deforum


def create_animation(
    deforum,
    prompts,
    negative_prompts,
    animations,
    durations,
    H,
    W,
    fps,
    save_samples,
    output_dir,
):
    animation = deforum(
        prompts=prompts,
        negative_prompts=negative_prompts,
        animations=animations,
        prompt_durations=durations,
        H=H,
        W=W,
        fps=fps,
        save_samples=save_samples,
    )

    frames = []

    pbar = tqdm(animation, total=len(deforum))

    for index, item in enumerate(pbar):
        frame = item["image"]
        frames.append(frame)
        for key, value in item.items():
            if not isinstance(value, (np.ndarray, torch.Tensor, Image.Image)):
                print(f"{key}: {value}")

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"animation-{uuid4()}.mp4")
    frames2video(frames, output_path, fps=24)
    return output_path


def frames2video(frames, output_path, fps=24):
    writer = iio.get_writer(output_path, fps=fps)
    for frame in tqdm(frames):
        writer.append_data(np.array(frame))
    writer.close()
