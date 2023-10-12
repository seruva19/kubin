from IPython.display import Video
from deforum_kandinsky import KandinskyV22Img2ImgPipeline, DeforumKandinsky
from diffusers import KandinskyV22PriorPipeline
import imageio.v2 as iio
from PIL import Image
import numpy as np
import torch
import datetime
from tqdm.notebook import tqdm
import ipywidgets as widgets
from IPython import display

# def load_models(model, model_version, device):
#     if model_version == "2.2":
#         image_encoder = (
#             CLIPVisionModelWithProjection.from_pretrained(
#                 "kandinsky-community/kandinsky-2-2-prior", subfolder="image_encoder"
#             )
#             .to(torch.float16)
#             .to(device)
#         )

#         unet = (
#             UNet2DConditionModel.from_pretrained(
#                 "kandinsky-community/kandinsky-2-2-decoder", subfolder="unet"
#             )
#             .to(torch.float16)
#             .to(device)
#         )

#         prior = KandinskyV22PriorPipeline.from_pretrained(
#             "kandinsky-community/kandinsky-2-2-prior",
#             image_encoder=image_encoder,
#             torch_dtype=torch.float16,
#         ).to(device)
#         decoder = KandinskyV22Img2ImgPipeline.from_pretrained(
#             "kandinsky-community/kandinsky-2-2-decoder",
#             unet=unet,
#             torch_dtype=torch.float16,
#         ).to(device)

#     elif model_version == 2.1:
#         image_encoder = CLIPVisionModelWithProjection.from_pretrained(
#             "kandinsky-community/kandinsky-2-1-prior",
#             subfolder="image_encoder",
#             torch_dtype=torch.float16,
#         ).to(device)
#         unet = UNet2DConditionModel.from_pretrained(
#             "kandinsky-community/kandinsky-2-1",
#             subfolder="unet",
#             torch_dtype=torch.float16,
#         ).to(device)
#         prior = KandinskyPriorPipeline.from_pretrained(
#             "kandinsky-community/kandinsky-2-1-prior", torch_dtype=torch.float16
#         ).to(device)
#         decoder = KandinskyImg2ImgPipeline.from_pretrained(
#             "kandinsky-community/kandinsky-2-1", unet=unet, torch_dtype=torch.float16
#         ).to(device)

#     return prior, decoder


def load_models(prior, decoder, device):
    prior = KandinskyV22PriorPipeline(**prior.components).to(device)
    decoder = KandinskyV22Img2ImgPipeline(**decoder.components).to(device)

    return prior, decoder


def create_deforum(prior, decoder, device):
    deforum = DeforumKandinsky(prior=prior, decoder_img2img=decoder, device=device)
    return deforum


def create_animation(
    deforum, prompts, negative_prompts, animations, durations, H, W, fps, save_samples
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

    # out = widgets.Output()
    pbar = tqdm(animation, total=len(deforum))
    # display.display(out)

    # with out:
    for index, item in enumerate(pbar):
        frame = item["image"]
        frames.append(frame)
        # display.clear_output(wait=True)
        # display.display(frame)
        for key, value in item.items():
            if not isinstance(value, (np.ndarray, torch.Tensor, Image.Image)):
                print(f"{key}: {value}")

    # display.clear_output(wait=True)
    frames2video(frames, "output_2_2.mp4", fps=24)
    # display.Video(url="output_2_2.mp4")
    return "output_2_2.mp4"


def frames2video(frames, output_path="video.mp4", fps=24, display=False):
    writer = iio.get_writer(output_path, fps=fps)
    for frame in tqdm(frames):
        writer.append_data(np.array(frame))
    writer.close()
    if display:
        display.Video(url=output_path)
