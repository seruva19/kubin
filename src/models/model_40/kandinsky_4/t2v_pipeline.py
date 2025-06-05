# This source code is licensed under the Apache License found in the
# LICENSE file in the current directory.
"""
The code has been adopted from Kandinsky-4
(https://github.com/ai-forever/Kandinsky-4/blob/main/kandinsky/t2v_pipeline.py)
"""


from typing import Union, List

import PIL
from PIL import Image

import numpy as np
from tqdm.auto import tqdm
import torch
import torchvision
from torchvision.transforms import ToPILImage
from einops import repeat
from diffusers import AutoencoderKLCogVideoX
from diffusers import CogVideoXDDIMScheduler

from models.model_40.model_kd40_env import Model_KD40_Environment

from .dit import DiffusionTransformer3D
from .text_embedders import T5TextEmbedder


@torch.no_grad()
def predict_x_0(noise_scheduler, model_output, timesteps, sample, device):
    init_alpha_device = noise_scheduler.alphas_cumprod.device
    alphas = noise_scheduler.alphas_cumprod.to(device)

    alpha_prod_t = alphas[timesteps][:, None, None, None]
    beta_prod_t = 1 - alpha_prod_t

    pred_original_sample = (alpha_prod_t**0.5) * sample - (
        beta_prod_t**0.5
    ) * model_output
    noise_scheduler.alphas_cumprod.to(init_alpha_device)
    return pred_original_sample


@torch.no_grad()
def get_velocity(
    model,
    x,
    t,
    text_embed,
    visual_cu_seqlens,
    text_cu_seqlens,
    num_goups=(1, 1, 1),
    scale_factor=(1.0, 1.0, 1.0),
):
    pred_velocity = model(
        x, text_embed, t, visual_cu_seqlens, text_cu_seqlens, num_goups, scale_factor
    )

    return pred_velocity


@torch.no_grad()
def diffusion_generate_renoise(
    model,
    noise_scheduler,
    shape,
    device,
    num_steps,
    text_embed,
    visual_cu_seqlens,
    text_cu_seqlens,
    num_goups=(1, 1, 1),
    scale_factor=(1.0, 1.0, 1.0),
    progress=False,
    seed=6554,
):
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)

    img = torch.randn(*shape, generator=generator).to(torch.bfloat16).to(device)
    noise_scheduler.set_timesteps(num_steps, device=device)

    timesteps = noise_scheduler.timesteps
    if progress:
        timesteps = tqdm(timesteps)
    for time in timesteps:
        model_time = time.unsqueeze(0).repeat(visual_cu_seqlens.shape[0] - 1)
        noise = (
            torch.randn(img.shape, generator=generator).to(torch.bfloat16).to(device)
        )
        img = noise_scheduler.add_noise(img, noise, time)

        pred_velocity = get_velocity(
            model,
            img.to(torch.bfloat16),
            model_time,
            text_embed.to(torch.bfloat16),
            visual_cu_seqlens,
            text_cu_seqlens,
            num_goups,
            scale_factor,
        )

        img = predict_x_0(
            noise_scheduler=noise_scheduler,
            model_output=pred_velocity.to(device),
            timesteps=model_time.to(device),
            sample=img.to(device),
            device=device,
        )

    return img


class Kandinsky4T2VPipeline:
    def __init__(
        self,
        environment: Model_KD40_Environment,
        device_map: Union[
            str, torch.device, dict
        ],  # {"dit": cuda:0, "vae": cuda:1, "text_embedder": cuda:1 }
        dit: DiffusionTransformer3D,
        text_embedder: T5TextEmbedder,
        vae: AutoencoderKLCogVideoX,
        noise_scheduler: CogVideoXDDIMScheduler,  # TODO base class
        resolution: int = 512,
        local_dit_rank=0,
        world_size=1,
    ):
        self.environment = environment

        if resolution not in [512]:
            raise ValueError("Resolution can be only 512")

        self.dit = dit
        self.noise_scheduler = noise_scheduler
        self.text_embedder = text_embedder
        self.vae = vae

        self.resolution = resolution

        self.device_map = device_map
        self.local_dit_rank = local_dit_rank
        self.world_size = world_size

        self.RESOLUTIONS = {
            512: [
                (512, 512),
                (352, 736),
                (736, 352),
                (384, 672),
                (672, 384),
                (480, 544),
                (544, 480),
            ],
        }

        self.progress_fn = lambda progress, desc: None

    def register_progress_bar(self, progress_fn=None):
        self.progress_fn = progress_fn if progress_fn is not None else self.progress_fn

    def update_progress(self, step, total_steps):
        if hasattr(self, "progress_fn"):
            try:
                self.progress_fn(
                    step / total_steps, desc=f"Generating {step}/{total_steps}"
                )
            except:
                self.progress_fn(step, total_steps)
        else:
            pass

    def __call__(
        self,
        text: str,
        save_path: str = "./test.mp4",
        bs: int = 1,
        time_length: int = 12,  # time in seconds 0 if you want generate image
        width: int = 512,
        height: int = 512,
        seed: int = None,
        return_frames: bool = False,
    ):
        num_steps = 4

        if seed is None:
            if self.local_dit_rank == 0:
                seed = torch.randint(2**63 - 1, (1,)).to(self.local_dit_rank)
            else:
                seed = torch.empty((1,), dtype=torch.int64).to(self.local_dit_rank)

            if self.world_size > 1:
                torch.distributed.broadcast(seed, 0)

            seed = seed.item()

        assert bs == 1

        if self.resolution != 512:
            raise NotImplementedError(f"Only 512 resolution is available for now")

        if (height, width) not in self.RESOLUTIONS[self.resolution]:
            raise ValueError(
                f"Wrong height, width pair. Available (height, width) are: {self.RESOLUTIONS[self.resolution]}"
            )

        if num_steps != 4:
            raise NotImplementedError(
                f"In the distilled version number of steps have to be strictly equal to 4"
            )

        num_frames = 1 if time_length == 0 else time_length * 8 // 4 + 1

        num_groups = (1, 1, 1) if self.resolution == 512 else (1, 2, 2)
        scale_factor = (1.0, 1.0, 1.0) if self.resolution == 512 else (1.0, 2.0, 2.0)

        if self.local_dit_rank == 0:
            with torch.no_grad():
                text_embed = (
                    self.text_embedder(text)
                    .squeeze(0)
                    .to(self.local_dit_rank, dtype=torch.bfloat16)
                )
        else:
            text_embed = torch.empty(224, 4096, dtype=torch.bfloat16).to(
                self.local_dit_rank
            )

        if self.world_size > 1:
            torch.distributed.broadcast(text_embed, 0)

        torch.cuda.empty_cache()

        visual_cu_seqlens = num_frames * torch.arange(
            bs + 1, dtype=torch.int32, device=self.device_map["dit"]
        )
        text_cu_seqlens = text_embed.shape[0] * torch.arange(
            bs + 1, dtype=torch.int32, device=self.device_map["dit"]
        )
        bs_text_embed = text_embed.repeat(bs, 1).to(self.device_map["dit"])
        shape = (bs * num_frames, height // 8, width // 8, 16)

        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                images = diffusion_generate_renoise(
                    self.dit,
                    self.noise_scheduler,
                    shape,
                    self.device_map["dit"],
                    num_steps,
                    bs_text_embed,
                    visual_cu_seqlens,
                    text_cu_seqlens,
                    num_groups,
                    scale_factor,
                    progress=True,
                    seed=seed,
                )

        torch.cuda.empty_cache()

        if self.local_dit_rank == 0:
            self.vae.num_latent_frames_batch_size = 1 if time_length == 0 else 2
            with torch.no_grad():
                images = (
                    1
                    / self.vae.config.scaling_factor
                    * images.to(device=self.device_map["vae"], dtype=torch.bfloat16)
                )
                images = (
                    images.permute(0, 3, 1, 2)
                    if time_length == 0
                    else images.permute(3, 0, 1, 2)
                )
                images = self.vae.decode(
                    images.unsqueeze(2 if time_length == 0 else 0)
                ).sample.float()
                images = torch.clip((images + 1.0) / 2.0, 0.0, 1.0)

        torch.cuda.empty_cache()

        if self.local_dit_rank == 0:
            if time_length == 0:
                return_images = []
                for i, image in enumerate(images.squeeze(2).cpu()):
                    return_images.append(ToPILImage()(image))
                return return_images
            else:
                if return_frames:
                    return_images = []
                    for i, image in enumerate(
                        images.squeeze(0).float().permute(1, 0, 2, 3).cpu()
                    ):
                        return_images.append(ToPILImage()(image))
                    return return_images
                else:
                    torchvision.io.write_video(
                        save_path,
                        255.0
                        * images.squeeze(0).float().permute(1, 2, 3, 0).cpu().numpy(),
                        fps=8,
                        options={"crf": "5"},
                    )
