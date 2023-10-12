# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The code has been adopted from diffusers
(https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/kandinsky2_2/pipeline_kandinsky2_2_img2img.py)
"""

from typing import Callable, List, Optional, Union

import numpy as np
import PIL
import torch
from PIL import Image

from diffusers.models import UNet2DConditionModel, VQModel
from diffusers.schedulers import DDPMScheduler
from diffusers.utils import (
    is_accelerate_available,
    is_accelerate_version,
    logging,
    randn_tensor,
)
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput


logger = logging.get_logger(__name__)


def downscale_height_and_width(height, width, scale_factor=8):
    new_height = height // scale_factor**2
    if height % scale_factor**2 != 0:
        new_height += 1
    new_width = width // scale_factor**2
    if width % scale_factor**2 != 0:
        new_width += 1
    return new_height * scale_factor, new_width * scale_factor


def prepare_image(pil_image, w=512, h=512):
    pil_image = pil_image.resize((w, h), resample=Image.BICUBIC, reducing_gap=1)
    arr = np.array(pil_image.convert("RGB"))
    arr = arr.astype(np.float32) / 127.5 - 1
    arr = np.transpose(arr, [2, 0, 1])
    image = torch.from_numpy(arr).unsqueeze(0)
    return image


class KandinskyV22Img2ImgPipelinePatched(DiffusionPipeline):
    def __init__(
        self,
        unet: UNet2DConditionModel,
        scheduler: DDPMScheduler,
        movq: VQModel,
    ):
        super().__init__()

        self.register_modules(
            unet=unet,
            scheduler=scheduler,
            movq=movq,
        )
        self.movq_scale_factor = 2 ** (len(self.movq.config.block_out_channels) - 1)

    def get_timesteps(self, num_inference_steps, strength, device):
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start:]

        return timesteps, num_inference_steps - t_start

    def prepare_latents(
        self,
        image,
        timestep,
        batch_size,
        num_images_per_prompt,
        dtype,
        device,
        generator=None,
    ):
        if not isinstance(image, (torch.Tensor, PIL.Image.Image, list)):
            raise ValueError(
                f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}"
            )

        image = image.to(device=device, dtype=dtype)

        batch_size = batch_size * num_images_per_prompt

        if image.shape[1] == 4:
            init_latents = image

        else:
            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )

            elif isinstance(generator, list):
                init_latents = [
                    self.movq.encode(image[i : i + 1]).latent_dist.sample(generator[i])
                    for i in range(batch_size)
                ]
                init_latents = torch.cat(init_latents, dim=0)
            else:
                init_latents = self.movq.encode(image).latent_dist.sample(generator)

            init_latents = self.movq.config.scaling_factor * init_latents

        init_latents = torch.cat([init_latents], dim=0)

        shape = init_latents.shape
        noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

        init_latents = self.scheduler.add_noise(init_latents, noise, timestep)

        latents = init_latents

        return latents

    def enable_model_cpu_offload(self, gpu_id=0):
        if is_accelerate_available() and is_accelerate_version(">=", "0.17.0.dev0"):
            from accelerate import cpu_offload_with_hook
        else:
            raise ImportError(
                "`enable_model_cpu_offload` requires `accelerate v0.17.0` or higher."
            )

        device = torch.device(f"cuda:{gpu_id}")

        if self.device.type != "cpu":
            self.to("cpu", silence_dtype_warnings=True)
            torch.cuda.empty_cache()  # otherwise we don't see the memory savings (but they probably exist)

        hook = None
        for cpu_offloaded_model in [self.unet, self.movq]:
            _, hook = cpu_offload_with_hook(
                cpu_offloaded_model, device, prev_module_hook=hook
            )

        self.final_offload_hook = hook

    @torch.no_grad()
    def __call__(
        self,
        image_embeds: Union[torch.FloatTensor, List[torch.FloatTensor]],
        image: Union[
            torch.FloatTensor,
            PIL.Image.Image,
            List[torch.FloatTensor],
            List[PIL.Image.Image],
        ],
        negative_image_embeds: Union[torch.FloatTensor, List[torch.FloatTensor]],
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 100,
        guidance_scale: float = 4.0,
        strength: float = 0.3,
        num_images_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: Optional[str] = "pil",
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        return_dict: bool = True,
    ):
        device = self._execution_device

        do_classifier_free_guidance = guidance_scale > 1.0

        if isinstance(image_embeds, list):
            image_embeds = torch.cat(image_embeds, dim=0)
        batch_size = image_embeds.shape[0]
        if isinstance(negative_image_embeds, list):
            negative_image_embeds = torch.cat(negative_image_embeds, dim=0)

        if do_classifier_free_guidance:
            image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
            negative_image_embeds = negative_image_embeds.repeat_interleave(
                num_images_per_prompt, dim=0
            )

            image_embeds = torch.cat([negative_image_embeds, image_embeds], dim=0).to(
                dtype=self.unet.dtype, device=device
            )

        if not isinstance(image, list):
            image = [image]
        if not all(isinstance(i, (PIL.Image.Image, torch.Tensor)) for i in image):
            raise ValueError(
                f"Input is in incorrect format: {[type(i) for i in image]}. Currently, we only support  PIL image and pytorch tensor"
            )

        image = torch.cat([prepare_image(i, width, height) for i in image], dim=0)
        image = image.to(dtype=image_embeds.dtype, device=device)

        latents = self.movq.encode(image)["latents"]
        latents = latents.repeat_interleave(num_images_per_prompt, dim=0)
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps, num_inference_steps = self.get_timesteps(
            num_inference_steps, strength, device
        )
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)
        height, width = downscale_height_and_width(
            height, width, self.movq_scale_factor
        )
        latents = self.prepare_latents(
            latents,
            latent_timestep,
            batch_size,
            num_images_per_prompt,
            image_embeds.dtype,
            device,
            generator,
        )
        for i, t in enumerate(self.progress_bar(timesteps)):
            latent_model_input = (
                torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            )

            added_cond_kwargs = {"image_embeds": image_embeds}
            noise_pred = self.unet(
                sample=latent_model_input,
                timestep=t,
                encoder_hidden_states=None,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

            if do_classifier_free_guidance:
                noise_pred, variance_pred = noise_pred.split(latents.shape[1], dim=1)
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                _, variance_pred_text = variance_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )
                noise_pred = torch.cat([noise_pred, variance_pred_text], dim=1)

            if not (
                hasattr(self.scheduler.config, "variance_type")
                and self.scheduler.config.variance_type in ["learned", "learned_range"]
            ):
                noise_pred, _ = noise_pred.split(latents.shape[1], dim=1)

            latents = self.scheduler.step(
                noise_pred,
                t,
                latents,
                generator=generator,
            )[0]

            if callback is not None and i % callback_steps == 0:
                callback(i, t, latents)

        image = self.movq.decode(latents, force_not_quantize=True)["sample"]

        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if output_type not in ["pt", "np", "pil"]:
            raise ValueError(
                f"Only the output types `pt`, `pil` and `np` are supported not output_type={output_type}"
            )

        if output_type in ["np", "pil"]:
            image = image * 0.5 + 0.5
            image = image.clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).float().numpy()

        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)
