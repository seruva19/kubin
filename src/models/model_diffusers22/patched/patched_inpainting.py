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
(https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/kandinsky2_2/pipeline_kandinsky2_2_inpainting.py)
"""

from copy import deepcopy
from typing import Callable, List, Optional, Union

import numpy as np
import PIL
import torch
import torch.nn.functional as F
from PIL import Image

from diffusers import __version__
from diffusers.models import UNet2DConditionModel, VQModel
from diffusers.schedulers import DDPMScheduler
from diffusers.utils import (
    is_accelerate_available,
    is_accelerate_version,
    logging,
    randn_tensor,
    replace_example_docstring,
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


def prepare_mask(masks):
    prepared_masks = []
    for mask in masks:
        old_mask = deepcopy(mask)
        for i in range(mask.shape[1]):
            for j in range(mask.shape[2]):
                if old_mask[0][i][j] == 1:
                    continue
                if i != 0:
                    mask[:, i - 1, j] = 0
                if j != 0:
                    mask[:, i, j - 1] = 0
                if i != 0 and j != 0:
                    mask[:, i - 1, j - 1] = 0
                if i != mask.shape[1] - 1:
                    mask[:, i + 1, j] = 0
                if j != mask.shape[2] - 1:
                    mask[:, i, j + 1] = 0
                if i != mask.shape[1] - 1 and j != mask.shape[2] - 1:
                    mask[:, i + 1, j + 1] = 0
        prepared_masks.append(mask)
    return torch.stack(prepared_masks, dim=0)


def prepare_mask_and_masked_image(image, mask, height, width):
    if image is None:
        raise ValueError("`image` input cannot be undefined.")

    if mask is None:
        raise ValueError("`mask_image` input cannot be undefined.")

    if isinstance(image, torch.Tensor):
        if not isinstance(mask, torch.Tensor):
            raise TypeError(
                f"`image` is a torch.Tensor but `mask` (type: {type(mask)} is not"
            )

        if image.ndim == 3:
            assert (
                image.shape[0] == 3
            ), "Image outside a batch should be of shape (3, H, W)"
            image = image.unsqueeze(0)

        if mask.ndim == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)

        if mask.ndim == 3:
            if mask.shape[0] == 1:
                mask = mask.unsqueeze(0)

            else:
                mask = mask.unsqueeze(1)

        assert (
            image.ndim == 4 and mask.ndim == 4
        ), "Image and Mask must have 4 dimensions"
        assert (
            image.shape[-2:] == mask.shape[-2:]
        ), "Image and Mask must have the same spatial dimensions"
        assert (
            image.shape[0] == mask.shape[0]
        ), "Image and Mask must have the same batch size"

        if image.min() < -1 or image.max() > 1:
            raise ValueError("Image should be in [-1, 1] range")

        if mask.min() < 0 or mask.max() > 1:
            raise ValueError("Mask should be in [0, 1] range")

        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1

        image = image.to(dtype=torch.float32)
    elif isinstance(mask, torch.Tensor):
        raise TypeError(
            f"`mask` is a torch.Tensor but `image` (type: {type(image)} is not"
        )
    else:
        if isinstance(image, (PIL.Image.Image, np.ndarray)):
            image = [image]

        if isinstance(image, list) and isinstance(image[0], PIL.Image.Image):
            image = [
                i.resize((width, height), resample=Image.BICUBIC, reducing_gap=1)
                for i in image
            ]
            image = [np.array(i.convert("RGB"))[None, :] for i in image]
            image = np.concatenate(image, axis=0)
        elif isinstance(image, list) and isinstance(image[0], np.ndarray):
            image = np.concatenate([i[None, :] for i in image], axis=0)

        image = image.transpose(0, 3, 1, 2)
        image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

        if isinstance(mask, (PIL.Image.Image, np.ndarray)):
            mask = [mask]

        if isinstance(mask, list) and isinstance(mask[0], PIL.Image.Image):
            mask = [i.resize((width, height), resample=PIL.Image.LANCZOS) for i in mask]
            mask = np.concatenate(
                [np.array(m.convert("L"))[None, None, :] for m in mask], axis=0
            )
            mask = mask.astype(np.float32) / 255.0
        elif isinstance(mask, list) and isinstance(mask[0], np.ndarray):
            mask = np.concatenate([m[None, None, :] for m in mask], axis=0)

        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        mask = torch.from_numpy(mask)

    mask = 1 - mask

    return mask, image


class KandinskyV22InpaintPipelinePatched(DiffusionPipeline):
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

    def prepare_latents(self, shape, dtype, device, generator, latents, scheduler):
        if latents is None:
            latents = randn_tensor(
                shape, generator=generator, device=device, dtype=dtype
            )
        else:
            if latents.shape != shape:
                raise ValueError(
                    f"Unexpected latents shape, got {latents.shape}, expected {shape}"
                )
            latents = latents.to(device)

        latents = latents * scheduler.init_noise_sigma
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
            torch.cuda.empty_cache()

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
        image: Union[torch.FloatTensor, PIL.Image.Image],
        mask_image: Union[torch.FloatTensor, PIL.Image.Image, np.ndarray],
        negative_image_embeds: Union[torch.FloatTensor, List[torch.FloatTensor]],
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 100,
        guidance_scale: float = 4.0,
        num_images_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        return_dict: bool = True,
    ):
        device = self._execution_device

        do_classifier_free_guidance = guidance_scale > 1.0

        if isinstance(image_embeds, list):
            image_embeds = torch.cat(image_embeds, dim=0)
        batch_size = image_embeds.shape[0] * num_images_per_prompt
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

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps_tensor = self.scheduler.timesteps

        mask_image, image = prepare_mask_and_masked_image(
            image, mask_image, height, width
        )

        image = image.to(dtype=image_embeds.dtype, device=device)
        image = self.movq.encode(image)["latents"]

        mask_image = mask_image.to(dtype=image_embeds.dtype, device=device)

        image_shape = tuple(image.shape[-2:])
        mask_image = F.interpolate(
            mask_image,
            image_shape,
            mode="nearest",
        )
        mask_image = prepare_mask(mask_image)
        masked_image = image * mask_image

        mask_image = mask_image.repeat_interleave(num_images_per_prompt, dim=0)
        masked_image = masked_image.repeat_interleave(num_images_per_prompt, dim=0)
        if do_classifier_free_guidance:
            mask_image = mask_image.repeat(2, 1, 1, 1)
            masked_image = masked_image.repeat(2, 1, 1, 1)

        num_channels_latents = self.movq.config.latent_channels

        height, width = downscale_height_and_width(
            height, width, self.movq_scale_factor
        )

        latents = self.prepare_latents(
            (batch_size, num_channels_latents, height, width),
            image_embeds.dtype,
            device,
            generator,
            latents,
            self.scheduler,
        )
        noise = torch.clone(latents)
        for i, t in enumerate(self.progress_bar(timesteps_tensor)):
            latent_model_input = (
                torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            )
            latent_model_input = torch.cat(
                [latent_model_input, masked_image, mask_image], dim=1
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
            init_latents_proper = image[:1]
            init_mask = mask_image[:1]

            if i < len(timesteps_tensor) - 1:
                noise_timestep = timesteps_tensor[i + 1]
                init_latents_proper = self.scheduler.add_noise(
                    init_latents_proper, noise, torch.tensor([noise_timestep])
                )

            latents = init_mask * init_latents_proper + (1 - init_mask) * latents

            if callback is not None and i % callback_steps == 0:
                callback(i, t, latents)

        latents = mask_image[:1] * image[:1] + (1 - mask_image[:1]) * latents
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
