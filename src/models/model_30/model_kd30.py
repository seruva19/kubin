import gc
import torch
import torch.backends
import torch

from models.model_30.kandinsky3 import (
    get_T2I_pipeline,
    get_inpainting_pipeline,
    Kandinsky3T2IPipeline,
    Kandinsky3InpaintingPipeline,
)

import itertools
import os
import secrets
from models.model_30.kandinsky3.inpainting_optimized_pipeline import (
    Kandinsky3InpaintingOptimizedPipeline,
)
from models.model_30.kandinsky3.t2i_optimized_pipeline import (
    Kandinsky3T2IOptimizedPipeline,
)
from models.model_30.model_kd30_env import Model_KD3_Environment
from params import KubinParams
from utils.file_system import save_output
from utils.image import create_inpaint_targets
from utils.logging import k_log

from model_utils.diffusers_samplers import use_sampler


class Model_KD3:
    def __init__(self, params: KubinParams):
        k_log("activating pipeline: native (3.0)")

        self.params = params
        self.t2i_pipe: Kandinsky3T2IPipeline | Kandinsky3T2IOptimizedPipeline | None = (
            None
        )
        self.inpainting_pipe: (
            Kandinsky3InpaintingPipeline | Kandinsky3InpaintingOptimizedPipeline | None
        ) = None

    def prepare_model(self, task):
        k_log(f"task queued: {task}")
        assert task in ["text2img", "inpainting"]

        cache_dir = self.params("general", "cache_dir")
        device = self.params("general", "device")
        text_encoder_path = self.params("native", "text_encoder")
        environment = Model_KD3_Environment().from_config(self.params)

        if task == "text2img":
            if self.t2i_pipe is not None:
                return
            else:
                self.flush(task)

                self.t2i_pipe = get_T2I_pipeline(
                    device=device,
                    environment=environment,
                    cache_dir=cache_dir,
                    movq_path=None,
                    text_encoder_path=text_encoder_path,
                    unet_path=None,
                    fp16=True,
                )

        if task == "inpainting":
            if self.inpainting_pipe is not None:
                return
            else:
                self.flush(task)

                self.inpainting_pipe = get_inpainting_pipeline(
                    device=device,
                    environment=environment,
                    cache_dir=cache_dir,
                    movq_path=None,
                    text_encoder_path=text_encoder_path,
                    unet_path=None,
                    fp16=False,
                )

    def create_batch_images(self, params, task, batch):
        params["task"] = task

        output_dir = params.get(
            ".output_dir",
            os.path.join(self.params("general", "output_dir"), task),
        )

        saved_batch = save_output(output_dir, batch, params)
        return saved_batch

    def t2i(self, params):
        task = "text2img"

        self.prepare_model(task)

        images = []
        batch = self.t2i_pipe(
            text=params["prompt"],
            negative_text=params["negative_prompt"],
            images_num=params["batch_count"],
            bs=params["batch_size"],
            width=params["w"],
            height=params["h"],
            guidance_scale=params["guidance_scale"],
            steps=params["num_steps"],
        )

        images += self.create_batch_images(params, "text2img", batch)
        k_log("text2img task: done")

        return images

    def i2i(self, params):
        task = "img2img"
        return []

    def mix(self, params):
        task = "mix"
        return []

    def inpaint(self, params):
        task = "inpainting"

        self.prepare_model(task)

        image_mask = params["image_mask"]
        pil_img = image_mask["image"]
        output_size = (pil_img.width, pil_img.height)
        mask = image_mask["mask"]

        inpaint_region = params["region"]
        inpaint_target = params["target"]

        image, mask = create_inpaint_targets(
            pil_img, mask, output_size, inpaint_region, inpaint_target
        )

        images = []
        batch = self.inpainting_pipe(
            text=params["prompt"],
            image=image,
            mask=mask,
            negative_text=params["negative_prompt"],
            images_num=params["batch_count"],
            bs=params["batch_size"],
            guidance_weight_text=params["guidance_scale"],
            steps=params["num_steps"],
        )

        images += self.create_batch_images(params, "text2img", batch)
        k_log("inpainting task: done")

        return images

    def outpaint(self, params):
        task = "outpainting"
        return []

    def flush(self, task=None):
        environment = Model_KD3_Environment().from_config(self.params)

        if not environment.kd30_low_vram:
            cleared = False
            if task == "text2img" or task is None:
                if self.t2i_pipe is not None:
                    k_log(f"moving t2i_pipe to cpu in order to release memory")

                    self.t2i_pipe.t5_encoder.to("cpu")
                    self.t2i_pipe.unet.to("cpu")
                    self.t2i_pipe.movq.to("cpu")

                    self.t2i_pipe = None
                    cleared = True

            elif task == "inpainting" or task is None:
                if self.inpainting_pipe is not None:
                    k_log(f"moving inpainting_pipe to cpu in order to release memory")

                    self.inpainting_pipe.t5_encoder.to("cpu")
                    self.inpainting_pipe.unet.to("cpu")
                    self.inpainting_pipe.movq.to("cpu")

                    self.inpainting_pipe = None
                    cleared = True
        else:
            cleared = True

        if cleared:
            gc.collect()
            device = self.params("general", "device")
            if device.startswith("cuda"):
                if torch.cuda.is_available():
                    with torch.cuda.device(device):
                        torch.cuda.empty_cache()
