import torch
import torch.backends
from diffusers import KandinskyV3Pipeline, KandinskyV3Img2ImgPipeline
import torch

from models.model_diffusers3.model_3_init import (
    flush_if_required_3,
    prepare_weights_for_task_3,
)

import itertools
import os
import secrets
from params import KubinParams
from utils.file_system import save_output
from utils.logging import k_log

from model_utils.diffusers_samplers import use_sampler


class Model_Diffusers3:
    def __init__(self, params: KubinParams):
        k_log("activating pipeline: diffusers (3)")
        self.params = params

        self.t2i_pipe: KandinskyV3Pipeline | None = None
        self.i2i_pipe: KandinskyV3Img2ImgPipeline | None = None
        self.inpaint_pipe: None = None

    def prepare_model(self, task):
        k_log(f"task queued: {task}")
        assert task in ["text2img", "img2img"]

        prepare_weights_for_task_3(self, task)

    def flush(self, target=None):
        flush_if_required_3(self, target)

    def prepare_params(self, params):
        input_seed = params["input_seed"]
        seed = secrets.randbelow(99999999999) if input_seed == -1 else input_seed

        k_log(f"seed generated: {seed}")
        params["input_seed"] = seed
        params["model_name"] = "diffusers3"

        generator = torch.Generator(
            device=self.params("general", "device")
        ).manual_seed(params["input_seed"])

        return params, generator

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
        params, generator = self.prepare_params(params)

        for _ in itertools.repeat(None, params["batch_count"]):
            current_batch = self.t2i_pipe(
                prompt=params["prompt"],
                negative_prompt=params["negative_prompt"],
                num_inference_steps=params["num_steps"],
                timesteps=None,
                guidance_scale=params["guidance_scale"],
                num_images_per_prompt=params["batch_size"],
                height=params["h"],
                width=params["w"],
                eta=0.0,
                generator=generator,
                prompt_embeds=None,
                negative_prompt_embeds=None,
                output_type="pil",
                return_dict=True,
                callback=None,
                callback_steps=1,
                latents=None,
                cut_context=True,
            ).images

            images += self.create_batch_images(params, "text2img", current_batch)
        k_log("text2img task: done")

        return images

    def i2i(self, params):
        task = "img2img"
        return []

    def mix(self, params):
        task = "mix"
        return []

    def inpaint(self, params):
        task = "mix"
        return []

    def outpaint(self, params):
        task = "outpainting"
        return []
