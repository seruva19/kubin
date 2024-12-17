import torch
import torch.backends
from diffusers import (
    Kandinsky3Pipeline,
    Kandinsky3Img2ImgPipeline,
    AutoPipelineForText2Image,
    AutoPipelineForImage2Image,
)
import torch

import itertools
import os
import secrets
from models.model_diffusers30.model_30_init import (
    flush_if_required,
    prepare_autopipeline_for_task,
)
from params import KubinParams
from utils.file_system import save_output
from utils.logging import k_log

from model_utils.diffusers_samplers import use_sampler


class Model_Diffusers3:
    def __init__(self, params: KubinParams):
        k_log("activating using: diffusers (3.0)")
        self.params = params

        self.auto_t2i_pipe: AutoPipelineForText2Image | None = None
        self.auto_i2i_pipe: AutoPipelineForImage2Image | None = None

        self.t2i_pipe: Kandinsky3Pipeline | None = None
        self.i2i_pipe: Kandinsky3Img2ImgPipeline | None = None
        self.inpaint_pipe: None = None

        self.optimizations = []

    def prepare_model(self, task):
        k_log(f"task queued: {task}")
        assert task in ["text2img", "img2img"]

        pipe = prepare_autopipeline_for_task(self, task)
        return pipe

    def flush(self, target=None):
        flush_if_required(self, target)

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

        pipe = self.prepare_model(task)
        params, generator = self.prepare_params(params)

        images = []
        for _ in itertools.repeat(None, params["batch_count"]):
            current_batch = pipe(
                prompt=params["prompt"],
                num_inference_steps=params["num_steps"],
                # timesteps=None,
                guidance_scale=params["guidance_scale"],
                negative_prompt=params["negative_prompt"],
                num_images_per_prompt=params["batch_size"],
                height=params["h"],
                width=params["w"],
                # eta=0.0,
                generator=generator,
                prompt_embeds=None,
                negative_prompt_embeds=None,
                # attention_mask=None,
                # negative_attention_mask=None,
                output_type="pil",
                return_dict=True,
                callback=None,
                callback_steps=1,
                # clean_caption=True,
                # cross_attention_kwargs=None,
                latents=None,
                # cut_context=True,
            ).images

            images += self.create_batch_images(params, "text2img", current_batch)
        k_log("text2img task: done")

        return images

    def i2i(self, params):
        task = "img2img"

        pipe = self.prepare_model(task)
        params, generator = self.prepare_params(params)

        images = []
        for _ in itertools.repeat(None, params["batch_count"]):
            current_batch = pipe(
                prompt=params["prompt"],
                image=params["init_image"],
                strength=params["strength"],
                num_inference_steps=params["num_steps"],
                guidance_scale=params["guidance_scale"],
                negative_prompt=params["negative_prompt"],
                num_images_per_prompt=params["batch_size"],
                # height=params["h"],
                # width=params["w"],
                generator=generator,
                prompt_embeds=None,
                negative_prompt_embeds=None,
                # attention_mask=None,
                # negative_attention_mask=None,
                output_type="pil",
                return_dict=True,
                callback=None,
                callback_steps=1,
                # callback_on_step_end=None
                # callback_on_step_end_tensor_inputs=None
                latents=None,
            ).images

            images += self.create_batch_images(params, "img2img", current_batch)
        k_log("img2img task: done")

        return images

    def mix(self, params):
        task = "mix"
        return []

    def inpaint(self, params):
        task = "mix"
        return []

    def outpaint(self, params):
        task = "outpainting"
        return []
