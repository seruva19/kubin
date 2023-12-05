import torch
import torch.backends
import torch

from huggingface_hub import hf_hub_download
from models.model_30.kandinsky3 import (
    get_T2I_pipeline,
    get_T2I_unet,
    get_T5encoder,
    get_movq,
    Kandinsky3T2IPipeline,
    Kandinsky3InpaintingPipeline,
)

import itertools
import os
import secrets
from models.model_30.model_kd30_patches import patch_kd30_pipelines
from params import KubinParams
from utils.file_system import save_output
from utils.logging import k_log

from model_utils.diffusers_samplers import use_sampler


class Model_KD3:
    def __init__(self, params: KubinParams):
        k_log("activating pipeline: native (3.0)")
        patch_kd30_pipelines(params)

        self.params = params

        self.text_encoder: Kandinsky3T2IPipeline | None = None
        self.t2i_pipe: Kandinsky3T2IPipeline | None = None
        self.inpaint_pipe: Kandinsky3InpaintingPipeline | None = None

    def flush(self, target=None):
        None

    def prepare_model(self, task):
        k_log(f"task queued: {task}")
        assert task in ["text2img", "inpainting"]

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

        cache_dir = self.params("general", "cache_dir")
        device = self.params("general", "device")

        unet_path = hf_hub_download(
            repo_id="ai-forever/Kandinsky3.0",
            filename="weights/kandinsky3.pt",
            cache_dir=cache_dir,
        )

        movq_path = hf_hub_download(
            repo_id="ai-forever/Kandinsky3.0",
            filename="weights/movq.pt",
            cache_dir=cache_dir,
        )

        unet, null_embedding, projections_state_dict = get_T2I_unet(
            device, unet_path, fp16=True
        )

        processor, condition_encoders = get_T5encoder(
            device,
            weights_path="google/flan-ul2",
            # weights_path="sigmareaver/flan-ul2-4bit-128g-gptq",
            # weights_path="google/flan-t5-small",
            # weights_path="google/flan-t5-base",
            projections_state_dict=projections_state_dict,
            low_cpu_mem_usage=True,
            fp16=True,
            device_map="sequential",
        )

        movq = get_movq(device, movq_path, fp16=True)
        self.t2i_pipe = Kandinsky3T2IPipeline(
            device, unet, null_embedding, processor, condition_encoders, movq, fp16=True
        )

        for _ in itertools.repeat(None, params["batch_count"]):
            current_batch = self.t2i_pipe(
                text=params["prompt"],
                negative_text=params["negative_prompt"],
                images_num=params["batch_size"],
                bs=1,
                width=params["w"],
                height=params["h"],
                guidance_scale=params["guidance_scale"],
                steps=params["num_steps"],
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
