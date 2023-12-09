import gc
import torch
import torch.backends
import torch

from huggingface_hub import hf_hub_download
from models.model_30.kandinsky3 import (
    get_T2I_pipeline,
    get_inpainting_pipeline,
    get_T2I_unet,
    get_inpainting_unet,
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
from utils.image import create_inpaint_targets
from utils.logging import k_log

from model_utils.diffusers_samplers import use_sampler


class Model_KD3:
    def __init__(self, params: KubinParams):
        k_log("activating pipeline: native (3.0)")
        patch_kd30_pipelines(params)

        self.params = params

        self.text_encoder: Kandinsky3T2IPipeline | None = None
        self.t2i_pipe: Kandinsky3T2IPipeline | None = None
        self.inpainting_pipe: Kandinsky3InpaintingPipeline | None = None

    def prepare_model(self, task):
        k_log(f"task queued: {task}")
        assert task in ["text2img", "inpainting"]

        cache_dir = self.params("general", "cache_dir")
        device = self.params("general", "device")

        if task == "text2img":
            if self.t2i_pipe is not None:
                return
            else:
                self.flush(task)

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
                    projections_state_dict=projections_state_dict,
                    low_cpu_mem_usage=True,
                    fp16=True,
                    device_map="auto",
                )

                movq = get_movq(device, movq_path, fp16=True)
                self.t2i_pipe = Kandinsky3T2IPipeline(
                    device,
                    unet,
                    null_embedding,
                    processor,
                    condition_encoders,
                    movq,
                    fp16=True,
                )

        if task == "inpainting":
            if self.inpaint_pipe is not None:
                return
            else:
                self.flush(task)

                unet_inpainting_path = hf_hub_download(
                    repo_id="ai-forever/Kandinsky3.0",
                    filename="weights/kandinsky3_inpainting.pt",
                    cache_dir=cache_dir,
                )

                movq_path = hf_hub_download(
                    repo_id="ai-forever/Kandinsky3.0",
                    filename="weights/movq.pt",
                    cache_dir=cache_dir,
                )

                unet, null_embedding, projections_state_dict = get_inpainting_unet(
                    device, unet_inpainting_path, fp16=True
                )

                processor, condition_encoders = get_T5encoder(
                    device,
                    weights_path="google/flan-ul2",
                    projections_state_dict=projections_state_dict,
                    low_cpu_mem_usage=True,
                    fp16=True,
                    device_map="auto",
                )

                movq = get_movq(device, movq_path, fp16=False)
                self.inpainting_pipe = Kandinsky3InpaintingPipeline(
                    device,
                    unet,
                    null_embedding,
                    processor,
                    condition_encoders,
                    movq,
                    fp16=True,
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
        cleared = False
        if task == "text2img" or task is None:
            if self.t2i_pipe is not None:
                k_log(f"moving t2i_pipe to cpu in order to release memory")
                self.t2i_pipe.to("cpu")
                self.t2i_pipe = None
                cleared = True

        elif task == "inpainting" or task is None:
            if self.inpainting_pipe is not None:
                k_log(f"moving inpainting_pipe to cpu in order to release memory")
                self.inpainting_pipe.to("cpu")
                self.inpainting_pipe = None
                cleared = True

        if cleared:
            gc.collect()
            device = self.params("general", "device")
            if device.startswith("cuda"):
                if torch.cuda.is_available():
                    with torch.cuda.device(device):
                        torch.cuda.empty_cache()
