import gc
from models.model_31.kandinsky31.t2i_lowvram_pipeline import (
    Kandinsky3T2ILowVRAMPipeline,
)
import torch
import torch.backends
import torch

from models.model_31.kandinsky31 import (
    get_T2I_Flash_pipeline,
    get_T2I_pipeline,
    get_inpainting_pipeline,
)
import os
from models.model_31.kandinsky31.inpainting_pipeline import Kandinsky3InpaintingPipeline
from models.model_31.kandinsky31.t2i_pipeline import Kandinsky3T2IPipeline
from models.model_31.model_kd31_env import Model_KD31_Environment
from params import KubinParams
from utils.file_system import save_output
from utils.image import composite_images, create_inpaint_targets
from utils.logging import k_log


class Model_KD31:
    def __init__(self, params: KubinParams):
        k_log("using pipeline: native (3.1)")

        self.params = params

        self.use_flash_pipeline = self.params("native", "use_kandinsky31_flash")
        self.t2i_pipe: Kandinsky3T2IPipeline | Kandinsky3T2ILowVRAMPipeline | None = (
            None
        )
        self.inpainting_pipe: Kandinsky3InpaintingPipeline | None = None

    def prepare_model(self, task):
        k_log(f"task queued: {task}")
        assert task in ["text2img", "inpainting"]

        cache_dir = self.params("general", "cache_dir")
        device = self.params("general", "device")

        use_flash_pipeline_before = self.use_flash_pipeline
        self.use_flash_pipeline = self.params("native", "use_kandinsky31_flash")

        text_encoder_path = self.params("native", "text_encoder")
        if text_encoder_path == "default":
            text_encoder_path = None

        environment = Model_KD31_Environment().from_config(self.params)

        if task == "text2img":
            if (
                self.t2i_pipe is None
                or use_flash_pipeline_before != self.use_flash_pipeline
            ):
                self.flush(task)

                if self.use_flash_pipeline:
                    k_log(f"preparing flash K3 pipeline")

                    self.t2i_pipe = get_T2I_Flash_pipeline(
                        environment=environment,
                        device_map=torch.device(device),
                        dtype_map={
                            "unet": torch.float32,
                            "text_encoder": torch.float16,
                            "movq": torch.float32,
                        },
                        low_cpu_mem_usage=True,
                        load_in_8bit=False,
                        load_in_4bit=False,
                        cache_dir=cache_dir,
                        unet_path=None,
                        text_encoder_path=text_encoder_path,
                        movq_path=None,
                    )
                else:
                    k_log(f"preparing regular K3 pipeline")

                    self.t2i_pipe = get_T2I_pipeline(
                        environment=environment,
                        device_map=torch.device(device),
                        dtype_map={
                            "unet": torch.float32,
                            "text_encoder": torch.float16,
                            "movq": torch.float32,
                        },
                        low_cpu_mem_usage=True,
                        load_in_8bit=False,
                        load_in_4bit=False,
                        cache_dir=cache_dir,
                        unet_path=None,
                        text_encoder_path=text_encoder_path,
                        movq_path=None,
                    )

        elif task == "inpainting":
            if self.inpainting_pipe is None:
                self.flush(task)

                self.inpainting_pipe = get_inpainting_pipeline(
                    environment=environment,
                    device_map=torch.device(device),
                    dtype_map={
                        "unet": torch.float32,
                        "text_encoder": torch.float16,
                        "movq": torch.float32,
                    },
                    low_cpu_mem_usage=True,
                    load_in_8bit=False,
                    load_in_4bit=False,
                    cache_dir=cache_dir,
                    unet_path=None,
                    text_encoder_path=text_encoder_path,
                    movq_path=None,
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
            negative_text=None,  # TODO: when using params["negative_prompt"], error is raised
            images_num=params["batch_count"],
            bs=params["batch_size"],
            guidance_weight_text=params["guidance_scale"],
            steps=params["num_steps"],
        )

        if inpaint_region == "mask":
            batch_composed = []
            for inpainted_image in batch:
                merged_image = composite_images(pil_img, inpainted_image, mask)
                batch_composed.append(merged_image)
            batch = batch_composed

        images += self.create_batch_images(params, "text2img", batch)
        k_log("inpainting task: done")

        return images

    def outpaint(self, params):
        task = "outpainting"
        return []

    def flush(self, task=None):
        environment = Model_KD31_Environment().from_config(self.params)
        cleared = False

        if task == "inpainting" or task is None:
            if self.t2i_pipe is not None:
                k_log(f"t2i_pipe -> cpu")

                if self.t2i_pipe.t5_encoder is not None:
                    self.t2i_pipe.t5_encoder.to("cpu")

                if self.t2i_pipe.unet is not None:
                    self.t2i_pipe.unet.to("cpu")

                if self.t2i_pipe.movq is not None:
                    self.t2i_pipe.movq.to("cpu")

                self.t2i_pipe = None
                cleared = True

        elif task == "text2img" or task is None:
            if self.inpainting_pipe is not None:
                k_log(f"inpainting_pipe -> cpu")

                if self.inpainting_pipe.t5_encoder is not None:
                    self.inpainting_pipe.t5_encoder.to("cpu")

                if self.inpainting_pipe.unet is not None:
                    self.inpainting_pipe.unet.to("cpu")

                if self.inpainting_pipe.movq is not None:
                    self.inpainting_pipe.movq.to("cpu")

                self.inpainting_pipe = None
                cleared = True

        if cleared:
            gc.collect()
            device = self.params("general", "device")
            if device.startswith("cuda"):
                if torch.cuda.is_available():
                    with torch.cuda.device(device):
                        torch.cuda.empty_cache()
