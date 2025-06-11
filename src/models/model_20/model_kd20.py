import gc
import itertools
import os
import secrets
from PIL import Image, ImageOps
import numpy as np
import torch
import torch.backends

from params import KubinParams
from utils.file_system import save_output
from utils.image import composite_images
from utils.env_data import load_env_value


class Model_KD20:
    def __init__(self, params: KubinParams):
        from kandinsky2 import Kandinsky2

        print("using pipeline: native (2.0)")
        self.params = params

        self.kd20: Kandinsky2 | None = None
        self.kd20_inpaint: Kandinsky2 | None = None

    def prepare_model(self, task):
        from model_utils.kd20_utils import get_kandinsky2_0

        print(f"task queued: {task}")
        assert task in ["text2img", "img2img", "inpainting"]

        clear_vram_on_switch = True

        cache_dir = self.params("general", "cache_dir")
        cache_dir = load_env_value("KD20_CACHE_DIR", cache_dir)

        device = self.params("general", "device")

        if task == "text2img" or task == "img2img":
            if self.kd20 is None:
                if clear_vram_on_switch:
                    self.flush()

                self.kd20 = get_kandinsky2_0(
                    device,
                    task_type="text2img",
                    cache_dir=cache_dir,
                    use_auth_token=None,
                )

                self.kd20.model.to(device)

        elif task == "inpainting":
            if self.kd20_inpaint is None:
                if clear_vram_on_switch:
                    self.flush()

                self.kd20_inpaint = get_kandinsky2_0(
                    device,
                    task_type="inpainting",
                    cache_dir=cache_dir,
                    use_auth_token=None,
                )

                self.kd20_inpaint.model.to(device)

        return self

    def flush(self, target=None):
        print(f"clearing memory")

        if target is None or target in ["text2img", "img2img"]:
            if self.kd20 is not None:
                self.kd20.model.to("cpu")
                self.kd20 = None

        if target is None or target in ["inpainting"]:
            if self.kd20_inpaint is not None:
                self.kd20_inpaint.model.to("cpu")
                self.kd20_inpaint = None

        gc.collect()

        if self.params("general", "device") == "cuda":
            if torch.cuda.is_available():
                with torch.cuda.device("cuda"):
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()

    def prepare_params(self, params):
        input_seed = params["input_seed"]
        seed = secrets.randbelow(99999999999) if input_seed == -1 else input_seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        print(f"seed generated: {seed}")
        params["input_seed"] = seed
        params["model_name"] = "kd2.0"

        return params

    def t2i(self, params):
        params = self.prepare_model("text2img").prepare_params(params)
        assert self.kd20 is not None

        images = []
        for _ in itertools.repeat(None, params["batch_count"]):
            current_batch = self.kd20.generate_text2img(
                prompt=params["prompt"],
                num_steps=params["num_steps"],
                batch_size=params["batch_size"],
                guidance_scale=params["guidance_scale"],
                progress=True,
                dynamic_threshold_v=99.5,
                denoised_type="dynamic_threshold",
                h=params["h"],
                w=params["w"],
                sampler=params["sampler"],
                ddim_eta=0.05,
            )
            output_dir = params.get(
                ".output_dir",
                os.path.join(self.params("general", "output_dir"), "text2img"),
            )
            saved_batch = save_output(output_dir, current_batch, params)
            images = images + saved_batch
        return images

    def i2i(self, params):
        params = self.prepare_model("img2img").prepare_params(params)
        assert self.kd20 is not None

        output_size = (params["w"], params["h"])
        image = params["init_image"]

        images = []
        for _ in itertools.repeat(None, params["batch_count"]):
            current_batch = self.kd20.generate_img2img(
                prompt=params["prompt"],
                pil_img=image,
                strength=params["strength"],
                num_steps=params["num_steps"],
                guidance_scale=params["guidance_scale"],
                progress=True,
                dynamic_threshold_v=99.5,
                denoised_type="dynamic_threshold",
                sampler=params["sampler"],
                ddim_eta=0.05,
            )

            output_dir = params.get(
                ".output_dir",
                os.path.join(self.params("general", "output_dir"), "img2img"),
            )
            saved_batch = save_output(output_dir, current_batch, params)
            images = images + saved_batch
        return images

    def inpaint(self, params):
        params = self.prepare_model("inpainting").prepare_params(params)
        assert self.kd20_inpaint is not None

        inpaint_region = params["region"]
        output_size = (params["w"], params["h"])
        image_mask = params["image_mask"]
        pil_img = image_mask["image"].resize(output_size, resample=Image.LANCZOS)

        mask_img = image_mask["mask"].resize(output_size)
        mask_arr = np.array(mask_img.convert("L")).astype(np.float32) / 255.0

        if params["target"] == "only mask":
            mask_arr = 1.0 - mask_arr

        images = []
        for _ in itertools.repeat(None, params["batch_count"]):
            current_batch = self.kd20_inpaint.generate_inpainting(
                prompt=params["prompt"],
                pil_img=pil_img,
                img_mask=mask_arr,
                num_steps=params["num_steps"],
                guidance_scale=params["guidance_scale"],
                progress=True,
                dynamic_threshold_v=99.5,
                denoised_type="dynamic_threshold",
                sampler=params["sampler"],
                ddim_eta=0.05,
            )

            if inpaint_region == "mask":
                current_batch_composed = []
                for inpainted_image in current_batch:
                    merged_image = composite_images(pil_img, inpainted_image, mask_arr)
                    current_batch_composed.append(merged_image)
                current_batch = current_batch_composed

            output_dir = params.get(
                ".output_dir",
                os.path.join(self.params("general", "output_dir"), "inpainting"),
            )
            saved_batch = save_output(output_dir, current_batch, params)
            images = images + saved_batch
        return images
