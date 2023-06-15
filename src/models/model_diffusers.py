import gc
import torch
import torch
import torch.backends

try:
    from diffusers import (
        KandinskyPipeline,
        KandinskyImg2ImgPipeline,
        KandinskyPriorPipeline,
        KandinskyInpaintPipeline,
    )
except:
    print(
        "warning: seems like diffusers are not installed, run 'pip install -r diffusers/requirements.txt' to install"
    )
    print("warning: if you are not going to use diffusers, just ignore this message")

from diffusers.models.attention_processor import AttnAddedKVProcessor2_0
import itertools
import os
import secrets
from PIL import Image, ImageOps
import numpy as np
from params import KubinParams
from utils.file_system import save_output


class Model_Diffusers:
    def __init__(self, params: KubinParams):
        print("activating pipeline: diffusers")
        self.params = params

        self.pipe_prior: KandinskyPriorPipeline | None = None
        self.t2i_pipe: KandinskyPipeline | None = None
        self.i2i_pipe: KandinskyImg2ImgPipeline | None = None
        self.inpaint_pipe: KandinskyInpaintPipeline | None = None

    def prepare(self, task):
        print(f"task queued: {task}")
        assert task in ["text2img", "img2img", "mix", "inpainting", "outpainting"]

        clear_vram_on_switch = False

        if self.params("diffusers", "use_deterministic_algorithms"):
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            torch.use_deterministic_algorithms(True)

        if self.params("diffusers", "use_tf32_mode"):
            torch.backends.cuda.matmul.allow_tf32 = True

        cache_dir = self.params("general", "cache_dir")
        device = self.params("general", "device")

        if self.pipe_prior is None:
            self.pipe_prior = KandinskyPriorPipeline.from_pretrained(
                "kandinsky-community/kandinsky-2-1-prior",
                torch_dtype=torch.float16
                if self.params("diffusers", "half_precision_weights")
                else "auto",
                cache_dir=cache_dir,
            )

            self.pipe_prior.to(device)

            if self.params("diffusers", "sequential_cpu_offload"):
                self.pipe_prior.enable_sequential_cpu_offload()

        if task == "text2img" or task == "mix":
            if self.t2i_pipe is None:
                if clear_vram_on_switch:
                    self.flush()

                self.t2i_pipe = KandinskyPipeline.from_pretrained(
                    "kandinsky-community/kandinsky-2-1",
                    torch_dtype=torch.float16
                    if self.params("diffusers", "half_precision_weights")
                    else "auto",
                    cache_dir=cache_dir,
                )

                if self.params("diffusers", "enable_xformers"):
                    self.t2i_pipe.enable_xformers_memory_efficient_attention()
                if self.params("diffusers", "enable_sdpa_attention"):
                    self.t2i_pipe.unet.set_attn_processor(AttnAddedKVProcessor2_0())
                if self.params("diffusers", "enable_sliced_attention"):
                    self.t2i_pipe.enable_attention_slicing()
                if self.params("diffusers", "channels_last_memory"):
                    self.t2i_pipe.unet.to(memory_format=torch.channels_last)
                if self.params("diffusers", "torch_code_compilation"):
                    self.t2i_pipe.unet = torch.compile(
                        self.t2i_pipe.unet, mode="reduce-overhead", fullgraph=True
                    )

                self.t2i_pipe.to(device)

                if self.params("diffusers", "sequential_cpu_offload"):
                    self.t2i_pipe.enable_sequential_cpu_offload()
                if self.params("diffusers", "full_model_offload"):
                    self.t2i_pipe.enable_model_cpu_offload()

        elif task == "img2img":
            if self.i2i_pipe is None:
                if clear_vram_on_switch:
                    self.flush()

                self.i2i_pipe = KandinskyImg2ImgPipeline.from_pretrained(
                    "kandinsky-community/kandinsky-2-1",
                    torch_dtype=torch.float16
                    if self.params("diffusers", "half_precision_weights")
                    else "auto",
                    cache_dir=cache_dir,
                )

                if self.params("diffusers", "enable_xformers"):
                    self.i2i_pipe.enable_xformers_memory_efficient_attention()
                if self.params("diffusers", "enable_sdpa_attention"):
                    self.i2i_pipe.unet.set_attn_processor(AttnAddedKVProcessor2_0())
                if self.params("diffusers", "enable_sliced_attention"):
                    self.i2i_pipe.enable_attention_slicing()
                if self.params("diffusers", "channels_last_memory"):
                    self.i2i_pipe.unet.to(memory_format=torch.channels_last)
                if self.params("diffusers", "torch_code_compilation"):
                    self.i2i_pipe.unet = torch.compile(
                        self.i2i_pipe.unet, mode="reduce-overhead", fullgraph=True
                    )

                self.i2i_pipe.to(device)

                if self.params("diffusers", "sequential_cpu_offload"):
                    self.i2i_pipe.enable_sequential_cpu_offload()
                if self.params("diffusers", "full_model_offload"):
                    self.i2i_pipe.enable_model_cpu_offload()

        elif task == "inpainting" or task == "outpainting":
            if self.inpaint_pipe is None:
                if clear_vram_on_switch:
                    self.flush()

                self.inpaint_pipe = KandinskyInpaintPipeline.from_pretrained(
                    "kandinsky-community/kandinsky-2-1-inpaint",
                    torch_dtype=torch.float16
                    if self.params("diffusers", "half_precision_weights")
                    else "auto",
                    cache_dir=cache_dir,
                )

                if self.params("diffusers", "enable_xformers"):
                    self.inpaint_pipe.enable_xformers_memory_efficient_attention()
                if self.params("diffusers", "enable_sdpa_attention"):
                    self.inpaint_pipe.unet.set_attn_processor(AttnAddedKVProcessor2_0())
                if self.params("diffusers", "enable_sliced_attention"):
                    self.inpaint_pipe.enable_attention_slicing()
                if self.params("diffusers", "channels_last_memory"):
                    self.inpaint_pipe.unet.to(memory_format=torch.channels_last)
                if self.params("diffusers", "torch_code_compilation"):
                    self.inpaint_pipe.unet = torch.compile(
                        self.inpaint_pipe.unet, mode="reduce-overhead", fullgraph=True
                    )

                self.inpaint_pipe.to(device)

                if self.params("diffusers", "sequential_cpu_offload"):
                    self.inpaint_pipe.enable_sequential_cpu_offload()
                if self.params("diffusers", "full_model_offload"):
                    self.inpaint_pipe.enable_model_cpu_offload()

        return self

    def flush(self, target=None):
        print(f"clearing memory")
        offload_enabled = self.params("diffusers", "sequential_cpu_offload")

        if self.pipe_prior is not None:
            if not offload_enabled:
                self.pipe_prior.to("cpu")
            self.pipe_prior = None

        if target is None or target in ["text2img", "mix"]:
            if self.t2i_pipe is not None:
                if not offload_enabled:
                    self.t2i_pipe.to("cpu")
                self.t2i_pipe = None

        if target is None or target in ["img2img"]:
            if self.i2i_pipe is not None:
                if not offload_enabled:
                    self.i2i_pipe.to("cpu")
                self.i2i_pipe = None

        if target is None or target in ["inpainting", "outpainting"]:
            if self.inpaint_pipe is not None:
                if not offload_enabled:
                    self.inpaint_pipe.to("cpu")
                self.inpaint_pipe = None

        gc.collect()

        if self.params("general", "device") == "cuda":
            if torch.cuda.is_available():
                with torch.cuda.device("cuda"):
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()

    def seed(self, params):
        input_seed = params["input_seed"]
        seed = secrets.randbelow(99999999999) if input_seed == -1 else input_seed

        print(f"seed generated: {seed}")
        params["input_seed"] = seed
        return params

    def t2i(self, params):
        params = self.prepare("text2img").seed(params)
        generator = torch.Generator(device="cuda").manual_seed(params["input_seed"])

        image_embeds, negative_image_embeds = self.pipe_prior(
            prompt=params["prompt"],
            negative_prompt=params["negative_prior_prompt"],
            num_images_per_prompt=1,
            num_inference_steps=params["prior_steps"],
            latents=None,
            guidance_scale=params["prior_cf_scale"],
            output_type="pt",
            generator=generator,
            return_dict=True,
        ).to_tuple()

        images = []
        for _ in itertools.repeat(None, params["batch_count"]):
            current_batch = self.t2i_pipe(
                prompt=params["prompt"],
                image_embeds=image_embeds,
                negative_image_embeds=negative_image_embeds,
                negative_prompt=params["negative_decoder_prompt"],
                width=params["w"],
                height=params["h"],
                num_inference_steps=params["num_steps"],
                guidance_scale=params["guidance_scale"],
                num_images_per_prompt=params["batch_size"],
                generator=generator,
                latents=None,
                output_type="pil",
                return_dict=True,
            ).images

            output_dir = params.get(
                ".output_dir",
                os.path.join(self.params("general", "output_dir"), "text2img"),
            )
            saved_batch = save_output(output_dir, current_batch, params)
            images = images + saved_batch
        return images

    def i2i(self, params):
        params = self.prepare("img2img").seed(params)
        generator = torch.Generator(device="cuda").manual_seed(params["input_seed"])

        image_embeds, negative_image_embeds = self.pipe_prior(
            prompt=params["prompt"],
            # TODO add negative prompt to UI
            # negative_prompt=params["negative_prior_prompt"],
            num_images_per_prompt=1,
            num_inference_steps=params["prior_steps"],
            latents=None,
            guidance_scale=params["prior_cf_scale"],
            output_type="pt",
            generator=generator,
            return_dict=True,
        ).to_tuple()

        images = []
        for _ in itertools.repeat(None, params["batch_count"]):
            current_batch = self.i2i_pipe(
                prompt=params["prompt"],
                image=params["init_image"],
                image_embeds=image_embeds,
                negative_image_embeds=negative_image_embeds,
                width=params["w"],
                height=params["h"],
                strength=1 - params["strength"],
                # TODO add negative prompt to UI
                # negative_prompt=params["negative_decoder_prompt"],
                num_inference_steps=params["num_steps"],
                guidance_scale=params["guidance_scale"],
                num_images_per_prompt=params["batch_size"],
                generator=generator,
                output_type="pil",
                return_dict=True,
            ).images

            output_dir = params.get(
                ".output_dir",
                os.path.join(self.params("general", "output_dir"), "img2img"),
            )
            saved_batch = save_output(output_dir, current_batch, params)
            images = images + saved_batch
        return images

    def mix(self, params):
        params = self.prepare("mix").seed(params)
        generator = torch.Generator(device="cuda").manual_seed(params["input_seed"])

        def images_or_texts(images, texts):
            images_texts = []
            for i in range(len(images)):
                images_texts.append(texts[i] if images[i] is None else images[i])

            return images_texts

        images_texts = images_or_texts(
            [params["image_1"], params["image_2"]],
            [params["text_1"], params["text_2"]],
        )
        weights = [params["weight_1"], params["weight_2"]]

        prompt = ""
        interpolation_params = self.pipe_prior.interpolate(
            images_and_prompts=images_texts,
            weights=weights,
            num_images_per_prompt=params["batch_size"],
            num_inference_steps=params["prior_steps"],
            generator=generator,
            latents=None,
            negative_prior_prompt=params["negative_prior_prompt"],
            negative_prompt=params["negative_decoder_prompt"],
            guidance_scale=params["prior_cf_scale"],
        )

        images = []
        prompt = ""
        for _ in itertools.repeat(None, params["batch_count"]):
            current_batch = self.t2i_pipe(
                prompt=prompt,
                **interpolation_params,
                width=params["w"],
                height=params["h"],
                num_inference_steps=params["num_steps"],
                guidance_scale=params["guidance_scale"],
                num_images_per_prompt=params["batch_size"],
                generator=generator,
                latents=None,
                output_type="pil",
                return_dict=True,
            ).images

            output_dir = params.get(
                ".output_dir", os.path.join(self.params("general", "output_dir"), "mix")
            )
            saved_batch = save_output(output_dir, current_batch, params)
            images = images + saved_batch
        return images

    def inpaint(self, params):
        params = self.prepare("inpainting").seed(params)
        generator = torch.Generator(device="cuda").manual_seed(params["input_seed"])

        prior_output = self.pipe_prior(
            prompt=params["prompt"],
            negative_prompt=params["negative_prior_prompt"],
            num_images_per_prompt=1,
            num_inference_steps=25,
            latents=None,
            guidance_scale=params["prior_cf_scale"],
            output_type="pt",
            generator=generator,
            return_dict=True,
        )

        output_size = (params["w"], params["h"])
        image_mask = params["image_mask"]
        pil_img = image_mask["image"].resize(output_size, resample=Image.LANCZOS)

        mask_img = image_mask["mask"].resize(output_size)
        mask_arr = np.array(mask_img.convert("L")).astype(np.float32) / 255.0

        if params["target"] == "only mask":
            mask_arr = 1.0 - mask_arr

        images = []
        for _ in itertools.repeat(None, params["batch_count"]):
            current_batch = self.inpaint_pipe(
                prompt=params["prompt"],
                image=pil_img,
                mask_image=mask_arr,
                **prior_output,
                width=params["w"],
                height=params["h"],
                num_inference_steps=params["num_steps"],
                guidance_scale=params["guidance_scale"],
                num_images_per_prompt=params["batch_size"],
                generator=generator,
                latents=None,
                output_type="pil",
                return_dict=True,
            ).images

            output_dir = params.get(
                ".output_dir",
                os.path.join(self.params("general", "output_dir"), "inpainting"),
            )
            saved_batch = save_output(output_dir, current_batch, params)
            images = images + saved_batch
        return images

    def outpaint(self, params):
        params = self.prepare("outpainting").seed(params)
        generator = torch.Generator(device="cuda").manual_seed(params["input_seed"])

        prior_output = self.pipe_prior(
            prompt=params["prompt"],
            negative_prompt=params["negative_prior_prompt"],
            num_images_per_prompt=1,
            num_inference_steps=25,
            latents=None,
            guidance_scale=params["prior_cf_scale"],
            output_type="pt",
            generator=generator,
            return_dict=True,
        ).to_tuple()

        image = params["image"]
        image_w, image_h = image.size

        offset = params["offset"]

        if offset is not None:
            top, right, bottom, left = offset
            inferred_mask_size = tuple(
                a + b for a, b in zip(image.size, (left + right, top + bottom))  # type: ignore
            )[::-1]
            mask = np.zeros(inferred_mask_size, dtype=np.float32)  # type: ignore
            mask[top : image_h + top, left : image_w + left] = 1
            image = ImageOps.expand(image, border=(left, top, right, bottom), fill=0)

        else:
            x1, y1, x2, y2 = image.getbbox()
            mask = np.ones((image_h, image_w), dtype=np.float32)
            mask[0:y1, :] = 0
            mask[:, 0:x1] = 0
            mask[y2:image_h, :] = 0
            mask[:, x2:image_w] = 0

        infer_size = params["infer_size"]
        if infer_size:
            height, width = mask.shape[:2]
        else:
            width = params["w"]
            height = params["h"]

        images = []
        for _ in itertools.repeat(None, params["batch_count"]):
            current_batch = self.inpaint_pipe(
                prompt=params["prompt"],
                image=image,
                mask_image=mask,
                **prior_output,
                width=width,
                height=height,
                num_inference_steps=params["num_steps"],
                guidance_scale=params["guidance_scale"],
                num_images_per_prompt=params["batch_size"],
                generator=generator,
                latents=None,
                output_type="pil",
                return_dict=True,
            ).images

            output_dir = params.get(
                ".output_dir",
                os.path.join(self.params("general", "output_dir"), "outpainting"),
            )
            saved_batch = save_output(output_dir, current_batch, params)
            images = images + saved_batch
        return images
