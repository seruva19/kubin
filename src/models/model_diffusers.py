import gc
import torch
import torch
import torch.backends

from utils.image import create_inpaint_targets, create_outpaint_targets

try:
    from model_utils.diffusers_utils import use_scheduler
    from diffusers import (
        KandinskyPipeline,
        KandinskyImg2ImgPipeline,
        KandinskyPriorPipeline,
        KandinskyInpaintPipeline,
    )
    from diffusers.models.attention_processor import AttnAddedKVProcessor2_0
except:
    print(
        "warning: seems like diffusers are not installed, run 'pip install -r diffusers/requirements.txt' to install"
    )
    print("warning: if you are not going to use diffusers, just ignore this message")

import itertools
import os
import secrets
from params import KubinParams
from utils.file_system import save_output


class Model_Diffusers:
    def __init__(self, params: KubinParams):
        print("activating pipeline: diffusers (2.1)")
        self.params = params

        self.pipe_prior: KandinskyPriorPipeline | None = None
        self.t2i_pipe: KandinskyPipeline | None = None
        self.i2i_pipe: KandinskyImg2ImgPipeline | None = None
        self.inpaint_pipe: KandinskyInpaintPipeline | None = None

        self.current_pipe = None
        self.cublas_config = os.environ.get("CUBLAS_WORKSPACE_CONFIG", None)

    def prepareModel(self, task):
        print(f"task queued: {task}")
        assert task in ["text2img", "img2img", "mix", "inpainting", "outpainting"]

        clear_vram_on_switch = False

        if self.params("diffusers", "use_deterministic_algorithms"):
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            torch.use_deterministic_algorithms(True)
        else:
            if self.cublas_config is not None:
                os.environ["CUBLAS_WORKSPACE_CONFIG"] = self.cublas_config
            torch.use_deterministic_algorithms(False)

        torch.backends.cuda.matmul.allow_tf32 = self.params(
            "diffusers", "use_tf32_mode"
        )

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

            self.pipe_prior.safety_checker = None
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
                self.current_pipe = self.t2i_pipe

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
                self.current_pipe = self.i2i_pipe

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

                self.current_pipe = self.inpaint_pipe

        if self.params("diffusers", "enable_xformers"):
            self.current_pipe.enable_xformers_memory_efficient_attention()
        else:
            self.current_pipe.disable_xformers_memory_efficient_attention()

        if self.params("diffusers", "enable_sdp_attention"):
            self.current_pipe.unet.set_attn_processor(AttnAddedKVProcessor2_0())

        if self.params("diffusers", "channels_last_memory"):
            self.current_pipe.unet.to(memory_format=torch.channels_last)

        if self.params("diffusers", "torch_code_compilation"):
            self.current_pipe.unet = torch.compile(
                self.current_pipe.unet, mode="reduce-overhead", fullgraph=True
            )

        self.current_pipe.to(device)

        if self.params("diffusers", "sequential_cpu_offload"):
            self.current_pipe.enable_sequential_cpu_offload()

        if self.params("diffusers", "full_model_offload"):
            self.current_pipe.enable_model_cpu_offload()

        if self.params("diffusers", "enable_sliced_attention"):
            self.current_pipe.enable_attention_slicing()
        else:
            self.current_pipe.disable_attention_slicing()

        self.current_pipe.safety_checker = None
        return self.current_pipe

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

    def prepareParams(self, params):
        input_seed = params["input_seed"]
        seed = secrets.randbelow(99999999999) if input_seed == -1 else input_seed

        print(f"seed generated: {seed}")
        params["input_seed"] = seed
        params["model_name"] = "diffusers2.1"

        return params

    def t2i(self, params):
        unet_pipe = self.prepareModel("text2img")
        params = self.prepareParams(params)

        generator = torch.Generator(
            device=self.params("general", "device")
        ).manual_seed(params["input_seed"])

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

        use_scheduler(unet_pipe, params["sampler"])

        images = []
        for _ in itertools.repeat(None, params["batch_count"]):
            current_batch = self.t2i_pipe(
                # progress_bar=params["_progress_bar"],
                prompt=params["prompt"],
                image_embeds=image_embeds,
                negative_prompt=params["negative_prompt"],
                negative_image_embeds=negative_image_embeds,
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
        unet_pipe = self.prepareModel("img2img")
        params = self.prepareParams(params)

        generator = torch.Generator(
            device=self.params("general", "device")
        ).manual_seed(params["input_seed"])

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

        use_scheduler(unet_pipe, params["sampler"])

        images = []
        for _ in itertools.repeat(None, params["batch_count"]):
            current_batch = self.i2i_pipe(
                prompt=params["prompt"],
                image=params["init_image"],
                image_embeds=image_embeds,
                negative_image_embeds=negative_image_embeds,
                width=params["w"],
                height=params["h"],
                strength=params["strength"],
                # TODO add negative prompt to UI
                # negative_prompt=params["negative_prompt"],
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
        unet_pipe = self.prepareModel("mix")
        params = self.prepareParams(params)

        generator = torch.Generator(
            device=self.params("general", "device")
        ).manual_seed(params["input_seed"])

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

        use_scheduler(unet_pipe, params["sampler"])

        prompt = ""
        interpolation_params = self.pipe_prior.interpolate(
            images_and_prompts=images_texts,
            weights=weights,
            num_images_per_prompt=params["batch_size"],
            num_inference_steps=params["prior_steps"],
            generator=generator,
            latents=None,
            negative_prior_prompt=params["negative_prior_prompt"],
            negative_prompt=params["negative_prompt"],
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
        unet_pipe = self.prepareModel("inpainting")
        params = self.prepareParams(params)

        generator = torch.Generator(
            device=self.params("general", "device")
        ).manual_seed(params["input_seed"])

        prior_output = self.pipe_prior(
            prompt=params["prompt"],
            negative_prompt=params["negative_prior_prompt"],
            num_images_per_prompt=1,
            num_inference_steps=params["prior_steps"],
            latents=None,
            guidance_scale=params["prior_cf_scale"],
            output_type="pt",
            generator=generator,
            return_dict=True,
        )

        image_mask = params["image_mask"]

        pil_img = image_mask["image"]
        width, height = (
            pil_img.width if params["infer_size"] else params["w"],
            pil_img.height if params["infer_size"] else params["h"],
        )
        output_size = (width, height)
        mask = image_mask["mask"]
        inpaint_region = params["region"]
        inpaint_target = params["target"]

        image, mask = create_inpaint_targets(
            pil_img, mask, output_size, inpaint_region, inpaint_target
        )

        use_scheduler(unet_pipe, params["sampler"])

        images = []
        for _ in itertools.repeat(None, params["batch_count"]):
            current_batch = self.inpaint_pipe(
                prompt=params["prompt"],
                image=image,
                mask_image=mask,
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
        unet_pipe = self.prepareModel("outpainting")
        params = self.prepareParams(params)

        generator = torch.Generator(
            device=self.params("general", "device")
        ).manual_seed(params["input_seed"])

        prior_output = self.pipe_prior(
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

        image = params["image"]
        offset = params["offset"]
        infer_size = params["infer_size"]
        width = params["w"]
        height = params["h"]

        image, mask, width, height = create_outpaint_targets(
            image, offset, infer_size, width, height
        )

        use_scheduler(unet_pipe, params["sampler"])

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
