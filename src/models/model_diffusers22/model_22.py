import torch
import torch.backends
from utils.image import create_inpaint_targets, create_outpaint_targets
import itertools
import os
import secrets
from params import KubinParams
from utils.file_system import save_output
from utils.logging import k_log

try:
    from model_utils.diffusers_utils import use_scheduler
    from models.model_diffusers22.model_22_cnet import generate_hint
    from models.model_diffusers22.model_22_utils import (
        flush_if_required,
        prepare_weights_for_task,
    )
    from transformers import CLIPVisionModelWithProjection
    from diffusers.models import UNet2DConditionModel
    from diffusers import (
        KandinskyV22PriorPipeline,
        KandinskyV22PriorEmb2EmbPipeline,
        KandinskyV22Pipeline,
        KandinskyV22Img2ImgPipeline,
        KandinskyV22InpaintPipeline,
        KandinskyV22ControlnetPipeline,
        KandinskyV22ControlnetImg2ImgPipeline,
    )
    from diffusers.models.attention_processor import AttnAddedKVProcessor2_0
except:
    k_log(
        "warning: seems like diffusers are not installed, run 'pip install -r diffusers/requirements.txt' to install"
    )
    k_log("warning: if you are not going to use diffusers, just ignore this message")


class Model_Diffusers22:
    def __init__(self, params: KubinParams):
        k_log("activating pipeline: diffusers (2.2)")
        self.params = params

        self.image_encoder: CLIPVisionModelWithProjection | None = None
        self.unet_2d: UNet2DConditionModel | None = None

        self.pipe_prior: KandinskyV22PriorPipeline | None = None
        self.pipe_prior_e2e: KandinskyV22PriorEmb2EmbPipeline | None = None
        self.t2i_pipe: KandinskyV22Pipeline | None = None
        self.i2i_pipe: KandinskyV22Img2ImgPipeline | None = None
        self.inpaint_pipe: KandinskyV22InpaintPipeline | None = None
        self.cnet_t2i_pipe: KandinskyV22ControlnetPipeline | None = None
        self.cnet_i2i_pipe: KandinskyV22ControlnetImg2ImgPipeline | None = None

        self.cublas_config = os.environ.get("CUBLAS_WORKSPACE_CONFIG", None)

    def prepareModel(self, task):
        k_log(f"task queued: {task}")
        assert task in [
            "text2img",
            "text2img_cnet",
            "img2img",
            "img2img_cnet",
            "mix",
            "inpainting",
            "outpainting",
        ]

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

        prior, decoder = prepare_weights_for_task(self, task)
        return (prior, decoder)

    def flush(self, target=None):
        flush_if_required(self, target)

    def prepareParams(self, params):
        input_seed = params["input_seed"]
        seed = secrets.randbelow(99999999999) if input_seed == -1 else input_seed

        k_log(f"seed generated: {seed}")
        params["input_seed"] = seed
        params["model_name"] = "diffusers2.2"

        prior_on_cpu = self.params("diffusers", "run_prior_on_cpu")
        decoder_generator = torch.Generator(
            device=self.params("general", "device")
        ).manual_seed(params["input_seed"])

        prior_generator = (
            torch.Generator(device="cpu").manual_seed(params["input_seed"])
            if prior_on_cpu
            else decoder_generator
        )

        return params, prior_generator, decoder_generator

    def create_batch_images(self, params, mode, batch):
        output_dir = params.get(
            ".output_dir",
            os.path.join(self.params("general", "output_dir"), mode),
        )
        saved_batch = save_output(output_dir, batch, params)
        return saved_batch

    def t2i(self, params):
        if params["cnet_enable"]:
            if params["cnet_pipeline"] == "ControlNetPipeline":
                return self.t2i_cnet(params)
            elif params["cnet_pipeline"] == "ControlNetImg2ImgPipeline":
                return self.i2i_cnet(params)

        prior, decoder = self.prepareModel("text2img")
        assert isinstance(prior, KandinskyV22PriorPipeline)
        assert isinstance(decoder, KandinskyV22Pipeline)
        params, prior_generator, decoder_generator = self.prepareParams(params)

        image_embeds, zero_embeds = prior(
            prompt=params["prompt"],
            negative_prompt=params["negative_prior_prompt"]
            if params["negative_prior_prompt"] != ""
            else None,
            num_images_per_prompt=params["batch_size"],
            num_inference_steps=params["prior_steps"],
            latents=None,
            guidance_scale=params["prior_cf_scale"],
            output_type="pt",
            generator=prior_generator,
            return_dict=False,
        )
        k_log("prior embeddings: done")

        negative_image_embeds = (
            prior(
                prompt=params["negative_prompt"],
                negative_prompt=None,
                num_images_per_prompt=params["batch_size"],
                num_inference_steps=params["prior_steps"],
                latents=None,
                guidance_scale=params["prior_cf_scale"],
                output_type="pt",
                generator=prior_generator,
                return_dict=True,
            ).image_embeds
            if params["negative_prompt"] != ""
            else zero_embeds
        )
        k_log("negative prior embeddings: done")

        use_scheduler(decoder, params["sampler"])

        images = []
        prior_on_cpu = self.params("diffusers", "run_prior_on_cpu")
        for _ in itertools.repeat(None, params["batch_count"]):
            current_batch = decoder(
                image_embeds=image_embeds.half() if prior_on_cpu else image_embeds,
                negative_image_embeds=negative_image_embeds.half()
                if prior_on_cpu
                else negative_image_embeds,
                width=params["w"],
                height=params["h"],
                num_inference_steps=params["num_steps"],
                guidance_scale=params["guidance_scale"],
                num_images_per_prompt=params["batch_size"],
                generator=decoder_generator,
                latents=None,
                output_type="pil",
                return_dict=True,
            ).images

            images += self.create_batch_images(params, "text2img", current_batch)
        k_log("decoder images: done")
        return images

    def i2i(self, params):
        if params["cnet_enable"]:
            return self.i2i_cnet(params)

        prior, decoder = self.prepareModel("img2img")
        assert isinstance(prior, KandinskyV22PriorPipeline)
        assert isinstance(decoder, KandinskyV22Img2ImgPipeline)
        params, prior_generator, decoder_generator = self.prepareParams(params)

        image_embeds, negative_image_embeds = prior(
            prompt=params["prompt"],
            negative_prompt=params["negative_prior_prompt"]
            if params["negative_prior_prompt"] != ""
            else None,
            num_images_per_prompt=params["batch_size"],
            num_inference_steps=params["prior_steps"],
            latents=None,
            guidance_scale=params["prior_cf_scale"],
            output_type="pt",
            generator=prior_generator,
            return_dict=False,
        )
        k_log("prior embeddings: done")

        use_scheduler(decoder, params["sampler"])

        images = []
        prior_on_cpu = self.params("diffusers", "run_prior_on_cpu")
        for _ in itertools.repeat(None, params["batch_count"]):
            current_batch = decoder(
                image=params["init_image"],
                image_embeds=image_embeds.half() if prior_on_cpu else image_embeds,
                negative_image_embeds=negative_image_embeds.half()
                if prior_on_cpu
                else negative_image_embeds,
                width=params["w"],
                height=params["h"],
                strength=params["strength"],
                num_inference_steps=params["num_steps"],
                guidance_scale=params["guidance_scale"],
                num_images_per_prompt=params["batch_size"],
                generator=decoder_generator,
                output_type="pil",
                return_dict=True,
            ).images

            images += self.create_batch_images(params, "img2img", current_batch)
        k_log("decoder images: done")
        return images

    def mix(self, params):
        prior, decoder = self.prepareModel("mix")
        assert isinstance(prior, KandinskyV22PriorPipeline)
        assert isinstance(decoder, KandinskyV22Pipeline)
        params, prior_generator, decoder_generator = self.prepareParams(params)

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

        embeds = prior.interpolate(
            images_and_prompts=images_texts,
            weights=weights,
            num_images_per_prompt=params["batch_size"],
            num_inference_steps=params["prior_steps"],
            generator=prior_generator,
            latents=None,
            negative_prompt=params["negative_prompt"],
            negative_prior_prompt=params["negative_prior_prompt"],
            guidance_scale=params["prior_cf_scale"],
        )
        k_log("prior embeddings: done")

        use_scheduler(decoder, params["sampler"])

        images = []
        prior_on_cpu = self.params("diffusers", "run_prior_on_cpu")
        for _ in itertools.repeat(None, params["batch_count"]):
            current_batch = decoder(
                image_embeds=embeds.image_embeds.half()
                if prior_on_cpu
                else embeds.image_embeds,
                negative_image_embeds=embeds.negative_image_embeds.half()
                if prior_on_cpu
                else embeds.negative_image_embeds,
                width=params["w"],
                height=params["h"],
                num_inference_steps=params["num_steps"],
                guidance_scale=params["guidance_scale"],
                num_images_per_prompt=params["batch_size"],
                generator=decoder_generator,
                latents=None,
                output_type="pil",
                return_dict=True,
            ).images

            images += self.create_batch_images(params, "mix", current_batch)
        k_log("decoder images: done")
        return images

    def inpaint(self, params):
        prior, decoder = self.prepareModel("inpainting")
        assert isinstance(prior, KandinskyV22PriorPipeline)
        assert isinstance(decoder, KandinskyV22InpaintPipeline)
        params, prior_generator, decoder_generator = self.prepareParams(params)

        image_embeds, zero_embeds = prior(
            prompt=params["prompt"],
            negative_prompt=params["negative_prior_prompt"]
            if params["negative_prior_prompt"] != ""
            else None,
            num_images_per_prompt=params["batch_size"],
            num_inference_steps=params["prior_steps"],
            latents=None,
            guidance_scale=params["prior_cf_scale"],
            output_type="pt",
            generator=prior_generator,
            return_dict=False,
        )
        k_log("prior embeddings: done")

        negative_image_embeds = (
            prior(
                prompt=params["negative_prompt"],
                negative_prompt=None,
                num_images_per_prompt=params["batch_size"],
                num_inference_steps=params["prior_steps"],
                latents=None,
                guidance_scale=params["prior_cf_scale"],
                output_type="pt",
                generator=prior_generator,
                return_dict=True,
            ).image_embeds
            if params["negative_prompt"] != ""
            else zero_embeds
        )
        k_log("negative prior embeddings: done")

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

        use_scheduler(decoder, params["sampler"])

        images = []
        prior_on_cpu = self.params("diffusers", "run_prior_on_cpu")
        for _ in itertools.repeat(None, params["batch_count"]):
            current_batch = decoder(
                image=image,
                mask_image=mask,
                image_embeds=image_embeds.half() if prior_on_cpu else image_embeds,
                negative_image_embeds=negative_image_embeds.half()
                if prior_on_cpu
                else negative_image_embeds,
                width=params["w"],
                height=params["h"],
                num_inference_steps=params["num_steps"],
                guidance_scale=params["guidance_scale"],
                num_images_per_prompt=params["batch_size"],
                generator=decoder_generator,
                latents=None,
                output_type="pil",
                return_dict=True,
            ).images

            images += self.create_batch_images(params, "inpainting", current_batch)
        k_log("decoder images: done")
        return images

    def outpaint(self, params):
        prior, decoder = self.prepareModel("outpainting")
        assert isinstance(prior, KandinskyV22PriorPipeline)
        assert isinstance(decoder, KandinskyV22InpaintPipeline)
        params, prior_generator, decoder_generator = self.prepareParams(params)

        image_embeds, zero_embeds = prior(
            prompt=params["prompt"],
            negative_prompt=params["negative_prior_prompt"]
            if params["negative_prior_prompt"] != ""
            else None,
            num_images_per_prompt=params["batch_size"],
            num_inference_steps=params["prior_steps"],
            latents=None,
            guidance_scale=params["prior_cf_scale"],
            output_type="pt",
            generator=prior_generator,
            return_dict=False,
        )
        k_log("prior embeddings: done")

        negative_image_embeds = (
            prior(
                prompt=params["negative_prompt"],
                negative_prompt=None,
                num_images_per_prompt=params["batch_size"],
                num_inference_steps=params["prior_steps"],
                latents=None,
                guidance_scale=params["prior_cf_scale"],
                output_type="pt",
                generator=prior_generator,
                return_dict=True,
            ).image_embeds
            if params["negative_prompt"] != ""
            else zero_embeds
        )
        k_log("negative prior embeddings: done")

        image = params["image"]
        offset = params["offset"]
        infer_size = params["infer_size"]
        width = params["w"]
        height = params["h"]

        image, mask, width, height = create_outpaint_targets(
            image, offset, infer_size, width, height
        )

        use_scheduler(decoder, params["sampler"])

        images = []
        prior_on_cpu = self.params("diffusers", "run_prior_on_cpu")
        for _ in itertools.repeat(None, params["batch_count"]):
            current_batch = decoder(
                image=image,
                mask_image=mask,
                image_embeds=image_embeds.half() if prior_on_cpu else image_embeds,
                negative_image_embeds=negative_image_embeds.half()
                if prior_on_cpu
                else negative_image_embeds,
                width=params["w"],
                height=params["h"],
                num_inference_steps=params["num_steps"],
                guidance_scale=params["guidance_scale"],
                num_images_per_prompt=params["batch_size"],
                generator=decoder_generator,
                latents=None,
                output_type="pil",
                return_dict=True,
            ).images

            images += self.create_batch_images(params, "outpainting", current_batch)
        k_log("decoder images: done")
        return images

    def t2i_cnet(self, params):
        prior, decoder = self.prepareModel("text2img_cnet")
        assert isinstance(prior, KandinskyV22PriorPipeline)
        assert isinstance(decoder, KandinskyV22ControlnetPipeline)
        params, prior_generator, decoder_generator = self.prepareParams(params)

        cnet_image = params["cnet_image"]
        cnet_condition = params["cnet_condition"]

        image_embeds, zero_embeds = prior(
            prompt=params["prompt"],
            negative_prompt=params["negative_prior_prompt"]
            if params["negative_prior_prompt"] != ""
            else None,
            num_images_per_prompt=params["batch_size"],
            num_inference_steps=params["prior_steps"],
            latents=None,
            guidance_scale=params["prior_cf_scale"],
            output_type="pt",
            generator=prior_generator,
            return_dict=False,
        )
        k_log("prior embeddings: done")

        negative_image_embeds = (
            prior(
                prompt=params["negative_prompt"],
                negative_prompt=None,
                num_images_per_prompt=params["batch_size"],
                num_inference_steps=params["prior_steps"],
                latents=None,
                guidance_scale=params["prior_cf_scale"],
                output_type="pt",
                generator=prior_generator,
                return_dict=True,
            ).image_embeds
            if params["negative_prompt"] != ""
            else zero_embeds
        )
        k_log("negative prior embeddings: done")

        use_scheduler(decoder, params["sampler"])

        cnet_image = cnet_image.resize((params["w"], params["h"]))
        hint = generate_hint(cnet_image, cnet_condition, self.params)

        images = []
        prior_on_cpu = self.params("diffusers", "run_prior_on_cpu")
        for _ in itertools.repeat(None, params["batch_count"]):
            current_batch = decoder(
                image_embeds=image_embeds.half() if prior_on_cpu else image_embeds,
                negative_image_embeds=negative_image_embeds.half()
                if prior_on_cpu
                else negative_image_embeds,
                hint=hint,
                width=params["w"],
                height=params["h"],
                num_inference_steps=params["num_steps"],
                guidance_scale=params["guidance_scale"],
                num_images_per_prompt=params["batch_size"],
                generator=decoder_generator,
                latents=None,
                output_type="pil",
                return_dict=True,
            ).images

            images += self.create_batch_images(params, "text2img_cnet", current_batch)
        k_log("decoder images: done")
        return images

    def i2i_cnet(self, params):
        prior, decoder = self.prepareModel("img2img_cnet")
        assert isinstance(prior, KandinskyV22PriorEmb2EmbPipeline)
        assert isinstance(decoder, KandinskyV22ControlnetImg2ImgPipeline)
        params, prior_generator, decoder_generator = self.prepareParams(params)

        init_image = params["init_image"]
        i2i_cnet_image = params["cnet_image"]
        i2i_cnet_condition = params["cnet_condition"]
        i2i_cnet_emb_transform_strength = params["cnet_emb_transform_strength"]
        i2i_cnet_neg_emb_transform_strength = params["cnet_neg_emb_transform_strength"]
        i2i_cnet_img_strength = params["cnet_img_strength"]

        image_embeds, zero_embeds = prior(
            prompt=params["prompt"],
            negative_prompt=params["negative_prior_prompt"]
            if params["negative_prior_prompt"] != ""
            else None,
            image=init_image if init_image is not None else i2i_cnet_image,
            strength=i2i_cnet_emb_transform_strength,
            num_images_per_prompt=params["batch_size"],
            num_inference_steps=params["prior_steps"],
            latents=None,
            guidance_scale=params["prior_cf_scale"],
            output_type="pt",
            generator=prior_generator,
            return_dict=False,
        )
        k_log("prior embeddings: done")

        negative_image_embeds = (
            prior(
                prompt=params["negative_prompt"],
                negative_prompt=None,
                image=init_image if init_image is not None else i2i_cnet_image,
                strength=i2i_cnet_neg_emb_transform_strength,
                num_images_per_prompt=params["batch_size"],
                num_inference_steps=params["prior_steps"],
                latents=None,
                guidance_scale=params["prior_cf_scale"],
                output_type="pt",
                generator=prior_generator,
                return_dict=True,
            ).image_embeds
            if params["negative_prompt"] != ""
            else zero_embeds
        )
        k_log("negative prior embeddings: done")

        use_scheduler(decoder, params["sampler"])

        i2i_cnet_image = i2i_cnet_image.resize((params["w"], params["h"]))
        hint = generate_hint(i2i_cnet_image, i2i_cnet_condition, self.params)

        images = []
        prior_on_cpu = self.params("diffusers", "run_prior_on_cpu")
        for _ in itertools.repeat(None, params["batch_count"]):
            current_batch = decoder(
                image=i2i_cnet_image,
                image_embeds=image_embeds.half() if prior_on_cpu else image_embeds,
                negative_image_embeds=negative_image_embeds.half()
                if prior_on_cpu
                else negative_image_embeds,
                strength=i2i_cnet_img_strength,
                hint=hint,
                width=params["w"],
                height=params["h"],
                num_inference_steps=params["num_steps"],
                guidance_scale=params["guidance_scale"],
                num_images_per_prompt=params["batch_size"],
                generator=decoder_generator,
                output_type="pil",
                return_dict=True,
            ).images

            images += self.create_batch_images(params, "img2img_cnet", current_batch)
        k_log("decoder images: done")
        return images
