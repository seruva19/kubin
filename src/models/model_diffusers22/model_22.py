import torch
import torch.backends
from model_utils.diffusers_utils import use_scheduler
from models.model_diffusers22.model_22_cnet import generate_hint
from models.model_diffusers22.model_22_utils import (
    flush_if_required,
    prepare_weights_for_task,
)
from utils.image import create_inpaint_targets, create_outpaint_targets
import itertools
import os
import secrets
from params import KubinParams
from utils.file_system import save_output

try:
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
    print(
        "warning: seems like diffusers are not installed, run 'pip install -r diffusers/requirements.txt' to install"
    )
    print("warning: if you are not going to use diffusers, just ignore this message")


class Model_Diffusers22:
    def __init__(self, params: KubinParams):
        print("activating pipeline: diffusers (2.2)")
        self.params = params

        self.pipe_prior: KandinskyV22PriorPipeline | None = None
        self.pipe_prior_e2e: KandinskyV22PriorEmb2EmbPipeline | None = None
        self.t2i_pipe: KandinskyV22Pipeline | None = None
        self.i2i_pipe: KandinskyV22Img2ImgPipeline | None = None
        self.inpaint_pipe: KandinskyV22InpaintPipeline | None = None
        self.cnet_t2i_pipe: KandinskyV22ControlnetPipeline | None = None
        self.cnet_i2i_pipe: KandinskyV22ControlnetImg2ImgPipeline | None = None

        self.cublas_config = os.environ.get("CUBLAS_WORKSPACE_CONFIG", None)

    def prepareModel(self, task):
        print(f"task queued: {task}")
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

        prior, unet = prepare_weights_for_task(self, task)
        return (prior, unet)

    def flush(self, target=None):
        flush_if_required(self, target)

    def prepareParams(self, params):
        input_seed = params["input_seed"]
        seed = secrets.randbelow(99999999999) if input_seed == -1 else input_seed

        print(f"seed generated: {seed}")
        params["input_seed"] = seed
        params["model_name"] = "diffusers2.2"

        generator = torch.Generator(
            device=self.params("general", "device")
        ).manual_seed(params["input_seed"])
        return params, generator

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

        prior, unet = self.prepareModel("text2img")
        assert isinstance(prior, KandinskyV22PriorPipeline)
        assert isinstance(unet, KandinskyV22Pipeline)
        params, generator = self.prepareParams(params)

        print("generating prior embeddings")
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
            generator=generator,
            return_dict=False,
        )

        print("generating negative prior embeddings")
        negative_image_embeds = (
            prior(
                prompt=params["negative_prompt"],
                negative_prompt=None,
                num_images_per_prompt=params["batch_size"],
                num_inference_steps=params["prior_steps"],
                latents=None,
                guidance_scale=params["prior_cf_scale"],
                output_type="pt",
                generator=generator,
                return_dict=True,
            ).image_embeds
            if params["negative_prompt"] != ""
            else zero_embeds
        )

        use_scheduler(unet, params["sampler"])

        images = []
        for _ in itertools.repeat(None, params["batch_count"]):
            print("generating unet images")
            current_batch = unet(
                image_embeds=image_embeds,
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

            images = images + self.create_batch_images(
                params, "text2img", current_batch
            )
        return images

    def i2i(self, params):
        prior, unet = self.prepareModel("img2img")
        assert isinstance(prior, KandinskyV22PriorPipeline)
        assert isinstance(unet, KandinskyV22Img2ImgPipeline)
        params, generator = self.prepareParams(params)

        image_embeds, negative_embeds = prior(
            prompt=params["prompt"],
            negative_prompt=params["negative_prior_prompt"]
            if params["negative_prior_prompt"] != ""
            else None,
            num_images_per_prompt=params["batch_size"],
            num_inference_steps=params["prior_steps"],
            latents=None,
            guidance_scale=params["prior_cf_scale"],
            output_type="pt",
            generator=generator,
            return_dict=False,
        )

        use_scheduler(unet, params["sampler"])

        images = []
        for _ in itertools.repeat(None, params["batch_count"]):
            current_batch = unet(
                image=params["init_image"],
                image_embeds=image_embeds,
                negative_image_embeds=negative_embeds,
                width=params["w"],
                height=params["h"],
                strength=params["strength"],
                num_inference_steps=params["num_steps"],
                guidance_scale=params["guidance_scale"],
                num_images_per_prompt=params["batch_size"],
                generator=generator,
                output_type="pil",
                return_dict=True,
            ).images

            images = images + self.create_batch_images(params, "img2img", current_batch)
        return images

    def mix(self, params):
        prior, unet = self.prepareModel("mix")
        assert isinstance(prior, KandinskyV22PriorPipeline)
        assert isinstance(unet, KandinskyV22Pipeline)
        params, generator = self.prepareParams(params)

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
            generator=generator,
            latents=None,
            negative_prompt=params["negative_prompt"],
            negative_prior_prompt=params["negative_prior_prompt"],
            guidance_scale=params["prior_cf_scale"],
        )

        use_scheduler(unet, params["sampler"])

        images = []
        for _ in itertools.repeat(None, params["batch_count"]):
            current_batch = unet(
                image_embeds=embeds.image_embeds,
                negative_image_embeds=embeds.negative_image_embeds,
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

            images = images + self.create_batch_images(params, "mix", current_batch)
        return images

    def inpaint(self, params):
        prior, unet = self.prepareModel("inpainting")
        assert isinstance(prior, KandinskyV22PriorPipeline)
        assert isinstance(unet, KandinskyV22InpaintPipeline)
        params, generator = self.prepareParams(params)

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
            generator=generator,
            return_dict=False,
        )

        negative_image_embeds = (
            prior(
                prompt=params["negative_prompt"],
                negative_prompt=None,
                num_images_per_prompt=params["batch_size"],
                num_inference_steps=params["prior_steps"],
                latents=None,
                guidance_scale=params["prior_cf_scale"],
                output_type="pt",
                generator=generator,
                return_dict=True,
            ).image_embeds
            if params["negative_prompt"] != ""
            else zero_embeds
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

        use_scheduler(unet, params["sampler"])

        images = []
        for _ in itertools.repeat(None, params["batch_count"]):
            current_batch = unet(
                image=image,
                mask_image=mask,
                image_embeds=image_embeds,
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

            images = images + self.create_batch_images(
                params, "inpainting", current_batch
            )
        return images

    def outpaint(self, params):
        prior, unet = self.prepareModel("outpainting")
        assert isinstance(prior, KandinskyV22PriorPipeline)
        assert isinstance(unet, KandinskyV22InpaintPipeline)
        params, generator = self.prepareParams(params)

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
            generator=generator,
            return_dict=False,
        )

        negative_image_embeds = (
            prior(
                prompt=params["negative_prompt"],
                negative_prompt=None,
                num_images_per_prompt=params["batch_size"],
                num_inference_steps=params["prior_steps"],
                latents=None,
                guidance_scale=params["prior_cf_scale"],
                output_type="pt",
                generator=generator,
                return_dict=True,
            ).image_embeds
            if params["negative_prompt"] != ""
            else zero_embeds
        )

        image = params["image"]
        offset = params["offset"]
        infer_size = params["infer_size"]
        width = params["w"]
        height = params["h"]

        image, mask, width, height = create_outpaint_targets(
            image, offset, infer_size, width, height
        )

        use_scheduler(unet, params["sampler"])

        images = []
        for _ in itertools.repeat(None, params["batch_count"]):
            current_batch = unet(
                image=image,
                mask_image=mask,
                image_embeds=image_embeds,
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

            images = images + self.create_batch_images(
                params, "outpainting", current_batch
            )
        return images

    def t2i_cnet(self, params):
        prior, unet = self.prepareModel("text2img_cnet")
        assert isinstance(prior, KandinskyV22PriorPipeline)
        assert isinstance(unet, KandinskyV22ControlnetPipeline)
        params, generator = self.prepareParams(params)

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
            generator=generator,
            return_dict=False,
        )

        negative_image_embeds = (
            prior(
                prompt=params["negative_prompt"],
                negative_prompt=None,
                num_images_per_prompt=params["batch_size"],
                num_inference_steps=params["prior_steps"],
                latents=None,
                guidance_scale=params["prior_cf_scale"],
                output_type="pt",
                generator=generator,
                return_dict=True,
            ).image_embeds
            if params["negative_prompt"] != ""
            else zero_embeds
        )

        use_scheduler(unet, params["sampler"])

        cnet_image = cnet_image.resize((params["w"], params["h"]))
        hint = generate_hint(cnet_image, cnet_condition)

        images = []
        for _ in itertools.repeat(None, params["batch_count"]):
            current_batch = unet(
                image_embeds=image_embeds,
                negative_image_embeds=negative_image_embeds,
                hint=hint,
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

            images = images + self.create_batch_images(
                params, "text2img_cnet", current_batch
            )
        return images

    def i2i_cnet(self, params):
        prior, unet = self.prepareModel("img2img_cnet")
        assert isinstance(prior, KandinskyV22PriorEmb2EmbPipeline)
        assert isinstance(unet, KandinskyV22ControlnetImg2ImgPipeline)
        params, generator = self.prepareParams(params)

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
            image=i2i_cnet_image,
            strength=i2i_cnet_emb_transform_strength,
            num_images_per_prompt=params["batch_size"],
            num_inference_steps=params["prior_steps"],
            latents=None,
            guidance_scale=params["prior_cf_scale"],
            output_type="pt",
            generator=generator,
            return_dict=False,
        )

        negative_image_embeds = (
            prior(
                prompt=params["negative_prompt"],
                negative_prompt=None,
                image=i2i_cnet_image,
                strength=i2i_cnet_neg_emb_transform_strength,
                num_images_per_prompt=params["batch_size"],
                num_inference_steps=params["prior_steps"],
                latents=None,
                guidance_scale=params["prior_cf_scale"],
                output_type="pt",
                generator=generator,
                return_dict=True,
            ).image_embeds
            if params["negative_prompt"] != ""
            else zero_embeds
        )

        use_scheduler(unet, params["sampler"])

        i2i_cnet_image = i2i_cnet_image.resize((params["w"], params["h"]))
        hint = generate_hint(i2i_cnet_image, i2i_cnet_condition)

        images = []
        for _ in itertools.repeat(None, params["batch_count"]):
            current_batch = unet(
                image=i2i_cnet_image,
                image_embeds=image_embeds,
                negative_image_embeds=negative_image_embeds,
                strength=i2i_cnet_img_strength,
                hint=hint,
                width=params["w"],
                height=params["h"],
                num_inference_steps=params["num_steps"],
                guidance_scale=params["guidance_scale"],
                num_images_per_prompt=params["batch_size"],
                generator=generator,
                output_type="pil",
                return_dict=True,
            ).images

            images = images + self.create_batch_images(
                params, "img2img_cnet", current_batch
            )
        return images
