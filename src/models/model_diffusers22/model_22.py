from typing import Any
import torch
import torch.backends
from models.model_diffusers22.patched.patched import KandinskyV22PipelinePatched
from models.model_diffusers22.patched.patched_controlnet import (
    KandinskyV22ControlnetPipelinePatched,
)
from models.model_diffusers22.patched.patched_controlnet_img2img import (
    KandinskyV22ControlnetImg2ImgPipelinePatched,
)
from models.model_diffusers22.patched.patched_img2img import (
    KandinskyV22Img2ImgPipelinePatched,
)
from models.model_diffusers22.patched.patched_inpainting import (
    KandinskyV22InpaintPipelinePatched,
)
from models.model_diffusers22.patched.patched_prior import (
    KandinskyV22PriorPipelinePatched,
)
from models.model_diffusers22.patched.patched_prior_emb2emb import (
    KandinskyV22PriorEmb2EmbPipelinePatched,
)
from progress import report_progress

from utils.image import (
    composite_images,
    create_inpaint_targets,
    create_outpaint_targets,
    round_to_nearest,
)
import itertools
import os
import secrets
from params import KubinParams
from utils.file_system import save_output
from utils.logging import k_log

from model_utils.diffusers_samplers import use_sampler
from models.model_diffusers22.model_22_cnet import generate_hint
from models.model_diffusers22.model_22_init import (
    flush_if_required,
    prepare_weights_for_task,
    images_or_texts,
    clear_pipe_info,
    execute_forced_hooks,
)

from transformers import (
    CLIPImageProcessor,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)
from diffusers.models import (
    UNet2DConditionModel,
    PriorTransformer,
    VQModel,
)

from diffusers.schedulers import UnCLIPScheduler
from hooks.hooks import HOOK


class Model_Diffusers22:
    def __init__(self, params: KubinParams):
        k_log("activating pipeline: diffusers (2.2)")
        self.params = params

        self.prior_transformer: PriorTransformer | None = None
        self.image_encoder: CLIPVisionModelWithProjection | None = None
        self.text_encoder: CLIPTextModelWithProjection | None = None
        self.tokenizer: CLIPTokenizer | None = None
        self.unclip_scheduler: UnCLIPScheduler | None = None
        self.image_processor: CLIPImageProcessor | None = None

        self.scheduler: Any = None
        self.unet_2d: UNet2DConditionModel | None = None
        self.movq: VQModel | None = None

        self.pipe_prior: KandinskyV22PriorPipelinePatched | None = None
        self.pipe_prior_e2e: KandinskyV22PriorEmb2EmbPipelinePatched | None = None
        self.t2i_pipe: KandinskyV22PipelinePatched | None = None
        self.i2i_pipe: KandinskyV22Img2ImgPipelinePatched | None = None
        self.inpaint_pipe: KandinskyV22InpaintPipelinePatched | None = None
        self.cnet_t2i_pipe: KandinskyV22ControlnetPipelinePatched | None = None
        self.cnet_i2i_pipe: KandinskyV22ControlnetImg2ImgPipelinePatched | None = None

        self.config = {}
        self.cublas_config = os.environ.get("CUBLAS_WORKSPACE_CONFIG", None)

        clear_pipe_info(self)

    def prepare_model(self, task):
        k_log(f"task queued: {task}")
        assert task in [
            "text2img",
            "text2img_cnet",
            "img2img",
            "img2img_cnet",
            "mix",
            "mix_cnet",
            "inpainting",
            "outpainting",
        ]

        prior, decoder = prepare_weights_for_task(self, task)
        return (prior, decoder)

    def flush(self, target=None):
        hook_params = {"model": self, "target": target}
        self.params.hook_store.call(
            HOOK.BEFORE_FLUSH_MODEL,
            **hook_params,
        )
        flush_if_required(self, target)
        self.params.hook_store.call(
            HOOK.AFTER_FLUSH_MODEL,
            **hook_params,
        )

    def prepare_params(self, params):
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
        params[".ui-task"] = task

        if params["cnet_enable"]:
            if params["cnet_pipeline"] == "ControlNetPipeline":
                return self.t2i_cnet(params)
            elif params["cnet_pipeline"] == "ControlNetImg2ImgPipeline":
                return self.i2i_cnet(params)

        hooks = self.params.hook_store
        hook_params = {"model": self, "params": params, "task": task}
        execute_forced_hooks(HOOK.BEFORE_PREPARE_MODEL, params, hook_params)
        hooks.call(
            HOOK.BEFORE_PREPARE_MODEL,
            **hook_params,
        )

        prior, decoder = self.prepare_model(task)

        assert isinstance(prior, KandinskyV22PriorPipelinePatched)
        assert isinstance(decoder, KandinskyV22PipelinePatched)

        hook_params["prior"] = prior
        hook_params["decoder"] = decoder
        execute_forced_hooks(HOOK.BEFORE_PREPARE_PARAMS, params, hook_params)
        hooks.call(
            HOOK.BEFORE_PREPARE_PARAMS,
            **hook_params,
        )

        params, prior_generator, decoder_generator = self.prepare_params(params)

        hook_params["prior_generator"] = prior_generator
        hook_params["decoder_generator"] = decoder_generator
        execute_forced_hooks(HOOK.BEFORE_PREPARE_EMBEDS, params, hook_params)
        self.params.hook_store.call(
            HOOK.BEFORE_PREPARE_EMBEDS,
            **hook_params,
        )

        image_embeds, zero_embeds = prior(
            prompt=params["prompt"],
            negative_prompt=(
                params["negative_prior_prompt"]
                if params["negative_prior_prompt"] != ""
                else None
            ),
            num_images_per_prompt=params["batch_size"],
            num_inference_steps=params["prior_steps"],
            latents=None,
            guidance_scale=params["prior_cf_scale"],
            output_type="pt",
            generator=prior_generator,
            return_dict=False,
            callback=lambda s, ts, ft: report_progress(
                task, "prior", params["prior_steps"], s, ts, ft
            ),
            callback_steps=1,
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
                callback=lambda s, ts, ft: report_progress(
                    task, "prior_negative", params["prior_steps"], s, ts, ft
                ),
            ).image_embeds
            if params["negative_prompt"] != ""
            else zero_embeds
        )
        k_log("negative prior embeddings: done")

        use_sampler(decoder, params["sampler"], task)

        images = []
        prior_on_cpu = self.params("diffusers", "run_prior_on_cpu")

        hook_params["image_embeds"] = image_embeds
        hook_params["negative_image_embeds"] = negative_image_embeds
        hook_params["scheduler"] = decoder.scheduler
        execute_forced_hooks(HOOK.BEFORE_PREPARE_DECODER, params, hook_params)
        self.params.hook_store.call(
            HOOK.BEFORE_PREPARE_DECODER,
            **hook_params,
        )

        for _ in itertools.repeat(None, params["batch_count"]):
            current_batch = decoder(
                image_embeds=image_embeds.half() if prior_on_cpu else image_embeds,
                negative_image_embeds=(
                    negative_image_embeds.half()
                    if prior_on_cpu
                    else negative_image_embeds
                ),
                width=params["w"],
                height=params["h"],
                num_inference_steps=params["num_steps"],
                guidance_scale=params["guidance_scale"],
                num_images_per_prompt=params["batch_size"],
                generator=decoder_generator,
                latents=None,
                output_type="pil",
                return_dict=True,
                callback=lambda s, ts, ft: report_progress(
                    task,
                    "decoder",
                    params["num_steps"] * params["batch_count"],
                    s,
                    ts,
                    ft,
                ),
                callback_steps=1,
            ).images

            hook_params["batch"] = current_batch
            execute_forced_hooks(HOOK.BEFORE_BATCH_SAVE, params, hook_params)
            self.params.hook_store.call(
                HOOK.BEFORE_BATCH_SAVE,
                **hook_params,
            )

            images += self.create_batch_images(params, "text2img", current_batch)
        k_log("decoder images: done")

        execute_forced_hooks(HOOK.BEFORE_TASK_QUIT, params, hook_params)
        self.params.hook_store.call(
            HOOK.BEFORE_TASK_QUIT,
            **hook_params,
        )

        return images

    def i2i(self, params):
        task = "img2img"
        params[".ui-task"] = task

        if params["cnet_enable"]:
            return self.i2i_cnet(params)

        hooks = self.params.hook_store
        hook_params = {"model": self, "params": params, "task": task}
        execute_forced_hooks(HOOK.BEFORE_PREPARE_MODEL, params, hook_params)
        hooks.call(
            HOOK.BEFORE_PREPARE_MODEL,
            **hook_params,
        )

        prior, decoder = self.prepare_model(task)
        assert isinstance(prior, KandinskyV22PriorPipelinePatched)
        assert isinstance(decoder, KandinskyV22Img2ImgPipelinePatched)

        hook_params["prior"] = prior
        hook_params["decoder"] = decoder
        execute_forced_hooks(HOOK.BEFORE_PREPARE_PARAMS, params, hook_params)
        hooks.call(
            HOOK.BEFORE_PREPARE_PARAMS,
            **hook_params,
        )

        params, prior_generator, decoder_generator = self.prepare_params(params)

        hook_params["prior_generator"] = prior_generator
        hook_params["decoder_generator"] = decoder_generator
        execute_forced_hooks(HOOK.BEFORE_PREPARE_EMBEDS, params, hook_params)
        self.params.hook_store.call(
            HOOK.BEFORE_PREPARE_EMBEDS,
            **hook_params,
        )

        image_embeds, negative_image_embeds = prior(
            prompt=params["prompt"],
            negative_prompt=(
                params["negative_prior_prompt"]
                if params["negative_prior_prompt"] != ""
                else None
            ),
            num_images_per_prompt=params["batch_size"],
            num_inference_steps=params["prior_steps"],
            latents=None,
            guidance_scale=params["prior_cf_scale"],
            output_type="pt",
            generator=prior_generator,
            return_dict=False,
        )
        k_log("prior embeddings: done")

        use_sampler(decoder, params["sampler"], task)

        images = []
        prior_on_cpu = self.params("diffusers", "run_prior_on_cpu")

        hook_params["image_embeds"] = image_embeds
        hook_params["negative_image_embeds"] = negative_image_embeds
        hook_params["scheduler"] = decoder.scheduler
        execute_forced_hooks(HOOK.BEFORE_PREPARE_DECODER, params, hook_params)
        self.params.hook_store.call(
            HOOK.BEFORE_PREPARE_DECODER,
            **hook_params,
        )

        for _ in itertools.repeat(None, params["batch_count"]):
            current_batch = decoder(
                image=params["init_image"],
                image_embeds=image_embeds.half() if prior_on_cpu else image_embeds,
                negative_image_embeds=(
                    negative_image_embeds.half()
                    if prior_on_cpu
                    else negative_image_embeds
                ),
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

            hook_params["batch"] = current_batch
            execute_forced_hooks(HOOK.BEFORE_BATCH_SAVE, params, hook_params)
            self.params.hook_store.call(
                HOOK.BEFORE_BATCH_SAVE,
                **hook_params,
            )

            images += self.create_batch_images(params, task, current_batch)
        k_log("decoder images: done")

        execute_forced_hooks(HOOK.BEFORE_TASK_QUIT, params, hook_params)
        self.params.hook_store.call(
            HOOK.BEFORE_TASK_QUIT,
            **hook_params,
        )

        return images

    def mix(self, params):
        task = "mix"
        params[".ui-task"] = task

        if params["cnet_enable"]:
            return self.mix_cnet(params)

        hooks = self.params.hook_store
        hook_params = {"model": self, "params": params, "task": task}
        execute_forced_hooks(HOOK.BEFORE_PREPARE_MODEL, params, hook_params)
        hooks.call(
            HOOK.BEFORE_PREPARE_MODEL,
            **hook_params,
        )

        prior, decoder = self.prepare_model(task)
        assert isinstance(prior, KandinskyV22PriorPipelinePatched)
        assert isinstance(decoder, KandinskyV22PipelinePatched)

        hook_params["prior"] = prior
        hook_params["decoder"] = decoder
        execute_forced_hooks(HOOK.BEFORE_PREPARE_PARAMS, params, hook_params)
        hooks.call(
            HOOK.BEFORE_PREPARE_PARAMS,
            **hook_params,
        )

        params, prior_generator, decoder_generator = self.prepare_params(params)

        images_texts = images_or_texts(
            [params["image_1"], params["image_2"]],
            [params["text_1"], params["text_2"]],
        )
        weights = [params["weight_1"], params["weight_2"]]

        hook_params["prior_generator"] = prior_generator
        hook_params["decoder_generator"] = decoder_generator
        hook_params["images_texts"] = images_texts
        hook_params["weights"] = weights
        execute_forced_hooks(HOOK.BEFORE_PREPARE_EMBEDS, params, hook_params)
        self.params.hook_store.call(
            HOOK.BEFORE_PREPARE_EMBEDS,
            **hook_params,
        )

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
        k_log("interpolation prior embeddings: done")

        use_sampler(decoder, params["sampler"], task)

        images = []
        prior_on_cpu = self.params("diffusers", "run_prior_on_cpu")

        hook_params["image_embeds"] = embeds.image_embeds
        hook_params["negative_image_embeds"] = embeds.negative_image_embeds
        hook_params["scheduler"] = decoder.scheduler
        execute_forced_hooks(HOOK.BEFORE_PREPARE_DECODER, params, hook_params)
        self.params.hook_store.call(
            HOOK.BEFORE_PREPARE_DECODER,
            **hook_params,
        )

        for _ in itertools.repeat(None, params["batch_count"]):
            current_batch = decoder(
                image_embeds=(
                    embeds.image_embeds.half() if prior_on_cpu else embeds.image_embeds
                ),
                negative_image_embeds=(
                    embeds.negative_image_embeds.half()
                    if prior_on_cpu
                    else embeds.negative_image_embeds
                ),
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

            hook_params["batch"] = current_batch
            execute_forced_hooks(HOOK.BEFORE_BATCH_SAVE, params, hook_params)
            self.params.hook_store.call(
                HOOK.BEFORE_BATCH_SAVE,
                **hook_params,
            )

            images += self.create_batch_images(params, task, current_batch)
        k_log("decoder images: done")

        execute_forced_hooks(HOOK.BEFORE_TASK_QUIT, params, hook_params)
        self.params.hook_store.call(
            HOOK.BEFORE_TASK_QUIT,
            **hook_params,
        )

        return images

    def inpaint(self, params):
        task = "inpainting"
        params[".ui-task"] = task

        hooks = self.params.hook_store
        hook_params = {"model": self, "params": params, "task": task}
        execute_forced_hooks(HOOK.BEFORE_PREPARE_MODEL, params, hook_params)
        hooks.call(
            HOOK.BEFORE_PREPARE_MODEL,
            **hook_params,
        )

        prior, decoder = self.prepare_model(task)
        assert isinstance(prior, KandinskyV22PriorPipelinePatched)
        assert isinstance(decoder, KandinskyV22InpaintPipelinePatched)

        hook_params["prior"] = prior
        hook_params["decoder"] = decoder
        execute_forced_hooks(HOOK.BEFORE_PREPARE_PARAMS, params, hook_params)
        hooks.call(
            HOOK.BEFORE_PREPARE_PARAMS,
            **hook_params,
        )

        params, prior_generator, decoder_generator = self.prepare_params(params)

        hook_params["prior_generator"] = prior_generator
        hook_params["decoder_generator"] = decoder_generator
        execute_forced_hooks(HOOK.BEFORE_PREPARE_EMBEDS, params, hook_params)
        self.params.hook_store.call(
            HOOK.BEFORE_PREPARE_EMBEDS,
            **hook_params,
        )

        image_embeds, zero_embeds = prior(
            prompt=params["prompt"],
            negative_prompt=(
                params["negative_prior_prompt"]
                if params["negative_prior_prompt"] != ""
                else None
            ),
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
            (
                round_to_nearest(pil_img.width, 64)
                if params["infer_size"]
                else params["w"]
            ),
            (
                round_to_nearest(pil_img.height, 64)
                if params["infer_size"]
                else params["h"]
            ),
        )
        output_size = (width, height)
        mask = image_mask["mask"]

        inpaint_region = params["region"]
        inpaint_target = params["target"]

        image, mask = create_inpaint_targets(
            pil_img, mask, output_size, inpaint_region, inpaint_target
        )

        use_sampler(decoder, params["sampler"], task)

        images = []
        prior_on_cpu = self.params("diffusers", "run_prior_on_cpu")

        hook_params["image_embeds"] = image_embeds
        hook_params["negative_image_embeds"] = negative_image_embeds
        hook_params["scheduler"] = decoder.scheduler
        hook_params["inpaint_image"] = image
        hook_params["inpaint_mask"] = mask
        execute_forced_hooks(HOOK.BEFORE_PREPARE_DECODER, params, hook_params)
        self.params.hook_store.call(
            HOOK.BEFORE_PREPARE_DECODER,
            **hook_params,
        )

        for _ in itertools.repeat(None, params["batch_count"]):
            current_batch = decoder(
                image=image,
                mask_image=mask,
                image_embeds=image_embeds.half() if prior_on_cpu else image_embeds,
                negative_image_embeds=(
                    negative_image_embeds.half()
                    if prior_on_cpu
                    else negative_image_embeds
                ),
                width=width,
                height=height,
                num_inference_steps=params["num_steps"],
                guidance_scale=params["guidance_scale"],
                num_images_per_prompt=params["batch_size"],
                generator=decoder_generator,
                latents=None,
                output_type="pil",
                return_dict=True,
            ).images

            hook_params["batch"] = current_batch
            execute_forced_hooks(HOOK.BEFORE_BATCH_SAVE, params, hook_params)
            self.params.hook_store.call(
                HOOK.BEFORE_BATCH_SAVE,
                **hook_params,
            )

            if inpaint_region == "mask":
                current_batch_composed = []
                for inpainted_image in current_batch:
                    merged_image = composite_images(pil_img, inpainted_image, mask)
                    current_batch_composed.append(merged_image)
                current_batch = current_batch_composed

            images += self.create_batch_images(params, task, current_batch)
        k_log("decoder images: done")

        execute_forced_hooks(HOOK.BEFORE_TASK_QUIT, params, hook_params)
        self.params.hook_store.call(
            HOOK.BEFORE_TASK_QUIT,
            **hook_params,
        )

        return images

    def outpaint(self, params):
        task = "outpainting"
        params[".ui-task"] = task

        hooks = self.params.hook_store
        hook_params = {"model": self, "params": params, "task": task}
        execute_forced_hooks(HOOK.BEFORE_PREPARE_MODEL, params, hook_params)
        hooks.call(
            HOOK.BEFORE_PREPARE_MODEL,
            **hook_params,
        )

        prior, decoder = self.prepare_model(task)
        assert isinstance(prior, KandinskyV22PriorPipelinePatched)
        assert isinstance(decoder, KandinskyV22InpaintPipelinePatched)

        hook_params["prior"] = prior
        hook_params["decoder"] = decoder
        execute_forced_hooks(HOOK.BEFORE_PREPARE_PARAMS, params, hook_params)
        hooks.call(
            HOOK.BEFORE_PREPARE_PARAMS,
            **hook_params,
        )

        params, prior_generator, decoder_generator = self.prepare_params(params)

        hook_params["prior_generator"] = prior_generator
        hook_params["decoder_generator"] = decoder_generator
        execute_forced_hooks(HOOK.BEFORE_PREPARE_EMBEDS, params, hook_params)
        self.params.hook_store.call(
            HOOK.BEFORE_PREPARE_EMBEDS,
            **hook_params,
        )

        image_embeds, zero_embeds = prior(
            prompt=params["prompt"],
            negative_prompt=(
                params["negative_prior_prompt"]
                if params["negative_prior_prompt"] != ""
                else None
            ),
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
        w = params["w"]
        h = params["h"]

        image, mask, width, height = create_outpaint_targets(
            image, offset, infer_size, w, h
        )

        use_sampler(decoder, params["sampler"], task)

        images = []
        prior_on_cpu = self.params("diffusers", "run_prior_on_cpu")

        hook_params["image_embeds"] = image_embeds
        hook_params["negative_image_embeds"] = negative_image_embeds
        hook_params["scheduler"] = decoder.scheduler
        hook_params["outpaint_image"] = image
        hook_params["outpaint_mask"] = mask
        hook_params["outpaint_width"] = width
        hook_params["outpaint_height"] = height
        execute_forced_hooks(HOOK.BEFORE_PREPARE_DECODER, params, hook_params)
        self.params.hook_store.call(
            HOOK.BEFORE_PREPARE_DECODER,
            **hook_params,
        )

        for _ in itertools.repeat(None, params["batch_count"]):
            current_batch = decoder(
                image=image,
                mask_image=mask,
                image_embeds=image_embeds.half() if prior_on_cpu else image_embeds,
                negative_image_embeds=(
                    negative_image_embeds.half()
                    if prior_on_cpu
                    else negative_image_embeds
                ),
                width=width,
                height=height,
                num_inference_steps=params["num_steps"],
                guidance_scale=params["guidance_scale"],
                num_images_per_prompt=params["batch_size"],
                generator=decoder_generator,
                latents=None,
                output_type="pil",
                return_dict=True,
            ).images

            hook_params["batch"] = current_batch
            execute_forced_hooks(HOOK.BEFORE_BATCH_SAVE, params, hook_params)
            self.params.hook_store.call(
                HOOK.BEFORE_BATCH_SAVE,
                **hook_params,
            )

            images += self.create_batch_images(params, task, current_batch)
        k_log("decoder images: done")

        execute_forced_hooks(HOOK.BEFORE_TASK_QUIT, params, hook_params)
        self.params.hook_store.call(
            HOOK.BEFORE_TASK_QUIT,
            **hook_params,
        )

        return images

    def t2i_cnet(self, params):
        task = "text2img_cnet"

        hooks = self.params.hook_store
        hook_params = {"model": self, "params": params, "task": task}
        execute_forced_hooks(HOOK.BEFORE_PREPARE_MODEL, params, hook_params)
        hooks.call(
            HOOK.BEFORE_PREPARE_MODEL,
            **hook_params,
        )

        prior, decoder = self.prepare_model(task)
        assert isinstance(prior, KandinskyV22PriorPipelinePatched)
        assert isinstance(decoder, KandinskyV22ControlnetPipelinePatched)

        hook_params["prior"] = prior
        hook_params["decoder"] = decoder
        execute_forced_hooks(HOOK.BEFORE_PREPARE_PARAMS, params, hook_params)
        hooks.call(
            HOOK.BEFORE_PREPARE_PARAMS,
            **hook_params,
        )

        params, prior_generator, decoder_generator = self.prepare_params(params)

        hook_params["prior_generator"] = prior_generator
        hook_params["decoder_generator"] = decoder_generator
        execute_forced_hooks(HOOK.BEFORE_PREPARE_EMBEDS, params, hook_params)
        self.params.hook_store.call(
            HOOK.BEFORE_PREPARE_EMBEDS,
            **hook_params,
        )

        cnet_image = params["cnet_image"]
        cnet_condition = params["cnet_condition"]
        cnet_depth_estimator = params["cnet_depth_estimator"]

        image_embeds, zero_embeds = prior(
            prompt=params["prompt"],
            negative_prompt=(
                params["negative_prior_prompt"]
                if params["negative_prior_prompt"] != ""
                else None
            ),
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

        use_sampler(decoder, params["sampler"], task)

        cnet_image = cnet_image.resize((params["w"], params["h"]))
        hint = generate_hint(
            self, cnet_image, cnet_condition, cnet_depth_estimator, self.params
        )

        images = []
        prior_on_cpu = self.params("diffusers", "run_prior_on_cpu")

        hook_params["image_embeds"] = image_embeds
        hook_params["negative_image_embeds"] = negative_image_embeds
        hook_params["scheduler"] = decoder.scheduler
        hook_params["cnet_hint"] = hint
        execute_forced_hooks(HOOK.BEFORE_PREPARE_DECODER, params, hook_params)
        self.params.hook_store.call(
            HOOK.BEFORE_PREPARE_DECODER,
            **hook_params,
        )

        for _ in itertools.repeat(None, params["batch_count"]):
            current_batch = decoder(
                image_embeds=image_embeds.half() if prior_on_cpu else image_embeds,
                negative_image_embeds=(
                    negative_image_embeds.half()
                    if prior_on_cpu
                    else negative_image_embeds
                ),
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

            hook_params["batch"] = current_batch
            execute_forced_hooks(HOOK.BEFORE_BATCH_SAVE, params, hook_params)
            self.params.hook_store.call(
                HOOK.BEFORE_BATCH_SAVE,
                **hook_params,
            )

            images += self.create_batch_images(params, "text2img_cnet", current_batch)
        k_log("decoder images: done")

        execute_forced_hooks(HOOK.BEFORE_TASK_QUIT, params, hook_params)
        self.params.hook_store.call(
            HOOK.BEFORE_TASK_QUIT,
            **hook_params,
        )

        return images

    def i2i_cnet(self, params):
        task = "img2img_cnet"

        hooks = self.params.hook_store
        hook_params = {"model": self, "params": params, "task": task}
        execute_forced_hooks(HOOK.BEFORE_PREPARE_MODEL, params, hook_params)
        hooks.call(
            HOOK.BEFORE_PREPARE_MODEL,
            **hook_params,
        )

        prior, decoder = self.prepare_model(task)
        assert isinstance(prior, KandinskyV22PriorEmb2EmbPipelinePatched)
        assert isinstance(decoder, KandinskyV22ControlnetImg2ImgPipelinePatched)

        hook_params["prior"] = prior
        hook_params["decoder"] = decoder
        execute_forced_hooks(HOOK.BEFORE_PREPARE_PARAMS, params, hook_params)
        hooks.call(
            HOOK.BEFORE_PREPARE_PARAMS,
            **hook_params,
        )

        params, prior_generator, decoder_generator = self.prepare_params(params)

        hook_params["prior_generator"] = prior_generator
        hook_params["decoder_generator"] = decoder_generator
        execute_forced_hooks(HOOK.BEFORE_PREPARE_EMBEDS, params, hook_params)
        self.params.hook_store.call(
            HOOK.BEFORE_PREPARE_EMBEDS,
            **hook_params,
        )

        init_image = params["init_image"]
        i2i_cnet_image = params["cnet_image"]
        i2i_cnet_condition = params["cnet_condition"]
        i2i_cnet_depth_estimator = params["cnet_depth_estimator"]
        i2i_cnet_emb_transform_strength = params["cnet_emb_transform_strength"]
        i2i_cnet_neg_emb_transform_strength = params["cnet_neg_emb_transform_strength"]
        i2i_cnet_img_strength = params["cnet_img_strength"]

        image_embeds, zero_embeds = prior(
            prompt=params["prompt"],
            negative_prompt=(
                params["negative_prior_prompt"]
                if params["negative_prior_prompt"] != ""
                else None
            ),
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

        use_sampler(decoder, params["sampler"], task)

        i2i_cnet_image = i2i_cnet_image.resize((params["w"], params["h"]))
        hint = generate_hint(
            self,
            i2i_cnet_image,
            i2i_cnet_condition,
            i2i_cnet_depth_estimator,
            self.params,
        )

        images = []
        prior_on_cpu = self.params("diffusers", "run_prior_on_cpu")

        hook_params["image_embeds"] = image_embeds
        hook_params["negative_image_embeds"] = negative_image_embeds
        hook_params["scheduler"] = decoder.scheduler
        hook_params["cnet_hint"] = hint
        execute_forced_hooks(HOOK.BEFORE_PREPARE_DECODER, params, hook_params)
        self.params.hook_store.call(
            HOOK.BEFORE_PREPARE_DECODER,
            **hook_params,
        )

        for _ in itertools.repeat(None, params["batch_count"]):
            current_batch = decoder(
                image=i2i_cnet_image,
                image_embeds=image_embeds.half() if prior_on_cpu else image_embeds,
                negative_image_embeds=(
                    negative_image_embeds.half()
                    if prior_on_cpu
                    else negative_image_embeds
                ),
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

            hook_params["batch"] = current_batch
            execute_forced_hooks(HOOK.BEFORE_BATCH_SAVE, params, hook_params)
            self.params.hook_store.call(
                HOOK.BEFORE_BATCH_SAVE,
                **hook_params,
            )

            images += self.create_batch_images(params, task, current_batch)
        k_log("decoder images: done")

        execute_forced_hooks(HOOK.BEFORE_TASK_QUIT, params, hook_params)
        self.params.hook_store.call(
            HOOK.BEFORE_TASK_QUIT,
            **hook_params,
        )

        return images

    def mix_cnet(self, params):
        task = "mix_cnet"

        hooks = self.params.hook_store
        hook_params = {"model": self, "params": params, "task": task}
        execute_forced_hooks(HOOK.BEFORE_PREPARE_MODEL, params, hook_params)
        hooks.call(
            HOOK.BEFORE_PREPARE_MODEL,
            **hook_params,
        )

        prior, decoder = self.prepare_model(task)
        assert isinstance(prior, KandinskyV22PriorPipelinePatched)
        assert isinstance(decoder, KandinskyV22ControlnetImg2ImgPipelinePatched)

        hook_params["prior"] = prior
        hook_params["decoder"] = decoder
        execute_forced_hooks(HOOK.BEFORE_PREPARE_PARAMS, params, hook_params)
        hooks.call(
            HOOK.BEFORE_PREPARE_PARAMS,
            **hook_params,
        )

        params, prior_generator, decoder_generator = self.prepare_params(params)

        mix_cnet_image = params["cnet_image"]
        mix_cnet_condition = params["cnet_condition"]
        mix_cnet_depth_estimator = params["cnet_depth_estimator"]
        mix_cnet_img_strength = params["cnet_img_strength"]

        images_texts = images_or_texts(
            [params["image_1"], params["image_2"]],
            [params["text_1"], params["text_2"]],
        )
        weights = [params["weight_1"], params["weight_2"]]

        hook_params["prior_generator"] = prior_generator
        hook_params["decoder_generator"] = decoder_generator
        hook_params["images_texts"] = images_texts
        hook_params["weights"] = weights
        execute_forced_hooks(HOOK.BEFORE_PREPARE_EMBEDS, params, hook_params)
        self.params.hook_store.call(
            HOOK.BEFORE_PREPARE_EMBEDS,
            **hook_params,
        )

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
        k_log("interpolation prior embeddings: done")

        use_sampler(decoder, params["sampler"], task)

        mix_cnet_image = mix_cnet_image.resize((params["w"], params["h"]))
        hint = generate_hint(
            self,
            mix_cnet_image,
            mix_cnet_condition,
            mix_cnet_depth_estimator,
            self.params,
        )

        images = []
        prior_on_cpu = self.params("diffusers", "run_prior_on_cpu")

        hook_params["image_embeds"] = embeds.image_embeds
        hook_params["negative_image_embeds"] = embeds.negative_image_embeds
        hook_params["scheduler"] = decoder.scheduler
        hook_params["cnet_hint"] = hint
        execute_forced_hooks(HOOK.BEFORE_PREPARE_DECODER, params, hook_params)
        self.params.hook_store.call(
            HOOK.BEFORE_PREPARE_DECODER,
            **hook_params,
        )

        for _ in itertools.repeat(None, params["batch_count"]):
            current_batch = decoder(
                image=mix_cnet_image,
                image_embeds=(
                    embeds.image_embeds.half() if prior_on_cpu else embeds.image_embeds
                ),
                negative_image_embeds=embeds.negative_image_embeds.half(),
                strength=mix_cnet_img_strength,
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

            hook_params["batch"] = current_batch
            execute_forced_hooks(HOOK.BEFORE_BATCH_SAVE, params, hook_params)
            self.params.hook_store.call(
                HOOK.BEFORE_BATCH_SAVE,
                **hook_params,
            )

            images += self.create_batch_images(params, task, current_batch)
        k_log("decoder images: done")

        execute_forced_hooks(HOOK.BEFORE_TASK_QUIT, params, hook_params)
        self.params.hook_store.call(
            HOOK.BEFORE_TASK_QUIT,
            **hook_params,
        )

        return images
