import gc
import torch

try:
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
    None


def prepare_weights_for_task(model, task):
    cache_dir = model.params("general", "cache_dir")
    device = model.params("general", "device")

    if model.pipe_prior is None:
        model.image_encoder = (
            CLIPVisionModelWithProjection.from_pretrained(
                "kandinsky-community/kandinsky-2-2-prior",
                subfolder="image_encoder",
                cache_dir=cache_dir,
            )
            .half()
            .to(device)
        )

        model.pipe_prior = KandinskyV22PriorPipeline.from_pretrained(
            "kandinsky-community/kandinsky-2-2-prior",
            image_encoder=model.image_encoder,
            torch_dtype=type_of_weights(model.params),
            cache_dir=cache_dir,
        )
    current_prior = model.pipe_prior

    if task == "text2img" or task == "mix" or task == "img2img":
        if model.t2i_pipe is None:
            flush_if_required(model, task)

            model.unet_2d = (
                UNet2DConditionModel.from_pretrained(
                    "kandinsky-community/kandinsky-2-2-decoder",
                    subfolder="unet",
                    cache_dir=cache_dir,
                )
                .half()
                .to(device)
            )

            model.t2i_pipe = KandinskyV22Pipeline.from_pretrained(
                "kandinsky-community/kandinsky-2-2-decoder",
                unet=model.unet_2d,
                torch_dtype=type_of_weights(model.params),
                cache_dir=cache_dir,
            )

        current_unet = model.t2i_pipe
        if task == "img2img":
            model.i2i_pipe = KandinskyV22Img2ImgPipeline(**model.t2i_pipe.components)
            current_unet = model.i2i_pipe

    if task == "text2img_cnet" or task == "img2img_cnet":
        if model.cnet_t2i_pipe is None:
            flush_if_required(model, task)

            model.unet_2d = (
                UNet2DConditionModel.from_pretrained(
                    "kandinsky-community/kandinsky-2-2-controlnet-depth",
                    subfolder="unet",
                    cache_dir=cache_dir,
                )
                .half()
                .to(device)
            )

            model.cnet_t2i_pipe = KandinskyV22ControlnetPipeline.from_pretrained(
                "kandinsky-community/kandinsky-2-2-controlnet-depth",
                unet=model.unet_2d,
                torch_dtype=type_of_weights(model.params),
                cache_dir=cache_dir,
            )

        current_unet = model.cnet_t2i_pipe
        if task == "img2img_cnet":
            model.pipe_prior_e2e = KandinskyV22PriorEmb2EmbPipeline(
                **model.pipe_prior.components
            )
            current_prior = model.pipe_prior_e2e
            model.cnet_i2i_pipe = KandinskyV22ControlnetImg2ImgPipeline(
                **model.cnet_t2i_pipe.components
            )
            current_unet = model.cnet_i2i_pipe

    elif task == "inpainting" or task == "outpainting":
        if model.inpaint_pipe is None:
            flush_if_required(model, task)

            model.unet_2d = (
                UNet2DConditionModel.from_pretrained(
                    "kandinsky-community/kandinsky-2-2-decoder-inpaint",
                    subfolder="unet",
                    cache_dir=cache_dir,
                )
                .half()
                .to(device)
            )

            model.inpaint_pipe = KandinskyV22InpaintPipeline.from_pretrained(
                "kandinsky-community/kandinsky-2-2-decoder-inpaint",
                torch_dtype=type_of_weights(model.params),
                unet=model.unet_2d,
                cache_dir=cache_dir,
            )

        current_unet = model.inpaint_pipe

    to_device(model.params, current_prior, current_unet)
    return current_prior, current_unet


def flush_if_required(model, target):
    clear_memory_targets = None

    if target in ["text2img_cnet", "img2img_cnet"]:
        clear_memory_targets = ["text2img", "inpainting"]
    elif target in ["text2img", "img2img", "mix"]:
        clear_memory_targets = ["text2img_cnet", "inpainting"]
    elif target in ["inpainting", "outpainting"]:
        clear_memory_targets = ["text2img", "text2img_cnet"]
    elif target is None:
        clear_memory_targets = [
            "text2img",
            "img2img",
            "mix",
            "inpainting",
            "outpainting",
            "text2img_cnet",
            "img2img_cnet",
        ]

    if clear_memory_targets is not None:
        print(
            f"following pipelines, if active, will be released for {target if target is not None else 'another model'}: {clear_memory_targets}"
        )
        offload_enabled = model.params("diffusers", "sequential_cpu_offload")

        if "prior" in clear_memory_targets:
            if model.pipe_prior is not None:
                print("releasing prior pipeline")
                if not offload_enabled:
                    model.pipe_prior.to("cpu")
                model.pipe_prior = None

            if model.pipe_prior_e2e is not None:
                print("releasing prior_e2e pipeline")
                if not offload_enabled:
                    model.pipe_prior_e2e.to("cpu")
                model.pipe_prior_e2e = None

        if any(
            value in clear_memory_targets for value in ["text2img", "img2img", "mix"]
        ):
            if model.t2i_pipe is not None:
                print("releasing t2i pipeline")
                if not offload_enabled:
                    model.t2i_pipe.to("cpu")
                model.t2i_pipe = None

            if model.i2i_pipe is not None:
                print("releasing i2i pipeline")
                if not offload_enabled:
                    model.i2i_pipe.to("cpu")
                model.i2i_pipe = None

        if any(
            value in clear_memory_targets for value in ["inpainting", "outpainting"]
        ):
            if model.inpaint_pipe is not None:
                print("releasing inpaint pipeline")
                if not offload_enabled:
                    model.inpaint_pipe.to("cpu")
                model.inpaint_pipe = None

        if any(
            value in clear_memory_targets for value in ["text2img_cnet", "img2img_cnet"]
        ):
            if model.cnet_t2i_pipe is not None:
                print("releasing t2i_cnet pipeline")
                if not offload_enabled:
                    model.cnet_t2i_pipe.to("cpu")
                model.cnet_t2i_pipe = None

            if model.cnet_i2i_pipe is not None:
                print("releasing i2i_cnet pipeline")
                if not offload_enabled:
                    model.cnet_i2i_pipe.to("cpu")
                model.cnet_i2i_pipe = None

        gc.collect()

        if model.params("general", "device") == "cuda":
            if torch.cuda.is_available():
                with torch.cuda.device("cuda"):
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()


def type_of_weights(k_params):
    return torch.float16 if k_params("diffusers", "half_precision_weights") else "auto"


def to_device(k_params, prior, unet):
    device = k_params("general", "device")

    prior.safety_checker = None
    prior.to(device)

    if k_params("diffusers", "sequential_cpu_offload"):
        prior.enable_sequential_cpu_offload()

    if k_params("diffusers", "enable_xformers"):
        unet.enable_xformers_memory_efficient_attention()
    else:
        unet.disable_xformers_memory_efficient_attention()

    if k_params("diffusers", "enable_sdpa_attention"):
        unet.unet.set_attn_processor(AttnAddedKVProcessor2_0())

    if k_params("diffusers", "channels_last_memory"):
        unet.unet.to(memory_format=torch.channels_last)

    if k_params("diffusers", "torch_code_compilation"):
        unet.unet = torch.compile(unet.unet, mode="reduce-overhead", fullgraph=True)

    unet.to(device)

    if k_params("diffusers", "sequential_cpu_offload"):
        unet.enable_sequential_cpu_offload()

    if k_params("diffusers", "full_model_offload"):
        unet.enable_model_cpu_offload()

    if k_params("diffusers", "enable_sliced_attention"):
        slice_size = k_params("diffusers", "attention_slice_size")
        unet.enable_attention_slicing(slice_size)
    else:
        unet.disable_attention_slicing()

    unet.safety_checker = None
