import gc
import torch
from models.model_diffusers22.patched.patched import KandinskyV22PipelinePatched
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
from utils.logging import k_log
import os

from transformers import CLIPVisionModelWithProjection
from diffusers.models import UNet2DConditionModel
from diffusers.models.attention_processor import AttnAddedKVProcessor2_0


def prepare_weights_for_task(model, task):
    if model.params("diffusers", "use_deterministic_algorithms"):
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
    else:
        if model.cublas_config is not None:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = model.cublas_config
        torch.use_deterministic_algorithms(False)

    torch.backends.cuda.matmul.allow_tf32 = model.params("diffusers", "use_tf32_mode")

    applied_optimizations = []

    cache_dir = model.params("general", "cache_dir")
    half_weights = model.params("diffusers", "half_precision_weights")
    run_prior_on_cpu = model.params("diffusers", "run_prior_on_cpu")
    device = model.params("general", "device")
    enable_xformers = model.params("diffusers", "enable_xformers")

    sequential_prior_offload = sequential_decoder_offload = model.params(
        "diffusers", "sequential_cpu_offload"
    )

    full_prior_offload = full_decoder_offload = model.params(
        "diffusers", "full_model_offload"
    )

    enable_sdp_attention = model.params("diffusers", "enable_sdp_attention")
    channels_last_memory = model.params("diffusers", "channels_last_memory")
    torch_code_compilation = model.params("diffusers", "torch_code_compilation")
    enable_sliced_attention = model.params("diffusers", "enable_sliced_attention")
    slice_size = model.params("diffusers", "attention_slice_size")

    xformers_available = False
    try:
        import xformers

        xformers_available = True
    except:
        None

    if model.pipe_prior is None:
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            "kandinsky-community/kandinsky-2-2-prior",
            subfolder="image_encoder",
            cache_dir=cache_dir,
            resume_download=True,
            # local_files_only=True,
            # device_map="auto",
        )

        if half_weights and not run_prior_on_cpu:
            image_encoder = image_encoder.half()

        if (
            not run_prior_on_cpu
            and not sequential_prior_offload
            and not full_prior_offload
        ):
            image_encoder = image_encoder.to(device)

        model.pipe_prior = KandinskyV22PriorPipelinePatched.from_pretrained(
            "kandinsky-community/kandinsky-2-2-prior",
            image_encoder=image_encoder,
            torch_dtype=torch.float32
            if run_prior_on_cpu
            else type_of_weights(model.params),
            cache_dir=cache_dir,
            resume_download=True,
        )

        if (
            not run_prior_on_cpu
            and not sequential_prior_offload
            and not full_prior_offload
        ):
            model.pipe_prior = model.pipe_prior.to(device)

    current_prior = model.pipe_prior

    if task == "text2img" or task == "mix" or task == "img2img":
        if model.t2i_pipe is None:
            flush_if_required(model, task)

            unet_2d = UNet2DConditionModel.from_pretrained(
                "kandinsky-community/kandinsky-2-2-decoder",
                subfolder="unet",
                cache_dir=cache_dir,
                resume_download=True,
            ).half()

            if not sequential_decoder_offload and not full_decoder_offload:
                unet_2d = unet_2d.to(device)

            model.t2i_pipe = KandinskyV22PipelinePatched.from_pretrained(
                "kandinsky-community/kandinsky-2-2-decoder",
                unet=unet_2d,
                torch_dtype=type_of_weights(model.params),
                cache_dir=cache_dir,
                resume_download=True,
            )

            if not sequential_decoder_offload and not full_decoder_offload:
                model.t2i_pipe = model.t2i_pipe.to(device)

        current_decoder = model.t2i_pipe
        if task == "img2img":
            model.i2i_pipe = KandinskyV22Img2ImgPipelinePatched(
                **model.t2i_pipe.components
            )
            current_decoder = model.i2i_pipe

    if task == "text2img_cnet" or task == "img2img_cnet" or task == "mix_cnet":
        if model.cnet_t2i_pipe is None:
            flush_if_required(model, task)

            unet_2d = UNet2DConditionModel.from_pretrained(
                "kandinsky-community/kandinsky-2-2-controlnet-depth",
                subfolder="unet",
                cache_dir=cache_dir,
                resume_download=True,
            ).half()

            if not sequential_decoder_offload and not full_decoder_offload:
                unet_2d = unet_2d.to(device)

            model.cnet_t2i_pipe = (
                KandinskyV22ControlnetImg2ImgPipelinePatched.from_pretrained(
                    "kandinsky-community/kandinsky-2-2-controlnet-depth",
                    unet=unet_2d,
                    torch_dtype=type_of_weights(model.params),
                    cache_dir=cache_dir,
                    resume_download=True,
                )
            )

            if not sequential_decoder_offload and not full_decoder_offload:
                model.cnet_t2i_pipe = model.cnet_t2i_pipe.to(device)

        current_decoder = model.cnet_t2i_pipe

        if task == "img2img_cnet":
            model.pipe_prior_e2e = KandinskyV22PriorEmb2EmbPipelinePatched(
                **model.pipe_prior.components
            )
            current_prior = model.pipe_prior_e2e

            model.cnet_i2i_pipe = KandinskyV22ControlnetImg2ImgPipelinePatched(
                **model.cnet_t2i_pipe.components
            )
            current_decoder = model.cnet_i2i_pipe

        if task == "mix_cnet":
            model.cnet_i2i_pipe = KandinskyV22ControlnetImg2ImgPipelinePatched(
                **model.cnet_t2i_pipe.components
            )
            current_decoder = model.cnet_i2i_pipe

    elif task == "inpainting" or task == "outpainting":
        if model.inpaint_pipe is None:
            flush_if_required(model, task)

            unet_2d = UNet2DConditionModel.from_pretrained(
                "kandinsky-community/kandinsky-2-2-decoder-inpaint",
                subfolder="unet",
                cache_dir=cache_dir,
                resume_download=True,
            ).half()

            if not sequential_decoder_offload and not full_decoder_offload:
                unet_2d = unet_2d.to(device)

            model.inpaint_pipe = KandinskyV22InpaintPipelinePatched.from_pretrained(
                "kandinsky-community/kandinsky-2-2-decoder-inpaint",
                torch_dtype=type_of_weights(model.params),
                unet=unet_2d,
                cache_dir=cache_dir,
                resume_download=True,
            )

            if not sequential_decoder_offload and not full_decoder_offload:
                model.inpaint_pipe = model.inpaint_pipe.to(device)

        current_decoder = model.inpaint_pipe

    prior_device = device
    if run_prior_on_cpu:
        prior_device = "cpu"
        applied_optimizations.append("prior on CPU")

    if enable_xformers:
        if xformers_available:
            from xformers.ops import (
                MemoryEfficientAttentionFlashAttentionOp,
                MemoryEfficientAttentionOp,
            )

            try:
                if prior_device != "cpu":
                    current_prior.enable_xformers_memory_efficient_attention(
                        attention_op=MemoryEfficientAttentionOp
                    )
                    applied_optimizations.append("xformers for prior")
            except:
                k_log("failed to apply xformers for prior")

            try:
                current_decoder.enable_xformers_memory_efficient_attention(
                    attention_op=MemoryEfficientAttentionOp
                )
                applied_optimizations.append("xformers for decoder")
            except:
                k_log("failed to apply xformers for decoder")
        else:
            k_log("xformers use requested, but no xformers installed")
    else:
        current_prior.disable_xformers_memory_efficient_attention()
        current_decoder.disable_xformers_memory_efficient_attention()

    if enable_sdp_attention:
        current_decoder.unet.set_attn_processor(AttnAddedKVProcessor2_0())
        applied_optimizations.append("forced sdp attention for decoder unet")

    if channels_last_memory:
        current_decoder.unet.to(memory_format=torch.channels_last)
        applied_optimizations.append("channels last memory for decoder unet")

    if torch_code_compilation:
        current_decoder.unet = torch.compile(
            current_decoder.unet, mode="reduce-overhead", fullgraph=True
        )
        applied_optimizations.append("torch compile for decoder unet")

    if sequential_prior_offload:
        if run_prior_on_cpu:
            k_log(
                "sequential offload for prior won't be applied, because prior generation on CPU is enabled"
            )
        else:
            if not model.pipe_info["sequential_prior_offload"]:
                current_prior.enable_sequential_cpu_offload()
                model.pipe_info["sequential_prior_offload"] = True
            applied_optimizations.append("sequential CPU offloading for prior")
    elif full_prior_offload:
        if run_prior_on_cpu:
            k_log(
                "full offload for prior won't be applied, because prior generation on CPU is enabled"
            )
        else:
            if not model.pipe_info["full_prior_offload"]:
                current_prior.enable_model_cpu_offload()
                model.pipe_info["full_prior_offload"] = True
            applied_optimizations.append("full model offloading for prior")
    else:
        # image_encoder.to(prior_device)
        # current_prior.to(prior_device)
        model.pipe_info["sequential_prior_offload"] = False
        model.pipe_info["full_prior_offload"] = False

    if sequential_decoder_offload:
        if not model.pipe_info["sequential_decoder_offload"]:
            current_decoder.enable_sequential_cpu_offload()
            model.pipe_info["sequential_decoder_offload"] = True
        applied_optimizations.append("sequential CPU offloading for decoder")
    elif full_decoder_offload:
        if not model.pipe_info["full_decoder_offload"]:
            current_decoder.enable_model_cpu_offload()
            model.pipe_info["full_decoder_offload"] = True
        applied_optimizations.append("full model offloading for decoder")
    else:
        # unet_2d.to(device)
        # current_decoder.to(device)
        model.pipe_info["sequential_decoder_offload"] = False
        model.pipe_info["full_decoder_offload"] = False

    if enable_sliced_attention:
        current_decoder.enable_attention_slicing(slice_size)
        applied_optimizations.append(
            f"attention slicing for decoder (slice_size={slice_size})"
        )
    else:
        current_decoder.disable_attention_slicing()

    k_log(
        f"optimizations: {'none' if len(applied_optimizations) == 0 else '; '.join(applied_optimizations)}"
    )

    current_prior.safety_checker = None
    current_decoder.safety_checker = None

    return current_prior, current_decoder


def clear_pipe_info(model):
    model.pipe_info = {
        "sequential_prior_offload": False,
        "sequential_decoder_offload": False,
        "full_prior_offload": False,
        "full_decoder_offload": False,
        "cnet_depth_estimator": None,
        "cnet_dmap_type": None,
    }


def flush_if_required(model, target):
    clear_memory_targets = None

    if target in ["text2img_cnet", "img2img_cnet", "mix_cnet"]:
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
            "mix_cnet",
        ]

    if clear_memory_targets is not None:
        k_log(
            f"following pipelines, if active, will be released{' for ' + target + ' task' if target is not None else ''}: {clear_memory_targets}"
        )
        offload_enabled = model.params("diffusers", "sequential_cpu_offload")

        if "prior" in clear_memory_targets:
            if model.pipe_prior is not None:
                k_log("releasing prior pipeline")
                if not offload_enabled:
                    model.pipe_prior.to("cpu")
                model.pipe_prior = None

            if model.pipe_prior_e2e is not None:
                k_log("releasing prior_e2e pipeline")
                if not offload_enabled:
                    model.pipe_prior_e2e.to("cpu")
                model.pipe_prior_e2e = None

        if any(
            value in clear_memory_targets for value in ["text2img", "img2img", "mix"]
        ):
            if model.t2i_pipe is not None:
                k_log("releasing t2i pipeline")
                if not offload_enabled:
                    model.t2i_pipe.to("cpu")
                model.t2i_pipe = None

            if model.i2i_pipe is not None:
                k_log("releasing i2i pipeline")
                if not offload_enabled:
                    model.i2i_pipe.to("cpu")
                model.i2i_pipe = None

        if any(
            value in clear_memory_targets for value in ["inpainting", "outpainting"]
        ):
            if model.inpaint_pipe is not None:
                k_log("releasing inpaint pipeline")
                if not offload_enabled:
                    model.inpaint_pipe.to("cpu")
                model.inpaint_pipe = None

        if any(
            value in clear_memory_targets for value in ["text2img_cnet", "img2img_cnet"]
        ):
            if model.cnet_t2i_pipe is not None:
                k_log("releasing t2i_cnet pipeline")
                if not offload_enabled:
                    model.cnet_t2i_pipe.to("cpu")
                model.cnet_t2i_pipe = None

            if model.cnet_i2i_pipe is not None:
                k_log("releasing i2i_cnet pipeline")
                if not offload_enabled:
                    model.cnet_i2i_pipe.to("cpu")
                model.cnet_i2i_pipe = None

            if model.pipe_info["cnet_depth_estimator"] is not None:
                model.pipe_info["cnet_depth_estimator"] = None

        gc.collect()
        device = model.params("general", "device")
        if device.startswith("cuda"):
            if torch.cuda.is_available():
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()

        model.config = {}
        clear_pipe_info(model)


def type_of_weights(k_params):
    return torch.float16 if k_params("diffusers", "half_precision_weights") else "auto"


def images_or_texts(images, texts):
    images_texts = []
    for i in range(len(images)):
        images_texts.append(texts[i] if images[i] is None else images[i])

    return images_texts


def execute_forced_hooks(hook_stage, params, hook_params):
    for hook in params.get(hook_stage, [lambda **hp: None]):
        hook(**hook_params)
