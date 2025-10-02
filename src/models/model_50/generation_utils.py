# This source code is licensed under the Apache License found in the
# LICENSE file in the current directory.
"""
The code has been adopted from Kandinsky-5
(https://github.com/ai-forever/Kandinsky-5/blob/main/kandinsky/generation_utils.py)
"""

import os

os.environ["TOKENIZERS_PARALLELISM"] = "False"

import torch
from tqdm import tqdm

from .models.utils import fast_sta_nabla


def get_sparse_params(conf, batch_embeds, device):
    assert conf.model.dit_params.patch_size[0] == 1
    T, H, W, _ = batch_embeds["visual"].shape
    T, H, W = (
        T // conf.model.dit_params.patch_size[0],
        H // conf.model.dit_params.patch_size[1],
        W // conf.model.dit_params.patch_size[2],
    )
    if conf.model.attention.type == "nabla":
        sta_mask = fast_sta_nabla(
            T,
            H // 8,
            W // 8,
            conf.model.attention.wT,
            conf.model.attention.wH,
            conf.model.attention.wW,
            device=device,
        )
        sparse_params = {
            "sta_mask": sta_mask.unsqueeze_(0).unsqueeze_(0),
            "attention_type": conf.model.attention.type,
            "to_fractal": True,
            "P": conf.model.attention.P,
            "wT": conf.model.attention.wT,
            "wW": conf.model.attention.wW,
            "wH": conf.model.attention.wH,
            "add_sta": conf.model.attention.add_sta,
            "visual_shape": (T, H, W),
            "method": getattr(conf.model.attention, "method", "topcdf"),
        }
    else:
        sparse_params = None

    return sparse_params


@torch.no_grad()
def get_velocity(
    dit,
    x,
    t,
    text_embeds,
    null_text_embeds,
    visual_rope_pos,
    text_rope_pos,
    null_text_rope_pos,
    guidance_weight,
    conf,
    sparse_params=None,
):
    pred_velocity = dit(
        x,
        text_embeds["text_embeds"],
        text_embeds["pooled_embed"],
        t * 1000,
        visual_rope_pos,
        text_rope_pos,
        scale_factor=conf.metrics.scale_factor,
        sparse_params=sparse_params,
    )
    if abs(guidance_weight - 1.0) > 1e-6:
        uncond_pred_velocity = dit(
            x,
            null_text_embeds["text_embeds"],
            null_text_embeds["pooled_embed"],
            t * 1000,
            visual_rope_pos,
            null_text_rope_pos,
            scale_factor=conf.metrics.scale_factor,
            sparse_params=sparse_params,
        )
        pred_velocity = uncond_pred_velocity + guidance_weight * (
            pred_velocity - uncond_pred_velocity
        )
    return pred_velocity


@torch.no_grad()
def generate(
    model,
    device,
    shape,
    num_steps,
    text_embeds,
    null_text_embeds,
    visual_rope_pos,
    text_rope_pos,
    null_text_rope_pos,
    guidance_weight,
    scheduler_scale,
    conf,
    progress=False,
    seed=6554,
):
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    img = torch.randn(*shape, device=device, generator=g)

    sparse_params = get_sparse_params(conf, {"visual": img}, device)
    timesteps = torch.linspace(1, 0, num_steps + 1, device=device)
    timesteps = scheduler_scale * timesteps / (1 + (scheduler_scale - 1) * timesteps)

    for timestep, timestep_diff in tqdm(
        list(zip(timesteps[:-1], torch.diff(timesteps)))
    ):
        time = timestep.unsqueeze(0)
        if model.visual_cond:
            visual_cond = torch.zeros_like(img)
            visual_cond_mask = torch.zeros(
                [*img.shape[:-1], 1], dtype=img.dtype, device=img.device
            )
            model_input = torch.cat([img, visual_cond, visual_cond_mask], dim=-1)
        else:
            model_input = img
        pred_velocity = get_velocity(
            model,
            model_input,
            time,
            text_embeds,
            null_text_embeds,
            visual_rope_pos,
            text_rope_pos,
            null_text_rope_pos,
            guidance_weight,
            conf,
            sparse_params=sparse_params,
        )
        img = img + timestep_diff * pred_velocity
    return img


def generate_sample(
    shape,
    caption,
    dit,
    vae,
    conf,
    text_embedder,
    num_steps=25,
    guidance_weight=5.0,
    scheduler_scale=1,
    negative_caption="",
    seed=6554,
    device="cuda",
    vae_device="cuda",
    text_embedder_device="cuda",
    progress=True,
    offload=False,
    dit_is_quantized=False,
    text_embedder_is_quantized=False,
):
    bs, duration, height, width, dim = shape
    if duration == 1:
        type_of_content = "image"
    else:
        type_of_content = "video"

    # Text embedder should already be on GPU if offload is enabled (moved in pipeline)
    print(f"Sequential offload: Phase 1 - Text encoding on {text_embedder_device}")
    with torch.no_grad():
        bs_text_embed, text_cu_seqlens = text_embedder.encode(
            [caption], type_of_content=type_of_content
        )
        bs_null_text_embed, null_text_cu_seqlens = text_embedder.encode(
            [negative_caption], type_of_content=type_of_content
        )

    if offload:
        if text_embedder_is_quantized:
            print("Sequential offload: Phase 1 complete - Moving quantized text embedder to CPU")
            print(f"  → Using .to() method to preserve quantization state")
            try:
                # Text embedder has nested wrappers - move all inner models using .to()
                if hasattr(text_embedder, 'embedder') and hasattr(text_embedder.embedder, 'model'):
                    # Move Qwen model
                    text_embedder.embedder.model = text_embedder.embedder.model.to("cpu")
                if hasattr(text_embedder, 'clip_embedder') and hasattr(text_embedder.clip_embedder, 'model'):
                    # Move CLIP model
                    text_embedder.clip_embedder.model = text_embedder.clip_embedder.model.to("cpu")
                print(f"  → Quantized text embedder moved to CPU")
            except Exception as e:
                print(f"  ⚠️  ERROR moving quantized text embedder: {e}")
            torch.cuda.empty_cache()
            import gc
            gc.collect()
        else:
            print("Sequential offload: Phase 1 complete - Moving text embedder to CPU (text processing done)")
            # Use the custom .to() method which handles the wrapper structure
            text_embedder.to("cpu")
            torch.cuda.empty_cache()
            import gc
            gc.collect()

    for key in bs_text_embed:
        bs_text_embed[key] = bs_text_embed[key].to(device=device)
        bs_null_text_embed[key] = bs_null_text_embed[key].to(device=device)
    text_cu_seqlens = text_cu_seqlens.to(device=device)[-1].item()
    null_text_cu_seqlens = null_text_cu_seqlens.to(device=device)[-1].item()

    visual_rope_pos = [
        torch.arange(duration),
        torch.arange(shape[-3] // conf.model.dit_params.patch_size[1]),
        torch.arange(shape[-2] // conf.model.dit_params.patch_size[2]),
    ]
    text_rope_pos = torch.arange(text_cu_seqlens)
    null_text_rope_pos = torch.arange(null_text_cu_seqlens)

    current_device = next(dit.parameters()).device

    if dit_is_quantized and offload and current_device.type == "cpu":
        print(f"Sequential offload: Phase 2 - Moving quantized DIT from CPU to {device}")
        print(f"  → Using .to() method to preserve quantization state")
        try:
            # Use .to() method to move the entire quantized model
            dit = dit.to(device)
            actual_device = next(dit.parameters()).device
            print(f"  → DIT now on: {actual_device}")
            print(f"  ✓ Quantized DIT successfully moved to GPU!")
        except Exception as e:
            print(f"  ⚠️  ERROR moving quantized DIT: {e}")
    elif dit_is_quantized and offload:
        print(f"Sequential offload: Phase 2 - Quantized DIT already on {current_device} (ready)")
    elif offload and current_device.type == "cpu":
        print(f"Sequential offload: Phase 2 - Moving DIT from CPU to {device} for latent generation")
        print(f"  → Target device: {device}, type: {device.type}")
        try:
            dit.to(device)  # This modifies in-place
            actual_device = next(dit.parameters()).device
            print(f"  → DIT now on: {actual_device} (type: {actual_device.type})")
            if actual_device.type != "cuda":
                print(f"  ⚠️  WARNING: DIT failed to move to GPU! Still on {actual_device}")
                print(f"  → This is why you see high CPU and low GPU usage!")
            else:
                print(f"  ✓ DIT successfully moved to GPU!")
                print("  ✓ TEXT PROCESSING COMPLETE - Starting DIT LATENT GENERATION")
        except Exception as e:
            print(f"  ⚠️  ERROR moving DIT to GPU: {e}")
    elif offload and current_device.type == "cuda":
        print(f"Sequential offload: Phase 2 - DIT already on {current_device} (ready for generation)")
    else:
        print(f"Sequential offload: Phase 2 - DIT already on {current_device} (offload disabled)")

    # Verify DIT device before generation
    dit_gen_device = next(dit.parameters()).device
    print(f"Sequential offload: Phase 2 - Starting latent generation with DIT on: {dit_gen_device}")
    if dit_gen_device.type == "cpu" and device.type == "cuda":
        print("  ⚠️  CRITICAL: DIT is on CPU but should be on GPU! This will be VERY slow!")

    with torch.no_grad():
        # Use autocast only if CUDA is available, otherwise run without it
        autocast_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else torch.no_grad()
        with autocast_context:
            latent_visual = generate(
                dit,
                device,
                (bs * duration, height, width, dim),
                num_steps,
                bs_text_embed,
                bs_null_text_embed,
                visual_rope_pos,
                text_rope_pos,
                null_text_rope_pos,
                guidance_weight,
                scheduler_scale,
                conf,
                seed=seed,
                progress=progress,
            )

    if offload:
        if dit_is_quantized:
            print("Sequential offload: Phase 2 complete - Moving quantized DIT back to CPU")
            print(f"  → Using .to() method to preserve quantization state")
            try:
                dit = dit.to("cpu")
                dit_after = next(dit.parameters()).device
                print(f"  → Quantized DIT moved back to: {dit_after}")
            except Exception as e:
                print(f"  ⚠️  ERROR moving quantized DIT to CPU: {e}")
        else:
            print("Sequential offload: Phase 2 complete - Moving DIT back to CPU")
            dit.to("cpu")
            dit_after = next(dit.parameters()).device
            print(f"  → DIT moved back to: {dit_after}")
        torch.cuda.empty_cache()
        import gc
        gc.collect()
    else:
        torch.cuda.empty_cache()

    if offload:
        print(f"Sequential offload: Phase 3 - Moving VAE to {vae_device} for video decoding")
        vae.to(vae_device)
        vae_actual = next(vae.parameters()).device
        print(f"  → VAE now on: {vae_actual}")

    print("Sequential offload: Phase 3 - VAE decoding latents to final video...")
    with torch.no_grad():
        # Use autocast only if CUDA is available, otherwise run without it
        autocast_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else torch.no_grad()
        with autocast_context:
            images = latent_visual.reshape(
                bs,
                -1,
                latent_visual.shape[-3],
                latent_visual.shape[-2],
                latent_visual.shape[-1],
            )
            images = images.to(device=vae_device)
            images = (images / vae.config.scaling_factor).permute(0, 4, 1, 2, 3)
            images = vae.decode(images).sample
            images = ((images.clamp(-1.0, 1.0) + 1.0) * 127.5).to(torch.uint8)

    if offload:
        print("Sequential offload: Phase 3 complete - Moving VAE back to CPU")
        vae.to("cpu")
        vae_after = next(vae.parameters()).device
        print(f"  → VAE moved back to: {vae_after}")
        torch.cuda.empty_cache()
        import gc
        gc.collect()
    else:
        torch.cuda.empty_cache()
    print("Sequential offload: All phases complete - Video generation finished!")

    return images
