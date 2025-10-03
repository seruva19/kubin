# This source code is licensed under the Apache License found in the
# LICENSE file in the current directory.
"""
The code has been adopted from Kandinsky-5
(https://github.com/ai-forever/Kandinsky-5/blob/main/kandinsky/generation_utils.py)
"""

import os

os.environ["TOKENIZERS_PARALLELISM"] = "False"

import torch
import time
from tqdm import tqdm

from .models.utils import fast_sta_nabla


def log_vram_usage(stage_name: str):
    """Log current and peak VRAM usage for a specific stage."""
    if torch.cuda.is_available():
        current_vram = torch.cuda.memory_allocated() / (1024**3)  # GB
        peak_vram = torch.cuda.max_memory_allocated() / (1024**3)  # GB
        print(
            f"  {stage_name} VRAM - Current: {current_vram:.2f} GB | Peak: {peak_vram:.2f} GB"
        )
    else:
        print(f"  {stage_name} - CUDA not available")


def get_sparse_params(conf, batch_embeds, device):
    assert conf.model.dit_params.patch_size[0] == 1
    T, H, W, _ = batch_embeds["visual"].shape
    print(f"\n{'='*80}")
    print(f"GET_SPARSE_PARAMS - Computing sequence length:")
    print(f"Visual embed shape (raw): {batch_embeds['visual'].shape} [T, H, W, C]")
    print(f"Patch size: {conf.model.dit_params.patch_size}")

    T, H, W = (
        T // conf.model.dit_params.patch_size[0],
        H // conf.model.dit_params.patch_size[1],
        W // conf.model.dit_params.patch_size[2],
    )

    print(f"After patching: T={T}, H={H}, W={W}")
    print(f"Attention type: {conf.model.attention.type}")

    if conf.model.attention.type == "nabla":
        print(f"Creating NABLA sparse params:")
        print(f"  STA dimensions: T={T}, H//8={H//8}, W//8={W//8}")
        print(f"  Sequence length will be: T*H*W = {T*H*W}")
        print(
            f"  Nabla params: P={conf.model.attention.P}, wT={conf.model.attention.wT}, wH={conf.model.attention.wH}, wW={conf.model.attention.wW}"
        )

        sta_mask = fast_sta_nabla(
            T,
            H // 8,
            W // 8,
            conf.model.attention.wT,
            conf.model.attention.wH,
            conf.model.attention.wW,
            device=device,
        )
        print(f"  STA mask created with shape: {sta_mask.shape}")

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
        print(f"  STA mask shape (after unsqueeze): {sparse_params['sta_mask'].shape}")
        print(f"{'='*80}\n")
    else:
        print(f"Using standard attention (no sparse params)")
        print(f"{'='*80}\n")
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
    return_loaded_models=False,
    magcache=False,
):
    bs, duration, height, width, dim = shape
    if duration == 1:
        type_of_content = "image"
    else:
        type_of_content = "video"

    # Start total timing
    total_start_time = time.time()

    # Log attention configuration
    attn_mode = os.environ.get("KD5_ATTENTION_MODE", "flash").lower()
    dit_arch = (
        conf.model.attention.type.upper()
        if hasattr(conf.model.attention, "type")
        else "STANDARD"
    )

    print(f"\n{'='*80}")
    print(f"🔧 ATTENTION CONFIGURATION:")
    print(f"  Kernel (all models): {attn_mode.upper()}")
    print(f"  DIT sparse pattern: {dit_arch}")
    print(f"{'='*80}\n")

    # Text embedder should already be on GPU if offload is enabled (moved in pipeline)
    print(f"Offload: Phase 1 - Text encoding on {text_embedder_device}")

    phase1_start = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        bs_text_embed, text_cu_seqlens = text_embedder.encode(
            [caption], type_of_content=type_of_content
        )
        bs_null_text_embed, null_text_cu_seqlens = text_embedder.encode(
            [negative_caption], type_of_content=type_of_content
        )

    phase1_time = time.time() - phase1_start
    print(f"  ⏱️  Phase 1 (Text Encoding) completed in {phase1_time:.2f}s")

    if offload:
        log_vram_usage("Text Embedder")
        if text_embedder_is_quantized:
            print("Offload: Phase 1 complete - Moving quantized text embedder to CPU")
            print(f"  → Using .to() method to preserve quantization state")
            try:
                if hasattr(text_embedder, "embedder") and hasattr(
                    text_embedder.embedder, "model"
                ):
                    text_embedder.embedder.model = text_embedder.embedder.model.to(
                        "cpu"
                    )
                if hasattr(text_embedder, "clip_embedder") and hasattr(
                    text_embedder.clip_embedder, "model"
                ):
                    text_embedder.clip_embedder.model = (
                        text_embedder.clip_embedder.model.to("cpu")
                    )
                print(f"  → Quantized text embedder moved to CPU")
            except Exception as e:
                print(f"  ⚠️  ERROR moving quantized text embedder: {e}")
            torch.cuda.empty_cache()
            import gc

            gc.collect()
        else:
            print(
                "Offload: Phase 1 complete - Moving text embedder to CPU (text processing done)"
            )
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

    # Load DIT if it wasn't loaded yet (deferred loading in offload mode)
    if dit is None and offload:
        print("DEFERRED LOADING: Loading DIT model now (after text processing)")
        from .models.dit import get_dit
        from .magcache_utils import set_magcache_params, disable_magcache
        from safetensors.torch import load_file

        # Get deferred loading params from somewhere - they should be passed in
        # For now we'll need to check a pipeline attribute or pass them as params
        # This is a limitation we need to address
        dit = get_dit(conf.model.dit_params)

        # Load checkpoint
        checkpoint_path = conf.model.checkpoint_path
        is_windows_path = len(checkpoint_path) > 1 and checkpoint_path[1] == ":"
        is_unix_path = checkpoint_path.startswith("/") or checkpoint_path.startswith(
            "./"
        )
        is_hf_repo = "/" in checkpoint_path and not is_windows_path and not is_unix_path

        if is_hf_repo:
            from huggingface_hub import snapshot_download

            cache_dir = os.environ.get("KD50_CACHE_DIR", "./weights/")
            cache_dir = os.path.abspath(os.path.normpath(cache_dir))
            model_path = snapshot_download(repo_id=checkpoint_path, cache_dir=cache_dir)
            checkpoint_path = model_path
        else:
            checkpoint_path = os.path.abspath(checkpoint_path)

        possible_filenames = [
            "model.safetensors",
            "kandinsky5lite_t2v_distilled16steps_5s.safetensors",
            "kandinsky5lite_t2v_distilled16steps_10s.safetensors",
            "kandinsky5lite_t2v_sft_5s.safetensors",
            "kandinsky5lite_t2v_pretrain_5s.safetensors",
            "kandinsky5lite_t2v_distil_5s.safetensors",
            "kandinsky5lite_t2v_nocfg_5s.safetensors",
            "kandinsky5lite_t2v_sft_10s.safetensors",
            "kandinsky5lite_t2v_pretrain_10s.safetensors",
            "kandinsky5lite_t2v_distil_10s.safetensors",
            "kandinsky5lite_t2v_nocfg_10s.safetensors",
        ]

        full_model_path = None
        for filename in possible_filenames:
            test_path = os.path.join(checkpoint_path, filename)
            if os.path.exists(test_path):
                full_model_path = test_path
                break
            test_path_in_model = os.path.join(checkpoint_path, "model", filename)
            if os.path.exists(test_path_in_model):
                full_model_path = test_path_in_model
                break

        if full_model_path is None:
            raise FileNotFoundError(f"No model file found in {checkpoint_path}")

        print(f"   → Loading DIT weights from: {full_model_path}")

        # Load weights - for CUDA, load directly to GPU to save one copy operation
        if device.type == "cuda":
            print(f"   → Loading weights directly to {device}")
            state_dict = load_file(full_model_path, device=str(device))
            dit.load_state_dict(state_dict, assign=True)
            # Parameters are now on GPU, but buffers are still on CPU
            # .to() is smart - it only moves what's not already there
            dit = dit.to(device)
            print(f"   → Model fully moved to {device}")
        else:
            state_dict = load_file(full_model_path)
            dit.load_state_dict(state_dict, assign=True)

        # Apply quantization if needed (passed via generate_sample params)
        if dit_is_quantized:
            from .model_kd50_env import quantize_with_torch_ao

            print("   → Applying int8 quantization to DIT")
            dit = quantize_with_torch_ao(dit)

        # Apply magcache if needed (deferred loading during offload)
        if (
            magcache
            and hasattr(conf, "magcache")
            and hasattr(conf.magcache, "mag_ratios")
        ):
            print("   → Setting up Magcache (deferred)")
            mag_ratios = conf.magcache.mag_ratios
            no_cfg = abs(guidance_weight - 1.0) < 0.01
            calibrate_mode = os.environ.get("MAGCACHE_CALIBRATE", "0") == "1"
            set_magcache_params(
                dit, mag_ratios, num_steps, no_cfg, calibrate=calibrate_mode
            )
        elif magcache:
            print("   ⚠️  Magcache requested but config missing mag_ratios")
        else:
            # Ensure magcache is disabled
            disable_magcache(dit)

        print(f"   → DIT loaded and ready (weights on {next(dit.parameters()).device})")

    current_device = next(dit.parameters()).device

    if dit_is_quantized and offload and current_device.type == "cpu":
        print(f"Offload: Phase 2 - Moving quantized DIT from CPU to {device}")
        print(f"  → Using .to() method to preserve quantization state")
        try:
            dit = dit.to(device)
            actual_device = next(dit.parameters()).device
            print(f"  → DIT now on: {actual_device}")
        except Exception as e:
            print(f"  ⚠️  ERROR moving quantized DIT: {e}")
    elif dit_is_quantized and offload:
        print(f"Offload: Phase 2 - Quantized DIT already on {current_device} (ready)")
    elif offload and current_device.type == "cpu":
        print(
            f"Offload: Phase 2 - Moving DIT from CPU to {device} for latent generation"
        )
        print(f"  → Target device: {device}, type: {device.type}")
        try:
            dit.to(device)  # This modifies in-place
            actual_device = next(dit.parameters()).device
            print(f"  → DIT now on: {actual_device} (type: {actual_device.type})")
            if actual_device.type != "cuda":
                print(
                    f"  ⚠️  WARNING: DIT failed to move to GPU! Still on {actual_device}"
                )
            else:
                print("  ✓ TEXT PROCESSING COMPLETE - Starting DIT LATENT GENERATION")
        except Exception as e:
            print(f"  ⚠️  ERROR moving DIT to GPU: {e}")
    elif offload and current_device.type == "cuda":
        print(
            f"Offload: Phase 2 - DIT already on {current_device} (ready for generation)"
        )
    else:
        print(f"Offload: Phase 2 - DIT already on {current_device} (offload disabled)")

    dit_gen_device = next(dit.parameters()).device
    print(
        f"Offload: Phase 2 - Starting latent generation with DIT on: {dit_gen_device}"
    )
    if dit_gen_device.type == "cpu" and device.type == "cuda":
        print(
            "  ⚠️  CRITICAL: DIT is on CPU but should be on GPU! This will be VERY slow!"
        )

    # Handle magcache state: setup/reset if enabled, disable if not requested
    from .magcache_utils import (
        reset_magcache_state,
        disable_magcache,
        set_magcache_params,
    )

    if magcache:
        # Check if magcache is already enabled, if not, set it up
        if not (hasattr(dit, "_magcache_enabled") and dit._magcache_enabled):
            # Magcache not enabled - set it up now
            if hasattr(conf, "magcache") and hasattr(conf.magcache, "mag_ratios"):
                print("   → Setting up Magcache (was disabled, now enabling)")
                mag_ratios = conf.magcache.mag_ratios
                no_cfg = abs(guidance_weight - 1.0) < 0.01
                calibrate_mode = os.environ.get("MAGCACHE_CALIBRATE", "0") == "1"
                set_magcache_params(
                    dit, mag_ratios, num_steps, no_cfg, calibrate=calibrate_mode
                )
            else:
                print("   ⚠️  Magcache requested but config missing mag_ratios")
        else:
            # Magcache already enabled - just reset state for new generation
            reset_magcache_state(dit)
    else:
        # Disable magcache if it was previously enabled but now disabled in UI
        disable_magcache(dit)

    phase2_start = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        # Use autocast only if CUDA is available, otherwise run without it
        autocast_context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device.type == "cuda"
            else torch.no_grad()
        )
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

    phase2_time = time.time() - phase2_start
    print(f"  ⏱️  Phase 2 (DIT Latent Generation) completed in {phase2_time:.2f}s")

    if offload:
        log_vram_usage("DIT")
        if dit_is_quantized:
            print("Offload: Phase 2 complete - Moving quantized DIT back to CPU")
            print(f"  → Using .to() method to preserve quantization state")
            try:
                dit = dit.to("cpu")
                dit_after = next(dit.parameters()).device
                print(f"  → Quantized DIT moved back to: {dit_after}")
            except Exception as e:
                print(f"  ⚠️  ERROR moving quantized DIT to CPU: {e}")
        else:
            print("Offload: Phase 2 complete - Moving DIT back to CPU")
            dit.to("cpu")
            dit_after = next(dit.parameters()).device
            print(f"  → DIT moved back to: {dit_after}")
        torch.cuda.empty_cache()
        import gc

        gc.collect()
    else:
        torch.cuda.empty_cache()

    # Load VAE if it wasn't loaded yet (deferred loading in offload mode)
    if vae is None and offload:
        print("DEFERRED LOADING: Loading VAE model now (for video decoding)")
        from .models.vae import build_vae

        vae_low_vram_mode = conf.model.vae.get("low_vram_mode", False)
        vae = build_vae(conf.model.vae, low_vram_mode=vae_low_vram_mode)
        vae = vae.eval()
        print(f"   → VAE loaded and ready")

    if offload:
        print(f"Offload: Phase 3 - Moving VAE to {vae_device} for video decoding")
        vae.to(vae_device)
        vae_actual = next(vae.parameters()).device
        print(f"  → VAE now on: {vae_actual}")

    print("Offload: Phase 3 - VAE decoding latents to final video...")

    phase3_start = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        # Use autocast only if CUDA is available, otherwise run without it
        autocast_context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device.type == "cuda"
            else torch.no_grad()
        )
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

    phase3_time = time.time() - phase3_start
    print(f"  ⏱️  Phase 3 (VAE Decoding) completed in {phase3_time:.2f}s")

    if offload:
        log_vram_usage("VAE")
        print("Offload: Phase 3 complete - Moving VAE back to CPU")
        vae.to("cpu")
        vae_after = next(vae.parameters()).device
        print(f"  → VAE moved back to: {vae_after}")
        torch.cuda.empty_cache()
        import gc

        gc.collect()
    else:
        torch.cuda.empty_cache()

    total_time = time.time() - total_start_time
    print(f"\n{'='*60}")
    print(f"⏱️  TOTAL GENERATION TIME: {total_time:.2f}s")
    print(
        f"  Phase 1 (Text Encoding):     {phase1_time:.2f}s ({phase1_time/total_time*100:.1f}%)"
    )
    print(
        f"  Phase 2 (DIT Generation):    {phase2_time:.2f}s ({phase2_time/total_time*100:.1f}%)"
    )
    print(
        f"  Phase 3 (VAE Decoding):      {phase3_time:.2f}s ({phase3_time/total_time*100:.1f}%)"
    )
    print(f"{'='*60}\n")
    print("Offload: All phases complete - Video generation finished!")

    if return_loaded_models:
        return images, dit, vae
    return images
