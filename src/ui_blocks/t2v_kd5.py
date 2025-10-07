import asyncio
import os
import gradio as gr
from omegaconf import OmegaConf
from ui_blocks.shared.ui_shared import SharedUI
from utils.gradio_ui import click_and_disable
from utils.storage import get_value
from utils.text import generate_prompt_from_wildcard
from models.model_50.ui_config_manager import UIConfigManager

block = "t2v_kd5"


def load_config_defaults(config_name, config_manager=None):
    try:
        if config_manager is None:
            config_manager = UIConfigManager()
        return config_manager.load_config(config_name)
    except Exception as e:
        print(f"Error loading config {config_name}: {e}")
        return None


def get_variant_defaults(config_defaults, variant):
    cfg = config_defaults.get(variant)

    if cfg:
        default_prompt = "A closeshot of beautiful blonde woman standing under the sun at the beach. Soft waves lapping at her feet and vibrant palm trees lining the distant coastline under a clear blue sky."
        default_negative = "Static, 2D cartoon, cartoon, 2d animation, paintings, images, worst quality, low quality, ugly, deformed, walking backwards"

        ui_settings = getattr(cfg, "ui_settings", None)

        return {
            "prompt": (
                ui_settings.prompt
                if ui_settings and hasattr(ui_settings, "prompt")
                else default_prompt
            ),
            "negative_prompt": (
                ui_settings.negative_prompt
                if ui_settings and hasattr(ui_settings, "negative_prompt")
                else default_negative
            ),
            "width": (
                ui_settings.width
                if ui_settings and hasattr(ui_settings, "width")
                else 512
            ),
            "height": (
                ui_settings.height
                if ui_settings and hasattr(ui_settings, "height")
                else 512
            ),
            "duration": cfg.model.get("duration", 10 if "10s" in variant else 5),
            "seed": (
                ui_settings.seed if ui_settings and hasattr(ui_settings, "seed") else -1
            ),
            "num_steps": cfg.model.num_steps,
            "guidance_weight": cfg.model.guidance_weight,
            "expand_prompts": (
                ui_settings.expand_prompts
                if ui_settings and hasattr(ui_settings, "expand_prompts")
                else False
            ),
            "use_offload": (
                ui_settings.use_offload
                if ui_settings and hasattr(ui_settings, "use_offload")
                else True
            ),
            "use_magcache": (
                ui_settings.use_magcache
                if ui_settings and hasattr(ui_settings, "use_magcache")
                else False
            ),
            "use_dit_int8_ao_quantization": (
                ui_settings.use_dit_int8_ao_quantization
                if ui_settings and hasattr(ui_settings, "use_dit_int8_ao_quantization")
                else False
            ),
            "use_save_quantized_weights": (
                ui_settings.use_save_quantized_weights
                if ui_settings and hasattr(ui_settings, "use_save_quantized_weights")
                else False
            ),
            "use_text_embedder_int8_ao_quantization": (
                ui_settings.use_text_embedder_int8_ao_quantization
                if ui_settings
                and hasattr(ui_settings, "use_text_embedder_int8_ao_quantization")
                else False
            ),
            "use_torch_compile_dit": (
                cfg.optimizations.use_torch_compile_dit
                if hasattr(cfg, "optimizations")
                and hasattr(cfg.optimizations, "use_torch_compile_dit")
                else True
            ),
            "use_torch_compile_vae": (
                cfg.optimizations.use_torch_compile_vae
                if hasattr(cfg, "optimizations")
                and hasattr(cfg.optimizations, "use_torch_compile_vae")
                else True
            ),
            "enhance_enable": (
                ui_settings.enhance_enable
                if ui_settings and hasattr(ui_settings, "enhance_enable")
                else False
            ),
            "enhance_weight": (
                float(ui_settings.enhance_weight)
                if ui_settings and hasattr(ui_settings, "enhance_weight")
                else 8
            ),
            "enhance_max_tokens": (
                int(ui_settings.enhance_max_tokens)
                if ui_settings and hasattr(ui_settings, "enhance_max_tokens")
                else 0
            ),
            "in_visual_dim": cfg.model.dit_params.in_visual_dim,
            "out_visual_dim": cfg.model.dit_params.out_visual_dim,
            "time_dim": cfg.model.dit_params.time_dim,
            "model_dim": cfg.model.dit_params.model_dim,
            "ff_dim": cfg.model.dit_params.ff_dim,
            "num_text_blocks": cfg.model.dit_params.num_text_blocks,
            "num_visual_blocks": cfg.model.dit_params.num_visual_blocks,
            "patch_size": str(cfg.model.dit_params.patch_size),
            "axes_dims": str(cfg.model.dit_params.axes_dims),
            "visual_cond": cfg.model.dit_params.visual_cond,
            "in_text_dim": cfg.model.dit_params.in_text_dim,
            "in_text_dim2": cfg.model.dit_params.in_text_dim2,
            "attention_type": cfg.model.attention.type,
            "attention_causal": cfg.model.attention.causal,
            "attention_local": cfg.model.attention.local,
            "attention_glob": cfg.model.attention.glob,
            "attention_window": cfg.model.attention.window,
            "nabla_P": cfg.model.attention.get("P", 0.9),
            "nabla_wT": cfg.model.attention.get("wT", 11),
            "nabla_wW": cfg.model.attention.get("wW", 3),
            "nabla_wH": cfg.model.attention.get("wH", 3),
            "nabla_add_sta": cfg.model.attention.get("add_sta", True),
            "nabla_method": cfg.model.attention.get("method", "topcdf"),
            "qwen_emb_size": cfg.model.text_embedder.qwen.emb_size,
            "qwen_max_length": cfg.model.text_embedder.qwen.max_length,
            "qwen_checkpoint": cfg.model.text_embedder.qwen.checkpoint_path.strip("/"),
            "clip_emb_size": cfg.model.text_embedder.clip.emb_size,
            "clip_max_length": cfg.model.text_embedder.clip.max_length,
            "clip_checkpoint": cfg.model.text_embedder.clip.checkpoint_path.strip("/"),
            "vae_checkpoint": cfg.model.vae.checkpoint_path,
            "vae_name": cfg.model.vae.name,
            "vae_tile_threshold": cfg.model.vae.get("tile_threshold", 450),
            "vae_low_vram_mode": cfg.model.vae.get("low_vram_mode", False),
            "model_checkpoint": cfg.model.checkpoint_path,
        }
    else:
        is_10s = "10s" in variant
        return {
            "prompt": "A closeshot of beautiful blonde woman standing under the sun at the beach. Soft waves lapping at her feet and vibrant palm trees lining the distant coastline under a clear blue sky.",
            "negative_prompt": "Static, 2D cartoon, cartoon, 2d animation, paintings, images, worst quality, low quality, ugly, deformed, walking backwards",
            "width": 512,
            "height": 512,
            "duration": 10 if is_10s else 5,
            "seed": -1,
            "num_steps": 50,
            "guidance_weight": 5.0,
            "expand_prompts": True,
            "use_offload": True,
            "use_magcache": False,
            "vae_tile_threshold": 450,
            "vae_low_vram_mode": False,
            "use_dit_int8_ao_quantization": False,
            "use_save_quantized_weights": False,
            "use_text_embedder_int8_ao_quantization": False,
            "use_torch_compile_dit": True,
            "use_torch_compile_vae": True,
            "enhance_enable": False,
            "enhance_weight": 3.4,
            "enhance_max_tokens": 256,
            "in_visual_dim": 16,
            "out_visual_dim": 16,
            "time_dim": 512,
            "model_dim": 1792,
            "ff_dim": 7168,
            "num_text_blocks": 2,
            "num_visual_blocks": 32,
            "patch_size": "[1, 2, 2]",
            "axes_dims": "[16, 24, 24]",
            "visual_cond": True,
            "in_text_dim": 3584,
            "in_text_dim2": 768,
            "attention_type": "nabla" if is_10s else "flash",
            "attention_causal": False,
            "attention_local": False,
            "attention_glob": False,
            "attention_window": 3,
            "nabla_P": 0.9,
            "nabla_wT": 11,
            "nabla_wW": 3,
            "nabla_wH": 3,
            "nabla_add_sta": True,
            "nabla_method": "topcdf",
            "qwen_emb_size": 3584,
            "qwen_max_length": 256,
            "qwen_checkpoint": "Qwen/Qwen2.5-VL-7B-Instruct",
            "clip_emb_size": 768,
            "clip_max_length": 77,
            "clip_checkpoint": "openai/clip-vit-large-patch14",
            "vae_checkpoint": "hunyuanvideo-community/HunyuanVideo",
            "vae_name": "hunyuan",
            "model_checkpoint": f"ai-forever/Kandinsky-5.0-T2V-Lite-{get_model_name_from_variant(variant)}",
        }


def get_model_name_from_variant(variant):
    variant_map = {
        "5s_sft": "sft-5s",
        "5s_pretrain": "pretrain-5s",
        "5s_nocfg": "nocfg-5s",
        "5s_distil": "distilled16steps-5s",
        "10s_sft": "sft-10s",
        "10s_pretrain": "pretrain-10s",
        "10s_nocfg": "nocfg-10s",
        "10s_distil": "distilled16steps-10s",
    }
    return variant_map.get(variant, "sft-5s")


def t2v_kd5_ui(generate_fn, shared: SharedUI, tabs, session):
    augmentations = shared.create_ext_augment_blocks("t2v_kd5")
    value = lambda name, def_value: get_value(shared.storage, block, name, def_value)

    config_manager = UIConfigManager()

    config_defaults = {}
    variants = [
        "5s_sft",
        "5s_pretrain",
        "5s_nocfg",
        "5s_distil",
        "10s_sft",
        "10s_pretrain",
        "10s_nocfg",
        "10s_distil",
    ]
    for variant in variants:
        config_defaults[variant] = load_config_defaults(variant, config_manager)

    initial_variant = value("config_variant", "5s_sft")
    defaults = get_variant_defaults(config_defaults, initial_variant)

    with gr.Row() as t2v_kd5_block:
        t2v_kd5_block.elem_classes = ["t2v_kd5_block"]

        with gr.Column(scale=2) as t2v_kd5_params:
            augmentations["ui_before_prompt"]()

            with gr.Row():
                config_variant = gr.Radio(
                    choices=[
                        ("5s-SFT", "5s_sft"),
                        ("5s-PT", "5s_pretrain"),
                        ("5s-CFG-DSTL", "5s_nocfg"),
                        ("5s-DSTL", "5s_distil"),
                        ("10s-SFT", "10s_sft"),
                        ("10s-PT", "10s_pretrain"),
                        ("10s-CFG-DSTL", "10s_nocfg"),
                        ("10s-DSTL", "10s_distil"),
                    ],
                    value=initial_variant,
                    label="Model Variant",
                    info="Select model variant. All UI settings are saved per variant on Generate. Use Reset to restore defaults.",
                )

            with gr.Row():
                prompt = gr.TextArea(
                    value=defaults["prompt"],
                    label="Prompt",
                    placeholder="",
                    lines=4,
                )

            with gr.Row():
                negative_prompt = gr.TextArea(
                    value=defaults.get(
                        "negative_prompt",
                        "Static, 2D cartoon, cartoon, 2d animation, paintings, images, worst quality, low quality, ugly, deformed, walking backwards",
                    ),
                    label="Negative Prompt",
                    placeholder="",
                    lines=2,
                )

            augmentations["ui_before_params"]()

            with gr.Row():
                with gr.Column():
                    width = gr.Number(
                        value=defaults["width"],
                        label="Width",
                        precision=0,
                        minimum=128,
                        maximum=2048,
                        step=64,
                    )
                    height = gr.Number(
                        value=defaults["height"],
                        label="Height",
                        precision=0,
                        minimum=128,
                        maximum=2048,
                        step=64,
                    )
                with gr.Column():
                    duration = gr.Slider(
                        minimum=1,
                        maximum=30,
                        value=defaults["duration"],
                        step=1,
                        label="Duration",
                    )
                    seed = gr.Number(value=defaults["seed"], label="Seed", precision=0)

            with gr.Accordion("MODEL CONFIGURATION", open=True):
                components = {}

                with gr.Row():
                    components["num_steps"] = gr.Slider(
                        minimum=1,
                        maximum=200,
                        value=defaults["num_steps"],
                        label="Number of Steps",
                        step=1,
                    )
                    components["guidance_weight"] = gr.Slider(
                        minimum=0.0,
                        maximum=20.0,
                        value=defaults["guidance_weight"],
                        label="Guidance Weight",
                        step=0.5,
                    )

                with gr.Row():
                    components["expand_prompts"] = gr.Checkbox(
                        value=defaults["expand_prompts"],
                        label="Expand Prompts",
                    )

                with gr.Accordion("Optimization Options", open=True):
                    with gr.Row():
                        with gr.Column():
                            components["use_offload"] = gr.Checkbox(
                                value=defaults["use_offload"],
                                label="Use Offloading",
                            )
                            components["use_magcache"] = gr.Checkbox(
                                value=defaults["use_magcache"],
                                label="Use MagCache",
                            )
                        with gr.Column():
                            components["use_torch_compile_dit"] = gr.Checkbox(
                                value=defaults["use_torch_compile_dit"],
                                label="Use torch.compile (DiT)",
                            )
                            components["use_torch_compile_vae"] = gr.Checkbox(
                                value=defaults["use_torch_compile_vae"],
                                label="Use torch.compile (VAE)",
                            )
                        with gr.Column(visible=False):
                            components["use_dit_int8_ao_quantization"] = gr.Checkbox(
                                value=defaults["use_dit_int8_ao_quantization"],
                                label="Use DiT INT8",
                                interactive=False,
                            )
                            components["use_text_embedder_int8_ao_quantization"] = (
                                gr.Checkbox(
                                    value=defaults[
                                        "use_text_embedder_int8_ao_quantization"
                                    ],
                                    label="Use Embedder INT8",
                                    interactive=False,
                                )
                            )
                            components["use_save_quantized_weights"] = gr.Checkbox(
                                value=defaults["use_save_quantized_weights"],
                                label="Save Quantized",
                            )

                with gr.Accordion("Attention Parameters", open=True):
                    with gr.Row():
                        if defaults["attention_type"] == "nabla":
                            attention_impl_default = "nabla"
                        else:
                            # For flash/sdpa, default to flash_fa2 (matching reset behavior)
                            attention_impl_default = "flash_fa2"

                        components["attention_implementation"] = gr.Radio(
                            choices=[
                                ("Nabla", "nabla"),
                                ("Flash", "flash_fa2"),
                                ("SDPA", "sdpa"),
                                ("Sage", "sage"),
                            ],
                            value=attention_impl_default,
                            label="Attention Implementation",
                            scale=2,
                        )
                        with gr.Column():
                            components["attention_causal"] = gr.Checkbox(
                                label="Causal",
                                value=defaults["attention_causal"],
                                interactive=False,
                            )
                            components["attention_local"] = gr.Checkbox(
                                label="Local",
                                value=defaults["attention_local"],
                                interactive=False,
                            )
                            components["attention_glob"] = gr.Checkbox(
                                label="Global",
                                value=defaults["attention_glob"],
                                interactive=False,
                            )
                        components["attention_window"] = gr.Number(
                            label="Window Size", value=defaults["attention_window"]
                        )

                with gr.Accordion("DiT Parameters", open=False):
                    with gr.Row():
                        components["in_visual_dim"] = gr.Number(
                            interactive=True,
                            label="Input Visual Dimension",
                            precision=0,
                            value=defaults["in_visual_dim"],
                        )
                        components["out_visual_dim"] = gr.Number(
                            interactive=True,
                            precision=0,
                            label="Output Visual Dimension",
                            value=defaults["out_visual_dim"],
                        )
                        components["time_dim"] = gr.Number(
                            interactive=True,
                            label="Time Dimension",
                            value=defaults["time_dim"],
                            precision=0,
                        )
                        components["model_dim"] = gr.Number(
                            precision=0,
                            interactive=True,
                            label="Model Dimension",
                            value=defaults["model_dim"],
                        )

                    with gr.Row():
                        components["ff_dim"] = gr.Number(
                            interactive=True,
                            precision=0,
                            label="Feed Forward Dimension",
                            value=defaults["ff_dim"],
                        )
                        components["num_text_blocks"] = gr.Number(
                            precision=0,
                            interactive=True,
                            label="Number of Text Blocks",
                            value=defaults["num_text_blocks"],
                        )
                        components["num_visual_blocks"] = gr.Number(
                            precision=0,
                            interactive=True,
                            label="Number of Visual Blocks",
                            value=defaults["num_visual_blocks"],
                        )

                    with gr.Row():
                        components["patch_size"] = gr.Textbox(
                            interactive=True,
                            max_lines=1,
                            label="Patch Size",
                            value=defaults["patch_size"],
                        )
                        components["axes_dims"] = gr.Textbox(
                            interactive=True,
                            max_lines=1,
                            label="Axes Dimensions",
                            value=defaults["axes_dims"],
                        )
                        components["visual_cond"] = gr.Checkbox(
                            value=defaults["visual_cond"],
                            label="Visual Conditioning",
                        )

                    with gr.Row():
                        components["in_text_dim"] = gr.Number(
                            interactive=True,
                            precision=0,
                            label="Input Text Dimension (Qwen)",
                            value=defaults["in_text_dim"],
                        )
                        components["in_text_dim2"] = gr.Number(
                            interactive=True,
                            precision=0,
                            label="Input Text Dimension 2 (CLIP)",
                            value=defaults["in_text_dim2"],
                        )

                with gr.Accordion("Nabla Attention Parameters", open=False):
                    with gr.Row():
                        components["nabla_P"] = gr.Number(
                            interactive=True,
                            label="P",
                            value=defaults["nabla_P"],
                        )
                        components["nabla_wT"] = gr.Number(
                            interactive=True,
                            precision=0,
                            label="wT",
                            value=defaults["nabla_wT"],
                        )
                        components["nabla_wW"] = gr.Number(
                            interactive=True,
                            precision=0,
                            label="wW",
                            value=defaults["nabla_wW"],
                        )
                        components["nabla_wH"] = gr.Number(
                            interactive=True,
                            precision=0,
                            label="wH",
                            value=defaults["nabla_wH"],
                        )
                    with gr.Row():
                        components["nabla_add_sta"] = gr.Checkbox(
                            value=defaults["nabla_add_sta"],
                            label="Add STA",
                        )
                        components["nabla_method"] = gr.Dropdown(
                            interactive=True,
                            label="Method",
                            value=defaults["nabla_method"],
                            choices=["topcdf", "top"],
                        )

                with gr.Accordion("Text Embedder Configuration", open=False):
                    with gr.Row():
                        components["qwen_emb_size"] = gr.Number(
                            precision=0,
                            interactive=True,
                            label="Qwen Embedding Size",
                            value=defaults["qwen_emb_size"],
                        )
                        components["qwen_max_length"] = gr.Number(
                            precision=0,
                            interactive=True,
                            label="Qwen Max Length",
                            value=defaults["qwen_max_length"],
                        )
                        components["qwen_checkpoint"] = gr.Textbox(
                            interactive=True,
                            max_lines=1,
                            label="Qwen Checkpoint Path",
                            value=defaults["qwen_checkpoint"],
                        )

                    with gr.Row():
                        components["clip_emb_size"] = gr.Number(
                            precision=0,
                            interactive=True,
                            label="CLIP Embedding Size",
                            value=defaults["clip_emb_size"],
                        )
                        components["clip_max_length"] = gr.Number(
                            precision=0,
                            interactive=True,
                            label="CLIP Max Length",
                            value=defaults["clip_max_length"],
                        )
                        components["clip_checkpoint"] = gr.Textbox(
                            interactive=True,
                            max_lines=1,
                            label="CLIP Checkpoint Path",
                            value=defaults["clip_checkpoint"],
                        )

                with gr.Accordion("VAE & Model Paths", open=False):
                    with gr.Row():
                        components["vae_checkpoint"] = gr.Textbox(
                            interactive=True,
                            max_lines=1,
                            label="VAE Checkpoint Path",
                            value=defaults["vae_checkpoint"],
                        )
                        components["vae_name"] = gr.Textbox(
                            interactive=True,
                            max_lines=1,
                            label="VAE Name",
                            value=defaults["vae_name"],
                        )
                        components["vae_tile_threshold"] = gr.Slider(
                            minimum=0,
                            maximum=1024,
                            value=defaults["vae_tile_threshold"],
                            label="VAE Tile Threshold",
                            step=10,
                        )
                    with gr.Row():
                        components["vae_low_vram_mode"] = gr.Checkbox(
                            value=defaults.get("vae_low_vram_mode", False),
                            label="VAE Low VRAM Mode",
                        )
                    with gr.Row():
                        components["model_checkpoint"] = gr.Textbox(
                            interactive=True,
                            max_lines=1,
                            label="Model Checkpoint Path",
                            value=defaults["model_checkpoint"],
                        )

                with gr.Accordion("Enhance-A-Video", open=False):
                    components["enhance_enable"] = gr.Checkbox(
                        value=defaults["enhance_enable"],
                        label="Enable Enhance-A-Video",
                    )
                    with gr.Row():
                        components["enhance_weight"] = gr.Slider(
                            minimum=0.0,
                            maximum=10.0,
                            step=0.1,
                            value=float(defaults["enhance_weight"]),
                            label="Enhance Weight",
                            info="Strength of cross-frame boost.",
                        )
                        components["enhance_max_tokens"] = gr.Number(
                            value=int(defaults["enhance_max_tokens"]),
                            precision=0,
                            minimum=0,
                            label="Max Spatial Tokens per Frame",
                            info="Down-samples per-frame tokens when >0 to limit memory (0 = auto).",
                        )

                with gr.Row():
                    reset_config_btn = gr.Button(
                        "Reset to default",
                        variant="secondary",
                        size="sm",
                    )

                def reset_to_defaults(variant):
                    config_manager.reset_ui_config(variant)
                    config_defaults[variant] = load_config_defaults(
                        variant, config_manager
                    )
                    defaults = get_variant_defaults(config_defaults, variant)

                    if defaults["attention_type"] == "nabla":
                        attention_impl_value = "nabla"
                    elif defaults["attention_type"] == "sage":
                        attention_impl_value = "sage"
                    else:
                        attention_impl_value = "flash_fa2"

                    return [
                        gr.update(value=defaults["prompt"]),
                        gr.update(value=defaults["negative_prompt"]),
                        gr.update(value=defaults["width"]),
                        gr.update(value=defaults["height"]),
                        gr.update(value=defaults["duration"]),
                        gr.update(value=defaults["seed"]),
                        gr.update(value=defaults["num_steps"]),
                        gr.update(value=defaults["guidance_weight"]),
                        gr.update(value=defaults["expand_prompts"]),
                        gr.update(value=defaults["use_offload"]),
                        gr.update(value=defaults["use_magcache"]),
                        gr.update(value=defaults["use_dit_int8_ao_quantization"]),
                        gr.update(value=defaults["use_save_quantized_weights"]),
                        gr.update(
                            value=defaults["use_text_embedder_int8_ao_quantization"]
                        ),
                        gr.update(value=defaults["use_torch_compile_dit"]),
                        gr.update(value=defaults["use_torch_compile_vae"]),
                        gr.update(value=defaults["enhance_enable"]),
                        gr.update(value=float(defaults["enhance_weight"])),
                        gr.update(value=int(float(defaults["enhance_max_tokens"]))),
                        gr.update(value=attention_impl_value),
                        gr.update(value=defaults["in_visual_dim"]),
                        gr.update(value=defaults["out_visual_dim"]),
                        gr.update(value=defaults["time_dim"]),
                        gr.update(value=defaults["model_dim"]),
                        gr.update(value=defaults["ff_dim"]),
                        gr.update(value=defaults["num_text_blocks"]),
                        gr.update(value=defaults["num_visual_blocks"]),
                        gr.update(value=defaults["patch_size"]),
                        gr.update(value=defaults["axes_dims"]),
                        gr.update(value=defaults["visual_cond"]),
                        gr.update(value=defaults["in_text_dim"]),
                        gr.update(value=defaults["in_text_dim2"]),
                        gr.update(value=defaults["attention_causal"]),
                        gr.update(value=defaults["attention_local"]),
                        gr.update(value=defaults["attention_glob"]),
                        gr.update(value=defaults["attention_window"]),
                        gr.update(value=defaults["nabla_P"]),
                        gr.update(value=defaults["nabla_wT"]),
                        gr.update(value=defaults["nabla_wW"]),
                        gr.update(value=defaults["nabla_wH"]),
                        gr.update(value=defaults["nabla_add_sta"]),
                        gr.update(value=defaults["nabla_method"]),
                        gr.update(value=defaults["qwen_emb_size"]),
                        gr.update(value=defaults["qwen_max_length"]),
                        gr.update(value=defaults["qwen_checkpoint"]),
                        gr.update(value=defaults["clip_emb_size"]),
                        gr.update(value=defaults["clip_max_length"]),
                        gr.update(value=defaults["clip_checkpoint"]),
                        gr.update(value=defaults["vae_checkpoint"]),
                        gr.update(value=defaults["vae_name"]),
                        gr.update(value=defaults["vae_tile_threshold"]),
                        gr.update(value=defaults.get("vae_low_vram_mode", False)),
                        gr.update(value=defaults["model_checkpoint"]),
                    ]

                reset_config_btn.click(
                    fn=reset_to_defaults,
                    inputs=[config_variant],
                    outputs=[
                        prompt,
                        negative_prompt,
                        width,
                        height,
                        duration,
                        seed,
                        components["num_steps"],
                        components["guidance_weight"],
                        components["expand_prompts"],
                        components["use_offload"],
                        components["use_magcache"],
                        components["use_dit_int8_ao_quantization"],
                        components["use_save_quantized_weights"],
                        components["use_text_embedder_int8_ao_quantization"],
                        components["use_torch_compile_dit"],
                        components["use_torch_compile_vae"],
                        components["enhance_enable"],
                        components["enhance_weight"],
                        components["enhance_max_tokens"],
                        components["attention_implementation"],
                        components["in_visual_dim"],
                        components["out_visual_dim"],
                        components["time_dim"],
                        components["model_dim"],
                        components["ff_dim"],
                        components["num_text_blocks"],
                        components["num_visual_blocks"],
                        components["patch_size"],
                        components["axes_dims"],
                        components["visual_cond"],
                        components["in_text_dim"],
                        components["in_text_dim2"],
                        components["attention_causal"],
                        components["attention_local"],
                        components["attention_glob"],
                        components["attention_window"],
                        components["nabla_P"],
                        components["nabla_wT"],
                        components["nabla_wW"],
                        components["nabla_wH"],
                        components["nabla_add_sta"],
                        components["nabla_method"],
                        components["qwen_emb_size"],
                        components["qwen_max_length"],
                        components["qwen_checkpoint"],
                        components["clip_emb_size"],
                        components["clip_max_length"],
                        components["clip_checkpoint"],
                        components["vae_checkpoint"],
                        components["vae_name"],
                        components["vae_tile_threshold"],
                        components["vae_low_vram_mode"],
                        components["model_checkpoint"],
                    ],
                )

                def load_variant_params(variant):
                    config_defaults[variant] = load_config_defaults(
                        variant, config_manager
                    )
                    defaults = get_variant_defaults(config_defaults, variant)
                    supports_magcache = (
                        "sft" in variant.lower() or "nocfg" in variant.lower()
                    )

                    # Convert attention_type to attention_implementation radio value
                    if defaults["attention_type"] == "nabla":
                        attention_impl_value = "nabla"
                    elif defaults["attention_type"] == "sage":
                        attention_impl_value = "sage"
                    else:
                        # For flash/sdpa, default to flash_fa2 (was the old default with use_flash_attention=True)
                        attention_impl_value = "flash_fa2"

                    return [
                        gr.update(value=defaults["prompt"]),
                        gr.update(value=defaults["negative_prompt"]),
                        gr.update(value=defaults["width"]),
                        gr.update(value=defaults["height"]),
                        gr.update(value=defaults["duration"]),
                        gr.update(value=defaults["seed"]),
                        gr.update(value=defaults["num_steps"]),
                        gr.update(value=defaults["guidance_weight"]),
                        gr.update(value=defaults["expand_prompts"]),
                        gr.update(value=defaults["use_offload"]),
                        gr.update(
                            value=(
                                defaults["use_magcache"] if supports_magcache else False
                            ),
                            interactive=supports_magcache,
                        ),
                        gr.update(value=defaults["use_dit_int8_ao_quantization"]),
                        gr.update(value=defaults["use_save_quantized_weights"]),
                        gr.update(
                            value=defaults["use_text_embedder_int8_ao_quantization"]
                        ),
                        gr.update(value=defaults["use_torch_compile_dit"]),
                        gr.update(value=defaults["use_torch_compile_vae"]),
                        gr.update(value=defaults["enhance_enable"]),
                        gr.update(value=float(defaults["enhance_weight"])),
                        gr.update(value=int(float(defaults["enhance_max_tokens"]))),
                        gr.update(value=attention_impl_value),
                        gr.update(value=defaults["in_visual_dim"]),
                        gr.update(value=defaults["out_visual_dim"]),
                        gr.update(value=defaults["time_dim"]),
                        gr.update(value=defaults["model_dim"]),
                        gr.update(value=defaults["ff_dim"]),
                        gr.update(value=defaults["num_text_blocks"]),
                        gr.update(value=defaults["num_visual_blocks"]),
                        gr.update(value=defaults["patch_size"]),
                        gr.update(value=defaults["axes_dims"]),
                        gr.update(value=defaults["visual_cond"]),
                        gr.update(value=defaults["in_text_dim"]),
                        gr.update(value=defaults["in_text_dim2"]),
                        gr.update(value=defaults["attention_causal"]),
                        gr.update(value=defaults["attention_local"]),
                        gr.update(value=defaults["attention_glob"]),
                        gr.update(value=defaults["attention_window"]),
                        gr.update(value=defaults["nabla_P"]),
                        gr.update(value=defaults["nabla_wT"]),
                        gr.update(value=defaults["nabla_wW"]),
                        gr.update(value=defaults["nabla_wH"]),
                        gr.update(value=defaults["nabla_add_sta"]),
                        gr.update(value=defaults["nabla_method"]),
                        gr.update(value=defaults["qwen_emb_size"]),
                        gr.update(value=defaults["qwen_max_length"]),
                        gr.update(value=defaults["qwen_checkpoint"]),
                        gr.update(value=defaults["clip_emb_size"]),
                        gr.update(value=defaults["clip_max_length"]),
                        gr.update(value=defaults["clip_checkpoint"]),
                        gr.update(value=defaults["vae_checkpoint"]),
                        gr.update(value=defaults["vae_name"]),
                        gr.update(value=defaults["vae_tile_threshold"]),
                        gr.update(value=defaults.get("vae_low_vram_mode", False)),
                        gr.update(value=defaults["model_checkpoint"]),
                    ]

                config_variant.change(
                    fn=load_variant_params,
                    inputs=[config_variant],
                    outputs=[
                        prompt,
                        negative_prompt,
                        width,
                        height,
                        duration,
                        seed,
                        components["num_steps"],
                        components["guidance_weight"],
                        components["expand_prompts"],
                        components["use_offload"],
                        components["use_magcache"],
                        components["use_dit_int8_ao_quantization"],
                        components["use_save_quantized_weights"],
                        components["use_text_embedder_int8_ao_quantization"],
                        components["use_torch_compile_dit"],
                        components["use_torch_compile_vae"],
                        components["enhance_enable"],
                        components["enhance_weight"],
                        components["enhance_max_tokens"],
                        components["attention_implementation"],
                        components["in_visual_dim"],
                        components["out_visual_dim"],
                        components["time_dim"],
                        components["model_dim"],
                        components["ff_dim"],
                        components["num_text_blocks"],
                        components["num_visual_blocks"],
                        components["patch_size"],
                        components["axes_dims"],
                        components["visual_cond"],
                        components["in_text_dim"],
                        components["in_text_dim2"],
                        components["attention_causal"],
                        components["attention_local"],
                        components["attention_glob"],
                        components["attention_window"],
                        components["nabla_P"],
                        components["nabla_wT"],
                        components["nabla_wW"],
                        components["nabla_wH"],
                        components["nabla_add_sta"],
                        components["nabla_method"],
                        components["qwen_emb_size"],
                        components["qwen_max_length"],
                        components["qwen_checkpoint"],
                        components["clip_emb_size"],
                        components["clip_max_length"],
                        components["clip_checkpoint"],
                        components["vae_checkpoint"],
                        components["vae_name"],
                        components["vae_tile_threshold"],
                        components["vae_low_vram_mode"],
                        components["model_checkpoint"],
                    ],
                )

            t2v_kd5_params.elem_classes = ["block-params", "t2v_kd5_params"]

        with gr.Column(
            scale=1, elem_classes=["t2v-kd5-output-block", "clear-flex-grow"]
        ):
            augmentations["ui_before_generate"]()

            with gr.Row():
                generate_t2v_kd5 = gr.Button("Generate", variant="primary", scale=2)
                cancel_t2v_kd5 = gr.Button(
                    "Cancel", variant="stop", scale=1, visible=False
                )

            with gr.Column():
                t2v_kd5_output = gr.Video(
                    label="Video output",
                    elem_classes=["t2v-kd5-output-video"],
                    autoplay=True,
                    show_share_button=True,
                )

            augmentations["ui_after_generate"]()

            async def generate(
                session,
                text,
                negative_text,
                variant,
                width,
                height,
                duration,
                seed,
                num_steps,
                guidance_weight,
                expand_prompts,
                use_offload,
                use_magcache,
                use_dit_int8_ao_quantization,
                use_save_quantized_weights,
                use_text_embedder_int8_ao_quantization,
                use_torch_compile_dit,
                use_torch_compile_vae,
                enhance_enable,
                enhance_weight,
                enhance_max_tokens,
                attention_implementation,
                in_visual_dim,
                out_visual_dim,
                time_dim,
                model_dim,
                ff_dim,
                num_text_blocks,
                num_visual_blocks,
                patch_size,
                axes_dims,
                visual_cond,
                in_text_dim,
                in_text_dim2,
                attention_causal,
                attention_local,
                attention_glob,
                attention_window,
                nabla_P,
                nabla_wT,
                nabla_wW,
                nabla_wH,
                nabla_add_sta,
                nabla_method,
                qwen_emb_size,
                qwen_max_length,
                qwen_checkpoint,
                clip_emb_size,
                clip_max_length,
                clip_checkpoint,
                vae_checkpoint,
                vae_name,
                vae_tile_threshold,
                vae_low_vram_mode,
                model_checkpoint,
                *injections,
            ):
                text = generate_prompt_from_wildcard(text)

                if attention_implementation == "nabla":
                    attention_type = "nabla"
                    use_flash_attention = True  # Doesn't matter for nabla
                    os.environ["KD5_ATTENTION_MODE"] = "flash"
                elif attention_implementation == "flash_fa2":
                    attention_type = "flash"
                    use_flash_attention = True
                    os.environ["KD5_ATTENTION_MODE"] = "flash"
                elif attention_implementation == "sage":
                    attention_type = "sage"  # Save sage as the attention type
                    use_flash_attention = True  # Enable attention mode switching
                    os.environ["KD5_ATTENTION_MODE"] = "sage"
                    print(f"→ Set KD5_ATTENTION_MODE=sage")
                else:  # flash_sdpa
                    attention_type = "flash"
                    use_flash_attention = False
                    os.environ["KD5_ATTENTION_MODE"] = "flash"

                try:
                    enhance_weight = float(enhance_weight)
                except (TypeError, ValueError):
                    enhance_weight = 3.4
                try:
                    enhance_max_tokens = int(enhance_max_tokens)
                except (TypeError, ValueError):
                    enhance_max_tokens = 0
                if enhance_max_tokens < 0:
                    enhance_max_tokens = 0

                config_data = config_manager.build_config_from_ui_params(
                    variant=variant,
                    prompt=text,
                    negative_prompt=negative_text,
                    width=width,
                    height=height,
                    duration=duration,
                    seed=seed,
                    num_steps=num_steps,
                    guidance_weight=guidance_weight,
                    expand_prompts=expand_prompts,
                    use_offload=use_offload,
                    use_magcache=use_magcache,
                    use_dit_int8_ao_quantization=use_dit_int8_ao_quantization,
                    use_save_quantized_weights=use_save_quantized_weights,
                    use_text_embedder_int8_ao_quantization=use_text_embedder_int8_ao_quantization,
                    use_torch_compile_dit=use_torch_compile_dit,
                    use_torch_compile_vae=use_torch_compile_vae,
                    enhance_enable=enhance_enable,
                    enhance_weight=enhance_weight,
                    enhance_max_tokens=enhance_max_tokens,
                    in_visual_dim=in_visual_dim,
                    out_visual_dim=out_visual_dim,
                    time_dim=time_dim,
                    model_dim=model_dim,
                    ff_dim=ff_dim,
                    num_text_blocks=num_text_blocks,
                    num_visual_blocks=num_visual_blocks,
                    patch_size=patch_size,
                    axes_dims=axes_dims,
                    visual_cond=visual_cond,
                    in_text_dim=in_text_dim,
                    in_text_dim2=in_text_dim2,
                    attention_type=attention_type,
                    attention_causal=attention_causal,
                    attention_local=attention_local,
                    attention_glob=attention_glob,
                    attention_window=attention_window,
                    nabla_P=nabla_P,
                    nabla_wT=nabla_wT,
                    nabla_wW=nabla_wW,
                    nabla_wH=nabla_wH,
                    nabla_add_sta=nabla_add_sta,
                    nabla_method=nabla_method,
                    qwen_emb_size=qwen_emb_size,
                    qwen_max_length=qwen_max_length,
                    qwen_checkpoint=qwen_checkpoint,
                    clip_emb_size=clip_emb_size,
                    clip_max_length=clip_max_length,
                    clip_checkpoint=clip_checkpoint,
                    vae_checkpoint=vae_checkpoint,
                    vae_name=vae_name,
                    vae_tile_threshold=vae_tile_threshold,
                    vae_low_vram_mode=vae_low_vram_mode,
                    model_checkpoint=model_checkpoint,
                )
                config_manager.save_ui_config(variant, config_data)

                while True:
                    variant_config = {
                        "model": {
                            "num_steps": num_steps,
                            "guidance_weight": guidance_weight,
                            "checkpoint_path": model_checkpoint,
                            "dit_params": {
                                "in_visual_dim": in_visual_dim,
                                "out_visual_dim": out_visual_dim,
                                "time_dim": time_dim,
                                "model_dim": model_dim,
                                "ff_dim": ff_dim,
                                "num_text_blocks": num_text_blocks,
                                "num_visual_blocks": num_visual_blocks,
                                "patch_size": (
                                    eval(patch_size)
                                    if isinstance(patch_size, str)
                                    else patch_size
                                ),
                                "axes_dims": (
                                    eval(axes_dims)
                                    if isinstance(axes_dims, str)
                                    else axes_dims
                                ),
                                "visual_cond": visual_cond,
                                "in_text_dim": in_text_dim,
                                "in_text_dim2": in_text_dim2,
                            },
                            "attention": {
                                "type": attention_type,
                                "causal": attention_causal,
                                "local": attention_local,
                                "glob": attention_glob,
                                "window": attention_window,
                                "P": nabla_P,
                                "wT": nabla_wT,
                                "wW": nabla_wW,
                                "wH": nabla_wH,
                                "add_sta": nabla_add_sta,
                                "method": nabla_method,
                            },
                            "text_embedder": {
                                "qwen": {
                                    "emb_size": qwen_emb_size,
                                    "max_length": qwen_max_length,
                                    "checkpoint_path": qwen_checkpoint,
                                },
                                "clip": {
                                    "emb_size": clip_emb_size,
                                    "max_length": clip_max_length,
                                    "checkpoint_path": clip_checkpoint,
                                },
                            },
                            "vae": {
                                "checkpoint_path": vae_checkpoint,
                                "name": vae_name,
                                "tile_threshold": vae_tile_threshold,
                            },
                        },
                        "expand_prompts": expand_prompts,
                        "use_offload": use_offload,
                        "use_magcache": use_magcache,
                        "use_dit_int8_ao_quantization": use_dit_int8_ao_quantization,
                        "use_save_quantized_weights": use_save_quantized_weights,
                        "use_text_embedder_int8_ao_quantization": use_text_embedder_int8_ao_quantization,
                    }

                    params = {
                        ".session": session,
                        "config_variant": variant,
                        "pipeline_args": {
                            "config_name": variant,
                            "kd50_conf": {variant: variant_config},
                        },
                        "prompt": text,
                        "negative_prompt": negative_text,
                        "width": width,
                        "height": height,
                        "time_length": duration,
                        "seed": seed,
                        "num_steps": num_steps,
                        "guidance_weight": guidance_weight,
                        "expand_prompts": expand_prompts,
                        "magcache": use_magcache,
                        "enhance_enable": enhance_enable,
                        "enhance_weight": enhance_weight,
                        "enhance_max_tokens": enhance_max_tokens,
                    }

                    params = augmentations["exec"](params, injections)

                    try:
                        yield generate_fn(params)
                    except InterruptedError as e:
                        # Handle cancellation specifically
                        print(f"Generation cancelled: {e}")
                        yield "Generation cancelled by user"
                        break
                    except Exception as e:
                        import traceback

                        error_msg = f"Error in KD5 generation: {str(e)}\n\nFull traceback:\n{traceback.format_exc()}"
                        print(error_msg)
                        yield f"Generation failed: {error_msg}"

                    if not shared.check("LOOP_T2V_KD5", False):
                        break

            generate_event = click_and_disable(
                element=generate_t2v_kd5,
                fn=generate,
                inputs=[
                    session,
                    prompt,
                    negative_prompt,
                    config_variant,
                    width,
                    height,
                    duration,
                    seed,
                    components["num_steps"],
                    components["guidance_weight"],
                    components["expand_prompts"],
                    components["use_offload"],
                    components["use_magcache"],
                    components["use_dit_int8_ao_quantization"],
                    components["use_save_quantized_weights"],
                    components["use_text_embedder_int8_ao_quantization"],
                    components["use_torch_compile_dit"],
                    components["use_torch_compile_vae"],
                    components["enhance_enable"],
                    components["enhance_weight"],
                    components["enhance_max_tokens"],
                    components["attention_implementation"],
                    components["in_visual_dim"],
                    components["out_visual_dim"],
                    components["time_dim"],
                    components["model_dim"],
                    components["ff_dim"],
                    components["num_text_blocks"],
                    components["num_visual_blocks"],
                    components["patch_size"],
                    components["axes_dims"],
                    components["visual_cond"],
                    components["in_text_dim"],
                    components["in_text_dim2"],
                    components["attention_causal"],
                    components["attention_local"],
                    components["attention_glob"],
                    components["attention_window"],
                    components["nabla_P"],
                    components["nabla_wT"],
                    components["nabla_wW"],
                    components["nabla_wH"],
                    components["nabla_add_sta"],
                    components["nabla_method"],
                    components["qwen_emb_size"],
                    components["qwen_max_length"],
                    components["qwen_checkpoint"],
                    components["clip_emb_size"],
                    components["clip_max_length"],
                    components["clip_checkpoint"],
                    components["vae_checkpoint"],
                    components["vae_name"],
                    components["vae_tile_threshold"],
                    components["vae_low_vram_mode"],
                    components["model_checkpoint"],
                ]
                + augmentations["injections"],
                outputs=[t2v_kd5_output],
                js=(
                    "args => kubin.UI.taskStarted('Text To Video')",
                    "args => kubin.UI.taskFinished('Text To Video')",
                ),
            )

            # Cancel button placeholder for future implementation
            # cancel_t2v_kd5.click(...)

    return t2v_kd5_block
