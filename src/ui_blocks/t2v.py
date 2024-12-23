import asyncio
import gradio as gr
from ui_blocks.shared.compatibility import (
    batch_size_classes,
    negative_prompt_classes,
    prior_block_classes,
)
from ui_blocks.shared.samplers import samplers_controls
from ui_blocks.shared.ui_shared import SharedUI
from utils.gradio_ui import click_and_disable
from utils.storage import get_value
from utils.text import generate_prompt_from_wildcard

block = "t2v"


def t2v_ui(generate_fn, shared: SharedUI, tabs, session):
    augmentations = shared.create_ext_augment_blocks("t2v")
    value = lambda name, def_value: get_value(shared.storage, block, name, def_value)

    with gr.Row() as t2v_block:
        t2v_block.elem_classes = ["t2v_block"]

        with gr.Column(scale=2) as t2v_params:
            with gr.Accordion("PRESETS", open=False, visible=False):
                pass

            augmentations["ui_before_prompt"]()

            with gr.Row():
                prompt = gr.TextArea(
                    value=lambda: value(
                        "prompt",
                        "A closeshot of beautiful blonde woman standing under the sun at the beach. Soft waves lapping at her feet and vibrant palm trees lining the distant coastline under a clear blue sky.",
                    ),
                    label="Prompt",
                    placeholder="",
                    lines=4,
                )
            augmentations["ui_before_params"]()

            with gr.Row():
                with gr.Column():
                    render_resolution = gr.Dropdown(
                        [
                            "1:1 [512x512]",
                            "1:2 [352x736]",
                            "2:1 [736x352]",
                            "9:16 [384x672]",
                            "16:9 [672x384]",
                            "3:4 [480x544]",
                            "4:3 [544x480]",
                        ],
                        value=lambda: value("resolution", "1:1 [512x512]"),
                        label="Resolution",
                    )
                    use_flash = gr.Checkbox(
                        value=lambda: value("pipeline_args.use_flash", True),
                        label="Use flash pipeline",
                        interactive=False,
                        elem_classes=["cnet-enable"],
                    )
                with gr.Column():
                    time_length = gr.Slider(
                        minimum=1,
                        maximum=20,
                        value=lambda: value("time_length", 12),
                        step=1,
                        label="Length",
                    )

                    generate_image = gr.Checkbox(
                        value=lambda: value("generate_image", False),
                        label="Generate image instead of video",
                    )

                seed = gr.Number(
                    value=lambda: value("seed", -1), label="Seed", precision=0
                )

            with gr.Column():
                with gr.Accordion("INITIALIZATION PARAMETERS", open=True):
                    with gr.Row():
                        optimization_flags = [
                            value.strip()
                            for value in shared.native_params(
                                "optimization_flags"
                            ).split(";")
                        ]

                        if "kd40_flash_attention" in optimization_flags:
                            mh_attention_type = "flash"
                        elif "kd40_sage_attention" in optimization_flags:
                            mh_attention_type = "sage"
                        else:
                            mh_attention_type = "none"

                    with gr.Accordion("DiT Parameters", open=True):
                        with gr.Row():
                            k_attention_type = gr.Dropdown(
                                interactive=True,
                                label="Attention Type",
                                value=lambda: value(
                                    "pipeline_args.kd40_conf.dit.params.k_attention_type",
                                    mh_attention_type,
                                ),
                                choices=["none", "flash", "sage"],
                            )
                            in_visual_dim = gr.Number(
                                interactive=True,
                                label="Input Visual Dimension",
                                precision=0,
                                value=lambda: value(
                                    "pipeline_args.kd40_conf.dit.params.in_visual_dim",
                                    16,
                                ),
                            )
                            in_text_dim = gr.Number(
                                interactive=True,
                                precision=0,
                                label="Input Text Dimension",
                                value=lambda: value(
                                    "pipeline_args.kd40_conf.dit.params.in_text_dim",
                                    4096,
                                ),
                            )
                            out_visual_dim = gr.Number(
                                interactive=True,
                                precision=0,
                                label="Output Visual Dimension",
                                value=lambda: value(
                                    "pipeline_args.kd40_conf.dit.params.out_visual_dim",
                                    16,
                                ),
                            )
                        with gr.Row():
                            time_dim = gr.Number(
                                interactive=True,
                                label="Time Dimension",
                                value=lambda: value(
                                    "pipeline_args.kd40_conf.dit.params.time_dim",
                                    512,
                                ),
                                precision=0,
                            )
                            patch_size = gr.Textbox(
                                interactive=True,
                                max_lines=1,
                                label="Patch Size",
                                value=lambda: value(
                                    "pipeline_args.kd40_conf._dit.patch_size",
                                    "[1, 2, 2]",
                                ),
                            )
                            model_dim = gr.Number(
                                precision=0,
                                interactive=True,
                                label="Model Dimension",
                                value=lambda: value(
                                    "pipeline_args.kd40_conf.dit.params.model_dim",
                                    3072,
                                ),
                            )
                            ff_dim = gr.Number(
                                interactive=True,
                                precision=0,
                                label="Feed Forward Dimension",
                                value=lambda: value(
                                    "pipeline_args.kd40_conf.dit.params.ff_dim",
                                    12288,
                                ),
                            )
                            num_blocks = gr.Number(
                                precision=0,
                                interactive=True,
                                label="Number of Blocks",
                                value=lambda: value(
                                    "pipeline_args.kd40_conf.dit.params.num_blocks",
                                    21,
                                ),
                            )
                            axes_dims = gr.Textbox(
                                interactive=True,
                                max_lines=1,
                                label="Axes Dimensions",
                                value=lambda: value(
                                    "pipeline_args.kd40_conf._dit.axes_dims",
                                    "[16, 24, 24]",
                                ),
                            )

                    with gr.Accordion("Configuration", open=True):
                        with gr.Row():
                            vae_checkpoint = gr.Textbox(
                                interactive=True,
                                max_lines=1,
                                label="VAE Checkpoint Path",
                                value=lambda: value(
                                    "pipeline_args.kd40_conf.vae.checkpoint_path",
                                    "",
                                ),
                            )
                            tokenizer_path = gr.Textbox(
                                interactive=True,
                                max_lines=1,
                                label="Tokenizer Path",
                                value=lambda: value(
                                    "pipeline_args.kd40_conf.text_embedder.params.tokenizer_path",
                                    "",
                                ),
                            )
                        with gr.Row():
                            text_emb_size = gr.Number(
                                precision=0,
                                interactive=True,
                                label="Text Embedder Embedding Size",
                                value=lambda: value(
                                    "pipeline_args.kd40_conf.text_embedder.emb_size",
                                    4096,
                                ),
                            )
                            text_tokens_length = gr.Number(
                                precision=0,
                                interactive=True,
                                label="Text Tokens Length",
                                value=lambda: value(
                                    "pipeline_args.kd40_conf.text_embedder.tokens_length",
                                    224,
                                ),
                            )
                            text_encoder_checkpoint = gr.Textbox(
                                interactive=True,
                                max_lines=1,
                                label="Text Encoder Checkpoint Path",
                                value=lambda: value(
                                    "pipeline_args.kd40_conf.text_embedder.checkpoint_path",
                                    "",
                                ),
                            )
                        with gr.Row():
                            dit_scheduler = gr.Textbox(
                                interactive=True,
                                max_lines=1,
                                label="DiT Scheduler Path",
                                value=lambda: value(
                                    "pipeline_args.kd40_conf.dit.scheduler",
                                    "",
                                ),
                            )
                            dit_checkpoint = gr.Textbox(
                                interactive=True,
                                max_lines=1,
                                label="DiT Checkpoint Path",
                                value=lambda: value(
                                    "pipeline_args.kd40_conf.dit.dit_checkpoint",
                                    "",
                                    # "models/kandinsky-4/kandinsky4_distil_512-bf16.safetensors",
                                ),
                            )
                            resolution = gr.Number(
                                label="Resolution",
                                value=lambda: value(
                                    "pipeline_args.kd40_conf.resolution", 512
                                ),
                                precision=0,
                            )
                    with gr.Column():
                        gr.HTML(
                            "<p>Changes to these parameters will only take effect after a model reload.</p>"
                        )

                        reload_model = gr.Button(
                            "Force model reload",
                            variant="secondary",
                            elem_classes=["options-medium"],
                        )
                        reload_model.click(
                            lambda: shared._kubin.model.flush(), queue=False
                        ).then(
                            fn=None,
                            _js='_ => kubin.notify.success("Model reload forced")',
                        )

            t2v_params.elem_classes = ["block-params", "t2v_params"]

        with gr.Column(scale=1, elem_classes=["t2v-output-block", "clear-flex-grow"]):
            augmentations["ui_before_generate"]()

            generate_t2v = gr.Button("Generate", variant="primary")

            with gr.Column(
                visible=not value("generate_image", "false")
            ) as t2v_video_block:
                with gr.Column():
                    t2v_output = gr.Video(
                        label="Video output",
                        elem_classes=["t2v-output-video"],
                        autoplay=True,
                        show_share_button=True,
                    )
                    send_v2a_btn = gr.Button(
                        "🔉 Send to Video2Audio", variant="secondary", size="sm"
                    )

                    send_v2a_btn.click(
                        fn=shared.open_another_tab,
                        inputs=[gr.State(7)],
                        outputs=tabs,
                        queue=False,
                    ).then(
                        fn=shared.send_video_to_another_tab,
                        inputs=[t2v_output],
                        outputs=[shared.input_video_to_audio],
                    )

            with gr.Column(visible=value("generate_image", "false")) as t2v_image_block:
                with gr.Column():
                    t2v_image_output = gr.Gallery(
                        label="Image output",
                        columns=2,
                        preview=True,
                        elem_classes=["t2v-output"],
                    )

                    t2v_image_output.select(
                        fn=None,
                        _js=f"() => kubin.UI.setImageIndex('t2v-output')",
                        show_progress=False,
                        outputs=gr.State(None),
                    )

                with gr.Column():
                    shared.create_base_send_targets(
                        t2v_image_output, "t2v-output", tabs
                    )
                    shared.create_ext_send_targets(t2v_image_output, "t2v-output", tabs)

            augmentations["ui_after_generate"]()

            generate_image.change(
                lambda x: [
                    gr.update(visible=not x),
                    gr.update(visible=x),
                    gr.update(interactive=not x),
                ],
                inputs=[generate_image],
                outputs=[t2v_video_block, t2v_image_block, time_length],
            )

            async def generate(
                session,
                text,
                render_resolution,
                length,
                generate_image_instead_video,
                seed,
                use_flash,
                k_attention_type,
                in_visual_dim,
                in_text_dim,
                out_visual_dim,
                time_dim,
                patch_size,
                model_dim,
                ff_dim,
                num_blocks,
                axes_dims,
                vae_checkpoint,
                text_emb_size,
                text_tokens_length,
                text_encoder_checkpoint,
                dit_scheduler,
                dit_checkpoint,
                tokenizer_path,
                resolution,
                *injections,
                # progress=gr.Progress(),
            ):
                text = generate_prompt_from_wildcard(text)

                while True:
                    params = {
                        ".session": session,
                        "pipeline_args": {
                            "use_flash": use_flash,
                            "kd40_conf": {
                                "vae": {"checkpoint_path": vae_checkpoint},
                                "text_embedder": {
                                    "emb_size": text_emb_size,
                                    "tokens_length": text_tokens_length,
                                    "params": {
                                        "checkpoint_path": text_encoder_checkpoint,
                                        "tokenizer_path": tokenizer_path,
                                    },
                                },
                                "dit": {
                                    "scheduler": dit_scheduler,
                                    "checkpoint_path": dit_checkpoint,
                                    "params": {
                                        "k_attention_type": k_attention_type,
                                        "in_visual_dim": in_visual_dim,
                                        "in_text_dim": in_text_dim,
                                        "out_visual_dim": out_visual_dim,
                                        "time_dim": time_dim,
                                        "patch_size": eval(patch_size),
                                        "model_dim": model_dim,
                                        "ff_dim": ff_dim,
                                        "num_blocks": num_blocks,
                                        "axes_dims": eval(axes_dims),
                                    },
                                },
                                "_dit": {
                                    "patch_size": patch_size,
                                    "axes_dims": axes_dims,
                                },
                                "resolution": resolution,
                            },
                        },
                        "prompt": text,
                        "time_length": length,
                        "resolution": render_resolution,
                        "generate_image": generate_image_instead_video,
                        "seed": seed,
                    }

                    shared.storage.save(block, params)

                    params = augmentations["exec"](params, injections)
                    yield generate_fn(params)

                    if not shared.check("LOOP_T2V", False):
                        break

            click_and_disable(
                element=generate_t2v,
                fn=generate,
                inputs=[
                    session,
                    prompt,
                    render_resolution,
                    time_length,
                    generate_image,
                    seed,
                    use_flash,
                    k_attention_type,
                    in_visual_dim,
                    in_text_dim,
                    out_visual_dim,
                    time_dim,
                    patch_size,
                    model_dim,
                    ff_dim,
                    num_blocks,
                    axes_dims,
                    vae_checkpoint,
                    text_emb_size,
                    text_tokens_length,
                    text_encoder_checkpoint,
                    dit_scheduler,
                    dit_checkpoint,
                    tokenizer_path,
                    resolution,
                ]
                + augmentations["injections"],
                outputs=[t2v_output, t2v_image_output],
                js=[
                    "args => kubin.UI.taskStarted('Text To Video')",
                    "args => kubin.UI.taskFinished('Text To Video')",
                ],
            )

    return t2v_block
