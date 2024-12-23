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

block = "v2a"


def v2a_ui(generate_fn, shared: SharedUI, tabs, session):
    augmentations = shared.create_ext_augment_blocks("v2a")
    value = lambda name, def_value: get_value(shared.storage, block, name, def_value)

    with gr.Row() as v2a_block:
        v2a_block.elem_classes = ["v2a_block"]

        with gr.Column(scale=2) as v2a_params:
            with gr.Accordion("PRESETS", open=False, visible=False):
                pass

            augmentations["ui_before_prompt"]()

            with gr.Row():
                shared.input_video_to_audio.render()

                # input_image = gr.Image(
                #     autoplay=False,
                #     source="upload",
                #     label="Input video",
                # )
                with gr.Column():
                    prompt = gr.TextArea(
                        value=lambda: value(
                            "prompt",
                            "clean, clear, good quality",
                        ),
                        label="Prompt",
                        placeholder="",
                        lines=4,
                    )

                    negative_prompt = gr.TextArea(
                        value=lambda: value(
                            "negative_prompt",
                            "hissing noise, drumming rhythm, saying, poor quality",
                        ),
                        label="Negative prompt",
                        placeholder="",
                        lines=4,
                    )
                    seed = gr.Number(
                        value=lambda: value("seed", -1), label="Seed", precision=0
                    )

                    steps = gr.Number(
                        minimum=1,
                        maximum=100,
                        value=lambda: value("steps", 50),
                        label="Steps",
                        precision=0,
                    )

                    guidance_scale = gr.Number(
                        minimum=1,
                        maximum=30,
                        value=lambda: value("guidance_scale", 8),
                        label="Guidance scale",
                        precision=0,
                    )

                    generate_only_sound = gr.Checkbox(
                        value=lambda: value("generate_only_sound", False),
                        label="Generate only sound",
                    )

            augmentations["ui_before_params"]()

            v2a_params.elem_classes = ["block-params", "v2a_params"]

        with gr.Column(scale=1, elem_classes=["v2a-output-block", "clear-flex-grow"]):
            augmentations["ui_before_generate"]()

            generate_v2a = gr.Button("Generate", variant="primary")

            with gr.Column(
                visible=not value("generate_only_sound", "false")
            ) as v2a_video_block:
                v2a_output = gr.Video(
                    label="Video output",
                    elem_classes=["v2a-output-video"],
                    autoplay=False,
                    show_share_button=True,
                )

            with gr.Column(
                visible=value("generate_only_sound", "false")
            ) as v2a_sound_block:
                with gr.Column():
                    v2a_sound_output = gr.Audio(
                        autoplay=False,
                        label="Sound output",
                        elem_classes=["v2a-output"],
                    )

            augmentations["ui_after_generate"]()

            generate_only_sound.change(
                lambda x: [
                    gr.update(visible=not x),
                    gr.update(visible=x),
                ],
                inputs=[generate_only_sound],
                outputs=[v2a_video_block, v2a_sound_block],
            )

            async def generate(
                session,
                input_video,
                prompt,
                negative_prompt,
                seed,
                steps,
                guidance_scale,
                generate_only_sound,
                *injections
            ):
                prompt = generate_prompt_from_wildcard(prompt)

                while True:
                    params = {
                        ".session": session,
                        "input_video": input_video,
                        "prompt": prompt,
                        "negative_prompt": negative_prompt,
                        "seed": seed,
                        "steps": steps,
                        "guidance_scale": guidance_scale,
                        "generate_only_sound": generate_only_sound,
                    }

                    shared.storage.save(block, params)

                    params = augmentations["exec"](params, injections)
                    yield generate_fn(params)

                    if not shared.check("LOOP_V2A", False):
                        break

            click_and_disable(
                element=generate_v2a,
                fn=generate,
                inputs=[
                    session,
                    shared.input_video_to_audio,
                    prompt,
                    negative_prompt,
                    seed,
                    steps,
                    guidance_scale,
                    generate_only_sound,
                ]
                + augmentations["injections"],
                outputs=[v2a_output, v2a_sound_output],
                js=[
                    "args => kubin.UI.taskStarted('Video To Audio')",
                    "args => kubin.UI.taskFinished('Video To Audio')",
                ],
            )

    return v2a_block
