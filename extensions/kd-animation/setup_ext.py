import shutil
import subprocess
import tempfile
import gradio as gr
import os

title = "Animation"


def setup(kubin):
    def animation_ui(ui_shared, ui_tabs):
        with gr.Row() as animation_block:
            with gr.Tabs():
                with gr.Tab("Deforum"):
                    with gr.Row():
                        with gr.Column(scale=2) as deforum_params_block:
                            with gr.Row():
                                prompts = gr.TextArea(
                                    value=str.join(
                                        "\n",
                                        [
                                            "winter forest, snowflakes, Van Gogh style",
                                            "spring forest, flowers, sun rays, Van Gogh style",
                                            "summer forest, lake, reflections on the water, summer sun, Van Gogh style",
                                            "autumn forest, rain, Van Gogh style",
                                            "winter forest, snowflakes, Van Gogh style",
                                        ],
                                    ),
                                    label="Prompts",
                                    lines=6,
                                )
                            with gr.Row():
                                negative_prompts = gr.TextArea(
                                    value="", label="Negative prompts", lines=6
                                )
                            with gr.Row():
                                with gr.Column():
                                    animations = gr.TextArea(
                                        value=str.join(
                                            "\n",
                                            ["live", "right", "right", "right", "live"],
                                        ),
                                        label="Animations",
                                        lines=6,
                                    )
                                    gr.HTML(
                                        "Possible values: 'right', 'left', 'up', 'down', 'spin_clockwise', 'spin_counterclockwise', 'zoomin', 'zoomout', 'rotate_right', 'rotate_left', 'rotate_up', 'rotate_down','around_right', 'around_left', 'zoomin_sinus_x', 'zoomout_sinus_y', 'right_sinus_y', 'left_sinus_y', 'flipping_phi', 'live'",
                                        elem_classes=["deforum-animations-info"],
                                    )

                                durations = gr.TextArea(
                                    value=str.join("\n", ["1", "1", "1", "1", "1"]),
                                    label="Durations",
                                    lines=6,
                                )
                                with gr.Column():
                                    width = gr.Number(value="640", label="Width")
                                    height = gr.Number(value="640", label="Height")
                                    fps = gr.Number(value="24", label="FPS")

                        with gr.Column(scale=1) as deforum_output_block:
                            create_animation_btn = gr.Button(
                                "Create animation", variant="primary"
                            )
                            deforum_output = gr.Video()

                            kubin.ui_utils.click_and_disable(
                                create_animation_btn,
                                fn=lambda *params: create_deforum_animation(
                                    kubin, *params
                                ),
                                inputs=[
                                    prompts,
                                    negative_prompts,
                                    animations,
                                    durations,
                                    width,
                                    height,
                                    fps,
                                ],
                                outputs=deforum_output,
                            )

            deforum_params_block.elem_classes = ["block-params"]
        return animation_block

    return {"title": title, "tab_ui": lambda ui_s, ts: animation_ui(ui_s, ts)}


def create_deforum_animation(
    kubin,
    prompts_string,
    negative_prompts_string,
    animations_string,
    durations_string,
    width,
    height,
    fps,
):
    from deforum.deforum import create_deforum, create_animation

    prompts = prompts_string.split("\n")
    negative_prompts = (
        ["low quality, bad image, cropped, out of frame"] * len(prompts)
        if negative_prompts_string == ""
        else negative_prompts_string.split("\n")
    )
    animations = animations_string.split("\n")
    durations = durations_string.split("\n")

    prior, decoder = kubin.model.prepare_model("img2img")
    device = kubin.params("general", "device")

    deforum = create_deforum(prior, decoder, device)
    animation = create_animation(
        deforum=deforum,
        prompts=prompts.split("\n"),
        negative_prompts=negative_prompts.split("\n"),
        animations=animations.split("\n"),
        durations=durations.split("\n"),
        W=width,
        H=height,
        fps=fps,
        save_samples=False,
    )
    return animation


def mount(kubin):
    deforum_repo = "https://github.com/ai-forever/deforum-kandinsky"
    commit = "7bb8d8c"
    destination_dir = "extensions/kd-animation/deforum_kandinsky"

    if not os.path.exists(destination_dir):
        current_path = os.getcwd()
        temp_dir = tempfile.mkdtemp()
        temp_deforum = os.path.join(temp_dir, "deforum")

        subprocess.run(["git", "clone", deforum_repo, temp_deforum, "-q"])
        os.chdir(temp_deforum)
        subprocess.run(["git", "checkout", commit, "-q"])
        os.chdir(current_path)

        repo_path = os.path.join(temp_deforum, "deforum_kandinsky")
        if os.path.exists(repo_path) and os.path.isdir(repo_path):
            shutil.copytree(repo_path, destination_dir)
        try:
            shutil.rmtree(temp_dir)
        except:
            pass
