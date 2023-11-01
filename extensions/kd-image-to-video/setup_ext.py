import shutil
import subprocess
import tempfile
import gradio as gr
import os
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf
import torch

title = "Image To Video"


def setup(kubin):
    source_image = gr.Image(type="numpy", label="Source image")

    cache_dir = kubin.params("general", "cache_dir")
    path = f"{cache_dir}/i2v_512_v1"

    def download_model():
        if not os.path.exists(path):
            os.makedirs(path)

        for filename in ["model.ckpt"]:
            local_file = os.path.join(path, filename)
            if not os.path.exists(local_file):
                hf_hub_download(
                    repo_id="VideoCrafter/Image2Video-512-v1.0",
                    filename=filename,
                    cache_dir=cache_dir,
                    local_dir=path,
                    local_dir_use_symlinks=False,
                )

    def create_video(source_image, prompt, steps, cfg_scale, eta, fps):
        from scripts.evaluation.funcs import (
            load_model_checkpoint,
            load_image_batch,
            save_videos,
            batch_ddim_sampling,
        )
        from utils.utils import instantiate_from_config

        download_model()
        torch.cuda.empty_cache()

        result_dir = kubin.params("general", "output_dir")
        save_path = f"{result_dir}/video"
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        ckpt_path = f"{path}/model.ckpt"
        config_file = "extensions/kd-image-to-video/configs/inference_i2v_512_v1.0.yaml"
        config = OmegaConf.load(config_file)
        model_config = config.pop("model", OmegaConf.create())
        model_config["params"]["unet_config"]["params"]["use_checkpoint"] = False

        model = instantiate_from_config(model_config)
        model = model.cuda(0)
        model = load_model_checkpoint(model, ckpt_path)
        model.eval()

        save_fps = 8
        batch_size = 1
        channels = model.model.diffusion_model.in_channels
        frames = model.temporal_length
        h, w = 320 // 8, 512 // 8
        noise_shape = [batch_size, channels, frames, h, w]

        # prompts = batch_size * [""]
        text_emb = model.get_learned_conditioning([prompt])

        # cond_images = load_image_batch([image_path])
        img_tensor = torch.from_numpy(source_image).permute(2, 0, 1).float()
        img_tensor = (img_tensor / 255.0 - 0.5) * 2
        img_tensor = img_tensor.unsqueeze(0)
        cond_images = img_tensor.to(model.device)
        img_emb = model.get_image_embeds(cond_images)
        imtext_cond = torch.cat([text_emb, img_emb], dim=1)
        cond = {"c_crossattn": [imtext_cond], "fps": fps}

        ## inference
        batch_samples = batch_ddim_sampling(
            model,
            cond,
            noise_shape,
            n_samples=1,
            ddim_steps=steps,
            ddim_eta=eta,
            cfg_scale=cfg_scale,
        )
        ## b,samples,c,t,h,w
        prompt_str = prompt.replace("/", "_slash_") if "/" in prompt else prompt
        prompt_str = prompt_str.replace(" ", "_") if " " in prompt else prompt_str
        prompt_str = prompt_str[:30]

        save_videos(batch_samples, result_dir, filenames=[prompt_str], fps=save_fps)
        return os.path.join(result_dir, f"{prompt_str}.mp4")

    def i2v_ui(ui_shared, ui_tabs):
        with gr.Row() as i2v_block:
            with gr.Column(scale=1) as i2v_params_block:
                with gr.Row():
                    source_image.render()
                with gr.Row():
                    prompt = gr.Textbox("", label="Prompt")
                with gr.Row():
                    steps = gr.Slider(0, 60, value=60, label="Steps")
                    cfg_scale = gr.Slider(0, 15, value=12, label="CFG scale")
                    eta = gr.Slider(0, 1, value=1, label="ETA")
                    fps = gr.Slider(0, 24, value=16, label="FPS")

            with gr.Column(scale=1):
                create_btn = gr.Button(
                    "Generate video", label="Generate video", variant="primary"
                )
                video_output = gr.Image(label="Generated video")

            kubin.ui_utils.click_and_disable(
                create_btn,
                fn=create_video,
                inputs=[source_image, prompt, steps, cfg_scale, eta, fps],
                outputs=video_output,
                js=[
                    f"args => kubin.UI.taskStarted('{title}')",
                    f"args => kubin.UI.taskFinished('{title}')",
                ],
            )

            i2v_params_block.elem_classes = ["block-params"]
        return i2v_block

    return {
        "send_to": f"📇 Send to Img2Video",
        "title": title,
        "tab_ui": lambda ui_s, ts: i2v_ui(ui_s, ts),
        "send_target": source_image,
    }


def mount(kubin):
    video_crafter_repo = "https://github.com/AILab-CVC/VideoCrafter"
    commit = "33f113b"
    required_folders = ["configs", "lvdm", "scripts", "utils"]
    destination_dirs = [
        "extensions/kd-image-to-video/" + path for path in required_folders
    ]

    all_exist = True
    for path in destination_dirs:
        if not os.path.exists(path):
            all_exist = False

    if not all_exist:
        current_path = os.getcwd()
        temp_dir = tempfile.mkdtemp()
        temp_video_crafter = os.path.join(temp_dir, "video_crafter")

        subprocess.run(["git", "clone", video_crafter_repo, temp_video_crafter, "-q"])
        os.chdir(temp_video_crafter)
        subprocess.run(["git", "checkout", commit, "-q"])
        os.chdir(current_path)

        for path in ["configs", "lvdm", "scripts", "utils"]:
            shutil.copytree(
                os.path.join(temp_video_crafter, path),
                os.path.join("extensions/kd-image-to-video/", path),
            )

        try:
            shutil.rmtree(temp_dir)
        except:
            pass
