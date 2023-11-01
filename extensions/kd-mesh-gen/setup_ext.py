from typing import Union
import uuid
import gradio as gr
import numpy as np
import torch
import os
from typing import Union
from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models import download
from shap_e.models.download import load_model, load_config
from shap_e.models.nn.camera import (
    DifferentiableCameraBatch,
    DifferentiableProjectiveCamera,
)
from shap_e.models.transmitter.base import Transmitter, VectorDecoder
from shap_e.rendering.torch_mesh import TorchMesh
from shap_e.util.collections import AttrDict

title = "Mesh Generator"


def setup(kubin):
    source_image = gr.Image(
        type="pil", label="Source image", elem_classes=["full-height"]
    )

    def model_3d_ui(ui_shared, ui_tabs):
        with gr.Row() as model_3d_block:
            with gr.Column(scale=1) as model_3d_params_block:
                with gr.Row():
                    source_image.render()

            with gr.Column(scale=1):
                create_btn = gr.Button(
                    "Generate model", label="Generate model", variant="primary"
                )
                model_output = gr.Model3D(
                    clear_color=[0.0, 0.0, 0.0, 0.0], label="Generated model"
                )

            kubin.ui_utils.click_and_disable(
                create_btn,
                fn=lambda *p: create_model(kubin, *p),
                inputs=[
                    source_image,
                    gr.State(kubin.params("general", "output_dir")),
                    gr.State(kubin.params("general", "device")),
                ],
                outputs=model_output,
                js=[
                    f"args => kubin.UI.taskStarted('{title}')",
                    f"args => kubin.UI.taskFinished('{title}')",
                ],
            )

            model_3d_params_block.elem_classes = ["block-params"]

        return model_3d_block

    return {
        "send_to": f"🗿 Send to {title}",
        "title": title,
        "tab_ui": lambda ui_s, ts: model_3d_ui(ui_s, ts),
        "send_target": source_image,
    }


def patch(kubin):
    old_method = download.default_cache_dir
    download.default_cache_dir = lambda: f"{kubin.params('general','cache_dir')}/shap-e"
    return old_method


def unpatch(old_method):
    download.default_cache_dir = old_method


def create_model(kubin, source_image, output_dir, device):
    p = patch(kubin)

    xm = load_model("transmitter", device=device)
    model = load_model("image300M", device=device)
    diffusion = diffusion_from_config(load_config("diffusion"))

    batch_size = 1
    guidance_scale = 3.0

    latents = sample_latents(
        batch_size=batch_size,
        model=model, 
        diffusion=diffusion,
        guidance_scale=guidance_scale,
        model_kwargs=dict(images=[source_image] * batch_size),
        progress=True,
        clip_denoised=True,
        use_fp16=True,
        use_karras=True,
        karras_steps=64,
        sigma_min=1e-3,
        sigma_max=160,
        s_churn=0,
    )

    objects = []
    save_path = f"{output_dir}/obj"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for _, latent in enumerate(latents):
        t = decode_latent_mesh(xm, latent).tri_mesh()
        uid = uuid.uuid4()

        obj_path = f"{save_path}/{uid}.obj"
        with open(obj_path, "w") as f:
            t.write_obj(f)
            objects.append(obj_path)

    unpatch(p)
    return objects[0]


@torch.no_grad()
def decode_latent_mesh(
    xm: Union[Transmitter, VectorDecoder],
    latent: torch.Tensor,
) -> TorchMesh:
    decoded = xm.renderer.render_views(
        AttrDict(cameras=create_pan_cameras(2, latent.device)),
        params=(xm.encoder if isinstance(xm, Transmitter) else xm).bottleneck_to_params(
            latent[None]
        ),
        options=AttrDict(rendering_mode="stf", render_with_direction=False),
    )
    return decoded.raw_meshes[0]


def create_pan_cameras(size: int, device: torch.device) -> DifferentiableCameraBatch:
    origins = []
    xs = []
    ys = []
    zs = []
    for theta in np.linspace(0, 2 * np.pi, num=20):
        z = np.array([np.sin(theta), np.cos(theta), -0.5])
        z /= np.sqrt(np.sum(z**2))
        origin = -z * 4
        x = np.array([np.cos(theta), -np.sin(theta), 0.0])
        y = np.cross(z, x)
        origins.append(origin)
        xs.append(x)
        ys.append(y)
        zs.append(z)

    return DifferentiableCameraBatch(
        shape=(1, len(xs)),
        flat_camera=DifferentiableProjectiveCamera(
            origin=torch.from_numpy(np.stack(origins, axis=0)).float().to(device),
            x=torch.from_numpy(np.stack(xs, axis=0)).float().to(device),
            y=torch.from_numpy(np.stack(ys, axis=0)).float().to(device),
            z=torch.from_numpy(np.stack(zs, axis=0)).float().to(device),
            width=size,
            height=size,
            x_fov=0.7,
            y_fov=0.7,
        ),
    )
