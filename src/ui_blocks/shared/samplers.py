import gradio as gr

from ui_blocks.shared.compatibility import (
    sampler20_classes,
    sampler21_classes,
    sampler_diffusers_classes,
)


def samplers_controls():
    sampler_20 = gr.Radio(
        ["ddim_sampler", "p_sampler"],
        value="p_sampler",
        label="Sampler",
        interactive=True,
    )
    sampler_20.elem_classes = sampler20_classes() + ["t2i_sampler"]

    sampler_21_native = gr.Radio(
        ["ddim_sampler", "p_sampler", "plms_sampler"],
        value="p_sampler",
        label="Sampler",
        interactive=True,
    )
    sampler_21_native.elem_classes = sampler21_classes() + ["t2i_sampler"]

    sampler_diffusers = gr.Dropdown(
        [
            "DDPM",
            "DDIM",
            "DPMSM",
            "DEISM",
            "DPMSSDE",
            "DPMSS",
            "Euler",
            "EulerA",
            "Heun",
            "LMS",
            "KDPM2",
            "KDPM2A",
            "PNDM",
            "UniPC",
        ],
        value="DDPM",
        label="Sampler",
        interactive=True,
    )
    sampler_diffusers.elem_classes = sampler_diffusers_classes() + ["t2i_sampler"]
    return sampler_20, sampler_21_native, sampler_diffusers
