import gradio as gr

from ui_blocks.shared.compatibility import (
    sampler20_classes,
    sampler21_classes,
    sampler_diffusers_classes,
)


def samplers_controls(default_samplers=["p_sampler", "p_sampler", "DDPM"]):
    default_sampler_20, default_sampler_21_native, default_sampler_diffusers = (
        default_samplers
    )

    samplers_20 = ["ddim_sampler", "p_sampler"]
    sampler_20 = gr.Radio(
        choices=samplers_20,
        value=default_sampler_20,
        label="Sampler",
        interactive=True,
    )
    sampler_20.elem_classes = sampler20_classes() + ["t2i_sampler"]

    samplers_21 = ["ddim_sampler", "p_sampler", "plms_sampler"]
    sampler_21_native = gr.Radio(
        choices=samplers_21,
        value=default_sampler_21_native,
        label="Sampler",
        interactive=True,
    )
    sampler_21_native.elem_classes = sampler21_classes() + ["t2i_sampler"]

    samplers_diffusers = [
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
    ]

    sampler_diffusers = gr.Dropdown(
        choices=samplers_diffusers,
        value=default_sampler_diffusers,
        label="Sampler",
        interactive=True,
    )
    sampler_diffusers.elem_classes = sampler_diffusers_classes() + ["t2i_sampler"]
    return sampler_20, sampler_21_native, sampler_diffusers
