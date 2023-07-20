import gradio as gr


def samplers_controls():
    sampler_20 = gr.Radio(
        ["ddim_sampler", "p_sampler"],
        value="p_sampler",
        label="Sampler",
        interactive=True,
    )
    sampler_20.elem_classes = [
        "t2i_sampler",
        "unsupported_21",
        "unsupported_d21",
        "unsupported_22",
        "unsupported_d22",
    ]

    sampler_21_native = gr.Radio(
        ["ddim_sampler", "p_sampler", "plms_sampler"],
        value="p_sampler",
        label="Sampler",
        interactive=True,
    )
    sampler_21_native.elem_classes = [
        "t2i_sampler",
        "unsupported_20",
        "unsupported_d21",
        "unsupported_22",
    ]

    sampler_diffusers = gr.Radio(
        ["ddim_sampler", "ddpm_sampler"],
        value="ddpm_sampler",
        label="Sampler",
        interactive=True,
    )
    sampler_diffusers.elem_classes = [
        "t2i_sampler",
        "unsupported_20",
        "unsupported_21",
    ]

    return sampler_20, sampler_21_native, sampler_diffusers
