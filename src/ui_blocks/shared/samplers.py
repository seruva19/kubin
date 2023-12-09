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
        "unsupported_30",
        "unsupported_d30",
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
        "unsupported_30",
        "unsupported_d30",
    ]

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
    sampler_diffusers.elem_classes = [
        "t2i_sampler",
        "unsupported_20",
        "unsupported_21",
        "unsupported_30",
        "unsupported_d30",
    ]

    return sampler_20, sampler_21_native, sampler_diffusers
