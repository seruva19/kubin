from dataclasses import dataclass
import gradio as gr
from env import Kubin


def options_tab_diffusers(kubin: Kubin):
    on_change = "(key, value, requiresRestart) => kubin.utils.optionsChanged(key, value, requiresRestart)"

    with gr.Column(
        elem_classes=["options-block", "options-diffusers"]
    ) as diffusers_options:
        half_precision_weights = gr.Checkbox(
            value=kubin.params("diffusers", "half_precision_weights"),
            label="Enable half precision weights",
        )
        enable_xformers = gr.Checkbox(
            value=kubin.params("diffusers", "enable_xformers"),
            label="Enable xformers memory efficient attention",
        )
        enable_sliced_attention = gr.Checkbox(
            value=kubin.params("diffusers", "enable_sliced_attention"),
            label="Enable attention slicing",
        )
        sequential_cpu_offload = gr.Checkbox(
            value=kubin.params("diffusers", "sequential_cpu_offload"),
            label="Enable sequential CPU offload",
        )
        full_model_offload = gr.Checkbox(
            value=kubin.params("diffusers", "full_model_offload"),
            label="Enable full-model offloading",
        )
        channels_last_memory = gr.Checkbox(
            value=kubin.params("diffusers", "channels_last_memory"),
            label="Enable channels last memory format",
        )
        torch_code_compilation = gr.Checkbox(
            value=kubin.params("diffusers", "torch_code_compilation"),
            label="Enable torch code compilation",
        )
        use_deterministic_algorithms = gr.Checkbox(
            value=kubin.params("diffusers", "use_deterministic_algorithms"),
            label="Enable torch deterministic algorithms",
        )
        enable_sdp_attention = gr.Checkbox(
            value=kubin.params("diffusers", "enable_sdp_attention"),
            label="Enable forced SDP attention",
        )
        use_tf32_mode = gr.Checkbox(
            value=kubin.params("diffusers", "use_tf32_mode"),
            label="Enable TensorFloat32 mode",
        )
        run_prior_on_cpu = gr.Checkbox(
            value=kubin.params("diffusers", "run_prior_on_cpu"),
            label="Enable prior generation on CPU",
        )

        half_precision_weights.change(
            fn=None,
            _js=on_change,
            inputs=[
                gr.Text("diffusers.half_precision_weights", visible=False),
                half_precision_weights,
                gr.Checkbox(False, visible=False),
            ],
            show_progress=False,
        )
        enable_xformers.change(
            fn=None,
            _js=on_change,
            inputs=[
                gr.Text("diffusers.enable_xformers", visible=False),
                enable_xformers,
                gr.Checkbox(False, visible=False),
            ],
            show_progress=False,
        )
        enable_sdp_attention.change(
            fn=None,
            _js=on_change,
            inputs=[
                gr.Text("diffusers.enable_sdp_attention", visible=False),
                enable_sdp_attention,
                gr.Checkbox(True, visible=False),
            ],
            show_progress=False,
        )
        enable_sliced_attention.change(
            fn=None,
            _js=on_change,
            inputs=[
                gr.Text("diffusers.enable_sliced_attention", visible=False),
                enable_sliced_attention,
                gr.Checkbox(False, visible=False),
            ],
            show_progress=False,
        )
        sequential_cpu_offload.change(
            fn=None,
            _js=on_change,
            inputs=[
                gr.Text("diffusers.sequential_cpu_offload", visible=False),
                sequential_cpu_offload,
                gr.Checkbox(True, visible=False),
            ],
            show_progress=False,
        )
        full_model_offload.change(
            fn=None,
            _js=on_change,
            inputs=[
                gr.Text("diffusers.full_model_offload", visible=False),
                full_model_offload,
                gr.Checkbox(True, visible=False),
            ],
            show_progress=False,
        )
        channels_last_memory.change(
            fn=None,
            _js=on_change,
            inputs=[
                gr.Text("diffusers.channels_last_memory", visible=False),
                channels_last_memory,
                gr.Checkbox(True, visible=False),
            ],
            show_progress=False,
        )
        torch_code_compilation.change(
            fn=None,
            _js=on_change,
            inputs=[
                gr.Text("diffusers.torch_code_compilation", visible=False),
                torch_code_compilation,
                gr.Checkbox(True, visible=False),
            ],
            show_progress=False,
        )
        use_deterministic_algorithms.change(
            fn=None,
            _js=on_change,
            inputs=[
                gr.Text("diffusers.use_deterministic_algorithms", visible=False),
                use_deterministic_algorithms,
                gr.Checkbox(False, visible=False),
            ],
            show_progress=False,
        )
        use_tf32_mode.change(
            fn=None,
            _js=on_change,
            inputs=[
                gr.Text("diffusers.use_tf32_mode", visible=False),
                use_tf32_mode,
                gr.Checkbox(False, visible=False),
            ],
            show_progress=False,
        )
        run_prior_on_cpu.change(
            fn=None,
            _js=on_change,
            inputs=[
                gr.Text("diffusers.run_prior_on_cpu", visible=False),
                run_prior_on_cpu,
                gr.Checkbox(False, visible=False),
            ],
            show_progress=False,
        )
    return diffusers_options
