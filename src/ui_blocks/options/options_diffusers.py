from dataclasses import dataclass
import gradio as gr
from env import Kubin


def options_tab_diffusers(kubin: Kubin):
    updated_config = kubin.params._updated

    with gr.Column(elem_classes=["options-block", "options-diffusers"]) as diffusers:
        half_precision_weights = gr.Checkbox(
            value=kubin.params("diffusers", "half_precision_weights"),
            label="Enable half precision weights",
        )
        enable_xformers = gr.Checkbox(
            value=kubin.params("diffusers", "enable_xformers"),
            label="Enable xformers memory efficient attention",
        )
        enable_sdpa_attention = gr.Checkbox(
            value=kubin.params("diffusers", "enable_sdpa_attention"),
            label="Enable SDPA attention",
        )
        enable_sliced_attention = gr.Checkbox(
            value=kubin.params("diffusers", "enable_sliced_attention"),
            label="Enable sliced attention",
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
        use_tf32_mode = gr.Checkbox(
            value=kubin.params("diffusers", "use_tf32_mode"),
            label="Enable TensorFloat32 mode",
        )

        def change_value(key, value):
            updated_config["diffusers"][key] = value

        half_precision_weights.change(
            change_value,
            inputs=[gr.State("half_precision_weights"), half_precision_weights],
        )
        enable_xformers.change(
            change_value, inputs=[gr.State("enable_xformers"), enable_xformers]
        )
        enable_sdpa_attention.change(
            change_value,
            inputs=[gr.State("enable_sdpa_attention"), enable_sdpa_attention],
        )
        enable_sliced_attention.change(
            change_value,
            inputs=[gr.State("enable_sliced_attention"), enable_sliced_attention],
        )
        sequential_cpu_offload.change(
            change_value,
            inputs=[gr.State("sequential_cpu_offload"), sequential_cpu_offload],
        )
        full_model_offload.change(
            change_value, inputs=[gr.State("full_model_offload"), full_model_offload]
        )
        channels_last_memory.change(
            change_value,
            inputs=[gr.State("channels_last_memory"), channels_last_memory],
        )
        torch_code_compilation.change(
            change_value,
            inputs=[gr.State("torch_code_compilation"), torch_code_compilation],
        )
        use_deterministic_algorithms.change(
            change_value,
            inputs=[
                gr.State("use_deterministic_algorithms"),
                use_deterministic_algorithms,
            ],
        )
        use_tf32_mode.change(
            change_value, inputs=[gr.State("use_tf32_mode"), use_tf32_mode]
        )
    return diffusers
