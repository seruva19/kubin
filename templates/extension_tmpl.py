import gradio as gr


def setup(kubin):
    image = gr.Image()

    def extension_tab_ui(ui_shared, ui_tabs):
        with gr.Column() as extension_block:
            image.render()

        return extension_block

    def extension_injection_ui(target):
        state = gr.State({"key": "value"})

        with gr.Column() as extension_injection_block:
            None

        return extension_injection_block, state

    def extension_injection_fn(params, state):
        return params

    def extension_settings_ui():
        with gr.Column() as extension_settings_block:
            None
        return extension_settings_block

    def extension_hook_fn(params):
        return params

    return {
        "title": "Example",
        "tab_ui": lambda ui_shared, ui_tabs: extension_tab_ui(ui_shared, ui_tabs),
        "inject_ui": lambda target: extension_injection_ui(target),
        "inject_fn": lambda target, params, augmentations: extension_injection_fn(
            params, augmentations[0]
        ),
        "inject_position": "before_cnet",
        "targets": ["t2i", "i2i", "mix", "inpaint", "outpaint"],
        "hook_fn": lambda hook, params: extension_hook_fn(params),
        "settings_ui": lambda: extension_settings_ui(),
        "send_to": "✋ Open in 'Example' extension",
        "send_target": image,
        "api": {"method": lambda api_params: kubin.log(api_params)},
        "supports_pipeline": None,
        "supports_model": None,
    }
