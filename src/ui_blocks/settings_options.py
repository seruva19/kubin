from dataclasses import dataclass
import gradio as gr
import json
from deepdiff import DeepDiff
from env import Kubin
from utils.gradio_ui import click_and_disable
from ui_blocks.options.options_native import options_tab_native
from ui_blocks.options.options_diffusers import options_tab_diffusers
from ui_blocks.options.options_general import options_tab_general
from ui_blocks.options.options_gradio import options_tab_gradio
from ui_blocks.options.options_ui import options_tab_ui


def options_ui(kubin: Kubin, start_fn, ui):
    with gr.Row() as options_block:
        with gr.Column(scale=1, elem_classes="options-left"):
            with gr.Box():
                gr.HTML(
                    "General",
                    elem_id="options-general",
                    elem_classes=["options-select", "selected"],
                )
                gr.HTML(
                    "Gradio", elem_id="options-gradio", elem_classes="options-select"
                )
                gr.HTML("UI", elem_id="options-ui", elem_classes="options-select")
                gr.HTML(
                    "Native", elem_id="options-native", elem_classes="options-select"
                )
                gr.HTML(
                    "Diffusers",
                    elem_id="options-diffusers",
                    elem_classes="options-select",
                )

        with gr.Column(scale=5):
            options_tab_general(kubin)
            options_tab_gradio(kubin)
            options_tab_ui(kubin)
            options_tab_native(kubin)
            options_tab_diffusers(kubin)

    with gr.Row():
        apply_changes = gr.Button(
            value="🆗 Apply changes",
            label="Apply changes",
            interactive=True,
        )

        save_changes = gr.Button(
            value="💾 Save applied changes",
            label="Save changes",
            interactive=False,
        )

        reset_changes = gr.Button(value="⏮️ Reset to default", label="Reset to default")
        restart_app = gr.Button(value="♻️ Restart app", label="Restart app")

    with gr.Row():
        options_info = gr.HTML("", elem_id="options-info", elem_classes=["block-info"])

    def apply(json_changes):
        changes = json.loads(json_changes)
        for key, value in changes.items():
            key_path = key.split(".")
            updated_conf = kubin.params._updated

            for key_item in key_path[:-1]:
                updated_conf = updated_conf[key_item]
            updated_conf[key_path[-1]] = value

        diff = DeepDiff(kubin.params.conf, kubin.params._updated, ignore_order=True)
        print(f"applying changes: {diff}")

        requires_reload = kubin.params.apply_config_changes()
        if requires_reload:
            kubin.with_pipeline()

        return [True]

    apply_changes_success = gr.Checkbox(False, visible=False)
    applied_changes = gr.Text("", visible=False)

    apply_changes.click(
        _js="(c) => kubin.utils.changedOptions()",
        inputs=[applied_changes],
        fn=apply,
        queue=False,
        outputs=[apply_changes_success],
    ).then(
        _js="(res) => kubin.utils.processOptionsChanges(res)",
        inputs=[apply_changes_success],
        fn=lambda success: gr.update(interactive=success),
        outputs=[save_changes],
    )

    save_changes.click(
        fn=lambda: kubin.params.save_user_config(), queue=False, show_progress=False
    ).then(
        fn=None,
        _js='(s) => (kubin.notify.success("Custom configuration file saved"), s)',
        inputs=[apply_changes_success],
        outputs=[apply_changes_success],
    )

    reset_changes.click(
        fn=lambda: kubin.params.reset_config(), queue=False, show_progress=False
    ).then(
        fn=None,
        _js='() => kubin.notify.success("Restored default config. Restart the app for changes to take effect")',
    )

    restart_app.click(
        fn=lambda: start_fn(kubin, ui),
        queue=False,
        show_progress=False,
        _js="() => kubin.utils.reloadUI()",
    )

    return options_block
