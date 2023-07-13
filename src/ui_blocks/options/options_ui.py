from dataclasses import dataclass
import gradio as gr
from env import Kubin


def options_tab_ui(kubin: Kubin):
    on_change = "(key, value, requiresRestart) => kubin.utils.optionsChanged(key, value, requiresRestart)"

    with gr.Column(elem_classes=["options-block", "options-ui"]) as ui_options:
        allow_params_panel_resize = gr.Checkbox(
            value=kubin.params("ui", "allow_params_panel_resize"),
            label="Allow resize of parameters panel",
        )

        enable_vertical_alignment = gr.Checkbox(
            value=kubin.params("ui", "enable_vertical_alignment"),
            label="Enable panels vertical alignment",
        )

        collapse_advanced_params = gr.Checkbox(
            value=kubin.params("ui", "collapse_advanced_params"),
            label="Collapse parameters panel on launch",
        )

        full_screen_panel = gr.Checkbox(
            value=kubin.params("ui", "full_screen_panel"),
            label="Allow UI stretch to the screen edges",
        )

        side_tabs = gr.Checkbox(
            value=kubin.params("ui", "side_tabs"),
            label="Put tabs to left side of the screen",
        )

    allow_params_panel_resize.change(
        fn=None,
        _js=on_change,
        inputs=[
            gr.Text("ui.allow_params_panel_resize", visible=False),
            allow_params_panel_resize,
            gr.Checkbox(False, visible=False),
        ],
        show_progress=False,
    )

    enable_vertical_alignment.change(
        fn=None,
        _js=on_change,
        inputs=[
            gr.Text("ui.enable_vertical_alignment", visible=False),
            enable_vertical_alignment,
            gr.Checkbox(False, visible=False),
        ],
        show_progress=False,
    )

    collapse_advanced_params.change(
        fn=None,
        _js=on_change,
        inputs=[
            gr.Text("ui.collapse_advanced_params", visible=False),
            collapse_advanced_params,
            gr.Checkbox(True, visible=False),
        ],
        show_progress=False,
    )

    full_screen_panel.change(
        fn=None,
        _js=on_change,
        inputs=[
            gr.Text("ui.full_screen_panel", visible=False),
            full_screen_panel,
            gr.Checkbox(False, visible=False),
        ],
        show_progress=False,
    )

    side_tabs.change(
        fn=None,
        _js=on_change,
        inputs=[
            gr.Text("ui.side_tabs", visible=False),
            side_tabs,
            gr.Checkbox(True, visible=False),
        ],
        show_progress=False,
    )

    return ui_options
