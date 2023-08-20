import gradio as gr
from env import Kubin
from ui_blocks.i2i import i2i_ui
from ui_blocks.inpaint import inpaint_ui
from ui_blocks.mix import mix_ui
from ui_blocks.outpaint import outpaint_ui
from ui_blocks.settings import settings_ui
from ui_blocks.settings_system import system_ui
from ui_blocks.extensions import extensions_ui
from ui_blocks.shared.ui_shared import SharedUI
from ui_blocks.t2i import t2i_ui
from ui_blocks.shared.client import css_styles, js_loader


def gradio_ui(kubin: Kubin, start_fn):
    ext_standalone = kubin.ext_registry.standalone()

    ext_start_tab_index = 5
    ext_target_images = create_ext_targets(ext_standalone, ext_start_tab_index)
    ext_client_folders, ext_client_resources = kubin.ext_registry.locate_resources()

    ui_shared = SharedUI(kubin, ext_target_images, kubin.ext_registry.injectable())
    kubin.ui = ui_shared

    with gr.Blocks(
        title="Kubin: Web-GUI for Kandinsky 2.x",
        theme=ui_shared.select_theme(kubin.params("gradio", "theme")),
        css=css_styles,
        analytics_enabled=kubin.params("gradio", "analytics"),
    ) as ui:
        session = gr.Textbox("-1", visible=False)

        ui.load(
            fn=None,
            _js=js_loader(ext_client_resources, kubin.params.to_json()),
            inputs=[session],
            outputs=[session],
        )

        with gr.Tabs() as ui_tabs:
            with gr.TabItem("Text To Image", id=0) as t2i_tabitem:
                t2i_ui(
                    generate_fn=lambda params: kubin.model.t2i(params),
                    shared=ui_shared,
                    tabs=ui_tabs,
                    session=session,
                )

            with gr.TabItem("Image To Image", id=1) as i2i_tabitem:
                i2i_ui(
                    generate_fn=lambda params: kubin.model.i2i(params),
                    shared=ui_shared,
                    tabs=ui_tabs,
                    session=session,
                )

            with gr.TabItem("Mix Images", id=2) as mix_tabitem:
                mix_ui(
                    generate_fn=lambda params: kubin.model.mix(params),
                    shared=ui_shared,
                    tabs=ui_tabs,
                    session=session,
                )

            with gr.TabItem("Inpainting", id=3) as inpaint_tabitem:
                inpaint_ui(
                    generate_fn=lambda params: kubin.model.inpaint(params),
                    shared=ui_shared,
                    tabs=ui_tabs,
                    session=session,
                )

            with gr.TabItem("Outpainting", id=4) as outpaint_tabitem:
                outpaint_ui(
                    generate_fn=lambda params: kubin.model.outpaint(params),
                    shared=ui_shared,
                    tabs=ui_tabs,
                    session=session,
                )

            create_ext_tabs(ext_standalone, ext_start_tab_index, ui_shared, ui_tabs)
            next_id = len(ext_standalone) + ext_start_tab_index

            with gr.TabItem("Extensions", id=next_id + 2) as extensions_tabitem:
                extensions_ui(kubin)

            with gr.TabItem("Settings", id=next_id + 3) as settings_tabitem:
                settings_ui(kubin, start_fn, ui)

            with gr.TabItem("System/Debug", id=next_id + 4) as system_tabitem:
                system_ui(kubin)

        ui_tabs.elem_classes = [
            "ui-tabs",
            "left" if kubin.params("ui", "side_tabs") else "",
        ]
    return ui, ext_client_folders


def create_ext_targets(exts, ext_start_tab_index):
    ext_targets = []
    for tab_index, ext in enumerate(exts):
        target = ext.get("send_target", None)
        if target is not None:
            ext_targets.append((ext, target, ext_start_tab_index + tab_index))

    return ext_targets


def create_ext_tabs(exts, ext_start_tab_index, ui_shared, tabs):
    ext_ui = []
    for tab_index, ext in enumerate(exts):
        title = ext.get("tab_title", ext["title"])
        with gr.TabItem(title, id=ext_start_tab_index + tab_index):
            ext_ui.append(ext["tab_ui"](ui_shared, tabs))
