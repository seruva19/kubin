from dataclasses import dataclass
import gradio as gr
from env import Kubin


@dataclass
class ExtensionInfo:
    name: str
    title: str
    info: str
    path: str
    enabled: bool
    description: str = ""
    url: str = ""


def create_extensions_info(kubin: Kubin):
    extensions = [(key, value) for key, value in kubin.ext_registry.extensions.items()]
    get_path = lambda x: f"{kubin.params('general','extensions_path')}/{x}"

    extensions_info = []
    if len(extensions) > 0:
        for extension in extensions:
            extensions_info.append(
                ExtensionInfo(
                    name=extension[0],
                    title=extension[1]["title"],
                    info=f'is {"not" if not extension[1].get("tab_ui", None) else ""} standalone, is {"not" if not extension[1].get("inject_ui", None) else ""} injectable',
                    path=get_path(extension[0]),
                    url="",
                    enabled=True,
                )
            )

    disabled_exts = kubin.ext_registry.get_disabled_extensions()
    if len(disabled_exts) > 0:
        disabled_exts = [ext.strip() for ext in disabled_exts]
        for extension_name in disabled_exts:
            extensions_info.append(
                ExtensionInfo(
                    name=extension_name,
                    title="unknown (not loaded)",
                    info="unknown (not loaded)",
                    path=get_path(extension_name),
                    url="",
                    enabled=False,
                )
            )

    return extensions_info


def extensions_ui(kubin: Kubin):
    extensions_data = create_extensions_info(kubin)

    with gr.Column() as extensions_block:
        gr.HTML(
            f"Local extensions found: {len(extensions_data)}<br>Activated extensions: {len(list(filter(lambda x: x.enabled, extensions_data)))}"
        )

        for index, extension_info in enumerate(extensions_data):
            extension_info: ExtensionInfo = extension_info

            with gr.Accordion(
                f"{str(index+1)}. {extension_info.name} {'- disabled' if not extension_info.enabled else ''}",
                open=False,
            ):
                with gr.Box():
                    gr.HTML(f"title: {extension_info.title}")
                    gr.HTML(f"url: {extension_info.url}")
                    gr.HTML(f"description: {extension_info.description}")
                    gr.HTML(f"info: {extension_info.info}")
                    gr.HTML(f"path: {extension_info.path}")
                    gr.HTML("")

                    with gr.Row():
                        clear_ext_install_btn = gr.Button(
                            value="🔧 Force reinstall",
                            interactive=True,
                            size="sm",
                            scale=0,
                        )
                        clear_ext_install_btn.click(
                            lambda name=extension_info.name: kubin.ext_registry.force_reinstall(
                                name
                            ),
                            queue=False,
                        ).then(
                            fn=None,
                            _js=f'_ => kubin.notify.success("Extension {extension_info.name} will be reinstalled on next launch")',
                        )

        clear_ext_install_all_btn = gr.Button(
            value="🔧 Force reinstall of all extensions on next launch",
            label="Force reinstall",
            interactive=True,
        )
        clear_ext_install_all_btn.click(
            lambda: kubin.ext_registry.force_reinstall(),
            queue=False,
        ).then(
            fn=None,
            _js=f'_ => kubin.notify.success("All extensions will be reinstalled on next launch")',
        )

    return extensions_block
