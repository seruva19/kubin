from dataclasses import dataclass
import shutil
import subprocess
import gradio as gr
from env import Kubin
from urllib.parse import urlparse
import os


@dataclass
class ExtensionInfo:
    name: str
    title: str
    role: str
    path: str
    enabled: bool
    description: str = ""
    url: str = ""
    has_settings: bool = False
    settings_ui: any = None


def create_extensions_info(kubin: Kubin):
    extensions = [(key, value) for key, value in kubin.ext_registry.extensions.items()]
    get_path = lambda x: f"{kubin.params('general','extensions_path')}/{x}"

    extensions_info = []
    if len(extensions) > 0:
        for extension in extensions:
            ext_id = extension[0]
            ext_props = extension[1]
            extensions_info.append(
                ExtensionInfo(
                    name=ext_id,
                    title=ext_props["title"],
                    role=";".join(
                        filter(
                            lambda a: a is not None,
                            [
                                "tab_ui"
                                if ext_props.get("tab_ui", None) is not None
                                else None,
                                "injectable_ui"
                                if ext_props.get("inject_ui", None) is not None
                                else None,
                            ],
                        )
                    ),
                    path=get_path(ext_id),
                    url="",
                    enabled=True,
                    description=ext_props.get("description", ""),
                    settings_ui=ext_props.get("settings_ui", lambda: None),
                    has_settings=ext_props.get("settings_ui", None) is not None,
                )
            )

    disabled_exts = kubin.ext_registry.get_disabled_extensions()
    if len(disabled_exts) > 0:
        disabled_exts = [ext.strip() for ext in disabled_exts]
        for extension_name in disabled_exts:
            extensions_info.append(
                ExtensionInfo(
                    name=extension_name,
                    title="",
                    role="",
                    path=get_path(extension_name),
                    url="",
                    enabled=False,
                    description="",
                    settings_ui=lambda: None,
                    has_settings=False,
                )
            )

    return sorted(extensions_info, key=lambda extension: extension.name)


def install_extension(kubin, path):
    extensions_root = kubin.params("general", "extensions_path")
    try:
        parsed_url = urlparse(path)
        if parsed_url.scheme and parsed_url.netloc:
            repo_name = parsed_url.path.split("/")[-1]
            target_path = os.path.join(extensions_root, repo_name)
            if os.path.exists(target_path):
                return "extension with the same name already installed"
            subprocess.run(["git", "clone", path, target_path])
            return ""
        elif os.path.isdir(path):
            folder_name = os.path.basename(path)
            target_path = os.path.join(extensions_root, folder_name)
            if os.path.exists(target_path):
                return "extension with the same name already installed"
            else:
                shutil.copytree(path, target_path)
            return ""
        else:
            return "the provided string is neither URL not path"
    except Exception as e:
        return e


def extensions_ui(kubin: Kubin):
    extensions_data = create_extensions_info(kubin)

    with gr.Column() as extensions_block:
        gr.HTML(
            f"Local extensions found: {len(extensions_data)}<br>Enabled extensions: {len(list(filter(lambda x: x.enabled, extensions_data)))}"
        )

        for index, extension_info in enumerate(extensions_data):
            extension_info: ExtensionInfo = extension_info
            disabled_info = (
                "<h3 style='color: red; display: inline'> - disabled</h3>"
                if not extension_info.enabled
                else ""
            )

            with gr.Box() as extension_box:
                extension_box.elem_classes = ["kd-extension-container"]

                gr.HTML(
                    f"<h3 style='display: inline'>{str(index+1)}. {extension_info.name}{disabled_info}</h3>"
                )
                gr.HTML(f"<hr />")
                gr.HTML(f"<br />")

                gr.HTML(f"title: {extension_info.title}")
                gr.HTML(f"description: {extension_info.description}")
                gr.HTML(f"url: {extension_info.url}")
                gr.HTML(f"role: {extension_info.role}")
                gr.HTML(f"path: {extension_info.path}")

                gr.HTML(f"<br />")

                with gr.Row():
                    if extension_info.has_settings:
                        open_settings_btn = gr.Button(
                            value="🛠️ Settings",
                            interactive=True,
                            size="sm",
                            scale=0,
                        )
                        open_settings_btn.click(
                            fn=None,
                            _js=f'_ => (document.querySelector(".kd-extension-settings-container-{extension_info.name}").classList.toggle("hidden"), void 0)',
                        )

                    if extension_info.enabled:
                        disable_ext_btn = gr.Button(
                            value="🛑 Disable",
                            interactive=False,
                            size="sm",
                            scale=0,
                        )
                    else:
                        enable_ext_btn = gr.Button(
                            value="🟢 Enable",
                            interactive=False,
                            size="sm",
                            scale=0,
                        )

                    clear_ext_install_btn = gr.Button(
                        value="🔧 Force reinstall",
                        interactive=True,
                        size="sm",
                        scale=0,
                    )

                    clear_ext_install_btn = gr.Button(
                        value="🗑️ Remove",
                        interactive=False,
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

                with gr.Column() as settings_column:
                    settings_column.elem_classes = [
                        f"kd-extension-settings-container-{extension_info.name}",
                        f"kd-extension-settings-container",
                        "hidden",
                    ]
                    extension_info.settings_ui()

        install_error = gr.Textbox("", visible=False)
        with gr.Accordion("Install extension"):
            extension_url_or_path = gr.Textbox(label="Github URL or path to folder")
            install_extension_btn = gr.Button(
                value="📥 Install extension",
                label="Install",
                interactive=True,
                scale=0,
                size="sm",
            )
            install_extension_btn.click(
                fn=lambda path: install_extension(kubin, path),
                inputs=extension_url_or_path,
                outputs=[install_error],
                queue=False,
            ).then(
                fn=None,
                inputs=[install_error],
                outputs=[install_error],
                show_progress=False,
                _js="""(e) => e == ''
                    ? kubin.notify.success("Successfuly installed extension, restart app to activate it")
                    : kubin.notify.error(`Error installing extension: ${e}`)
                """,
            )

        clear_ext_install_all_btn = gr.Button(
            value="🔧 Force reinstall of all extensions on next launch",
            label="Force reinstall",
            interactive=True,
            scale=0,
            size="sm",
        )
        clear_ext_install_all_btn.click(
            lambda: kubin.ext_registry.force_reinstall(),
            queue=False,
        ).then(
            fn=None,
            _js=f'_ => kubin.notify.success("All extensions will be reinstalled on next launch")',
        )

    extensions_block.elem_classes = ["kd-extensions-list"]
    return extensions_block
