from arguments import parse_arguments
from env import Kubin
from utils.platform import is_windows
from web_gui import gradio_ui
from pathlib import Path
import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import gradio.analytics

check_executed = False


def custom_version_check():
    global check_executed
    if not check_executed:
        check_executed = True
        print(
            "fyi: kubin uses an old version of Gradio (3.50.2), which is now considered deprecated for security reasons.\nhowever, the author is too stubborn to upgrade (https://github.com/seruva19/kubin/blob/main/DOCS.md#gradio-4)."
        )


gradio.analytics.version_check = custom_version_check

kubin = Kubin()
args = parse_arguments()


def init_kubin(kubin: Kubin):
    kubin.with_args(args)
    kubin.with_utils()
    kubin.with_extensions()
    kubin.with_hooks()
    kubin.with_pipeline()


def reload_app(ui, kubin):
    kubin.model.flush()

    from subprocess import Popen

    Popen(
        ["start.bat"]
        if is_windows()
        else [
            "/bin/bash",
            "-c",
            f"chmod u+x '{Path(__file__).parent.parent.absolute()}/start.sh' && ./start.sh",
        ]
    )

    try:
        ui.close()
        raise SystemExit
    finally:
        None


def start(kubin, ui):
    if ui is not None:
        reload_app(ui, kubin)

    init_kubin(kubin)
    ui, resources = gradio_ui(kubin, start)

    app, local, shared = ui.queue(
        concurrency_count=kubin.params("gradio", "concurrency_count"), api_open=True
    ).launch(
        prevent_thread_lock=True,
        show_api=False,
        debug=kubin.params("gradio", "debug"),
        favicon_path="client/favicon.png",
        show_error=True,
        share=kubin.params("general", "share") == "gradio",
        server_name=kubin.params("gradio", "server_name"),
        server_port=kubin.params("gradio", "server_port"),
        allowed_paths=[f"{Path(__file__).parent.parent.absolute()}/client"] + resources,
    )


start(kubin, None)
