from arguments import parse_arguments
from env import Kubin
from utils.platform import is_windows
from web_gui import gradio_ui
from pathlib import Path

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
