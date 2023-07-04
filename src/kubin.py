from arguments import parse_arguments
from env import Kubin
from webui import gradio_ui
import gradio as gr
from pathlib import Path

kubin = Kubin()
args = parse_arguments()


def init_kubin(kubin: Kubin):
    kubin.with_args(args)
    kubin.with_pipeline()
    kubin.with_utils()
    kubin.with_extensions()


def start(kubin, ui: gr.Blocks):
    if ui is not None:
        ui.close()

    init_kubin(kubin)
    ui, resources = gradio_ui(kubin, start)

    ui.queue(
        concurrency_count=kubin.params("gradio", "concurrency_count"), api_open=False
    ).launch(
        prevent_thread_lock=True,
        show_api=False,
        debug=kubin.params("gradio", "debug"),
        show_error=True,
        share=kubin.params("general", "share") == "gradio",
        server_name=kubin.params("gradio", "server_name"),
        server_port=kubin.params("gradio", "server_port"),
        allowed_paths=[f"{Path(__file__).parent.parent.absolute()}/client"] + resources,
    )


start(kubin, None)
