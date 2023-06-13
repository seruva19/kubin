import argparse
from env import Kubin
from webui import gradio_ui
from pathlib import Path

parser = argparse.ArgumentParser(description="Run Kubin")
parser.add_argument("--from-config", type=str, default=None)
parser.add_argument("--device", type=str, default=None)
parser.add_argument("--flash-attention", type=str, default=None)
parser.add_argument("--cache-dir", type=str, default=None)
parser.add_argument("--output-dir", type=str, default=None)
parser.add_argument("--share", type=str, default=None)
parser.add_argument("--server-name", type=str, default=None)
parser.add_argument("--server-port", type=int, default=None)
parser.add_argument("--concurrency-count", type=int, default=None)
parser.add_argument("--debug", type=str, default=None)
parser.add_argument("--extensions-path", type=str, default=None)
parser.add_argument("--enabled-extensions", type=str, default=None)
parser.add_argument("--disabled-extensions", type=str, default=None)
parser.add_argument(
    "--extensions-order",
    type=str,
    default=None,
)
parser.add_argument("--skip-install", type=str, default=None)
parser.add_argument("--safe-mode", type=str, default=None)
parser.add_argument("--pipeline", type=str, default=None)
parser.add_argument("--mock", type=str, default=None)
parser.add_argument("--theme", type=str, default=None)

args = parser.parse_args()
args_preview = {key: value for key, value in vars(args).items() if value is not None}
print(f"command line arguments: {args_preview}")

kubin = Kubin(args)
kubin.with_pipeline()
kubin.with_utils()
kubin.with_extensions()

ui, resources = gradio_ui(kubin)
ui.queue(
    concurrency_count=kubin.params("gradio", "concurrency_count"), api_open=False
).launch(
    show_api=False,
    debug=kubin.params("gradio", "debug"),
    show_error=True,
    share=kubin.params("general", "share") == "gradio",
    server_name=kubin.params("gradio", "server_name"),
    server_port=kubin.params("gradio", "server_port"),
    allowed_paths=[f"{Path(__file__).parent.parent.absolute()}/client"] + resources,
)
