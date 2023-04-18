import argparse
from env import Kubin
from webui import gradio_ui

parser = argparse.ArgumentParser(description='Run Kubin')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--model-version', type=str, default='2.1')
parser.add_argument('--use-flash-attention', default=False, action='store_true')
parser.add_argument('--cache-dir', type=str, default='models')
parser.add_argument('--output-dir', type=str, default='output')
parser.add_argument('--task-type', type=str, default='text2img')
parser.add_argument('--share', type=str, default='none')
parser.add_argument('--server-name', type=str, default='0.0.0.0')
parser.add_argument('--server-port', type=int, default=7860)
parser.add_argument('--concurrency-count', type=int, default=2)
parser.add_argument('--debug', default=True, action='store_true')
parser.add_argument('--locale', type=str, default='en-us') # yet unused 
parser.add_argument('--model-config', type=str, default='config.kd2') # yet unused

args = parser.parse_args()
print(f'launching with: {vars(args)}')

kubin = Kubin(args)
ui = gradio_ui(kubin)

ui.queue(concurrency_count=kubin.args.concurrency_count) \
  .launch(debug=kubin.args.debug, share=kubin.args.share=='gradio', server_name=kubin.args.server_name, server_port=kubin.args.server_port)