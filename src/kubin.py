import argparse

from webui import launch as ui_launch
from model import Model

parser = argparse.ArgumentParser(description='Run Kubin')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--model-version', type=str, default='2.1')
parser.add_argument('--use-flash-attention', type=bool, default=False)
parser.add_argument('--cache-dir', type=str, default='models')
parser.add_argument('--output-dir', type=str, default='output')
parser.add_argument('--task-type', type=str, default='text2img')
parser.add_argument('--share', type=str, default='none')
#parser.add_argument('--config', type=str, default='config.yaml')
#parser.add_argument('--locale', type=str, default='en-us')

args = parser.parse_args()
print(f'launching with: {vars(args)}')

model = Model(args.device, args.task_type, args.cache_dir, args.model_version, args.use_flash_attention, args.output_dir)
ui_launch(model).queue(concurrency_count=1).launch(debug=True, share=args.share=='gradio', server_name='0.0.0.0', server_port=7860)