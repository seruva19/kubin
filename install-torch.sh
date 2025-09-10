#!/bin/bash

. venv/bin/activate
pip uninstall -y torch
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118 --force-reinstall --no-deps
