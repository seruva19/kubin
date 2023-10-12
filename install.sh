#!/bin/bash

python -m venv venv
. venv/bin/activate
pip install -r requirements.txt --no-cache-dir --ignore-installed --force-reinstall
