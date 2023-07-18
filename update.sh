#!/bin/bash

git pull
git checkout .
source venv/bin/activate
pip install -r requirements.txt