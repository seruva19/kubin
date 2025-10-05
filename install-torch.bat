call venv\Scripts\activate.bat
pip uninstall -y torch
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126 --force-reinstall --no-deps

PAUSE