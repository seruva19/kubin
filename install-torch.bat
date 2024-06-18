call venv\Scripts\activate.bat
pip uninstall -y torch
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu118 --force-reinstall --no-deps

PAUSE