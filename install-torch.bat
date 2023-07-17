call venv\Scripts\activate.bat
pip uninstall -y torch
pip install torch --index-url https://download.pytorch.org/whl/cu118 --force-reinstall --no-deps
