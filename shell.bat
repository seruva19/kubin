@echo off
if not exist "venv\Scripts\activate.bat" (
    echo Virtual environment not found. Creating venv...
    python -m venv venv
)

call venv\Scripts\activate.bat

echo Virtual environment activated.
echo You can now use pip install within this environment.
cmd /K
