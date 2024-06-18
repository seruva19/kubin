python -m venv venv
call venv\Scripts\activate.bat
pip install -r requirements.txt --no-cache-dir --ignore-installed --force-reinstall
call extensions\reset.bat