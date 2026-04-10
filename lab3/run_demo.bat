@echo off
cd /d "%~dp0"
python check_env.py || exit /b 1
python scripts\inference_compare.py
pause
