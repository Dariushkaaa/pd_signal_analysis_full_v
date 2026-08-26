@echo off
title PD Signal Analysis
echo ========================================
echo PD Signal Analysis - Auto Start
echo ========================================
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed!
    echo Please download it from https://www.python.org/downloads/
    pause
    exit /b 1
)

if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

call venv\Scripts\activate.bat

echo Installing base dependencies...
pip install -r requirements.txt --quiet

echo Installing stable CPU version of PyTorch...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --quiet

echo.
echo ========================================
echo Server is starting...
echo Open your browser: http://localhost:8000
echo ========================================
echo.

cd backend
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000

pause