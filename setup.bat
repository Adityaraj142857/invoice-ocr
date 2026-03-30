@echo off
REM ============================================================
REM setup.bat -- Windows setup script
REM Run: setup.bat  (in Command Prompt or PowerShell)
REM ============================================================

echo ==========================================
echo  Invoice OCR -- Setup (Windows)
echo ==========================================

REM Check Python
python --version >nul 2>&1
IF ERRORLEVEL 1 (
    echo ERROR: Python not found. Install from python.org
    pause
    exit /b 1
)

echo.
echo [1/5] Creating virtual environment...
python -m venv venv

echo [2/5] Activating venv...
call venv\Scripts\activate.bat

echo [3/5] Upgrading pip...
python -m pip install --upgrade pip --quiet

echo [4/5] Installing PyTorch (CUDA 12.1 - change if your CUDA version differs)...
REM For CUDA: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
REM For CPU only:
pip install torch torchvision --quiet

echo [5/5] Installing requirements...
pip install -r requirements.txt 

echo.
echo ==========================================
echo  Setup complete!
echo.
echo  NOTE: For PDF support, install poppler:
echo  1. Download from: https://github.com/oschwartz10612/poppler-windows/releases
echo  2. Extract and add the bin/ folder to your PATH
echo.
echo  To run:
echo    venv\Scripts\activate
echo    python executable.py --input data\train_docs\
echo ==========================================
pause
