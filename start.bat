@echo off
setlocal enableextensions enabledelayedexpansion

REM Start script for local Whisper Desktop Dictation on Windows
REM - Creates a virtual environment in .venv
REM - Installs dependencies (repo package + desktop requirements)
REM - Launches the application

set VENV_DIR=.venv
set VENV_PY=%VENV_DIR%\Scripts\python.exe
set VENV_PIP=%VENV_DIR%\Scripts\pip.exe

where %VENV_PY% >nul 2>nul
if %errorlevel% neq 0 (
  echo [Setup] Creating virtual environment...
  py -3 -m venv %VENV_DIR%
)

echo [Setup] Upgrading pip...
%VENV_PY% -m pip install --upgrade pip setuptools wheel

REM Optional: install CUDA-enabled PyTorch first for GPU use
echo [Setup] Ensuring CUDA-enabled PyTorch (GPU)...
REM Remove any existing CPU-only torch first
%VENV_PIP% uninstall -y torch torchvision torchaudio >nul 2>nul
REM Try CUDA 12.1 channel (GPU wheel)
%VENV_PIP% install --no-cache-dir --force-reinstall --index-url https://download.pytorch.org/whl/cu121 torch -U
if %errorlevel% neq 0 (
  echo [Warn] CUDA 12.1 torch install failed; trying CUDA 12.4 channel...
  %VENV_PIP% install --no-cache-dir --force-reinstall --index-url https://download.pytorch.org/whl/cu124 torch -U
)
echo [Check] Verifying CUDA availability in PyTorch...
%VENV_PY% - <<PYCODE
import torch
print("torch:", torch.__version__)
print("torch.version.cuda:", getattr(torch.version, 'cuda', None))
print("cuda.is_available:", torch.cuda.is_available())
PYCODE

echo [Setup] Installing core Whisper package...
REM Install this repository as a package (uses pyproject.toml)
%VENV_PIP% install -e .
if %errorlevel% neq 0 (
  echo [Error] Failed to install core package. Aborting.
  exit /b 1
)

echo [Setup] Installing desktop app dependencies...
%VENV_PIP% install -r desktop\requirements.txt
if %errorlevel% neq 0 (
  echo [Error] Failed to install desktop dependencies. Aborting.
  exit /b 1
)

echo [Run] Launching desktop app...
%VENV_PY% desktop\app.py

endlocal
pause 
