@echo off
setlocal

set ENV_NAME=transtrack_test
set PYTHON_VERSION=3.11

echo =============================================
echo  TransTrack Cloud Review — Environment Setup
echo =============================================
echo.

set /p GPU_CHOICE="Do you have an NVIDIA GPU with CUDA? (y/n): "

echo.
echo [1/4] Creating conda environment: %ENV_NAME% (Python %PYTHON_VERSION%)...
call conda create -n %ENV_NAME% python=%PYTHON_VERSION% -y
if errorlevel 1 goto error

echo.
echo [2/4] Installing core dependencies...
call conda run -n %ENV_NAME% pip install -r requirements.txt
if errorlevel 1 goto error

echo.
echo [3/4] Installing PyTorch...
if /i "%GPU_CHOICE%"=="y" (
    echo Installing GPU version ^(CUDA 12.1^)...
    call conda run -n %ENV_NAME% pip install -r requirements-torch-gpu.txt
) else (
    echo Installing CPU version...
    call conda run -n %ENV_NAME% pip install -r requirements-torch-cpu.txt
)
if errorlevel 1 goto error

echo.
echo [4/4] Installing dev dependencies ^(pytest^)...
call conda run -n %ENV_NAME% pip install -r requirements-dev.txt
if errorlevel 1 goto error

echo.
echo =============================================
echo  Setup complete!
echo.
echo  Activate with:
echo    conda activate %ENV_NAME%
echo.
echo  Then verify model:
echo    python scripts/check_model.py
echo.
echo  Run tests:
echo    pytest -v --ignore=tests/test_integration.py
echo =============================================
goto end

:error
echo.
echo [ERROR] Setup failed. Check the output above.
exit /b 1

:end
endlocal
