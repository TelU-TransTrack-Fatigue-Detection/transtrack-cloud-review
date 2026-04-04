@echo off
setlocal

set ENV_NAME=transtrack_test
set PROJECT_DIR=C:\Users\lulay\Desktop\transtrack-api

echo [1/5] Creating conda environment: %ENV_NAME%...
call conda create -n %ENV_NAME% python=3.11 -y
if errorlevel 1 goto error

echo.
echo [2/5] Activating and installing core requirements...
call conda activate %ENV_NAME%
cd /d %PROJECT_DIR%
pip install -r requirements.txt
if errorlevel 1 goto error

echo.
echo [3/5] Installing PyTorch (GPU)...
pip install -r requirements-torch-gpu.txt
if errorlevel 1 goto error

echo.
echo [4/5] Installing dev requirements...
pip install -r requirements-dev.txt
if errorlevel 1 goto error

echo.
echo [5/5] Running tests...
pytest -v --ignore=tests/test_integration.py
if errorlevel 1 goto testfail

echo.
echo =============================================
echo  All tests passed!
echo =============================================
goto end

:testfail
echo.
echo =============================================
echo  Tests finished with failures (see above).
echo =============================================
goto end

:error
echo.
echo [ERROR] Setup failed at the step above.
exit /b 1

:end
endlocal
