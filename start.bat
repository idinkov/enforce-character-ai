@echo off
setlocal

:: Step 1: Check for Python 3.10
echo Checking for Python 3.10...

for /f "delims=" %%i in ('python --version 2^>^&1') do set PYVER=%%i
echo %PYVER% | find "3.10" >nul
if %errorlevel% neq 0 (
    echo Python 3.10 not found.
    echo Downloading Python 3.10 installer...
    set PY_INSTALLER=python-3.10.11-amd64.exe
    powershell -Command "Invoke-WebRequest https://www.python.org/ftp/python/3.10.11/%PY_INSTALLER% -OutFile %PY_INSTALLER%"
    echo Installing Python 3.10 silently...
    start /wait %PY_INSTALLER% /quiet InstallAllUsers=1 PrependPath=1 Include_test=0
    del %PY_INSTALLER%
)

:: Step 2: Create virtual environment
if not exist .venv (
    echo Creating virtual environment...
    python -m venv .venv
)

:: Step 3: Activate and install requirements
echo Activating virtual environment and installing dependencies...
call .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

:: Step 4: Launch the app
echo Launching app...
python main.py

endlocal
pause
