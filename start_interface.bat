@echo off
setlocal
cd /d "%~dp0"
if not exist "venv\Scripts\python.exe" (
    echo Could not find venv\Scripts\python.exe in the project folder.
    pause
    exit /b 1
)
"venv\Scripts\python.exe" -m streamlit run "interface_handler.py"
if errorlevel 1 (
    echo.
    echo Streamlit could not start. See the error above.
    pause
    exit /b 1
)
endlocal
