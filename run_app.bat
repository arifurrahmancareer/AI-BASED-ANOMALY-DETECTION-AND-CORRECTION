@echo off
title SmartDetect - AI Image Anomaly Detection
cd /d "%~dp0"

echo ========================================
echo   SmartDetect - Starting Application
echo ========================================
echo.

:: Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python from https://python.org
    pause
    exit /b 1
)

:: Start the Streamlit app with auto-open browser
echo Starting SmartDetect...
echo The application will open in your browser automatically.
echo.
echo Press Ctrl+C to stop the server.
echo ========================================

python -m streamlit run app.py --server.headless=true --browser.gatherUsageStats=false

pause

