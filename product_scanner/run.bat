@echo off
title Product Tag Scanner & Official Image Finder

:: Set directory to script folder
cd /d "%~dp0"

echo ========================================================
echo   Product Tag Scanner & Official Image Finder
echo ========================================================
echo.

:: Check if virtual environment python exists
if not exist "venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found at .\venv
    echo Please make sure the venv is set up properly.
    echo.
    pause
    exit /b 1
)

echo Starting Streamlit server...
echo Opening app at http://localhost:8501
echo.

:: Launch Streamlit app using python -m streamlit (bypasses WDAC block on wrapper exe)
venv\Scripts\python.exe -m streamlit run app.py

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Streamlit closed with an error code.
    pause
)
