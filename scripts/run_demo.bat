@echo off
REM Demo script for Windows

echo ====================================
echo Worker Detection System - Demo
echo ====================================
echo.

REM Check if virtual environment exists
if not exist "venv\" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt

REM Download models
echo Downloading and converting models...
python -m src.models.download_models

REM Run demo
echo Running demo...
python -m src.main demo

echo.
echo Demo complete! Check data/outputs/ for results.
echo.
echo To view the dashboard, run:
echo   streamlit run src/app.py
echo.

pause

