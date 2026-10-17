@echo off
title Autonomous ZIB Materials Acceleration Platform (MAP)
cd /d "%~dp0"
echo =====================================================================
echo  Autonomous Materials Acceleration Platform (MAP) for Aqueous ZIBs
echo =====================================================================
echo Launching with project virtual environment (.venv)...
if exist ".venv\Scripts\streamlit.exe" (
    ".venv\Scripts\streamlit.exe" run app.py
) else (
    streamlit run app.py
)
pause
