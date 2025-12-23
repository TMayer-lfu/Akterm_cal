@echo off
setlocal
rem Launch Streamlit UI using Python 3.11 from repo root so app package is importable
cd /d "%~dp0"
set PYTHONPATH=%CD%
py -3.11 -m streamlit run app/ui/streamlit_app.py
echo.
echo Zum Beenden dieses Fensters schliessen oder Strg+C druecken.
pause
