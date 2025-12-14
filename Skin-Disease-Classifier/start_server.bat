@echo off
echo ======================================================================
echo HTML Sunucusu Baslatiliyor...
echo ======================================================================
echo.
echo Tarayicida su adresi acin:
echo http://localhost:8000/analyze.html
echo.
echo Durdurmak icin CTRL+C basin
echo ======================================================================
echo.
cd /d "%~dp0"
python -m http.server 8000

