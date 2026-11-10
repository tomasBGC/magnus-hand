@echo off
cd /d "%~dp0"

echo =============================================
echo   eFlesh Microstructure Generator
echo =============================================
echo.

python generate_eflesh_gui.py

echo.
pause
