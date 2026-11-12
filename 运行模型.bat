@echo off
REM One-click launcher for the interactive playground.
REM Double-click this file, or run it from a terminal in this folder.
chcp 65001 >nul
cd /d "%~dp0"
".venv\Scripts\python.exe" "src\chat.py" %*
pause
