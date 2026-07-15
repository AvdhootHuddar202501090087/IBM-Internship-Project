@echo off
title Chef Aanya ? Recipe Agent
color 0A
echo.
echo  ==========================================
echo   Chef Aanya - Recipe Preparation Agent
echo  ==========================================
echo.
echo  Starting server...
echo  Open your browser at: http://127.0.0.1:5000
echo.
echo  Press Ctrl+C to stop the server.
echo  ==========================================
echo.

cd /d "%~dp0"
call .venv\Scripts\activate.bat
flask run

echo.
echo  Server stopped. Press any key to close.
pause > nul
