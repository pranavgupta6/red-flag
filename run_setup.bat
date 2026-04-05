@echo off
cd /d "D:\zz projects\red-flag"
python setup_project.py
if %ERRORLEVEL% EQU 0 (
    echo.
    echo Verifying created structure...
    dir /s /b "financial-audit-openenv"
) else (
    echo Script execution failed with error code %ERRORLEVEL%
)
