@echo off
rem Launch the handwritten digit recognizer from this folder.
cd /d "%~dp0"
"%LOCALAPPDATA%\Programs\Python\Python312\python.exe" digits.py
if errorlevel 1 pause
