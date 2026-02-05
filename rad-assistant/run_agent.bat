@echo off
REM Windows batch script to run the Radiology Assistant

cd /d "%~dp0"
python run_agent.py %*
