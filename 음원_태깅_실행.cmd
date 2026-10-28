@echo off
chcp 65001 >nul
setlocal
set "APP_PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%APP_PYTHON%" set "APP_PYTHON=%~dp0work\test-env\Scripts\python.exe"
if not exist "%APP_PYTHON%" (
  echo 태깅 프로그램의 실행 환경이 없습니다.
  pause
  exit /b 2
)
"%APP_PYTHON%" -u -X utf8 "%~dp0tag_audio.py" %*
set "TAGGING_EXIT=%ERRORLEVEL%"
pause
exit /b %TAGGING_EXIT%
