@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set "APP_PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%APP_PYTHON%" set "APP_PYTHON=%~dp0work\web-env\Scripts\python.exe"
if not exist "%APP_PYTHON%" set "APP_PYTHON=%~dp0work\test-env\Scripts\python.exe"
if defined MUSIC_CLASSIFICATION_WEB_PYTHON set "APP_PYTHON=%MUSIC_CLASSIFICATION_WEB_PYTHON%"
if not exist "%APP_PYTHON%" (
  echo 실행 환경이 없습니다. docs/local-web.md의 설치 안내를 확인해주세요.
  pause
  exit /b 2
)
"%APP_PYTHON%" -c "import flask" >nul 2>&1
if errorlevel 1 (
  echo 웹 실행 도구를 먼저 설치해주세요:
  echo "%APP_PYTHON%" -m pip install -e ".[web]"
  pause
  exit /b 2
)
"%APP_PYTHON%" -u -X utf8 "%~dp0local_tagging_server.py" %*
set "APP_EXIT=%ERRORLEVEL%"
pause
exit /b %APP_EXIT%
