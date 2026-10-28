@echo off
chcp 65001 >nul
setlocal
set "APP_PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%APP_PYTHON%" set "APP_PYTHON=%~dp0work\test-env\Scripts\python.exe"
if not exist "%APP_PYTHON%" (
  echo 분류 프로그램의 실행 환경이 없습니다.
  pause
  exit /b 2
)
"%APP_PYTHON%" -u -X utf8 "%~dp0obsidian_classifier_app.py" %*
if errorlevel 1 (
  echo 분류 또는 저장 중 오류가 발생했습니다. 위 내용을 확인해주세요.
  pause
  exit /b 1
)
endlocal
