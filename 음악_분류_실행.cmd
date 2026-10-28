@echo off
chcp 65001 >nul
setlocal
set "APP_PYTHON=%~dp0work\research-package-r1\consumer-env\Scripts\python.exe"
if not exist "%APP_PYTHON%" (
  echo 분류 프로그램의 Python 환경이 없습니다. 설치 환경을 확인해주세요.
  pause
  exit /b 2
)
"%APP_PYTHON%" -I -u -X utf8 "%~dp0music_classifier_app.py"
if errorlevel 1 (
  echo 프로그램 실행 중 오류가 발생했습니다. 위 내용을 확인해주세요.
  pause
  exit /b 1
)
endlocal
