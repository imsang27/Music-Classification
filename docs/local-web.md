# 브라우저에서 음악 분류하기

현재 MuQ 모델은 Python과 모델 가중치가 필요합니다. GitHub Pages는 정적 파일만 제공하며 Python 서버를 실행하지 않으므로, 현재 프로그램을 github.io에서 단독 실행할 수 없습니다. [GitHub 공식 설명](https://docs.github.com/en/pages/getting-started-with-github-pages/creating-a-github-pages-site)

대신 이 컴퓨터에서 서버를 열고 브라우저로 사용할 수 있습니다. 새 로컬 웹 화면은 기존 familiar 구성과 같은 분류 함수를 사용합니다. 이전 Flask/Wav2Vec2 웹앱과 별도의 실행 프로그램입니다.

## 처음 설치할 때

프로젝트 폴더의 PowerShell에서 실행합니다. 기존 `.venv` 환경이 있으면 첫 번째 명령은 생략합니다.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[desktop,web]"
.\.venv\Scripts\python.exe scripts/setup_tagging_model.py
```

이미 같은 프로젝트에서 모델을 설치했다면 모델 준비 명령을 다시 실행할 필요가 없습니다. 모델은 약 2.65GB이며 설치·사용 조건과 YouTube용 FFmpeg 설정은 [모델 설치 안내](desktop-tagging.md)를 따릅니다. 실행 프로그램은 모델을 자동으로 새로 학습하거나 변경하지 않습니다.

## 실행과 사용

1. **로컬_웹_실행.cmd**를 더블클릭합니다.
2. 브라우저가 자동으로 열립니다. 열리지 않으면 **http://127.0.0.1:8765**에 접속합니다.
3. 음원 파일 또는 YouTube 링크를 선택하고 **분류 시작**을 누릅니다.
4. 장르 대표값과 감정·분위기 각각 최대 3개 후보를 확인합니다.
5. **Markdown 저장**, **태그 JSON 저장**, **상세 결과 저장**으로 원하는 결과를 내려받습니다.

서버 창을 열어 둔 동안 사용할 수 있습니다. 종료하려면 해당 창에서 **Ctrl+C**를 누릅니다. 진행 중인 분석이 있으면 정리를 마친 후 종료됩니다. 브라우저 탭을 닫아도 시작한 분석은 계속됩니다. 같은 서버 실행 중에는 새로고침하면 진행 중이거나 이 탭에서 마지막으로 확인한 작업을 다시 볼 수 있습니다. 서버 재시작 후에는 새 작업을 시작하며, 이전 결과는 디스크에 남아 있습니다.

파일 입력은 WAV·MP3·FLAC·OGG·M4A·AAC를 지원하며 최대 20개, 파일당 100MiB, 합계 500MiB입니다. 파일 복사본을 임시로 분석한 뒤 삭제합니다. 같은 이름의 파일도 서로 덮어쓰지 않습니다. 원본 음원과 Obsidian 노트는 수정하지 않습니다.

YouTube 입력은 로그인 없이 열리는 20분 이하 영상 한 곡입니다. 재생목록 전체는 지원하지 않습니다. 분석용 다운로드는 분류 후 삭제하며 영상 제목·채널 정보는 분류 근거로 사용하지 않습니다. 접근 제한이나 네트워크 상태로 링크 입력이 실패할 수 있습니다.

분석은 한 작업씩 실행합니다. 첫 실행과 여러 곡 분석에는 시간이 걸릴 수 있으며, 화면은 작업 상태를 표시합니다. 실제 진행률을 계산한 백분율은 표시하지 않습니다. 실패한 파일은 실패로 남기고 임의의 태그를 만들지 않습니다.

결과는 프로젝트의 `outputs/local-web/<작업ID>/`에 자동 저장됩니다. 업로드 오디오는 결과 폴더에 보관하지 않습니다. JSON의 대표값과 대안 후보, 원시 점수의 의미는 [분석·평가 안내](research/README.md)를 참고하세요.

## 다른 포트 또는 기존 모델 설정 사용

```powershell
.\.venv\Scripts\python.exe local_tagging_server.py --port 8766
.\.venv\Scripts\python.exe local_tagging_server.py --settings "C:\path\model_paths.local.json"
```

기존 다른 폴더의 모델을 재사용하려면 `config/model_paths.local.json`에 MuQ 설치·Python·캐시 경로를 설정하거나 `--settings`로 설정 파일을 지정합니다. `MUSIC_CLASSIFICATION_SETTINGS`도 지원합니다. 경로는 개인 설정이며 GitHub에 올리지 않습니다. 화면의 준비 상태는 설정·실행 파일·런타임 기록의 존재 확인이고, 가중치 검증과 실제 모델 적재는 분류 실행 시 진행합니다.

서버는 `127.0.0.1`에만 열립니다. 외부 컴퓨터·휴대전화·github.io 화면에서 이 서버를 호출하는 구성은 포함하지 않습니다. HTML·CSS·JavaScript도 로컬에서 제공하며 외부 UI 서비스나 CDN에 의존하지 않습니다.

## 개발 확인

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[desktop-test]"
.\.venv\Scripts\python.exe scripts/test_desktop.py
```

웹 검사에는 업로드·복수 파일·중복 이름·실패·용량 제한·임시 파일 정리·동시 요청·링크 전달·결과 다운로드와 다른 출처 요청 차단을 포함합니다. 자동 검사에서는 모델·YouTube 접근을 대체하고 실제 분류 품질을 채점하지 않습니다.
