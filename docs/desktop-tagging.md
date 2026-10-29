# 로컬 음악 태깅 설치와 실행

모델 비교·학습·오류 분석과 수치 해석은 [분석·평가 안내](research/README.md)에서 확인할 수 있습니다. 과거 실험의 수치는 현재 familiar 구성 전체의 정확도와 구분합니다.

브라우저로 사용하려면 [로컬 웹 실행 안내](local-web.md)를 참고하세요. 같은 familiar 모델·설명을 사용하며 별도의 로컬 서버를 엽니다.

## 지원하는 기능

이 문서는 Windows 태깅 창의 안내입니다. 브라우저에서는 같은 MuQ 기능을 기존 Flask 페이지에 통합했으며 [통합 웹 안내](local-web.md)에서 AI·규칙·직접 분류와 업로드 관리를 사용할 수 있습니다.

- 음원 하나/여러 개 또는 공개 YouTube 영상 한 곡을 분류합니다.
- 장르25개·감정9개·분위기11개 안에서 가까운 표현을 선택합니다.
- 감정·분위기는 상위3후보와 대표값을 구분해서 보여줍니다. 모두 동시에 맞는 확정 태그라는 뜻은 아닙니다.
- 파일·링크의 소리를 기준으로 분석하며, 가사 의미·아티스트·앨범 정보를 분류 근거로 쓰지 않습니다.
- 결과 JSON과 Markdown을 저장합니다. 원본 음원이나 Obsidian 노트는 수정하지 않습니다.

## 설치

Windows와 Python3.12를 기준으로 합니다. 모델·음원·개인 설정은 GitHub에 포함하지 않습니다.

프로젝트 폴더의 PowerShell에서:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[desktop]"
.\.venv\Scripts\python.exe scripts/setup_tagging_model.py
```

마지막 명령은 공개된 고정 버전의 모델과 소스를 받아 해시를 검사하고 `config/model_paths.local.json`을 생성합니다. MuQ-MuLan 가중치만 약2.65GB이며 여유 디스크와 인터넷 연결이 필요합니다. 이미 파일이 있으면 검사한 뒤 재사용하고 다른 파일로 조용히 덮어쓰지 않습니다. 기존 설치를 재사용하려면 `--asset-cache "기존 설치 폴더" --offline`을 사용할 수 있습니다.

PyTorch와 torchaudio는 서로 맞는 버전/장치로 설치해야 합니다. NVIDIA GPU를 사용하려면 [PyTorch 공식 설치 안내](https://pytorch.org/get-started/locally/)에 따라 환경에 맞는 CUDA 빌드를 설치하세요. CPU도 지원하지만 처리 속도와 긴 영상의 완료 시간은 PC에 따라 달라집니다.

YouTube 입력에는 [FFmpeg](https://ffmpeg.org/download.html)가 필요합니다. 설치 후 PATH에 추가하거나 `MUSIC_CLASSIFICATION_FFMPEG` 환경변수에 실행 파일 경로를 지정하세요. 프로젝트 아래 `ffmpeg-master-latest-win64-gpl/bin/ffmpeg.exe`에 둔 기존 설치도 사용할 수 있습니다. FFmpeg 바이너리는 이 저장소에서 재배포하지 않습니다.

## 실행

**익숙한_표현_태깅_실행.cmd**를 더블클릭합니다.

1. 파일은 **음원 선택 → 분류 시작**으로 처리합니다.
2. 링크는 **YouTube 링크 → 링크로 분류**로 처리합니다.
3. 화면에서 후보를 보고 **결과 폴더 열기**로 저장 파일을 확인합니다.

명령줄에서도 실행할 수 있습니다.

```powershell
.\.venv\Scripts\python.exe tag_audio.py --profile familiar "D:\Music\song.flac"
.\.venv\Scripts\python.exe tag_audio.py --youtube "https://youtu.be/영상ID"
```

링크는 영상 한 곡·최대20분·다운로드100MB를 지원합니다. 재생목록만 있는 링크, 진행 중인 방송, 로그인이 필요한 영상은 지원하지 않습니다. 시작 시각 정보가 붙어 있어도 전체 오디오를 분석합니다. 브라우저 쿠키나 로그인 정보는 가져오지 않습니다. 임시 오디오는 처리 후 삭제하고 결과만 남깁니다. YouTube의 접근 제한이나 인터넷 상태에 따라 링크 처리가 실패할 수 있습니다.

## 결과와 설정

- `tags.json`: `labels`는 대표값 하나씩, `candidates`는 순위가 있는 검토 후보입니다.
- `태깅결과.md`: 감정·분위기 후보3개와 대표값을 보여줍니다.
- `classification.json`: 전체 원시 분류 결과를 보존합니다.
- `status.json`: 성공/실패 수와 링크 임시 오디오 정리 여부입니다.

설정은 `config/model_paths.local.json`을 우선 읽고, 기존 로컬 배포 폴더가 있으면 이전 경로로도 동작합니다. 다른 설정은 `--settings "설정 경로"` 또는 `MUSIC_CLASSIFICATION_SETTINGS`로 선택합니다. `config/model_paths.example.json`의 경로는 설정 파일을 기준으로 해석합니다. 개인 설정과 결과는 Git에 올리지 않습니다.

기존 `음원_태깅_실행.cmd`와 `Music표현_분류_실행.cmd`는 이전69개 표현 구성입니다. 계층형 실행기는 이전44개 감정·분위기 표현을 정리해 보여주는 실험 구성이며, 익숙한 표현 구성과 목록이 다릅니다. `음악_분류_실행.cmd` 및 Music2Emo 관련 실험 API는 별도 백엔드 준비가 필요하고 위 MuQ 설치만으로 모든 실험 모델이 준비되지는 않습니다.

## 모델과 검증 범위

주 모델은 [OpenMuQ/MuQ-MuLan-large](https://huggingface.co/OpenMuQ/MuQ-MuLan-large)입니다. 고정 영어 설명과 오디오의 유사도를 비교하며, 추가 학습한 전용 감정 분류기는 아닙니다. 유사도는 정답 확률이 아닙니다. 현재 감정은 신남·슬픔·분노·불안·설렘·그리움·외로움·희망·사랑이고, 분위기는 잔잔·몽환·따뜻·시원·웅장·발랄·편안·어둠·낭만·Energetic·Chill입니다.

익숙한 곡의 사용 적합성을 확인한 개발용 구성입니다. 모든 노래나 청자에게 맞는 정확도를 보장하지 않습니다. 대표값을 검수 없이 확정 태그로 자동 적용하지 않고 후보를 검토하는 용도로 시작하세요.

모델 가중치의 라이선스는 CC-BY-NC4.0이고 MuQ 소스는 MIT입니다. 소스 프로그램과 모델 가중치의 사용 조건은 구분해야 합니다. 설치기는 원본 출처의 라이선스 문서도 받아 보존합니다.

## 개발 검사

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[desktop-test]"
.\.venv\Scripts\python.exe scripts/test_desktop.py
```

이 검사는 모델·음원을 다운로드하지 않습니다. 숨긴 실제 Tk 창과 이벤트 루프를 쓰되 모델/네트워크 결과는 모의 자료로 확인합니다. 로컬 연구 자료가 있어야 수행할 수 있는 두 감사는 공개 배포에서 건너뜁니다. 실제 모델 실행 확인은 별도 로컬 검증이며 CI 통과를 정확도 증명으로 해석하지 않습니다.
