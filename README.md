# 🎵 음악 분류 시스템

AI를 활용한 음악 장르 및 감정 분류 시스템입니다. Wav2Vec2 모델을 기반으로 한 고성능 음악 분류와 웹 인터페이스를 제공합니다.

## 📁 프로젝트 구조

```
Music Classification/
├── core/                    # 핵심 분류 기능
│   ├── __init__.py
│   ├── feature_extractor.py # 오디오 특성 추출 (librosa 기반)
│   ├── predictor.py         # 예측 기능 (Wav2Vec2, CNN, ML)
│   └── classifier.py        # 분류기 (규칙 기반, 수동, 하이브리드)
├── models/                  # 모델 관련 기능
│   ├── __init__.py
│   ├── cnn_model.py        # CNN 모델
│   ├── ml_model.py         # 전통적 ML 모델 (SVM, 로지스틱 회귀)
│   ├── wav2vec2_model.py   # Wav2Vec2 모델 (Hugging Face)
│   └── models--dima806--music_genres_classification/  # 사전 훈련된 모델
├── utils/                   # 유틸리티 함수들
│   ├── __init__.py
│   ├── downloader.py       # 다운로드 기능 (YouTube, 직접 링크)
│   ├── validator.py        # 검증 기능 (URL, 파일)
│   ├── monitor.py          # 모니터링 (성능, 메모리)
│   ├── cache.py           # 캐시 기능 (특성 추출 결과)
│   ├── memory_optimizer.py # 메모리 최적화
│   └── audio_processor.py  # 오디오 처리
├── webapp/                  # 웹 애플리케이션
│   ├── __init__.py
│   ├── app.py             # Flask 앱 (RESTful API)
│   ├── static/            # 정적 파일 (CSS, JS)
│   └── templates/         # HTML 템플릿
├── data/                   # 데이터 처리
│   ├── __init__.py
│   ├── processor.py       # 데이터 처리
│   └── reporter.py        # 보고서 생성 (JSON, 통계)
├── scripts/                # 설치 및 설정 스크립트
│   ├── install_packages.py
│   └── setup_optimization.py
├── test/                   # 테스트 파일
│   └── integrated_test.py
├── uploads/               # 업로드된 파일들 (임시 저장)
├── ffmpeg-master-latest-win64-gpl/  # FFmpeg 바이너리
├── main.py               # 메인 실행 파일
└── requirements.txt      # 의존성 패키지
```

## 🚀 설치 및 실행

### 0. 시스템 요구사항

- **Python**: 3.8 이상
- **메모리**: 최소 4GB RAM (8GB 권장)
- **GPU**: CUDA 지원 GPU (선택사항, CPU도 가능)
- **디스크**: 최소 2GB 여유 공간
- **OS**: Windows, macOS, Linux 지원

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

**주요 의존성 패키지:**
- **오디오 처리**: librosa, soundfile, ffmpeg-python
- **머신러닝**: tensorflow, scikit-learn, torch, transformers
- **웹 프레임워크**: Flask
- **다운로드**: yt-dlp, requests
- **시스템 모니터링**: psutil
- **시각화**: matplotlib

### 2. 실행

```bash
# 메인 실행 파일 사용 (권장)
python main.py

# 또는 웹앱 앱 파일 직접 실행
python webapp/app.py
```

## 🔧 주요 기능

### 1. 핵심 분류 기능 (`core/`)
- **특성 추출**: librosa 기반 오디오 특성 추출 (멜 스펙트로그램, MFCC, 템포, 크로마그램)
- **Wav2Vec2 예측**: Hugging Face 사전 훈련 모델을 사용한 고성능 분류
- **규칙 기반 분류**: 템포 기반 간단한 분류 시스템
- **하이브리드 분류**: 여러 방법을 결합한 통합 분류

### 2. 모델 관리 (`models/`)
- **Wav2Vec2**: Hugging Face 기반 고성능 음악 장르 분류 모델
- **CNN 모델**: 딥러닝 기반 음악 분류 (TensorFlow/Keras)
- **전통적 ML**: SVM, 로지스틱 회귀 등 scikit-learn 기반 모델
- **메모리 최적화**: 대용량 모델의 효율적인 메모리 관리

### 3. 유틸리티 (`utils/`)
- **다운로더**: YouTube, 직접 링크 다운로드 (yt-dlp, FFmpeg)
- **URL 검증**: 다양한 플랫폼 지원 (YouTube, Spotify, 직접 링크)
- **메모리 모니터링**: 실시간 CPU/GPU 메모리 사용량 추적
- **캐시 시스템**: 특성 추출 결과 캐싱으로 성능 향상
- **오디오 처리**: 파일 형식 검증, 크기 제한, 전처리

### 4. 웹 인터페이스 (`webapp/`)
- **Flask 앱**: 웹 기반 음악 분류 인터페이스
- **RESTful API**: JSON 기반 API 엔드포인트
- **실시간 분류**: 파일 업로드 및 URL 분류
- **진행률 추적**: Server-Sent Events를 통한 실시간 진행률 표시
- **일괄 처리**: 여러 URL 동시 분류 지원
- **반응형 UI**: 모바일/데스크톱 최적화된 인터페이스

### 5. 데이터 처리 (`data/`)
- **보고서 생성**: 분류 결과 통계 및 시각화 (JSON)
- **결과 저장**: 구조화된 JSON 형태로 결과 저장
- **통계 분석**: 장르/감정 분포, 신뢰도 통계 제공

## 📊 지원하는 분류 방법

1. **Wav2Vec2 (AI)**: Hugging Face 사전 훈련 모델 - 가장 정확한 분류
2. **규칙 기반**: 템포 기반 간단한 분류 (빠른 처리)
3. **수동 분류**: 사용자가 직접 장르/감정 입력
4. **전통적 ML**: SVM, 로지스틱 회귀 등 scikit-learn 모델
5. **하이브리드**: 여러 방법을 결합한 통합 분류

## 🎯 지원하는 장르 (Wav2Vec2 모델 기준)

- **블루스** (blues)
- **클래식** (classical) 
- **컨트리** (country)
- **디스코** (disco)
- **힙합** (hiphop)
- **재즈** (jazz)
- **메탈** (metal)
- **팝** (pop)
- **레게** (reggae)
- **록** (rock)

## 😊 지원하는 감정 (규칙 기반)

- **행복한**: 빠른 템포 (120+ BPM)
- **슬픈**: 느린 템포 (80-100 BPM)
- **평화로운**: 매우 느린 템포 (80- BPM)
- **열정적인**: 매우 빠른 템포 (140+ BPM)

## 🔗 URL 지원

- **YouTube**: youtube.com, youtu.be 링크
- **직접 링크**: MP3, WAV, M4A, FLAC 파일
- **일괄 처리**: 여러 URL 동시 분류
- **링크 미리보기**: URL 정보 미리보기 기능

## 📈 성능 모니터링

- **실시간 메모리 추적**: CPU/GPU 메모리 사용량 모니터링
- **진행률 표시**: Server-Sent Events 기반 실시간 진행률
- **캐시 시스템**: 동일 파일 재분석 시 성능 향상
- **메모리 최적화**: 대용량 모델 효율적 관리
- **결과 저장**: JSON 형태로 분류 결과 저장

## 🌐 API 엔드포인트

### 웹 인터페이스
- `GET /`: 메인 페이지
- `POST /classify`: 파일 업로드 분류
- `POST /classify_url`: URL 분류
- `POST /batch_classify`: 일괄 URL 분류

### RESTful API
- `GET /api/model_status`: 모델 상태 조회
- `POST /api/validate_url`: URL 검증
- `POST /api/link_preview`: 링크 미리보기
- `GET /api/progress/<task_id>`: 분류 진행률 조회
- `GET /api/progress_stream/<task_id>`: 실시간 진행률 스트리밍
- `POST /clear_uploads`: 업로드 폴더 정리
- `GET /get_uploads_info`: 업로드 폴더 정보

### 사용 예시

#### 파일 업로드 분류
```bash
curl -X POST -F "audio_file=@music.mp3" -F "action=wav2vec2" http://localhost:5000/classify
```

#### URL 분류
```bash
curl -X POST -d "url=https://youtube.com/watch?v=example&action=wav2vec2" http://localhost:5000/classify_url
```

#### 일괄 분류
```bash
curl -X POST -d "urls=https://youtube.com/watch?v=1
https://youtube.com/watch?v=2&action=wav2vec2" http://localhost:5000/batch_classify
```

## 🛠️ 개발자 정보

Github - [imsang27](https://github.com/imsang27)

### 모듈 추가 방법

1. 해당 디렉토리에 새 파일 생성
2. `__init__.py`에 함수/클래스 추가
3. 메인 파일에서 import

### 테스트

```bash
# 통합 테스트 실행
python test/integrated_test.py

# 특정 모듈 테스트
python -c "from core import *; print('Core module loaded successfully')"

# 웹앱 테스트
python webapp/app.py
```

## 📝 라이선스

MIT License

## 🤝 기여

### Issue 제보

버그 리포트, 기능 요청, 질문 등이 있으시면 새로운 [Issues](https://github.com/imsang27/music-classification/issues)를 생성해주세요.

**Issue 작성 시 포함해주세요:**
- 제목: 간결하고 명확한 제목
- 설명: 문제나 요청사항에 대한 자세한 설명
- 환경: OS, Python 버전, 라이브러리 버전 등
- 재현 방법: 버그의 경우 재현 단계
- 예상 결과: 기대하는 동작이나 결과

### Pull Request

코드 기여를 원하시면 다음 단계를 따라주세요:

1. [Fork](https://github.com/imsang27/Music-Classification/fork) the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Create a [Pull Request](https://github.com/imsang27/Music-Classification/pulls)

**PR 작성 시 포함해주세요:**
- 제목: 변경사항을 요약한 제목
- 설명: 변경사항에 대한 자세한 설명
- 테스트: 변경사항에 대한 테스트 방법
- 관련 Issue: 연결된 Issue 번호 (있다면)
