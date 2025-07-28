# 🎵 음악 분류 시스템

AI를 활용한 음악 장르 및 감정 분류 시스템입니다.

## 📁 프로젝트 구조

```
Music Classification/
├── core/                    # 핵심 분류 기능
│   ├── __init__.py
│   ├── feature_extractor.py # 오디오 특성 추출
│   ├── predictor.py         # 예측 기능
│   └── classifier.py        # 분류기
├── models/                  # 모델 관련 기능
│   ├── __init__.py
│   ├── cnn_model.py        # CNN 모델
│   ├── ml_model.py         # 전통적 ML 모델
│   └── wav2vec2_model.py   # Wav2Vec2 모델
├── utils/                   # 유틸리티 함수들
│   ├── __init__.py
│   ├── downloader.py       # 다운로드 기능
│   ├── validator.py        # 검증 기능
│   ├── monitor.py          # 모니터링
│   └── cache.py           # 캐시 기능
├── webapp/                  # 웹 애플리케이션
│   ├── __init__.py
│   ├── app.py             # Flask 앱
│   ├── static/            # 정적 파일 (CSS, JS)
│   └── templates/         # HTML 템플릿
├── data/                   # 데이터 처리
│   ├── __init__.py
│   ├── processor.py       # 데이터 처리
│   └── reporter.py        # 보고서 생성
├── models/                 # 저장된 모델들
├── uploads/               # 업로드된 파일들
├── main.py               # 메인 실행 파일
├── Music-Classification.py # 기존 파일 (하위 호환성)
└── requirements.txt
```

## 🚀 설치 및 실행

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. 실행

```bash
# 메인 실행 파일 사용 (권장)
python main.py

# 또는 웹앱 앱 파일 직접 실행
python webapp/app.py
```

## 🔧 주요 기능

### 1. 핵심 분류 기능 (`core/`)
- **특성 추출**: 오디오 파일에서 멜 스펙트로그램, MFCC, 템포 등 추출
- **예측**: 다양한 모델을 사용한 음악 분류
- **분류기**: 규칙 기반, 수동, 하이브리드 분류

### 2. 모델 관리 (`models/`)
- **CNN 모델**: 딥러닝 기반 음악 분류
- **전통적 ML**: SVM, 로지스틱 회귀 등
- **Wav2Vec2**: Hugging Face 기반 고성능 모델

### 3. 유틸리티 (`utils/`)
- **다운로더**: YouTube, 직접 링크 다운로드
- **검증기**: URL, 데이터셋 검증
- **모니터링**: 성능, 메모리 사용량 모니터링
- **캐시**: 특성 추출 결과 캐싱

### 4. 웹 인터페이스 (`webapp/`)
- **Flask 앱**: 웹 기반 음악 분류 인터페이스
- **API**: RESTful API 제공
- **실시간 분류**: 파일 업로드 및 URL 분류
- **템플릿**: HTML, CSS, JavaScript 파일 포함

### 5. 데이터 처리 (`data/`)
- **보고서 생성**: 분류 결과 통계 및 시각화
- **결과 저장**: JSON 형태로 결과 저장

## 📊 지원하는 분류 방법

1. **Wav2Vec2 (AI)**: 가장 정확한 분류
2. **규칙 기반**: 템포 기반 간단한 분류
3. **수동 분류**: 사용자가 직접 입력
4. **전통적 ML**: SVM 등 기계학습 모델
5. **하이브리드**: 여러 방법 결합

## 🎯 지원하는 장르

- 클래식, 재즈, 록, 팝
- 블루스, 컨트리, 디스코, 힙합
- 메탈, 레게

## 😊 지원하는 감정

- 행복한, 슬픈, 평화로운, 열정적인
- 차분한, 긴장된, 기쁨, 슬픔

## 🔗 URL 지원

- YouTube 링크
- 직접 음악 파일 링크
- 일괄 처리 지원

## 📈 성능 모니터링

- 실시간 메모리 사용량 추적
- 분류 정확도 모니터링
- 학습 진행 상황 저장

## 🛠️ 개발자 정보

Github - [imsang27](https://github.com/imsang27)

### 모듈 추가 방법

1. 해당 디렉토리에 새 파일 생성
2. `__init__.py`에 함수/클래스 추가
3. 메인 파일에서 import

### 테스트

```bash
# 기본 테스트
python -m pytest tests/

# 특정 모듈 테스트
python -c "from core import *; print('Core module loaded successfully')"
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
