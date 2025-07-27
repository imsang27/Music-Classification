# 음악 분류 시스템 (Music Classification System)

딥러닝 기반의 음악 장르 및 감정 분류 시스템입니다. CNN을 활용하여 오디오 특성을 분석하고, 멀티태스크 학습을 통해 장르와 감정을 동시에 분류합니다.

## 주요 기능

### 1. 오디오 특성 추출
- 멜 스펙트로그램 분석
- 템포 검출
- 크로마그램 (음조 특성) 분석
- MFCC (음색 특성) 분석
- 스펙트럴 대비 분석

### 2. 장르 및 감정 분류
- 다중 장르 분류
- 음악 감정 분류
- 신뢰도 점수 제공

### 3. 링크 기반 분류 (NEW!)
- **YouTube 링크 지원**: yt-dlp를 사용한 오디오 다운로드 및 분류
- **직접 음악 파일 링크 지원**: HTTP/HTTPS를 통한 음악 파일 다운로드
- **일괄 분류 기능**: 여러 링크를 동시에 처리하고 결과 요약
- **실시간 URL 검증**: 입력된 링크의 유효성 검사
- **링크 미리보기 기능**: YouTube 정보 및 파일 정보 미리보기
- **Spotify 링크 검증**: 향후 지원 예정 (현재는 검증만 지원)

### 4. 데이터 관리
- 데이터셋 검증
- 오디오 파일 품질 검사
- 클래스 밸런스 모니터링

### 5. 모델 관리
- 자동 모델 백업
- 버전 관리
- 학습 히스토리 추적

### 6. 추가 분류 기능
- 수동 분류 인터페이스
- 규칙 기반 분류
- 전통적인 머신 러닝 모델
- 가사 분석
- 하이브리드 접근 방식

## 설치 방법

프로젝트 의존성은 `requirements.txt` 파일로 관리합니다.

```bash
pip install -r requirements.txt
```

## 사용 방법

### 1. 모델 학습
```python
# 장르와 감정 레이블 정의
genres = ['클래식', '재즈', '록', '팝']
emotions = ['행복한', '슬픈', '평화로운', '열정적인']

# 모델 학습
model = train_music_classifier('음악_데이터_경로', genres, emotions)
```

### 2. 음악 분류

#### 파일 기반 분류
```python
# 새로운 음악 파일 분류
result = predict_music(model, "새로운_음악.mp3", genres, emotions)
print("장르:", result['genres'])
print("감정:", result['emotions'])
```

#### 링크 기반 분류
```python
# YouTube 링크로 분류
youtube_url = "https://youtube.com/watch?v=..."
result = classify_music_from_url(model, youtube_url, genres, emotions)
print("장르:", result['genres'])
print("감정:", result['emotions'])
print("YouTube 정보:", result['youtube_info'])

# 여러 링크 일괄 분류
urls = [
    "https://youtube.com/watch?v=...",
    "https://example.com/music.mp3"
]
results = batch_classify_urls(model, urls, genres, emotions)
for result in results:
    print(f"URL: {result['url']}")
    print(f"장르: {result['genres']}")
    print(f"감정: {result['emotions']}")
```

### 3. 웹 인터페이스

간단한 웹 서버를 실행하여 브라우저에서 음악 파일을 업로드하고 분류할 수 있습니다.

```bash
python webapp/app.py
```

웹 페이지에서 다음 기능들을 사용할 수 있습니다:

#### 📁 파일로 분류하기
- 음악 파일 업로드
- AI 기반 분류
- 규칙 기반 분류
- 수동 분류

#### 🔗 링크로 분류하기
- YouTube 링크 입력
- 직접 음악 파일 링크 입력
- 실시간 URL 검증 및 링크 미리보기
- AI 기반 및 규칙 기반 분류

#### 📋 여러 링크 일괄 분류하기
- 여러 링크를 한 번에 입력
- 일괄 처리 및 결과 요약
- JSON 파일로 결과 저장

## 지원하는 링크 형식

### YouTube 링크
- `https://youtube.com/watch?v=VIDEO_ID`
- `https://youtu.be/VIDEO_ID`
- `https://youtube.com/embed/VIDEO_ID`

### 직접 음악 파일 링크
- `.mp3`, `.wav`, `.m4a`, `.flac`, `.ogg`, `.aac` 확장자 지원
- HTTP/HTTPS 프로토콜 지원

## 시스템 요구사항

- Python 3.8 이상
- CUDA 지원 GPU (권장)
- 최소 8GB RAM
- 충분한 저장 공간 (데이터셋 크기에 따라 다름)
- 인터넷 연결 (링크 분류 기능 사용 시)

## 주의사항

1. 입력 파일 형식
   - 지원 형식: .mp3, .wav, .m4a, .flac, .ogg, .aac
   - 최대 파일 크기: 100MB

2. 링크 분류 제한사항
   - YouTube 링크: yt-dlp 라이브러리 필요
   - 일괄 분류: 최대 20개 URL
   - 처리 시간: 링크 길이에 따라 다름

3. 메모리 사용
   - 대용량 데이터셋 처리 시 메모리 사용량 모니터링 필요
   - GPU 메모리 관리 설정 확인

4. 데이터셋 요구사항
   - 장르당 최소 50개 이상의 샘플
   - 균형잡힌 클래스 분포 권장
   - 고품질 오디오 파일 권장

## 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다.

## 기여 방법

1. 이 저장소를 포크합니다.
2. 새로운 브랜치를 생성합니다.
3. 변경사항을 커밋합니다.
4. 브랜치에 푸시합니다.
5. Pull Request를 생성합니다.

## 문의사항

문제가 발생하거나 제안사항이 있으시면 [Issues](https://github.com/imsang27/Music-Classification/issues)에 등록해 주세요.
