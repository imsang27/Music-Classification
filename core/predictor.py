"""
음악 분류 예측 모듈
"""

import numpy as np
import random
import os
import sys
from .feature_extractor import extract_audio_features

# 순환 import 방지를 위해 함수 내에서 import
def _get_download_functions():
    """다운로드 함수들을 지연 로드"""
    try:
        from utils.downloader import download_youtube_audio, download_direct_audio
        return download_youtube_audio, download_direct_audio
    except ImportError:
        # 상대 경로에서 실행될 때를 위한 대안
        sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
        from utils.downloader import download_youtube_audio, download_direct_audio
        return download_youtube_audio, download_direct_audio


def predict_music(model, audio_path, genres, emotions, confidence_threshold=0.5):
    """음악 분류 예측 함수"""
    features = extract_audio_features(audio_path)
    input_data = features['mel_spectrogram'].reshape(1, *features['mel_spectrogram'].shape, 1)
    genre_pred, emotion_pred = model.predict(input_data)
    
    # 신뢰도가 높은 예측만 반환
    genre_results = [
        (genres[i], float(prob)) 
        for i, prob in enumerate(genre_pred[0]) 
        if prob >= confidence_threshold
    ]
    emotion_results = [
        (emotions[i], float(prob)) 
        for i, prob in enumerate(emotion_pred[0]) 
        if prob >= confidence_threshold
    ]
    
    return {
        'genres': sorted(genre_results, key=lambda x: x[1], reverse=True),
        'emotions': sorted(emotion_results, key=lambda x: x[1], reverse=True)
    }


def predict_music_wav2vec2(model, processor, audio_path, confidence_threshold=0.5, max_duration=30):
    """Wav2Vec2 모델을 사용하여 음악 분류 (메모리 최적화 버전)"""
    try:
        import torch
        import librosa
        from utils.memory_optimizer import monitor_memory_usage, optimize_memory_usage
        
        # 메모리 사용량 모니터링
        initial_memory = monitor_memory_usage()
        print(f"초기 메모리 사용량: {initial_memory['rss']:.1f} MB")
        
        # 오디오 파일 로드 및 전처리 (메모리 효율적)
        y, sr = librosa.load(audio_path, sr=16000, duration=max_duration)
        
        # 오디오 길이 검증 및 조정
        min_length = 16000  # 최소 1초 (16kHz * 1초)
        max_length = 16000 * max_duration  # 최대 지정된 시간
        
        if len(y) < min_length:
            # 너무 짧은 경우 반복하여 최소 길이 확보
            repeat_times = (min_length // len(y)) + 1
            y = np.tile(y, repeat_times)[:min_length]
        elif len(y) > max_length:
            # 너무 긴 경우 자르기 (중간 부분 사용)
            start_idx = len(y) // 2 - max_length // 2
            y = y[start_idx:start_idx + max_length]
        
        # 오디오 정규화
        y = librosa.util.normalize(y)
        
        # 메모리 사용량 체크
        current_memory = monitor_memory_usage()
        print(f"오디오 로드 후 메모리: {current_memory['rss']:.1f} MB")
        
        # 모델 입력 형식으로 변환 (메모리 효율적 설정)
        try:
            inputs = processor(
                y, 
                sampling_rate=16000, 
                return_tensors="pt", 
                padding=True,
                truncation=True,
                max_length=max_length
            )
        except Exception as padding_error:
            # padding 오류 발생 시 더 보수적인 설정으로 재시도
            print(f"Padding 오류 발생, 재시도 중: {padding_error}")
            inputs = processor(
                y, 
                sampling_rate=16000, 
                return_tensors="pt", 
                padding=True,
                truncation=True,
                max_length=max_length // 2  # 절반으로 제한
            )
        
        # 예측 수행 (메모리 효율적)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][predicted_class].item()
            
            # 메모리 정리
            del outputs, logits, probabilities
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # 최종 메모리 사용량 체크
        final_memory = monitor_memory_usage()
        memory_diff = final_memory['rss'] - initial_memory['rss']
        print(f"예측 완료 후 메모리: {final_memory['rss']:.1f} MB (변화: {memory_diff:+.1f} MB)")
        
        # 결과 매핑
        genre = model.config.id2label[predicted_class]
        
        # 한국어 장르명 매핑
        genre_mapping = {
            'blues': '블루스',
            'classical': '클래식',
            'country': '컨트리',
            'disco': '디스코',
            'hiphop': '힙합',
            'jazz': '재즈',
            'metal': '메탈',
            'pop': '팝',
            'reggae': '레게',
            'rock': '록'
        }
        
        korean_genre = genre_mapping.get(genre, genre)
        
        result = {
            'genre': korean_genre,
            'confidence': confidence,
            'method': 'Wav2Vec2 (Hugging Face)',
            'note': f'원본 장르: {genre}',
            'success': True,
            'memory_usage': {
                'initial': initial_memory['rss'],
                'final': final_memory['rss'],
                'difference': memory_diff
            }
        }
        
        if confidence < confidence_threshold:
            result['note'] += f' (낮은 신뢰도: {confidence:.2f})'
        
        return result
        
    except Exception as e:
        # 오류 발생 시에도 메모리 정리
        optimize_memory_usage()
        return {
            'error': f'Wav2Vec2 분류 중 오류 발생: {str(e)}',
            'success': False
        }


def dummy_classification(url, genres=None, emotions=None):
    """Wav2Vec2 모델이 없을 때 사용하는 더미 분류"""
    # 기본 장르와 감정 설정
    if genres is None:
        genres = ['블루스', '클래식', '컨트리', '디스코', '힙합', '재즈', '메탈', '팝', '레게', '록']
    if emotions is None:
        emotions = ['행복한', '슬픈', '평화로운', '열정적인']
    
    # URL에서 힌트 추출
    url_lower = url.lower()
    
    # URL 기반 간단한 규칙
    if any(word in url_lower for word in ['classical', 'classic', 'orchestra', 'symphony']):
        genre = '클래식'
    elif any(word in url_lower for word in ['jazz', 'smooth']):
        genre = '재즈'
    elif any(word in url_lower for word in ['rock', 'guitar', 'band']):
        genre = '록'
    elif any(word in url_lower for word in ['pop', 'popular']):
        genre = '팝'
    elif any(word in url_lower for word in ['hip', 'rap', 'urban']):
        genre = '힙합'
    elif any(word in url_lower for word in ['metal', 'heavy']):
        genre = '메탈'
    elif any(word in url_lower for word in ['country', 'folk']):
        genre = '컨트리'
    elif any(word in url_lower for word in ['blues', 'soul']):
        genre = '블루스'
    elif any(word in url_lower for word in ['reggae', 'caribbean']):
        genre = '레게'
    elif any(word in url_lower for word in ['disco', 'dance']):
        genre = '디스코'
    else:
        genre = random.choice(genres)
    
    # 감정도 추측
    if any(word in url_lower for word in ['happy', 'joy', 'upbeat']):
        emotion = '행복한'
    elif any(word in url_lower for word in ['sad', 'melancholy', 'slow']):
        emotion = '슬픈'
    elif any(word in url_lower for word in ['peaceful', 'calm', 'relaxing']):
        emotion = '평화로운'
    elif any(word in url_lower for word in ['passionate', 'energetic', 'fast']):
        emotion = '열정적인'
    else:
        emotion = random.choice(emotions)
    
    genre_confidence = random.uniform(0.3, 0.7)  # 낮은 신뢰도
    emotion_confidence = random.uniform(0.3, 0.7)  # 낮은 신뢰도
    
    # 새로운 형식 (Wav2Vec2 스타일)
    result = {
        'genre': genre,
        'confidence': genre_confidence,
        'method': '더미 분류 (Wav2Vec2 모델 없음)',
        'note': '실제 AI 모델이 로드되지 않아 더미 결과를 반환합니다.',
        'url': url,
        'success': True
    }
    
    # 기존 형식도 지원 (하위 호환성)
    result['genres'] = [(genre, genre_confidence)]
    result['emotions'] = [(emotion, emotion_confidence)]
    
    return result


def analyze_prediction(model, audio_path, genres, emotions):
    """예측 결과 분석"""
    features = extract_audio_features(audio_path)
    predictions = model.predict(features)
    
    # 특성 중요도 분석
    feature_importance = {
        'mel_spectrogram': np.mean(features['mel_spectrogram']),
        'tempo': features['tempo'],
        'chromagram': np.mean(features['chromagram']),
        'mfcc': np.mean(features['mfcc'])
    }
    
    return {
        'predictions': predictions,
        'feature_importance': feature_importance
    } 


def classify_music_from_url_wav2vec2(model, processor, url, confidence_threshold=0.5):
    """
    URL에서 음악을 다운로드하고 Wav2Vec2 모델로 분류
    """
    try:
        # 다운로드 함수들을 지연 로드
        download_youtube_audio, download_direct_audio = _get_download_functions()
        
        # URL에서 오디오 다운로드
        temp_file = None
        try:
            if 'youtube.com' in url or 'youtu.be' in url:
                temp_file = download_youtube_audio(url)
            else:
                temp_file = download_direct_audio(url)
            
            if not temp_file or not os.path.exists(temp_file):
                return {'error': '오디오 파일 다운로드 실패', 'success': False}
            
            # Wav2Vec2 모델로 분류
            result = predict_music_wav2vec2(model, processor, temp_file, confidence_threshold)
            result['url'] = url
            
            return result
            
        finally:
            # 임시 파일 정리
            if temp_file and os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass
                    
    except Exception as e:
        return {
            'error': f'URL 분류 중 오류 발생: {str(e)}',
            'url': url,
            'success': False
        }

def batch_classify_urls_wav2vec2(model, processor, urls, confidence_threshold=0.5):
    """
    여러 URL을 Wav2Vec2 모델로 일괄 분류 (메모리 최적화 버전)
    """
    import psutil
    import gc
    
    results = []
    process = psutil.Process()
    
    print(f"배치 처리 시작 - 총 {len(urls)}개 URL")
    print(f"초기 메모리 사용량: {process.memory_info().rss / 1024 / 1024:.1f} MB")
    
    for i, url in enumerate(urls):
        try:
            print(f"\n분류 중... ({i+1}/{len(urls)}): {url}")
            
            # 각 URL을 개별적으로 처리하여 메모리 문제 방지
            result = classify_music_from_url_wav2vec2(model, processor, url, confidence_threshold)
            result['url'] = url
            result['status'] = 'success' if result.get('success', False) else 'error'
            
            results.append(result)
            
            # 적극적인 메모리 정리
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except:
                pass
            
            # Python 가비지 컬렉션 강제 실행
            gc.collect()
            
            # 메모리 사용량 모니터링
            current_memory = process.memory_info().rss / 1024 / 1024
            print(f"현재 메모리 사용량: {current_memory:.1f} MB")
            
            # 메모리 사용량이 너무 높으면 경고
            if current_memory > 2000:  # 2GB 이상
                print("⚠️  메모리 사용량이 높습니다. 잠시 대기...")
                import time
                time.sleep(2)  # 2초 대기
                
        except Exception as e:
            print(f"URL 처리 중 오류 발생: {url} - {str(e)}")
            results.append({
                'url': url,
                'status': 'error',
                'error': str(e),
                'success': False
            })
    
    # 최종 메모리 정리
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except:
        pass
    
    final_memory = process.memory_info().rss / 1024 / 1024
    print(f"\n배치 처리 완료 - 최종 메모리: {final_memory:.1f} MB")
    
    return results 