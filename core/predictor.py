"""
음악 분류 예측 모듈
"""

import numpy as np
import random
from .feature_extractor import extract_audio_features


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


def predict_music_wav2vec2(model, processor, audio_path, confidence_threshold=0.5):
    """Wav2Vec2 모델을 사용하여 음악 분류"""
    try:
        import torch
        import librosa
        
        # 오디오 파일 로드 및 전처리
        y, sr = librosa.load(audio_path, sr=16000)  # Wav2Vec2는 16kHz 사용
        
        # 모델 입력 형식으로 변환
        inputs = processor(y, sampling_rate=16000, return_tensors="pt", padding=True)
        
        # 예측 수행
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][predicted_class].item()
        
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
            'success': True
        }
        
        if confidence < confidence_threshold:
            result['note'] += f' (낮은 신뢰도: {confidence:.2f})'
        
        return result
        
    except Exception as e:
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