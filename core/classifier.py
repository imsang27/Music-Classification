"""
음악 분류기 모듈
"""

import numpy as np
import random
from .feature_extractor import extract_audio_features


def rule_based_classification(audio_path):
    """규칙 기반 분류 - 템포와 오디오 특성을 기반으로 분류"""
    try:
        features = extract_audio_features(audio_path)
        tempo = features['tempo']
        
        # 장르 분류 (템포 기반)
        if tempo < 80:
            genre = '클래식'
            genre_confidence = 0.8
        elif tempo < 110:
            genre = '재즈'
            genre_confidence = 0.7
        elif tempo < 140:
            genre = '팝'
            genre_confidence = 0.75
        else:
            genre = '록'
            genre_confidence = 0.8
        
        # 감정 분류 (템포 기반)
        if tempo < 80:
            emotion = '평화로운'
            emotion_confidence = 0.8
        elif tempo < 100:
            emotion = '슬픈'
            emotion_confidence = 0.7
        elif tempo < 120:
            emotion = '행복한'
            emotion_confidence = 0.75
        else:
            emotion = '열정적인'
            emotion_confidence = 0.8
        
        # 결과를 웹 인터페이스 형식에 맞게 반환
        return {
            'genres': [(genre, genre_confidence)],
            'emotions': [(emotion, emotion_confidence)],
            'tempo': tempo,
            'method': '규칙 기반 분류'
        }
        
    except Exception as e:
        # 오류 발생 시 기본값 반환
        return {
            'genres': [('팝', 0.5)],
            'emotions': [('행복한', 0.5)],
            'tempo': 120,
            'method': '규칙 기반 분류 (기본값)',
            'error': f'분류 중 오류 발생: {str(e)}'
        }


def manual_classification(audio_path, genres, emotions, input_func=input):
    """사용자가 직접 장르와 감정을 입력하여 분류합니다"""
    print(f"수동 분류 대상 파일: {audio_path}")
    genre = input_func(f"장르를 선택하세요 {genres}: ")
    emotion = input_func(f"감정을 선택하세요 {emotions}: ")
    return {'genre': genre, 'emotion': emotion}


def hybrid_classification(cnn_model, ml_model, vec, lyr_model, audio_path, lyrics,
                          genres, emotions):
    """딥러닝, 규칙 기반, 가사 분석을 결합한 하이브리드 분류"""
    from .predictor import predict_music
    
    cnn_result = predict_music(cnn_model, audio_path, genres, emotions)
    cnn_genre = cnn_result['genres'][0][0] if cnn_result['genres'] else None
    cnn_emotion = cnn_result['emotions'][0][0] if cnn_result['emotions'] else None

    rule_result = rule_based_classification(audio_path)
    ml_genre = predict_traditional_ml_model(ml_model, audio_path)

    votes_genre = [g for g in [cnn_genre, rule_result['genre'], ml_genre] if g]
    final_genre = max(set(votes_genre), key=votes_genre.count)

    emotion_votes = [cnn_emotion, rule_result['emotion']]
    if lyrics and vec and lyr_model:
        emotion_votes.append(predict_lyrics(vec, lyr_model, lyrics))
    emotion_votes = [e for e in emotion_votes if e]
    final_emotion = max(set(emotion_votes), key=emotion_votes.count)

    return {'genre': final_genre, 'emotion': final_emotion}


def predict_traditional_ml_model(model, audio_path):
    """전통적인 머신 러닝 모델 예측"""
    features = extract_audio_features(audio_path)
    X = np.concatenate([
        features['mfcc'],
        features['spectral_contrast'],
        [features['tempo']]
    ]).reshape(1, -1)
    return model.predict(X)[0]


def predict_lyrics(vectorizer, model, lyrics):
    """가사 분석 예측"""
    X = vectorizer.transform([lyrics])
    return model.predict(X)[0]

def train_traditional_ml_model(feature_list, labels):
    """전통적인 머신 러닝 모델(SVM) 학습"""
    from sklearn.svm import SVC
    X = [
        np.concatenate([
            f['mfcc'],
            f['spectral_contrast'],
            [f['tempo']]
        ])
        for f in feature_list
    ]
    model = SVC(probability=True)
    model.fit(X, labels)
    return model

def train_lyrics_model(lyrics_list, labels):
    """가사 분석을 위한 간단한 텍스트 분류 모델"""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(lyrics_list)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, labels)
    return vectorizer, clf 