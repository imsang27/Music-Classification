"""
핵심 음악 분류 기능 모듈
"""

from .classifier import *
from .feature_extractor import *

# predictor 모듈은 조건부로 import
try:
    from .predictor import *
except ImportError as e:
    print(f"Warning: predictor 모듈 import 실패: {e}")
    # 기본 함수들만 정의
    def predict_music_wav2vec2(*args, **kwargs):
        return {'error': 'predictor 모듈을 로드할 수 없습니다', 'success': False}
    
    def dummy_classification(*args, **kwargs):
        return {'error': 'predictor 모듈을 로드할 수 없습니다', 'success': False}

__all__ = [
    'extract_audio_features',
    'predict_music',
    'predict_music_wav2vec2',
    'classify_music_from_url_wav2vec2',
    'batch_classify_urls_wav2vec2',
    'rule_based_classification',
    'manual_classification',
    'hybrid_classification',
    'train_traditional_ml_model',
    'predict_traditional_ml_model',
    'train_lyrics_model',
    'predict_lyrics',
    'analyze_prediction',
    'dummy_classification'
] 