"""
캐시 관련 유틸리티
"""

import os
import hashlib
import numpy as np
from core.feature_extractor import extract_audio_features


class FeatureCache:
    def __init__(self, cache_dir='./feature_cache'):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def get_features(self, audio_path):
        """캐시된 특성을 가져오거나 새로 계산"""
        cache_path = os.path.join(self.cache_dir, 
                                f"{hashlib.md5(audio_path.encode()).hexdigest()}.npy")
        
        if os.path.exists(cache_path):
            return np.load(cache_path, allow_pickle=True).item()
        
        features = extract_audio_features(audio_path)
        np.save(cache_path, features)
        return features
    
    def clear_cache(self):
        """캐시 정리"""
        import shutil
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)
            os.makedirs(self.cache_dir, exist_ok=True)
    
    def get_cache_size(self):
        """캐시 크기 반환 (MB)"""
        total_size = 0
        if os.path.exists(self.cache_dir):
            for dirpath, dirnames, filenames in os.walk(self.cache_dir):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    total_size += os.path.getsize(filepath)
        return total_size / (1024 * 1024)  # MB로 변환 