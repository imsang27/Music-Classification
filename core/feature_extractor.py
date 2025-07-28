"""
오디오 특성 추출 모듈
"""

import librosa
import numpy as np
import os


def extract_audio_features(audio_path, duration=60):
    """오디오 파일에서 특성을 추출합니다."""
    try:
        # 파일 존재 여부 확인
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {audio_path}")
            
        y, sr = librosa.load(audio_path, duration=duration)
        if len(y) == 0:
            raise ValueError("오디오 파일이 비어있습니다")
        
        # 샘플링 레이트 체크
        if sr != 22050:  # librosa 기본값
            y = librosa.resample(y, orig_sr=sr, target_sr=22050)
            sr = 22050
            
        # 멜 스펙트로그램
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
        mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
        mel_spectrogram = (mel_spectrogram - np.mean(mel_spectrogram)) / np.std(mel_spectrogram)
        
        # 템포
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        
        # 크로마그램 (음조 특성)
        chromagram = librosa.feature.chroma_stft(y=y, sr=sr)
        chromagram = (chromagram - np.mean(chromagram)) / np.std(chromagram)
        
        # MFCC (음색 특성)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)
        
        # 스펙트럴 대비 (음악의 다이나믹 특성)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        
        return {
            'mel_spectrogram': mel_spectrogram,
            'tempo': tempo,
            'chromagram': np.mean(chromagram, axis=1),
            'mfcc': np.mean(mfcc, axis=1),
            'spectral_contrast': np.mean(spectral_contrast, axis=1)
        }
    except Exception as e:
        print(f"오디오 파일 처리 중 오류 발생: {str(e)}")
        raise


def preprocess_audio(audio_path, sr=22050):
    """오디오 전처리 함수"""
    try:
        # 오디오 로드
        y, sr = librosa.load(audio_path, sr=sr)
        
        # 노이즈 제거
        y = librosa.effects.preemphasis(y)
        
        # 무음 구간 제거
        y, _ = librosa.effects.trim(y, top_db=20)
        
        # 볼륨 정규화
        y = librosa.util.normalize(y)
        
        return y, sr
    except Exception as e:
        print(f"전처리 중 오류 발생: {str(e)}")
        raise


def validate_audio_file(audio_path):
    """오디오 파일 유효성 검사"""
    try:
        # 파일 존재 여부 확인
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {audio_path}")
            
        # 파일 크기 검사
        file_size = os.path.getsize(audio_path)
        if file_size == 0:
            raise ValueError("빈 파일입니다")
        if file_size > 100 * 1024 * 1024:  # 100MB 제한
            raise ValueError("파일이 너무 큽니다")
            
        # 파일 형식 검사
        valid_extensions = ['.mp3', '.wav', '.m4a', '.flac']
        if not any(audio_path.lower().endswith(ext) for ext in valid_extensions):
            raise ValueError("지원하지 않는 파일 형식입니다")
            
        return True
    except Exception as e:
        print(f"파일 검증 실패: {str(e)}")
        return False 