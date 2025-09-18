#!/usr/bin/env python3
"""
오디오 처리 관련 함수들
"""

import os
import librosa
import numpy as np
import psutil
import gc
from .memory_optimizer import monitor_memory_usage

def extract_audio_features(audio_path, duration=60):
    """오디오 특성을 추출하는 함수"""
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

def preprocess_audio(audio_path, sr=22050, duration=60, memory_optimized=True):
    """오디오 전처리 (메모리 최적화 옵션 포함)"""
    try:
        if memory_optimized:
            # 메모리 사용량 모니터링
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024
            
            # 오디오 로드 (메모리 효율적)
            y, sr = librosa.load(audio_path, sr=sr, duration=duration)
            
            # 오디오 길이 검증
            if len(y) == 0:
                raise ValueError("오디오 파일이 비어있습니다")
            
            # 메모리 사용량 체크
            current_memory = process.memory_info().rss / 1024 / 1024
            memory_diff = current_memory - initial_memory
            
            if memory_diff > 100:  # 100MB 이상 증가하면 경고
                print(f"⚠️  오디오 로드 메모리 사용량 높음: {memory_diff:.1f} MB")
        else:
            # 기본 전처리
            y, sr = librosa.load(audio_path, sr=sr, duration=duration)
            if len(y) == 0:
                raise ValueError("오디오 파일이 비어있습니다")
        
        # 노이즈 제거 (메모리 효율적)
        try:
            y = librosa.effects.preemphasis(y)
        except:
            # preemphasis 실패 시 스킵
            pass
        
        # 무음 구간 제거 (메모리 효율적)
        try:
            y, _ = librosa.effects.trim(y, top_db=20)
        except:
            # trim 실패 시 스킵
            pass
        
        # 볼륨 정규화
        y = librosa.util.normalize(y)
        
        if memory_optimized:
            # 최종 메모리 사용량 체크
            final_memory = process.memory_info().rss / 1024 / 1024
            total_memory_diff = final_memory - initial_memory
            
            if total_memory_diff > 200:  # 200MB 이상 증가하면 경고
                print(f"⚠️  오디오 전처리 메모리 사용량 높음: {total_memory_diff:.1f} MB")
        
        return y, sr
    except Exception as e:
        print(f"전처리 중 오류 발생: {str(e)}")
        raise

def optimize_audio_processing(audio_path, max_duration=30, target_sr=16000):
    """오디오 처리 최적화 (메모리 효율적)"""
    try:
        # 메모리 사용량 모니터링
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024
        
        # 오디오 로드 (길이 제한)
        y, sr = librosa.load(audio_path, sr=target_sr, duration=max_duration)
        
        # 오디오 길이 검증 및 조정
        min_length = target_sr  # 최소 1초
        max_length = target_sr * max_duration  # 최대 지정된 시간
        
        if len(y) < min_length:
            # 너무 짧은 경우 반복
            repeat_times = (min_length // len(y)) + 1
            y = np.tile(y, repeat_times)[:min_length]
        elif len(y) > max_length:
            # 너무 긴 경우 자르기
            start_idx = len(y) // 2 - max_length // 2
            y = y[start_idx:start_idx + max_length]
        
        # 오디오 정규화
        y = librosa.util.normalize(y)
        
        # 메모리 정리
        gc.collect()
        
        # 최종 메모리 사용량 체크
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_diff = final_memory - initial_memory
        
        if memory_diff > 50:  # 50MB 이상 증가하면 경고
            print(f"⚠️  오디오 처리 메모리 사용량 높음: {memory_diff:.1f} MB")
        
        return y, sr
        
    except Exception as e:
        print(f"오디오 처리 최적화 중 오류 발생: {str(e)}")
        raise

def validate_audio_file(audio_path, max_size_mb=100):
    """오디오 파일 검증 (크기 제한 옵션 포함)"""
    try:
        # 파일 존재 여부 확인
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {audio_path}")
            
        # 파일 크기 검사
        file_size = os.path.getsize(audio_path)
        if file_size == 0:
            raise ValueError("빈 파일입니다")
        if file_size > max_size_mb * 1024 * 1024:  # 지정된 크기 제한
            raise ValueError(f"파일이 너무 큽니다 (최대 {max_size_mb}MB)")
            
        # 파일 형식 검사
        valid_extensions = ['.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac']
        if not any(audio_path.lower().endswith(ext) for ext in valid_extensions):
            raise ValueError("지원하지 않는 파일 형식입니다")
            
        return True
    except Exception as e:
        print(f"파일 검증 실패: {str(e)}")
        return False

def get_audio_info(audio_path, duration=10):
    """오디오 파일 정보 추출 (지속 시간 옵션 포함)"""
    try:
        y, sr = librosa.load(audio_path, duration=duration)  # 지정된 시간만 로드
        
        info = {
            'sample_rate': sr,
            'duration': len(y) / sr,
            'channels': 1 if y.ndim == 1 else y.shape[0],
            'file_size': os.path.getsize(audio_path),
            'format': os.path.splitext(audio_path)[1].lower(),
            'file_path': audio_path
        }
        
        return info
    except Exception as e:
        print(f"오디오 정보 추출 실패: {str(e)}")
        return None

def normalize_audio_length(y, sr, target_duration=30):
    """오디오 길이 정규화 (지정된 지속 시간으로 조정)"""
    target_length = int(target_duration * sr)
    
    if len(y) < target_length:
        # 너무 짧은 경우 반복
        repeat_times = (target_length // len(y)) + 1
        y = np.tile(y, repeat_times)[:target_length]
    elif len(y) > target_length:
        # 너무 긴 경우 중간 부분 자르기
        start_idx = len(y) // 2 - target_length // 2
        y = y[start_idx:start_idx + target_length]
    
    return y


def batch_process_audio(audio_paths, max_duration=30, target_sr=16000):
    """여러 오디오 파일을 배치로 처리"""
    results = []
    
    for i, audio_path in enumerate(audio_paths):
        try:
            print(f"처리 중... ({i+1}/{len(audio_paths)}): {os.path.basename(audio_path)}")
            
            # 파일 검증
            if not validate_audio_file(audio_path):
                results.append({
                    'path': audio_path,
                    'success': False,
                    'error': '파일 검증 실패'
                })
                continue
            
            # 오디오 처리
            y, sr = optimize_audio_processing(audio_path, max_duration, target_sr)
            
            # 특성 추출
            features = extract_audio_features(audio_path, max_duration)
            
            results.append({
                'path': audio_path,
                'success': True,
                'audio_data': y,
                'sample_rate': sr,
                'features': features
            })
            
        except Exception as e:
            results.append({
                'path': audio_path,
                'success': False,
                'error': str(e)
            })
    
    return results
