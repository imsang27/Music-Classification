"""
검증 관련 유틸리티
"""

import os
import librosa
import shutil
from datetime import datetime, timedelta


def validate_dataset(data_path, genres, emotions):
    """데이터셋 유효성 검사"""
    errors = []
    for genre in genres:
        genre_path = os.path.join(data_path, genre)
        if not os.path.exists(genre_path):
            errors.append(f"장르 폴더를 찾을 수 없음: {genre}")
        else:
            files = os.listdir(genre_path)
            if len(files) < 10:  # 최소 파일 수 체크
                errors.append(f"장르 {genre}의 데이터가 부족함 (현재: {len(files)}개)")
    
    if errors:
        raise ValueError("\n".join(errors))


class DatasetValidator:
    def __init__(self, min_samples=50, min_duration=10):
        self.min_samples = min_samples
        self.min_duration = min_duration
        self.validation_results = {
            'passed': [],
            'warnings': [],
            'errors': []
        }
    
    def validate_dataset(self, data_path, genres, emotions):
        """데이터셋 검증"""
        try:
            # 데이터셋 구조 검증
            self._validate_directory_structure(data_path, genres)
            
            # 각 장르별 샘플 수 검증
            self._validate_sample_counts(data_path, genres)
            
            # 오디오 파일 품질 검증
            self._validate_audio_quality(data_path)
            
            # 클래스 밸런스 검증
            self._validate_class_balance(data_path, genres)
            
            return self.validation_results
            
        except Exception as e:
            self.validation_results['errors'].append(f"검증 중 오류 발생: {str(e)}")
            return self.validation_results
    
    def _validate_directory_structure(self, data_path, genres):
        """디렉토리 구조 검증"""
        for genre in genres:
            genre_path = os.path.join(data_path, genre)
            if not os.path.exists(genre_path):
                self.validation_results['errors'].append(f"장르 폴더 없음: {genre}")
            elif not os.path.isdir(genre_path):
                self.validation_results['errors'].append(f"잘못된 장르 폴더 형식: {genre}")
    
    def _validate_sample_counts(self, data_path, genres):
        """샘플 수 검증"""
        for genre in genres:
            genre_path = os.path.join(data_path, genre)
            if os.path.exists(genre_path):
                samples = len([
                    f for f in os.listdir(genre_path)
                    if f.endswith(('.mp3', '.wav', '.m4a', '.flac'))
                ])
                if samples < self.min_samples:
                    self.validation_results['warnings'].append(
                        f"장르 '{genre}'의 샘플 수가 부족합니다. (현재: {samples}, 필요: {self.min_samples})"
                    )
    
    def _validate_audio_quality(self, data_path):
        """오디오 품질 검증"""
        for root, _, files in os.walk(data_path):
            for file in files:
                if file.endswith(('.mp3', '.wav', '.m4a', '.flac')):
                    file_path = os.path.join(root, file)
                    try:
                        y, sr = librosa.load(file_path, duration=5)  # 첫 5초만 검사
                        if len(y) == 0:
                            self.validation_results['errors'].append(f"빈 오디오 파일: {file_path}")
                        if sr < 22050:
                            self.validation_results['warnings'].append(
                                f"낮은 샘플링 레이트: {file_path} ({sr}Hz)"
                            )
                    except Exception as e:
                        self.validation_results['errors'].append(
                            f"파일 로드 실패: {file_path} - {str(e)}"
                        )
    
    def _validate_class_balance(self, data_path, genres):
        """클래스 밸런스 검증"""
        sample_counts = {}
        for genre in genres:
            genre_path = os.path.join(data_path, genre)
            if os.path.exists(genre_path):
                sample_counts[genre] = len([f for f in os.listdir(genre_path) 
                                          if f.endswith(('.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac'))])
        
        if sample_counts:
            avg_samples = sum(sample_counts.values()) / len(sample_counts)
            for genre, count in sample_counts.items():
                if count < avg_samples * 0.5:
                    self.validation_results['warnings'].append(
                        f"장르 '{genre}'의 샘플 수가 평균({avg_samples:.0f})의 50% 미만입니다: {count}"
                    ) 

def cleanup_uploads_folder(max_age_hours=24):
    """
    uploads 폴더에서 오래된 파일들을 정리합니다
    
    Args:
        max_age_hours (int): 파일 보관 시간 (시간 단위, 기본값: 24시간)
    """
    uploads_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'uploads')
    
    if not os.path.exists(uploads_dir):
        return
    
    cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
    removed_count = 0
    
    try:
        for filename in os.listdir(uploads_dir):
            file_path = os.path.join(uploads_dir, filename)
            
            # 파일의 수정 시간 확인
            file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
            
            if file_mtime < cutoff_time:
                try:
                    os.remove(file_path)
                    removed_count += 1
                    print(f"오래된 파일 삭제: {filename}")
                except Exception as e:
                    print(f"파일 삭제 실패 {filename}: {str(e)}")
        
        if removed_count > 0:
            print(f"총 {removed_count}개의 오래된 파일을 정리했습니다.")
        else:
            print("정리할 오래된 파일이 없습니다.")
            
    except Exception as e:
        print(f"uploads 폴더 정리 중 오류 발생: {str(e)}")


def cleanup_all_uploads():
    """uploads 폴더의 모든 파일을 삭제합니다"""
    uploads_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'uploads')
    
    if not os.path.exists(uploads_dir):
        print("uploads 폴더가 존재하지 않습니다.")
        return
    
    try:
        shutil.rmtree(uploads_dir)
        print("uploads 폴더의 모든 파일을 삭제했습니다.")
    except Exception as e:
        print(f"uploads 폴더 삭제 중 오류 발생: {str(e)}")


def get_uploads_folder_size():
    """uploads 폴더의 크기를 반환합니다 (MB 단위)"""
    uploads_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'uploads')
    
    if not os.path.exists(uploads_dir):
        return 0
    
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(uploads_dir):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                total_size += os.path.getsize(file_path)
        
        return round(total_size / (1024 * 1024), 2)  # MB 단위로 변환
    except Exception as e:
        print(f"폴더 크기 계산 중 오류 발생: {str(e)}")
        return 0 
