"""
모니터링 관련 유틸리티
"""

import psutil
import time
import json
import os


def monitor_memory_usage():
    """메모리 사용량 모니터링"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    return {
        'rss': memory_info.rss / 1024 / 1024,  # MB
        'vms': memory_info.vms / 1024 / 1024,  # MB
        'percent': process.memory_percent()
    }


class PerformanceMonitor:
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.history = []
        
    def on_epoch_end(self, epoch, logs={}):
        """에포크 종료 시 성능 모니터링"""
        metrics = {
            'epoch': epoch,
            'genre_acc': logs.get('val_genre_output_accuracy', 0),
            'emotion_acc': logs.get('val_emotion_output_accuracy', 0),
            'timestamp': time.time()
        }
        
        self.history.append(metrics)
        
        # 성능 경고
        if metrics['genre_acc'] < self.threshold:
            print(f"경고: 낮은 장르 분류 정확도 ({metrics['genre_acc']:.2f})")
        if metrics['emotion_acc'] < self.threshold:
            print(f"경고: 낮은 감정 분류 정확도 ({metrics['emotion_acc']:.2f})")
        
        # 학습 진행 상황 저장
        self.save_history()
    
    def save_history(self):
        """학습 히스토리 저장"""
        with open('training_history.json', 'w') as f:
            json.dump(self.history, f, indent=2)


class ModelVersionControl:
    def __init__(self, version_dir='./model_versions'):
        self.version_dir = version_dir
        os.makedirs(version_dir, exist_ok=True)
        
    def save_version(self, model, version_info):
        """모델 버전 저장"""
        version_path = os.path.join(
            self.version_dir, 
            f"v{version_info['version']}_{time.strftime('%Y%m%d')}"
        )
        os.makedirs(version_path, exist_ok=True)
        
        # 모델 저장
        model.save(os.path.join(version_path, 'model.h5'))
        
        # 버전 정보 저장
        version_info.update({
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'model_hash': self._get_model_hash(model)
        })
        
        with open(os.path.join(version_path, 'version_info.json'), 'w') as f:
            json.dump(version_info, f, indent=2)
    
    def _get_model_hash(self, model):
        """모델 해시 계산"""
        import hashlib
        return hashlib.md5(str(model.get_weights()).encode()).hexdigest()


def run_model_tests(model, test_data_dir):
    """모델 테스트 실행"""
    test_results = {
        'passed': [],
        'failed': []
    }
    
    try:
        # 기본 예측 테스트
        test_file = os.path.join(test_data_dir, 'test_sample.mp3')
        if os.path.exists(test_file):
            from core.predictor import predict_music
            prediction = predict_music(model, test_file, ['클래식', '재즈', '록', '팝'], ['행복한', '슬픈', '평화로운', '열정적인'])
            test_results['passed'].append('기본 예측 테스트')
        
        # 메모리 누수 테스트
        initial_memory = monitor_memory_usage()
        for _ in range(10):
            if os.path.exists(test_file):
                predict_music(model, test_file, ['클래식', '재즈', '록', '팝'], ['행복한', '슬픈', '평화로운', '열정적인'])
        final_memory = monitor_memory_usage()
        
        if (final_memory['rss'] - initial_memory['rss']) < 100:  # 100MB 이하 증가 허용
            test_results['passed'].append('메모리 누수 테스트')
        else:
            test_results['failed'].append('메모리 누수 테스트')
            
    except Exception as e:
        test_results['failed'].append(f'테스트 실패: {str(e)}')
    
    return test_results 