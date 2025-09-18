"""
메모리 최적화 관련 유틸리티
"""

import gc
import psutil
import os


def monitor_memory_usage():
    """메모리 사용량 모니터링 (최적화된 버전)"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    # GPU 메모리도 확인
    gpu_memory = None
    try:
        import torch
        if torch.cuda.is_available():
            gpu_memory = {
                'allocated': torch.cuda.memory_allocated() / 1024 / 1024,  # MB
                'cached': torch.cuda.memory_reserved() / 1024 / 1024,  # MB
                'total': torch.cuda.get_device_properties(0).total_memory / 1024 / 1024  # MB
            }
    except:
        pass
    
    return {
        'rss': memory_info.rss / 1024 / 1024,  # MB
        'vms': memory_info.vms / 1024 / 1024,  # MB
        'percent': process.memory_percent(),
        'gpu': gpu_memory
    }


def optimize_memory_usage():
    """메모리 사용량 최적화"""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except:
        pass
    gc.collect()


def check_memory_health():
    """메모리 상태 건강성 검사"""
    memory_info = monitor_memory_usage()
    
    # CPU 메모리 검사
    cpu_memory_gb = memory_info['rss'] / 1024
    if cpu_memory_gb > 4:  # 4GB 이상
        print(f"⚠️  CPU 메모리 사용량이 높습니다: {cpu_memory_gb:.1f} GB")
        return False
    elif cpu_memory_gb > 2:  # 2GB 이상
        print(f"⚠️  CPU 메모리 사용량이 보통입니다: {cpu_memory_gb:.1f} GB")
    else:
        print(f"✅ CPU 메모리 사용량 정상: {cpu_memory_gb:.1f} GB")
    
    # GPU 메모리 검사
    if memory_info['gpu']:
        gpu_allocated_gb = memory_info['gpu']['allocated'] / 1024
        gpu_total_gb = memory_info['gpu']['total'] / 1024
        gpu_usage_percent = (memory_info['gpu']['allocated'] / memory_info['gpu']['total']) * 100
        
        if gpu_usage_percent > 80:
            print(f"⚠️  GPU 메모리 사용량이 높습니다: {gpu_allocated_gb:.1f}/{gpu_total_gb:.1f} GB ({gpu_usage_percent:.1f}%)")
            return False
        elif gpu_usage_percent > 50:
            print(f"⚠️  GPU 메모리 사용량이 보통입니다: {gpu_allocated_gb:.1f}/{gpu_total_gb:.1f} GB ({gpu_usage_percent:.1f}%)")
        else:
            print(f"✅ GPU 메모리 사용량 정상: {gpu_allocated_gb:.1f}/{gpu_total_gb:.1f} GB ({gpu_usage_percent:.1f}%)")
    
    return True


def get_memory_usage_info():
    """메모리 사용량 정보를 상세히 반환"""
    memory_info = monitor_memory_usage()
    
    info = {
        'cpu': {
            'rss_mb': memory_info['rss'],
            'vms_mb': memory_info['vms'],
            'percent': memory_info['percent']
        }
    }
    
    if memory_info['gpu']:
        info['gpu'] = memory_info['gpu']
    
    return info


def force_cleanup():
    """강제 메모리 정리"""
    print("🧹 강제 메모리 정리 중...")
    
    # Python 가비지 컬렉션
    collected = gc.collect()
    print(f"   Python 가비지 컬렉션: {collected}개 객체 정리")
    
    # PyTorch GPU 메모리 정리
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("   PyTorch GPU 캐시 정리 완료")
    except:
        pass
    
    # 메모리 사용량 확인
    final_memory = monitor_memory_usage()
    print(f"   정리 후 메모리: {final_memory['rss']:.1f} MB")
    
    return final_memory