#!/usr/bin/env python3
"""
필요한 패키지들을 설치하는 스크립트
"""

import subprocess
import sys

def install_package(package):
    """개별 패키지를 설치합니다."""
    print(f"📦 {package} 설치 중...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ {package} 설치 완료")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {package} 설치 실패: {e}")
        return False

def main():
    """메인 실행 함수"""
    print("🚀 Wav2Vec2 최적화를 위한 패키지 설치")
    print("=" * 40)
    
    # 필수 패키지 목록
    packages = [
        "numpy",
        "librosa", 
        "transformers",
        "torch",
        "tensorflow",
        "scikit-learn",
        "psutil",
        "soundfile"
    ]
    
    success_count = 0
    total_count = len(packages)
    
    for package in packages:
        if install_package(package):
            success_count += 1
    
    print("\n" + "=" * 40)
    print(f"📊 설치 결과: {success_count}/{total_count} 성공")
    
    if success_count == total_count:
        print("🎉 모든 패키지 설치 완료!")
        print("이제 Wav2Vec2 모델을 사용할 수 있습니다.")
    else:
        print("⚠️ 일부 패키지 설치 실패")
        print("실패한 패키지를 수동으로 설치해주세요.")

if __name__ == "__main__":
    main() 