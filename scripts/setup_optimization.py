#!/usr/bin/env python3
"""
Wav2Vec2 메모리 최적화 적용 스크립트
"""

import subprocess
import sys
import os

def install_packages():
    """필요한 패키지들을 설치합니다."""
    print("📦 필요한 패키지들을 설치합니다...")
    
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
    
    for package in packages:
        print(f"설치 중: {package}")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ {package} 설치 완료")
        except subprocess.CalledProcessError as e:
            print(f"❌ {package} 설치 실패: {e}")
            return False
    
    return True

def test_optimization():
    """메모리 최적화가 제대로 작동하는지 테스트합니다."""
    print("\n🧪 메모리 최적화 테스트를 실행합니다...")
    
    try:
        # quick_test.py 실행
        result = subprocess.run([sys.executable, "test/quick_test.py"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ 메모리 최적화 테스트 성공!")
            print(result.stdout)
            return True
        else:
            print("❌ 메모리 최적화 테스트 실패:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ 테스트 실행 중 오류: {e}")
        return False

def start_webapp():
    """웹앱을 시작합니다."""
    print("\n🌐 웹앱을 시작합니다...")
    print("웹앱이 시작되면 http://127.0.0.1:5000 에서 접속할 수 있습니다.")
    print("종료하려면 Ctrl+C를 누르세요.")
    
    try:
        subprocess.run([sys.executable, "../webapp/app.py"])
    except KeyboardInterrupt:
        print("\n웹앱이 종료되었습니다.")

def main():
    """메인 실행 함수"""
    print("🚀 Wav2Vec2 메모리 최적화 적용 시작")
    print("=" * 50)
    
    # 1. 패키지 설치
    if not install_packages():
        print("❌ 패키지 설치 실패")
        return
    
    print("\n✅ 모든 패키지 설치 완료!")
    
    # 2. 최적화 테스트
    if not test_optimization():
        print("❌ 최적화 테스트 실패")
        return
    
    print("\n✅ 메모리 최적화 적용 완료!")
    
    # 3. 웹앱 시작
    start_webapp()

if __name__ == "__main__":
    main() 