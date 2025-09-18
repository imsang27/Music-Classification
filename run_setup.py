#!/usr/bin/env python3
"""
최적화 설정 실행 스크립트
"""

import subprocess
import sys
import os

def main():
    """메인 실행 함수"""
    print("🚀 최적화 설정 시작")
    print("=" * 40)
    
    # scripts 디렉토리로 이동
    script_path = os.path.join("scripts", "setup_optimization.py")
    
    if not os.path.exists(script_path):
        print(f"❌ 스크립트를 찾을 수 없습니다: {script_path}")
        return
    
    try:
        subprocess.run([sys.executable, script_path], check=True)
        print("\n✅ 최적화 설정 완료!")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 최적화 설정 실패: {e}")

if __name__ == "__main__":
    main() 