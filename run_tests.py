#!/usr/bin/env python3
"""
테스트 실행 스크립트
"""

import subprocess
import sys
import os
import argparse

def run_test(test_name):
    """특정 테스트를 실행합니다."""
    test_path = os.path.join("test", f"{test_name}.py")
    
    if not os.path.exists(test_path):
        print(f"❌ 테스트를 찾을 수 없습니다: {test_path}")
        return False
    
    print(f"🧪 {test_name} 테스트 실행 중...")
    try:
        subprocess.run([sys.executable, test_path], check=True)
        print(f"✅ {test_name} 테스트 완료!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {test_name} 테스트 실패: {e}")
        return False

def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description="테스트 실행 스크립트")
    parser.add_argument("--test", choices=["url_classification", "unified_optimization", "all"], 
                       default="all", help="실행할 테스트 선택")
    
    args = parser.parse_args()
    
    print("🚀 테스트 실행 시작")
    print("=" * 40)
    
    if args.test == "url_classification":
        success = run_test("test_url_classification")
    elif args.test == "unified_optimization":
        success = run_test("unified_optimization_test")
    elif args.test == "all":
        print("🔄 모든 테스트 실행 중...")
        success1 = run_test("test_url_classification")
        print("\n" + "-" * 40)
        success2 = run_test("unified_optimization_test")
        success = success1 and success2
    
    if success:
        print("\n🎉 모든 테스트 통과!")
    else:
        print("\n⚠️ 일부 테스트가 실패했습니다.")

if __name__ == "__main__":
    main() 