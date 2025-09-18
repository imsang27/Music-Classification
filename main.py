"""
음악 분류 시스템 메인 파일
"""

import os
import sys

# 모듈 경로 추가
sys.path.append(os.path.dirname(__file__))

from core import *
from models import *
from utils import *
from data import *
from webapp.app import app

def main():
    """메인 함수"""
    print("🎵 음악 분류 시스템 시작")
    print("=" * 50)
    
    # 기본 설정
    genres = ['클래식', '재즈', '록', '팝']
    emotions = ['행복한', '슬픈', '평화로운', '열정적인']
    
    print(f"📊 지원 장르: {', '.join(genres)}")
    print(f"💭 지원 감정: {', '.join(emotions)}")
    print("=" * 50)
    
    # 웹 애플리케이션 실행
    print("🌐 웹 애플리케이션을 시작합니다...")
    print("📍 접속 주소: http://localhost:5000")
    print("🛑 종료하려면 Ctrl+C를 누르세요")
    print("=" * 50)
    
    app.run(debug=True, host='0.0.0.0', port=5000)

if __name__ == "__main__":
    main()