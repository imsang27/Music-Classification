"""
데이터 처리 관련 기능
"""

import json
import os
from datetime import datetime


def create_url_classification_report(results):
    """URL 분류 결과를 보고서 형태로 생성합니다"""
    report = {
        'summary': {
            'total': len(results),
            'success': len([r for r in results if r['status'] == 'success']),
            'failed': len([r for r in results if r['status'] == 'error'])
        },
        'genre_distribution': {},
        'emotion_distribution': {},
        'detailed_results': results
    }
    
    # 장르 및 감정 분포 계산
    for result in results:
        if result['status'] == 'success':
            # 장르 분포
            if 'genres' in result:
                for genre, confidence in result['genres']:
                    if genre not in report['genre_distribution']:
                        report['genre_distribution'][genre] = 0
                    report['genre_distribution'][genre] += 1
            
            # 감정 분포
            if 'emotions' in result:
                for emotion, confidence in result['emotions']:
                    if emotion not in report['emotion_distribution']:
                        report['emotion_distribution'][emotion] = 0
                    report['emotion_distribution'][emotion] += 1
    
    return report


def save_url_classification_results(results, output_file='url_classification_results.json'):
    """URL 분류 결과를 JSON 파일로 저장합니다"""
    report = create_url_classification_report(results)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"결과가 {output_file}에 저장되었습니다.")


def print_url_classification_summary(results):
    """URL 분류 결과 요약 출력"""
    if not results:
        print("분류 결과가 없습니다.")
        return
    
    total = len(results)
    successful = sum(1 for r in results if r.get('status') == 'success')
    failed = total - successful
    
    print(f"\n=== URL 분류 결과 요약 ===")
    print(f"총 URL 수: {total}")
    print(f"성공: {successful}")
    print(f"실패: {failed}")
    
    if successful > 0:
        # 장르별 통계
        genre_counts = {}
        for result in results:
            if result.get('status') == 'success' and 'genre' in result:
                genre = result['genre']
                genre_counts[genre] = genre_counts.get(genre, 0) + 1
        
        print(f"\n장르별 분포:")
        for genre, count in sorted(genre_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / successful) * 100
            print(f"  {genre}: {count}개 ({percentage:.1f}%)") 