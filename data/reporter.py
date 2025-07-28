"""
리포터 관련 기능
"""

import json
import os
from datetime import datetime


def generate_classification_report(results, output_file='classification_report.json'):
    """분류 결과 보고서 생성"""
    report = {
        'timestamp': datetime.now().isoformat(),
        'total_samples': len(results),
        'successful_classifications': len([r for r in results if r.get('success', False)]),
        'failed_classifications': len([r for r in results if not r.get('success', False)]),
        'genre_distribution': {},
        'emotion_distribution': {},
        'confidence_stats': {
            'min': float('inf'),
            'max': 0,
            'avg': 0
        },
        'detailed_results': results
    }
    
    # 통계 계산
    confidences = []
    for result in results:
        if result.get('success', False):
            # 장르 분포
            if 'genre' in result:
                genre = result['genre']
                report['genre_distribution'][genre] = report['genre_distribution'].get(genre, 0) + 1
            
            # 감정 분포
            if 'emotions' in result:
                for emotion, confidence in result['emotions']:
                    report['emotion_distribution'][emotion] = report['emotion_distribution'].get(emotion, 0) + 1
            
            # 신뢰도 통계
            if 'confidence' in result:
                confidences.append(result['confidence'])
    
    if confidences:
        report['confidence_stats']['min'] = min(confidences)
        report['confidence_stats']['max'] = max(confidences)
        report['confidence_stats']['avg'] = sum(confidences) / len(confidences)
    
    # 파일 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    return report


def print_detailed_report(report):
    """상세 보고서 출력"""
    print("\n" + "="*50)
    print("음악 분류 결과 상세 보고서")
    print("="*50)
    
    print(f"총 샘플 수: {report['total_samples']}")
    print(f"성공한 분류: {report['successful_classifications']}")
    print(f"실패한 분류: {report['failed_classifications']}")
    
    if report['successful_classifications'] > 0:
        success_rate = (report['successful_classifications'] / report['total_samples']) * 100
        print(f"성공률: {success_rate:.1f}%")
        
        print(f"\n신뢰도 통계:")
        print(f"  최소: {report['confidence_stats']['min']:.3f}")
        print(f"  최대: {report['confidence_stats']['max']:.3f}")
        print(f"  평균: {report['confidence_stats']['avg']:.3f}")
        
        if report['genre_distribution']:
            print(f"\n장르별 분포:")
            for genre, count in sorted(report['genre_distribution'].items(), key=lambda x: x[1], reverse=True):
                percentage = (count / report['successful_classifications']) * 100
                print(f"  {genre}: {count}개 ({percentage:.1f}%)")
        
        if report['emotion_distribution']:
            print(f"\n감정별 분포:")
            for emotion, count in sorted(report['emotion_distribution'].items(), key=lambda x: x[1], reverse=True):
                percentage = (count / report['successful_classifications']) * 100
                print(f"  {emotion}: {count}개 ({percentage:.1f}%)")
    
    print("="*50) 