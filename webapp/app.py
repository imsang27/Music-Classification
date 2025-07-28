"""
Flask 웹 애플리케이션
"""

from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
import sys

# 모듈 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from core import *
from models import *
from utils import *
from data import *

app = Flask(__name__)

# Try to load Wav2Vec2 model if available
try:
    from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
    import torch
    
    MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'models--dima806--music_genres_classification', 'snapshots', '5f71fb1e2c6bedcddb2bfb1e929fc70655780902')
    
    if os.path.exists(MODEL_PATH):
        try:
            # Load the Wav2Vec2 model and feature extractor
            wav2vec2_model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_PATH)
            wav2vec2_processor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_PATH)
            
            # Set model to evaluation mode
            wav2vec2_model.eval()
            
            print("✅ Wav2Vec2 모델 로드 성공")
            print(f"   - 지원 장르: {list(wav2vec2_model.config.id2label.values())}")
        except Exception as e:
            print(f"❌ Wav2Vec2 모델 로드 실패: {str(e)}")
            wav2vec2_model = None
            wav2vec2_processor = None
    else:
        print("❌ 모델 폴더가 존재하지 않습니다: models/models--dima806--music_genres_classification")
        wav2vec2_model = None
        wav2vec2_processor = None
except Exception as e:
    print(f"❌ Transformers/PyTorch 임포트 실패: {str(e)}")
    wav2vec2_model = None
    wav2vec2_processor = None

@app.route('/')
def index():
    model_status = "로드됨" if wav2vec2_model else "로드되지 않음"
    return render_template('index.html', prediction=None, model_status=model_status)

@app.route('/classify', methods=['POST'])
def classify():
    if 'audio_file' not in request.files:
        return redirect(url_for('index'))

    audio_file = request.files['audio_file']
    if audio_file.filename == '':
        return redirect(url_for('index'))

    os.makedirs('uploads', exist_ok=True)
    file_path = os.path.join('uploads', audio_file.filename)
    audio_file.save(file_path)

    action = request.form.get('action')

    # 기존 분류 방법들을 위한 장르와 감정 정의
    genres = ['클래식', '재즈', '록', '팝']
    emotions = ['행복한', '슬픈', '평화로운', '열정적인']

    if action == 'wav2vec2' and wav2vec2_model:
        # Wav2Vec2 모델을 사용한 분류
        result = predict_music_wav2vec2(wav2vec2_model, wav2vec2_processor, file_path)
    elif action == 'wav2vec2' and not wav2vec2_model:
        # Wav2Vec2 모델이 없을 때 기존 CNN 모델 사용 (만약 있다면)
        result = {'error': 'Wav2Vec2 모델이 로드되지 않았습니다. 규칙 기반 분류를 사용하세요.'}
    elif action == 'rule':
        result = rule_based_classification(file_path)
    elif action == 'manual':
        result = manual_classification(file_path, genres, emotions)
    elif action == 'ml':
        result = predict_traditional_ml_model(None, file_path)  # 모델이 없으면 오류
    elif action == 'lyrics':
        result = {'error': '가사 분석 기능은 아직 구현되지 않았습니다.'}
    elif action == 'hybrid':
        result = {'error': '통합 분류 기능은 아직 구현되지 않았습니다.'}
    else:
        result = {'error': '선택한 기능을 사용할 수 없습니다.'}

    return render_template('index.html', prediction=result)

@app.route('/classify_url', methods=['POST'])
def classify_url():
    url = request.form.get('url', '').strip()
    if not url:
        return render_template('index.html', prediction={'error': 'URL을 입력해주세요.'})

    action = request.form.get('action')
    
    try:
        if action == 'wav2vec2':
            if wav2vec2_model:
                result = classify_music_from_url_wav2vec2(wav2vec2_model, wav2vec2_processor, url)
            else:
                # Wav2Vec2 모델이 없을 때 더미 분류 사용
                result = dummy_classification(url)
        elif action == 'rule':
            # 규칙 기반 분류는 로컬 파일만 지원하므로 임시 다운로드 후 처리
            temp_file = None
            try:
                if 'youtube.com' in url or 'youtu.be' in url:
                    temp_file = download_youtube_audio(url)
                else:
                    temp_file = download_direct_audio(url)
                result = rule_based_classification(temp_file)
            finally:
                if temp_file and os.path.exists(temp_file):
                    os.remove(temp_file)
        else:
            result = {'error': '선택한 기능을 사용할 수 없습니다.'}
        
        return render_template('index.html', prediction=result, url=url)
        
    except Exception as e:
        return render_template('index.html', prediction={'error': f'분류 중 오류 발생: {str(e)}'}, url=url)

@app.route('/batch_classify', methods=['POST'])
def batch_classify():
    urls_text = request.form.get('urls', '').strip()
    if not urls_text:
        return render_template('index.html', prediction={'error': 'URL 목록을 입력해주세요.'})

    # URL 목록 파싱 (줄바꿈이나 쉼표로 구분)
    urls = [url.strip() for url in urls_text.replace(',', '\n').split('\n') if url.strip()]
    
    if not urls:
        return render_template('index.html', prediction={'error': '유효한 URL이 없습니다.'})

    action = request.form.get('action')
    
    try:
        if action == 'wav2vec2':
            if wav2vec2_model:
                results = batch_classify_urls_wav2vec2(wav2vec2_model, wav2vec2_processor, urls)
            else:
                # Wav2Vec2 모델이 없을 때 더미 분류로 일괄 처리
                results = []
                for url in urls:
                    try:
                        result = dummy_classification(url)
                        result['url'] = url
                        result['status'] = 'success'
                        results.append(result)
                    except Exception as e:
                        results.append({
                            'url': url,
                            'status': 'error',
                            'error': str(e)
                        })
            
            # 결과 저장
            save_url_classification_results(results)
            
            return render_template('index.html', batch_results=results, urls=urls)
        else:
            return render_template('index.html', prediction={'error': '일괄 분류는 Wav2Vec2 모델만 지원합니다.'})
            
    except Exception as e:
        return render_template('index.html', prediction={'error': f'일괄 분류 중 오류 발생: {str(e)}'})

@app.route('/api/validate_url', methods=['POST'])
def validate_url_api():
    """URL 검증 API"""
    url = request.json.get('url', '').strip()
    if not url:
        return jsonify({'valid': False, 'message': 'URL을 입력해주세요.'})
    
    try:
        is_valid, message = validate_url(url)
        return jsonify({'valid': is_valid, 'message': message})
    except Exception as e:
        return jsonify({'valid': False, 'message': f'검증 중 오류 발생: {str(e)}'})

@app.route('/api/link_preview', methods=['POST'])
def link_preview_api():
    """링크 미리보기 API"""
    url = request.json.get('url', '').strip()
    if not url:
        return jsonify({'error': 'URL을 입력해주세요.'})
    
    try:
        preview = get_link_preview(url)
        return jsonify(preview)
    except Exception as e:
        return jsonify({'error': f'미리보기 생성 중 오류 발생: {str(e)}'})

@app.route('/api/progress/<task_id>', methods=['GET'])
def get_progress_api(task_id):
    """분류 진행률 조회 API"""
    # 간단한 진행률 시뮬레이션 (실제로는 Celery나 Redis를 사용해야 함)
    return jsonify({
        'task_id': task_id,
        'status': 'processing',
        'progress': 50,
        'message': '분류 중...'
    })

if __name__ == '__main__':
    app.run(debug=True)