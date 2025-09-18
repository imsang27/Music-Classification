"""
Flask 웹 애플리케이션
"""

from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
import sys

# 모듈 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# 안전한 import
try:
    from core.predictor import predict_music_wav2vec2, dummy_classification, classify_music_from_url_wav2vec2, batch_classify_urls_wav2vec2
    from core.classifier import rule_based_classification, manual_classification
except ImportError as e:
    print(f"Warning: core 모듈 import 실패: {e}")
    # 기본 함수들 정의
    def predict_music_wav2vec2(*args, **kwargs):
        return {'error': 'core 모듈을 로드할 수 없습니다', 'success': False}
    
    def rule_based_classification(*args, **kwargs):
        return {'error': 'core 모듈을 로드할 수 없습니다', 'success': False}
    
    def dummy_classification(*args, **kwargs):
        return {'error': 'core 모듈을 로드할 수 없습니다', 'success': False}
    
    def classify_music_from_url_wav2vec2(*args, **kwargs):
        return {'error': 'core 모듈을 로드할 수 없습니다', 'success': False}
    
    def batch_classify_urls_wav2vec2(*args, **kwargs):
        return [{'error': 'core 모듈을 로드할 수 없습니다', 'success': False}]
    
    def manual_classification(*args, **kwargs):
        return {'error': 'core 모듈을 로드할 수 없습니다', 'success': False}

try:
    from utils.downloader import download_youtube_audio, download_direct_audio, validate_url, get_link_preview
    from utils.memory_optimizer import monitor_memory_usage, optimize_memory_usage
except ImportError as e:
    print(f"Warning: utils 모듈 import 실패: {e}")
    # 기본 함수들 정의
    def download_youtube_audio(*args, **kwargs):
        return None
    
    def download_direct_audio(*args, **kwargs):
        return None
    
    def validate_url(*args, **kwargs):
        return False, "utils 모듈을 로드할 수 없습니다"
    
    def get_link_preview(*args, **kwargs):
        return {'error': 'utils 모듈을 로드할 수 없습니다'}
    
    def monitor_memory_usage(*args, **kwargs):
        return {'rss': 0, 'vms': 0, 'percent': 0}
    
    def optimize_memory_usage(*args, **kwargs):
        pass

try:
    from data.reporter import save_url_classification_results
except ImportError as e:
    print(f"Warning: data 모듈 import 실패: {e}")
    # 기본 함수들 정의
    def save_url_classification_results(*args, **kwargs):
        pass

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
            
            # GPU 사용 가능 시 GPU로 이동
            if torch.cuda.is_available():
                wav2vec2_model = wav2vec2_model.cuda()
                print("✅ GPU 사용 가능 - Wav2Vec2 모델을 GPU로 이동")
            
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

    # 업로드 폴더에 원본 파일 저장
    os.makedirs('uploads', exist_ok=True)
    original_file_path = os.path.join('uploads', audio_file.filename)
    audio_file.save(original_file_path)

    # 임시 파일 생성 (처리용)
    import tempfile
    import shutil
    
    temp_file = None
    try:
        # 임시 파일 생성
        temp_fd, temp_file = tempfile.mkstemp(suffix=os.path.splitext(audio_file.filename)[1])
        os.close(temp_fd)
        
        # 원본 파일을 임시 파일로 복사
        shutil.copy2(original_file_path, temp_file)
        
        action = request.form.get('action')

        # 기존 분류 방법들을 위한 장르와 감정 정의
        genres = ['클래식', '재즈', '록', '팝']
        emotions = ['행복한', '슬픈', '평화로운', '열정적인']

        if action == 'wav2vec2' and wav2vec2_model:
            # Wav2Vec2 모델을 사용한 분류 (임시 파일 사용, 메모리 최적화)
            try:
                # 메모리 사용량 모니터링
                initial_memory = monitor_memory_usage()
                print(f"웹앱 분류 시작 - 메모리: {initial_memory['rss']:.1f} MB")
                
                result = predict_music_wav2vec2(wav2vec2_model, wav2vec2_processor, temp_file)
                
                # 메모리 정리
                optimize_memory_usage()
                
                final_memory = monitor_memory_usage()
                print(f"웹앱 분류 완료 - 메모리: {final_memory['rss']:.1f} MB")
                
            except Exception as e:
                result = {'error': f'Wav2Vec2 분류 중 오류 발생: {str(e)}', 'success': False}
        elif action == 'wav2vec2' and not wav2vec2_model:
            # Wav2Vec2 모델이 없을 때 기존 CNN 모델 사용 (만약 있다면)
            result = {'error': 'Wav2Vec2 모델이 로드되지 않았습니다. 규칙 기반 분류를 사용하세요.'}
        elif action == 'rule':
            result = rule_based_classification(temp_file)
        elif action == 'manual':
            result = manual_classification(temp_file, genres, emotions)
        elif action == 'ml':
            result = {'error': '전통적인 ML 모델 기능은 아직 구현되지 않았습니다.'}
        elif action == 'lyrics':
            result = {'error': '가사 분석 기능은 아직 구현되지 않았습니다.'}
        elif action == 'hybrid':
            result = {'error': '통합 분류 기능은 아직 구현되지 않았습니다.'}
        else:
            result = {'error': '선택한 기능을 사용할 수 없습니다.'}

    except Exception as e:
        result = {'error': f'파일 처리 중 오류 발생: {str(e)}'}
    finally:
        # 임시 파일 삭제
        if temp_file and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except Exception as e:
                print(f"임시 파일 삭제 실패: {str(e)}")

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
                # 개선된 일괄 처리 함수 사용
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

@app.route('/clear_uploads', methods=['POST'])
def clear_uploads():
    """업로드 폴더 비우기"""
    try:
        uploads_dir = 'uploads'
        if os.path.exists(uploads_dir):
            # 폴더 내 모든 파일 삭제
            for filename in os.listdir(uploads_dir):
                file_path = os.path.join(uploads_dir, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            
            return jsonify({
                'success': True,
                'message': '업로드 폴더가 성공적으로 비워졌습니다.'
            })
        else:
            return jsonify({
                'success': False,
                'message': '업로드 폴더가 존재하지 않습니다.'
            })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'폴더 정리 중 오류 발생: {str(e)}'
        })

@app.route('/get_uploads_info', methods=['GET'])
def get_uploads_info():
    """업로드 폴더 정보 조회"""
    try:
        uploads_dir = 'uploads'
        if os.path.exists(uploads_dir):
            files = []
            total_size = 0
            
            for filename in os.listdir(uploads_dir):
                file_path = os.path.join(uploads_dir, filename)
                if os.path.isfile(file_path):
                    file_size = os.path.getsize(file_path)
                    files.append({
                        'name': filename,
                        'size': file_size,
                        'size_formatted': format_file_size(file_size)
                    })
                    total_size += file_size
            
            return jsonify({
                'success': True,
                'file_count': len(files),
                'total_size': total_size,
                'total_size_formatted': format_file_size(total_size),
                'files': files
            })
        else:
            return jsonify({
                'success': True,
                'file_count': 0,
                'total_size': 0,
                'total_size_formatted': '0 B',
                'files': []
            })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'폴더 정보 조회 중 오류 발생: {str(e)}'
        })

def format_file_size(size_bytes):
    """파일 크기를 사람이 읽기 쉬운 형태로 변환"""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"

if __name__ == '__main__':
    app.run(debug=True)