"""
Flask 웹 애플리케이션
"""

# TensorFlow oneDNN 경고 메시지 억제
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from flask import Flask, render_template, request, redirect, url_for, jsonify, Response
import sys
import threading
import time
import uuid
import json

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

# 진행률 추적을 위한 전역 변수
progress_tracker = {}
progress_lock = threading.Lock()

# 분류 결과 저장을 위한 전역 변수
classification_results = {}
results_lock = threading.Lock()

def update_progress(task_id, stage, message, progress=0):
    """진행률 업데이트"""
    with progress_lock:
        progress_tracker[task_id] = {
            'stage': stage,
            'message': message,
            'progress': progress,
            'timestamp': time.time()
        }

def get_progress(task_id):
    """진행률 조회"""
    with progress_lock:
        return progress_tracker.get(task_id, {
            'stage': 'unknown',
            'message': '작업을 찾을 수 없습니다.',
            'progress': 0,
            'timestamp': time.time()
        })

def clear_progress(task_id):
    """진행률 정리"""
    with progress_lock:
        if task_id in progress_tracker:
            del progress_tracker[task_id]

def save_classification_result(task_id, result):
    """분류 결과 저장"""
    with results_lock:
        classification_results[task_id] = result

def get_classification_result(task_id):
    """분류 결과 조회"""
    with results_lock:
        return classification_results.get(task_id, None)

def clear_classification_result(task_id):
    """분류 결과 정리"""
    with results_lock:
        if task_id in classification_results:
            del classification_results[task_id]

# Try to load Wav2Vec2 model if available
wav2vec2_model_name = None  # 전역 변수로 모델 이름 저장

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
            
            # 모델 이름 저장 - models 폴더부터 경로 분석
            wav2vec2_model_name = 'music_genres_classification'  # 기본값
            
            try:
                # MODEL_PATH에서 models 폴더부터의 경로 추출
                # 예: .../models/models--dima806--music_genres_classification/snapshots/...
                model_path_str = str(MODEL_PATH)
                print(f"   - 전체 모델 경로: {model_path_str}")
                
                # models 폴더 찾기
                if 'models' in model_path_str:
                    # models 폴더 이후의 경로 추출
                    models_index = model_path_str.find('models')
                    models_path = model_path_str[models_index:]
                    print(f"   - models 폴더부터 경로: {models_path}")
                    
                    # 경로를 / 또는 \ 로 분할
                    path_parts = models_path.replace('\\', '/').split('/')
                    print(f"   - 경로 분할: {path_parts}")
                    
                    # models 폴더 다음에 오는 폴더명 찾기
                    for i, part in enumerate(path_parts):
                        if part == 'models' and i + 1 < len(path_parts):
                            model_folder = path_parts[i + 1]
                            print(f"   - 모델 폴더명: {model_folder}")
                            
                            # models--dima806--music_genres_classification 형태에서 마지막 부분 추출
                            if '--' in model_folder:
                                wav2vec2_model_name = model_folder.split('--')[-1]
                                print(f"   - 추출된 모델명: {wav2vec2_model_name}")
                            else:
                                wav2vec2_model_name = model_folder
                                print(f"   - 폴더명 그대로 사용: {wav2vec2_model_name}")
                            break
                else:
                    print("   - models 폴더를 찾을 수 없음, 기본값 사용")
                    
            except Exception as e:
                print(f"   - 모델명 추출 중 오류: {e}")
                wav2vec2_model_name = 'music_genres_classification'
            
            print(f"   - 최종 모델명: {wav2vec2_model_name}")
            
            # GPU 사용 가능 시 GPU로 이동
            if torch.cuda.is_available():
                wav2vec2_model = wav2vec2_model.cuda()
                print("✅ GPU 사용 가능 - Wav2Vec2 모델을 GPU로 이동")
            
            print("✅ Wav2Vec2 모델 로드 성공")
            print(f"   - 모델 이름: {wav2vec2_model_name}")
            print(f"   - 지원 장르: {list(wav2vec2_model.config.id2label.values())}")
        except Exception as e:
            print(f"❌ Wav2Vec2 모델 로드 실패: {str(e)}")
            wav2vec2_model = None
            wav2vec2_processor = None
            wav2vec2_model_name = None
    else:
        print("❌ 모델 폴더가 존재하지 않습니다: models/models--dima806--music_genres_classification")
        wav2vec2_model = None
        wav2vec2_processor = None
        wav2vec2_model_name = None
except Exception as e:
    print(f"❌ Transformers/PyTorch 임포트 실패: {str(e)}")
    wav2vec2_model = None
    wav2vec2_processor = None
    wav2vec2_model_name = None

@app.route('/')
def index():
    model_status = "로드됨" if wav2vec2_model else "로드되지 않음"
    return render_template('index.html', prediction=None, model_status=model_status)

@app.route('/progress/<task_id>')
def progress_page(task_id):
    """진행률 표시 페이지"""
    url = request.args.get('url', '')
    return render_template('progress.html', task_id=task_id, url=url)

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

    # 임시 파일 생성 (처리용) - 최적화된 방식
    import tempfile
    import shutil
    
    temp_file = None
    try:
        # NamedTemporaryFile을 사용하여 더 안전한 임시 파일 생성
        with tempfile.NamedTemporaryFile(
            suffix=os.path.splitext(audio_file.filename)[1], 
            delete=False
        ) as temp_f:
            temp_file = temp_f.name
        
        # 원본 파일을 임시 파일로 복사
        shutil.copy2(original_file_path, temp_file)
        
        action = request.form.get('action')
        print(f"DEBUG: 받은 action 값: '{action}'")  # 디버깅용
        print(f"DEBUG: request.form 전체: {dict(request.form)}")  # 디버깅용

        # 기존 분류 방법들을 위한 장르와 감정 정의
        genres = ['블루스', '클래식', '컨트리', '디스코', '힙합', '재즈', '메탈', '팝', '레게', '록']
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
            print(f"DEBUG: 예상되지 않은 action 값: '{action}'")  # 디버깅용
            result = {'error': f'선택한 기능을 사용할 수 없습니다. (action: {action})'}

    except Exception as e:
        result = {'error': f'파일 처리 중 오류 발생: {str(e)}'}
    finally:
        # 임시 파일 정리 (더 안전한 방식)
        if temp_file:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    print(f"파일 업로드 임시 파일 정리: {temp_file}")
            except Exception as cleanup_error:
                print(f"임시 파일 정리 실패: {cleanup_error}")

    return render_template('index.html', prediction=result)

@app.route('/classify_url', methods=['POST'])
def classify_url():
    url = request.form.get('url', '').strip()
    if not url:
        return render_template('index.html', prediction={'error': 'URL을 입력해주세요.'})

    action = request.form.get('action')
    print(f"DEBUG: URL 분류 - 받은 action 값: '{action}'")  # 디버깅용
    print(f"DEBUG: URL 분류 - request.form 전체: {dict(request.form)}")  # 디버깅용
    task_id = str(uuid.uuid4())
    
    try:
        # 진행률 초기화
        update_progress(task_id, 'starting', '분류 작업을 시작합니다...', 0)
        
        if action == 'wav2vec2':
            if wav2vec2_model:
                # 백그라운드에서 실행
                def run_classification():
                    try:
                        # 진행률 콜백 함수 정의
                        def progress_callback(stage, message, progress):
                            update_progress(task_id, stage, message, progress)
                        
                        result = classify_music_from_url_wav2vec2(
                            wav2vec2_model, 
                            wav2vec2_processor, 
                            url, 
                            progress_callback=progress_callback
                        )
                        # 결과 저장
                        save_classification_result(task_id, result)
                        return result
                    except Exception as e:
                        error_result = {'error': str(e), 'success': False}
                        update_progress(task_id, 'error', f'분류 중 오류 발생: {str(e)}', 0)
                        # 오류 결과도 저장
                        save_classification_result(task_id, error_result)
                        return error_result
                
                # 백그라운드 스레드에서 실행
                thread = threading.Thread(target=run_classification)
                thread.daemon = True
                thread.start()
                
                # 진행률 페이지로 리다이렉트
                return redirect(url_for('progress_page', task_id=task_id, url=url))
            else:
                # Wav2Vec2 모델이 없을 때 더미 분류 사용
                result = dummy_classification(url)
        elif action == 'rule':
            # 규칙 기반 분류는 로컬 파일만 지원하므로 임시 다운로드 후 처리
            temp_file = None
            try:
                update_progress(task_id, 'downloading', '음악을 다운로드하는 중...', 30)
                if 'youtube.com' in url or 'youtu.be' in url:
                    temp_file = download_youtube_audio(url, max_duration=30)  # 길이 제한으로 최적화
                else:
                    temp_file = download_direct_audio(url, max_size_mb=50)  # 크기 제한으로 최적화
                
                update_progress(task_id, 'processing', '규칙 기반으로 분류하는 중...', 70)
                result = rule_based_classification(temp_file)
                update_progress(task_id, 'completed', '분류가 완료되었습니다!', 100)
            finally:
                # 임시 파일 정리 (더 안전한 방식)
                if temp_file:
                    try:
                        if os.path.exists(temp_file):
                            os.remove(temp_file)
                            print(f"규칙 기반 분류 임시 파일 정리: {temp_file}")
                    except Exception as cleanup_error:
                        print(f"임시 파일 정리 실패: {cleanup_error}")
        else:
            print(f"DEBUG: URL 분류 - 예상되지 않은 action 값: '{action}'")  # 디버깅용
            result = {'error': f'선택한 기능을 사용할 수 없습니다. (action: {action})'}
        
        return render_template('index.html', prediction=result, url=url)
        
    except Exception as e:
        update_progress(task_id, 'error', f'분류 중 오류 발생: {str(e)}', 0)
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
    """링크 미리보기 API (상세 정보)"""
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
    progress = get_progress(task_id)
    result = get_classification_result(task_id)
    
    response = {
        'task_id': task_id,
        'stage': progress['stage'],
        'message': progress['message'],
        'progress': progress['progress'],
        'timestamp': progress['timestamp']
    }
    
    # 완료된 경우 결과도 포함
    if result:
        response['result'] = result
    
    return jsonify(response)

@app.route('/api/progress_stream/<task_id>')
def progress_stream(task_id):
    """Server-Sent Events로 실시간 진행률 스트리밍"""
    def generate():
        while True:
            progress = get_progress(task_id)
            result = get_classification_result(task_id)
            
            # JSON 데이터를 SSE 형식으로 전송
            data = {
                'task_id': task_id,
                'stage': progress['stage'],
                'message': progress['message'],
                'progress': progress['progress'],
                'timestamp': progress['timestamp']
            }
            
            # 완료된 경우 결과도 포함
            if result:
                data['result'] = result
            
            yield f"data: {json.dumps(data)}\n\n"
            
            # 완료되면 종료
            if progress['stage'] in ['completed', 'error']:
                break
                
            time.sleep(0.5)  # 0.5초마다 업데이트
    
    return Response(generate(), mimetype='text/event-stream', headers={
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Cache-Control'
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

@app.route('/api/model_status', methods=['GET'])
def get_model_status():
    """모델 상태 조회 API"""
    try:
        model_loaded = wav2vec2_model is not None
        status = "로드됨" if model_loaded else "로드되지 않음"
        
        # 모델 이름 결정 - 저장된 모델 이름 사용
        model_name = wav2vec2_model_name if model_loaded else None
        print(f"API 호출 - 모델 로드됨: {model_loaded}, 모델명: {model_name}")
        
        # 추가 정보
        info = {
            'model_loaded': model_loaded,
            'status': status,
            'model_type': model_name,
            'model_name': model_name,  # 호환성을 위해 추가
            'gpu_available': torch.cuda.is_available() if 'torch' in globals() else False
        }
        
        if model_loaded and hasattr(wav2vec2_model, 'config'):
            info['supported_genres'] = list(wav2vec2_model.config.id2label.values())
        
        return jsonify({
            'success': True,
            'model_status': info
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'모델 상태 조회 중 오류 발생: {str(e)}'
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