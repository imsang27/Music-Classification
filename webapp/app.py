from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
import importlib.util
import tempfile

# Load the main classification module dynamically because of the hyphen in the filename
MODULE_PATH = os.path.join(os.path.dirname(__file__), '..', 'Music-Classification.py')
spec = importlib.util.spec_from_file_location('music_classification', MODULE_PATH)
mc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mc)

app = Flask(__name__)

# Try to load CNN model if available
try:
    from tensorflow.python.keras.models import load_model
    MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'best_model.h5')
    cnn_model = load_model(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
except Exception:
    cnn_model = None

@app.route('/')
def index():
    return render_template('index.html', prediction=None)

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

    if action == 'cnn' and cnn_model:
        genres = ['클래식', '재즈', '록', '팝']
        emotions = ['행복한', '슬픈', '평화로운', '열정적인']
        result = mc.predict_music(cnn_model, file_path, genres, emotions)
    elif action == 'rule':
        result = mc.rule_based_classification(file_path)
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
        if action == 'cnn' and cnn_model:
            genres = ['클래식', '재즈', '록', '팝']
            emotions = ['행복한', '슬픈', '평화로운', '열정적인']
            result = mc.classify_music_from_url(cnn_model, url, genres, emotions)
        elif action == 'rule':
            # 규칙 기반 분류는 로컬 파일만 지원하므로 임시 다운로드 후 처리
            temp_file = None
            try:
                if 'youtube.com' in url or 'youtu.be' in url:
                    temp_file = mc.download_youtube_audio(url)
                else:
                    temp_file = mc.download_direct_audio(url)
                result = mc.rule_based_classification(temp_file)
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
        if action == 'cnn' and cnn_model:
            genres = ['클래식', '재즈', '록', '팝']
            emotions = ['행복한', '슬픈', '평화로운', '열정적인']
            results = mc.batch_classify_urls(cnn_model, urls, genres, emotions)
            
            # 결과 저장
            mc.save_url_classification_results(results)
            
            return render_template('index.html', batch_results=results, urls=urls)
        else:
            return render_template('index.html', prediction={'error': '일괄 분류는 CNN 모델만 지원합니다.'})
            
    except Exception as e:
        return render_template('index.html', prediction={'error': f'일괄 분류 중 오류 발생: {str(e)}'})

@app.route('/api/validate_url', methods=['POST'])
def validate_url_api():
    """URL 검증 API"""
    url = request.json.get('url', '').strip()
    if not url:
        return jsonify({'valid': False, 'message': 'URL을 입력해주세요.'})
    
    try:
        is_valid, message = mc.validate_url(url)
        return jsonify({'valid': is_valid, 'message': message})
    except Exception as e:
        return jsonify({'valid': False, 'message': f'검증 중 오류 발생: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True)
