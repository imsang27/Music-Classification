from flask import Flask, render_template, request, redirect, url_for
import os
import importlib.util

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

if __name__ == '__main__':
    app.run(debug=True)
