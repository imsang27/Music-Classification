import librosa
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import KFold
import os
from sklearn.metrics import classification_report, confusion_matrix
import json
from datetime import datetime
import hashlib
import time
import psutil
import matplotlib.pyplot as plt

# 오디오 특성을 추출하는 함수
def extract_audio_features(audio_path, duration=60):
    try:
        # 파일 존재 여부 확인
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {audio_path}")
            
        y, sr = librosa.load(audio_path, duration=duration)
        if len(y) == 0:
            raise ValueError("오디오 파일이 비어있습니다")
        
        # 샘플링 레이트 체크
        if sr != 22050:  # librosa 기본값
            y = librosa.resample(y, orig_sr=sr, target_sr=22050)
            sr = 22050
            
        # 멜 스펙트로그램
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
        mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
        mel_spectrogram = (mel_spectrogram - np.mean(mel_spectrogram)) / np.std(mel_spectrogram)
        
        # 템포
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        
        # 크로마그램 (음조 특성)
        chromagram = librosa.feature.chroma_stft(y=y, sr=sr)
        chromagram = (chromagram - np.mean(chromagram)) / np.std(chromagram)
        
        # MFCC (음색 특성)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)
        
        # 스펙트럴 대비 (음악의 다이나믹 특성)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        
        return {
            'mel_spectrogram': mel_spectrogram,
            'tempo': tempo,
            'chromagram': np.mean(chromagram, axis=1),
            'mfcc': np.mean(mfcc, axis=1),
            'spectral_contrast': np.mean(spectral_contrast, axis=1)
        }
    except Exception as e:
        print(f"오디오 파일 처리 중 오류 발생: {str(e)}")
        raise

# CNN 모델 생성 (장르와 감정 모두 분류)
def create_model(input_shape, num_genres, num_emotions, dropout_rate=0.5, learning_rate=0.001):
    # 공통 특성 추출 레이어
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # 장르 분류 브랜치
    genre_output = layers.Dense(num_genres, activation='softmax', name='genre_output')(x)
    
    # 감정 분류 브랜치
    emotion_output = layers.Dense(num_emotions, activation='softmax', name='emotion_output')(x)
    
    model = models.Model(inputs=inputs, outputs=[genre_output, emotion_output])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss={
            'genre_output': 'sparse_categorical_crossentropy',
            'emotion_output': 'sparse_categorical_crossentropy'
        },
        metrics=['accuracy']
    )
    return model

# 데이터 전처리 및 모델 학습
def train_music_classifier(data_path, genres, emotions):
    X = []
    y_genre = []
    y_emotion = []

    # 데이터 로드 및 전처리
    for genre_idx, genre in enumerate(genres):
        genre_path = os.path.join(data_path, genre)
        if not os.path.exists(genre_path):
            continue
        for file in os.listdir(genre_path):
            if file.endswith(('.mp3', '.wav', '.m4a', '.flac')):
                file_path = os.path.join(genre_path, file)
                try:
                    features = extract_audio_features(file_path)
                    # mel_spectrogram만 CNN 입력으로 사용 (예시)
                    mel = features['mel_spectrogram']
                    if mel.shape[1] < 128:  # 예시: 최소 프레임 길이 보장
                        continue
                    mel = mel[:, :128]  # (n_mels, 128)로 자르기
                    X.append(mel)
                    y_genre.append(genre_idx)
                    # 감정 레이블은 임의로 지정하거나 파일명/메타데이터에서 추출 필요
                    y_emotion.append(0)  # 예시: 임시로 0번 감정
                except Exception as e:
                    print(f"특성 추출 실패: {file_path} - {e}")

    X = np.array(X)
    y_genre = np.array(y_genre)
    y_emotion = np.array(y_emotion)

    print(f"총 수집된 오디오 샘플 수: {len(X)}")
    if len(X) == 0:
        print("ERROR: 오디오 데이터를 하나도 불러오지 못했습니다. 데이터 경로와 폴더 구조, 파일 확장자를 확인하세요.")
        exit()

    # 차원 맞추기 (CNN 입력: (샘플수, n_mels, 프레임수, 1))
    X = X[..., np.newaxis]

    # 데이터 분할
    X_train, X_test, y_genre_train, y_genre_test, y_emotion_train, y_emotion_test = train_test_split(
        X, y_genre, y_emotion, test_size=0.2, random_state=42
    )
    X_train, X_val, y_genre_train, y_genre_val, y_emotion_train, y_emotion_val = train_test_split(
        X_train, y_genre_train, y_emotion_train, test_size=0.2, random_state=42
    )

    # 모델 생성 및 컴파일
    model = create_model(
        input_shape=(X.shape[1], X.shape[2], 1),
        num_genres=len(genres),
        num_emotions=len(emotions)
    )
    
    # 모델 학습
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6
    )
    
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        'best_model.h5',
        monitor='val_loss',
        save_best_only=True,
        mode='min'
    )
    
    callbacks = [
        early_stopping,
        lr_scheduler,
        checkpoint
    ]
    
    # 클래스 가중치 계산
    genre_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_genre_train),
        y=y_genre_train
    )
    emotion_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_emotion_train),
        y=y_emotion_train
    )
    
    genre_weight_dict = dict(zip(range(len(genres)), genre_weights))
    emotion_weight_dict = dict(zip(range(len(emotions)), emotion_weights))
    
    model.fit(
        X_train,
        {'genre_output': y_genre_train, 'emotion_output': y_emotion_train},
        class_weight={
            'genre_output': genre_weight_dict,
            'emotion_output': emotion_weight_dict
        },
        epochs=50,
        callbacks=callbacks,
        validation_data=(X_val, {'genre_output': y_genre_val, 'emotion_output': y_emotion_val})
    )
    
    # 모델 평가 추가
    evaluation = model.evaluate(
        X_test,
        {'genre_output': y_genre_test, 'emotion_output': y_emotion_test},
        verbose=1
    )
    print(f"Genre accuracy: {evaluation[3]:.4f}")
    print(f"Emotion accuracy: {evaluation[4]:.4f}")
    
    return model

# 음악 분류 예측 함수
def predict_music(model, audio_path, genres, emotions, confidence_threshold=0.5):
    features = extract_audio_features(audio_path)
    input_data = features['mel_spectrogram'].reshape(1, *features['mel_spectrogram'].shape, 1)
    genre_pred, emotion_pred = model.predict(input_data)
    
    # 신뢰도가 높은 예측만 반환
    genre_results = [
        (genres[i], float(prob)) 
        for i, prob in enumerate(genre_pred[0]) 
        if prob >= confidence_threshold
    ]
    emotion_results = [
        (emotions[i], float(prob)) 
        for i, prob in enumerate(emotion_pred[0]) 
        if prob >= confidence_threshold
    ]
    
    return {
        'genres': sorted(genre_results, key=lambda x: x[1], reverse=True),
        'emotions': sorted(emotion_results, key=lambda x: x[1], reverse=True)
    }

# 모델 성능 시각화
def plot_training_history(history):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['genre_output_accuracy'], label='Genre Training')
    plt.plot(history.history['val_genre_output_accuracy'], label='Genre Validation')
    plt.title('Genre Classification Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['emotion_output_accuracy'], label='Emotion Training')
    plt.plot(history.history['val_emotion_output_accuracy'], label='Emotion Validation')
    plt.title('Emotion Classification Accuracy')
    plt.legend()
    
    plt.show()

# 교차 검증 함수
def cross_validate_model(X, y_genre, y_emotion, n_splits=5):
    kfold = KFold(n_splits=n_splits, shuffle=True)
    genre_scores = []
    emotion_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_genre_train, y_genre_val = y_genre[train_idx], y_genre[val_idx]
        y_emotion_train, y_emotion_val = y_emotion[train_idx], y_emotion[val_idx]
        
        model = create_model(...)  # 모델 생성
        # ... 모델 학습 ...
        
        scores = model.evaluate(X_val, [y_genre_val, y_emotion_val])
        genre_scores.append(scores[3])  # genre accuracy
        emotion_scores.append(scores[4])  # emotion accuracy
    
    return np.mean(genre_scores), np.mean(emotion_scores)

# 사용 예시
if __name__ == "__main__":
    genres = ['클래식', '재즈', '록', '팝']
    emotions = ['행복한', '슬픈', '평화로운', '열정적인',
                '차분한', '긴장된', '기쁨', '슬픔', '차분함', '신남', '분노']
    model = train_music_classifier('음악_데이터_경로', genres, emotions)

def save_model_with_metadata(model, genres, emotions, metrics, save_path):
    # 모델 저장
    model.save(f"{save_path}/model.h5")
    
    # 메타데이터 저장
    metadata = {
        'genres': genres,
        'emotions': emotions,
        'metrics': metrics,
        'timestamp': datetime.now().isoformat(),
        'version': '1.0'
    }
    
    with open(f"{save_path}/metadata.json", 'w') as f:
        json.dump(metadata, f)

def evaluate_model(model, X_test, y_genre_test, y_emotion_test):
    from sklearn.metrics import classification_report, confusion_matrix
    
    genre_pred, emotion_pred = model.predict(X_test)
    
    # 장르 평가
    genre_report = classification_report(
        np.argmax(y_genre_test, axis=1),
        np.argmax(genre_pred, axis=1),
        target_names=genres
    )
    
    # 감정 평가
    emotion_report = classification_report(
        np.argmax(y_emotion_test, axis=1),
        np.argmax(emotion_pred, axis=1),
        target_names=emotions
    )
    
    return {
        'genre_report': genre_report,
        'emotion_report': emotion_report,
        'genre_confusion': confusion_matrix(
            np.argmax(y_genre_test, axis=1),
            np.argmax(genre_pred, axis=1)
        ),
        'emotion_confusion': confusion_matrix(
            np.argmax(y_emotion_test, axis=1),
            np.argmax(emotion_pred, axis=1)
        )
    }

def validate_dataset(data_path, genres, emotions):
    errors = []
    for genre in genres:
        genre_path = os.path.join(data_path, genre)
        if not os.path.exists(genre_path):
            errors.append(f"장르 폴더를 찾을 수 없음: {genre}")
        else:
            files = os.listdir(genre_path)
            if len(files) < 10:  # 최소 파일 수 체크
                errors.append(f"장르 {genre}의 데이터가 부족함 (현재: {len(files)}개)")
    
    if errors:
        raise ValueError("\n".join(errors))

class PerformanceMonitor:
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.history = []
        
    def on_epoch_end(self, epoch, logs={}):
        metrics = {
            'epoch': epoch,
            'genre_acc': logs.get('val_genre_output_accuracy', 0),
            'emotion_acc': logs.get('val_emotion_output_accuracy', 0),
            'timestamp': time.time()
        }
        
        self.history.append(metrics)
        
        # 성능 경고
        if metrics['genre_acc'] < self.threshold:
            print(f"경고: 낮은 장르 분류 정확도 ({metrics['genre_acc']:.2f})")
        if metrics['emotion_acc'] < self.threshold:
            print(f"경고: 낮은 감정 분류 정확도 ({metrics['emotion_acc']:.2f})")
        
        # 학습 진행 상황 저장
        self.save_history()
    
    def save_history(self):
        with open('training_history.json', 'w') as f:
            json.dump(self.history, f, indent=2)

def preprocess_audio(audio_path, sr=22050):
    try:
        # 오디오 로드
        y, sr = librosa.load(audio_path, sr=sr)
        
        # 노이즈 제거
        y = librosa.effects.preemphasis(y)
        
        # 무음 구간 제거
        y, _ = librosa.effects.trim(y, top_db=20)
        
        # 볼륨 정규화
        y = librosa.util.normalize(y)
        
        return y, sr
    except Exception as e:
        print(f"전처리 중 오류 발생: {str(e)}")
        raise

def create_ensemble_model(models, input_shape):
    inputs = layers.Input(shape=input_shape)
    
    genre_outputs = []
    emotion_outputs = []
    
    for model in models:
        g_out, e_out = model(inputs)
        genre_outputs.append(g_out)
        emotion_outputs.append(e_out)
    
    # 앙상블 결과 평균
    genre_output = layers.Average(name='genre_output')(genre_outputs)
    emotion_output = layers.Average(name='emotion_output')(emotion_outputs)
    
    ensemble_model = models.Model(inputs=inputs, outputs=[genre_output, emotion_output])
    return ensemble_model

def configure_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
        except RuntimeError as e:
            print(f"GPU 설정 오류: {e}")

def analyze_prediction(model, audio_path, genres, emotions):
    features = extract_audio_features(audio_path)
    predictions = model.predict(features)
    
    # 특성 중요도 분석
    feature_importance = {
        'mel_spectrogram': np.mean(features['mel_spectrogram']),
        'tempo': features['tempo'],
        'chromagram': np.mean(features['chromagram']),
        'mfcc': np.mean(features['mfcc'])
    }
    
    return {
        'predictions': predictions,
        'feature_importance': feature_importance
    }

class FeatureCache:
    def __init__(self, cache_dir='./feature_cache'):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def get_features(self, audio_path):
        cache_path = os.path.join(self.cache_dir, 
                                f"{hashlib.md5(audio_path.encode()).hexdigest()}.npy")
        
        if os.path.exists(cache_path):
            return np.load(cache_path, allow_pickle=True).item()
        
        features = extract_audio_features(audio_path)
        np.save(cache_path, features)
        return features

def validate_audio_file(audio_path):
    try:
        # 파일 존재 여부 확인
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {audio_path}")
            
        # 파일 크기 검사
        file_size = os.path.getsize(audio_path)
        if file_size == 0:
            raise ValueError("빈 파일입니다")
        if file_size > 100 * 1024 * 1024:  # 100MB 제한
            raise ValueError("파일이 너무 큽니다")
            
        # 파일 형식 검사
        valid_extensions = ['.mp3', '.wav', '.m4a', '.flac']
        if not any(audio_path.lower().endswith(ext) for ext in valid_extensions):
            raise ValueError("지원하지 않는 파일 형식입니다")
            
        return True
    except Exception as e:
        print(f"파일 검증 실패: {str(e)}")
        return False

def backup_model(model, backup_dir='./model_backups'):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(backup_dir, f'model_backup_{timestamp}')
    os.makedirs(backup_path, exist_ok=True)
    
    # 모델 저장
    model.save(os.path.join(backup_path, 'model.h5'))
    
    # 설정 저장
    config = {
        'timestamp': timestamp,
        'architecture': model.get_config(),
        'weights_hash': hashlib.md5(str(model.get_weights()).encode()).hexdigest()
    }
    
    with open(os.path.join(backup_path, 'config.json'), 'w') as f:
        json.dump(config, f)

def monitor_memory_usage():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    return {
        'rss': memory_info.rss / 1024 / 1024,  # MB
        'vms': memory_info.vms / 1024 / 1024,  # MB
        'percent': process.memory_percent()
    }

def run_model_tests(model, test_data_dir):
    test_results = {
        'passed': [],
        'failed': []
    }
    
    try:
        # 기본 예측 테스트
        test_file = os.path.join(test_data_dir, 'test_sample.mp3')
        prediction = predict_music(model, test_file, genres, emotions)
        test_results['passed'].append('기본 예측 테스트')
        
        # 메모리 누수 테스트
        initial_memory = monitor_memory_usage()
        for _ in range(10):
            predict_music(model, test_file, genres, emotions)
        final_memory = monitor_memory_usage()
        
        if (final_memory['rss'] - initial_memory['rss']) < 100:  # 100MB 이하 증가 허용
            test_results['passed'].append('메모리 누수 테스트')
        else:
            test_results['failed'].append('메모리 누수 테스트')
            
    except Exception as e:
        test_results['failed'].append(f'테스트 실패: {str(e)}')
    
    return test_results

# ------------------- 추가 기능 -------------------

def manual_classification(audio_path, genres, emotions, input_func=input):
    """사용자가 직접 장르와 감정을 입력하여 분류합니다."""
    print(f"수동 분류 대상 파일: {audio_path}")
    genre = input_func(f"장르를 선택하세요 {genres}: ")
    emotion = input_func(f"감정을 선택하세요 {emotions}: ")
    return {'genre': genre, 'emotion': emotion}


def rule_based_classification(audio_path):
    """단순 규칙 기반 분류 예시."""
    features = extract_audio_features(audio_path)
    tempo = features['tempo']

    if tempo < 80:
        genre = '클래식'
    elif tempo < 110:
        genre = '재즈'
    elif tempo < 140:
        genre = '팝'
    else:
        genre = '록'

    if tempo < 80:
        emotion = '평화로운'
    elif tempo < 120:
        emotion = '행복한'
    else:
        emotion = '열정적인'

    return {'genre': genre, 'emotion': emotion}


def train_traditional_ml_model(feature_list, labels):
    """전통적인 머신 러닝 모델(SVM) 학습."""
    from sklearn.svm import SVC
    X = [
        np.concatenate([
            f['mfcc'],
            f['spectral_contrast'],
            [f['tempo']]
        ])
        for f in feature_list
    ]
    model = SVC(probability=True)
    model.fit(X, labels)
    return model


def predict_traditional_ml_model(model, audio_path):
    features = extract_audio_features(audio_path)
    X = np.concatenate([
        features['mfcc'],
        features['spectral_contrast'],
        [features['tempo']]
    ]).reshape(1, -1)
    return model.predict(X)[0]


def train_lyrics_model(lyrics_list, labels):
    """가사 분석을 위한 간단한 텍스트 분류 모델."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(lyrics_list)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, labels)
    return vectorizer, clf


def predict_lyrics(vectorizer, model, lyrics):
    X = vectorizer.transform([lyrics])
    return model.predict(X)[0]


def hybrid_classification(cnn_model, ml_model, vec, lyr_model, audio_path, lyrics,
                          genres, emotions):
    """딥러닝, 규칙 기반, 가사 분석을 결합한 하이브리드 분류."""
    cnn_result = predict_music(cnn_model, audio_path, genres, emotions)
    cnn_genre = cnn_result['genres'][0][0] if cnn_result['genres'] else None
    cnn_emotion = cnn_result['emotions'][0][0] if cnn_result['emotions'] else None

    rule_result = rule_based_classification(audio_path)
    ml_genre = predict_traditional_ml_model(ml_model, audio_path)

    votes_genre = [g for g in [cnn_genre, rule_result['genre'], ml_genre] if g]
    final_genre = max(set(votes_genre), key=votes_genre.count)

    emotion_votes = [cnn_emotion, rule_result['emotion']]
    if lyrics and vec and lyr_model:
        emotion_votes.append(predict_lyrics(vec, lyr_model, lyrics))
    emotion_votes = [e for e in emotion_votes if e]
    final_emotion = max(set(emotion_votes), key=emotion_votes.count)

    return {'genre': final_genre, 'emotion': final_emotion}


class ModelVersionControl:
    def __init__(self, version_dir='./model_versions'):
        self.version_dir = version_dir
        os.makedirs(version_dir, exist_ok=True)
        
    def save_version(self, model, version_info):
        version_path = os.path.join(
            self.version_dir, 
            f"v{version_info['version']}_{time.strftime('%Y%m%d')}"
        )
        os.makedirs(version_path, exist_ok=True)
        
        # 모델 저장
        model.save(os.path.join(version_path, 'model.h5'))
        
        # 버전 정보 저장
        version_info.update({
            'timestamp': datetime.now().isoformat(),
            'model_hash': hashlib.md5(str(model.get_weights()).encode()).hexdigest()
        })
        
        with open(os.path.join(version_path, 'version_info.json'), 'w') as f:
            json.dump(version_info, f, indent=2)

class DatasetValidator:
    def __init__(self, min_samples=50, min_duration=10):
        self.min_samples = min_samples
        self.min_duration = min_duration
        self.validation_results = {
            'passed': [],
            'warnings': [],
            'errors': []
        }
    
    def validate_dataset(self, data_path, genres, emotions):
        try:
            # 데이터셋 구조 검증
            self._validate_directory_structure(data_path, genres)
            
            # 각 장르별 샘플 수 검증
            self._validate_sample_counts(data_path, genres)
            
            # 오디오 파일 품질 검증
            self._validate_audio_quality(data_path)
            
            # 클래스 밸런스 검증
            self._validate_class_balance(data_path, genres)
            
            return self.validation_results
            
        except Exception as e:
            self.validation_results['errors'].append(f"검증 중 오류 발생: {str(e)}")
            return self.validation_results
    
    def _validate_directory_structure(self, data_path, genres):
        for genre in genres:
            genre_path = os.path.join(data_path, genre)
            if not os.path.exists(genre_path):
                self.validation_results['errors'].append(f"장르 폴더 없음: {genre}")
            elif not os.path.isdir(genre_path):
                self.validation_results['errors'].append(f"잘못된 장르 폴더 형식: {genre}")
    
    def _validate_sample_counts(self, data_path, genres):
        for genre in genres:
            genre_path = os.path.join(data_path, genre)
            if os.path.exists(genre_path):
                samples = len([f for f in os.listdir(genre_path) 
                             if f.endswith(('.mp3', '.wav', '.m4a', '.flac'))])
                if samples < self.min_samples:
                    self.validation_results['warnings'].append(
                        f"장르 '{genre}'의 샘플 수가 부족합니다. (현재: {samples}, 필요: {self.min_samples})"
                    )
    
    def _validate_audio_quality(self, data_path):
        for root, _, files in os.walk(data_path):
            for file in files:
                if file.endswith(('.mp3', '.wav', '.m4a', '.flac')):
                    file_path = os.path.join(root, file)
                    try:
                        y, sr = librosa.load(file_path, duration=5)  # 첫 5초만 검사
                        if len(y) == 0:
                            self.validation_results['errors'].append(f"빈 오디오 파일: {file_path}")
                        if sr < 22050:
                            self.validation_results['warnings'].append(
                                f"낮은 샘플링 레이트: {file_path} ({sr}Hz)"
                            )
                    except Exception as e:
                        self.validation_results['errors'].append(
                            f"파일 로드 실패: {file_path} - {str(e)}"
                        )
    
    def _validate_class_balance(self, data_path, genres):
        sample_counts = {}
        for genre in genres:
            genre_path = os.path.join(data_path, genre)
            if os.path.exists(genre_path):
                sample_counts[genre] = len([f for f in os.listdir(genre_path) 
                                          if f.endswith(('.mp3', '.wav', '.m4a', '.flac'))])
        
        if sample_counts:
            avg_samples = sum(sample_counts.values()) / len(sample_counts)
            for genre, count in sample_counts.items():
                if count < avg_samples * 0.5:
                    self.validation_results['warnings'].append(
                        f"장르 '{genre}'의 샘플 수가 평균({avg_samples:.0f})의 50% 미만입니다: {count}"
                    )

# 사용 예시
    validator = DatasetValidator(min_samples=50, min_duration=10)
    validation_results = validator.validate_dataset('음악_데이터_경로', genres, emotions)

    if validation_results['errors']:
        print("\n오류:")
        for error in validation_results['errors']:
            print(f"- {error}")

    if validation_results['warnings']:
        print("\n경고:")
        for warning in validation_results['warnings']:
            print(f"- {warning}")

    if validation_results['passed']:
        print("\n통과한 검증:")
        for passed in validation_results['passed']:
            print(f"- {passed}")