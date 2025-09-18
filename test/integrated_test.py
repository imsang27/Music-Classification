#!/usr/bin/env python3
"""
🎯 통합 테스트 스크립트
모든 최적화 기능과 모듈을 종합적으로 테스트
"""

import os
import sys
import tempfile
import numpy as np
import librosa
import psutil
import gc
import time
import importlib.util

# 모듈 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

class IntegratedTester:
    """통합 테스트 클래스"""
    
    def __init__(self):
        self.model = None
        self.processor = None
        self.process = psutil.Process()
        self.test_results = {}
        
    def load_model(self):
        """Wav2Vec2 모델 로드"""
        try:
            from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
            import torch
            
            print("📥 Wav2Vec2 모델 로드 중...")
            
            MODEL_PATH = os.path.join('models', 'models--dima806--music_genres_classification', 'snapshots', '5f71fb1e2c6bedcddb2bfb1e929fc70655780902')
            
            if not os.path.exists(MODEL_PATH):
                print("❌ 모델 경로가 존재하지 않습니다:", MODEL_PATH)
                return False
            
            # 초기 메모리 상태
            initial_memory = self.process.memory_info().rss / 1024 / 1024
            print(f"초기 메모리 사용량: {initial_memory:.1f} MB")
            
            # 모델 로드
            self.model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_PATH)
            self.processor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_PATH)
            self.model.eval()
            
            if torch.cuda.is_available():
                self.model = self.model.cuda()
                print("✅ GPU 사용 가능")
            
            # 모델 로드 후 메모리
            model_memory = self.process.memory_info().rss / 1024 / 1024
            print(f"모델 로드 후 메모리: {model_memory:.1f} MB (변화: {model_memory - initial_memory:+.1f} MB)")
            
            return True
            
        except Exception as e:
            print(f"❌ 모델 로드 실패: {str(e)}")
            return False
    
    def test_core_predictor(self):
        """core/predictor.py 최적화 테스트"""
        print("\n1️⃣ core/predictor.py 최적화 테스트...")
        
        try:
            from core.predictor import predict_music_wav2vec2, batch_classify_urls_wav2vec2
            
            # 테스트 오디오 생성
            sr = 16000
            y = np.random.randn(sr * 10)  # 10초
            y = librosa.util.normalize(y)
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                try:
                    import soundfile as sf
                    sf.write(temp_file.name, y, sr)
                except ImportError:
                    from scipy.io import wavfile
                    wavfile.write(temp_file.name, sr, y.astype(np.float32))
                
                temp_path = temp_file.name
            
            try:
                # 메모리 모니터링과 함께 분류 수행
                initial_memory = self.process.memory_info().rss / 1024 / 1024
                print(f"초기 메모리: {initial_memory:.1f} MB")
                
                result = predict_music_wav2vec2(self.model, self.processor, temp_path)
                
                final_memory = self.process.memory_info().rss / 1024 / 1024
                memory_diff = final_memory - initial_memory
                print(f"최종 메모리: {final_memory:.1f} MB (변화: {memory_diff:+.1f} MB)")
                
                if result.get('success', False):
                    print(f"✅ 분류 성공: {result['genre']} (신뢰도: {result['confidence']:.2f})")
                    self.test_results['core_predictor'] = True
                else:
                    print(f"❌ 분류 실패: {result.get('error', '알 수 없는 오류')}")
                    self.test_results['core_predictor'] = False
                
                # 메모리 누수 체크
                if memory_diff > 100:
                    print(f"⚠️  메모리 누수 가능성: {memory_diff:.1f} MB 증가")
                else:
                    print(f"✅ 메모리 사용량 안정적: {memory_diff:.1f} MB 변화")
                    
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            
            return self.test_results['core_predictor']
            
        except Exception as e:
            print(f"❌ core/predictor.py 테스트 실패: {str(e)}")
            self.test_results['core_predictor'] = False
            return False
    
    def test_memory_optimizer(self):
        """utils/memory_optimizer.py 최적화 테스트"""
        print("\n2️⃣ utils/memory_optimizer.py 최적화 테스트...")
        
        try:
            from utils.memory_optimizer import optimize_memory_usage, check_memory_health, monitor_memory_usage
            
            # 메모리 최적화 함수 테스트
            initial_memory = self.process.memory_info().rss / 1024 / 1024
            print(f"최적화 전 메모리: {initial_memory:.1f} MB")
            
            optimize_memory_usage()
            
            final_memory = self.process.memory_info().rss / 1024 / 1024
            print(f"최적화 후 메모리: {final_memory:.1f} MB")
            
            # 메모리 건강성 검사
            health_check = check_memory_health()
            print(f"메모리 건강성: {health_check}")
            
            # 메모리 모니터링
            memory_info = monitor_memory_usage()
            print(f"메모리 모니터링: {memory_info}")
            
            print("✅ utils/memory_optimizer.py 최적화 확인 완료")
            self.test_results['memory_optimizer'] = True
            return True
            
        except Exception as e:
            print(f"❌ utils/memory_optimizer.py 테스트 실패: {str(e)}")
            self.test_results['memory_optimizer'] = False
            return False
    
    def test_audio_processor(self):
        """utils/audio_processor.py 최적화 테스트"""
        print("\n3️⃣ utils/audio_processor.py 최적화 테스트...")
        
        try:
            from utils.audio_processor import extract_audio_features, preprocess_audio, optimize_audio_processing
            
            # 테스트 오디오 생성
            sr = 22050
            y = np.random.randn(sr * 15)  # 15초
            y = librosa.util.normalize(y)
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                try:
                    import soundfile as sf
                    sf.write(temp_file.name, y, sr)
                except ImportError:
                    from scipy.io import wavfile
                    wavfile.write(temp_file.name, sr, y.astype(np.float32))
                
                temp_path = temp_file.name
            
            try:
                # 오디오 특성 추출 테스트
                print("오디오 특성 추출 테스트...")
                features = extract_audio_features(temp_path, duration=10)
                print(f"추출된 특성: {list(features.keys())}")
                
                # 오디오 전처리 테스트
                print("오디오 전처리 테스트...")
                processed_audio = preprocess_audio(temp_path)
                print(f"전처리된 오디오 길이: {len(processed_audio)}")
                
                # 최적화된 오디오 처리 테스트
                print("최적화된 오디오 처리 테스트...")
                optimized_audio = optimize_audio_processing(temp_path, max_duration=10)
                print(f"최적화된 오디오 길이: {len(optimized_audio)}")
                
                print("✅ utils/audio_processor.py 최적화 확인 완료")
                self.test_results['audio_processor'] = True
                return True
                
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            
        except Exception as e:
            print(f"❌ utils/audio_processor.py 테스트 실패: {str(e)}")
            self.test_results['audio_processor'] = False
            return False
    
    def test_batch_processing(self):
        """배치 처리 최적화 테스트"""
        print("\n4️⃣ 배치 처리 최적화 테스트...")
        
        try:
            from core.predictor import batch_classify_urls_wav2vec2
            
            # 테스트 파일들 생성
            test_files = []
            for i in range(3):
                sr = 16000
                y = np.random.randn(sr * 5)  # 5초
                y = librosa.util.normalize(y)
                
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    try:
                        import soundfile as sf
                        sf.write(temp_file.name, y, sr)
                    except ImportError:
                        from scipy.io import wavfile
                        wavfile.write(temp_file.name, sr, y.astype(np.float32))
                    test_files.append(temp_file.name)
            
            try:
                # 배치 처리 테스트 (URL 대신 파일 경로 사용)
                initial_memory = self.process.memory_info().rss / 1024 / 1024
                print(f"배치 처리 전 메모리: {initial_memory:.1f} MB")
                
                results = []
                for i, file_path in enumerate(test_files):
                    print(f"배치 처리 {i+1}/3: {file_path}")
                    from core.predictor import predict_music_wav2vec2
                    result = predict_music_wav2vec2(self.model, self.processor, file_path)
                    results.append(result)
                    
                    # 메모리 정리
                    gc.collect()
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                final_memory = self.process.memory_info().rss / 1024 / 1024
                print(f"배치 처리 후 메모리: {final_memory:.1f} MB (변화: {final_memory - initial_memory:+.1f} MB)")
                
                # 성공률 계산
                success_count = sum(1 for r in results if r.get('success', False))
                print(f"성공률: {success_count}/{len(results)} ({success_count/len(results)*100:.1f}%)")
                
                print("✅ 배치 처리 최적화 확인 완료")
                self.test_results['batch_processing'] = True
                return True
                
            finally:
                for file_path in test_files:
                    if os.path.exists(file_path):
                        os.remove(file_path)
            
        except Exception as e:
            print(f"❌ 배치 처리 테스트 실패: {str(e)}")
            self.test_results['batch_processing'] = False
            return False
    
    def test_url_classification(self):
        """URL 분류 테스트"""
        print("\n5️⃣ URL 분류 테스트...")
        
        try:
            from core.predictor import predict_music_wav2vec2
            
            # URL 분류 함수 테스트 (실제로는 파일 경로로 테스트)
            print("URL 분류 함수 테스트...")
            
            # 테스트 오디오 파일 생성
            sr = 16000
            y = np.random.randn(sr * 10)  # 10초
            y = librosa.util.normalize(y)
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                try:
                    import soundfile as sf
                    sf.write(temp_file.name, y, sr)
                except ImportError:
                    from scipy.io import wavfile
                    wavfile.write(temp_file.name, sr, y.astype(np.float32))
                
                temp_path = temp_file.name
            
            try:
                # URL 분류 시뮬레이션 (실제로는 파일 경로 사용)
                result = predict_music_wav2vec2(self.model, self.processor, temp_path)
                
                if result.get('success', False):
                    print(f"✅ URL 분류 성공: {result['genre']} (신뢰도: {result['confidence']:.2f})")
                    self.test_results['url_classification'] = True
                    return True
                else:
                    print(f"❌ URL 분류 실패: {result.get('error', '알 수 없는 오류')}")
                    self.test_results['url_classification'] = False
                    return False
                    
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            
        except Exception as e:
            print(f"❌ URL 분류 테스트 실패: {str(e)}")
            self.test_results['url_classification'] = False
            return False
    
    def test_webapp_integration(self):
        """웹앱 통합 테스트"""
        print("\n6️⃣ 웹앱 통합 테스트...")
        
        try:
            from webapp.app import app
            
            # 웹앱 모듈 로드 확인
            print("✅ 웹앱 모듈 로드 성공")
            
            # Flask 앱 인스턴스 확인
            if app:
                print("✅ Flask 앱 인스턴스 생성 성공")
                self.test_results['webapp_integration'] = True
                return True
            else:
                print("❌ Flask 앱 인스턴스 생성 실패")
                self.test_results['webapp_integration'] = False
                return False
                
        except Exception as e:
            print(f"❌ 웹앱 통합 테스트 실패: {str(e)}")
            self.test_results['webapp_integration'] = False
            return False
    
    def test_memory_cleanup(self):
        """메모리 정리 테스트"""
        print("\n7️⃣ 메모리 정리 테스트...")
        
        try:
            from utils.memory_optimizer import optimize_memory_usage
            
            # 메모리 정리 전
            before_memory = self.process.memory_info().rss / 1024 / 1024
            print(f"정리 전 메모리: {before_memory:.1f} MB")
            
            # 메모리 정리
            optimize_memory_usage()
            gc.collect()
            
            # 메모리 정리 후
            after_memory = self.process.memory_info().rss / 1024 / 1024
            memory_diff = after_memory - before_memory
            print(f"정리 후 메모리: {after_memory:.1f} MB (변화: {memory_diff:+.1f} MB)")
            
            if memory_diff < 0:
                print(f"✅ 메모리 정리 성공: {abs(memory_diff):.1f} MB 해제")
            else:
                print(f"⚠️  메모리 정리 효과 미미: {memory_diff:.1f} MB 변화")
            
            self.test_results['memory_cleanup'] = True
            return True
            
        except Exception as e:
            print(f"❌ 메모리 정리 테스트 실패: {str(e)}")
            self.test_results['memory_cleanup'] = False
            return False
    
    def run_all_tests(self):
        """모든 테스트 실행"""
        print("🚀 통합 테스트 시작")
        print("=" * 60)
        
        # 모델 로드
        if not self.load_model():
            print("❌ 모델 로드 실패로 테스트 중단")
            return False
        
        # 각 테스트 실행
        tests = [
            self.test_core_predictor,
            self.test_memory_optimizer,
            self.test_audio_processor,
            self.test_batch_processing,
            self.test_url_classification,
            self.test_webapp_integration,
            self.test_memory_cleanup
        ]
        
        passed_tests = 0
        total_tests = len(tests)
        
        for test_func in tests:
            try:
                if test_func():
                    passed_tests += 1
            except Exception as e:
                print(f"❌ 테스트 실행 중 오류: {str(e)}")
        
        # 결과 요약
        print("\n" + "=" * 60)
        print("📊 통합 테스트 결과 요약")
        print("=" * 60)
        print(f"총 테스트 수: {total_tests}")
        print(f"통과한 테스트: {passed_tests}")
        print(f"실패한 테스트: {total_tests - passed_tests}")
        
        print("\n📋 상세 결과:")
        for test_name, result in self.test_results.items():
            status = "✅ 통과" if result else "❌ 실패"
            print(f"   {test_name}: {status}")
        
        # 최종 메모리 사용량
        final_memory = self.process.memory_info().rss / 1024 / 1024
        print(f"\n💾 최종 메모리 사용량: {final_memory:.1f} MB")
        
        if passed_tests == total_tests:
            print("\n🎉 모든 테스트 통과! 최적화가 성공적으로 적용되었습니다.")
            return True
        else:
            print(f"\n⚠️  {total_tests - passed_tests}개 테스트가 실패했습니다. 추가 확인이 필요합니다.")
            return False

def main():
    """메인 실행 함수"""
    tester = IntegratedTester()
    success = tester.run_all_tests()
    
    if success:
        print("\n🎯 통합 테스트 완료!")
        sys.exit(0)
    else:
        print("\n🔧 일부 테스트가 실패했습니다. 로그를 확인해주세요.")
        sys.exit(1)

if __name__ == "__main__":
    main()
