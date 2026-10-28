"""Worker-local models reused serially; application code never imports them."""

from copy import deepcopy
import importlib.metadata
import os
from pathlib import Path
import sys

from .muq_runtime import digest


class BaseEngine:
    def __init__(self, base, job, cache):
        self.base, self.job, self.cache = map(Path, (base, job, cache))
        self.loaded = False
        self.cache.mkdir(parents=True, exist_ok=True)
        os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HOME=str(self.cache / 'hf-home'),
                          NUMBA_CACHE_DIR=str(self.cache / 'numba-cache'))

    def _capture_rng(self):
        import numpy as np
        import torch
        self._numpy_rng = np.random.get_state()
        self._torch_rng = torch.get_rng_state()
        self._cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    def _restore_rng(self):
        import numpy as np
        import torch
        np.random.set_state(self._numpy_rng)
        torch.set_rng_state(self._torch_rng)
        if self._cuda_rng is not None:
            torch.cuda.set_rng_state_all(self._cuda_rng)

    def predict(self, path):
        import numpy as np
        import soundfile as sf
        path = Path(path).resolve()
        try:
            if not path.is_file() or not 0 < path.stat().st_size <= 100 * 1024 * 1024:
                raise ValueError('Audio missing, empty, or exceeds100 MB')
            original_hash = digest(path)
            decoded, sr = sf.read(path, dtype='float32')
            if len(decoded) < sr or not np.isfinite(decoded).all() or np.max(np.abs(decoded)) <= 1e-7:
                raise ValueError('At least1 second of finite, non-silent audio required')
        except Exception as exc:
            return {'status': 'invalid_audio', 'error': f'Audio validation failed: {exc}'}
        if not self.loaded:
            self._load()
            self._capture_rng()
            self.loaded = True
        # Reproduce a fresh worker's post-initialization RNG state for each item.
        self._restore_rng()
        result = self._infer(path, decoded, sr, original_hash)
        if digest(path) != original_hash:
            return {'status': 'invalid_audio', 'error': 'Audio changed during inference; no prediction'}
        return result


class MuQEngine(BaseEngine):
    def _load(self, *, shared=None):
        import numpy as np
        import torch
        import librosa
        from .portable_runtime import load_muq, resource_json
        torch.set_num_threads(4)
        if shared is None:
            torch.manual_seed(20261025)
            np.random.seed(20261025)
        self.prompt_sha = 'aadf7a046ee96ec2c77e4427af0cc6c63ca3df4b145578397df57233f9104ec0'
        self.taxonomy_sha = 'a74d6d7544c9d4afb4d0bd360f7ac533f1e61e53c984c80e5d0c82f068459bc8'
        config = resource_json('clap_prompts.json', self.prompt_sha)
        resource_json('classification_taxonomy.json', self.taxonomy_sha)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if shared is None:
            self.model, self.info = load_muq(self.base, self.job / 'runtime', self.device)
        else:
            self.model, info = shared
            self.info = deepcopy(info)
        prompts = config['prompts']['emotion']
        self.labels = list(prompts)
        with torch.inference_mode():
            self.text = self.model(texts=list(prompts.values()))
        if self.text.shape != (8, 512) or not torch.isfinite(self.text).all():
            raise ValueError('Invalid MuQ text embeddings')
        self.info['prompt_sha256'], self.info['taxonomy_sha256'] = self.prompt_sha, self.taxonomy_sha
        self.info['packages'] = {name: importlib.metadata.version(name) for name in
                                ('torch', 'torchaudio', 'transformers', 'numpy', 'librosa', 'einops', 'x-clip')}

    def _infer(self, path, decoded, original_sr, original_hash):
        import numpy as np
        import torch
        import librosa
        from .muq import validated_scores
        audio, sr = librosa.load(str(path), sr=24000)
        if len(audio) < sr or not np.isfinite(audio).all():
            raise ValueError('Invalid resampled audio')
        segments = np.array_split(np.arange(len(audio)), int(np.ceil(len(audio) / 240000)))
        bounds, similarities = [], []
        with torch.inference_mode():
            for segment in segments:
                start, end = int(segment[0]), int(segment[-1]) + 1
                wave = torch.as_tensor(audio[start:end], device=self.device).unsqueeze(0)
                if wave.shape[-1] < 240000:
                    wave = wave.repeat(1, int(np.ceil(240000 / wave.shape[-1])))[:, :240000]
                latent = self.model(wavs=wave, parallel_processing=False)
                similarities.append(self.model.calc_similarity(latent, self.text)[0].cpu())
                bounds.append([start / sr, end / sr])
            scores = validated_scores(dict(zip(self.labels, torch.stack(similarities).mean(0).tolist())))
        return {'status': 'ok', 'scores': scores, 'model': deepcopy(self.info),
                'analysis': {'sample_rate': sr, 'input_sample_rate': original_sr,
                    'duration_seconds': len(decoded) / original_sr, 'audio_sha256': original_hash,
                    'strategy': 'balanced_full_audio_windows_mean_native_similarity', 'segments_seconds': bounds,
                    'omitted_tail_seconds': 0., 'padding': 'repeat segment prefix to10s before native wrapper'}}


class Music2EmoEngine(BaseEngine):
    def _load(self):
        import json
        import numpy as np
        import torch
        from .music2emo_worker import PINS, verified_file
        from .portable_runtime import derive_runtime
        for name, value in {'HF_MODULES_CACHE': self.cache / 'hf-modules',
                            'MPLCONFIGDIR': self.cache / 'matplotlib',
                            'MUSIC_CLASSIFICATION_MERT_PATH': self.base / 'mert'}.items():
            os.environ[name] = str(value)
        manifests = {name: json.loads(verified_file(self.base, name, sha).read_text(encoding='utf-8'))
                     for name, sha in PINS.items()}
        runtime = self.job / 'runtime'
        relocation = derive_runtime(self.base / 'runtime-modefix', manifests['runtime-patches-modefix.json']['files_sha256'], runtime, kind='music2emo')
        for item in manifests['mert-source.json']['files']:
            verified_file(self.base / 'mert', item['path'], item['sha256'])
        verified_file(self.base / 'mert', 'pytorch_model.bin', manifests['mert-weights-source.json']['sha256'])
        self.labels = [str(t).replace('mood/theme---', '') for t in np.load(runtime / 'inference/data/tag_list.npy', allow_pickle=False)[127:]]
        os.chdir(runtime)
        sys.path.insert(0, str(runtime))
        torch.set_num_threads(4)
        torch.manual_seed(20261025)
        np.random.seed(20261025)
        from music2emo import Music2emo
        self.model = Music2emo()
        self.info = {'name': 'Music2Emotion/J_all', 'experimental': True,
            'upstream_revision': 'c5f7984f78d9928b841f25bd05a47377c12c58ca',
            'mert_revision': manifests['mert-source.json']['revision'], 'artifact_manifests_sha256': PINS,
            'installation': str(self.base), 'runtime_relocation': relocation,
            'head_sha256': manifests['runtime-patches-modefix.json']['files_sha256']['saved_models/J_all.ckpt'],
            'mert_weights_sha256': manifests['mert-weights-source.json']['sha256'],
            'device': str(self.model.device), 'mode_feature_fix': True}

    def _infer(self, path, audio, sr, original_hash):
        import torch
        from .music2emo_results import validated_scores, segment_ranges
        with torch.inference_mode():
            raw = self.model.predict(path.as_posix())
        checked = validated_scores(raw, self.labels)
        return {'status': 'ok', 'labels': self.labels,
                'raw': {'raw_mood_scores': checked['native_scores'], 'valence': checked['valence'], 'arousal': checked['arousal']},
                'analysis': {'strategy': 'complete_audio_equal_mean_MERT_layers_5_6',
                    'duration_seconds': len(audio) / sr, 'input_sample_rate': sr,
                    'segments_seconds': [[start / sr, end / sr] for start, end in segment_ranges(len(audio), sr)],
                    'audio_sha256': original_hash}, 'model': deepcopy(self.info)}
