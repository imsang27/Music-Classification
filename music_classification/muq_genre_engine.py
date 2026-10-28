"""Worker-local reproduction of the frozen common10 public-study protocol."""

from copy import deepcopy
import math

from .inference_engines import BaseEngine


def source_ranges(frames):
    if frames < 48000:
        raise ValueError('At least one second of canonical audio required')
    count = math.ceil(frames / 480000)
    quotient, remainder = divmod(frames, count)
    start, bounds = 0, []
    for i in range(count):
        end = start + quotient + (i < remainder)
        bounds.append([start, end])
        start = end
    return bounds


class MuQGenreEngine(BaseEngine):
    def _load(self, *, shared=None):
        import numpy as np
        import torch
        from .portable_runtime import load_muq
        from .muq_genre import prompt_spec, PROMPT_SHA256
        torch.set_num_threads(4)
        if shared is None:
            torch.manual_seed(20261025)
            np.random.seed(20261025)
        spec = prompt_spec()
        self.labels = spec['labels']
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if shared is None:
            self.model, self.info = load_muq(self.base, self.job / 'runtime', self.device)
        else:
            self.model, info = shared
            self.info = deepcopy(info)
        with torch.inference_mode():
            self.text = self.model(texts=list(spec['prompts'].values()))
        if self.text.shape != (10, 512) or not torch.isfinite(self.text).all():
            raise ValueError('Invalid genre text embeddings')
        self.info.update(genre_prompt_sha256=PROMPT_SHA256, genre_prompt_version=spec['version'])

    def _infer(self, path, decoded, original_sr, original_hash):
        import numpy as np
        import torch
        import librosa
        from .muq_genre import validated_scores
        audio = decoded.reshape(-1, 1).mean(axis=1) if decoded.ndim == 1 else decoded.mean(axis=1)
        frames = (len(audio) * 48000 + original_sr - 1) // original_sr
        if original_sr != 48000:
            audio = librosa.resample(audio, orig_sr=original_sr, target_sr=48000)
            audio = librosa.util.fix_length(audio, size=frames)
        if len(audio) != frames or not np.isfinite(audio).all():
            raise ValueError('Invalid canonical genre audio')
        bounds = source_ranges(frames)
        raw_scores = []
        with torch.inference_mode():
            for start, end in bounds:
                wave = librosa.resample(audio[start:end], orig_sr=48000, target_sr=24000)
                wave = np.tile(wave, math.ceil(240000 / len(wave)))[:240000]
                latent = self.model(wavs=torch.as_tensor(wave, device=self.device).unsqueeze(0), parallel_processing=False)
                raw_scores.append(self.model.calc_similarity(latent, self.text)[0].cpu().tolist())
        # Match the study's per-label float64 NumPy mean, not a float32 Torch mean.
        scores = validated_scores({label: float(np.mean([window[i] for window in raw_scores]))
                                   for i, label in enumerate(self.labels)})
        return {'status': 'ok', 'scores': scores, 'model': deepcopy(self.info),
                'analysis': {'audio_sha256': original_hash, 'input_sample_rate': original_sr,
                             'duration_seconds': len(decoded) / original_sr,
                             'canonical_sample_rate': 48000, 'canonical_frames': frames,
                             'model_sample_rate': 24000, 'source_sample_ranges_48k': bounds,
                             'segments_seconds': [[a / 48000, b / 48000] for a, b in bounds],
                             'strategy': 'common10_balanced_canonical_windows_mean_native_similarity',
                             'input_scope': 'complete_audio', 'omitted_tail_seconds': 0.,
                             'padding': 'repeat segment prefix to10s before native wrapper'}}
