"""Private offline MuQ worker; no application imports of heavy model dependencies."""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys


def predict(base, audio_path):
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HOME=str(base / 'hf-home'),
                      NUMBA_CACHE_DIR=str(base / 'numba-cache'))
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from music_classification.muq_runtime import load_local_model, verified, digest
    from music_classification.muq import validated_scores
    import numpy as np
    import soundfile as sf
    try:
        decoded, original_sr = sf.read(audio_path, dtype='float32')
        if len(decoded) < original_sr or not np.isfinite(decoded).all() or np.max(np.abs(decoded)) <= 1e-7:
            raise ValueError('At least1 second of finite, non-silent audio required')
    except Exception as exc:
        return {'status': 'invalid_audio', 'error': f'Audio validation failed: {exc}'}
    original_hash = digest(audio_path)
    prompt_path = verified(root, 'config/clap_prompts.json', 'aadf7a046ee96ec2c77e4427af0cc6c63ca3df4b145578397df57233f9104ec0')
    taxonomy_path = verified(root, 'config/classification_taxonomy.json', 'a74d6d7544c9d4afb4d0bd360f7ac533f1e61e53c984c80e5d0c82f068459bc8')
    import torch
    import librosa
    torch.set_num_threads(4)
    torch.manual_seed(20261025)
    np.random.seed(20261025)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, info = load_local_model(base, device)
    prompts = json.loads(prompt_path.read_text(encoding='utf-8'))['prompts']['emotion']
    labels = list(prompts)
    with torch.inference_mode():
        text = model(texts=list(prompts.values()))
    if text.shape != (8, 512) or not torch.isfinite(text).all():
        raise ValueError('Invalid MuQ text embeddings')
    audio, sr = librosa.load(str(audio_path), sr=24000)
    if len(audio) < sr or not np.isfinite(audio).all():
        raise ValueError('Invalid resampled audio')
    segments = np.array_split(np.arange(len(audio)), int(np.ceil(len(audio) / 240000)))
    bounds, similarities = [], []
    with torch.inference_mode():
        for segment in segments:
            start, end = int(segment[0]), int(segment[-1]) + 1
            wave = torch.as_tensor(audio[start:end], device=device).unsqueeze(0)
            if wave.shape[-1] < 240000:
                wave = wave.repeat(1, int(np.ceil(240000 / wave.shape[-1])))[:, :240000]
            latent = model(wavs=wave, parallel_processing=False)
            similarities.append(model.calc_similarity(latent, text)[0].cpu())
            bounds.append([start / sr, end / sr])
        scores = validated_scores(dict(zip(labels, torch.stack(similarities).mean(0).tolist())))
    info['prompt_sha256'], info['taxonomy_sha256'] = digest(prompt_path), digest(taxonomy_path)
    info['packages'] = {name: importlib.metadata.version(name) for name in ('torch', 'torchaudio', 'transformers', 'numpy', 'librosa', 'einops', 'x-clip')}
    if digest(audio_path) != original_hash:
        return {'status': 'invalid_audio', 'error': 'Audio changed during inference; no prediction'}
    return {'status': 'ok', 'scores': scores, 'model': info,
            'analysis': {'sample_rate': sr, 'input_sample_rate': original_sr,
                'duration_seconds': len(decoded) / original_sr, 'audio_sha256': original_hash,
                'strategy': 'balanced_full_audio_windows_mean_native_similarity', 'segments_seconds': bounds,
                'omitted_tail_seconds': 0., 'padding': 'repeat segment prefix to10s before native wrapper'}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('base', 'job', 'audio'):
        parser.add_argument('--' + option, required=True, type=Path)
    args = parser.parse_args()
    try:
        payload = predict(args.base.resolve(), args.audio.resolve())
    except Exception as exc:
        payload = {'status': 'error', 'error': f'{type(exc).__name__}: {exc}'}
    (args.job / 'result.json').write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False), encoding='utf-8')
