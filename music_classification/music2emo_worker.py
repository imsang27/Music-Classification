"""Private subprocess entry point for the pinned local Music2Emo runtime."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

PINS = {
    'runtime-patches-modefix.json': '6e98c04c42a78f6ca7509c30cec9c4cb7fa3195b75546f91d9bcd8458286e6a4',
    'mert-source.json': 'd618d6fce1b2d46e9284cb4e0fbcb52172746c5004497641e422ba99b020d23c',
    'mert-weights-source.json': 'eaa1528756ac531fcf9345494c7aacee1b79fbe350724907e55c99b159698321',
}


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verified_file(root, name, sha):
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()) or digest(path) != sha:
        raise ValueError(f'Local runtime integrity check failed: {name}')
    return path


def predict(base, job, audio_path):
    # The reviewed upstream source embeds this absolute checkpoint path.
    # Never verify one installation's weights and silently load another's.
    reviewed_mert = Path('C:/Python/Music Classification/work/music2emo-experiment/mert').resolve()
    if (base / 'mert').resolve() != reviewed_mert:
        return {'status': 'unavailable', 'error': 'This pinned runtime requires its reviewed absolute MERT installation path'}
    for name, value in {'HF_MODULES_CACHE': base / 'hf-modules', 'HF_HOME': base / 'hf-home',
                        'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                        'MPLCONFIGDIR': base / 'matplotlib', 'NUMBA_CACHE_DIR': base / 'numba-cache'}.items():
        os.environ[name] = str(value)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import numpy as np
    import soundfile as sf
    from music_classification.music2emo_results import validated_scores, segment_ranges
    try:
        audio, sr = sf.read(audio_path, dtype='float32')
        if len(audio) < sr or not np.isfinite(audio).all() or np.max(np.abs(audio)) <= 1e-7:
            raise ValueError('Audio must contain at least one second of finite, non-silent samples')
    except Exception as exc:
        return {'status': 'invalid_audio', 'error': f'Audio validation failed: {exc}'}
    manifests = {name: json.loads(verified_file(base, name, sha).read_text(encoding='utf-8'))
                 for name, sha in PINS.items()}
    runtime = job / 'runtime'
    runtime.mkdir()
    for name, sha in manifests['runtime-patches-modefix.json']['files_sha256'].items():
        source = verified_file(base / 'runtime-modefix', name, sha)
        target = runtime / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        verified_file(runtime, name, sha)
    for item in manifests['mert-source.json']['files']:
        verified_file(base / 'mert', item['path'], item['sha256'])
    verified_file(base / 'mert', 'pytorch_model.bin', manifests['mert-weights-source.json']['sha256'])
    labels = [str(t).replace('mood/theme---', '') for t in
              np.load(runtime / 'inference/data/tag_list.npy', allow_pickle=False)[127:]]
    # Only the child changes cwd/sys.path. All upstream scratch data is private.
    os.chdir(runtime)
    sys.path.insert(0, str(runtime))
    import torch
    torch.set_num_threads(4)
    torch.manual_seed(20261025)
    np.random.seed(20261025)
    from music2emo import Music2emo
    model = Music2emo()
    with torch.inference_mode():
        raw = model.predict(audio_path.as_posix())
    checked = validated_scores(raw, labels)
    return {'status': 'ok', 'labels': labels,
            'raw': {'raw_mood_scores': checked['native_scores'], 'valence': checked['valence'], 'arousal': checked['arousal']},
            'analysis': {'strategy': 'complete_audio_equal_mean_MERT_layers_5_6',
                         'duration_seconds': len(audio) / sr, 'input_sample_rate': sr,
                         'segments_seconds': [[start / sr, end / sr] for start, end in segment_ranges(len(audio), sr)],
                         'audio_sha256': digest(audio_path)},
            'model': {'name': 'Music2Emotion/J_all', 'experimental': True,
                      'upstream_revision': 'c5f7984f78d9928b841f25bd05a47377c12c58ca',
                      'mert_revision': manifests['mert-source.json']['revision'],
                      'artifact_manifests_sha256': PINS,
                      'head_sha256': manifests['runtime-patches-modefix.json']['files_sha256']['saved_models/J_all.ckpt'],
                      'mert_weights_sha256': manifests['mert-weights-source.json']['sha256'],
                      'device': str(model.device), 'mode_feature_fix': True}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('base', 'job', 'audio'):
        parser.add_argument('--' + option, required=True, type=Path)
    args = parser.parse_args()
    try:
        output = predict(args.base.resolve(), args.job.resolve(), args.audio.resolve())
    except Exception as exc:
        output = {'status': 'error', 'error': f'{type(exc).__name__}: {exc}'}
    (args.job / 'result.json').write_text(json.dumps(output, ensure_ascii=False, allow_nan=False), encoding='utf-8')
