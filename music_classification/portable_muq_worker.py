"""One-shot entry point using the same worker-local engine as model-reuse batches."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from music_classification.inference_engines import MuQEngine


def predict(base, audio_path, job, cache):
    return MuQEngine(base, job, cache).predict(audio_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('base', 'job', 'audio', 'cache'):
        parser.add_argument('--' + option, required=True, type=Path)
    args = parser.parse_args()
    try:
        payload = predict(args.base.resolve(), args.audio.resolve(), args.job.resolve(), args.cache.resolve())
    except Exception as exc:
        payload = {'status': 'error', 'error': f'{type(exc).__name__}: {exc}'}
    (args.job / 'result.json').write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False), encoding='utf-8')
