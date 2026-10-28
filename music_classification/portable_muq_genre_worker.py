"""One-shot genre worker; uses the same engine as genre batch sessions."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from music_classification.muq_genre_engine import MuQGenreEngine


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('base', 'job', 'audio', 'cache'):
        parser.add_argument('--' + option, required=True, type=Path)
    args = parser.parse_args()
    try:
        payload = MuQGenreEngine(args.base.resolve(), args.job.resolve(), args.cache.resolve()).predict(args.audio.resolve())
    except Exception as exc:
        payload = {'status': 'error', 'error': f'{type(exc).__name__}: {exc}'}
    (args.job / 'result.json').write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False), encoding='utf-8')
