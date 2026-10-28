"""Private serial JSON-command worker with atomic file responses."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def atomic_json(path, value):
    temporary = path.with_suffix('.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def serve(args):
    from music_classification.inference_engines import MuQEngine, Music2EmoEngine
    if args.kind == 'muq_shared':
        from music_classification.shared_muq_engine import SharedMuQEngine
        engine_class = SharedMuQEngine
    elif args.kind == 'muq_genre':
        from music_classification.muq_genre_engine import MuQGenreEngine
        engine_class = MuQGenreEngine
    else:
        engine_class = MuQEngine if args.kind == 'muq' else Music2EmoEngine
    engine = engine_class(args.base.resolve(), args.job.resolve(), args.cache.resolve())
    requests_root = args.requests_root.resolve()
    processed = 0
    for line in sys.stdin:
        command = json.loads(line)
        output = Path(command['output']).resolve()
        if (output.name != 'result.json' or output.parent.parent != requests_root
                or not output.parent.name.startswith('request-') or not output.parent.is_dir()):
            raise ValueError('Unsafe batch response path')
        try:
            payload = (engine.predict(Path(command['audio']), axis=command.get('axis')) if args.kind == 'muq_shared'
                       else engine.predict(Path(command['audio'])))
        except Exception as exc:
            payload = {'status': 'error', 'error': f'{type(exc).__name__}: {exc}'}
        processed += 1
        stats = {'model_loaded': engine.loaded, 'processed': processed}
        if args.kind in ('muq', 'muq_genre', 'muq_shared') and engine.loaded:
            import torch
            if torch.cuda.is_available():
                stats['cuda_memory'] = {'allocated_bytes': torch.cuda.memory_allocated(),
                                       'reserved_bytes': torch.cuda.memory_reserved(),
                                       'peak_allocated_bytes': torch.cuda.max_memory_allocated()}
        atomic_json(output.with_name('worker-stats.json'), stats)
        atomic_json(output, payload)
        # Do not reuse potentially contaminated model state after inference errors.
        if payload['status'] == 'error':
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', required=True, choices=['muq', 'music2emo', 'muq_genre', 'muq_shared'])
    for option in ('base', 'job', 'cache', 'requests-root'):
        parser.add_argument('--' + option, required=True, type=Path)
    serve(parser.parse_args())
