"""Original Wav2Vec2 genre inference through the common local web job API."""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from music_classification.settings import ModelSettings
from music_classification.muq_runtime import digest

NAMES = dict(blues='블루스', classical='클래식', country='컨트리', disco='디스코',
             hiphop='힙합', jazz='재즈', metal='메탈', pop='팝', reggae='레게', rock='록')


def status(settings_path):
    try:
        settings = ModelSettings.from_file(settings_path)
        path = settings.genre_checkpoint
        python = settings.muq.python_executable if settings.muq else Path(sys.executable)
        if path is None or not (path / 'config.json').is_file() or not python.is_file():
            raise ValueError('Missing original genre setup')
        if not any((path / name).is_file() for name in ('model.safetensors', 'pytorch_model.bin')):
            raise ValueError('Missing original genre weights')
        return {'ready': True, 'message': '기존 10장르 분류를 사용할 수 있습니다.'}
    except (OSError, ValueError, TypeError):
        return {'ready': False, 'message': '기존 Wav2Vec2 모델 경로를 설정해주세요. 현재 MuQ 방식은 그대로 사용할 수 있습니다.'}


def markdown(tags):
    lines = ['# 기존 AI 장르 분류 결과', '',
             'Wav2Vec2 · 앞부분 최대 30초 · 10장르 후보. 감정·분위기는 미지원입니다.',
             '점수는 정답 확률이 아니며 원본 음원과 노트를 수정하지 않았습니다.', '',
             '| 음원 | 장르 대표 | 대안 후보 | 감정 | 분위기 |', '|---|---|---|---|---|']
    for row in tags['results']:
        name = (row.get('display_name') or Path(row['audio_path']).name).replace('|', '\\|')
        if row['labels'] is None:
            lines.append(f'| {name} | 처리 실패 | — | 미지원 | 미지원 |')
        else:
            alternatives = ' · '.join(c['name'] for c in row['candidates']['genre'][1:])
            lines.append(f'| {name} | {row["labels"]["genre"][0]} | {alternatives} | 미지원 | 미지원 |')
    return '\n'.join(lines) + '\n'


def run(paths, output_directory, settings_path, *, source=None):
    paths = [Path(p).resolve() for p in paths]
    if not paths:
        raise ValueError('No audio selected')
    settings_path = Path(settings_path).resolve()
    settings = ModelSettings.from_file(settings_path)
    if not status(settings_path)['ready']:
        raise ValueError('Original genre model is not configured')
    python = settings.muq.python_executable if settings.muq else Path(sys.executable)
    target = Path(output_directory).resolve()
    target.mkdir(parents=True, exist_ok=False)
    job = target / 'worker-request.json'
    job.write_text(json.dumps({'settings': str(settings_path), 'paths': [str(p) for p in paths]}), encoding='utf-8')
    (target / 'status.json').write_text('{"status":"running"}', encoding='utf-8')
    try:
        with (target / 'worker.log').open('w', encoding='utf-8') as log:
            completed = subprocess.run([str(python), '-X', 'utf8', str(Path(__file__).resolve()), str(job)],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=600 * len(paths),
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        if completed.returncode:
            raise ValueError('Original genre worker failed; see worker.log')
        raw = json.loads((target / 'classification.json').read_text(encoding='utf-8'))
        if len(raw['results']) != len(paths):
            raise ValueError('Mismatched original genre results')
        rows = []
        for path, result in zip(paths, raw['results']):
            row = {'audio_path': str(path), 'display_name': path.name, 'status': result['status'],
                   'labels': None, 'candidates': None}
            if result['success']:
                candidates = [{**c, 'name': NAMES.get(c['label'], c['label']), 'rank': rank,
                               'representative': rank == 1}
                              for rank, c in enumerate(result['axes']['genre']['candidates'], 1)]
                row.update(labels={'genre': [candidates[0]['name']], 'emotion': [], 'mood': []},
                           candidates={'genre': candidates, 'emotion': [], 'mood': []},
                           audio_sha256=digest(path))
            else:
                row['error'] = result.get('error')
            if source:
                if row['labels'] is not None and row['audio_sha256'] != source['audio_sha256']:
                    raise ValueError('YouTube audio changed')
                row.update(audio_path=None, display_name=source['title'], source_url=source['url'])
            rows.append(row)
        tags = {'schema': 'music_legacy_genre_tags_v1', 'classifier': 'wav2vec2', 'results': rows,
                'lyrics_analyzed': False, 'human_verified': False, 'accuracy_validated': False,
                'unsupported_axes': ['emotion', 'mood'], 'analysis_scope': 'first_30_seconds'}
        if source:
            tags['input_source'] = source
        (target / 'tags.json').write_text(json.dumps(tags, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
        (target / '태깅결과.md').write_text(markdown(tags), encoding='utf-8')
        good = sum(r['labels'] is not None for r in rows)
        (target / 'status.json').write_text(json.dumps({'status': 'completed', 'successful': good,
            'failed': len(paths) - good}), encoding='utf-8')
        return target, 0 if good == len(paths) else 1
    except Exception:
        (target / 'status.json').write_text('{"status":"failed"}', encoding='utf-8')
        raise
    finally:
        job.unlink(missing_ok=True)


def run_youtube(url, target, settings_path):
    from music_classification.youtube_input import temporary_audio
    with temporary_audio(url) as audio:
        folder, code = run([audio.path], target, settings_path, source=audio.metadata)
    path = folder / 'status.json'
    state = json.loads(path.read_text(encoding='utf-8'))
    state.update(input_kind='youtube', temporary_audio_removed=True)
    path.write_text(json.dumps(state), encoding='utf-8')
    return folder, code


def worker(job_path):
    from music_classification.inference import classify_audio
    job_path = Path(job_path)
    job = json.loads(job_path.read_text(encoding='utf-8'))
    settings = ModelSettings.from_file(job['settings'])
    model, processor = settings.load_genre_model()
    if model is None or processor is None:
        raise ValueError('No original genre model')
    results = [classify_audio(path, model, processor, max_duration=30, top_k=3) for path in job['paths']]
    (job_path.parent / 'classification.json').write_text(json.dumps({'classifier': 'wav2vec2',
        'analysis_scope': 'first_30_seconds', 'results': results}, ensure_ascii=False, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    worker(sys.argv[1])
