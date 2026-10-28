"""Opt-in descriptive controls. No calibration, voting, or prediction override."""

from pathlib import Path
from statistics import mean
import hashlib
import math
import shutil
import tempfile
from uuid import uuid4


def fingerprint(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def compare(first, second, *, mood=False):
    if first.get('status') != 'ok' or second.get('status') != 'ok':
        return {'status': 'unavailable_comparison',
                'axis_statuses': [first.get('status'), second.get('status')]}
    a = {row['label'] for row in first['candidates'][:3]}
    b = {row['label'] for row in second['candidates'][:3]}
    result = {'status': 'ok', 'primary_same': first['primary']['label'] == second['primary']['label'],
              'top_candidates_jaccard': len(a & b) / len(a | b),
              'candidate_counts': [len(a), len(b)]}
    def vector(axis):
        values = axis.get('similarity_scores', axis.get('native_scores'))
        return values if values is not None else {r['label']: r['score'] for r in axis['candidates']}
    left, right = vector(first), vector(second)
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
           for values in (left, right) for v in values.values()):
        raise ValueError('Non-finite diagnostic score vector')
    # Truncated genre candidates can cover different labels. Never fill missing scores with zero.
    same = set(left) == set(right)
    result['score_vector_comparable'] = same
    result['score_label_count'] = len(left) if same else None
    result['score_scope'] = 'native_vector' if 'native_scores' in first or 'similarity_scores' in first else 'returned_candidates_only'
    differences = [abs(left[k] - right[k]) for k in left] if same else []
    result['maximum_absolute_score_change'] = max(differences) if differences else None
    result['mean_absolute_score_change'] = mean(differences) if differences else None
    if mood:
        x, y = set(first['labels']), set(second['labels'])
        result['threshold_tags_both_empty'] = not x and not y
        result['threshold_tag_jaccard'] = len(x & y) / len(x | y) if x | y else None
    return result


def classify_with_diagnostics(classifier, audio_path, options, directory=None):
    if not getattr(classifier, '_diagnostic_session_active', False):
        batch = classifier.classify_many([audio_path], diagnostics=True,
                                        diagnostics_directory=directory, **options)
        result = batch['results'][0]
        result['diagnostics']['execution'] = batch['execution']
        return result
    result = classifier._classify_base(audio_path, **options)
    diagnostic = {'schema_version': '1.0', 'status': 'unavailable', 'is_accuracy': False,
                  'is_confidence': False, 'prediction_override': False,
                  'protocol': 'three_contiguous_windows_gain_minus3db_repeat_v1',
                  'limitations': ['Context and pooling changes are not proven model errors',
                                 'No human accuracy reference or J-pop validation',
                                 'Each condition uses the configured per-model duration rules',
                                 'Repeat is immediate after this track controls, not a long batch stress test']}
    result['diagnostics'] = diagnostic
    if not result.get('success'):
        diagnostic['reason'] = 'No successful baseline axis; controls skipped'
        return result
    job = None
    try:
        import numpy as np
        import soundfile as sf
        path = Path(audio_path).resolve()
        expected = result['input']['audio_sha256']
        if fingerprint(path) != expected:
            raise ValueError('Source changed after baseline inference')
        info = sf.info(path)
        if info.frames < 3 * info.samplerate:
            diagnostic['reason'] = 'Three windows require at least 3 seconds of audio'
            return result
        if info.frames * info.channels * 4 > 256 * 1024 * 1024:
            diagnostic['reason'] = 'Diagnostic decode exceeds the 256 MB float32 budget'
            return result
        audio, sr = sf.read(path, dtype='float32', always_2d=True)
        if not np.isfinite(audio).all() or fingerprint(path) != expected:
            raise ValueError('Source changed or contains non-finite samples')
        root = Path(directory if directory is not None else Path(tempfile.gettempdir()) / 'music-classification-diagnostics').resolve()
        root.mkdir(parents=True, exist_ok=True)
        job = root / ('diagnostic-' + uuid4().hex)
        job.mkdir()
        quotient, remainder = divmod(len(audio), 3)
        start = 0
        conditions = []
        for i in range(3):
            end = start + quotient + (i < remainder)
            output = job / f'window_{i + 1}.wav'
            sf.write(output, audio[start:end], sr, subtype='FLOAT')
            conditions.append((f'window_{i + 1}', output, [start, end]))
            start = end
        factor = math.pow(10, -3 / 20)
        output = job / 'gain_minus3db.wav'
        sf.write(output, np.multiply(audio, factor, dtype=np.float32), sr, subtype='FLOAT')
        conditions += [('gain_minus3db', output, [0, len(audio)]), ('repeat', path, [0, len(audio)])]
        controls = {}
        for name, control_path, bounds in conditions:
            controls[name] = {'source_sample_range': bounds,
                              'audio_sha256': fingerprint(control_path),
                              'result': classifier._classify_base(control_path, **options)}
        if fingerprint(path) != expected:
            raise ValueError('Source changed during diagnostic controls; comparisons discarded')
        axes = {}
        complete = True
        for axis in ('genre', 'emotion', 'mood'):
            initial = result['axes'][axis]
            windows = [controls[f'window_{i}']['result']['axes'][axis] for i in range(1, 4)]
            comparisons = {name: compare(initial, controls[name]['result']['axes'][axis], mood=axis == 'mood')
                           for name in ('repeat', 'gain_minus3db', 'window_1', 'window_2', 'window_3')}
            labels = [window['primary']['label'] if window.get('status') == 'ok' else None for window in windows]
            axes[axis] = {'comparisons': comparisons, 'window_primary_labels': labels,
                          'primary_unanimous': len(set(labels)) == 1 if all(labels) else None}
            complete &= all(row['status'] == 'ok' for row in comparisons.values())
        diagnostic.update(status='ok' if complete else 'partial', axes=axes, controls=controls,
                          source_sha256=expected, sample_rate=sr, source_frames=len(audio),
                          gain_db=-3, gain_factor=factor)
    except Exception as exc:
        diagnostic.update(status='error', error=f'{type(exc).__name__}: {exc}')
    finally:
        if job is not None:
            try:
                if job.is_symlink() or job.resolve().parent != root or not job.name.startswith('diagnostic-'):
                    raise ValueError('Unsafe diagnostic cleanup path')
                shutil.rmtree(job)
            except Exception as exc:
                diagnostic['cleanup_error'] = f'{type(exc).__name__}: {exc}'
                diagnostic['status'] = 'error'
    return result
