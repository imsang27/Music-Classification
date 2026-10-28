"""Opt-in offline Music2Emo adapter; upstream runs in an isolated interpreter."""

import json
import math
from pathlib import Path
import shutil
import subprocess
from uuid import uuid4

from .inference import _axis
from .music2emo_results import validated_scores


class ExperimentalMusic2EmoClassifier:
    """Use the locally verified mode-corrected runtime without downloading files.

    Single calls use a fresh process; serial batches reuse a private model worker.
    Verified runtime bytes remain immutable; a request-local copy reads the
    configured installation path. Work and cache folders can be set separately.
    """

    def __init__(self, *, installation=None, python_executable=None, timeout=600, work_directory=None, cache_directory=None):
        root = Path(__file__).resolve().parents[1]
        self.installation = Path(installation or root / 'work/music2emo-experiment').resolve()
        self.python = Path(python_executable or root / 'work/music2emo-env/Scripts/python.exe').resolve()
        self.work_directory = Path(work_directory or self.installation / 'api-jobs').resolve()
        self.cache_directory = Path(cache_directory or self.work_directory.parent / '.inference-cache').resolve()
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError('timeout must be positive finite seconds')
        self.timeout = timeout

    def batch_session(self):
        from .batching import backend_session
        return backend_session(self, 'music2emo')

    def classify_many(self, audio_paths, *, top_k=3):
        from .batching import backend_many
        return backend_many(self, 'music2emo', audio_paths, top_k=top_k)

    def classify(self, audio_path, *, top_k=3):
        result = {'schema_version': '1.0', 'status': 'error', 'success': False,
                  'axes': {name: _axis() for name in ('genre', 'emotion', 'mood')},
                  'analysis': None, 'model': None, 'warnings': []}

        def fail(status, message):
            result.update(status=status, error=message)
            result['axes'] = {name: _axis() for name in ('genre', 'emotion', 'mood')}
            result['analysis'], result['model'] = None, None
            result.pop('valence_arousal', None)
            return result

        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 56:
            return fail('error', 'top_k must be an integer from 1 to 56')
        try:
            path = Path(audio_path).resolve()
            if not path.is_file() or not 0 < path.stat().st_size <= 100 * 1024 * 1024:
                return fail('invalid_audio', 'Audio missing, empty, or exceeds 100 MB')
        except (TypeError, ValueError, OSError) as exc:
            return fail('invalid_audio', str(exc))
        if not self.python.is_file() or not (self.installation / 'runtime-patches-modefix.json').is_file():
            return fail('unavailable', 'Verified local Music2Emo installation or interpreter missing')
        job = None
        try:
            jobs = self.work_directory
            jobs.mkdir(parents=True, exist_ok=True)
            # Resolve and check the final target before any later recursive cleanup.
            if jobs.resolve() != self.work_directory:
                raise ValueError('Unsafe job directory')
            job = (jobs / ('request-' + uuid4().hex)).resolve()
            if job.parent != jobs.resolve():
                raise ValueError('Unsafe request directory')
            job.mkdir()
            worker = Path(__file__).with_name('portable_music2emo_worker.py')
            with (job / 'worker.log').open('w', encoding='utf-8') as log:
                session = getattr(self, '_worker_session', None)
                run = session.run if session is not None else subprocess.run
                completed = run(
                    [str(self.python), str(worker), '--base', str(self.installation),
                     '--job', str(job), '--audio', str(path), '--cache', str(self.cache_directory)],
                    cwd=str(job), stdout=log, stderr=subprocess.STDOUT,
                    timeout=self.timeout, check=False)
            output = job / 'result.json'
            if not output.is_file():
                return fail('error', f'Music2Emo worker failed (exit {completed.returncode}); no prediction')
            payload = json.loads(output.read_text(encoding='utf-8'))
            if payload.get('status') != 'ok':
                return fail(payload.get('status') if payload.get('status') in ('invalid_audio', 'unavailable') else 'error',
                            str(payload.get('error', 'Music2Emo inference failed')))
            if completed.returncode != 0:
                return fail('error', f'Music2Emo worker exited {completed.returncode}')
            labels = payload['labels']
            checked = validated_scores(payload['raw'], labels)
            scores = checked['native_scores']
            candidates = sorted(({'label': label, 'score': score} for label, score in scores.items()),
                                key=lambda item: item['score'], reverse=True)[:top_k]
            result['axes']['mood'] = {
                'status': 'ok', 'primary': candidates[0], 'candidates': candidates,
                'labels': [label for label in labels if scores[label] >= .5],
                'native_scores': scores, 'score_type': 'sigmoid', 'calibration': 'uncalibrated',
                'label_space': 'mtg_jamendo_native_mood_theme_56', 'multilabel': True,
                'threshold': .5, 'nearest_supported': True,
                'validation': {'evaluated_labels': ['happy', 'sad'], 'jpop_validated': False,
                               'scope': 'small public-audio human-reference happy/sad comparison; other 54 labels unvalidated'}}
            result['axes']['emotion']['reason'] = 'Native emotion/mood/theme labels are retained together under mood; no independent emotion taxonomy mapping'
            result['axes']['genre']['reason'] = 'Music2Emo does not supply genre'
            result['valence_arousal'] = {key: checked[key] for key in
                                       ('valence', 'arousal', 'va_outside_nominal_1_to_9')}
            result['valence_arousal']['validation'] = 'not evaluated; raw regression, not calibrated'
            result['analysis'] = payload['analysis']
            result['model'] = payload['model']
            result['warnings'] = ['Experimental opt-in backend; default classifier unchanged',
                                  'Native scores are not correctness probabilities',
                                  'Only happy/sad evaluated; J-pop and valence/arousal unvalidated',
                                  'Genre and separate emotion taxonomy unavailable',
                                  'Digital silence check does not reject noise or speech']
            result.update(status='partial', success=True)
            return result
        except subprocess.TimeoutExpired:
            return fail('error', f'Music2Emo inference exceeded {self.timeout} seconds; no prediction')
        except Exception as exc:
            return fail('error', f'Music2Emo inference failed: {exc}')
        finally:
            if job is not None and job.parent == self.work_directory and job.name.startswith('request-'):
                shutil.rmtree(job)
