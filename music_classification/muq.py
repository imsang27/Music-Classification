"""Opt-in emotion prompt ranking with raw, possibly negative, MuQ similarities."""

import json
import math
from pathlib import Path
import shutil
import subprocess
from uuid import uuid4

from .inference import _axis
from .muq_runtime import digest

EMOTIONS = {'joy': '기쁨', 'sadness': '슬픔', 'anticipation': '설렘', 'longing': '그리움',
            'poignancy': '애절함', 'hope': '희망', 'anxiety': '불안', 'anger': '분노'}


def validated_scores(scores):
    if (not isinstance(scores, dict) or set(scores) != set(EMOTIONS)
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                   or not -1.000001 <= v <= 1.000001 for v in scores.values())):
        raise ValueError('Expected eight finite raw MuQ similarity scores')
    return {label: float(scores[label]) for label in EMOTIONS}


class ExperimentalMuQClassifier:
    """Load the reviewed model in an isolated worker; batches reuse that worker.

    This ranks fixed draft emotion prompts; it does not provide calibrated
    probabilities, verified multi-label tags, or emotion-intensity estimates.
    Verified model installations may live separately from writable request/cache
    directories; no project-absolute model path is required by this adapter.
    """

    def __init__(self, *, installation=None, python_executable=None, timeout=600, work_directory=None, cache_directory=None):
        root = Path(__file__).resolve().parents[1]
        self.installation = Path(installation or root / 'work/muq-experiment').resolve()
        self.python = Path(python_executable or root / 'work/muq-env/Scripts/python.exe').resolve()
        self.work_directory = Path(work_directory or self.installation / 'api-jobs').resolve()
        self.cache_directory = Path(cache_directory or self.work_directory.parent / '.inference-cache').resolve()
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError('timeout must be positive finite seconds')
        self.timeout = timeout

    def batch_session(self):
        from .batching import backend_session
        return backend_session(self, 'muq')

    def classify_many(self, audio_paths, *, top_k=3):
        from .batching import backend_many
        return backend_many(self, 'muq', audio_paths, top_k=top_k)

    def classify(self, audio_path, *, top_k=3):
        result = {'schema_version': '1.0', 'status': 'error', 'success': False, 'experimental': True,
                  'axes': {name: _axis() for name in ('genre', 'emotion', 'mood')},
                  'analysis': None, 'model': None, 'warnings': []}

        def fail(status, message):
            result.update(status=status, error=message)
            return result

        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 8:
            return fail('error', 'top_k must be an integer from1 to8')
        try:
            path = Path(audio_path).resolve()
            if not path.is_file() or not 0 < path.stat().st_size <= 100 * 1024 * 1024:
                return fail('invalid_audio', 'Audio missing, empty, or exceeds100 MB')
            original_hash = digest(path)
        except (OSError, ValueError, TypeError) as exc:
            return fail('invalid_audio', str(exc))
        if not self.python.is_file() or not (self.installation / 'runtime-manifest.json').is_file():
            return fail('unavailable', 'Verified local MuQ installation or interpreter missing')
        job = None
        try:
            jobs = self.work_directory
            jobs.mkdir(parents=True, exist_ok=True)
            if jobs.resolve() != self.work_directory:
                raise ValueError('Unsafe MuQ job directory')
            candidate = (jobs / ('request-' + uuid4().hex)).resolve()
            if candidate.parent != jobs.resolve():
                raise ValueError('Unsafe MuQ request path')
            candidate.mkdir()
            job = candidate
            with (job / 'worker.log').open('w', encoding='utf-8') as log:
                session = getattr(self, '_worker_session', None)
                run = session.run if session is not None else subprocess.run
                completed = run([str(self.python), str(Path(__file__).with_name('portable_muq_worker.py')),
                    '--base', str(self.installation), '--job', str(job), '--audio', str(path), '--cache', str(self.cache_directory)],
                    cwd=str(job), stdout=log, stderr=subprocess.STDOUT, timeout=self.timeout, check=False)
            output = job / 'result.json'
            if not output.is_file():
                return fail('error', f'MuQ worker failed (exit{completed.returncode}); no prediction')
            payload = json.loads(output.read_text(encoding='utf-8'))
            if not isinstance(payload, dict):
                raise ValueError('Invalid MuQ response')
            if payload.get('status') != 'ok':
                return fail(payload.get('status') if payload.get('status') in ('invalid_audio', 'unavailable') else 'error',
                            str(payload.get('error', 'MuQ inference failed')))
            if completed.returncode != 0:
                return fail('error', f'MuQ worker exited{completed.returncode}')
            scores = validated_scores(payload['scores'])
            analysis, model = payload['analysis'], payload['model']
            if not isinstance(analysis, dict) or not isinstance(model, dict):
                raise ValueError('Missing MuQ provenance')
            if analysis.get('audio_sha256') != original_hash or digest(path) != original_hash:
                return fail('invalid_audio', 'Audio changed during inference; predictions discarded')
            candidates = sorted(({'label': label, 'name': EMOTIONS[label], 'score': score}
                                 for label, score in scores.items()), key=lambda r: r['score'], reverse=True)[:top_k]
            result['axes']['emotion'] = {
                'status': 'ok', 'primary': candidates[0], 'candidates': candidates,
                'labels': [candidates[0]['label']], 'similarity_scores': scores,
                'score_type': 'mean_dot_similarity', 'calibration': 'uncalibrated',
                'label_space': 'draft_emotion_prompts_v0.1.0', 'multilabel': False,
                'nearest_supported': True, 'experimental': True,
                'validation': {'release_ready': False, 'jpop_validated': False,
                    'all_eight_emotions_validated': False, 'emotion_presence_accuracy_established': False,
                    'evaluated_related_references': {'joy': ['happy', 'joyful'], 'sadness': ['sad'], 'anxiety': ['anxious']},
                    'evaluated_direct_references': {'anger': {
                        'study_id': 'soundtracks_anger_evaluation_v2',
                        'reference': 'Soundtracks Set2 native Anger mean rating, scale1..9',
                        'excerpts': 49, 'source_album_track_groups': 45,
                        'metric': 'Spearman correlation of group-mean ratings and group-mean raw scores',
                        'spearman_rho': 0.7665536587542701,
                        'exploratory_bootstrap_95_interval': [0.5979067547997016, 0.8675461022526018],
                        'bootstrap_draws': 5000,
                        'results_sha256': '219a15d569376e1f641b447cd074b5fb29286f51763872bf1c4c7a19637352d4',
                        'scope': 'Short film soundtrack excerpts; not a final independent J-pop test',
                        'presence_accuracy_established': False, 'intensity_calibrated': False,
                        'jpop_validated': False, 'model_superiority_established': False}},
                    'unevaluated_labels': ['anticipation', 'longing', 'poignancy', 'hope'],
                    'scope': 'Related selected-only intensity correlations plus a direct anger group-ordering study; '
                             'not all8 emotion presence accuracy, calibrated intensity, model superiority, or J-pop validation'}}
            result['analysis'], result['model'] = analysis, model
            result['axes']['genre']['reason'] = 'MuQ adapter is configured only for emotion prompt ranking'
            result['axes']['mood']['reason'] = 'Mood supplied separately; no emotion-to-mood mapping'
            result['warnings'] = ['Experimental nearest emotion candidates; not verified emotion tags',
                'Raw similarities can be negative; they are not probabilities, accuracy, or calibrated intensity',
                'Top-k alternatives do not imply simultaneous emotions',
                'All8 custom emotions and J-pop accuracy remain unvalidated',
                'Digital silence check does not reject noise or speech']
            result.update(status='partial', success=True)
            return result
        except subprocess.TimeoutExpired:
            return fail('error', f'MuQ inference exceeded{self.timeout} seconds; no prediction')
        except Exception as exc:
            return fail('error', f'MuQ inference failed: {exc}')
        finally:
            if job is not None and job.parent == self.work_directory and job.name.startswith('request-'):
                try:
                    shutil.rmtree(job)
                except OSError:
                    result['warnings'].append('MuQ temporary request cleanup failed')
