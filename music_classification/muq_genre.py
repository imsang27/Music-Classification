"""Explicit fixed common10 genre ranking; raw similarities are not probabilities."""

from importlib.resources import files
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from uuid import uuid4
import math

from .inference import _axis
from .muq import ExperimentalMuQClassifier
from .muq_runtime import digest

GENRES = {'disco': '디스코', 'metal': '메탈', 'reggae': '레게', 'blues': '블루스',
          'rock': '록', 'classical': '클래식', 'jazz': '재즈', 'hiphop': '힙합',
          'country': '컨트리', 'pop': '팝'}
PROMPT_SHA256 = '521c933927cf56e23d49f75fb71411ed57e8e087e6b74e36806d0fd1439fbdba'


def prompt_spec():
    raw = files('music_classification').joinpath('resources', 'muq_genre_prompts.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != PROMPT_SHA256:
        raise ValueError('Fixed MuQ genre prompt resource changed')
    return json.loads(raw)


def validated_scores(scores):
    if (not isinstance(scores, dict) or set(scores) != set(GENRES)
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                   or not -1.000001 <= v <= 1.000001 for v in scores.values())):
        raise ValueError('Expected ten finite raw MuQ genre similarities')
    return {label: float(scores[label]) for label in GENRES}


class ExperimentalMuQGenreClassifier(ExperimentalMuQClassifier):
    """Use the same explicit local installation settings as the emotion adapter.

    This independent adapter ranks only the common10 prompts evaluated on public
    uploader tags. It does not claim J-pop, custom9, or subgenre validation.
    """

    def batch_session(self):
        from .batching import backend_session
        return backend_session(self, 'muq_genre')

    def classify_many(self, audio_paths, *, top_k=3):
        from .batching import backend_many
        return backend_many(self, 'muq_genre', audio_paths, top_k=top_k)

    def classify(self, audio_path, *, top_k=3):
        result = {'schema_version': '1.0', 'status': 'error', 'success': False, 'experimental': True,
                  'axes': {name: _axis() for name in ('genre', 'emotion', 'mood')},
                  'analysis': None, 'model': None, 'warnings': []}

        def fail(status, message):
            result.update(status=status, error=message)
            return result

        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 10:
            return fail('error', 'top_k must be an integer from1 to10')
        try:
            path = Path(audio_path).resolve()
            if not path.is_file() or not 0 < path.stat().st_size <= 100 * 1024 * 1024:
                raise ValueError('Audio missing, empty, or exceeds100 MB')
            original_hash = digest(path)
        except (OSError, TypeError, ValueError) as exc:
            return fail('invalid_audio', str(exc))
        if not self.python.is_file() or not (self.installation / 'runtime-manifest.json').is_file():
            return fail('unavailable', 'Verified local MuQ installation or interpreter missing')
        job = None
        try:
            spec = prompt_spec()
            jobs = self.work_directory
            jobs.mkdir(parents=True, exist_ok=True)
            if jobs.resolve() != jobs:
                raise ValueError('Unsafe MuQ genre work directory')
            candidate = (jobs / ('request-' + uuid4().hex)).resolve()
            if candidate.parent != jobs:
                raise ValueError('Unsafe MuQ genre request path')
            candidate.mkdir()
            job = candidate
            with (job / 'worker.log').open('w', encoding='utf-8') as log:
                session = getattr(self, '_worker_session', None)
                run = session.run if session is not None else subprocess.run
                completed = run([str(self.python), str(Path(__file__).with_name('portable_muq_genre_worker.py')),
                                 '--base', str(self.installation), '--job', str(job), '--audio', str(path),
                                 '--cache', str(self.cache_directory)], cwd=str(job), stdout=log,
                                stderr=subprocess.STDOUT, timeout=self.timeout, check=False)
            output = job / 'result.json'
            if not output.is_file():
                return fail('error', f'MuQ genre worker failed (exit{completed.returncode}); no prediction')
            payload = json.loads(output.read_text(encoding='utf-8'))
            if not isinstance(payload, dict):
                raise ValueError('Invalid MuQ genre response')
            if payload.get('status') != 'ok':
                return fail(payload.get('status') if payload.get('status') in ('invalid_audio', 'unavailable') else 'error',
                            str(payload.get('error', 'MuQ genre inference failed')))
            if completed.returncode != 0:
                return fail('error', f'MuQ genre worker exited{completed.returncode}')
            scores = validated_scores(payload['scores'])
            analysis, model = payload['analysis'], payload['model']
            if (not isinstance(analysis, dict) or not isinstance(model, dict)
                    or model.get('name') != 'OpenMuQ/MuQ-MuLan-large'
                    or model.get('genre_prompt_sha256') != PROMPT_SHA256):
                raise ValueError('Missing or mismatched MuQ genre provenance')
            if analysis.get('audio_sha256') != original_hash or digest(path) != original_hash:
                return fail('invalid_audio', 'Audio changed during genre inference; predictions discarded')
            candidates = sorted(({'label': label, 'name': GENRES[label], 'score': value} for label, value in scores.items()),
                                key=lambda item: item['score'], reverse=True)[:top_k]
            result['axes']['genre'] = {'status': 'ok', 'primary': candidates[0], 'candidates': candidates,
                'labels': [candidates[0]['label']], 'similarity_scores': scores, 'supported_labels': list(GENRES),
                'score_type': 'mean_dot_similarity', 'calibration': 'uncalibrated',
                'label_space': 'muq_genre_common10_v1', 'multilabel': False,
                'nearest_supported': True, 'experimental': True,
                'validation': {'release_ready': False, 'jpop_validated': False, 'subgenres_validated': False,
                               'custom9_validated': False, 'scope': 'Selected public uploader-tag retrieval only',
                               'public_reference': spec['evaluation']}}
            result['analysis'], result['model'] = analysis, model
            result['warnings'] = ['Experimental common10 genre candidates; J-pop and subgenres remain unvalidated',
                'Raw similarities can be negative; they are not probabilities or correctness confidence',
                'Complete audio uses the fixed public-study window protocol; no genre remapping or score fusion',
                'Top-k alternatives do not imply simultaneous genres',
                'Digital silence check does not reject noise or speech']
            result.update(status='partial', success=True)
            return result
        except subprocess.TimeoutExpired:
            return fail('error', f'MuQ genre inference exceeded{self.timeout} seconds; no prediction')
        except Exception as exc:
            return fail('error', f'MuQ genre inference failed: {exc}')
        finally:
            if job is not None and job.parent == self.work_directory and job.name.startswith('request-'):
                try:
                    shutil.rmtree(job)
                except OSError:
                    result['warnings'].append('MuQ genre temporary request cleanup failed')
