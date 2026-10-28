"""Explicit composition of genre and native Music2Emo inference."""

from copy import deepcopy, copy
from contextlib import ExitStack
import hashlib
import math
import time
from pathlib import Path

from .inference import _axis, classify_audio


class ExperimentalCombinedClassifier:
    """Run both supplied backends independently, preserving scores and provenance.

    No automatic loading, downloads, taxonomy mapping, or cross-model score fusion.
    A partial success means at least one supported axis produced a prediction.
    """

    def __init__(self, genre_model, genre_processor, *, mood_classifier, emotion_classifier=None,
                 genre_classifier=None, share_muq_model=True):
        if not isinstance(share_muq_model, bool):
            raise ValueError('share_muq_model must be a bool')
        self.share_muq_model = share_muq_model
        if not callable(getattr(mood_classifier, 'classify', None)):
            raise ValueError('Supply an explicit mood classifier with classify()')
        self.genre_model = genre_model
        self.genre_processor = genre_processor
        if genre_classifier is not None:
            if not callable(getattr(genre_classifier, 'classify', None)):
                raise ValueError('Supply an explicit genre classifier with classify()')
            if genre_model is not None or genre_processor is not None:
                raise ValueError('Choose either an explicit genre classifier or the supplied genre model/processor')
        self.genre_classifier = genre_classifier
        self.mood_classifier = mood_classifier
        if emotion_classifier is not None and not callable(getattr(emotion_classifier, 'classify', None)):
            raise ValueError('Supply an explicit emotion classifier with classify()')
        self.emotion_classifier = emotion_classifier

    def classify_many(self, audio_paths, **options):
        """Preserve order and per-file results while reusing supported workers."""
        from .batching import input_list, batch_result
        paths = input_list(audio_paths)
        started = time.perf_counter()
        pooled = copy(self)
        pooled._diagnostic_session_active = True
        sessions = {}
        with ExitStack() as stack:
            from .shared_muq import compatible, shared_session
            if self.share_muq_model and compatible(self.genre_classifier, self.emotion_classifier):
                genre, emotion = stack.enter_context(shared_session(self.genre_classifier, self.emotion_classifier))
                pooled.genre_classifier, pooled.emotion_classifier = genre, emotion
                sessions['muq_shared'] = genre
            for attribute, kind in (('mood_classifier', 'music2emo'), ('emotion_classifier', 'muq'),
                                    ('genre_classifier', 'muq_genre')):
                backend = getattr(pooled, attribute)
                if backend is not None and not hasattr(backend, '_worker_session') and callable(getattr(backend, 'batch_session', None)):
                    backend = stack.enter_context(backend.batch_session())
                    setattr(pooled, attribute, backend)
                    sessions[kind] = backend
            results = [pooled.classify(path, **options) for path in paths]
        statistics = {kind: dict(backend._worker_session.statistics) if hasattr(backend, '_worker_session')
                      else {'mode': 'external_session_statistics_unavailable'} for kind, backend in sessions.items()}
        return batch_result(paths, results, statistics, started)

    def classify(self, audio_path, *, genre_max_duration=30, genre_top_k=3,
                 mood_top_k=3, genre_warning_threshold=.5, emotion_top_k=3,
                 diagnostics=False, diagnostics_directory=None):
        if not isinstance(diagnostics, bool):
            raise ValueError('diagnostics must be a bool')
        options = dict(genre_max_duration=genre_max_duration, genre_top_k=genre_top_k,
                       mood_top_k=mood_top_k, genre_warning_threshold=genre_warning_threshold,
                       emotion_top_k=emotion_top_k)
        from .shared_muq import compatible
        if (self.share_muq_model and not getattr(self, '_diagnostic_session_active', False)
                and compatible(self.genre_classifier, self.emotion_classifier)):
            batch = self.classify_many([audio_path], diagnostics=diagnostics,
                                      diagnostics_directory=diagnostics_directory, **options)
            result = batch['results'][0]
            if diagnostics:
                result['diagnostics']['execution'] = batch['execution']
            else:
                result['execution'] = batch['execution']
            return result
        if diagnostics:
            from .diagnostics import classify_with_diagnostics
            return classify_with_diagnostics(self, audio_path, options, diagnostics_directory)
        return self._classify_base(audio_path, **options)

    def _classify_base(self, audio_path, *, genre_max_duration=30, genre_top_k=3,
                       mood_top_k=3, genre_warning_threshold=.5, emotion_top_k=3):
        result = {'schema_version': '1.0', 'status': 'error', 'success': False,
                  'experimental': True,
                  'axes': {axis: _axis() for axis in ('genre', 'emotion', 'mood')},
                  'components': {}, 'analysis': {}, 'model': {}, 'warnings': []}

        def fingerprint(path):
            with path.open('rb') as stream:
                return hashlib.file_digest(stream, 'sha256').hexdigest()

        try:
            path = Path(audio_path).resolve()
            size = path.stat().st_size
            if not path.is_file() or not 0 < size <= 100 * 1024 * 1024:
                raise ValueError('Audio must be a non-empty file no larger than 100 MB')
            original_hash = fingerprint(path)
        except (OSError, ValueError, TypeError) as exc:
            result.update(status='invalid_audio', error=str(exc))
            return result

        def invoke(callback, axis):
            try:
                value = callback()
                if (not isinstance(value, dict) or not isinstance(value.get('axes'), dict)
                        or not isinstance(value.get('success'), bool)
                        or value.get('status') not in ('partial', 'ok', 'unavailable', 'invalid_audio', 'error')):
                    raise ValueError('Backend returned an invalid response contract')
                if value['success']:
                    if value['status'] not in ('ok', 'partial'):
                        raise ValueError('Backend success conflicts with failure status')
                    prediction = value['axes'].get(axis)
                    if not isinstance(prediction, dict):
                        raise ValueError('Backend omitted the supported axis')
                    primary = prediction.get('primary')
                    similarity = (axis == 'emotion' or axis == 'genre' and self.genre_classifier is not None) and prediction.get('score_type') in ('mean_dot_similarity', 'mean_cosine_similarity')
                    lower, upper = (-1.000001, 1.000001) if similarity else (0, 1)

                    def valid_candidate(candidate):
                        return (isinstance(candidate, dict) and isinstance(candidate.get('label'), str) and bool(candidate['label'])
                                and not isinstance(candidate.get('score'), bool) and isinstance(candidate.get('score'), (int, float))
                                and math.isfinite(candidate['score']) and lower <= candidate['score'] <= upper)

                    candidates = prediction.get('candidates')
                    if (prediction.get('status') != 'ok' or not isinstance(primary, dict)
                            or not valid_candidate(primary) or not isinstance(candidates, list) or not candidates
                            or primary != candidates[0] or any(not valid_candidate(item) for item in candidates)
                            or len({item['label'] for item in candidates}) != len(candidates)):
                        raise ValueError('Backend returned an invalid supported-axis prediction')
                return value
            except Exception as exc:
                return {'status': 'error', 'success': False, 'axes': {},
                        'error': f'{type(exc).__name__}: {exc}'}

        # Execute independently: an unavailable or failed genre model must not
        # suppress valid full-track mood inference (or vice versa).
        components = result['components']
        if self.genre_classifier is None:
            components['genre'] = invoke(lambda: classify_audio(
                path, self.genre_model, self.genre_processor, max_duration=genre_max_duration,
                warning_threshold=genre_warning_threshold, top_k=genre_top_k), 'genre')
        else:
            components['genre'] = invoke(lambda: self.genre_classifier.classify(path, top_k=genre_top_k), 'genre')
        components['music2emo'] = invoke(lambda: self.mood_classifier.classify(path, top_k=mood_top_k), 'mood')
        if self.emotion_classifier is not None:
            components['muq'] = invoke(lambda: self.emotion_classifier.classify(path, top_k=emotion_top_k), 'emotion')
        try:
            unchanged = path.stat().st_size == size and fingerprint(path) == original_hash
        except OSError:
            unchanged = False
        if not unchanged:
            # Do not present predictions that may have read different inputs.
            result.update(status='invalid_audio', error='Audio changed during inference; predictions discarded')
            result['components'] = {name: {'status': value['status'], 'success': False,
                                          'error': 'Input changed; predictions discarded'}
                                    for name, value in components.items()}
            return result
        result['input'] = {'audio_sha256': original_hash, 'bytes': size}

        sources = [('genre', 'genre'), ('mood', 'music2emo')]
        if self.emotion_classifier is not None:
            sources.append(('emotion', 'muq'))
        for axis, source in sources:
            component = components[source]
            prediction = component['axes'].get(axis, {})
            if component['success'] and prediction.get('status') == 'ok':
                result['axes'][axis] = deepcopy(prediction)
                result['axes'][axis]['source'] = source
                result['success'] = True
            else:
                result['axes'][axis].update(status=component['status'] if component['status'] in
                                           ('invalid_audio', 'error', 'unavailable') else 'unavailable',
                                           reason=component.get('error', f'{source} axis unavailable'), source=source)
            result['analysis'][source] = deepcopy(component.get('analysis'))
            result['model'][source] = deepcopy(component.get('model'))
        if self.emotion_classifier is None:
            result['axes']['emotion']['reason'] = 'Native emotion/mood/theme labels are preserved together under mood; separate emotion taxonomy unavailable'
        if result['axes']['mood']['status'] == 'ok' and 'valence_arousal' in components['music2emo']:
            result['valence_arousal'] = deepcopy(components['music2emo']['valence_arousal'])
        if result['success']:
            result['status'] = 'partial'
        else:
            statuses = {component['status'] for component in components.values()}
            result['status'] = next((status for status in ('error', 'invalid_audio', 'unavailable')
                                     if status in statuses), 'error')
            result['error'] = 'No supported axis produced a prediction; inspect components'
        result['warnings'] = ['Experimental composition; genre and mood scores are not comparable or calibrated',
                              'Genre uses its configured initial segment; Music2Emo uses the complete audio',
                              'Separate emotion taxonomy and J-pop accuracy remain unvalidated']
        if self.genre_classifier is not None:
            result['warnings'][1] = 'Explicit genre backend defines its analysis scope; see analysis.genre; Music2Emo uses complete audio'
            result['warnings'].append('genre_max_duration and genre_warning_threshold apply only to the Wav2Vec2 backend; explicit genre backend receives top_k only')
        if self.emotion_classifier is not None:
            result['warnings'].append('MuQ emotion prompt candidates enabled explicitly; similarities can be negative and are not probabilities')
        return result
