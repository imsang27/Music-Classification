"""HTTP-to-export integration checks without model downloads or inference."""
import hashlib
import gc
import io
import json
from pathlib import Path
import re
import sys
import tempfile
import threading
import time
import unittest
import tomllib
from unittest.mock import patch

from local_tagging_server import create_app
from music_classification.familiar_candidates import PROFILE_SHA256, profile_spec, ranked_axes
from music_classification.youtube_input import YoutubeInputError, environment_status
import tag_audio


def fake_raw(paths, settings):
    profile = profile_spec()
    scores = {row['id']: 0.1 + index * 0.001 for rows in profile['axes'].values()
              for index, row in enumerate(rows)}
    results = []
    for path in paths:
        success = Path(path).read_bytes() != b'broken'
        results.append({'success': success, 'status': 'ok' if success else 'invalid_audio',
                        'axes': ranked_axes(scores) if success else {}, 'error': 'invalid audio' if not success else None})
    return {'profile_sha256': PROFILE_SHA256, 'vocabulary_version': profile['version'],
            'batch': {'input_paths': [str(p) for p in paths], 'results': results}}


def fake_run(*args, **kwargs):
    with patch('music_classification.familiar_candidates.classify_paths', side_effect=fake_raw):
        return tag_audio.run(*args, **kwargs)


class LocalWebTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Earlier GUI tests destroy Tk windows; collect their cycles on the
        # main thread before any HTTP inference worker can trigger collection.
        gc.collect()

    def setUp(self):
        dependency_patch = patch('local_tagging_server.environment_status', return_value={'ready': True, 'message': 'ready'})
        dependency_patch.start()
        self.addCleanup(dependency_patch.stop)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        installation = self.root / 'model'
        installation.mkdir()
        (installation / 'runtime-manifest.json').write_text('{}', encoding='utf-8')
        self.settings = self.root / 'settings.json'
        self.settings.write_text(json.dumps({'version': 1, 'muq': {
            'installation': str(installation), 'python_executable': sys.executable}}), encoding='utf-8')
        self.app = create_app(self.settings, root=self.root, runner=fake_run, youtube_runner=self.youtube)
        self.addCleanup(self.app.extensions['tagging_jobs'].close)
        self.client = self.app.test_client()
        self.token = re.search(r'name="app-token" content="([^"]+)"', self.client.get('/').get_data(as_text=True))[1]
        self.headers = {'X-App-Token': self.token}
        self.youtube_calls = []

    def youtube(self, url, target, settings):
        self.youtube_calls.append(url)
        with tempfile.TemporaryDirectory(dir=self.root) as folder:
            path = Path(folder) / 'audio.wav'; path.write_bytes(b'fixture')
            source = {'kind': 'youtube', 'url': url, 'title': '<test> 곡',
                      'audio_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
            result = fake_run([path], target, settings, profile='familiar', source=source)
        status = Path(target) / 'status.json'
        value = json.loads(status.read_text(encoding='utf-8'))
        value['temporary_audio_removed'] = True
        status.write_text(json.dumps(value), encoding='utf-8')
        return result

    def submit(self, names=('노래.wav',), contents=None):
        contents = contents or [b'fixture'] * len(names)
        return self.client.post('/api/jobs/files', data={'audio': [
            (io.BytesIO(content), name) for name, content in zip(names, contents)]}, headers=self.headers)

    def wait(self, response):
        self.assertEqual(response.status_code, 202, response.get_data(as_text=True))
        url = '/api/jobs/' + response.json['id']
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            result = self.client.get(url)
            if result.json['status'] in ('completed', 'partial', 'failed'):
                return result.json
            time.sleep(.01)
        self.fail('job did not finish')

    def test_multiple_uploads_duplicate_names_export_and_cleanup(self):
        job = self.wait(self.submit(('노래.wav', '노래.wav')))
        self.assertEqual(job['status'], 'completed')
        rows = job['result']['results']
        self.assertEqual([r['display_name'] for r in rows], ['노래.wav'] * 2)
        self.assertTrue(all(r['audio_path'] is None for r in rows))
        for row in rows:
            self.assertEqual(len(row['candidates']['emotion']), 3)
            self.assertEqual(len(row['candidates']['mood']), 3)
            self.assertEqual(len(row['labels']['emotion']), 1)
        self.assertEqual(list((self.root / 'work/local-web-uploads').iterdir()), [])
        downloaded = self.client.get(job['downloads']['태깅결과.md'])
        self.assertEqual(downloaded.status_code, 200)
        self.assertIn('attachment', downloaded.headers['Content-Disposition'])
        self.assertIn('노래.wav', downloaded.get_data(as_text=True))
        status = self.client.get(job['downloads']['status.json']).json
        self.assertTrue(status['temporary_uploads_removed'])
        self.assertEqual(self.client.get('/api/status').json['active_job'], None)

    def test_partial_result_has_no_fabricated_tags(self):
        job = self.wait(self.submit(('좋음.wav', '오류.wav'), [b'ok', b'broken']))
        self.assertEqual(job['status'], 'partial')
        self.assertIsNone(job['result']['results'][1]['labels'])
        self.assertEqual(self.client.get(job['downloads']['tags.json']).status_code, 200)

    def test_failure_cleans_audio_and_allows_next_job(self):
        with patch.object(self.app.extensions['tagging_jobs'], 'runner', side_effect=RuntimeError('fixture failure')):
            job = self.wait(self.submit())
        self.assertEqual(job['status'], 'failed')
        self.assertNotIn('downloads', job)
        self.assertEqual(list((self.root / 'work/local-web-uploads').iterdir()), [])
        self.assertEqual(self.wait(self.submit())['status'], 'completed')

    def test_reject_empty_unsupported_too_many_or_oversize_uploads(self):
        for names, content in [(('a.wav',), [b'']), (('a.exe',), [b'a']),
                               (tuple(f'{i}.wav' for i in range(21)), None)]:
            self.assertEqual(self.submit(names, content).status_code, 400)
        self.app.config['MAX_FILE_BYTES'] = 2
        self.assertEqual(self.submit(contents=[b'long']).status_code, 413)
        self.assertEqual(list((self.root / 'work/local-web-uploads').iterdir()), [])
        self.assertIsNone(self.client.get('/api/status').json['active_job'])
        self.app.config['MAX_CONTENT_LENGTH'] = 20
        self.assertEqual(self.submit().status_code, 413)

    def test_cross_origin_bad_host_and_missing_token_rejected(self):
        self.assertEqual(self.client.post('/api/jobs/youtube', json={'url': 'x'}).status_code, 403)
        self.assertEqual(self.client.get('/', headers={'Host': 'evil.example'}).status_code, 403)
        self.assertEqual(self.client.get('/', environ_overrides={'REMOTE_ADDR': '10.0.0.2'}).status_code, 403)
        self.assertEqual(self.client.post('/api/jobs/youtube', json={'url': 'x'},
            headers={**self.headers, 'Origin': 'https://evil.example'}).status_code, 403)
        self.assertEqual(self.client.get('/', headers={'Origin': 'null'}).status_code, 403)
        self.assertEqual(self.client.get('/').headers['Cache-Control'], 'no-store')
        self.assertIn("frame-ancestors 'none'", self.client.get('/').headers['Content-Security-Policy'])

    def test_youtube_normalization_and_validation(self):
        response = self.client.post('/api/jobs/youtube', json={'url': 'https://youtu.be/-VKIqrvVOpo?list=ignored'}, headers=self.headers)
        job = self.wait(response)
        self.assertEqual(self.youtube_calls, ['https://www.youtube.com/watch?v=-VKIqrvVOpo'])
        self.assertEqual(job['result']['results'][0]['display_name'], '<test> 곡')
        self.assertTrue(self.client.get(job['downloads']['status.json']).json['temporary_audio_removed'])
        for value in [None, [], {'url': 'http://localhost/private'}, {'url': 'https://youtube.com/playlist?list=abc'}]:
            self.assertEqual(self.client.post('/api/jobs/youtube', json=value, headers=self.headers).status_code, 400)

    def test_only_one_job_runs_at_a_time(self):
        entered, release = threading.Event(), threading.Event()
        def blocked(*args, **kwargs):
            entered.set(); release.wait(3)
            return fake_run(*args, **kwargs)
        with patch.object(self.app.extensions['tagging_jobs'], 'runner', side_effect=blocked):
            first = self.submit()
            try:
                self.assertTrue(entered.wait(2))
                self.assertEqual(self.submit().status_code, 409)
                self.assertEqual(self.client.get('/api/status').json['active_job'], first.json['id'])
            finally:
                release.set()
            self.assertEqual(self.wait(first)['status'], 'completed')

    def test_download_whitelist_and_input_filename_is_not_a_path(self):
        job = self.wait(self.submit(('../../한국어.wav',)))
        self.assertEqual(job['result']['results'][0]['display_name'], '한국어.wav')
        self.assertEqual(self.client.get(f'/api/jobs/{job["id"]}/downloads/settings.json').status_code, 404)
        self.assertEqual(self.client.get('/api/jobs/missing').status_code, 404)
        self.assertFalse((self.root / '한국어.wav').exists())

    def test_missing_model_is_explained_without_starting_a_job(self):
        self.settings.unlink()
        self.assertFalse(self.client.get('/api/status').json['model']['ready'])
        self.assertEqual(self.submit().status_code, 503)
        self.assertIsNone(self.client.get('/api/status').json['active_job'])

    def test_youtube_environment_is_reported_before_starting_a_job(self):
        missing = {'ready': False, 'message': 'YouTube 연결 도구가 없습니다.'}
        with patch('local_tagging_server.environment_status', return_value=missing):
            self.assertEqual(self.client.get('/api/status').json['youtube'], missing)
            response = self.client.post('/api/jobs/youtube', json={'url': 'https://youtu.be/-VKIqrvVOpo'}, headers=self.headers)
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json['error'], missing['message'])
        self.assertEqual(self.youtube_calls, [])
        self.assertIsNone(self.client.get('/api/status').json['active_job'])

    def test_specific_youtube_error_is_visible_and_job_released(self):
        message = 'YouTube 오디오를 가져오지 못했습니다. 공개 영상 여부를 확인해주세요.'
        with patch.object(self.app.extensions['tagging_jobs'], 'youtube_runner', side_effect=YoutubeInputError(message)):
            job = self.wait(self.client.post('/api/jobs/youtube', json={'url': 'https://youtu.be/-VKIqrvVOpo'}, headers=self.headers))
        self.assertEqual(job['status'], 'failed')
        self.assertEqual(job['message'], message)
        self.assertIsNone(self.client.get('/api/status').json['active_job'])

    def test_web_install_includes_link_dependencies_and_detects_missing_tools(self):
        project = tomllib.loads((Path(__file__).resolve().parents[1] / 'pyproject.toml').read_text(encoding='utf-8'))
        extras = project['project']['optional-dependencies']
        self.assertTrue(set(extras['youtube']) <= set(extras['web']))
        with patch('music_classification.youtube_input.importlib.util.find_spec', return_value=None):
            state = environment_status()
            self.assertFalse(state['ready'])
            self.assertIn('YouTube', state['message'])
        with patch('music_classification.youtube_input.importlib.util.find_spec', return_value=object()), \
             patch('music_classification.youtube_input.ffmpeg_executable', return_value=self.root / 'missing-ffmpeg'):
            self.assertIn('FFmpeg', environment_status()['message'])

    def test_legacy_tab_uses_original_runner_and_exports_unsupported_axes(self):
        def original(paths, target, settings):
            folder, code = fake_run(paths, target, settings, profile='familiar')
            path = folder / 'tags.json'
            tags = json.loads(path.read_text(encoding='utf-8'))
            tags.update(schema='music_legacy_genre_tags_v1', classifier='wav2vec2')
            for row in tags['results']:
                row['labels'].update(emotion=[], mood=[])
                row['candidates'].update(emotion=[], mood=[])
            path.write_text(json.dumps(tags), encoding='utf-8')
            return folder, code
        with patch('local_tagging_server.legacy_genre_web.status', return_value={'ready': True}), \
             patch('local_tagging_server.legacy_genre_web.run', side_effect=original) as runner:
            response = self.client.post('/api/jobs/files', data={'method':'wav2vec2',
                'audio': (io.BytesIO(b'original audio'), '기존.wav')}, headers=self.headers)
            job = self.wait(response)
        self.assertEqual(runner.call_count, 1)
        self.assertEqual(job['method'], 'wav2vec2')
        self.assertEqual(job['result']['results'][0]['labels']['emotion'], [])
        self.assertIn('미지원', self.client.get(job['downloads']['태깅결과.md']).get_data(as_text=True))

    def test_legacy_youtube_dispatch_and_unavailable_model_are_explicit(self):
        url = 'https://www.youtube.com/watch?v=-VKIqrvVOpo'
        with patch('local_tagging_server.legacy_genre_web.status', return_value={'ready': True}), \
             patch('local_tagging_server.legacy_genre_web.run_youtube', side_effect=self.youtube) as runner:
            job = self.wait(self.client.post('/api/jobs/youtube', json={'url':url,'method':'wav2vec2'}, headers=self.headers))
        self.assertEqual(runner.call_args.args[0], url)
        self.assertEqual(job['method'], 'wav2vec2')
        with patch('local_tagging_server.legacy_genre_web.status', return_value={'ready': False, 'message':'기존 모델 없음'}):
            response = self.client.post('/api/jobs/youtube', json={'url':url,'method':'wav2vec2'}, headers=self.headers)
        self.assertEqual(response.status_code, 503)
        self.assertEqual(self.client.post('/api/jobs/youtube', json={'url':url,'method':'unknown'}, headers=self.headers).status_code, 400)

    def test_original_entry_point_uses_same_ai_screen(self):
        from webapp.app import app
        self.addCleanup(app.extensions['tagging_jobs'].close)
        page = app.test_client().get('/').get_data(as_text=True)
        self.assertIn('AI 분류 방식', page)
        self.assertIn('sticky-header', page)
        self.assertIn('Wav2Vec2', page)
        self.assertIn('MuQ', page)


if __name__ == '__main__':
    unittest.main()
