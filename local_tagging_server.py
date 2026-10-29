"""Loopback-only browser UI for the existing familiar audio classifier."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import logging
from pathlib import Path
import secrets
import shutil
import tempfile
import threading
from urllib.parse import urlsplit
from uuid import uuid4
import webbrowser
import urllib.request

from flask import Flask, abort, jsonify, render_template, request, send_file
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge
from werkzeug.serving import make_server

from music_classification.app_paths import default_settings_path
from music_classification.settings import ModelSettings
from music_classification.youtube_input import canonical_url, environment_status, YoutubeInputError
from music_classification import legacy_genre_web

ROOT = Path(__file__).resolve().parent
EXTENSIONS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac'}
EXPORTS = {'tags.json', 'classification.json', '태깅결과.md', 'status.json'}
MAX_FILE_BYTES = 100 * 1024 * 1024
MAX_REQUEST_BYTES = 500 * 1024 * 1024


def model_status(settings_path):
    try:
        settings = ModelSettings.from_file(settings_path)
        if settings.muq is None:
            raise ValueError('MuQ 설정 없음')
        if not settings.muq.python_executable.is_file() or not (settings.muq.installation / 'runtime-manifest.json').is_file():
            raise ValueError('MuQ 실행 환경 없음')
        return {'ready': True, 'message': '분류 준비가 되었습니다.'}
    except (OSError, ValueError, TypeError):
        return {'ready': False, 'message': '모델 준비가 필요합니다. 설치 안내에 따라 모델을 설치한 뒤 서버를 다시 실행해주세요.'}


class Jobs:
    def __init__(self, root, settings, runner, youtube_runner):
        self.root, self.settings = Path(root), Path(settings)
        self.runner, self.youtube_runner = runner, youtube_runner
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='music-tagging')
        self.lock = threading.Lock()
        self.active = None
        self.items = {}

    def reserve(self):
        with self.lock:
            if self.active:
                abort(409, description='이미 분류 중입니다. 현재 작업이 끝난 뒤 다시 시도해주세요.')
            job_id = uuid4().hex
            # Keep bounded session history; result files remain on disk.
            while len(self.items) >= 20:
                del self.items[next(iter(self.items))]
            self.items[job_id] = {'id': job_id, 'status': 'preparing', 'message': '입력을 준비하고 있습니다.'}
            self.active = job_id
            return job_id

    def update(self, job_id, **values):
        with self.lock:
            self.items[job_id].update(values)

    def release(self, job_id):
        with self.lock:
            if self.active == job_id:
                self.active = None

    def get(self, job_id):
        with self.lock:
            item = self.items.get(job_id)
            if item is None:
                abort(404, description='이 서버 실행 중에 생성한 작업을 찾을 수 없습니다.')
            return deepcopy(item)

    def start(self, job_id, *, upload=None, names=None, url=None, method='familiar'):
        self.pool.submit(self.execute, job_id, upload, names, url, method)

    def execute(self, job_id, upload, names, url, method):
        result = None
        try:
            self.update(job_id, status='running', message='오디오를 분석하고 있습니다. 처음에는 모델 준비에 시간이 걸릴 수 있습니다.')
            target = self.root / 'outputs' / 'local-web' / job_id
            if method == 'wav2vec2':
                if url:
                    folder, code = legacy_genre_web.run_youtube(url, target, self.settings)
                else:
                    folder, code = legacy_genre_web.run(sorted(Path(upload).iterdir()), target, self.settings)
            elif url:
                folder, code = self.youtube_runner(url, target, self.settings)
            else:
                paths = sorted(Path(upload).iterdir())
                folder, code = self.runner(paths, target, self.settings, profile='familiar')
            folder = Path(folder).resolve()
            if folder != target.resolve():
                raise ValueError('결과 폴더가 예상 위치와 다릅니다.')
            result = json.loads((folder / 'tags.json').read_text(encoding='utf-8'))
            if upload:
                if len(result['results']) != len(names):
                    raise ValueError('입력과 결과의 수가 다릅니다.')
                for row, name in zip(result['results'], names):
                    row.update(display_name=name, audio_path=None, input_kind='local_upload')
                result['input_retention'] = 'temporary_uploads_deleted_after_processing'
                (folder / 'tags.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
                from tag_audio import markdown as familiar_markdown
                markdown = legacy_genre_web.markdown if method == 'wav2vec2' else familiar_markdown
                (folder / '태깅결과.md').write_text(markdown(result), encoding='utf-8')
            successful = sum(row['labels'] is not None for row in result['results'])
            final = {'status': 'completed' if code == 0 else 'partial',
                     'method': method,
                     'message': f'{successful}/{len(result["results"])}개 분류 완료', 'result': result,
                     'downloads': {name: f'/api/jobs/{job_id}/downloads/{name}' for name in sorted(EXPORTS)}}
        except YoutubeInputError as exc:
            logging.exception('Local YouTube job %s failed', job_id)
            final = {'status': 'failed', 'message': str(exc)}
        except Exception:
            logging.exception('Local web tagging job %s failed', job_id)
            final = {'status': 'failed', 'message': '분류하지 못했습니다. 모델 설정과 입력한 음원·공개 링크를 확인해주세요.'}
        finally:
            try:
                if upload:
                    safe_parent = (self.root / 'work' / 'local-web-uploads').resolve()
                    if Path(upload).resolve().parent != safe_parent:
                        raise ValueError('Unexpected upload directory')
                    shutil.rmtree(upload)
                # Publish completion only after temporary upload cleanup.
                if result is not None:
                    status_path = self.root / 'outputs' / 'local-web' / job_id / 'status.json'
                    status = json.loads(status_path.read_text(encoding='utf-8'))
                    status['temporary_uploads_removed'] = bool(upload)
                    status_path.write_text(json.dumps(status, ensure_ascii=False) + '\n', encoding='utf-8')
            except Exception:
                logging.exception('Local web cleanup failed for %s', job_id)
                final = {'status': 'failed', 'message': '임시 파일 정리를 완료하지 못했습니다. 서버 창의 오류 내용을 확인해주세요.'}
            with self.lock:
                self.items[job_id].update(final)
                if self.active == job_id:
                    self.active = None

    def close(self):
        self.pool.shutdown(wait=True)


def create_app(settings_path=None, *, root=ROOT, runner=None, youtube_runner=None):
    from tag_audio import run, run_youtube
    app = Flask(__name__, template_folder=str(ROOT / 'local_web' / 'templates'),
                static_folder=str(ROOT / 'local_web' / 'static'))
    app.config.update(MAX_CONTENT_LENGTH=MAX_REQUEST_BYTES + 1024 * 1024,
                      MAX_FILE_BYTES=MAX_FILE_BYTES, MAX_FORM_MEMORY_SIZE=1024 * 1024,
                      MAX_FORM_PARTS=30)
    settings_path = Path(settings_path or default_settings_path(root))
    token = secrets.token_urlsafe(32)
    jobs = Jobs(root, settings_path, runner or run, youtube_runner or run_youtube)
    app.extensions['tagging_jobs'] = jobs

    @app.before_request
    def local_only():
        try:
            host = urlsplit('http://' + request.host).hostname
        except ValueError:
            host = None
        if request.remote_addr not in ('127.0.0.1', '::1') or host not in ('127.0.0.1', 'localhost', '::1'):
            abort(403, description='이 프로그램은 같은 컴퓨터에서만 사용할 수 있습니다.')
        origin = request.headers.get('Origin')
        if origin and origin != request.host_url.rstrip('/'):
            abort(403, description='프로그램 화면에서 요청해주세요.')
        if request.method not in ('GET', 'HEAD', 'OPTIONS') and not secrets.compare_digest(request.headers.get('X-App-Token', ''), token):
            abort(403, description='프로그램 화면을 새로고침한 뒤 다시 시도해주세요.')

    @app.after_request
    def response_headers(response):
        response.headers['Cache-Control'] = 'no-store'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
        return response

    @app.errorhandler(HTTPException)
    def request_error(exc):
        message = '한 번에 올릴 수 있는 파일 크기를 초과했습니다.' if isinstance(exc, RequestEntityTooLarge) else exc.description
        return jsonify(error=message), exc.code

    @app.errorhandler(Exception)
    def unexpected_error(exc):
        logging.exception('Local web request failed')
        return jsonify(error='처리하지 못했습니다. 서버 창의 오류 내용을 확인해주세요.'), 500

    @app.get('/')
    def index():
        return render_template('index.html', token=token)

    @app.get('/api/status')
    def status():
        with jobs.lock:
            active = jobs.active
        return jsonify(application='music-classification-local-web-v1', model=model_status(settings_path),
                       models={'familiar': model_status(settings_path), 'wav2vec2': legacy_genre_web.status(settings_path)},
                       youtube=environment_status(), active_job=active)

    def ready(method):
        if method not in ('familiar', 'wav2vec2'):
            abort(400, description='지원하지 않는 AI 분류 방식입니다.')
        state = legacy_genre_web.status(settings_path) if method == 'wav2vec2' else model_status(settings_path)
        if not state['ready']:
            abort(503, description=state['message'])

    @app.post('/api/jobs/files')
    def files_job():
        method = request.form.get('method', 'familiar')
        ready(method)
        files = request.files.getlist('audio')
        if not 1 <= len(files) <= 20:
            abort(400, description='음원 파일을 1~20개 선택해주세요.')
        names = []
        for file in files:
            name = (file.filename or '').replace('\\', '/').rsplit('/', 1)[-1]
            if not name or len(name) > 200 or any(ord(c) < 32 for c in name) or Path(name).suffix.lower() not in EXTENSIONS:
                abort(400, description='WAV·MP3·FLAC·OGG·M4A·AAC 음원을 선택해주세요.')
            names.append(name)
        job_id = jobs.reserve()
        folder = None
        try:
            parent = Path(root) / 'work' / 'local-web-uploads'
            parent.mkdir(parents=True, exist_ok=True)
            folder = Path(tempfile.mkdtemp(prefix='upload-', dir=parent))
            total = 0
            for index, (file, name) in enumerate(zip(files, names)):
                size = 0
                with (folder / f'{index:03d}{Path(name).suffix.lower()}').open('wb') as stream:
                    while chunk := file.stream.read(1024 * 1024):
                        size += len(chunk)
                        total += len(chunk)
                        if size > app.config['MAX_FILE_BYTES'] or total > MAX_REQUEST_BYTES:
                            abort(413)
                        stream.write(chunk)
                if size == 0:
                    abort(400, description='빈 파일은 분석할 수 없습니다.')
            jobs.start(job_id, upload=folder, names=names, method=method)
        except Exception:
            try:
                if folder is not None:
                    shutil.rmtree(folder)
            finally:
                jobs.update(job_id, status='failed', message='입력을 준비하지 못했습니다.')
                jobs.release(job_id)
            raise
        return jsonify(id=job_id), 202

    @app.post('/api/jobs/youtube')
    def youtube_job():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            abort(400, description='YouTube 영상 한 곡의 링크를 입력해주세요.')
        method = data.get('method', 'familiar')
        ready(method)
        try:
            url = canonical_url(data.get('url'))
        except ValueError as exc:
            abort(400, description=str(exc))
        youtube_state = environment_status()
        if not youtube_state['ready']:
            abort(503, description=youtube_state['message'])
        job_id = jobs.reserve()
        try:
            jobs.start(job_id, url=url, method=method)
        except Exception:
            jobs.release(job_id)
            jobs.update(job_id, status='failed', message='작업을 시작하지 못했습니다.')
            raise
        return jsonify(id=job_id), 202

    @app.get('/api/jobs/<job_id>')
    def job(job_id):
        return jsonify(jobs.get(job_id))

    @app.get('/api/jobs/<job_id>/downloads/<name>')
    def download(job_id, name):
        item = jobs.get(job_id)
        if name not in EXPORTS or item['status'] not in ('completed', 'partial'):
            abort(404)
        return send_file(Path(root).resolve() / 'outputs' / 'local-web' / job_id / name,
                         as_attachment=True, download_name=name)

    return app


def main(argv=None):
    parser = argparse.ArgumentParser(description='음악 태깅을 이 컴퓨터의 브라우저에서 실행합니다.')
    parser.add_argument('--settings', type=Path)
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args(argv)
    if not 1024 <= args.port <= 65535:
        parser.error('포트는 1024~65535 중 하나를 선택해주세요.')
    app = create_app(args.settings)
    url = f'http://127.0.0.1:{args.port}'
    from tag_audio import print_status
    try:
        server = make_server('127.0.0.1', args.port, app, threaded=True)
    except (OSError, SystemExit):
        app.extensions['tagging_jobs'].close()
        try:
            with urllib.request.urlopen(url + '/api/status', timeout=2) as response:
                existing = json.load(response)
            if existing.get('application') == 'music-classification-local-web-v1':
                print_status('이미 실행 중인 음악 분류 화면을 엽니다: ' + url)
                if not args.no_browser:
                    webbrowser.open(url)
                return 0
        except (OSError, ValueError):
            pass
        print_status('서버를 열지 못했습니다. --port로 다른 번호를 선택해주세요.')
        return 2
    print_status(f'음악 분류가 준비되었습니다: {url}\n종료하려면 이 창에서 Ctrl+C를 누르세요.')
    if not args.no_browser:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print_status('진행 중인 작업을 정리하고 종료합니다.')
    finally:
        server.server_close()
        app.extensions['tagging_jobs'].close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
