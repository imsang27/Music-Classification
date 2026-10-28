"""Bounded serial worker sessions; each inference keeps its original timeout."""

from contextlib import contextmanager
from copy import copy
import errno
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from uuid import uuid4


def input_list(paths):
    if isinstance(paths, (str, bytes, os.PathLike)):
        raise TypeError('Pass an iterable of audio paths, not a single path')
    return list(paths)


class WorkerSession:
    def __init__(self, adapter, kind, *, worker_script=None):
        self.adapter, self.kind = adapter, kind
        self.script = Path(worker_script or Path(__file__).with_name('batch_worker.py')).resolve()
        self.root = None
        self.process = None
        self.log = None
        self.closed = False
        self._loaded_generation = False
        self.statistics = {'worker_starts': 0, 'model_loads': 0, 'requests': 0, 'timeouts': 0,
                           'worker_failures': 0, 'cleanup_errors': 0, 'cleanup_retries': 0}
        if kind == 'muq_shared':
            self.statistics['axis_requests'] = {'genre': 0, 'emotion': 0}

    def _start(self):
        self.statistics.pop('current_generation_cuda_memory', None)
        requests = self.adapter.work_directory
        requests.mkdir(parents=True, exist_ok=True)
        if self.root is None:
            self.root = (requests / ('batch-' + uuid4().hex)).resolve()
            if self.root.parent != requests.resolve():
                raise ValueError('Unsafe batch session directory')
            self.root.mkdir()
        generation = self.root / ('generation-' + uuid4().hex)
        generation.mkdir()
        self.log = (generation / 'worker.log').open('w', encoding='utf-8')
        command = [str(self.adapter.python), str(self.script), '--kind', self.kind,
                   '--base', str(self.adapter.installation), '--job', str(generation),
                   '--cache', str(self.adapter.cache_directory), '--requests-root', str(requests)]
        try:
            self.process = subprocess.Popen(command, cwd=str(generation), stdin=subprocess.PIPE,
                stdout=self.log, stderr=subprocess.STDOUT, text=True, encoding='utf-8', bufsize=1,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        except Exception:
            self.log.close()
            self.log = None
            raise
        self.statistics['worker_starts'] += 1
        self._loaded_generation = False

    def _stop(self, *, force=False):
        process, self.process = self.process, None
        if process is not None:
            if force and process.poll() is None:
                process.kill()
            if process.stdin is not None:
                try:
                    process.stdin.close()
                except OSError:
                    pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        if self.log is not None:
            self.log.close()
            self.log = None

    def run(self, argv, *, cwd, timeout, axis=None, **kwargs):
        if self.closed:
            raise RuntimeError('Batch session is closed')
        if self.kind == 'muq_shared':
            if axis not in ('genre', 'emotion'):
                raise ValueError('Explicit shared MuQ axis required')
            self.statistics['axis_requests'][axis] += 1
        self.statistics['requests'] += 1
        output = Path(cwd) / 'result.json'
        audio = argv[argv.index('--audio') + 1]
        started = time.monotonic()
        try:
            if self.process is not None and self.process.poll() is not None:
                self._stop()
            if self.process is None:
                self._start()
            command = {'audio': audio, 'output': str(output)}
            if self.kind == 'muq_shared':
                command['axis'] = axis
            self.process.stdin.write(json.dumps(command) + '\n')
            self.process.stdin.flush()
            while True:
                if output.is_file():
                    try:
                        payload = json.loads(output.read_text(encoding='utf-8'))
                        stats_path = output.with_name('worker-stats.json')
                        stats = json.loads(stats_path.read_text(encoding='utf-8')) if stats_path.is_file() else None
                    except PermissionError:
                        # Windows can briefly lock an atomically published file.
                        # Retry only these reads, using the original deadline.
                        pass
                    else:
                        if stats is not None:
                            if stats['model_loaded'] and not self._loaded_generation:
                                self.statistics['model_loads'] += 1
                                self._loaded_generation = True
                            if 'cuda_memory' in stats:
                                self.statistics['current_generation_cuda_memory'] = stats['cuda_memory']
                        if not isinstance(payload, dict) or payload.get('status') == 'error':
                            self.statistics['worker_failures'] += 1
                            self._stop(force=True)
                        return subprocess.CompletedProcess(argv, 0)
                code = self.process.poll()
                if code is not None:
                    self.statistics['worker_failures'] += 1
                    self._stop()
                    return subprocess.CompletedProcess(argv, code)
                if time.monotonic() - started >= timeout:
                    self.statistics['timeouts'] += 1
                    self._stop(force=True)
                    raise subprocess.TimeoutExpired(argv, timeout)
                time.sleep(.025)
        except subprocess.TimeoutExpired:
            raise
        except Exception:
            self.statistics['worker_failures'] += 1
            self._stop(force=True)
            raise

    def close(self):
        if self.closed:
            return
        try:
            self._stop()
        finally:
            self.closed = True
            if self.root is not None and self.root.parent == self.adapter.work_directory.resolve() and self.root.name.startswith('batch-'):
                delays = (.05, .1, .2, .4, .8)
                for attempt in range(len(delays) + 1):
                    # Recheck the resolved boundary before every recursive removal.
                    if (self.root.is_symlink() or getattr(self.root, 'is_junction', lambda: False)()
                            or self.root.resolve().parent != self.adapter.work_directory.resolve()):
                        self.statistics['cleanup_errors'] += 1
                        self.statistics['cleanup_error'] = 'Unsafe batch cleanup path; removal refused'
                        break
                    try:
                        shutil.rmtree(self.root)
                        break
                    except FileNotFoundError:
                        if not self.root.exists():
                            break
                        self.statistics['cleanup_errors'] += 1
                        self.statistics['cleanup_error'] = 'Batch content disappeared but root remains'
                        break
                    except OSError as exc:
                        transient = (getattr(exc, 'winerror', None) in (5, 32, 33, 145)
                                     or exc.errno in (errno.EACCES, errno.EPERM, errno.ENOTEMPTY))
                        if transient and attempt < len(delays):
                            self.statistics['cleanup_retries'] += 1
                            time.sleep(delays[attempt])
                            continue
                        self.statistics['cleanup_errors'] += 1
                        self.statistics['cleanup_error'] = f'{type(exc).__name__}: {exc}'
                        break


@contextmanager
def backend_session(adapter, kind):
    pooled = copy(adapter)
    session = WorkerSession(adapter, kind)
    pooled._worker_session = session
    try:
        yield pooled
    finally:
        session.close()


def backend_many(adapter, kind, audio_paths, **options):
    paths = input_list(audio_paths)
    started = time.perf_counter()
    with backend_session(adapter, kind) as pooled:
        results = [pooled.classify(path, **options) for path in paths]
    return batch_result(paths, results, {kind: pooled._worker_session.statistics}, started)


def batch_result(paths, results, workers, started):
    return {'schema_version': '1.0', 'results': results, 'input_paths': [str(path) for path in paths],
            'summary': {'total': len(results), 'succeeded': sum(bool(r['success']) for r in results),
                        'failed': sum(not r['success'] for r in results)},
            'execution': {'mode': 'serial_reused_model_workers', 'workers': workers,
                          'elapsed_seconds': time.perf_counter() - started}}
