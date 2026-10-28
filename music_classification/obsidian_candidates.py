"""Opt-in local69-label profile; existing adapters and defaults are unchanged."""
from hashlib import sha256
from importlib.resources import files
import json
import math
from pathlib import Path
import tempfile
import time

from .batching import WorkerSession, input_list, batch_result
from .muq_runtime import digest

PROFILE_SHA256='972ab8c78d0f26544937eb0db991252049b62356ffb7e5f5b1c7d248533d1c84'
AXIS_COUNTS={'genre':25,'emotion':22,'mood':22}


def profile_spec():
    raw=files('music_classification').joinpath('resources','obsidian_candidate_prompts_v1.json').read_bytes()
    if sha256(raw).hexdigest()!=PROFILE_SHA256:raise ValueError('Candidate profile checksum mismatch')
    profile=json.loads(raw)
    if {a:len(v) for a,v in profile['axes'].items()}!=AXIS_COUNTS:raise ValueError('Invalid label spaces')
    entries=[r for rows in profile['axes'].values() for r in rows]
    if len({r['id'] for r in entries})!=69:raise ValueError('Duplicate candidate IDs')
    return profile


def ranked_axes(scores, *, top_k=3):
    if isinstance(top_k,bool) or not isinstance(top_k,int) or not 1<=top_k<=22:
        raise ValueError('top_k must be an integer from1 to22')
    profile=profile_spec();expected={r['id'] for rows in profile['axes'].values() for r in rows}
    if (not isinstance(scores,dict) or set(scores)!=expected or any(
        isinstance(s,bool) or not isinstance(s,(int,float)) or not math.isfinite(s) or not -1.000001<=s<=1.000001
        for s in scores.values())):raise ValueError('Expected69 finite raw similarities with exact profile IDs')
    axes={}
    for axis,rows in profile['axes'].items():
        candidates=sorted([{'label':r['id'],'name':r['name'],'score':float(scores[r['id']])} for r in rows],key=lambda c:c['score'],reverse=True)[:top_k]
        axes[axis]={'status':'ok','primary':candidates[0],'candidates':candidates,
          'similarity_scores':{r['id']:float(scores[r['id']]) for r in rows},
          'score_type':'mean_dot_similarity','calibration':'uncalibrated',
          'label_space':profile['version']+'/'+axis,'multilabel':False,'nearest_supported':True,
          'validation':profile['validation']}
    return axes


class ObsidianCandidateClassifier:
    """Rank explicit69 prompts with one MuQ model per serial batch, no training."""
    def __init__(self, *, installation, python_executable, work_directory, cache_directory, timeout=600):
        if isinstance(timeout,bool) or not isinstance(timeout,(int,float)) or not math.isfinite(timeout) or timeout<=0:
            raise ValueError('timeout must be positive finite seconds')
        self.installation=Path(installation).resolve();self.python=Path(python_executable).resolve()
        self.work_directory=Path(work_directory).resolve();self.cache_directory=Path(cache_directory).resolve();self.timeout=timeout

    def classify(self,audio_path,*,top_k=3):
        return self.classify_many([audio_path],top_k=top_k)['results'][0]

    def classify_many(self,audio_paths,*,top_k=3):
        # Validate configuration before starting any worker, including empty batches.
        ranked_axes({r['id']:0. for rows in profile_spec()['axes'].values() for r in rows},top_k=top_k)
        paths=input_list(audio_paths)
        if not paths:raise ValueError('At least one audio path is required')
        paths=[str(Path(p).expanduser().resolve()) for p in paths]
        started=time.perf_counter();results=[]
        session=WorkerSession(self,'obsidian_candidates',worker_script=Path(__file__).with_name('obsidian_candidate_worker.py'))
        try:
            for path in paths:results.append(self._classify(path,session,top_k))
        finally:session.close()
        return batch_result(paths,results,{'obsidian_muq':session.statistics},started)

    def _classify(self,audio_path,session,top_k):
        result={'schema_version':'obsidian_candidates_v1','success':False,'status':'error','final_tags':False,
          'axes':{a:{'status':'unavailable','primary':None,'candidates':[]} for a in AXIS_COUNTS},
          'profile_sha256':PROFILE_SHA256,'model':None,'analysis':None}
        def fail(status,message):
            result.update(status=status,error=message)
            for value in result['axes'].values():value.update(status=status,error=message)
            return result
        path=Path(audio_path)
        try:
            if not path.is_file() or not 0<path.stat().st_size<=100*1024*1024:
                return fail('invalid_audio','Audio missing, empty, or exceeds100MiB')
            original=digest(path)
        except OSError as exc:return fail('invalid_audio',str(exc))
        if not self.python.is_file() or not (self.installation/'runtime-manifest.json').is_file():
            return fail('unavailable','Verified MuQ installation or interpreter missing')
        try:
            self.work_directory.mkdir(parents=True,exist_ok=True)
            with tempfile.TemporaryDirectory(prefix='request-',dir=self.work_directory) as request:
                job=Path(request)
                completed=session.run([str(self.python),'--audio',str(path)],cwd=job,timeout=self.timeout)
                output=job/'result.json'
                if not output.is_file():return fail('error',f'Worker exited{completed.returncode} without a result')
                payload=json.loads(output.read_text(encoding='utf-8'))
                if payload.get('status')!='ok':
                    return fail('invalid_audio' if payload.get('status')=='invalid_audio' else 'error',str(payload.get('error','Worker failed')))
                if completed.returncode!=0:raise ValueError('Worker failed despite a result')
                if payload['model'].get('profile_sha256')!=PROFILE_SHA256:raise ValueError('Wrong worker profile')
                if payload['model'].get('name')!='OpenMuQ/MuQ-MuLan-large':raise ValueError('Wrong worker model')
                if payload['analysis'].get('audio_sha256')!=original or digest(path)!=original:
                    return fail('invalid_audio','Audio changed during inference; result discarded')
                result.update(success=True,status='ok',axes=ranked_axes(payload['scores'],top_k=top_k),
                              model=payload['model'],analysis=payload['analysis'])
                result['warnings']=['Unvalidated profile; no accuracy metrics transfer from common10/emotion8/native56',
                  'Raw similarities may be negative and are not confidence probabilities',
                  'Alternative candidates do not establish simultaneous labels; overlapping concepts may not separate',
                  'Only supplied audio is analyzed; no lyrics semantics, singer identity, cover status or Obsidian writing']
                return result
        except Exception as exc:return fail('error',f'{type(exc).__name__}: {exc}')
