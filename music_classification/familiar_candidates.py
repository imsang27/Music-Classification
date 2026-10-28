"""Separate familiar-word profile; original69 profile and workers stay intact."""
from hashlib import sha256
from importlib.resources import files
import json
import math
from pathlib import Path
import tempfile
import time

from .batching import WorkerSession, input_list, batch_result
from .muq_runtime import digest
from .obsidian_candidates import ObsidianCandidateClassifier

PROFILE_SHA256='fe379859e27d8fb74157481de9e0d915c79446ee63b128d02d8280db678f9408'
DISPLAY_POLICY_SHA256='bf1a9d1f6b2e9afe7609014d2404d081fb2ca0b3c4c4cc22acf094466e0bec73'
COUNTS={'genre':25,'emotion':9,'mood':11}


def profile_spec():
    raw=files('music_classification').joinpath('resources/familiar_candidate_prompts_v2.json').read_bytes()
    if sha256(raw).hexdigest()!=PROFILE_SHA256:
        raise ValueError('Familiar vocabulary checksum mismatch')
    profile=json.loads(raw)
    if {a:len(v) for a,v in profile['axes'].items()}!=COUNTS:
        raise ValueError('Wrong familiar vocabulary')
    rows=[r for rows in profile['axes'].values() for r in rows]
    if len({r['id'] for r in rows})!=sum(COUNTS.values()):
        raise ValueError('Duplicate familiar IDs')
    display_raw=files('music_classification').joinpath('resources/familiar_display_names_v1.json').read_bytes()
    if sha256(display_raw).hexdigest()!=DISPLAY_POLICY_SHA256:
        raise ValueError('Display policy checksum mismatch')
    display=json.loads(display_raw)
    indexed={r['id']:r for r in rows}
    for label_id,override in display['overrides'].items():
        if label_id not in indexed or indexed[label_id]['name']!=override['previous_name']:
            raise ValueError('Display override does not match the fixed label')
        indexed[label_id]['name']=override['name']
        indexed[label_id]['accepted_input_aliases']=[override['previous_name'],override['name']]
    profile['display_policy_version']=display['version']
    profile['display_policy_sha256']=DISPLAY_POLICY_SHA256
    return profile


def ranked_axes(scores,top_k=3):
    if isinstance(top_k,bool) or not isinstance(top_k,int) or not 1<=top_k<=9:
        raise ValueError('top_k must be1..9')
    profile=profile_spec()
    expected={r['id'] for rows in profile['axes'].values() for r in rows}
    if (not isinstance(scores,dict) or set(scores)!=expected or any(
            isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or not -1.000001<=v<=1.000001
            for v in scores.values())):
        raise ValueError('Expected45 finite raw similarities')
    axes={}
    for axis,rows in profile['axes'].items():
        candidates=sorted([{'label':r['id'],'name':r['name'],'score':float(scores[r['id']])} for r in rows],
                          key=lambda r:r['score'],reverse=True)[:top_k]
        axes[axis]={'status':'ok','primary':candidates[0],'candidates':candidates,
                    'similarity_scores':{r['id']:float(scores[r['id']]) for r in rows},
                    'score_type':'mean_dot_similarity','calibration':'uncalibrated',
                    'label_space':profile['version']+'/'+axis,'multilabel':False,
                    'nearest_supported':True,'validation':profile['validation']}
    return axes


class FamiliarCandidateClassifier(ObsidianCandidateClassifier):
    def classify_many(self,audio_paths,*,top_k=3):
        ranked_axes({r['id']:0. for rows in profile_spec()['axes'].values() for r in rows},top_k)
        paths=[str(Path(p).expanduser().resolve()) for p in input_list(audio_paths)]
        if not paths:raise ValueError('At least one audio path required')
        started=time.perf_counter();results=[]
        worker=WorkerSession(self,'familiar_candidates',worker_script=Path(__file__).with_name('familiar_candidate_worker.py'))
        try:
            for path in paths:results.append(self._classify(path,worker,top_k))
        finally:worker.close()
        return batch_result(paths,results,{'familiar_muq':worker.statistics},started)

    def _classify(self,audio_path,session,top_k):
        result={'schema_version':'familiar_candidates_v1','success':False,'status':'error','final_tags':False,
                'axes':{a:{'status':'unavailable','primary':None,'candidates':[]} for a in COUNTS},
                'profile_sha256':PROFILE_SHA256,'model':None,'analysis':None}
        def fail(status,message):
            result.update(status=status,error=message)
            for axis in result['axes'].values():axis.update(status=status,error=message)
            return result
        path=Path(audio_path)
        try:
            if not path.is_file() or not 0<path.stat().st_size<=100*1024*1024:
                return fail('invalid_audio','Audio missing, empty, or exceeds100MiB')
            original=digest(path)
            if not self.python.is_file() or not (self.installation/'runtime-manifest.json').is_file():
                return fail('unavailable','Verified MuQ installation or interpreter missing')
            self.work_directory.mkdir(parents=True,exist_ok=True)
            with tempfile.TemporaryDirectory(prefix='request-',dir=self.work_directory) as request:
                job=Path(request)
                completed=session.run([str(self.python),'--audio',str(path)],cwd=job,timeout=self.timeout)
                output=job/'result.json'
                if not output.is_file():return fail('error','Worker returned no result')
                payload=json.loads(output.read_text(encoding='utf-8'))
                if payload.get('status')!='ok':
                    return fail('invalid_audio' if payload.get('status')=='invalid_audio' else 'error',str(payload.get('error','Worker failed')))
                if completed.returncode!=0:raise ValueError('Worker failed despite result')
                if payload['model'].get('profile_sha256')!=PROFILE_SHA256 or payload['model'].get('name')!='OpenMuQ/MuQ-MuLan-large':
                    raise ValueError('Wrong worker profile/model')
                if payload['analysis'].get('audio_sha256')!=original or digest(path)!=original:
                    return fail('invalid_audio','Audio changed during inference')
                result.update(success=True,status='ok',axes=ranked_axes(payload['scores'],top_k),
                              model=payload['model'],analysis=payload['analysis'])
                result['warnings']=['Fixed familiar vocabulary; no arbitrary tag generation.',
                    'Not yet reviewed against familiar songs; scores are not probabilities.',
                    'One nearest label per axis; alternatives do not establish simultaneous tags.']
                return result
        except Exception as exc:return fail('error',f'{type(exc).__name__}: {exc}')


def classify_paths(paths,settings_path):
    from .settings import ModelSettings
    root=Path(__file__).resolve().parents[1]
    settings=ModelSettings.from_file(settings_path)
    if settings.muq is None:raise ValueError('MuQ 모델 설정이 없습니다.')
    options=settings.muq.kwargs()
    options['work_directory']=root/'work/familiar-profile-requests'
    return {'application_schema':'familiar_candidates_v1','final_tags':False,
            'accuracy_validated':False,'profile_sha256':PROFILE_SHA256,
            'vocabulary_version':profile_spec()['version'],
            'display_policy_sha256':DISPLAY_POLICY_SHA256,
            'batch':FamiliarCandidateClassifier(**options).classify_many(paths,top_k=3)}
