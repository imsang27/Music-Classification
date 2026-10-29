"""Browser-safe replacements for the old manual and tempo-rule workflows."""
import json
from pathlib import Path
import subprocess
import sys
import importlib.util

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def rule_python(settings_path):
    from music_classification.settings import ModelSettings
    try:settings=ModelSettings.from_file(settings_path)
    except FileNotFoundError:return Path(sys.executable)
    return settings.muq.python_executable if settings.muq else Path(sys.executable)


def rule_status(settings_path):
    try:
        python=rule_python(settings_path)
        if not python.is_file() or (python.resolve()==Path(sys.executable).resolve() and importlib.util.find_spec('librosa') is None):
            return {'ready':False,'message':'규칙 분류용 오디오 실행 환경이 필요합니다. desktop 설치 항목을 준비해주세요.'}
        return {'ready':True,'message':'기존 템포 규칙의 참고 분류입니다.'}
    except (ValueError,OSError):return {'ready':False,'message':'규칙 분류의 실행 환경 설정을 확인해주세요.'}


def vocabulary():
    from music_classification.familiar_candidates import profile_spec
    return {axis: [row['name'] for row in rows] for axis, rows in profile_spec()['axes'].items()}


def manual_values(data):
    if not isinstance(data, dict):
        raise ValueError('곡 이름과 분류 항목을 입력해주세요.')
    title = data.get('title')
    if not isinstance(title, str) or not title.strip() or len(title) > 200 or any(ord(c) < 32 for c in title):
        raise ValueError('곡 이름을 1~200자로 입력해주세요.')
    labels = data.get('labels')
    choices = vocabulary()
    if not isinstance(labels, dict) or set(labels) != set(choices):
        raise ValueError('장르·감정·분위기 항목을 확인해주세요.')
    for axis, values in labels.items():
        maximum = 1 if axis == 'genre' else 3
        if (not isinstance(values, list) or not 1 <= len(values) <= maximum
                or any(not isinstance(v, str) or v not in choices[axis] for v in values)
                or len(set(values)) != len(values)):
            raise ValueError('장르는 1개, 감정·분위기는 각각 1~3개를 목록에서 선택해주세요.')
    return {'title': title.strip(), 'labels': labels}


def markdown(tags):
    manual = tags['classifier'] == 'manual'
    lines = ['# 직접 분류 결과' if manual else '# 규칙 분류 참고 결과', '',
        '사용자가 선택한 값이며 모델 예측·학습용 정답 인증이 아닙니다.' if manual else
        '기존 템포 규칙의 참고 결과입니다. 음악적 장르·감정의 정확한 판정이나 확률이 아닙니다.', '',
        '| 곡 | 장르 | 감정 | 분위기 |', '|---|---|---|---|']
    for row in tags['results']:
        title = (row.get('display_name') or str(row.get('audio_path', '음원'))).replace('|', '\\|')
        cells = [' · '.join(row['labels'][a]) or '미지원' for a in ('genre','emotion','mood')] if row['labels'] else ['처리 실패','—','—']
        lines.append('| ' + title + ' | ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines) + '\n'


def save(target, tags, raw):
    target = Path(target)
    target.mkdir(parents=True, exist_ok=False)
    for name, value in [('tags.json', tags), ('classification.json', raw)]:
        (target/name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    (target/'태깅결과.md').write_text(markdown(tags), encoding='utf-8')
    good = sum(row['labels'] is not None for row in tags['results'])
    (target/'status.json').write_text(json.dumps({'status':'completed','successful':good,'failed':len(tags['results'])-good}), encoding='utf-8')
    return target, 0 if good == len(tags['results']) else 1


def run_manual(data, target):
    data = manual_values(data)
    row = {'display_name':data['title'], 'audio_path':None, 'status':'ok', 'labels':data['labels'],
           'candidates':{a:[{'name':name,'rank':i,'representative':True,'score':None}
                            for i,name in enumerate(names,1)] for a,names in data['labels'].items()}}
    tags = {'schema':'music_manual_tags_v1','classifier':'manual','origin':'human_entered',
            'accuracy_validated':False,'lyrics_analyzed':False,'results':[row]}
    return save(target,tags,{'origin':'human_entered','selection':data})


def rule_labels(tempo):
    import math
    if isinstance(tempo,bool) or not isinstance(tempo,(int,float)) or not math.isfinite(tempo) or tempo <= 0:
        raise ValueError('유효한 템포를 추출하지 못했습니다.')
    genre = '클래식' if tempo < 80 else '재즈' if tempo < 110 else '팝' if tempo < 140 else '록'
    emotion = '평화로운' if tempo < 80 else '슬픈' if tempo < 100 else '행복한' if tempo < 120 else '열정적인'
    return {'genre':[genre],'emotion':[emotion],'mood':[]}


def run_rule(paths, target, settings_path, *, source=None):
    python = rule_python(settings_path)
    target = Path(target); target.mkdir(parents=True,exist_ok=False)
    job = target/'rule-request.json'
    job.write_text(json.dumps([str(Path(p).resolve()) for p in paths]),encoding='utf-8')
    try:
        with (target/'worker.log').open('w',encoding='utf-8') as log:
            result = subprocess.run([str(python),'-X','utf8',str(Path(__file__).resolve()),str(job)],
                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=300*len(paths),
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        if result.returncode:raise ValueError('규칙 분류 실행 환경을 확인해주세요.')
        rows = json.loads((target/'rule-results.json').read_text(encoding='utf-8'))
        if len(rows)!=len(paths):raise ValueError('규칙 결과 수가 다릅니다.')
        for row,path in zip(rows,paths):
            row.update(audio_path=str(path),display_name=Path(path).name)
            if source:
                if row['labels'] is not None and row['audio_sha256']!=source['audio_sha256']:
                    raise ValueError('입력 링크의 오디오가 바뀌었습니다.')
                row.update(audio_path=None,display_name=source['title'],source_url=source['url'])
        tags={'schema':'music_rule_tags_v1','classifier':'rule','origin':'tempo_rules',
              'accuracy_validated':False,'lyrics_analyzed':False,'analysis_scope':'first_60_seconds',
              'unsupported_axes':['mood'],'results':rows}
        if source:tags['input_source']=source
        # The worker directory is already present; write only the export files.
        for name,value in [('tags.json',tags),('classification.json',{'classifier':'rule','results':rows})]:
            (target/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
        (target/'태깅결과.md').write_text(markdown(tags),encoding='utf-8')
        good=sum(r['labels'] is not None for r in rows)
        (target/'status.json').write_text(json.dumps({'status':'completed','successful':good,'failed':len(rows)-good}),encoding='utf-8')
        return target,0 if good==len(rows) else 1
    except Exception:
        (target/'status.json').write_text('{"status":"failed"}',encoding='utf-8')
        raise
    finally:job.unlink(missing_ok=True)


def worker(job):
    import librosa
    import numpy as np
    from music_classification.muq_runtime import digest
    rows=[]
    for path in json.loads(job.read_text(encoding='utf-8')):
        try:
            audio,sr=librosa.load(path,sr=22050,duration=60)
            if len(audio)<sr or not np.isfinite(audio).all() or np.max(np.abs(audio))<=1e-7:
                raise ValueError('정상적인 소리가 있는 1초 이상 음원이 필요합니다.')
            tempo,_=librosa.beat.beat_track(y=audio,sr=sr)
            tempo=float(np.asarray(tempo).reshape(-1)[0]); labels=rule_labels(tempo)
            rows.append({'status':'ok','labels':labels,'tempo_bpm':tempo,'audio_sha256':digest(path),
                         'candidates':{a:[{'name':v,'rank':1,'representative':True,'score':None} for v in vals] for a,vals in labels.items()}})
        except Exception as exc:
            rows.append({'status':'invalid_audio','labels':None,'candidates':None,'error':str(exc)})
    job.with_name('rule-results.json').write_text(json.dumps(rows,ensure_ascii=False,allow_nan=False),encoding='utf-8')


if __name__=='__main__':worker(Path(sys.argv[1]))
