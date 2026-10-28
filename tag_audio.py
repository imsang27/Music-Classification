"""Assign one canonical audio label per axis using the existing fixed model."""
import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import json
from pathlib import Path

from obsidian_classifier_app import ROOT, SETTINGS, classify_paths


def assigned_tags(document):
    """Keep failed inputs as failures; never promote alternatives to extra tags."""
    paths = document['batch']['input_paths']
    results = document['batch']['results']
    if len(paths) != len(results):
        raise ValueError('음원과 분류 결과의 개수가 다릅니다.')
    rows = []
    for path, result in zip(paths, results):
        row = {'audio_path':str(path), 'status':result['status'], 'labels':None}
        if result['success']:
            labels, details = {}, {}
            for axis in ('genre','emotion','mood'):
                values = result['axes'][axis]
                if values['status'] != 'ok' or not values['candidates']:
                    raise ValueError('성공 결과에 분류 항목이 없습니다.')
                primary = values['primary']
                if primary != values['candidates'][0]:
                    raise ValueError('첫 후보와 선택 항목이 일치하지 않습니다.')
                labels[axis] = [primary['name']]
                details[axis] = {'selected_id':primary['label'], 'raw_score':primary['score'],
                                 'alternatives':values['candidates'][1:]}
            source = Path(path)
            row.update(labels=labels, details=details,
                       audio_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
        else:
            row['error'] = result
        rows.append(row)
    return {'schema':'music_auto_tags_v1', 'assignment':'top1_per_axis',
            'origin':'model_audio_only', 'human_verified':False,
            'accuracy_validated':False, 'lyrics_analyzed':False,
            'profile_sha256':document['profile_sha256'], 'results':rows}


def prepare_export(raw, profile='single'):
    if profile not in ('single','hierarchical','familiar'):
        raise ValueError('알 수 없는 태깅 구성입니다.')
    tags=assigned_tags(raw)
    if profile=='familiar':
        from music_classification.familiar_candidates import profile_spec,PROFILE_SHA256
        if raw['profile_sha256']!=PROFILE_SHA256:
            raise ValueError('현재 익숙한 표현 구성과 다른 결과입니다.')
        tags.update(schema='music_familiar_tags_v1',
                    vocabulary=raw.get('vocabulary_version') or profile_spec()['version'],
                    display_policy_sha256=raw.get('display_policy_sha256'),
                    candidate_semantics='ranked_alternatives_not_simultaneous_confirmed_tags')
        for row,result in zip(tags['results'],raw['batch']['results']):
            row['candidates']=None
            if row['labels'] is None:continue
            row['candidates']={axis:[{**deepcopy(c),'rank':rank,'representative':rank==1}
                for rank,c in enumerate(result['axes'][axis]['candidates'][:3],1)]
                for axis in ('genre','emotion','mood')}
    if profile=='hierarchical':
        from music_classification.hierarchical_tagging import apply_hierarchy
        tags=apply_hierarchy(tags,raw)
    return tags


def format_candidate_results(document):
    lines=['음악 분류 후보',
        '감정·분위기는 각각 상위3후보입니다. 1순위가 대표값이며 모두 확정 태그는 아닙니다.','']
    for row in document['results']:
        lines+=['',row.get('display_name') or Path(row['audio_path']).name]
        if row['labels'] is None:
            lines.append('  처리하지 못했습니다. 다른 음원 파일로 확인해주세요.');continue
        lines.append('  장르 대표: '+row['labels']['genre'][0])
        for axis,title in (('emotion','감정'),('mood','분위기')):
            values=row['candidates'][axis]
            lines.append('  '+title+' 후보: '+' / '.join(
                f"{c['rank']}. {c['name']}"+(' (대표)' if c['representative'] else '') for c in values))
    return '\n'.join(lines)


def markdown(document):
    familiar=document.get('schema')=='music_familiar_tags_v1'
    assignment_text = ('장르는 대표 태그 하나, 감정·분위기는 학습 근거가 있을 때 최대3개까지 부여합니다. 근거가 없으면 가까운 대표 표현 하나를 유지합니다.'
                       if document.get('schema')=='music_hierarchical_tags_v1' else
                       '각 음원에 장르·감정·분위기를 하나씩 부여했습니다. 고정된 후보 중 가장 가까운 표현이며 사람이 확인한 정답은 아닙니다.')
    if familiar:
        assignment_text='감정·분위기의 상위3후보를 순서대로 보여줍니다. 1순위는 대표값이며, 세 단어 모두가 동시에 맞는 확정 태그는 아닙니다.'
    lines = ['# 음원 자동 태깅 결과', '',
             assignment_text,
             '가사·아티스트 정보는 분석하지 않았습니다. 원본 음원과 Obsidian 노트는 수정하지 않았습니다.', '',
             ('| 음원 | 장르 대표 | 감정 후보1~3 | 분위기 후보1~3 |' if familiar else '| 음원 | 장르 | 감정 | 분위기 |'), '|---|---|---|---|']
    if document.get('input_source',{}).get('kind')=='youtube':
        lines[3:3]=['원본 링크: '+document['input_source']['url'],
                    '분석용 오디오는 처리 후 삭제하고 결과만 저장합니다. 영상 제목은 화면 표시용이며 분류 근거로 사용하지 않습니다.','']
    for row in document['results']:
        name = (row.get('display_name') or Path(row['audio_path']).name).replace('|','\\|')
        if row['labels'] is None:
            lines.append(f'| {name} | 처리 실패: {row["status"]} | — | — |')
        else:
            labels = row['labels']
            if familiar:
                cells=[labels['genre'][0]]
                for axis in ('emotion','mood'):
                    cells.append(' · '.join(f"{c['rank']}. {c['name']}"+(' (대표)' if c['representative'] else '') for c in row['candidates'][axis]))
                lines.append('| '+name+' | '+' | '.join(cells)+' |')
            else:
                lines.append('| '+name+' | '+' | '.join(' · '.join(labels[a]) for a in ('genre','emotion','mood'))+' |')
    if document.get('schema')=='music_hierarchical_tags_v1':
        lines += ['', '## 큰 범주와 세부 표현', '',
                  '큰 범주는 기존 표현을 정리한 구조입니다. 학습된 상위 범주 판정이나 정답 확률이 아닙니다.',
                  '학습 근거가 있는 복수 태그 결정이 없으면 기존 방식의 대표 태그 하나만 추정값으로 유지합니다.', '']
        for row in document['results']:
            if row['labels'] is None:
                continue
            lines.append('**'+Path(row['audio_path']).name+'**')
            for axis,title in (('emotion','감정'),('mood','분위기')):
                value = row['hierarchy'][axis]
                origin = '학습 결정' if value['assignment']=='supported_supervised_decisions' else '유사도 기반 추정'
                lines.append('- '+title+': '+' / '.join(value['parents'])+' → '+' · '.join(value['labels'])+' ('+origin+')')
            lines.append('')
    lines += ['', ('## 대표값만 담은 노트 속성 — 검토 후 사용' if familiar else '## 노트 속성으로 사용할 값'), '',
              '아래 값은 Music 폴더의 표기 규칙을 유지합니다. 띄어쓰기를 보존하기 위해 속성 값으로 제공합니다. 기존 노트에는 자동으로 기록하지 않습니다.', '']
    if familiar:
        lines+=['후보3개를 모두 노트 태그로 복사한 값이 아닙니다. JSON의 labels는 대표값, candidates는 순위가 있는 검토 후보입니다.','']
    for row in document['results']:
        if row['labels'] is None:
            continue
        lines += ['### '+(row.get('display_name') or Path(row['audio_path']).name), '', '```yaml']
        for axis in ('genre','emotion','mood'):
            lines.append(axis+': '+json.dumps(row['labels'][axis],ensure_ascii=False))
        lines += ['```', '']
    return '\n'.join(lines)


def run(paths, output_directory=None, settings=SETTINGS, *, profile='single',source=None):
    if profile not in ('single','hierarchical','familiar'):
        raise ValueError('알 수 없는 태깅 구성입니다.')
    paths = [Path(p).expanduser().resolve() for p in paths]
    if not paths:
        raise ValueError('태깅할 음원을 선택해주세요.')
    if source is not None and (profile!='familiar' or len(paths)!=1 or source.get('kind')!='youtube'):
        raise ValueError('YouTube 입력은 익숙한 표현 구성에서 한 곡씩 지원합니다.')
    target = Path(output_directory) if output_directory else ROOT/'outputs/tagged_audio'/datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    # Always create a new directory so exports cannot overwrite audio or settings.
    target.mkdir(parents=True,exist_ok=False)
    (target/'status.json').write_text('{"status":"running"}\n',encoding='utf-8')
    try:
        print(f'{len(paths)}개 음원 태깅 중입니다.',flush=True)
        if profile=='familiar':
            from music_classification.familiar_candidates import classify_paths as classify_familiar
            raw = classify_familiar(paths,settings)
        else:
            raw = classify_paths(paths,settings)
        tags = prepare_export(raw,profile)
        if source is not None:
            source=deepcopy(source)
            tags['input_source']=source
            raw=deepcopy(raw);raw['input_source']=source
            raw['batch']['input_paths']=[source['url']]
            for row in tags['results']:
                if row['labels'] is not None and row['audio_sha256']!=source['audio_sha256']:
                    raise ValueError('YouTube 입력과 분류 결과의 오디오가 다릅니다.')
                row.update(audio_path=None,source_url=source['url'],display_name=source['title'])
        for name, data in (('classification.json',raw),('tags.json',tags)):
            (target/name).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        (target/'태깅결과.md').write_text(markdown(tags),encoding='utf-8')
        successful = sum(r['labels'] is not None for r in tags['results'])
        (target/'status.json').write_text(json.dumps({'status':'completed','successful':successful,'failed':len(paths)-successful},ensure_ascii=False)+'\n',encoding='utf-8')
        if profile=='familiar':print(format_candidate_results(tags),flush=True)
        print(f'완료: {successful}/{len(paths)}개\n결과 폴더: {target.resolve()}',flush=True)
        return target, 0 if successful == len(paths) else 1
    except Exception as exc:
        (target/'status.json').write_text(json.dumps({'status':'failed','error':str(exc)},ensure_ascii=False)+'\n',encoding='utf-8')
        raise


def run_youtube(url,output_directory=None,settings=SETTINGS):
    from music_classification.youtube_input import temporary_audio
    with temporary_audio(url) as audio:
        target,code=run([audio.path],output_directory,settings,profile='familiar',source=audio.metadata)
    status=json.loads((target/'status.json').read_text(encoding='utf-8'))
    status.update(input_kind='youtube',temporary_audio_removed=True)
    (target/'status.json').write_text(json.dumps(status,ensure_ascii=False)+'\n',encoding='utf-8')
    return target,code


def main(argv=None):
    parser = argparse.ArgumentParser(description='음원에 장르·감정·분위기 태그를 부여합니다.')
    parser.add_argument('audio',nargs='*')
    parser.add_argument('--output-directory',type=Path)
    parser.add_argument('--settings',type=Path,default=SETTINGS)
    parser.add_argument('--profile',choices=('single','hierarchical','familiar'),default='single',help='single: 기존 대표 태그 / hierarchical: 계층형 / familiar: 익숙한 감정·분위기 표현')
    parser.add_argument('--youtube',help='YouTube 영상 한 곡의 링크; 익숙한 표현 구성으로 분류')
    args = parser.parse_args(argv)
    if args.youtube:
        if args.audio:parser.error('YouTube 링크와 음원 파일을 동시에 지정할 수 없습니다.')
        try:
            _,code=run_youtube(args.youtube,args.output_directory,args.settings)
            return code
        except Exception as exc:
            print('YouTube 분류 오류: '+str(exc),flush=True)
            return 2
    paths = args.audio
    if not paths and args.profile=='familiar':
        from familiar_tagging_app import launch_window
        launch_window(args.settings,args.output_directory)
        return 0
    if not paths:
        import tkinter as tk
        from tkinter import filedialog
        window = tk.Tk(); window.withdraw()
        try:
            paths = list(filedialog.askopenfilenames(parent=window,title='태깅할 음원 선택',filetypes=[('오디오','*.wav *.mp3 *.flac *.ogg *.m4a *.aac'),('모든 파일','*.*')]))
        finally:
            window.destroy()
        if not paths:
            print('선택을 취소했습니다.'); return 0
    try:
        _, code = run(paths,args.output_directory,args.settings,profile=args.profile)
        return code
    except Exception as exc:
        print('태깅 오류: '+str(exc),flush=True)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
