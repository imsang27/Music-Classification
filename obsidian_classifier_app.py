"""Local69-label candidate viewer; optional file selection, no note writing."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
from music_classification.app_paths import default_settings_path
SETTINGS=default_settings_path(ROOT)
AXES={'genre':'장르·스타일','emotion':'감정','mood':'분위기'}


def classify_paths(paths,settings_path=SETTINGS):
    from music_classification.settings import ModelSettings
    from music_classification.obsidian_candidates import ObsidianCandidateClassifier,PROFILE_SHA256
    settings=ModelSettings.from_file(settings_path)
    if settings.muq is None:raise ValueError('MuQ 모델 설정이 없습니다.')
    config=settings.muq.kwargs();config['work_directory']=ROOT/'work/obsidian-profile-requests'
    classifier=ObsidianCandidateClassifier(**config)
    return {'application_schema':'obsidian_candidates_v1','final_tags':False,'accuracy_validated':False,
      'profile_sha256':PROFILE_SHA256,'batch':classifier.classify_many(paths,top_k=3)}


def format_results(document):
    batch=document['batch']
    lines=['Music 폴더 표현으로 분류한 연구용 후보',
      '장르25개 · 감정22개 · 분위기22개 중 축별 상위3후보입니다.',
      '점수는 정답 확률이 아니며, 여러 후보의 동시 존재를 확정하지 않습니다.',
      '제공한 음원 구간만 분석합니다. 가까운 표현의 구별 정확도는 미검증입니다.','']
    for path,result in zip(batch['input_paths'],batch['results']):
        lines+=['',Path(path).name]
        if not result['success']:
            lines.append('  결과 없음: '+{'invalid_audio':'정상적인 음원으로 읽을 수 없습니다.','unavailable':'모델이나 실행 환경을 확인해주세요.'}.get(result['status'],'분류 중 오류가 발생했습니다. 상세 내용은 JSON에 보존됩니다.'));continue
        for axis,name in AXES.items():
            values=result['axes'][axis]['candidates']
            lines.append('  '+name+': '+' / '.join(f"{i}순위 {c['name']} ({c['score']:.4f})" for i,c in enumerate(values,1)))
    return '\n'.join(lines)


def save_result(document,path,settings_path=SETTINGS,*,overwrite=False):
    from music_classification.obsidian_candidates import profile_spec
    target=Path(path).expanduser().resolve()
    if target.suffix.lower()!='.json':raise ValueError('JSON 파일로 저장해주세요.')
    protected={Path(p).resolve() for p in document['batch']['input_paths']}|{
      Path(settings_path).resolve(),ROOT/'music_classification/resources/obsidian_candidate_prompts_v1.json'}
    if target in protected:raise ValueError('음원·설정·분류 설명 파일에는 덮어쓸 수 없습니다.')
    profile_spec()
    with target.open('w' if overwrite else 'x',encoding='utf-8') as stream:
        stream.write(json.dumps(document,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def launch_window(settings_path=SETTINGS):
    import tkinter as tk
    from tkinter import ttk,filedialog,messagebox,scrolledtext
    root=tk.Tk();root.title('음악 분류 — Music 폴더 표현');root.geometry('1050x720')
    frame=ttk.Frame(root,padding=16);frame.pack(fill='both',expand=True)
    ttk.Label(frame,text='음원을 선택해 Music 폴더 표현의 후보를 확인하세요.').pack(anchor='w')
    ttk.Label(frame,text='장르25 · 감정22 · 분위기22 / 아직 정확도를 검증하지 않은 별도 구성입니다.').pack(anchor='w',pady=8)
    bar=ttk.Frame(frame);bar.pack(fill='x');status=tk.StringVar(value='음원 하나 또는 여러 개를 선택해주세요.')
    ttk.Label(frame,textvariable=status).pack(anchor='w',pady=8)
    output=scrolledtext.ScrolledText(frame,wrap='word');output.pack(fill='both',expand=True)
    output.insert('end','기존 10/8/56 출력과 다른 직접 오디오·설명 비교입니다.\n가사 의미 분석이나 Obsidian 노트 쓰기는 수행하지 않습니다.')
    output.configure(state='disabled')
    pool=ThreadPoolExecutor(max_workers=1);state={'paths':[],'document':None,'future':None}
    def show(text):
        output.configure(state='normal');output.delete('1.0','end');output.insert('end',text);output.configure(state='disabled')
    def choose():
        paths=filedialog.askopenfilenames(parent=root,title='분류할 음원 선택',filetypes=[('오디오','*.wav *.mp3 *.flac *.ogg *.m4a *.aac'),('모든 파일','*.*')])
        if not paths:return
        state.update(paths=list(paths),document=None);status.set(f'{len(paths)}개 선택됨');run.configure(state='normal');save.configure(state='disabled')
    def done():
        future=state['future']
        if not future.done():root.after(150,done);return
        state['future']=None;select.configure(state='normal');run.configure(state='normal')
        try:
            state['document']=future.result();show(format_results(state['document']));save.configure(state='normal');status.set('완료 — 필요하면 JSON을 저장할 수 있습니다.')
        except Exception as exc:status.set('분류를 완료하지 못했습니다.');messagebox.showerror('실행 오류',str(exc),parent=root)
    def start():
        state['document']=None
        for button in (select,run,save):button.configure(state='disabled')
        status.set('분류 중입니다. 처음에는 모델 준비에 시간이 걸릴 수 있습니다.')
        state['future']=pool.submit(classify_paths,state['paths'],settings_path);root.after(150,done)
    def export():
        path=filedialog.asksaveasfilename(parent=root,defaultextension='.json',filetypes=[('JSON','*.json')],confirmoverwrite=True)
        if not path:return
        try:save_result(state['document'],path,settings_path,overwrite=True)
        except Exception as exc:messagebox.showerror('저장 오류',str(exc),parent=root)
    def close():
        if state['future'] is not None:
            messagebox.showinfo('분류 중','분류가 완료된 뒤 닫아주세요.',parent=root);return
        pool.shutdown(wait=False);root.destroy()
    select=ttk.Button(bar,text='음원 선택',command=choose);select.pack(side='left')
    run=ttk.Button(bar,text='분류 시작',command=start,state='disabled');run.pack(side='left',padx=8)
    save=ttk.Button(bar,text='JSON 저장',command=export,state='disabled');save.pack(side='left')
    root.protocol('WM_DELETE_WINDOW',close);root.mainloop()


def main(argv=None):
    p=argparse.ArgumentParser(description='Music 폴더 표현으로 음원의 연구용 후보를 분류합니다.')
    p.add_argument('audio',nargs='*');p.add_argument('--settings',type=Path,default=SETTINGS);p.add_argument('--output',type=Path)
    args=p.parse_args(argv)
    if not args.audio:
        if args.output:p.error('--output은 음원 경로와 함께 사용해주세요.')
        launch_window(args.settings);return 0
    try:
        document=classify_paths(args.audio,args.settings);print(format_results(document),flush=True)
        if args.output:save_result(document,args.output,args.settings)
        return 0 if all(r['success'] for r in document['batch']['results']) else 1
    except Exception as exc:print('실행 오류: '+str(exc));return 2


if __name__=='__main__':raise SystemExit(main())
