"""Local file selection and research candidate display; no Obsidian writes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
from music_classification.app_paths import default_settings_path
DEFAULT_SETTINGS = default_settings_path(ROOT)
AXES = (('genre', '장르 후보'), ('emotion', '감정 후보'), ('mood', '원래 분위기·테마 후보'))


def classify_paths(paths, settings_path=DEFAULT_SETTINGS):
    from music_classification import ModelSettings
    paths = [str(Path(p).expanduser().resolve()) for p in paths]
    if not paths:
        raise ValueError('음원 파일을 먼저 선택해주세요.')
    settings = ModelSettings.from_file(settings_path)
    settings = replace(settings,
        muq=replace(settings.muq, work_directory=ROOT / 'work/app-requests/muq'),
        music2emo=replace(settings.music2emo, work_directory=ROOT / 'work/app-requests/music2emo'))
    classifier = settings.create_classifier(genre_backend='muq', include_emotion=True, share_muq_model=True)
    batch = classifier.classify_many(paths, genre_top_k=3, emotion_top_k=3, mood_top_k=3)
    return {'application_schema': 'research_candidates_v1', 'final_tags': False, 'jpop_validated': False, 'batch': batch}


def all_axes_succeeded(document):
    results = document['batch']['results']
    return bool(results) and all(all(r.get('axes', {}).get(axis, {}).get('status') == 'ok' for axis, _ in AXES) for r in results)


def format_results(document):
    batch = document['batch']
    lines = ['연구용 분류 결과 — 확정 태그가 아닌 후보입니다.',
             '점수는 정답 확률이 아닙니다. 서로 다른 축의 점수를 비교하지 마세요.',
             '분위기는 원래56개 감정·분위기·테마이며, 자체8개 태그와 다릅니다.',
             'J-pop 정확도는 미검증입니다. 상위3개는 동시에 존재한다고 확인한 태그가 아닙니다.', '']
    good = sum(all(r.get('axes', {}).get(a, {}).get('status') == 'ok' for a, _ in AXES) for r in batch['results'])
    lines.append(f"전체 {len(batch['results'])}개 / 세 축 모두 결과 있음 {good}개 / 일부 또는 전체 결과 없음 {len(batch['results'])-good}개")
    for index, (path, result) in enumerate(zip(batch['input_paths'], batch['results']), 1):
        lines += ['', f'{index}. {Path(path).name}', f'   파일: {path}']
        for axis, title in AXES:
            value = result.get('axes', {}).get(axis, {})
            if value.get('status') != 'ok':
                status = value.get('status', 'unavailable')
                reason = {'invalid_audio': '파일이 없거나 정상적인 음원으로 읽을 수 없습니다.',
                          'unavailable': '이 분류에 필요한 모델이나 설정이 준비되지 않았습니다.',
                          'error': '이 항목의 분류를 완료하지 못했습니다.'}.get(status, '사용 가능한 결과가 없습니다.')
                lines.append(f'   {title}: 결과 없음 ({status}) — {reason}')
                continue
            candidates = value.get('candidates', [])
            def name(c):
                label = c.get('name', c['label'])
                return label if label == c['label'] else f"{label} ({c['label']})"
            lines.append(f"   {title}: "+' / '.join(f"{i}순위 {name(c)} · {c['score']:.4f}" for i, c in enumerate(candidates, 1)))
        if result.get('error'):
            lines.append('   오류의 원인과 실행 상세는 저장할 수 있는 JSON 결과에 보존됩니다.')
    lines += ['', '분노의 영화음악 평가 근거는 전체 감정 유무 정확도나 J-pop 검증을 뜻하지 않습니다.',
              'Obsidian 노트와 원본 음원은 수정하지 않았습니다.']
    return '\n'.join(lines)


def save_result(document, path, settings_path=DEFAULT_SETTINGS, *, overwrite=False):
    path = Path(path).expanduser().resolve()
    if path.suffix.lower() != '.json':
        raise ValueError('저장 파일 이름은 .json으로 끝나야 합니다.')
    protected = {Path(p).resolve() for p in document['batch']['input_paths']} | {Path(settings_path).resolve()}
    if path in protected:
        raise ValueError('원본 음원이나 모델 설정 파일에 결과를 덮어쓸 수 없습니다.')
    payload = json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False)+'\n'
    with path.open('w' if overwrite else 'x', encoding='utf-8') as stream:
        stream.write(payload)


def launch_window(settings_path):
    import tkinter as tk
    from tkinter import filedialog, messagebox, scrolledtext, ttk
    root = tk.Tk(); root.title('음악 분류 — 연구용 후보 보기'); root.geometry('1050x720'); root.minsize(760, 520)
    frame = ttk.Frame(root, padding=16); frame.pack(fill='both', expand=True)
    ttk.Label(frame, text='음원을 선택해 장르·감정·분위기 후보를 확인하세요.', font=('', 14, 'bold')).pack(anchor='w')
    ttk.Label(frame, text='하나 또는 여러 파일을 선택할 수 있습니다. 노트와 원본 음원은 수정하지 않습니다.').pack(anchor='w', pady=(6, 10))
    bar = ttk.Frame(frame); bar.pack(fill='x')
    status = tk.StringVar(value='먼저 음원을 선택해주세요.')
    chosen = tk.StringVar(value='선택한 파일 없음')
    state = {'paths': [], 'result': None, 'future': None}
    pool = ThreadPoolExecutor(max_workers=1)
    def show(text):
        output.configure(state='normal'); output.delete('1.0', 'end'); output.insert('1.0', text); output.configure(state='disabled')
    def choose():
        paths = filedialog.askopenfilenames(parent=root, title='분류할 음원 선택 — 여러 개 선택 가능',
            filetypes=[('오디오 파일', '*.mp3 *.wav *.flac *.ogg *.m4a *.aac *.wma *.aiff'), ('모든 파일', '*.*')])
        if not paths: return
        state.update(paths=list(paths), result=None)
        chosen.set(f"{len(paths)}개 선택: "+', '.join(Path(p).name for p in paths))
        run_button.configure(state='normal'); save_button.configure(state='disabled')
        show('선택한 파일:\n'+'\n'.join(paths)); status.set('선택이 끝났으면 분류 시작을 눌러주세요.')
    def finish():
        future = state['future']
        if not future.done(): root.after(150, finish); return
        progress.stop(); choose_button.configure(state='normal'); run_button.configure(state='normal')
        try:
            document = future.result(); state['result'] = document
            show(format_results(document)); save_button.configure(state='normal')
            status.set('분류 완료 — 모든 파일에서 세 축 결과를 받았습니다.' if all_axes_succeeded(document)
                       else '처리 완료 — 일부 파일이나 축의 결과가 없습니다. 아래 오류를 확인해주세요.')
        except Exception as exc:
            state['result'] = None; show('분류를 완료하지 못했습니다.\n'+str(exc))
            status.set('실행 오류 — 설정과 모델 환경을 확인해주세요.')
        state['future'] = None
    def run():
        if not state['paths']: return
        choose_button.configure(state='disabled'); run_button.configure(state='disabled'); save_button.configure(state='disabled')
        state['result'] = None
        status.set(f"{len(state['paths'])}개 분류 중 — 첫 실행의 모델 준비는 몇 분 걸릴 수 있습니다.")
        progress.start(); show('분류 중입니다. 결과가 준비되면 여기에 표시됩니다.')
        state['future'] = pool.submit(classify_paths, list(state['paths']), settings_path); root.after(150, finish)
    def export():
        if state['result'] is None: return
        path = filedialog.asksaveasfilename(parent=root, title='분류 결과 저장', defaultextension='.json',
                                          initialfile='music-classification-results.json', filetypes=[('JSON 결과', '*.json')])
        if not path: return
        try:
            save_result(state['result'], path, settings_path, overwrite=True)
            status.set('결과 저장 완료: '+path)
        except Exception as exc: messagebox.showerror('저장하지 못했습니다', str(exc), parent=root)
    def close():
        if state['future'] is not None:
            messagebox.showinfo('분류 중입니다', '작업이 끝난 뒤 창을 닫아주세요. 모델 준비와 분류를 진행 중입니다.', parent=root); return
        pool.shutdown(wait=False); root.destroy()
    choose_button = ttk.Button(bar, text='음원 선택…', command=choose); choose_button.pack(side='left')
    run_button = ttk.Button(bar, text='분류 시작', command=run, state='disabled'); run_button.pack(side='left', padx=8)
    save_button = ttk.Button(bar, text='결과 저장…', command=export, state='disabled'); save_button.pack(side='left')
    ttk.Label(frame, textvariable=chosen, wraplength=960).pack(anchor='w', pady=(10, 6))
    progress = ttk.Progressbar(frame, mode='indeterminate'); progress.pack(fill='x')
    ttk.Label(frame, textvariable=status, wraplength=960).pack(anchor='w', pady=6)
    output = scrolledtext.ScrolledText(frame, wrap='word', font=('맑은 고딕', 10), state='disabled'); output.pack(fill='both', expand=True)
    show('연구용 후보를 보여주는 프로그램입니다.\n\n장르10개 · 감정8개 · 원래 분위기/테마56개\nJ-pop 정확도와 최종 태그 확정은 아직 검증되지 않았습니다.\n\n음원을 선택하면 시작할 수 있습니다.')
    root.protocol('WM_DELETE_WINDOW', close); root.mainloop()


def main(argv=None):
    parser = argparse.ArgumentParser(description='로컬 음원의 연구용 장르·감정·분위기 후보를 확인합니다. 파일 인수가 없으면 선택 창을 엽니다.')
    parser.add_argument('audio', nargs='*', help='음원 파일 경로. 여러 개 지정 가능')
    parser.add_argument('--settings', type=Path, default=DEFAULT_SETTINGS, help='모델 경로 설정 JSON')
    parser.add_argument('--output', type=Path, help='결과를 새 JSON 파일로 저장. 기존 파일은 덮어쓰지 않음')
    args = parser.parse_args(argv)
    if not args.audio:
        if args.output: parser.error('--output은 음원 경로와 함께 사용해주세요.')
        launch_window(args.settings); return 0
    try:
        print('음악 분류 중입니다. 첫 모델 준비는 몇 분 걸릴 수 있습니다.', flush=True)
        document = classify_paths(args.audio, args.settings)
        print(format_results(document), flush=True)
        if args.output:
            save_result(document, args.output, args.settings); print('\n결과 저장: '+str(args.output.resolve()))
        return 0 if all_axes_succeeded(document) else 1
    except Exception as exc:
        print('실행하지 못했습니다: '+str(exc), file=sys.stderr); return 2


if __name__ == '__main__':
    raise SystemExit(main())
