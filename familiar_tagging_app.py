"""Simple local window for ranked familiar-word candidates."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def launch_window(settings_path,output_directory=None):
    import tkinter as tk
    from tkinter import ttk,filedialog,messagebox,scrolledtext
    from tag_audio import run,run_youtube,format_candidate_results
    import json
    root=tk.Tk();root.title('음악 분류 — 감정·분위기 후보');root.geometry('1050x720')
    frame=ttk.Frame(root,padding=16);frame.pack(fill='both',expand=True)
    ttk.Label(frame,text='음원 파일이나 YouTube 링크로 감정·분위기 후보를 각각3개씩 확인하세요.').pack(anchor='w')
    ttk.Label(frame,text='1순위는 대표값입니다. 다른 후보까지 모두 확정 태그로 부여하지 않습니다.').pack(anchor='w',pady=8)
    bar=ttk.Frame(frame);bar.pack(fill='x')
    linkbar=ttk.Frame(frame);linkbar.pack(fill='x',pady=8)
    ttk.Label(linkbar,text='YouTube 링크').pack(side='left')
    link_value=tk.StringVar()
    link_entry=ttk.Entry(linkbar,textvariable=link_value);link_entry.pack(side='left',fill='x',expand=True,padx=8)
    status=tk.StringVar(value='음원을 선택하거나, 한 곡의 YouTube 링크를 입력해주세요.')
    ttk.Label(frame,textvariable=status).pack(anchor='w',pady=8)
    output=scrolledtext.ScrolledText(frame,wrap='word');output.pack(fill='both',expand=True)
    output.insert('end','감정 예: 신남·슬픔·그리움 / 분위기 예: 발랄·시원·잔잔\n제공한 음원의 소리만 분석합니다. 원본 음원과 Obsidian 노트는 수정하지 않습니다.')
    output.configure(state='disabled')
    pool=ThreadPoolExecutor(max_workers=1)
    state={'paths':[],'future':None,'folder':None}
    def choose():
        paths=filedialog.askopenfilenames(parent=root,title='분류할 음원 선택',filetypes=[('오디오','*.wav *.mp3 *.flac *.ogg *.m4a *.aac'),('모든 파일','*.*')])
        if not paths:return
        state.update(paths=list(paths),folder=None)
        start_button.configure(state='normal');folder_button.configure(state='disabled')
        status.set(f'{len(paths)}개 선택됨')
        output.configure(state='normal');output.delete('1.0','end')
        output.insert('end','선택한 음원:\n'+'\n'.join(Path(p).name for p in paths)+'\n\n분류 시작을 누르면 후보를 보여드립니다.')
        output.configure(state='disabled')
    def poll():
        future=state['future']
        if not future.done():root.after(150,poll);return
        state['future']=None
        select_button.configure(state='normal');link_button.configure(state='normal');link_entry.configure(state='normal')
        start_button.configure(state='normal' if state['paths'] else 'disabled')
        try:
            folder,code=future.result()
            document=json.loads((Path(folder)/'tags.json').read_text(encoding='utf-8'))
            output.configure(state='normal');output.delete('1.0','end')
            output.insert('end',format_candidate_results(document));output.configure(state='disabled')
            state['folder']=Path(folder)
            folder_button.configure(state='normal')
            status.set(('완료' if code==0 else '일부 파일을 처리하지 못했습니다.')+' — 결과가 저장되었습니다: '+str(folder))
        except Exception as exc:
            status.set('분류를 완료하지 못했습니다.')
            output.configure(state='normal');output.delete('1.0','end')
            output.insert('end','결과를 생성하지 못했습니다. 입력한 링크나 음원 파일을 확인해주세요.')
            output.configure(state='disabled')
            messagebox.showerror('실행 오류',str(exc),parent=root)
    def launch_job(function,*args,**kwargs):
        state['folder']=None
        for button in (select_button,start_button,folder_button,link_button,link_entry):button.configure(state='disabled')
        status.set('분류 중입니다. 처음에는 모델 준비에 시간이 걸릴 수 있습니다.')
        output.configure(state='normal');output.delete('1.0','end')
        output.insert('end','오디오를 준비하고 분류하는 중입니다. 완료되면 후보를 보여드립니다.')
        output.configure(state='disabled')
        state['future']=pool.submit(function,*args,**kwargs)
        root.after(150,poll)
    def start():
        launch_job(run,state['paths'],output_directory,settings_path,profile='familiar')
    def start_link():
        from music_classification.youtube_input import canonical_url,YoutubeInputError
        try:url=canonical_url(link_value.get())
        except YoutubeInputError as exc:
            messagebox.showerror('YouTube 링크',str(exc),parent=root);return
        state['paths']=[]
        launch_job(run_youtube,url,output_directory,settings_path)
        status.set('YouTube 오디오 준비 및 분류 중입니다. 잠시 기다려주세요.')
    def open_folder():
        if state['folder'] is not None:
            import os
            try:os.startfile(state['folder'])
            except OSError as exc:messagebox.showerror('폴더 열기',str(exc),parent=root)
    def close():
        if state['future'] is not None:
            messagebox.showinfo('분류 중','분류가 완료된 뒤 닫아주세요.',parent=root);return
        for callback in root.tk.call('after','info'):
            root.after_cancel(callback)
        pool.shutdown(wait=False);root.destroy()
    select_button=ttk.Button(bar,text='음원 선택',command=choose);select_button.pack(side='left')
    start_button=ttk.Button(bar,text='분류 시작',command=start,state='disabled');start_button.pack(side='left',padx=8)
    folder_button=ttk.Button(bar,text='결과 폴더 열기',command=open_folder,state='disabled');folder_button.pack(side='left')
    link_button=ttk.Button(linkbar,text='링크로 분류',command=start_link);link_button.pack(side='left')
    root.protocol('WM_DELETE_WINDOW',close);root.mainloop()
