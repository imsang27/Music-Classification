from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import familiar_tagging_app
import tag_audio
from music_classification.familiar_candidates import PROFILE_SHA256,profile_spec,ranked_axes


class CandidatePresentationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.folder=Path(self.temp.name);self.audio=self.folder/'테스트.wav';self.audio.write_bytes(b'fixture')
        values={r['id']:-.9+i*.01 for i,r in enumerate(r for rows in profile_spec()['axes'].values() for r in rows)}
        self.raw={'profile_sha256':PROFILE_SHA256,'vocabulary_version':profile_spec()['version'],
                  'batch':{'input_paths':[str(self.audio)],'results':[{'success':True,'status':'ok',
                      'axes':ranked_axes(values),'analysis':{'audio_sha256':hashlib.sha256(self.audio.read_bytes()).hexdigest()}}]}}

    def test_three_ranked_candidates_are_not_promoted_to_three_assigned_labels(self):
        before=deepcopy(self.raw);tags=tag_audio.prepare_export(self.raw,'familiar')
        self.assertEqual(self.raw,before)
        for axis in ('emotion','mood'):
            candidates=tags['results'][0]['candidates'][axis]
            self.assertEqual([c['rank'] for c in candidates],[1,2,3])
            self.assertEqual([c['representative'] for c in candidates],[True,False,False])
            self.assertEqual([c['name'] for c in candidates],[c['name'] for c in self.raw['batch']['results'][0]['axes'][axis]['candidates']])
            self.assertEqual(tags['results'][0]['labels'][axis],[candidates[0]['name']])
        text=tag_audio.format_candidate_results(tags);md=tag_audio.markdown(tags)
        self.assertIn('모두 확정 태그는 아닙니다',text)
        for axis in ('emotion','mood'):
            for c in tags['results'][0]['candidates'][axis]:
                self.assertIn(str(c['rank'])+'. '+c['name'],text)
                self.assertIn(str(c['rank'])+'. '+c['name'],md)
        block=md.split('```yaml\n',1)[1].split('```',1)[0]
        for line in block.strip().splitlines():self.assertEqual(len(json.loads(line.split(': ',1)[1])),1)

    def test_failed_audio_has_no_candidates_and_export_is_printed(self):
        self.raw['batch']['input_paths'].append(str(self.folder/'missing.wav'))
        self.raw['batch']['results'].append({'success':False,'status':'invalid_audio'})
        with patch('music_classification.familiar_candidates.classify_paths',return_value=self.raw),redirect_stdout(io.StringIO()) as console:
            target,code=tag_audio.run([self.audio,self.folder/'missing.wav'],self.folder/'output',profile='familiar')
        self.assertEqual(code,1)
        result=json.loads((target/'tags.json').read_text(encoding='utf-8'))
        self.assertIsNone(result['results'][1]['labels']);self.assertIsNone(result['results'][1]['candidates'])
        self.assertIn('감정 후보:',console.getvalue());self.assertIn('처리하지 못했습니다',console.getvalue())

    def test_wrong_profile_rejected_and_familiar_no_args_opens_window(self):
        wrong=deepcopy(self.raw);wrong['profile_sha256']='wrong'
        with self.assertRaises(ValueError):tag_audio.prepare_export(wrong,'familiar')
        with patch('familiar_tagging_app.launch_window') as launch:
            self.assertEqual(tag_audio.main(['--profile','familiar']),0)
        launch.assert_called_once_with(tag_audio.SETTINGS,None)

    def test_hidden_window_busy_guard_three_candidates_folder_and_reset(self):
        import tkinter as tk
        root=tk.Tk();root.withdraw();real_loop=root.mainloop
        gate=threading.Event();errors=[];messages=[];checks=[];ticks=[]
        result_folder=self.folder/'export';result_folder.mkdir()
        tags=tag_audio.prepare_export(self.raw,'familiar')
        (result_folder/'tags.json').write_text(json.dumps(tags,ensure_ascii=False),encoding='utf-8')
        def widgets(parent):
            for child in parent.winfo_children():
                yield child
                yield from widgets(child)
        def buttons():return {w.cget('text'):w for w in widgets(root) if w.winfo_class()=='TButton'}
        def execute(paths,out,settings,*,profile):
            self.assertEqual(paths,[str(self.audio)]);self.assertEqual(profile,'familiar')
            gate.wait(5);return result_folder,0
        def protect(callback):
            try:callback()
            except BaseException as exc:errors.append(exc);gate.set();root.destroy()
        def finish():
            controls=buttons()
            if str(controls['결과 폴더 열기']['state'])!='normal':root.after(30,lambda:protect(finish));return
            self.assertTrue(ticks)
            text=next(w for w in widgets(root) if w.winfo_class()=='Text').get('1.0','end')
            for axis in ('emotion','mood'):
                for c in tags['results'][0]['candidates'][axis]:self.assertIn(str(c['rank'])+'. '+c['name'],text)
            controls['결과 폴더 열기'].invoke();open_folder.assert_called_once_with(result_folder)
            controls['음원 선택'].invoke();self.assertEqual(str(controls['결과 폴더 열기']['state']),'normal')
            controls['음원 선택'].invoke();self.assertEqual(str(controls['결과 폴더 열기']['state']),'disabled')
            checks.append('display_folder_cancel_reset');root.tk.call(root.protocol('WM_DELETE_WINDOW'))
        def begin():
            controls=buttons();controls['음원 선택'].invoke();controls['분류 시작'].invoke()
            self.assertTrue(all(str(b['state'])=='disabled' for b in controls.values()))
            root.tk.call(root.protocol('WM_DELETE_WINDOW'));self.assertTrue(messages)
            checks.append('busy_guard');root.after(80,gate.set);root.after(30,lambda:protect(finish))
        def tick():ticks.append(1);root.after(20,tick)
        def mainloop():
            root.after(10,lambda:protect(begin));root.after(20,tick)
            root.after(7000,lambda:protect(lambda:self.fail('GUI flow timed out')));real_loop()
        with (patch('tkinter.Tk',return_value=root),patch.object(root,'mainloop',mainloop),
              patch('tag_audio.run',side_effect=execute),
              patch('tkinter.filedialog.askopenfilenames',side_effect=[(str(self.audio),),(),(str(self.folder/'other.wav'),)]),
              patch('tkinter.messagebox.showinfo',side_effect=lambda *a,**k:messages.append(a)),
              patch('tkinter.messagebox.showerror',side_effect=lambda *a,**k:errors.append(AssertionError(str(a)))),
              patch('os.startfile') as open_folder):
            familiar_tagging_app.launch_window(tag_audio.SETTINGS)
        gate.set()
        if errors:raise errors[0]
        self.assertEqual(checks,['busy_guard','display_folder_cancel_reset'])

    def test_link_window_validation_busy_guard_and_return_to_idle(self):
        import tkinter as tk
        root=tk.Tk();root.withdraw();real_loop=root.mainloop
        gate=threading.Event();errors=[];messages=[];validation=[];checks=[]
        result_folder=self.folder/'link_export';result_folder.mkdir()
        tags=tag_audio.prepare_export(self.raw,'familiar')
        tags['results'][0]['display_name']='YouTube 테스트 곡'
        (result_folder/'tags.json').write_text(json.dumps(tags,ensure_ascii=False),encoding='utf-8')
        url='https://www.youtube.com/watch?v=-VKIqrvVOpo'
        def widgets(parent):
            for child in parent.winfo_children():
                yield child
                yield from widgets(child)
        def buttons():return {w.cget('text'):w for w in widgets(root) if w.winfo_class()=='TButton'}
        def entry():return next(w for w in widgets(root) if w.winfo_class()=='TEntry')
        def execute(value,out,settings):
            self.assertEqual(value,url);gate.wait(5);return result_folder,0
        def protect(callback):
            try:callback()
            except BaseException as exc:errors.append(exc);gate.set();root.destroy()
        def finish():
            controls=buttons()
            if str(controls['결과 폴더 열기']['state'])!='normal':root.after(30,lambda:protect(finish));return
            text=next(w for w in widgets(root) if w.winfo_class()=='Text').get('1.0','end')
            self.assertIn('YouTube 테스트 곡',text)
            self.assertIn('3.',text)
            self.assertEqual(str(controls['분류 시작']['state']),'disabled')
            self.assertEqual(str(controls['링크로 분류']['state']),'normal')
            self.assertEqual(str(entry()['state']),'normal')
            checks.append('link_result');root.tk.call(root.protocol('WM_DELETE_WINDOW'))
        def begin():
            controls=buttons();controls['음원 선택'].invoke()
            entry().insert(0,'invalid link');controls['링크로 분류'].invoke()
            self.assertEqual(len(validation),1)
            entry().delete(0,'end');entry().insert(0,'https://youtu.be/-VKIqrvVOpo')
            controls['링크로 분류'].invoke()
            self.assertTrue(all(str(b['state'])=='disabled' for b in controls.values()))
            self.assertEqual(str(entry()['state']),'disabled')
            root.tk.call(root.protocol('WM_DELETE_WINDOW'));self.assertTrue(messages)
            checks.append('link_busy_guard');root.after(80,gate.set);root.after(30,lambda:protect(finish))
        def mainloop():
            root.after(10,lambda:protect(begin))
            root.after(7000,lambda:protect(lambda:self.fail('Link GUI flow timed out')));real_loop()
        with (patch('tkinter.Tk',return_value=root),patch.object(root,'mainloop',mainloop),
              patch('tag_audio.run_youtube',side_effect=execute) as classify,
              patch('tkinter.filedialog.askopenfilenames',return_value=(str(self.audio),)),
              patch('tkinter.messagebox.showinfo',side_effect=lambda *a,**k:messages.append(a)),
              patch('tkinter.messagebox.showerror',side_effect=lambda *a,**k:validation.append(a))):
            familiar_tagging_app.launch_window(tag_audio.SETTINGS)
        gate.set()
        if errors:raise errors[0]
        self.assertEqual(checks,['link_busy_guard','link_result'])
        classify.assert_called_once_with(url,None,tag_audio.SETTINGS)


if __name__=='__main__':unittest.main()
