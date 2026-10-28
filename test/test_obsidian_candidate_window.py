"""Exercise real hidden Tk widgets/event loop and thread, with synthetic scores."""
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import obsidian_classifier_app as app
from music_classification.obsidian_candidates import profile_spec,ranked_axes


class WindowTest(unittest.TestCase):
    def test_choose_busy_close_guard_export_cancel_and_reset(self):
        import tkinter as tk
        root=tk.Tk();root.withdraw();loop=root.mainloop
        gate=threading.Event();errors=[];checks=[];messages=[];ticks=[]
        def widgets(parent):
            for child in parent.winfo_children():
                yield child
                yield from widgets(child)
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);output=folder/'export.json';first=folder/'한 곡.wav';second=folder/'다른 곡.wav'
            values={r['id']:-.2 for rows in profile_spec()['axes'].values() for r in rows}
            document={'batch':{'input_paths':[str(first)],'results':[{'success':True,'status':'ok','axes':ranked_axes(values)}]}}
            def classify(paths,settings):
                gate.wait(5);self.assertEqual(paths,[str(first)]);return document
            def protect(callback):
                try:callback()
                except BaseException as exc:
                    errors.append(exc);gate.set();root.destroy()
            def finish():
                buttons={w.cget('text'):w for w in widgets(root) if w.winfo_class()=='TButton'}
                if str(buttons['JSON 저장']['state'])!='normal':
                    root.after(30,lambda:protect(finish));return
                self.assertTrue(ticks)
                buttons['JSON 저장'].invoke();self.assertEqual(json.loads(output.read_text(encoding='utf-8')),document)
                buttons['음원 선택'].invoke();self.assertEqual(str(buttons['JSON 저장']['state']),'normal')
                buttons['음원 선택'].invoke();self.assertEqual(str(buttons['JSON 저장']['state']),'disabled')
                checks.append('export_cancel_reset');root.tk.call(root.protocol('WM_DELETE_WINDOW'))
            def begin():
                buttons={w.cget('text'):w for w in widgets(root) if w.winfo_class()=='TButton'}
                buttons['음원 선택'].invoke();buttons['분류 시작'].invoke()
                self.assertTrue(all(str(b['state'])=='disabled' for b in buttons.values()))
                root.tk.call(root.protocol('WM_DELETE_WINDOW'));self.assertTrue(messages)
                checks.append('busy_close_guard');root.after(80,gate.set);root.after(30,lambda:protect(finish))
            def tick():ticks.append(1);root.after(20,tick)
            def mainloop():
                root.after(10,lambda:protect(begin));root.after(20,tick)
                root.after(7000,lambda:protect(lambda:self.fail('GUI flow timed out')))
                loop()
            with (patch('tkinter.Tk',return_value=root),patch.object(root,'mainloop',mainloop),
                 patch.object(app,'classify_paths',side_effect=classify),
                 patch('tkinter.filedialog.askopenfilenames',side_effect=[(str(first),),(),(str(second),)]),
                 patch('tkinter.filedialog.asksaveasfilename',return_value=str(output)),
                 patch('tkinter.messagebox.showinfo',side_effect=lambda *a,**k:messages.append(a)),
                 patch('tkinter.messagebox.showerror',side_effect=lambda *a,**k:errors.append(AssertionError(str(a))))):
                app.launch_window()
            gate.set()
        if errors:raise errors[0]
        self.assertEqual(checks,['busy_close_guard','export_cancel_reset'])


if __name__=='__main__':unittest.main()
