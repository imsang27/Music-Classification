from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf
import tag_audio
from music_classification.familiar_candidates import PROFILE_SHA256,profile_spec,ranked_axes
from music_classification.youtube_input import canonical_url,validate_video,temporary_audio,YoutubeInputError,MAX_BYTES

VIDEO='-VKIqrvVOpo'
URL='https://www.youtube.com/watch?v='+VIDEO


class Downloader:
    mode='ok'
    def __init__(self,options):self.options=options
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def extract_info(self,url,download):
        assert url==URL and download
        info={'id':VIDEO,'title':'테스트 곡','duration':2,'channel':'테스트 채널','availability':'public'}
        if self.mode=='network':raise RuntimeError('Network unavailable')
        self.options['match_filter'](info,incomplete=False)
        for hook in self.options['progress_hooks']:
            hook({'downloaded_bytes':MAX_BYTES+1 if self.mode=='oversize' else 100})
        self.source=Path(self.options['outtmpl'].replace('%(ext)s','webm'))
        self.source.write_bytes(b'synthetic compressed fixture')
        return info
    def prepare_filename(self,info):return str(self.source)


def convert(argv,**kwargs):
    sf.write(argv[-1],np.full(48000,.1,dtype=np.float32),24000,subtype='PCM_16')
    return SimpleNamespace(returncode=0)


class YoutubeInputTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.folder=Path(self.temp.name);self.jobs=self.folder/'requests'
        self.decoder=self.folder/'ffmpeg.exe';self.decoder.write_bytes(b'fixture')
        Downloader.mode='ok'

    def test_common_urls_normalize_to_one_video_not_playlist(self):
        for value in [URL+'&list=ignored&t=90','https://youtu.be/'+VIDEO+'?si=abc',
                      'https://music.youtube.com/watch?v='+VIDEO,
                      'https://m.youtube.com/shorts/'+VIDEO,'https://youtube.com/embed/'+VIDEO]:
            self.assertEqual(canonical_url(value),URL)
        for value in ['https://www.youtube.com/playlist?list=abc','https://www.youtube.com.evil.test/watch?v='+VIDEO,
                      'file:///C:/audio.wav','https://localhost/watch?v='+VIDEO,'https://x@youtube.com/watch?v='+VIDEO,
                      'https://youtu.be/'+VIDEO+'/extra',URL+'&v=duplicate','https://[','not a link']:
            with self.subTest(value=value),self.assertRaises(YoutubeInputError):canonical_url(value)

    def test_live_private_unknown_or_long_video_rejected(self):
        valid={'id':VIDEO,'duration':200,'availability':'public'}
        validate_video(valid,VIDEO)
        for change in [{'is_live':True},{'live_status':'is_upcoming'},{'availability':'private'},
                       {'duration':1201},{'duration':None},{'duration':True},{'duration':float('nan')},{'_type':'playlist'},{'id':'different'}]:
            with self.subTest(change=change),self.assertRaises(YoutubeInputError):validate_video({**valid,**change},VIDEO)

    def test_temporary_audio_is_deleted_on_success_and_body_failure(self):
        for failure in (False,True):
            with self.subTest(failure=failure),patch('yt_dlp.YoutubeDL',Downloader),patch('music_classification.youtube_input.subprocess.run',side_effect=convert):
                try:
                    with temporary_audio(URL,work_directory=self.jobs,ffmpeg_path=self.decoder) as item:
                        path=item.path;self.assertTrue(path.is_file())
                        self.assertEqual(item.metadata['title'],'테스트 곡')
                        if failure:raise RuntimeError('simulated inference error')
                except RuntimeError:
                    self.assertTrue(failure)
                self.assertFalse(path.exists());self.assertFalse(list(self.jobs.iterdir()))

    def test_download_and_conversion_failures_also_clean_up(self):
        for mode in ('network','oversize','conversion'):
            Downloader.mode=mode
            decoder=convert if mode!='conversion' else lambda *a,**k:SimpleNamespace(returncode=1)
            with self.subTest(mode=mode),patch('yt_dlp.YoutubeDL',Downloader),patch('music_classification.youtube_input.subprocess.run',side_effect=decoder):
                with self.assertRaises(YoutubeInputError):
                    with temporary_audio(URL,work_directory=self.jobs,ffmpeg_path=self.decoder):self.fail('Unexpected audio')
                self.assertFalse(list(self.jobs.iterdir()))

    def test_link_pipeline_has_title_source_and_three_candidates_after_cleanup(self):
        values={r['id']:-.9+i*.01 for i,r in enumerate(r for rows in profile_spec()['axes'].values() for r in rows)}
        def classify(paths,settings):
            self.assertEqual(len(paths),1);path=Path(paths[0]);self.assertTrue(path.is_file())
            return {'profile_sha256':PROFILE_SHA256,'vocabulary_version':profile_spec()['version'],
                    'batch':{'input_paths':[str(path)],'results':[{'success':True,'status':'ok','axes':ranked_axes(values),
                        'analysis':{'audio_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}}]}}
        def fetch(url):return temporary_audio(url,work_directory=self.jobs,ffmpeg_path=self.decoder)
        with (patch('yt_dlp.YoutubeDL',Downloader),patch('music_classification.youtube_input.subprocess.run',side_effect=convert),
              patch('music_classification.youtube_input.temporary_audio',side_effect=fetch),
              patch('music_classification.familiar_candidates.classify_paths',side_effect=classify),redirect_stdout(io.StringIO())):
            folder,code=tag_audio.run_youtube(URL,self.folder/'output')
        self.assertEqual(code,0);self.assertFalse(list(self.jobs.iterdir()))
        tags=json.loads((folder/'tags.json').read_text(encoding='utf-8'))
        row=tags['results'][0];self.assertIsNone(row['audio_path'])
        self.assertEqual(row['display_name'],'테스트 곡');self.assertEqual(row['source_url'],URL)
        self.assertEqual(len(row['candidates']['emotion']),3);self.assertEqual(len(row['candidates']['mood']),3)
        self.assertTrue(json.loads((folder/'status.json').read_text())['temporary_audio_removed'])
        raw=json.loads((folder/'classification.json').read_text(encoding='utf-8'))
        self.assertEqual(raw['batch']['input_paths'],[URL])
        self.assertIn(URL,(folder/'태깅결과.md').read_text(encoding='utf-8'))

    def test_cli_routes_link_and_rejects_mixed_file_input(self):
        with patch('tag_audio.run_youtube',return_value=(self.folder,0)) as runner:
            self.assertEqual(tag_audio.main(['--youtube',URL]),0)
        runner.assert_called_once_with(URL,None,tag_audio.SETTINGS)
        with redirect_stdout(io.StringIO()),patch('sys.stderr',new_callable=io.StringIO),self.assertRaises(SystemExit):
            tag_audio.main(['--youtube',URL,'file.wav'])


if __name__=='__main__':unittest.main()
