import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from music_classification.web_audio_input import normalize_link, public_target, PublicHTTPS, temporary_input
from music_classification.youtube_input import YoutubeInputError
from music_classification.upload_management import Uploads


class LinkInputTests(unittest.TestCase):
    def test_only_public_https_audio_or_valid_youtube_urls(self):
        self.assertEqual(normalize_link('https://example.org/a.wav?x=1#ignored'),'https://example.org/a.wav?x=1')
        self.assertEqual(normalize_link('https://youtu.be/-VKIqrvVOpo?t=10'),'https://www.youtube.com/watch?v=-VKIqrvVOpo')
        for value in ['http://example.org/a.wav','file:///C:/a.wav','https://localhost/a.wav','https://127.0.0.1/a.wav',
                      'https://[::1]/a.wav','https://user:pass@example.org/a.wav','https://example.org/page',None]:
            with self.subTest(value=value),self.assertRaises(YoutubeInputError):normalize_link(value)

    def test_dns_results_and_redirect_targets_must_be_public(self):
        def answer(address):return [(2,1,6,'',(address,443))]
        for addresses in [answer('127.0.0.1'),answer('10.0.0.1'),answer('1.1.1.1')+answer('192.168.1.1')]:
            with patch('music_classification.web_audio_input.socket.getaddrinfo',return_value=addresses),self.assertRaises(YoutubeInputError):
                public_target('https://example.org/a.wav')
        with patch('music_classification.web_audio_input.socket.getaddrinfo',return_value=answer('1.1.1.1')):
            self.assertEqual(public_target('https://example.org/a.wav')[1],'1.1.1.1')
        with self.assertRaises(YoutubeInputError):public_target('http://example.org/a.wav')

    def test_tls_connection_is_pinned_to_checked_ip_and_checks_hostname(self):
        client=PublicHTTPS('example.org','1.1.1.1');context=MagicMock();client._context=context
        with patch('music_classification.web_audio_input.socket.create_connection') as connect:
            client.connect()
            connect.assert_called_once_with(('1.1.1.1',443),15)
            self.assertEqual(context.wrap_socket.call_args.kwargs['server_hostname'],'example.org')

    def test_download_size_type_and_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            response=MagicMock();response.status=200;response.getheader.side_effect=lambda name,default=None: {'Content-Length':'7','Content-Type':'audio/wav'}.get(name,default)
            connection=MagicMock();connection.getresponse.return_value=response
            with patch('music_classification.web_audio_input.ROOT',Path(directory)), \
                 patch('music_classification.web_audio_input.public_target',return_value=(__import__('urllib.parse',fromlist=['urlsplit']).urlsplit('https://example.org/a.wav'),'1.1.1.1')), \
                 patch('music_classification.web_audio_input.PublicHTTPS',return_value=connection):
                response.read.side_effect=[b'fixture',b'']
                with temporary_input('https://example.org/a.wav') as audio:
                    self.assertEqual(audio.path.read_bytes(),b'fixture');saved=audio.path
                    self.assertEqual(audio.metadata['kind'],'direct')
                self.assertFalse(saved.exists());self.assertTrue(connection.close.called)
                response.getheader.side_effect=lambda name,default=None: 'text/html' if name=='Content-Type' else default
                with self.assertRaises(YoutubeInputError):
                    with temporary_input('https://example.org/a.wav'):pass
                response.getheader.side_effect=lambda name,default=None: str(101*1024*1024) if name=='Content-Length' else default
                with self.assertRaises(YoutubeInputError):
                    with temporary_input('https://example.org/a.wav'):pass
                self.assertEqual(list((Path(directory)/'work/direct-audio-requests').iterdir()),[])

    def test_restore_cannot_overwrite_new_audio_or_escape_folder(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);folder=root/'uploads';folder.mkdir();(folder/'song.wav').write_bytes(b'original')
            store=Uploads(root,folder);record=store.recycle(store.inventory()['version'])
            (folder/'song.wav').write_bytes(b'new')
            with self.assertRaises(ValueError):store.restore(record['id'])
            self.assertEqual((folder/'song.wav').read_bytes(),b'new')
            path=store.trash/record['id']/'record.json'
            data=json.loads(path.read_text(encoding='utf-8'));data['names']=['../outside.wav'];path.write_text(json.dumps(data),encoding='utf-8')
            with self.assertRaises(ValueError):store.restore(record['id'])


if __name__=='__main__':unittest.main()
