"""Bounded public HTTPS audio links alongside the existing YouTube input."""
from contextlib import contextmanager
import hashlib
import http.client
import ipaddress
from pathlib import Path
import socket
import ssl
import tempfile
import time
from urllib.parse import urljoin, urlsplit, unquote, urlunsplit

from music_classification.youtube_input import temporary_audio, canonical_url, YoutubeAudio, YoutubeInputError, MAX_BYTES, ROOT

EXTENSIONS={'.wav','.mp3','.flac','.ogg','.m4a','.aac'}


def is_public(address):
    return address.is_global and not (address.is_multicast or address.is_reserved or address.is_unspecified)


def preview_link(url):
    normalized=normalize_link(url)
    if urlsplit(normalized).hostname!='www.youtube.com':
        return {'url':normalized,'kind':'direct','title':Path(unquote(urlsplit(normalized).path)).name,
                'message':'HTTPS 음원 파일 · 접근 가능 여부는 분류할 때 확인합니다.'}
    try:
        import yt_dlp
        from music_classification.youtube_input import validate_video, _QuietLogger
        with yt_dlp.YoutubeDL({'skip_download':True,'quiet':True,'noplaylist':True,'cachedir':False,
            'socket_timeout':10,'retries':0,'extractor_retries':0,'logger':_QuietLogger()}) as client:
            info=client.extract_info(normalized,download=False)
        validate_video(info,normalized.rsplit('=',1)[1])
        return {'url':normalized,'kind':'youtube','title':' '.join(str(info.get('title','')).split())[:300],
                'channel':' '.join(str(info.get('channel','')).split())[:200],'duration_seconds':info['duration']}
    except YoutubeInputError:raise
    except Exception as exc:raise YoutubeInputError('영상 정보를 가져오지 못했습니다. 공개 영상 여부와 연결을 확인해주세요.') from exc


def normalize_link(value):
    try:return canonical_url(value)
    except ValueError:pass
    if not isinstance(value,str):raise YoutubeInputError('YouTube 영상 또는 HTTPS 음원 링크를 입력해주세요.')
    try:
        parsed=urlsplit(value.strip())
        valid=(parsed.scheme=='https' and parsed.hostname and not parsed.username and not parsed.password
               and parsed.port in (None,443) and Path(unquote(parsed.path)).suffix.lower() in EXTENSIONS)
    except ValueError:valid=False
    if not valid:raise YoutubeInputError('YouTube 영상 또는 WAV·MP3 등 HTTPS 음원 파일 링크를 입력해주세요.')
    if parsed.hostname.lower() in ('localhost','localhost.localdomain'):raise YoutubeInputError('공개 음원 링크를 사용해주세요.')
    try:
        address=ipaddress.ip_address(parsed.hostname)
    except ValueError:address=None
    if address is not None and not is_public(address):raise YoutubeInputError('공개 음원 링크를 사용해주세요.')
    return urlunsplit((parsed.scheme,parsed.netloc,parsed.path,parsed.query,''))


def public_target(url):
    parsed=urlsplit(url)
    if parsed.scheme!='https' or not parsed.hostname or parsed.username or parsed.password or parsed.port not in (None,443):
        raise YoutubeInputError('HTTPS 공개 음원만 지원합니다.')
    try:
        addresses=socket.getaddrinfo(parsed.hostname,443,type=socket.SOCK_STREAM)
        ips=[row[4][0] for row in addresses]
        if not ips or any(not is_public(ipaddress.ip_address(ip)) for ip in ips):raise ValueError('Non-public address')
    except (OSError,ValueError) as exc:raise YoutubeInputError('공개 음원 주소를 확인할 수 없습니다.') from exc
    return parsed,ips[0]


class PublicHTTPS(http.client.HTTPSConnection):
    def __init__(self,host,address):
        super().__init__(host,timeout=15,context=ssl.create_default_context());self.address=address
    def connect(self):
        # Connect to the checked IP, retaining normal hostname/TLS verification.
        self.sock=self._context.wrap_socket(socket.create_connection((self.address,443),self.timeout),server_hostname=self.host)


@contextmanager
def temporary_input(url):
    normalized=normalize_link(url)
    if urlsplit(normalized).hostname=='www.youtube.com':
        with temporary_audio(normalized) as audio:yield audio
        return
    parent=ROOT/'work/direct-audio-requests';parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='request-',dir=parent) as folder:
        parsed=urlsplit(normalized);title=Path(unquote(parsed.path)).name[:200]
        if any(ord(c)<32 for c in title):raise YoutubeInputError('음원 파일 이름이 올바르지 않습니다.')
        path=Path(folder)/('audio'+Path(title).suffix.lower());current=normalized
        try:
            for redirect in range(4):
                target,address=public_target(current)
                connection=PublicHTTPS(target.hostname,address)
                try:
                    connection.request('GET',urlunsplit(('', '',target.path or '/',target.query,'')),headers={'User-Agent':'MusicClassification/0.6','Accept':'audio/*,application/octet-stream'})
                    response=connection.getresponse()
                    if response.status in (301,302,303,307,308):
                        location=response.getheader('Location')
                        if not location or redirect==3:raise YoutubeInputError('음원 링크 이동 횟수를 초과했습니다.')
                        current=urljoin(current,location);continue
                    if response.status!=200:raise YoutubeInputError('음원 링크를 열 수 없습니다.')
                    length=response.getheader('Content-Length')
                    if length is not None and (not length.isdigit() or int(length)>MAX_BYTES):raise YoutubeInputError('100MiB 이하 음원만 지원합니다.')
                    content=response.getheader('Content-Type','').split(';')[0].lower()
                    if content and not content.startswith('audio/') and content not in ('application/octet-stream','application/ogg'):
                        raise YoutubeInputError('음원 파일에 직접 연결되는 링크를 사용해주세요.')
                    total=0;started=time.monotonic()
                    with path.open('wb') as stream:
                        while chunk:=response.read(65536):
                            total+=len(chunk)
                            if total>MAX_BYTES or time.monotonic()-started>180:raise YoutubeInputError('음원 크기 또는 준비 시간을 초과했습니다.')
                            stream.write(chunk)
                    if not total:raise YoutubeInputError('빈 음원 파일입니다.')
                    break
                finally:connection.close()
        except YoutubeInputError:raise
        except (OSError,http.client.HTTPException) as exc:raise YoutubeInputError('음원 링크를 가져오지 못했습니다. 주소와 인터넷 연결을 확인해주세요.') from exc
        with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
        yield YoutubeAudio(path,{'kind':'direct','url':normalized,'title':title,'audio_sha256':digest,
                                'retention':'temporary_audio_deleted_after_processing','title_usage':'display_only_not_classification_input'})
