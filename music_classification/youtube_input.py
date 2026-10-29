"""One public YouTube video -> bounded, request-local audio, then cleanup."""
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
import importlib.util
import math
import os
from pathlib import Path
import re
import subprocess
import shutil
import tempfile
import time
from urllib.parse import parse_qs,urlparse

ROOT=Path(__file__).resolve().parents[1]
MAX_SECONDS=20*60
MAX_BYTES=100*1024*1024


class YoutubeInputError(ValueError):
    pass


def ffmpeg_executable(ffmpeg_path=None):
    bundled=ROOT/'ffmpeg-master-latest-win64-gpl/bin/ffmpeg.exe'
    return Path(ffmpeg_path or os.environ.get('MUSIC_CLASSIFICATION_FFMPEG') or
                (str(bundled) if bundled.is_file() else shutil.which('ffmpeg') or bundled))


def environment_status():
    """Check the actual application interpreter, separately from model readiness."""
    missing=[name for name in ('yt_dlp','soundfile') if importlib.util.find_spec(name) is None]
    if missing:
        return {'ready':False,'message':'YouTube용 연결 도구가 없습니다. 웹 실행 환경에 .[web] 설치가 필요합니다.'}
    if not ffmpeg_executable().is_file():
        return {'ready':False,'message':'YouTube 오디오 변환 도구(FFmpeg)가 없습니다. 설치 안내에 따라 FFmpeg 경로를 설정해주세요.'}
    return {'ready':True,'message':'YouTube 링크로 분류할 수 있습니다.'}


def canonical_url(value):
    if not isinstance(value,str):raise YoutubeInputError('YouTube 영상 링크를 입력해주세요.')
    try:parsed=urlparse(value.strip())
    except ValueError as exc:raise YoutubeInputError('올바른 YouTube 영상 링크를 입력해주세요.') from exc
    if parsed.scheme not in ('https','http') or parsed.username or parsed.password:
        raise YoutubeInputError('https://로 시작하는 YouTube 영상 링크를 입력해주세요.')
    try:
        if parsed.port not in (None,80,443):raise YoutubeInputError('일반 YouTube 영상 링크를 입력해주세요.')
    except ValueError as exc:raise YoutubeInputError('올바른 YouTube 영상 링크를 입력해주세요.') from exc
    host=parsed.hostname
    if host in ('youtu.be','www.youtu.be'):
        video=parsed.path.strip('/')
    elif host in ('youtube.com','www.youtube.com','m.youtube.com','music.youtube.com'):
        parts=parsed.path.strip('/').split('/')
        if parts==['watch']:
            values=parse_qs(parsed.query).get('v',[])
            video=values[0] if len(values)==1 else ''
        elif len(parts)==2 and parts[0] in ('shorts','embed','live'):
            video=parts[1]
        else:video=''
    else:
        raise YoutubeInputError('YouTube 영상 링크만 지원합니다.')
    if not re.fullmatch(r'[A-Za-z0-9_-]{11}',video or ''):
        raise YoutubeInputError('재생목록 전체가 아닌, 노래 한 곡의 영상 링크를 입력해주세요.')
    return 'https://www.youtube.com/watch?v='+video


def validate_video(info,video_id):
    duration=info.get('duration')
    if info.get('id')!=video_id or info.get('_type','video')!='video':
        raise YoutubeInputError('선택한 영상 한 곡을 확인할 수 없습니다.')
    if info.get('is_live') or info.get('live_status') in ('is_live','is_upcoming'):
        raise YoutubeInputError('진행 중인 방송이나 공개 예정 영상은 지원하지 않습니다.')
    if info.get('availability') in ('private','premium_only','subscriber_only','needs_auth'):
        raise YoutubeInputError('로그인 없이 열 수 있는 공개 영상 링크를 사용해주세요.')
    if isinstance(duration,bool) or not isinstance(duration,(int,float)) or not math.isfinite(duration) or not 1<=duration<=MAX_SECONDS:
        raise YoutubeInputError('길이를 확인할 수 있는 20분 이하의 영상을 사용해주세요.')


@dataclass(frozen=True)
class YoutubeAudio:
    path: Path
    metadata: dict


class _QuietLogger:
    def debug(self,*args,**kwargs):pass
    def warning(self,*args,**kwargs):pass
    def error(self,*args,**kwargs):pass


@contextmanager
def temporary_audio(url,*,work_directory=None,ffmpeg_path=None):
    canonical=canonical_url(url);video_id=canonical.rsplit('=',1)[1]
    executable=ffmpeg_executable(ffmpeg_path)
    if not executable.is_file():raise YoutubeInputError('오디오 변환 도구가 없습니다. 프로그램 실행 환경을 확인해주세요.')
    try:
        import yt_dlp
        import soundfile as sf
    except ImportError as exc:raise YoutubeInputError('YouTube 연결 도구가 없습니다. 프로그램 실행 환경을 확인해주세요.') from exc
    parent=Path(work_directory or ROOT/'work/youtube-tagging-requests').resolve()
    parent.mkdir(parents=True,exist_ok=True)
    temporary=tempfile.TemporaryDirectory(prefix='request-',dir=parent)
    request=Path(temporary.name)
    # Check the absolute cleanup target against the explicitly chosen parent.
    if request.resolve().parent!=parent or not request.name.startswith('request-'):
        raise YoutubeInputError('임시 작업 폴더를 확인할 수 없습니다.')
    try:
        started=time.monotonic()
        def match(info,*,incomplete=False):
            if not incomplete:validate_video(info,video_id)
            return None
        def progress(state):
            if time.monotonic()-started>180:
                raise YoutubeInputError('오디오 준비 시간이 초과되었습니다. 나중에 다시 시도해주세요.')
            if state.get('downloaded_bytes',0)>MAX_BYTES:
                raise YoutubeInputError('오디오가 너무 큽니다. 더 짧은 영상을 사용해주세요.')
        options={'format':'bestaudio','outtmpl':str(request/'source.%(ext)s'),
            'noplaylist':True,'retries':0,'fragment_retries':0,'extractor_retries':0,
            'socket_timeout':15,'cachedir':False,'max_filesize':MAX_BYTES,
            'match_filter':match,'progress_hooks':[progress],'quiet':True,'noprogress':True,
            'logger':_QuietLogger(),'ffmpeg_location':str(executable.parent)}
        try:
            with yt_dlp.YoutubeDL(options) as downloader:
                info=downloader.extract_info(canonical,download=True)
                if not isinstance(info,dict):raise YoutubeInputError('영상의 오디오를 가져오지 못했습니다.')
                validate_video(info,video_id)
                source=Path(downloader.prepare_filename(info)).resolve()
        except YoutubeInputError:raise
        except Exception as exc:
            raise YoutubeInputError('YouTube 오디오를 가져오지 못했습니다. 인터넷 연결과 공개 영상 여부를 확인해주세요.') from exc
        if source.parent!=request.resolve() or not source.is_file() or not 0<source.stat().st_size<=MAX_BYTES:
            raise YoutubeInputError('분석할 수 있는 크기의 오디오를 받지 못했습니다.')
        target=request/'audio.wav'
        try:
            converted=subprocess.run([str(executable),'-v','error','-nostdin','-n','-i',str(source),
                '-vn','-t',str(MAX_SECONDS+1),'-ac','1','-ar','24000','-c:a','pcm_s16le',str(target)],
                capture_output=True,text=True,timeout=120,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        except (OSError,subprocess.TimeoutExpired) as exc:
            raise YoutubeInputError('영상 오디오를 변환하지 못했습니다. 다른 공개 영상으로 확인해주세요.') from exc
        if converted.returncode or not target.is_file() or not 0<target.stat().st_size<=MAX_BYTES:
            raise YoutubeInputError('영상 오디오를 변환하지 못했습니다. 다른 공개 영상으로 확인해주세요.')
        try:
            decoded=sf.info(target)
        except Exception as exc:raise YoutubeInputError('정상적인 음원으로 읽을 수 없습니다.') from exc
        if decoded.samplerate!=24000 or decoded.channels!=1 or not 1<=decoded.duration<=MAX_SECONDS:
            raise YoutubeInputError('20분 이하의 정상적인 음원만 지원합니다.')
        title=' '.join(str(info.get('title') or video_id).split())[:300]
        yield YoutubeAudio(target,{'kind':'youtube','url':canonical,'video_id':video_id,'title':title,
            'channel':info.get('channel'),'duration_seconds':decoded.duration,
            'audio_sha256':sha256(target.read_bytes()).hexdigest(),
            'retention':'temporary_audio_deleted_after_processing',
            'title_usage':'display_only_not_classification_input'})
    finally:
        if request.resolve().parent!=parent or not request.name.startswith('request-'):
            raise YoutubeInputError('임시 폴더 위치가 바뀌어 자동 정리를 중단했습니다.')
        temporary.cleanup()
