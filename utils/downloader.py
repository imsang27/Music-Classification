"""
다운로드 관련 유틸리티
"""

import os
import tempfile
import subprocess
import requests
import re
import json
from urllib.parse import urlparse


def validate_url(url, supported_platforms=None):
    """URL이 유효한 음악 링크인지 검증합니다 (지원 플랫폼 옵션 포함)"""
    if supported_platforms is None:
        supported_platforms = ['youtube', 'spotify', 'direct']
    
    try:
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            return False, "유효하지 않은 URL 형식입니다."
        
        # YouTube 링크 검증
        if 'youtube' in supported_platforms and ('youtube.com' in parsed.netloc or 'youtu.be' in parsed.netloc):
            return True, "YouTube 링크"
        
        # Spotify 링크 검증
        if 'spotify' in supported_platforms and 'spotify.com' in parsed.netloc:
            return True, "Spotify 링크"
        
        # 일반 음악 파일 링크 검증
        if 'direct' in supported_platforms:
            music_extensions = ['.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac']
            if any(parsed.path.lower().endswith(ext) for ext in music_extensions):
                return True, "직접 음악 파일 링크"
        
        return False, "지원하지 않는 링크 형식입니다."
        
    except Exception as e:
        return False, f"URL 검증 중 오류 발생: {str(e)}"


def download_youtube_audio(url, output_path=None, audio_format='mp3', audio_quality='0'):
    """YouTube 링크에서 오디오를 다운로드합니다 (형식 및 품질 옵션 포함)"""
    try:
        if output_path is None:
            output_path = tempfile.mktemp(suffix=f'.{audio_format}')
        
        # ffmpeg 경로 설정
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ffmpeg_path = os.path.join(current_dir, 'ffmpeg-master-latest-win64-gpl', 'bin')
        
        # yt-dlp 사용 (ffmpeg 경로 명시)
        cmd = [
            'yt-dlp',
            '--extract-audio',
            '--audio-format', audio_format,
            '--audio-quality', audio_quality,
            '--ffmpeg-location', ffmpeg_path,
            '--output', output_path,
            url
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise Exception(f"YouTube 다운로드 실패: {result.stderr}")
        
        # 실제 다운로드된 파일 경로 찾기
        if os.path.exists(output_path):
            return output_path
        else:
            # yt-dlp가 생성한 실제 파일명 찾기
            dir_path = os.path.dirname(output_path)
            base_name = os.path.splitext(os.path.basename(output_path))[0]
            for file in os.listdir(dir_path):
                if file.startswith(base_name) and file.endswith(f'.{audio_format}'):
                    return os.path.join(dir_path, file)
        
        raise Exception("다운로드된 파일을 찾을 수 없습니다.")
        
    except FileNotFoundError:
        raise Exception("yt-dlp가 설치되어 있지 않습니다. 'pip install yt-dlp'로 설치하세요.")
    except Exception as e:
        raise Exception(f"YouTube 다운로드 중 오류 발생: {str(e)}")


def download_direct_audio(url, output_path=None, timeout=30, chunk_size=8192):
    """직접 음악 파일 링크에서 다운로드합니다 (타임아웃 및 청크 크기 옵션 포함)"""
    try:
        if output_path is None:
            # URL에서 파일 확장자 추출
            parsed_url = urlparse(url)
            file_ext = os.path.splitext(parsed_url.path)[1]
            if not file_ext:
                file_ext = '.mp3'  # 기본값
            output_path = tempfile.mktemp(suffix=file_ext)
        
        response = requests.get(url, stream=True, timeout=timeout)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                f.write(chunk)
        
        return output_path
        
    except Exception as e:
        raise Exception(f"직접 다운로드 중 오류 발생: {str(e)}")


def extract_video_id_from_youtube(url):
    """YouTube URL에서 비디오 ID를 추출합니다"""
    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/)([^&\n?#]+)',
        r'youtube\.com\/embed\/([^&\n?#]+)',
        r'youtube\.com\/v\/([^&\n?#]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    
    return None


def get_youtube_info(url, max_description_length=200):
    """YouTube 비디오 정보를 가져옵니다 (설명 길이 옵션 포함)"""
    try:
        video_id = extract_video_id_from_youtube(url)
        if not video_id:
            raise Exception("YouTube 비디오 ID를 추출할 수 없습니다.")
        
        # yt-dlp를 사용하여 비디오 정보 가져오기
        cmd = ['yt-dlp', '--dump-json', url]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise Exception(f"YouTube 정보 가져오기 실패: {result.stderr}")
        
        info = json.loads(result.stdout)
        
        # 설명 길이 제한
        description = info.get('description', '')
        if description and len(description) > max_description_length:
            description = description[:max_description_length] + '...'
        
        return {
            'title': info.get('title', 'Unknown'),
            'duration': info.get('duration', 0),
            'uploader': info.get('uploader', 'Unknown'),
            'view_count': info.get('view_count', 0),
            'description': description,
            'video_id': video_id,
            'url': url
        }
        
    except Exception as e:
        return {
            'title': 'Unknown',
            'duration': 0,
            'uploader': 'Unknown',
            'view_count': 0,
            'description': f'정보 가져오기 실패: {str(e)}',
            'video_id': None,
            'url': url
        }


def get_link_preview(url, max_description_length=200):
    """링크의 미리보기 정보를 가져옵니다 (설명 길이 옵션 포함)"""
    try:
        # URL 검증
        is_valid, message = validate_url(url)
        if not is_valid:
            return {'error': message}
        
        # YouTube 링크 처리
        if 'youtube.com' in url or 'youtu.be' in url:
            try:
                info = get_youtube_info(url, max_description_length)
                return {
                    'platform': 'YouTube',
                    'title': info['title'],
                    'uploader': info['uploader'],
                    'duration': info['duration'],
                    'view_count': info['view_count'],
                    'description': info['description'],
                    'video_id': info.get('video_id'),
                    'url': url
                }
            except Exception as e:
                return {'error': f'YouTube 정보 가져오기 실패: {str(e)}'}
        
        # Spotify 링크 처리
        elif 'spotify.com' in url:
            return {
                'platform': 'Spotify',
                'error': 'Spotify 링크는 현재 지원하지 않습니다.',
                'url': url
            }
        
        # 직접 음악 파일 링크 처리
        else:
            try:
                # HEAD 요청으로 파일 정보 확인
                response = requests.head(url, timeout=10)
                response.raise_for_status()
                
                content_type = response.headers.get('content-type', '')
                content_length = response.headers.get('content-length', 0)
                
                # 파일명 추출
                parsed_url = urlparse(url)
                filename = os.path.basename(parsed_url.path)
                if not filename:
                    filename = 'unknown'
                
                return {
                    'platform': '직접 링크',
                    'title': filename,
                    'content_type': content_type,
                    'file_size': int(content_length) if content_length else '알 수 없음',
                    'url': url
                }
            except Exception as e:
                return {'error': f'파일 정보 가져오기 실패: {str(e)}'}
                
    except Exception as e:
        return {'error': f'미리보기 생성 중 오류 발생: {str(e)}'}


def batch_download_audio(urls, output_dir=None, audio_format='mp3', audio_quality='0'):
    """여러 URL에서 오디오를 배치로 다운로드"""
    if output_dir is None:
        output_dir = tempfile.mkdtemp()
    
    results = []
    
    for i, url in enumerate(urls):
        try:
            print(f"다운로드 중... ({i+1}/{len(urls)}): {url}")
            
            # URL 검증
            is_valid, message = validate_url(url)
            if not is_valid:
                results.append({
                    'url': url,
                    'success': False,
                    'error': message
                })
                continue
            
            # 다운로드 실행
            if 'youtube.com' in url or 'youtu.be' in url:
                output_path = os.path.join(output_dir, f"youtube_{i}.{audio_format}")
                downloaded_path = download_youtube_audio(url, output_path, audio_format, audio_quality)
            else:
                output_path = os.path.join(output_dir, f"direct_{i}.mp3")
                downloaded_path = download_direct_audio(url, output_path)
            
            results.append({
                'url': url,
                'success': True,
                'file_path': downloaded_path
            })
            
        except Exception as e:
            results.append({
                'url': url,
                'success': False,
                'error': str(e)
            })
    
    return results 