"""
다운로드 관련 유틸리티
"""

import os
import tempfile
import subprocess
import requests
import re
from urllib.parse import urlparse


def validate_url(url):
    """URL이 유효한 음악 링크인지 검증합니다"""
    try:
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            return False, "유효하지 않은 URL 형식입니다."
        
        # YouTube 링크 검증
        if 'youtube.com' in parsed.netloc or 'youtu.be' in parsed.netloc:
            return True, "YouTube 링크"
        
        # Spotify 링크 검증
        if 'spotify.com' in parsed.netloc:
            return True, "Spotify 링크"
        
        # 일반 음악 파일 링크 검증
        music_extensions = ['.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac']
        if any(parsed.path.lower().endswith(ext) for ext in music_extensions):
            return True, "직접 음악 파일 링크"
        
        return False, "지원하지 않는 링크 형식입니다."
        
    except Exception as e:
        return False, f"URL 검증 중 오류 발생: {str(e)}"


def download_youtube_audio(url, output_path=None):
    """YouTube 링크에서 오디오를 다운로드합니다"""
    try:
        if output_path is None:
            output_path = tempfile.mktemp(suffix='.mp3')
        
        # yt-dlp 사용
        cmd = [
            'yt-dlp',
            '--extract-audio',
            '--audio-format', 'mp3',
            '--audio-quality', '0',
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
                if file.startswith(base_name) and file.endswith('.mp3'):
                    return os.path.join(dir_path, file)
        
        raise Exception("다운로드된 파일을 찾을 수 없습니다.")
        
    except FileNotFoundError:
        raise Exception("yt-dlp가 설치되어 있지 않습니다. 'pip install yt-dlp'로 설치하세요.")
    except Exception as e:
        raise Exception(f"YouTube 다운로드 중 오류 발생: {str(e)}")


def download_direct_audio(url, output_path=None):
    """직접 음악 파일 링크에서 다운로드합니다"""
    try:
        if output_path is None:
            output_path = tempfile.mktemp(suffix='.mp3')
        
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
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


def get_youtube_info(url):
    """YouTube 비디오 정보를 가져옵니다"""
    try:
        video_id = extract_video_id_from_youtube(url)
        if not video_id:
            raise Exception("YouTube 비디오 ID를 추출할 수 없습니다.")
        
        # yt-dlp를 사용하여 비디오 정보 가져오기
        cmd = ['yt-dlp', '--dump-json', url]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise Exception(f"YouTube 정보 가져오기 실패: {result.stderr}")
        
        import json
        info = json.loads(result.stdout)
        
        return {
            'title': info.get('title', 'Unknown'),
            'duration': info.get('duration', 0),
            'uploader': info.get('uploader', 'Unknown'),
            'view_count': info.get('view_count', 0),
            'description': info.get('description', '')[:200] + '...' if info.get('description') else ''
        }
        
    except Exception as e:
        return {
            'title': 'Unknown',
            'duration': 0,
            'uploader': 'Unknown',
            'view_count': 0,
            'description': f'정보 가져오기 실패: {str(e)}'
        }


def get_link_preview(url):
    """링크의 미리보기 정보를 가져옵니다"""
    try:
        # URL 검증
        is_valid, message = validate_url(url)
        if not is_valid:
            return {'error': message}
        
        # YouTube 링크 처리
        if 'youtube.com' in url or 'youtu.be' in url:
            try:
                info = get_youtube_info(url)
                return {
                    'platform': 'YouTube',
                    'title': info['title'],
                    'uploader': info['uploader'],
                    'duration': info['duration'],
                    'view_count': info['view_count'],
                    'description': info['description']
                }
            except Exception as e:
                return {'error': f'YouTube 정보 가져오기 실패: {str(e)}'}
        
        # Spotify 링크 처리
        elif 'spotify.com' in url:
            return {
                'platform': 'Spotify',
                'error': 'Spotify 링크는 현재 지원하지 않습니다.'
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