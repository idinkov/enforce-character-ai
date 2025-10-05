"""
YouTube provider for downloading videos from YouTube channels or playlists and extracting frames.
Uses yt-dlp for downloading and scene detection for frame extraction.
"""

import os
import re
import json
import subprocess
from typing import Dict, List, Any, Optional
from .base_provider import BaseProvider
from ..utils.video_processor import VideoProcessor

try:
    import yt_dlp
    YT_DLP_AVAILABLE = True
except ImportError:
    YT_DLP_AVAILABLE = False
    yt_dlp = None


class YoutubeProvider(BaseProvider):
    """Provider for downloading videos from YouTube and extracting frames."""

    def __init__(self, character_name: str = None, progress_callback=None, log_callback=None):
        super().__init__(character_name, progress_callback, log_callback)
        self.video_processor = None

    def download(self, output_dir: str, **params) -> List[str]:
        """Download videos from YouTube and extract frames."""
        if not YT_DLP_AVAILABLE:
            raise ImportError("yt-dlp library is required. Install with: pip install yt-dlp")

        video_url = params.get('video_url')
        channel_url = params.get('channel_url')
        playlist_url = params.get('playlist_url')
        max_count = params.get('max_count', 50)
        quality = params.get('quality', 'highest')  # highest, lowest, 720p, 480p, etc.
        extract_frames = params.get('extract_frames', True)  # Extract frames instead of keeping video
        frames_per_scene = params.get('frames_per_scene', 1)  # Frames to extract per scene
        scene_threshold = params.get('scene_threshold', 30.0)  # Scene detection threshold
        keep_videos = params.get('keep_videos', False)  # Keep original video files after frame extraction

        # Initialize video processor if frame extraction is enabled
        if extract_frames:
            try:
                self.video_processor = VideoProcessor(
                    default_frames_per_scene=frames_per_scene,
                    scene_threshold=scene_threshold
                )
            except ImportError as e:
                print(f"Warning: Video processing not available: {e}")
                extract_frames = False

        downloaded_files = []

        try:
            if video_url:
                downloaded_files.extend(self._download_single_video(
                    video_url, output_dir, quality, extract_frames, frames_per_scene, keep_videos, max_count
                ))
            elif channel_url:
                downloaded_files.extend(self._download_channel_videos(
                    channel_url, output_dir, max_count, quality, extract_frames, frames_per_scene, keep_videos
                ))
            elif playlist_url:
                downloaded_files.extend(self._download_playlist_videos(
                    playlist_url, output_dir, max_count, quality, extract_frames, frames_per_scene, keep_videos
                ))
            else:
                raise ValueError("Must specify video_url, channel_url, or playlist_url")
        except Exception as e:
            print(f"Error downloading from YouTube: {e}")

        # Update history with downloaded items
        item_ids = [os.path.basename(f) for f in downloaded_files]
        self.update_history(item_ids)

        return downloaded_files

    def _download_single_video(self, video_url: str, output_dir: str, quality: str, extract_frames: bool, frames_per_scene: int, keep_videos: bool, max_count: int = 1) -> List[str]:
        """Download a single YouTube video using yt-dlp and optionally extract frames."""
        downloaded_files = []

        try:
            print(f"Processing video: {video_url}")

            # Clean the URL to remove extra parameters that might cause issues
            clean_url = self._clean_youtube_url(video_url)
            video_id = self._extract_video_id(clean_url)

            if not self.is_duplicate(video_id, output_dir):
                # Create temporary directory for video download if extracting frames
                if extract_frames and not keep_videos:
                    temp_dir = os.path.join(output_dir, 'temp_videos')
                    os.makedirs(temp_dir, exist_ok=True)
                    video_output_dir = temp_dir
                else:
                    video_output_dir = output_dir

                filepath = self._download_video_with_yt_dlp(clean_url, video_output_dir, quality)
                if filepath:
                    if extract_frames and self.video_processor:
                        # Extract frames from the downloaded video
                        extracted_frames = self._extract_frames_from_video(
                            filepath, output_dir, frames_per_scene
                        )
                        downloaded_files.extend(extracted_frames)

                        # Clean up video file if not keeping it
                        if not keep_videos:
                            try:
                                os.remove(filepath)
                            except OSError:
                                pass
                    else:
                        downloaded_files.append(filepath)
            else:
                print(f"Video {video_id} already downloaded, skipping...")

        except Exception as e:
            print(f"Error downloading video {video_url}: {e}")

        return downloaded_files

    def _download_channel_videos(self, channel_url: str, output_dir: str, max_count: int, quality: str, extract_frames: bool, frames_per_scene: int, keep_videos: bool) -> List[str]:
        """Download videos from a YouTube channel using yt-dlp and optionally extract frames."""
        downloaded_files = []

        try:
            if not YT_DLP_AVAILABLE:
                raise ImportError("yt-dlp library is required for channel downloads")

            # Create temporary directory for video downloads if extracting frames
            if extract_frames and not keep_videos:
                temp_dir = os.path.join(output_dir, 'temp_videos')
                os.makedirs(temp_dir, exist_ok=True)
                video_output_dir = temp_dir
            else:
                video_output_dir = output_dir

            # Get video URLs from channel using yt-dlp
            video_urls = self._get_channel_video_urls(channel_url, max_count)

            count = 0
            for video_url in video_urls:
                if count >= max_count:
                    break

                try:
                    video_id = self._extract_video_id(video_url)
                    if not self.is_duplicate(video_id, output_dir):
                        filepath = self._download_video_with_yt_dlp(video_url, video_output_dir, quality)
                        if filepath:
                            if extract_frames and self.video_processor:
                                # Extract frames from the downloaded video
                                extracted_frames = self._extract_frames_from_video(
                                    filepath, output_dir, frames_per_scene
                                )
                                downloaded_files.extend(extracted_frames)

                                # Clean up video file if not keeping it
                                if not keep_videos:
                                    try:
                                        os.remove(filepath)
                                    except OSError:
                                        pass
                            else:
                                downloaded_files.append(filepath)
                            count += 1
                except Exception as e:
                    print(f"Error downloading video from channel: {e}")
                    continue

        except Exception as e:
            print(f"Error downloading channel {channel_url}: {e}")

        return downloaded_files

    def _download_playlist_videos(self, playlist_url: str, output_dir: str, max_count: int, quality: str, extract_frames: bool, frames_per_scene: int, keep_videos: bool) -> List[str]:
        """Download videos from a YouTube playlist using yt-dlp and optionally extract frames."""
        downloaded_files = []

        try:
            if not YT_DLP_AVAILABLE:
                raise ImportError("yt-dlp library is required for playlist downloads")

            # Create temporary directory for video downloads if extracting frames
            if extract_frames and not keep_videos:
                temp_dir = os.path.join(output_dir, 'temp_videos')
                os.makedirs(temp_dir, exist_ok=True)
                video_output_dir = temp_dir
            else:
                video_output_dir = output_dir

            # Get video URLs from playlist using yt-dlp
            video_urls = self._get_playlist_video_urls(playlist_url, max_count)

            count = 0
            for video_url in video_urls:
                if count >= max_count:
                    break

                try:
                    video_id = self._extract_video_id(video_url)
                    if not self.is_duplicate(video_id, output_dir):
                        filepath = self._download_video_with_yt_dlp(video_url, video_output_dir, quality)
                        if filepath:
                            if extract_frames and self.video_processor:
                                # Extract frames from the downloaded video
                                extracted_frames = self._extract_frames_from_video(
                                    filepath, output_dir, frames_per_scene
                                )
                                downloaded_files.extend(extracted_frames)

                                # Clean up video file if not keeping it
                                if not keep_videos:
                                    try:
                                        os.remove(filepath)
                                    except OSError:
                                        pass
                            else:
                                downloaded_files.append(filepath)
                            count += 1
                except Exception as e:
                    print(f"Error downloading video from playlist: {e}")
                    continue

        except Exception as e:
            print(f"Error downloading playlist {playlist_url}: {e}")

        return downloaded_files

    def _extract_frames_from_video(self, video_path: str, output_dir: str, frames_per_scene: int) -> List[str]:
        """Extract frames from a downloaded video."""
        try:
            print(f"Extracting frames from video: {os.path.basename(video_path)}")
            extracted_frames = self.video_processor.extract_frames_from_video(
                video_path, output_dir, frames_per_scene
            )
            print(f"Extracted {len(extracted_frames)} frames from video")
            return extracted_frames
        except Exception as e:
            print(f"Error extracting frames from video {video_path}: {e}")
            return []

    def _download_video_with_yt_dlp(self, video_url: str, output_dir: str, quality: str) -> Optional[str]:
        """Download a YouTube video using yt-dlp with retry logic."""
        import time

        max_retries = 3
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                print(f"Downloading with yt-dlp (attempt {attempt + 1}): {video_url}")

                # Configure yt-dlp options
                ydl_opts = {
                    'outtmpl': os.path.join(output_dir, '%(title)s_%(id)s.%(ext)s'),
                    'format': self._get_format_selector(quality),
                    'noplaylist': True,
                    'extractaudio': False,
                    'ignoreerrors': True,
                    'no_warnings': False,
                }

                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    # Extract info first to get the filename
                    info = ydl.extract_info(video_url, download=False)
                    if not info:
                        print(f"Could not extract video info for: {video_url}")
                        continue

                    # Generate the expected filename
                    filename = ydl.prepare_filename(info)

                    # Download the video
                    ydl.download([video_url])

                    # Check if file exists and return path
                    if os.path.exists(filename):
                        print(f"Successfully downloaded: {os.path.basename(filename)}")
                        return filename
                    else:
                        print(f"Download completed but file not found: {filename}")
                        return None

            except Exception as e:
                error_msg = str(e)
                print(f"Attempt {attempt + 1} failed: {error_msg}")

                # Check for specific errors that might be resolved with retry
                if "HTTP Error 400" in error_msg or "Bad Request" in error_msg:
                    if attempt < max_retries - 1:
                        print(f"HTTP 400 error detected, retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        continue
                elif "unavailable" in error_msg.lower() or "private" in error_msg.lower():
                    print(f"Video is unavailable or private, skipping...")
                    return None
                elif attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    continue
                else:
                    print(f"All retry attempts failed for video: {video_url}")
                    return None

        return None

    def _get_format_selector(self, quality: str) -> str:
        """Get youtube-dl format selector based on quality preference."""
        if quality == 'highest':
            return 'best[ext=mp4]/best'
        elif quality == 'lowest':
            return 'worst[ext=mp4]/worst'
        elif quality in ['720p', '1080p', '480p', '360p']:
            # Try to get specific resolution, fallback to best
            height = quality.replace('p', '')
            return f'best[height<={height}][ext=mp4]/best[ext=mp4]/best'
        else:
            # Default to best quality
            return 'best[ext=mp4]/best'

    def _get_channel_video_urls(self, channel_url: str, max_count: int) -> List[str]:
        """Get video URLs from a YouTube channel using yt-dlp."""
        try:
            ydl_opts = {
                'quiet': True,
                'extract_flat': True,
                'playlistend': max_count,
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(channel_url, download=False)
                if 'entries' in info:
                    video_urls = []
                    for entry in info['entries'][:max_count]:
                        if entry and 'id' in entry:
                            video_urls.append(f"https://www.youtube.com/watch?v={entry['id']}")
                    return video_urls

        except Exception as e:
            print(f"Error extracting channel videos: {e}")

        return []

    def _get_playlist_video_urls(self, playlist_url: str, max_count: int) -> List[str]:
        """Get video URLs from a YouTube playlist using yt-dlp."""
        try:
            ydl_opts = {
                'quiet': True,
                'extract_flat': True,
                'playlistend': max_count,
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(playlist_url, download=False)
                if 'entries' in info:
                    video_urls = []
                    for entry in info['entries'][:max_count]:
                        if entry and 'id' in entry:
                            video_urls.append(f"https://www.youtube.com/watch?v={entry['id']}")
                    return video_urls

        except Exception as e:
            print(f"Error extracting playlist videos: {e}")

        return []

    def _extract_video_id(self, url: str) -> str:
        """Extract video ID from YouTube URL."""
        patterns = [
            r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
            r'(?:embed\/)([0-9A-Zaz_-]{11})',
            r'(?:watch\?v=)([0-9A-Za-z_-]{11})'
        ]

        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)

        return url

    def get_required_params(self) -> List[str]:
        """Return list of required parameters."""
        return []  # At least one of video_url, channel_url, or playlist_url is required

    def get_optional_params(self) -> List[str]:
        """Return list of optional parameters."""
        return ['video_url', 'channel_url', 'playlist_url', 'max_count', 'quality', 'extract_frames', 'frames_per_scene', 'scene_threshold', 'keep_videos']

    @classmethod
    def can_handle_url(cls, url: str) -> bool:
        """Check if this provider can handle the given URL."""
        return 'youtube.com' in url.lower() or 'youtu.be' in url.lower()

    @classmethod
    def extract_params_from_url(cls, url: str) -> Dict[str, Any]:
        """Extract provider parameters from YouTube URL."""
        params = {}

        if '/watch?v=' in url or 'youtu.be/' in url:
            # Single video URL
            params['video_url'] = url
        elif '/channel/' in url or '/c/' in url or '/user/' in url:
            # Channel URL
            params['channel_url'] = url
        elif '/playlist?list=' in url:
            # Playlist URL
            params['playlist_url'] = url

        return params

    def get_param_description(self, param_name: str) -> str:
        """Get human-readable description for parameter."""
        descriptions = {
            'video_url': 'URL of a single YouTube video to download',
            'channel_url': 'URL of a YouTube channel to download videos from',
            'playlist_url': 'URL of a YouTube playlist to download videos from',
            'max_count': 'Maximum number of videos to download from channel/playlist',
            'quality': 'Video quality (highest, lowest, 720p, 480p, etc.)',
            'extract_frames': 'Extract frames from videos instead of keeping video files',
            'frames_per_scene': 'Number of frames to extract per scene change',
            'scene_threshold': 'Threshold for detecting scene changes in videos',
            'keep_videos': 'Keep original video files after frame extraction'
        }
        return descriptions.get(param_name, param_name)

    def _clean_youtube_url(self, url: str) -> str:
        """Clean YouTube URL to remove parameters that might cause issues."""
        import urllib.parse

        # Parse the URL
        parsed = urllib.parse.urlparse(url)

        if 'youtube.com' in parsed.netloc:
            # For youtube.com URLs, keep only the v parameter
            query_params = urllib.parse.parse_qs(parsed.query)
            if 'v' in query_params:
                video_id = query_params['v'][0]
                return f"https://www.youtube.com/watch?v={video_id}"
        elif 'youtu.be' in parsed.netloc:
            # For youtu.be URLs, extract video ID from path
            video_id = parsed.path.lstrip('/')
            return f"https://www.youtube.com/watch?v={video_id}"

        return url


