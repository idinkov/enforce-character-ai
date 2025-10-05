"""
YouTube provider for downloading videos from YouTube channels or playlists and extracting frames.
Uses pytube library for downloading and scene detection for frame extraction.
"""

import os
import re
from typing import Dict, List, Any, Optional
from .base_provider import BaseProvider
from ..utils.video_processor import VideoProcessor

try:
    from pytube import YouTube, Channel, Playlist
    PYTUBE_AVAILABLE = True
except ImportError:
    PYTUBE_AVAILABLE = False
    # Define placeholder classes to avoid NameError
    YouTube = None
    Channel = None
    Playlist = None


class YoutubeProvider(BaseProvider):
    """Provider for downloading videos from YouTube and extracting frames."""

    def __init__(self, character_name: str = None, progress_callback=None, log_callback=None):
        super().__init__(character_name, progress_callback, log_callback)
        self.video_processor = None

    def download(self, output_dir: str, **params) -> List[str]:
        """Download videos from YouTube and extract frames."""
        if not PYTUBE_AVAILABLE:
            raise ImportError("pytube library is required. Install with: pip install pytube")

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
                    video_url, output_dir, quality, extract_frames, frames_per_scene, keep_videos
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

    def _download_single_video(self, video_url: str, output_dir: str, quality: str, extract_frames: bool, frames_per_scene: int, keep_videos: bool) -> List[str]:
        """Download a single YouTube video and optionally extract frames."""
        downloaded_files = []

        try:
            yt = YouTube(video_url)
            video_id = self._extract_video_id(video_url)

            if not self.is_duplicate(video_id, output_dir):
                # Create temporary directory for video download if extracting frames
                if extract_frames and not keep_videos:
                    temp_dir = os.path.join(output_dir, 'temp_videos')
                    os.makedirs(temp_dir, exist_ok=True)
                    video_output_dir = temp_dir
                else:
                    video_output_dir = output_dir

                filepath = self._download_video(yt, video_output_dir, quality)
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
        except Exception as e:
            print(f"Error downloading video {video_url}: {e}")

        return downloaded_files

    def _download_channel_videos(self, channel_url: str, output_dir: str, max_count: int, quality: str, extract_frames: bool, frames_per_scene: int, keep_videos: bool) -> List[str]:
        """Download videos from a YouTube channel and optionally extract frames."""
        downloaded_files = []

        try:
            channel = Channel(channel_url)
            count = 0

            # Create temporary directory for video downloads if extracting frames
            if extract_frames and not keep_videos:
                temp_dir = os.path.join(output_dir, 'temp_videos')
                os.makedirs(temp_dir, exist_ok=True)
                video_output_dir = temp_dir
            else:
                video_output_dir = output_dir

            for video in channel.video_urls:
                if count >= max_count:
                    break

                try:
                    yt = YouTube(video)
                    video_id = self._extract_video_id(video)

                    if not self.is_duplicate(video_id, output_dir):
                        filepath = self._download_video(yt, video_output_dir, quality)
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
        """Download videos from a YouTube playlist and optionally extract frames."""
        downloaded_files = []

        try:
            playlist = Playlist(playlist_url)
            count = 0

            # Create temporary directory for video downloads if extracting frames
            if extract_frames and not keep_videos:
                temp_dir = os.path.join(output_dir, 'temp_videos')
                os.makedirs(temp_dir, exist_ok=True)
                video_output_dir = temp_dir
            else:
                video_output_dir = output_dir

            for video_url in playlist.video_urls:
                if count >= max_count:
                    break

                try:
                    yt = YouTube(video_url)
                    video_id = self._extract_video_id(video_url)

                    if not self.is_duplicate(video_id, output_dir):
                        filepath = self._download_video(yt, video_output_dir, quality)
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

    def _download_video(self, yt, output_dir: str, quality: str) -> Optional[str]:
        """Download a YouTube video."""
        try:
            if quality == 'highest':
                stream = yt.streams.get_highest_resolution()
            elif quality == 'lowest':
                stream = yt.streams.get_lowest_resolution()
            else:
                stream = yt.streams.filter(res=quality).first()
                if not stream:
                    stream = yt.streams.get_highest_resolution()

            if stream:
                filename = f"{self.get_valid_filename(yt.title)}_{yt.video_id}.mp4"
                filepath = os.path.join(output_dir, filename)
                stream.download(output_path=output_dir, filename=filename)
                print(f"Downloaded video: {filename}")
                return filepath

        except Exception as e:
            print(f"Error downloading video {yt.title}: {e}")

        return None

    def _extract_video_id(self, url: str) -> str:
        """Extract video ID from YouTube URL."""
        patterns = [
            r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
            r'(?:embed\/)([0-9A-Za-z_-]{11})',
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
