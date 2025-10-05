"""
Vimeo provider for downloading videos from Vimeo users or albums and extracting frames.
Uses vimeo library and web scraping with scene detection for frame extraction.
"""

import os
import re
import requests
from typing import Dict, List, Any, Optional
from .base_provider import BaseProvider
from ..utils.video_processor import VideoProcessor

try:
    import vimeo
    VIMEO_AVAILABLE = True
except ImportError:
    vimeo = None  # type: ignore
    VIMEO_AVAILABLE = False


class VimeoProvider(BaseProvider):
    """Provider for downloading videos from Vimeo and extracting frames."""

    def __init__(self, character_name: str = None, progress_callback=None, log_callback=None):
        super().__init__(character_name, progress_callback, log_callback)
        # You'll need to get API credentials from Vimeo
        self.client_id = "your_vimeo_client_id"
        self.client_secret = "your_vimeo_client_secret"
        self.access_token = "your_vimeo_access_token"
        self.video_processor = None

    def download(self, output_dir: str, **params) -> List[str]:
        """Download videos from Vimeo and extract frames."""
        video_url = params.get('video_url')
        user_id = params.get('user_id')
        album_id = params.get('album_id')
        search_query = params.get('search_query')
        max_count = params.get('max_count', 50)
        quality = params.get('quality', 'highest')  # highest, lowest, hd, sd
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
            if VIMEO_AVAILABLE and vimeo is not None:
                client = vimeo.VimeoClient(  # type: ignore
                    token=self.access_token,
                    key=self.client_id,
                    secret=self.client_secret
                )

                if video_url:
                    downloaded_files.extend(self._download_single_video(
                        client, video_url, output_dir, quality, extract_frames, frames_per_scene, keep_videos
                    ))
                elif user_id:
                    downloaded_files.extend(self._download_user_videos(
                        client, user_id, output_dir, max_count, quality, extract_frames, frames_per_scene, keep_videos
                    ))
                elif album_id:
                    downloaded_files.extend(self._download_album_videos(
                        client, album_id, output_dir, max_count, quality, extract_frames, frames_per_scene, keep_videos
                    ))
                elif search_query:
                    downloaded_files.extend(self._download_search_results(
                        client, search_query, output_dir, max_count, quality, extract_frames, frames_per_scene, keep_videos
                    ))
                else:
                    raise ValueError("Must specify video_url, user_id, album_id, or search_query")
            else:
                # Fallback to basic scraping
                if video_url:
                    downloaded_files.extend(self._download_video_fallback(video_url, output_dir))

        except Exception as e:
            print(f"Error downloading from Vimeo: {e}")

        # Update history with downloaded items
        item_ids = [os.path.basename(f) for f in downloaded_files]
        self.update_history(item_ids)

        return downloaded_files

    def _download_single_video(self, client, video_url: str, output_dir: str, quality: str, extract_frames: bool, frames_per_scene: int, keep_videos: bool) -> List[str]:
        """Download a single Vimeo video and optionally extract frames."""
        downloaded_files = []

        try:
            video_id = self._extract_video_id(video_url)
            if video_id and not self.is_duplicate(video_id, output_dir):
                response = client.get(f'/videos/{video_id}')
                if response.status_code == 200:
                    video_data = response.json()

                    # Create temporary directory for video download if extracting frames
                    if extract_frames and not keep_videos:
                        temp_dir = os.path.join(output_dir, 'temp_videos')
                        os.makedirs(temp_dir, exist_ok=True)
                        video_output_dir = temp_dir
                    else:
                        video_output_dir = output_dir

                    filepath = self._download_video_file(video_data, video_output_dir, quality)
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

    def _download_user_videos(self, client, user_id: str, output_dir: str, max_count: int, quality: str, extract_frames: bool, frames_per_scene: int, keep_videos: bool) -> List[str]:
        """Download videos from a Vimeo user."""
        downloaded_files = []

        try:
            response = client.get(f'/users/{user_id}/videos', params={'per_page': min(max_count, 100)})
            if response.status_code == 200:
                videos = response.json()['data']

                for video in videos[:max_count]:
                    video_id = video['uri'].split('/')[-1]
                    if not self.is_duplicate(video_id, output_dir):
                        if extract_frames:
                            # For frame extraction, download to temp directory first
                            temp_dir = os.path.join(output_dir, 'temp_videos') if not keep_videos else output_dir
                            os.makedirs(temp_dir, exist_ok=True)
                            filepath = self._download_video_file(video, temp_dir, quality)
                            if filepath and self.video_processor:
                                extracted_frames = self._extract_frames_from_video(
                                    filepath, output_dir, frames_per_scene
                                )
                                downloaded_files.extend(extracted_frames)
                                if not keep_videos:
                                    try:
                                        os.remove(filepath)
                                    except OSError:
                                        pass
                            elif filepath and keep_videos:
                                downloaded_files.append(filepath)
                        else:
                            filepath = self._download_video_file(video, output_dir, quality)
                            if filepath:
                                downloaded_files.append(filepath)
        except Exception as e:
            print(f"Error downloading user videos: {e}")

        return downloaded_files

    def _download_album_videos(self, client, album_id: str, output_dir: str, max_count: int, quality: str, extract_frames: bool, frames_per_scene: int, keep_videos: bool) -> List[str]:
        """Download videos from a Vimeo album."""
        downloaded_files = []

        try:
            response = client.get(f'/albums/{album_id}/videos', params={'per_page': min(max_count, 100)})
            if response.status_code == 200:
                videos = response.json()['data']

                for video in videos[:max_count]:
                    video_id = video['uri'].split('/')[-1]
                    if not self.is_duplicate(video_id, output_dir):
                        if extract_frames:
                            # For frame extraction, download to temp directory first
                            temp_dir = os.path.join(output_dir, 'temp_videos') if not keep_videos else output_dir
                            os.makedirs(temp_dir, exist_ok=True)
                            filepath = self._download_video_file(video, temp_dir, quality)
                            if filepath and self.video_processor:
                                extracted_frames = self._extract_frames_from_video(
                                    filepath, output_dir, frames_per_scene
                                )
                                downloaded_files.extend(extracted_frames)
                                if not keep_videos:
                                    try:
                                        os.remove(filepath)
                                    except OSError:
                                        pass
                            elif filepath and keep_videos:
                                downloaded_files.append(filepath)
                        else:
                            filepath = self._download_video_file(video, output_dir, quality)
                            if filepath:
                                downloaded_files.append(filepath)
        except Exception as e:
            print(f"Error downloading album videos: {e}")

        return downloaded_files

    def _download_search_results(self, client, query: str, output_dir: str, max_count: int, quality: str, extract_frames: bool, frames_per_scene: int, keep_videos: bool) -> List[str]:
        """Download videos from search results."""
        downloaded_files = []

        try:
            response = client.get('/videos', params={'query': query, 'per_page': min(max_count, 100)})
            if response.status_code == 200:
                videos = response.json()['data']

                for video in videos[:max_count]:
                    video_id = video['uri'].split('/')[-1]
                    if not self.is_duplicate(video_id, output_dir):
                        if extract_frames:
                            # For frame extraction, download to temp directory first
                            temp_dir = os.path.join(output_dir, 'temp_videos') if not keep_videos else output_dir
                            os.makedirs(temp_dir, exist_ok=True)
                            filepath = self._download_video_file(video, temp_dir, quality)
                            if filepath and self.video_processor:
                                extracted_frames = self._extract_frames_from_video(
                                    filepath, output_dir, frames_per_scene
                                )
                                downloaded_files.extend(extracted_frames)
                                if not keep_videos:
                                    try:
                                        os.remove(filepath)
                                    except OSError:
                                        pass
                            elif filepath and keep_videos:
                                downloaded_files.append(filepath)
                        else:
                            filepath = self._download_video_file(video, output_dir, quality)
                            if filepath:
                                downloaded_files.append(filepath)
        except Exception as e:
            print(f"Error downloading search results: {e}")

        return downloaded_files

    def _download_video_file(self, video_data: dict, output_dir: str, quality: str) -> Optional[str]:
        """Download video file from Vimeo."""
        try:
            # Extract download links (requires Pro account)
            download_links = video_data.get('download', [])
            if not download_links:
                print(f"No download links available for video {video_data.get('name', 'Unknown')}")
                return None

            # Select quality
            selected_link = None
            if quality == 'highest':
                selected_link = max(download_links, key=lambda x: x.get('height', 0))
            elif quality == 'lowest':
                selected_link = min(download_links, key=lambda x: x.get('height', 999999))
            else:
                # Look for specific quality
                for link in download_links:
                    if quality.lower() in link.get('quality', '').lower():
                        selected_link = link
                        break
                if not selected_link:
                    selected_link = download_links[0]

            if selected_link:
                url = selected_link['link']
                filename = f"{self.get_valid_filename(video_data['name'])}_{video_data['uri'].split('/')[-1]}.mp4"
                filepath = os.path.join(output_dir, filename)

                response = requests.get(url, stream=True)
                response.raise_for_status()

                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                return filepath

        except Exception as e:
            print(f"Error downloading video file: {e}")

        return None

    def _download_video_fallback(self, video_url: str, output_dir: str) -> List[str]:
        """Fallback method for downloading without API."""
        # This is a simplified implementation
        print("Vimeo API library not available. Limited functionality.")
        print("For full functionality, install: pip install vimeo")
        return []

    def _extract_video_id(self, url: str) -> Optional[str]:
        """Extract video ID from Vimeo URL."""
        match = re.search(r'vimeo\.com/(\d+)', url)
        return match.group(1) if match else None

    def get_required_params(self) -> List[str]:
        """Return list of required parameters."""
        return []  # At least one of video_url, user_id, album_id, or search_query is required

    def get_optional_params(self) -> List[str]:
        """Return list of optional parameters."""
        return ['video_url', 'user_id', 'album_id', 'search_query', 'max_count', 'quality', 'extract_frames', 'frames_per_scene', 'scene_threshold', 'keep_videos']

    @classmethod
    def can_handle_url(cls, url: str) -> bool:
        """Check if this provider can handle the given URL."""
        return 'vimeo.com' in url.lower()

    @classmethod
    def extract_params_from_url(cls, url: str) -> Dict[str, Any]:
        """Extract provider parameters from Vimeo URL."""
        params = {}

        if '/user' in url:
            # User URL
            match = re.search(r'/user(\d+)', url)
            if match:
                params['user_id'] = match.group(1)
        elif '/album' in url:
            # Album URL
            match = re.search(r'/album/(\d+)', url)
            if match:
                params['album_id'] = match.group(1)
        else:
            # Video URL
            params['video_url'] = url

        return params
