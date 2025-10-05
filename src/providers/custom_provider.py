"""
Custom URL provider for downloading media files directly from URLs, including video files.
Supports frame extraction from downloaded videos.
"""

import os
import requests
from pathlib import Path
from typing import Dict, List, Any
from urllib.parse import urlparse
from .base_provider import BaseProvider
from ..utils.video_processor import VideoProcessor, is_video_file, get_supported_video_extensions


class CustomProvider(BaseProvider):
    """Provider for downloading media files directly from URLs with video processing support."""

    def __init__(self, character_name: str = None, progress_callback=None, log_callback=None):
        super().__init__(character_name, progress_callback, log_callback)
        self.video_processor = None

    def download(self, output_dir: str, **params) -> List[str]:
        """
        Download media files from direct URLs and extract frames from videos.

        Args:
            output_dir: Directory to save downloaded media and extracted frames
            **params: Parameters including 'urls', 'max_count', 'extract_frames', etc.

        Returns:
            List of downloaded/extracted file paths
        """
        urls = params.get('urls', [])
        if isinstance(urls, str):
            urls = [urls]  # Convert single URL to list

        max_count = params.get('max_count', 100)
        extract_frames = params.get('extract_frames', True)  # Extract frames from videos
        frames_per_scene = params.get('frames_per_scene', 1)  # Frames to extract per scene
        scene_threshold = params.get('scene_threshold', 30.0)  # Scene detection threshold
        keep_videos = params.get('keep_videos', False)  # Keep original video files after frame extraction

        if not urls:
            raise ValueError("At least one URL must be provided")

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
        supported_image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
        supported_video_extensions = set(get_supported_video_extensions())

        try:
            for i, url in enumerate(urls[:max_count]):
                try:
                    # Determine file type from URL
                    parsed_url = urlparse(url)
                    file_extension = Path(parsed_url.path).suffix.lower()

                    # Skip if already downloaded
                    url_filename = self._generate_filename_from_url(url)
                    if self.is_duplicate(url_filename, output_dir):
                        print(f"Skipping already downloaded URL: {url}")
                        continue

                    if file_extension in supported_image_extensions:
                        # Download as image
                        downloaded_files.extend(self._download_image(url, output_dir, i))
                    elif file_extension in supported_video_extensions or self._is_likely_video_url(url):
                        # Download as video and optionally extract frames
                        downloaded_files.extend(self._download_video(
                            url, output_dir, extract_frames, frames_per_scene, keep_videos, i
                        ))
                    else:
                        # Try to download and detect file type
                        downloaded_files.extend(self._download_unknown_media(
                            url, output_dir, extract_frames, frames_per_scene, keep_videos, i
                        ))

                except Exception as e:
                    print(f"Error processing URL {url}: {e}")
                    continue

        except Exception as e:
            print(f"Error downloading from URLs: {e}")

        # Update history with downloaded items
        item_ids = [os.path.basename(f) for f in downloaded_files]
        self.update_history(item_ids)

        print(f"Successfully processed {len(downloaded_files)} files from URLs")
        return downloaded_files

    def _download_image(self, url: str, output_dir: str, index: int) -> List[str]:
        """Download an image file from URL."""
        downloaded_files = []

        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

            # Generate filename
            filename = self._generate_filename_from_url(url, index, 'image')
            filepath = os.path.join(output_dir, filename)

            # Download the file
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            downloaded_files.append(filepath)
            print(f"Downloaded image: {filename}")

        except Exception as e:
            print(f"Error downloading image from {url}: {e}")

        return downloaded_files

    def _download_video(self, url: str, output_dir: str, extract_frames: bool, frames_per_scene: int, keep_videos: bool, index: int) -> List[str]:
        """Download a video file from URL and optionally extract frames."""
        downloaded_files = []

        try:
            # Create temporary directory for video download if extracting frames
            if extract_frames and not keep_videos:
                temp_dir = os.path.join(output_dir, 'temp_videos')
                os.makedirs(temp_dir, exist_ok=True)
                video_output_dir = temp_dir
            else:
                video_output_dir = output_dir

            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()

            # Generate filename
            filename = self._generate_filename_from_url(url, index, 'video')
            filepath = os.path.join(video_output_dir, filename)

            # Download the video file
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            print(f"Downloaded video: {filename}")

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
            print(f"Error downloading video from {url}: {e}")

        return downloaded_files

    def _download_unknown_media(self, url: str, output_dir: str, extract_frames: bool, frames_per_scene: int, keep_videos: bool, index: int) -> List[str]:
        """Download media file with unknown type and detect type from content."""
        downloaded_files = []

        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

            # Try to detect content type from headers
            content_type = response.headers.get('content-type', '').lower()

            if 'image' in content_type:
                # Handle as image
                filename = self._generate_filename_from_url(url, index, 'image')
                filepath = os.path.join(output_dir, filename)

                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                downloaded_files.append(filepath)
                print(f"Downloaded image: {filename}")

            elif 'video' in content_type:
                # Handle as video
                return self._download_video(url, output_dir, extract_frames, frames_per_scene, keep_videos, index)
            else:
                # Download and check file signature
                filename = self._generate_filename_from_url(url, index)
                filepath = os.path.join(output_dir, filename)

                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                # Check if it's a video file
                if is_video_file(filepath):
                    if extract_frames and self.video_processor:
                        # Extract frames and optionally remove video
                        extracted_frames = self._extract_frames_from_video(
                            filepath, output_dir, frames_per_scene
                        )
                        downloaded_files.extend(extracted_frames)

                        if not keep_videos:
                            try:
                                os.remove(filepath)
                            except OSError:
                                pass
                    else:
                        downloaded_files.append(filepath)
                else:
                    downloaded_files.append(filepath)

                print(f"Downloaded file: {filename}")

        except Exception as e:
            print(f"Error downloading unknown media from {url}: {e}")

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

    def _generate_filename_from_url(self, url: str, index: int = 0, file_type: str = None) -> str:
        """Generate a safe filename from URL."""
        parsed_url = urlparse(url)

        # Try to get filename from URL path
        url_filename = Path(parsed_url.path).name
        if url_filename and '.' in url_filename:
            base_name = Path(url_filename).stem
            extension = Path(url_filename).suffix
        else:
            # Generate name from domain and index
            domain = parsed_url.netloc.replace('www.', '')
            base_name = f"url_{domain}_{index:04d}"

            # Set extension based on file type
            if file_type == 'image':
                extension = '.jpg'
            elif file_type == 'video':
                extension = '.mp4'
            else:
                extension = ''

        filename = f"{base_name}{extension}"
        return self.get_valid_filename(filename)

    def _is_likely_video_url(self, url: str) -> bool:
        """Check if URL is likely to be a video file based on patterns."""
        video_patterns = ['mp4', 'avi', 'mov', 'mkv', 'webm', 'flv', 'm4v']
        url_lower = url.lower()
        return any(pattern in url_lower for pattern in video_patterns)

    def get_required_params(self) -> List[str]:
        """Return list of required parameters."""
        return ['urls']

    def get_optional_params(self) -> List[str]:
        """Return list of optional parameters."""
        return ['max_count', 'extract_frames', 'frames_per_scene', 'scene_threshold', 'keep_videos']

    @classmethod
    def can_handle_url(cls, url: str) -> bool:
        """Check if this provider can handle the given URL."""
        # This provider can handle any HTTP/HTTPS URL
        return url.startswith(('http://', 'https://'))

    @classmethod
    def extract_params_from_url(cls, url: str) -> Dict[str, Any]:
        """Extract provider parameters from URL."""
        params = {}
        if cls.can_handle_url(url):
            params['urls'] = [url]
        return params

    def get_param_description(self, param_name: str) -> str:
        """Get human-readable description for parameter."""
        descriptions = {
            'urls': 'List of direct URLs to download media files from',
            'max_count': 'Maximum number of URLs to process',
            'extract_frames': 'Extract frames from video files instead of keeping videos',
            'frames_per_scene': 'Number of frames to extract per scene change in videos',
            'scene_threshold': 'Threshold for detecting scene changes in videos',
            'keep_videos': 'Keep original video files after frame extraction'
        }
        return descriptions.get(param_name, param_name)
