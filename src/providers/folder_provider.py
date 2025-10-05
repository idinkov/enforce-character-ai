"""
Folder provider for importing images and videos from local directories.
"""

import os
import shutil
import threading
from pathlib import Path
from typing import Dict, List, Any
from .base_provider import BaseProvider
from ..utils.video_processor import VideoProcessor, is_video_file, get_supported_video_extensions


class FolderProvider(BaseProvider):
    """Provider for importing images and videos from local folders."""

    def __init__(self, character_name: str = None, progress_callback=None, log_callback=None):
        super().__init__(character_name, progress_callback, log_callback)
        self.video_processor = None

    def download(self, output_dir: str, **params) -> List[str]:
        """
        Copy images and extract frames from videos in local folder to output directory.

        Args:
            output_dir: Directory to save copied images and extracted frames
            **params: Parameters including 'folder_path', optional 'max_count', 'preserve_names',
                     'frames_per_scene', 'process_videos', 'scene_threshold'

        Returns:
            List of copied/extracted file paths
        """
        folder_path = params.get('folder_path', '')
        max_count = params.get('max_count', 10000)  # Default to 10000 max files
        preserve_names = params.get('preserve_names', True)  # Keep original filenames
        process_videos = params.get('process_videos', True)  # Process video files
        frames_per_scene = params.get('frames_per_scene', 1)  # Frames to extract per scene
        scene_threshold = params.get('scene_threshold', 30.0)  # Scene detection threshold

        if not folder_path:
            raise ValueError("folder_path parameter is required")

        if not os.path.exists(folder_path):
            raise ValueError(f"Folder path does not exist: {folder_path}")

        if not os.path.isdir(folder_path):
            raise ValueError(f"Path is not a directory: {folder_path}")

        # Initialize video processor if needed
        if process_videos:
            try:
                self.video_processor = VideoProcessor(
                    default_frames_per_scene=frames_per_scene,
                    scene_threshold=scene_threshold,
                    progress_callback=self.progress_callback,
                    log_callback=self.log_callback
                )
            except ImportError as e:
                self.log(f"Warning: Video processing not available: {e}")
                process_videos = False

        processed_files = []
        supported_image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
        supported_video_extensions = set(get_supported_video_extensions())

        try:
            # Get all image and video files from the folder
            source_folder = Path(folder_path)
            media_files = []

            self.log(f"Scanning folder: {folder_path}")

            for file_path in source_folder.iterdir():
                if file_path.is_file():
                    ext = file_path.suffix.lower()
                    if ext in supported_image_extensions:
                        media_files.append(('image', file_path))
                    elif process_videos and ext in supported_video_extensions:
                        media_files.append(('video', file_path))

            # Sort files by name for consistent ordering
            media_files.sort(key=lambda x: x[1].name.lower())

            self.log(f"Found {len(media_files)} media files to process")

            # Limit to max_count if specified
            if max_count > 0:
                media_files = media_files[:max_count]
                self.log(f"Limited to {len(media_files)} files (max_count: {max_count})")

            # Process each media file
            total_files = len(media_files)
            for i, (file_type, source_file) in enumerate(media_files):
                try:
                    # Update progress
                    if self.progress_callback:
                        self.progress_callback(i, total_files, f"Processing {source_file.name}")

                    self.log(f"Processing {file_type}: {source_file.name}")

                    if file_type == 'image':
                        processed_files.extend(self._process_image_file(
                            source_file, output_dir, preserve_names, i
                        ))
                    elif file_type == 'video':
                        processed_files.extend(self._process_video_file(
                            source_file, output_dir, frames_per_scene
                        ))

                except Exception as e:
                    self.log(f"Error processing file {source_file}: {e}")
                    continue

            # Final progress update
            if self.progress_callback:
                self.progress_callback(total_files, total_files, "Processing complete")

        except Exception as e:
            self.log(f"Error processing folder {folder_path}: {e}")

        # Update history with processed items
        item_ids = [os.path.basename(f) for f in processed_files]
        self.update_history(item_ids)

        self.log(f"Successfully processed {len(processed_files)} files from folder")
        return processed_files

    def _process_image_file(self, source_file: Path, output_dir: str, preserve_names: bool, index: int) -> List[str]:
        """Process a single image file."""
        processed_files = []

        # Check if this file was already imported
        source_filename = source_file.name
        if self.is_duplicate(source_filename, output_dir):
            self.log(f"Skipping already imported file: {source_filename}")
            return processed_files

        if preserve_names:
            # Keep original filename
            filename = source_file.name
            output_path = os.path.join(output_dir, filename)

            # Handle filename conflicts by adding a number suffix
            counter = 1
            while os.path.exists(output_path):
                name_part = source_file.stem
                ext_part = source_file.suffix
                filename = f"{name_part}_{counter}{ext_part}"
                output_path = os.path.join(output_dir, filename)
                counter += 1
        else:
            # Use numbered naming scheme
            file_extension = source_file.suffix.lower()
            filename = f"folder_import_{index:04d}_{source_file.stem}{file_extension}"
            filename = self.get_valid_filename(filename)
            output_path = os.path.join(output_dir, filename)

        # Copy the file
        shutil.copy2(source_file, output_path)
        processed_files.append(output_path)
        self.log(f"Copied image: {source_file.name} -> {filename}")

        return processed_files

    def _process_video_file(self, source_file: Path, output_dir: str, frames_per_scene: int) -> List[str]:
        """Process a single video file by extracting frames."""
        processed_files = []

        if not self.video_processor:
            self.log(f"Skipping video file (video processing not available): {source_file.name}")
            return processed_files

        # Check if this video was already processed by looking for existing frames
        video_stem = source_file.stem
        existing_frames = [f for f in os.listdir(output_dir) if f.startswith(f"{video_stem}_scene_")]

        if existing_frames:
            self.log(f"Skipping already processed video: {source_file.name}")
            return [os.path.join(output_dir, f) for f in existing_frames]

        try:
            self.log(f"Processing video: {source_file.name}")
            extracted_frames = self.video_processor.extract_frames_from_video(
                str(source_file), output_dir, frames_per_scene
            )
            processed_files.extend(extracted_frames)
            self.log(f"Extracted {len(extracted_frames)} frames from video: {source_file.name}")

        except Exception as e:
            self.log(f"Error processing video {source_file.name}: {e}")

        return processed_files

    def get_required_params(self) -> List[str]:
        """Return list of required parameters for folder provider."""
        return ['folder_path']

    def get_optional_params(self) -> List[str]:
        """Return list of optional parameters for folder provider."""
        return ['max_count', 'preserve_names', 'process_videos', 'frames_per_scene', 'scene_threshold']

    @classmethod
    def can_handle_url(cls, url: str) -> bool:
        """
        Check if this provider can handle the given URL.
        For folder provider, we check if it's a local file path.
        """
        # Check if it looks like a Windows or Unix file path
        if os.path.exists(url) and os.path.isdir(url):
            return True

        # Check for common folder path patterns
        if (url.startswith(('C:', 'D:', 'E:', 'F:', 'G:', 'H:')) or  # Windows drive letters
            url.startswith('/') or  # Unix absolute path
            url.startswith('./') or  # Relative path
            url.startswith('../')):  # Parent relative path
            return True

        return False

    @classmethod
    def extract_params_from_url(cls, url: str) -> Dict[str, Any]:
        """
        Extract provider parameters from folder path.
        """
        params = {}

        if cls.can_handle_url(url):
            params['folder_path'] = url

        return params

    def get_param_description(self, param_name: str) -> str:
        """Get human-readable description for parameter."""
        descriptions = {
            'folder_path': 'Local folder path containing images to import',
            'max_count': 'Maximum number of images to import (0 = no limit)',
            'preserve_names': 'Whether to preserve original file names',
            'process_videos': 'Whether to process video files and extract frames',
            'frames_per_scene': 'Number of frames to extract per scene change in videos',
            'scene_threshold': 'Threshold for detecting scene changes in videos'
        }
        return descriptions.get(param_name, param_name)

    def validate_params(self, params: Dict[str, Any]) -> tuple[bool, str]:
        """
        Validate provider parameters.

        Returns:
            Tuple of (is_valid, error_message)
        """
        folder_path = params.get('folder_path', '')

        if not folder_path:
            return False, "Folder path is required"

        if not os.path.exists(folder_path):
            return False, f"Folder path does not exist: {folder_path}"

        if not os.path.isdir(folder_path):
            return False, f"Path is not a directory: {folder_path}"

        # Check if folder contains any image or video files
        supported_image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
        supported_video_extensions = set(get_supported_video_extensions())
        source_folder = Path(folder_path)

        has_media_files = any(
            file_path.is_file() and (file_path.suffix.lower() in supported_image_extensions or
            (file_path.suffix.lower() in supported_video_extensions and is_video_file(str(file_path))))
            for file_path in source_folder.iterdir()
        )

        if not has_media_files:
            return False, f"No supported image or video files found in folder: {folder_path}"

        max_count = params.get('max_count', 100000)
        if isinstance(max_count, str):
            try:
                max_count = int(max_count)
            except ValueError:
                return False, "Max count must be a number"

        if max_count < 0:
            return False, "Max count cannot be negative"

        return True, ""
