"""
Video file provider for importing video files directly and extracting frames.
"""

import os
from pathlib import Path
from typing import Dict, List, Any
from .base_provider import BaseProvider
from ..utils.video_processor import VideoProcessor, is_video_file


class VideoFileProvider(BaseProvider):
    """Provider for importing individual video files and extracting frames."""

    def __init__(self, character_name: str = None):
        super().__init__(character_name)
        self.video_processor = None

    def download(self, output_dir: str, **params) -> List[str]:
        """
        Process video files and extract frames.

        Args:
            output_dir: Directory to save extracted frames
            **params: Parameters including 'video_path', 'frames_per_scene', etc.

        Returns:
            List of extracted frame file paths
        """
        video_path = params.get('video_path', '')
        frames_per_scene = params.get('frames_per_scene', 1)  # Frames to extract per scene
        scene_threshold = params.get('scene_threshold', 30.0)  # Scene detection threshold
        force_reprocess = params.get('force_reprocess', False)  # Force reprocessing of cached videos

        if not video_path:
            raise ValueError("video_path parameter is required")

        if not os.path.exists(video_path):
            raise ValueError(f"Video file does not exist: {video_path}")

        if not is_video_file(video_path):
            raise ValueError(f"File is not a supported video format: {video_path}")

        # Initialize video processor
        try:
            self.video_processor = VideoProcessor(
                default_frames_per_scene=frames_per_scene,
                scene_threshold=scene_threshold
            )
        except ImportError as e:
            raise ImportError(f"Video processing not available: {e}")

        extracted_files = []

        try:
            # Check if this video was already processed by looking for existing frames
            video_stem = Path(video_path).stem
            existing_frames = [f for f in os.listdir(output_dir) if f.startswith(f"{video_stem}_scene_")]

            if existing_frames and not force_reprocess:
                print(f"Using existing frames for video: {Path(video_path).name}")
                extracted_files = [os.path.join(output_dir, f) for f in existing_frames]
            else:
                print(f"Processing video: {Path(video_path).name}")
                extracted_files = self.video_processor.extract_frames_from_video(
                    video_path, output_dir, frames_per_scene, force_reprocess
                )

        except Exception as e:
            print(f"Error processing video {video_path}: {e}")

        # Update history with extracted items
        item_ids = [os.path.basename(f) for f in extracted_files]
        self.update_history(item_ids)

        print(f"Successfully extracted {len(extracted_files)} frames from video")
        return extracted_files

    def get_required_params(self) -> List[str]:
        """Return list of required parameters."""
        return ['video_path']

    def get_optional_params(self) -> List[str]:
        """Return list of optional parameters."""
        return ['frames_per_scene', 'scene_threshold', 'force_reprocess']

    @classmethod
    def can_handle_url(cls, url: str) -> bool:
        """Check if this provider can handle the given URL."""
        return os.path.exists(url) and is_video_file(url)

    @classmethod
    def extract_params_from_url(cls, url: str) -> Dict[str, Any]:
        """Extract provider parameters from video file path."""
        params = {}
        if cls.can_handle_url(url):
            params['video_path'] = url
        return params

    def get_param_description(self, param_name: str) -> str:
        """Get human-readable description for parameter."""
        descriptions = {
            'video_path': 'Path to the video file to process',
            'frames_per_scene': 'Number of frames to extract per scene change',
            'scene_threshold': 'Threshold for detecting scene changes in videos',
            'force_reprocess': 'Force reprocessing even if frames already exist'
        }
        return descriptions.get(param_name, param_name)

    def validate_params(self, params: Dict[str, Any]) -> tuple[bool, str]:
        """Validate provider parameters."""
        video_path = params.get('video_path', '')

        if not video_path:
            return False, "Video path is required"

        if not os.path.exists(video_path):
            return False, f"Video file does not exist: {video_path}"

        if not is_video_file(video_path):
            return False, f"File is not a supported video format: {video_path}"

        frames_per_scene = params.get('frames_per_scene', 1)
        if isinstance(frames_per_scene, str):
            try:
                frames_per_scene = int(frames_per_scene)
            except ValueError:
                return False, "Frames per scene must be a number"

        if frames_per_scene < 1:
            return False, "Frames per scene must be at least 1"

        return True, ""
