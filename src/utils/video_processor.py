"""
Video processing utilities for extracting frames from videos using scene detection.
"""

import os
import yaml
import cv2
import ffmpeg
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from datetime import datetime

try:
    from scenedetect import VideoManager, SceneManager
    from scenedetect.detectors import ContentDetector
    # Try to import FrameTimecode to check if we have the old API
    try:
        from scenedetect.frame_timecode import FrameTimecode
        HAS_FRAME_TIMECODE = True
    except ImportError:
        HAS_FRAME_TIMECODE = False
    SCENEDETECT_AVAILABLE = True
except ImportError:
    SCENEDETECT_AVAILABLE = False
    HAS_FRAME_TIMECODE = False


class VideoProcessor:
    """Handles video processing including scene detection and frame extraction."""

    def __init__(self, default_frames_per_scene: int = 1, scene_threshold: float = 30.0):
        """
        Initialize video processor.

        Args:
            default_frames_per_scene: Default number of frames to extract per scene
            scene_threshold: Threshold for scene detection sensitivity (lower = more sensitive)
        """
        if not SCENEDETECT_AVAILABLE:
            raise ImportError("PySceneDetect is required. Install with: pip install scenedetect")

        self.default_frames_per_scene = default_frames_per_scene
        self.scene_threshold = scene_threshold

    def _get_frame_number(self, time_or_tuple: Any, fps: float) -> int:
        """
        Extract frame number from either FrameTimecode object or tuple.
        This handles compatibility between different PySceneDetect versions.
        """
        if hasattr(time_or_tuple, 'get_frames'):
            # Old API: FrameTimecode object
            return time_or_tuple.get_frames()
        elif isinstance(time_or_tuple, (tuple, list)) and len(time_or_tuple) >= 2:
            # New API: tuple format (seconds, frame_number)
            return int(time_or_tuple[1])
        elif hasattr(time_or_tuple, 'frame_num'):
            # Alternative new API format
            return time_or_tuple.frame_num
        else:
            # Fallback: convert seconds to frame number
            try:
                seconds = float(time_or_tuple) if not hasattr(time_or_tuple, 'get_seconds') else time_or_tuple.get_seconds()
                return int(seconds * fps)
            except (ValueError, TypeError):
                return 0

    def _get_seconds(self, time_or_tuple: Any) -> float:
        """
        Extract seconds from either FrameTimecode object or tuple.
        This handles compatibility between different PySceneDetect versions.
        """
        if hasattr(time_or_tuple, 'get_seconds'):
            # Old API: FrameTimecode object
            return time_or_tuple.get_seconds()
        elif isinstance(time_or_tuple, (tuple, list)) and len(time_or_tuple) >= 1:
            # New API: tuple format (seconds, frame_number)
            return float(time_or_tuple[0])
        elif hasattr(time_or_tuple, 'seconds'):
            # Alternative new API format
            return time_or_tuple.seconds
        else:
            # Fallback: assume it's already seconds
            try:
                return float(time_or_tuple)
            except (ValueError, TypeError):
                return 0.0

    def extract_frames_from_video(
        self,
        video_path: str,
        output_dir: str,
        frames_per_scene: Optional[int] = None,
        force_reprocess: bool = False
    ) -> List[str]:
        """
        Extract frames from video using scene detection.

        Args:
            video_path: Path to the video file
            output_dir: Directory to save extracted frames
            frames_per_scene: Number of frames to extract per scene (defaults to instance setting)
            force_reprocess: If True, ignore cached scene data and reprocess

        Returns:
            List of extracted frame file paths
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        frames_per_scene = frames_per_scene or self.default_frames_per_scene

        # Check for cached scene data
        scene_data_path = video_path.parent / f"{video_path.stem}_scene_data.yaml"
        scene_data = None

        if not force_reprocess and scene_data_path.exists():
            try:
                scene_data = self._load_scene_data(scene_data_path, video_path)
            except Exception as e:
                print(f"Warning: Could not load cached scene data: {e}")
                scene_data = None

        # If no cached data or forced reprocessing, detect scenes
        if scene_data is None:
            print(f"Detecting scenes in video: {video_path.name}")
            scene_data = self._detect_scenes(video_path)
            self._save_scene_data(scene_data, scene_data_path, video_path)
        else:
            print(f"Using cached scene data for: {video_path.name}")

        # Extract frames from scenes
        extracted_frames = self._extract_frames_from_scenes(
            video_path, scene_data['scenes'], output_dir, frames_per_scene
        )

        return extracted_frames

    def _detect_scenes(self, video_path: Path) -> Dict[str, Any]:
        """Detect scenes in the video using PySceneDetect."""
        video_manager = VideoManager([str(video_path)])
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=self.scene_threshold))

        # Start video processing
        video_manager.start()
        scene_manager.detect_scenes(frame_source=video_manager)
        scene_list = scene_manager.get_scene_list()

        # Get video info
        fps = video_manager.get_framerate()

        # Handle different PySceneDetect API versions for getting total frames
        try:
            duration = video_manager.get_duration()
            if hasattr(duration, 'get_frames'):
                total_frames = duration.get_frames()
            else:
                total_frames = self._get_frame_number(duration, fps)
        except Exception as e:
            print(f"Warning: Could not get total frames, using fallback calculation: {e}")
            # Fallback: use video file info
            try:
                import cv2
                cap = cv2.VideoCapture(str(video_path))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
            except:
                total_frames = 0

        video_manager.release()

        # Convert scene list to serializable format
        scenes = []
        for i, (start_time, end_time) in enumerate(scene_list):
            start_frame = self._get_frame_number(start_time, fps)
            end_frame = self._get_frame_number(end_time, fps)
            start_seconds = self._get_seconds(start_time)
            end_seconds = self._get_seconds(end_time)

            scenes.append({
                'scene_number': i + 1,
                'start_frame': start_frame,
                'end_frame': end_frame,
                'start_seconds': start_seconds,
                'end_seconds': end_seconds,
                'duration_seconds': end_seconds - start_seconds
            })

        scene_data = {
            'video_info': {
                'filename': video_path.name,
                'fps': fps,
                'total_frames': total_frames,
                'detection_threshold': self.scene_threshold
            },
            'detection_timestamp': datetime.now().isoformat(),
            'scene_count': len(scenes),
            'scenes': scenes
        }

        print(f"Detected {len(scenes)} scenes in video")
        return scene_data

    def _save_scene_data(self, scene_data: Dict[str, Any], scene_data_path: Path, video_path: Path):
        """Save scene detection data to YAML file."""
        try:
            with open(scene_data_path, 'w', encoding='utf-8') as f:
                yaml.safe_dump(scene_data, f, default_flow_style=False, allow_unicode=True)
            print(f"Saved scene data to: {scene_data_path}")
        except Exception as e:
            print(f"Warning: Could not save scene data: {e}")

    def _load_scene_data(self, scene_data_path: Path, video_path: Path) -> Dict[str, Any]:
        """Load and validate cached scene detection data."""
        with open(scene_data_path, 'r', encoding='utf-8') as f:
            scene_data = yaml.safe_load(f)

        # Validate the scene data is for the current video
        if scene_data.get('video_info', {}).get('filename') != video_path.name:
            raise ValueError("Scene data filename mismatch")

        # Check if video file was modified since scene detection
        video_mtime = os.path.getmtime(video_path)
        detection_time = datetime.fromisoformat(scene_data.get('detection_timestamp', '1970-01-01'))

        if video_mtime > detection_time.timestamp():
            raise ValueError("Video file modified since scene detection")

        return scene_data

    def _extract_frames_from_scenes(
        self,
        video_path: Path,
        scenes: List[Dict[str, Any]],
        output_dir: Path,
        frames_per_scene: int
    ) -> List[str]:
        """Extract frames from detected scenes."""
        extracted_frames = []

        # Open video with OpenCV
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file: {video_path}")

        try:
            for scene in scenes:
                scene_num = scene['scene_number']
                start_frame = scene['start_frame']
                end_frame = scene['end_frame']

                # Calculate frame positions to extract
                frame_positions = self._calculate_frame_positions(
                    start_frame, end_frame, frames_per_scene
                )

                for i, frame_pos in enumerate(frame_positions):
                    # Set video position to desired frame
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
                    ret, frame = cap.read()

                    if ret:
                        # Generate filename
                        if frames_per_scene == 1:
                            filename = f"{video_path.stem}_scene_{scene_num:03d}.jpg"
                        else:
                            filename = f"{video_path.stem}_scene_{scene_num:03d}_frame_{i+1:02d}.jpg"

                        frame_path = output_dir / filename

                        # Save frame
                        cv2.imwrite(str(frame_path), frame)
                        extracted_frames.append(str(frame_path))

                        print(f"Extracted frame: {filename}")
                    else:
                        print(f"Warning: Could not read frame at position {frame_pos}")

        finally:
            cap.release()

        print(f"Extracted {len(extracted_frames)} frames from {len(scenes)} scenes")
        return extracted_frames

    def _calculate_frame_positions(
        self,
        start_frame: int,
        end_frame: int,
        frames_per_scene: int
    ) -> List[int]:
        """Calculate which frame positions to extract from a scene."""
        if frames_per_scene <= 0:
            return []

        scene_length = end_frame - start_frame

        if frames_per_scene == 1:
            # Extract middle frame
            return [start_frame + scene_length // 2]

        if frames_per_scene >= scene_length:
            # Extract all frames in scene
            return list(range(start_frame, end_frame))

        # Extract evenly spaced frames
        positions = []
        step = scene_length / (frames_per_scene + 1)  # +1 to avoid first and last frame

        for i in range(1, frames_per_scene + 1):
            pos = int(start_frame + step * i)
            positions.append(pos)

        return positions

    def get_video_info(self, video_path: str) -> Dict[str, Any]:
        """Get basic information about a video file."""
        try:
            probe = ffmpeg.probe(video_path)
            video_stream = next(
                (stream for stream in probe['streams'] if stream['codec_type'] == 'video'),
                None
            )

            if video_stream is None:
                raise ValueError("No video stream found")

            return {
                'duration': float(probe['format']['duration']),
                'width': int(video_stream['width']),
                'height': int(video_stream['height']),
                'fps': eval(video_stream['r_frame_rate']),  # Convert fraction to float
                'codec': video_stream['codec_name'],
                'format': probe['format']['format_name']
            }
        except Exception as e:
            print(f"Warning: Could not get video info for {video_path}: {e}")
            return {}


def is_video_file(file_path: str) -> bool:
    """Check if a file is a supported video format."""
    video_extensions = {
        '.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm',
        '.m4v', '.3gp', '.ogv', '.f4v', '.asf', '.rm', '.rmvb'
    }
    return Path(file_path).suffix.lower() in video_extensions


def get_supported_video_extensions() -> List[str]:
    """Get list of supported video file extensions."""
    return [
        '.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm',
        '.m4v', '.3gp', '.ogv', '.f4v', '.asf', '.rm', '.rmvb'
    ]
