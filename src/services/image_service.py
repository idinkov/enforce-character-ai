"""
Image management services with deleted images tracking.
"""
from pathlib import Path
from typing import List, Set, Optional, Dict, Tuple, Callable
import shutil
import threading
import time
import hashlib
from concurrent.futures import ThreadPoolExecutor
from PIL import Image, ImageTk
import yaml
from datetime import datetime
from src.config.app_config import config


class ThumbnailCache:
    """In-memory cache for thumbnails with LRU eviction."""

    def __init__(self, max_size: int = 500, max_memory_mb: int = 100):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.cache: Dict[str, Tuple[ImageTk.PhotoImage, float, int]] = {}
        self.access_times: Dict[str, float] = {}
        self.total_size_bytes = 0
        self.lock = threading.RLock()

    def _get_cache_key(self, image_path: Path, thumbnail_size: Tuple[int, int]) -> str:
        """Generate cache key from image path and size."""
        try:
            stat = image_path.stat()
            path_hash = hashlib.md5(f"{image_path}_{stat.st_mtime}_{stat.st_size}_{thumbnail_size}".encode()).hexdigest()
            return path_hash
        except:
            return hashlib.md5(f"{image_path}_{thumbnail_size}".encode()).hexdigest()

    def get(self, image_path: Path, thumbnail_size: Tuple[int, int]) -> Optional[ImageTk.PhotoImage]:
        """Get cached thumbnail."""
        key = self._get_cache_key(image_path, thumbnail_size)
        with self.lock:
            if key in self.cache:
                image, _, size_bytes = self.cache[key]
                self.access_times[key] = time.time()
                self.cache[key] = (image, self.access_times[key], size_bytes)
                return image
        return None

    def put(self, image_path: Path, thumbnail_size: Tuple[int, int], image: ImageTk.PhotoImage) -> None:
        """Cache thumbnail with LRU eviction."""
        key = self._get_cache_key(image_path, thumbnail_size)
        estimated_size = thumbnail_size[0] * thumbnail_size[1] * 4

        with self.lock:
            current_time = time.time()
            if key in self.cache:
                _, _, old_size = self.cache[key]
                self.total_size_bytes -= old_size

            self.cache[key] = (image, current_time, estimated_size)
            self.access_times[key] = current_time
            self.total_size_bytes += estimated_size
            self._evict_if_necessary()

    def _evict_if_necessary(self):
        """Evict oldest entries if cache is too large."""
        while len(self.cache) > self.max_size:
            self._evict_oldest()
        while self.total_size_bytes > self.max_memory_bytes and self.cache:
            self._evict_oldest()

    def _evict_oldest(self):
        """Remove the least recently used item."""
        if not self.cache:
            return
        oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        if oldest_key in self.cache:
            _, _, size_bytes = self.cache[oldest_key]
            self.total_size_bytes -= size_bytes
            del self.cache[oldest_key]
            del self.access_times[oldest_key]

    def clear(self):
        """Clear all cached thumbnails."""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
            self.total_size_bytes = 0

    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        with self.lock:
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'memory_mb': self.total_size_bytes // (1024 * 1024),
                'max_memory_mb': self.max_memory_bytes // (1024 * 1024)
            }


class AsyncThumbnailLoader:
    """Asynchronous thumbnail loader with prioritization."""

    def __init__(self, cache: ThumbnailCache, max_workers: int = 4):
        self.cache = cache
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="thumbnail_loader")
        self.pending_requests: Dict[str, threading.Event] = {}
        self.lock = threading.Lock()

    def load_thumbnail_async(self, image_path: Path, thumbnail_size: Tuple[int, int],
                            callback: Callable, priority: int = 0) -> None:
        """Load thumbnail asynchronously."""
        cache_key = self.cache._get_cache_key(image_path, thumbnail_size)

        cached = self.cache.get(image_path, thumbnail_size)
        if cached:
            callback(image_path, cached)
            return

        with self.lock:
            if cache_key in self.pending_requests:
                return
            event = threading.Event()
            self.pending_requests[cache_key] = event

        self.executor.submit(self._load_thumbnail_sync, image_path, thumbnail_size, callback, cache_key)

    def _load_thumbnail_sync(self, image_path: Path, thumbnail_size: Tuple[int, int],
                            callback: Callable, cache_key: str) -> None:
        """Synchronously load thumbnail in background thread."""
        try:
            cached = self.cache.get(image_path, thumbnail_size)
            if cached:
                callback(image_path, cached)
                return

            thumbnail = self._create_thumbnail_optimized(image_path, thumbnail_size)
            if thumbnail:
                self.cache.put(image_path, thumbnail_size, thumbnail)
                callback(image_path, thumbnail)
            else:
                callback(image_path, None)

        except Exception as e:
            print(f"Error loading thumbnail for {image_path}: {e}")
            callback(image_path, None)

        finally:
            with self.lock:
                if cache_key in self.pending_requests:
                    self.pending_requests[cache_key].set()
                    del self.pending_requests[cache_key]

    def _create_thumbnail_optimized(self, image_path: Path, thumbnail_size: Tuple[int, int]) -> Optional[ImageTk.PhotoImage]:
        """Create optimized thumbnail."""
        try:
            with Image.open(image_path) as img:
                if img.mode not in ('RGB', 'RGBA'):
                    img = img.convert('RGB')
                img.thumbnail(thumbnail_size, Image.Resampling.NEAREST)
                return ImageTk.PhotoImage(img)
        except Exception as e:
            print(f"Error creating optimized thumbnail for {image_path}: {e}")
            return None

    def shutdown(self):
        """Shutdown the async loader."""
        self.executor.shutdown(wait=True)


class ImageService:
    """Service for managing character images with deleted images tracking."""

    def __init__(self, characters_path: Path):
        self.characters_path = characters_path
        self.thumbnail_cache = ThumbnailCache(
            max_size=config.THUMBNAIL_CACHE_SIZE,
            max_memory_mb=config.THUMBNAIL_CACHE_MEMORY_MB
        )
        self.face_cache = ThumbnailCache(
            max_size=config.FACE_CACHE_SIZE,
            max_memory_mb=config.FACE_CACHE_MEMORY_MB
        )
        self.async_loader = AsyncThumbnailLoader(
            self.thumbnail_cache,
            max_workers=config.THUMBNAIL_WORKER_THREADS
        )

    def _get_deleted_file_path(self, character_name: str) -> Path:
        """Get the path to the deleted.yaml file for a character."""
        return self.characters_path / character_name / "deleted.yaml"

    def _load_deleted_images(self, character_name: str) -> Set[str]:
        """Load deleted images for a specific character."""
        deleted_file = self._get_deleted_file_path(character_name)
        if deleted_file.exists():
            try:
                with open(deleted_file, 'r', encoding='utf-8') as file:
                    deleted_data = yaml.safe_load(file) or {}
                    return set(deleted_data.get('deleted_images', []))
            except Exception as e:
                print(f"Error loading deleted images for {character_name}: {e}")
        return set()

    def _save_deleted_images(self, character_name: str, deleted_images: Set[str]):
        """Save deleted images for a specific character."""
        deleted_file = self._get_deleted_file_path(character_name)
        deleted_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            deleted_data = {
                'deleted_images': sorted(list(deleted_images)),
                'last_updated': datetime.now().isoformat()
            }
            with open(deleted_file, 'w', encoding='utf-8') as file:
                yaml.dump(deleted_data, file, default_flow_style=False, allow_unicode=True)
        except Exception as e:
            print(f"Error saving deleted images for {character_name}: {e}")

    def _add_to_deleted_list(self, character_name: str, image_path: Path):
        """Add an image to the deleted list for a character."""
        filename = image_path.name
        deleted_images = self._load_deleted_images(character_name)
        deleted_images.add(filename)
        self._save_deleted_images(character_name, deleted_images)

    def _is_image_deleted(self, character_name: str, filename: str) -> bool:
        """Check if an image filename is in the deleted list for a character."""
        deleted_images = self._load_deleted_images(character_name)
        return filename in deleted_images

    def _remove_from_deleted_list(self, character_name: str, filename: str):
        """Remove an image from the deleted list."""
        deleted_images = self._load_deleted_images(character_name)
        deleted_images.discard(filename)
        self._save_deleted_images(character_name, deleted_images)

    def _extract_character_name_from_path(self, img_path: Path) -> Optional[str]:
        """Extract character name from image path structure."""
        try:
            relative_path = img_path.relative_to(self.characters_path)
            return relative_path.parts[0] if relative_path.parts else None
        except ValueError:
            try:
                path_parts = img_path.parts
                if 'characters' in path_parts:
                    characters_idx = path_parts.index('characters')
                    if characters_idx + 1 < len(path_parts):
                        return path_parts[characters_idx + 1]
            except Exception as e:
                print(f"Error extracting character name from {img_path}: {e}")
        return None

    def upload_images(self, character_name: str, stage: str, file_paths: List[str]) -> int:
        """Upload images to a character's stage folder, skipping previously deleted images."""
        target_dir = self.characters_path / character_name / "images" / stage
        target_dir.mkdir(parents=True, exist_ok=True)

        count = 0
        skipped = 0

        for file_path in file_paths:
            try:
                filename = Path(file_path).name

                # Check if this image was previously deleted
                if self._is_image_deleted(character_name, filename):
                    print(f"Skipping previously deleted image: {filename}")
                    skipped += 1
                    continue

                target_path = target_dir / filename

                # Handle name conflicts
                counter = 1
                original_filename = filename
                while target_path.exists():
                    name, ext = Path(original_filename).stem, Path(original_filename).suffix
                    filename = f"{name}_{counter}{ext}"
                    target_path = target_dir / filename
                    counter += 1

                shutil.copy2(file_path, target_path)
                count += 1

                # If we had to rename the file, remove the original from deleted list
                if filename != original_filename:
                    self._remove_from_deleted_list(character_name, original_filename)

            except Exception as e:
                print(f"Error uploading {file_path}: {e}")

        if skipped > 0:
            print(f"Uploaded {count} images, skipped {skipped} previously deleted images")

        return count

    def delete_images(self, image_paths: Set[Path]) -> List[str]:
        """Delete selected images and add them to the character's deleted list."""
        errors = []

        for img_path in image_paths:
            try:
                character_name = self._extract_character_name_from_path(img_path)
                if not character_name:
                    errors.append(f"Could not determine character for {img_path}")
                    continue

                img_path.unlink()
                self._invalidate_image_cache(img_path)
                self._add_to_deleted_list(character_name, img_path)

            except Exception as e:
                errors.append(f"Error deleting {img_path}: {e}")

        return errors

    def count_stage_images(self, character_name: str, stage: str) -> int:
        """Count images in a specific stage for a character."""
        try:
            stage_path = self.characters_path / character_name / "images" / stage
            if not stage_path.exists():
                return 0

            count = 0
            for ext in config.IMAGE_EXTENSIONS:
                count += len(list(stage_path.rglob(ext)))
            return count
        except Exception:
            return 0

    def get_stage_images(self, character_name: str, stage: str) -> List[Path]:
        """Get all image files in a stage folder, with deleted images sorted to the bottom."""
        stage_path = self.characters_path / character_name / "images" / stage

        if not stage_path.exists():
            return []

        image_files = []
        seen_files = set()

        for ext in config.IMAGE_EXTENSIONS:
            files = stage_path.rglob(ext)
            for file_path in files:
                relative_path = file_path.relative_to(stage_path)
                file_key = str(relative_path).lower()
                if file_key not in seen_files:
                    seen_files.add(file_key)
                    image_files.append(file_path)

        # Sort images so that deleted images appear at the bottom
        # Load deleted images list once for efficiency
        deleted_images = self._load_deleted_images(character_name)

        def sort_key(image_path: Path):
            # Return tuple: (is_deleted, filename_lower)
            # is_deleted: False (0) for non-deleted, True (1) for deleted
            # This ensures non-deleted images come first, then deleted images
            # Within each group, sort alphabetically by filename
            is_deleted = image_path.name in deleted_images
            return is_deleted, image_path.name.lower()

        image_files.sort(key=sort_key)
        return image_files

    def stage_has_images(self, character_name: str, stage: str) -> bool:
        """Check if a stage has any images."""
        return self.count_stage_images(character_name, stage) > 0

    def get_current_stage(self, character_name: str) -> str:
        """Get the highest completed stage for a character."""
        try:
            stage_order = ["7_final_dataset", "6_rtt_1024", "5_processed_fixed_1024",
                          "4_processed_1024", "3_raw_upscaled", "2_raw_filtered", "1_raw"]

            for i, stage in enumerate(stage_order):
                if self.stage_has_images(character_name, stage):
                    return str(7 - i)
            return "0"
        except Exception:
            return "0"

    def is_character_completed(self, character_name: str) -> bool:
        """Check if character has completed all stages."""
        return self.stage_has_images(character_name, "7_final_dataset")

    def _invalidate_image_cache(self, image_path: Path):
        """Invalidate cache entries for an image."""
        # Simplified invalidation - could be enhanced to track specific cache keys
        pass

    def create_thumbnail_async(self, image_path: Path, callback: Callable, priority: int = 0):
        """Create thumbnail asynchronously."""
        self.async_loader.load_thumbnail_async(image_path, config.THUMBNAIL_SIZE, callback, priority)

    def create_face_thumbnail(self, image_path: Path) -> Optional[ImageTk.PhotoImage]:
        """Create a small face thumbnail for display."""
        cached = self.face_cache.get(image_path, config.FACE_IMAGE_SIZE)
        if cached:
            return cached

        try:
            with Image.open(image_path) as img:
                if img.mode not in ('RGB', 'RGBA'):
                    img = img.convert('RGB')
                img.thumbnail(config.FACE_IMAGE_SIZE, Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                self.face_cache.put(image_path, config.FACE_IMAGE_SIZE, photo)
                return photo
        except Exception as e:
            print(f"Error creating face thumbnail for {image_path}: {e}")
            return None

    def get_cache_stats(self) -> Dict[str, Dict[str, int]]:
        """Get cache statistics."""
        return {
            'thumbnail_cache': self.thumbnail_cache.get_stats(),
            'face_cache': self.face_cache.get_stats()
        }

    def clear_caches(self):
        """Clear all caches."""
        self.thumbnail_cache.clear()
        self.face_cache.clear()

    def shutdown(self):
        """Shutdown image service and cleanup resources."""
        self.async_loader.shutdown()
        self.clear_caches()


class StageProgressTracker:
    """Tracks completion status of character stages."""

    def __init__(self, image_service: ImageService):
        self.image_service = image_service
        self.completion_cache = {}

    def get_completion_status(self, character_name: str) -> dict:
        """Get completion status for all stages of a character."""
        status = {}
        completed_count = 0

        for stage_id, _, _ in config.STAGES:
            is_completed = self.image_service.stage_has_images(character_name, stage_id)
            status[stage_id] = is_completed
            if is_completed:
                completed_count += 1

        status['completed_count'] = completed_count
        status['total_stages'] = len(config.STAGES)
        status['progress_percentage'] = (completed_count / len(config.STAGES)) * 100

        return status

    def invalidate_cache(self, character_name: str):
        """Invalidate cached completion status for a character."""
        cache_key = f"{character_name}_completion"
        if cache_key in self.completion_cache:
            del self.completion_cache[cache_key]
