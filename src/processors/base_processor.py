"""
Base processor class for image processing stages
"""
import shutil
from pathlib import Path
from datetime import datetime
import threading
from abc import ABC, abstractmethod
import yaml
from concurrent.futures import ThreadPoolExecutor
import time
from ..config.app_config import config


class BaseProcessor(ABC):
    def __init__(self, characters_path, log_callback=None, progress_callback=None, provider_manager=None):
        self.characters_path = Path(characters_path)
        self.log_callback = log_callback or self._default_log
        self.progress_callback = progress_callback or self._default_progress
        self.provider_manager = provider_manager

    def _default_log(self, message):
        """Default logging function"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}")

    def _default_progress(self, progress):
        """Default progress function"""
        print(f"Progress: {progress:.1f}%")

    def log(self, message):
        """Log a message"""
        self.log_callback(message)

    def update_progress(self, progress):
        """Update progress"""
        self.progress_callback(progress)

    def get_image_files(self, directory, character_name=None):
        """Get all image files from a directory recursively, excluding deleted images"""
        image_files = []
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif', '*.tiff', '*.webp']

        # Use a set to avoid duplicates and make case-insensitive matching
        seen_files = set()

        # Get deleted images list if character_name is provided
        deleted_images = set()
        if character_name:
            deleted_images = self._load_deleted_images(character_name)

        for ext in extensions:
            # Use rglob for recursive search instead of glob
            files = directory.rglob(ext)
            for file_path in files:
                # Use lowercase path for deduplication check
                file_key = file_path.name.lower()
                if file_key not in seen_files:
                    # Check if this image is in the deleted list
                    if character_name and file_path.name in deleted_images:
                        self.log(f"Skipping deleted image: {file_path.name}")
                        continue

                    seen_files.add(file_key)
                    image_files.append(file_path)

        return sorted(image_files)

    def copy_with_unique_name(self, source_path, target_dir, filename=None):
        """Copy file to target directory with unique name if conflicts exist"""
        if filename is None:
            filename = source_path.name

        target_path = target_dir / filename
        counter = 1

        while target_path.exists():
            name, ext = Path(filename).stem, Path(filename).suffix
            target_path = target_dir / f"{name}_{counter}{ext}"
            counter += 1

        shutil.copy2(source_path, target_path)
        return target_path

    def process_character_async(self, char_name):
        """Process character in a separate thread"""
        threading.Thread(target=self.process_character, args=(char_name,), daemon=True).start()

    @abstractmethod
    def process_character(self, char_name):
        """Process character - to be implemented by subclasses"""
        pass

    @abstractmethod
    def get_source_stage(self):
        """Get the source stage name"""
        pass

    @abstractmethod
    def get_target_stage(self):
        """Get the target stage name"""
        pass

    def delete_images_in_target_stage_folder(self, folder_path, stage):
        """Delete all images in a character's target stage folder"""
        if not folder_path.exists():
            return False, "Stage folder does not exist"

        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif', '*.tiff', '*.webp']
        deleted_files = 0

        for ext in extensions:
            for img_file in folder_path.glob(ext):
                try:
                    img_file.unlink()
                    deleted_files += 1
                except Exception as e:
                    self.log(f"Error deleting file {img_file.name}: {e}")

        return True, f"Deleted {deleted_files} files from {stage}"

    def _load_deleted_images(self, character_name: str) -> set:
        """Load deleted images for a specific character."""
        deleted_file = self.characters_path / character_name / "deleted.yaml"
        if deleted_file.exists():
            try:
                with open(deleted_file, 'r', encoding='utf-8') as file:
                    deleted_data = yaml.safe_load(file) or {}
                    return set(deleted_data.get('deleted_images', []))
            except Exception as e:
                self.log(f"Error loading deleted images for {character_name}: {e}")
        return set()

    def _is_image_deleted(self, character_name: str, filename: str) -> bool:
        """Check if an image filename is in the deleted list for a character."""
        deleted_images = self._load_deleted_images(character_name)
        return filename in deleted_images

    def physically_delete_marked_images(self, character_name: str, source_dir: Path) -> int:
        """Physically delete images from source directory that are marked as deleted in deleted.yaml"""
        if not source_dir.exists():
            return 0

        deleted_images = self._load_deleted_images(character_name)
        if not deleted_images:
            return 0

        # Create a set of deleted stems (filenames without extensions) for faster lookup
        deleted_stems = {Path(filename).stem.lower() for filename in deleted_images}

        deleted_count = 0

        # Get all image files in the source directory
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp']

        for file_path in source_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                # Check if the file stem matches any deleted stem (case-insensitive)
                if file_path.stem.lower() in deleted_stems:
                    try:
                        file_path.unlink()
                        deleted_count += 1
                        self.log(f"Physically deleted marked image: {file_path.name}")
                    except Exception as e:
                        self.log(f"Error deleting marked image {file_path.name}: {e}")

        if deleted_count > 0:
            self.log(f"Physically deleted {deleted_count} images marked as deleted from {source_dir}")

        return deleted_count

    def _normalize_target_filenames(self, char_name, source_dir, target_dir):
        """Normalize target filenames by removing any prefixes and suffixes"""
        try:
            if not target_dir.exists():
                return

            # Get list of source filenames (without extension) for matching
            source_files = self.get_image_files(source_dir, char_name) if source_dir.exists() else []

            # Build lookup dictionaries for O(1) access
            source_stems_lower = {source_file.stem.lower() for source_file in source_files}
            # Map lowercase stem to original case stem
            source_stem_case_map = {sf.stem.lower(): sf.stem for sf in source_files}

            renamed_count = 0

            # Get all files in target directory
            for file_path in target_dir.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.gif',
                                                                        '.tiff', '.webp']:
                    current_stem = file_path.stem
                    current_stem_lower = current_stem.lower()

                    # Skip if already matches a source filename
                    if current_stem_lower in source_stems_lower:
                        continue

                    # Try to find matching source filename by removing prefix and/or suffix
                    best_match = None

                    # Method 1: Try removing prefix (everything up to first underscore)
                    if '_' in current_stem:
                        no_prefix = current_stem.split('_', 1)[1]
                        if no_prefix.lower() in source_stems_lower:
                            best_match = no_prefix

                    # Method 2: Try removing suffix (everything after last underscore)
                    if not best_match and '_' in current_stem:
                        no_suffix = current_stem.rsplit('_', 1)[0]
                        if no_suffix.lower() in source_stems_lower:
                            best_match = no_suffix

                    # Method 3: Try removing both prefix and suffix
                    if not best_match and current_stem.count('_') >= 2:
                        parts = current_stem.split('_')
                        # Try middle part(s) - remove first and last part
                        middle_part = '_'.join(parts[1:-1])
                        if middle_part and middle_part.lower() in source_stems_lower:
                            best_match = middle_part

                    # Method 4: Try finding source filename as substring (optimized)
                    if not best_match:
                        for source_stem_lower, source_stem_original in source_stem_case_map.items():
                            if source_stem_lower in current_stem_lower:
                                best_match = source_stem_original
                                break

                    # Rename if we found a match
                    if best_match:
                        # Get the original case from the lookup map
                        original_case_match = source_stem_case_map.get(best_match.lower(), best_match)
                        new_name = original_case_match + file_path.suffix
                        new_path = file_path.parent / new_name

                        # Rename file if new name is different and doesn't already exist
                        if new_name != file_path.name and not new_path.exists():
                            file_path.rename(new_path)
                            renamed_count += 1
                            self.log(f"Normalized target filename: {file_path.name} -> {new_name}")

            if renamed_count > 0:
                self.log(f"Normalized {renamed_count} target filenames to match source files")
            else:
                self.log("No target filenames needed normalization")

        except Exception as e:
            self.log(f"Error normalizing target filenames: {e}")
