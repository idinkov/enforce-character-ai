"""
Stage 0 to 1: Provider Import Processor
Copies images from provider import folders to raw images stage
"""
from .base_processor import BaseProcessor
from ..config.app_config import PROVIDER_IMPORTS_DIR


class ProviderImportProcessor(BaseProcessor):
    def __init__(self, characters_path, log_callback=None, progress_callback=None, provider_manager=None):
        """Initialize the provider import processor"""
        super().__init__(characters_path, log_callback, progress_callback, provider_manager)

    def get_source_stage(self):
        return PROVIDER_IMPORTS_DIR

    def get_target_stage(self):
        return "1_raw"

    def check_all_providers(self, char_name):
        """Check all providers for new content before processing"""
        if not self.provider_manager:
            self.log("Warning: Provider manager not available. Skipping provider checks.")
            return 0

        try:
            self.log(f"Checking all providers for {char_name}...")

            # Construct full character directory path
            character_dir = str(self.characters_path / char_name)
            providers = self.provider_manager.get_character_providers(character_dir)

            if not providers:
                self.log("No providers configured for this character")
                return 0

            total_downloaded = 0
            provider_count = len(providers)

            self.log(f"Found {provider_count} provider(s) to check")

            for i, provider_config in enumerate(providers):
                provider_id = provider_config.get('id')
                provider_type = provider_config.get('type', 'unknown')

                if provider_id:
                    # Update progress for current provider
                    if self.progress_callback:
                        self.progress_callback(i, provider_count, f"Checking {provider_type}")

                    self.log(f"Checking provider {i+1}/{provider_count}: {provider_type} (ID: {provider_id})")

                    try:
                        downloaded_count = self.provider_manager.check_provider_now(
                            character_dir, provider_id,
                            progress_callback=self.progress_callback,
                            log_callback=self.log
                        )

                        if downloaded_count >= 0:
                            total_downloaded += downloaded_count
                            if downloaded_count > 0:
                                self.log(f"✓ Downloaded {downloaded_count} new files from {provider_type}")
                            else:
                                self.log(f"✓ No new files found from {provider_type}")
                        else:
                            self.log(f"✗ Failed to check provider: {provider_type}")
                    except Exception as e:
                        self.log(f"✗ Error checking provider {provider_type}: {e}")
                        continue

            # Update progress to complete
            if self.progress_callback:
                self.progress_callback(provider_count, provider_count, "Provider checks complete")

            self.log(f"✓ Provider checking completed! Total new files downloaded: {total_downloaded}")
            return total_downloaded

        except Exception as e:
            self.log(f"✗ Error during provider checking: {e}")
            return 0

    def process_character(self, char_name):
        """Copy images from provider import folders to raw images stage"""
        try:
            self.log(f"Starting image copy from provider imports to raw stage...")

            source_dir = self.characters_path / char_name / "images" / self.get_source_stage()
            target_dir = self.characters_path / char_name / "images" / self.get_target_stage()

            if not source_dir.exists():
                self.log("Error: Source directory (provider imports) does not exist")
                return False

            # Physically delete images marked as deleted in deleted.yaml
            deleted_count = self.physically_delete_marked_images(char_name, source_dir)
            if deleted_count > 0:
                self.log(f"Physically deleted {deleted_count} images marked as deleted")

            # Create target directory if it doesn't exist
            target_dir.mkdir(parents=True, exist_ok=True)

            # Get all existing files in target directory to avoid duplicates
            existing_files = set()
            if target_dir.exists():
                for file_path in target_dir.iterdir():
                    if file_path.is_file():
                        existing_files.add(file_path.name.lower())

            # Process all provider folders within the source directory
            provider_folders = [d for d in source_dir.iterdir() if d.is_dir()]

            if not provider_folders:
                self.log("No provider folders found in source directory")
                return True

            total_files_to_copy = 0
            files_copied = 0
            files_skipped = 0

            # First pass: count total files to copy
            for provider_folder in provider_folders:
                image_files = self.get_image_files(provider_folder, char_name)
                for image_file in image_files:
                    if image_file.name.lower() not in existing_files:
                        total_files_to_copy += 1

            if total_files_to_copy == 0:
                self.log("All files have already been copied. No new files to process.")
                return True

            self.log(f"Found {total_files_to_copy} new files to copy from {len(provider_folders)} provider folders")

            # Second pass: copy files
            for provider_idx, provider_folder in enumerate(provider_folders):
                provider_name = provider_folder.name
                self.log(f"Processing provider folder: {provider_name}")

                image_files = self.get_image_files(provider_folder, char_name)

                for file_idx, image_file in enumerate(image_files):
                    # Check if file already exists in target (case-insensitive)
                    if image_file.name.lower() in existing_files:
                        files_skipped += 1
                        continue

                    try:
                        # Generate unique filename if needed
                        target_path = self.copy_with_unique_name(image_file, target_dir)

                        # Add to existing files set to prevent duplicates within this run
                        existing_files.add(target_path.name.lower())

                        files_copied += 1

                        # Update progress
                        progress = (files_copied / total_files_to_copy) * 100
                        self.update_progress(progress)

                        if files_copied % 10 == 0:  # Log every 10 files
                            self.log(f"Copied {files_copied}/{total_files_to_copy} files...")

                    except Exception as e:
                        self.log(f"Error copying {image_file.name}: {e}")
                        continue

            self.log(f"Provider import completed!")
            self.log(f"Files copied: {files_copied}")
            self.log(f"Files skipped (already exist): {files_skipped}")
            self.log(f"Total files in target: {len(list(target_dir.iterdir()))}")

            self.update_progress(100)
            return True

        except Exception as e:
            self.log(f"Error during provider import: {e}")
            return False

    def get_provider_folders(self, char_name):
        """Get list of provider folders for a character"""
        source_dir = self.characters_path / char_name / "images" / self.get_source_stage()
        if not source_dir.exists():
            return []

        return [d.name for d in source_dir.iterdir() if d.is_dir()]

    def get_provider_file_count(self, char_name, provider_name=None):
        """Get count of files in provider folders"""
        source_dir = self.characters_path / char_name / "images" / self.get_source_stage()
        if not source_dir.exists():
            return 0

        total_count = 0

        if provider_name:
            # Count files in specific provider folder
            provider_dir = source_dir / provider_name
            if provider_dir.exists():
                image_files = self.get_image_files(provider_dir, char_name)
                total_count = len(image_files)
        else:
            # Count files in all provider folders
            for provider_folder in source_dir.iterdir():
                if provider_folder.is_dir():
                    image_files = self.get_image_files(provider_folder, char_name)
                    total_count += len(image_files)

        return total_count

    def clear_provider_imports(self, char_name):
        """Clear all files from provider imports stage"""
        source_dir = self.characters_path / char_name / "images" / self.get_source_stage()
        if not source_dir.exists():
            return False, "Provider imports directory does not exist"

        try:
            deleted_count = 0
            for provider_folder in source_dir.iterdir():
                if provider_folder.is_dir():
                    # Delete all files in the provider folder but keep the folder
                    for file_path in provider_folder.iterdir():
                        if file_path.is_file():
                            file_path.unlink()
                            deleted_count += 1

            return True, f"Cleared {deleted_count} files from provider imports"
        except Exception as e:
            return False, f"Error clearing provider imports: {e}"
