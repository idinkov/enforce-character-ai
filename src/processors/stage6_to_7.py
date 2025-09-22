"""
Stage 6 to 7: Final Processor
Creates the final training-ready dataset with metadata and validation
"""
import os
import shutil
from .base_processor import BaseProcessor


class FinalProcessor(BaseProcessor):
    def get_source_stage(self):
        return "6_rtt_1024"

    def get_target_stage(self):
        return "7_final_dataset"

    def process_character(self, char_name):
        """Create final training dataset with metadata"""
        try:
            self.log(f"Starting final dataset creation for character: {char_name}")

            # Get character path
            char_path = self.characters_path / char_name
            if not char_path.exists():
                self.log(f"Character path does not exist: {char_path}")
                return False

            # Get source and target directories
            source_dir = char_path / "images" / self.get_source_stage()
            target_dir = char_path / "images" / self.get_target_stage()

            if not source_dir.exists():
                self.log(f"Source directory does not exist: {source_dir}")
                return False

            # Look for the 1_person folder in the source directory
            source_1_person_dir = source_dir / "1_person"
            if not source_1_person_dir.exists():
                self.log(f"1_person folder does not exist in source: {source_1_person_dir}")
                return False

            # Physically delete images marked as deleted in deleted.yaml
            deleted_count = self.physically_delete_marked_images(char_name, source_1_person_dir)
            if deleted_count > 0:
                self.log(f"Physically deleted {deleted_count} images marked as deleted")

            # Create target directory if it doesn't exist
            target_dir.mkdir(parents=True, exist_ok=True)

            # Normalize any existing target filenames to remove prefixes/suffixes
            self._normalize_target_filenames(char_name, source_1_person_dir, target_dir)

            # Get all image files from the 1_person folder
            image_files = self.get_image_files(source_1_person_dir, char_name)
            if not image_files:
                self.log(f"No image files found in {source_1_person_dir}")
                return True

            self.log(f"Found {len(image_files)} images to copy from 1_person folder")

            # Move each image to the target directory
            copied_count = 0
            skipped_count = 0
            for image_file in image_files:
                try:
                    # Check if file already exists in target directory
                    potential_target_path = target_dir / image_file.name
                    if potential_target_path.exists():
                        # Delete file
                        os.unlink(image_file)
                        self.log(f"Deleting {image_file.name} - already exists in target directory")
                        skipped_count += 1
                    else:
                        # Move the image to target directory
                        target_path = shutil.move(image_file, target_dir)
                        copied_count += 1
                        self.log(f"Moved: {image_file.name} -> {target_path.name}")

                    # Update progress
                    progress = ((copied_count + skipped_count) / len(image_files)) * 100
                    self.update_progress(progress)

                except Exception as e:
                    self.log(f"Error moving {image_file.name}: {e}")

            self.log(f"Successfully moved {copied_count} images, skipped {skipped_count} existing files to final dataset for {char_name}")
            return True

        except Exception as e:
            self.log(f"Error processing character {char_name}: {e}")
            return False
