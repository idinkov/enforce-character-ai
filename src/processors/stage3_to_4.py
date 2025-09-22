"""
Stage 3 to 4: Simple 1024x1024 Processor
Resizes images to 1024x1024 without face detection
"""
import cv2
import shutil
from PIL import Image
from .base_processor import BaseProcessor


class SimpleResizeProcessor(BaseProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # No face detector needed

    def get_source_stage(self):
        return "3_raw_upscaled"

    def get_target_stage(self):
        return "4_processed_1024"

    def process_character(self, char_name):
        """Process images to 1024x1024 with simple resize"""
        try:
            self.log(f"Processing images to 1024x1024 for {char_name}...")

            # Find source directory (prefer upscaled, then filtered, then raw)
            char_path = self.characters_path / char_name / "images"
            source_dir = None

            for stage in ["3_raw_upscaled", "2_raw_filtered", "1_raw"]:
                potential_source = char_path / stage
                if potential_source.exists():
                    image_files = self.get_image_files(potential_source, char_name)
                    if image_files:
                        source_dir = potential_source
                        self.log(f"Using source directory: {stage}")
                        break

            if not source_dir:
                self.log("Error: No source images found")
                return False

            # Physically delete images marked as deleted in deleted.yaml
            deleted_count = self.physically_delete_marked_images(char_name, source_dir)
            if deleted_count > 0:
                self.log(f"Physically deleted {deleted_count} images marked as deleted")

            target_dir = char_path / self.get_target_stage()
            target_dir.mkdir(parents=True, exist_ok=True)

            # Normalize any existing target filenames to remove prefixes/suffixes
            self._normalize_target_filenames(char_name, source_dir, target_dir)

            image_files = self.get_image_files(source_dir, char_name)

            total_files = len(image_files)
            processed = 0
            successful = 0
            skipped = 0

            self.update_progress(0)

            for img_file in image_files:
                try:
                    # Check if file already exists in target directory
                    output_path = target_dir / img_file.name
                    if output_path.exists():
                        self.log(f"Skipping {img_file.name} - already exists in target directory")
                        skipped += 1
                        processed += 1
                        progress = (processed / total_files) * 100
                        self.update_progress(progress)
                        continue

                    # Check if image is already 1024x1024
                    try:
                        # Use PIL to quickly check dimensions without loading full image
                        with Image.open(img_file) as pil_img:
                            if pil_img.size == (1024, 1024):
                                # Image is already 1024x1024, just copy it
                                shutil.copy2(img_file, output_path)
                                successful += 1
                                processed += 1
                                progress = (processed / total_files) * 100
                                self.update_progress(progress)
                                self.log(f"Copied (already 1024x1024): {img_file.name}")
                                continue
                    except Exception as e:
                        self.log(f"Error checking image dimensions for {img_file.name}: {e}")
                        # Fall through to normal processing if dimension check fails

                    # Load image
                    img = cv2.imread(str(img_file))
                    if img is None:
                        self.log(f"Could not load image: {img_file.name}")
                        processed += 1
                        continue

                    # Process image with simple resize
                    processed_img = self._resize_image(img)

                    if processed_img is not None:
                        # Save with original filename
                        cv2.imwrite(str(output_path), processed_img)
                        successful += 1
                        self.log(f"Processed and resized: {img_file.name}")

                    processed += 1
                    progress = (processed / total_files) * 100
                    self.update_progress(progress)

                except Exception as e:
                    self.log(f"Error processing {img_file.name}: {e}")
                    processed += 1

            self.log(f"Processing complete! Successfully processed {successful}, skipped {skipped} existing files out of {processed} images")
            self.update_progress(100)
            return True

        except Exception as e:
            self.log(f"Error during processing: {e}")
            return False

    def _resize_image(self, img):
        """Simple resize image to 1024x1024"""
        try:
            # Simply resize to 1024x1024
            result = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_LANCZOS4)
            return result

        except Exception as e:
            self.log(f"Error in resize processing: {e}")
            return None
