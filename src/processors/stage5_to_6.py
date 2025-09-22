"""
Stage 5 to 6: Count & Sort Processor
Here we are going to sort in folders {num_people}/{photo} by using face detection to count the faces
"""

from .base_processor import BaseProcessor
from ..detections.face_detection import FaceDetection
from ..services.gpu_service import get_gpu_service

class QualityControlProcessor(BaseProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gpu_service = get_gpu_service()
        self.face_detector = None

    def reinitialize_with_selected_gpu(self):
        """Reinitialize face detector with currently selected GPU."""
        # Face detector will be reinitialized when needed in process_character
        self.face_detector = None

    def get_source_stage(self):
        return "5_processed_fixed_1024"

    def get_target_stage(self):
        return "6_rtt_1024"

    def process_character(self, char_name):
        """
        Process a character by detecting faces in stage 5 images and sorting them
        into folders based on face count: {num_people}/{photo}
        """
        try:
            self.log(f"Starting face detection and sorting for character: {char_name}")

            # Get character path
            char_path = self.characters_path / char_name
            if not char_path.exists():
                self.log(f"Character path does not exist: {char_path}")
                return False

            # Initialize face detection for this character
            face_detector = FaceDetection(char_path, log_callback=self.log)

            # Get source and target directories
            source_dir = char_path / "images" / self.get_source_stage()
            target_dir = char_path / "images" / self.get_target_stage()

            if not source_dir.exists():
                self.log(f"Source directory does not exist: {source_dir}")
                return False

            # Physically delete images marked as deleted in deleted.yaml
            deleted_count = self.physically_delete_marked_images(char_name, source_dir)
            if deleted_count > 0:
                self.log(f"Physically deleted {deleted_count} images marked as deleted")

            # Create target directory if it doesn't exist
            target_dir.mkdir(parents=True, exist_ok=True)

            # Normalize any existing target filenames to remove prefixes/suffixes
            self._normalize_target_filenames(char_name, source_dir, target_dir)

            # Get all image files from source
            source_image_files = self.get_image_files(source_dir, char_name)
            if not source_image_files:
                self.log(f"No image files found in {source_dir}")
                return True

            # Preload existing target filenames to avoid repeated file existence checks
            self.log("Preloading existing target filenames...")
            existing_target_files = set()

            if target_dir.exists():
                # Recursively scan all subdirectories in target to get existing files
                for subdir in target_dir.iterdir():
                    if subdir.is_dir():
                        for file_path in subdir.iterdir():
                            if file_path.is_file() and file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp']:
                                existing_target_files.add(file_path.name)

            # Create set of source filenames for efficient comparison
            source_filenames = {img_file.name for img_file in source_image_files}

            # Calculate files that need to be processed (source files not in target)
            files_to_process_names = source_filenames - existing_target_files

            # Filter source image files to only include those that need processing
            files_to_process = [img_file for img_file in source_image_files if img_file.name in files_to_process_names]

            skipped_count = len(source_image_files) - len(files_to_process)

            self.log(f"Found {len(source_image_files)} total images in source")
            self.log(f"Found {len(existing_target_files)} existing images in target")
            self.log(f"Need to process {len(files_to_process)} new images")
            self.log(f"Skipping {skipped_count} already processed images")

            if not files_to_process:
                self.log("All images have already been processed!")
                return True

            # Process each image that needs processing
            processed_count = 0
            face_count_stats = {}

            for i, image_file in enumerate(files_to_process):
                try:
                    self.log(f"Processing image {i+1}/{len(files_to_process)}: {image_file.name}")

                    # Detect faces in the image
                    face_boxes = face_detector.detect_faces_in_image(image_file)
                    num_faces = len(face_boxes)

                    # Create target folder based on number of faces
                    # Use "0_people", "1_person", "2_people", etc.
                    if num_faces == 0:
                        face_folder = "0_people"
                    elif num_faces == 1:
                        face_folder = "1_person"
                    else:
                        face_folder = f"{num_faces}_people"

                    face_target_dir = target_dir / face_folder
                    face_target_dir.mkdir(parents=True, exist_ok=True)

                    # Copy image to the appropriate folder (no need to check existence since we pre-filtered)
                    target_path = self.copy_with_unique_name(image_file, face_target_dir)
                    processed_count += 1
                    self.log(f"  → Found {num_faces} face(s), copied to {face_folder}/")

                    # Update statistics
                    if num_faces not in face_count_stats:
                        face_count_stats[num_faces] = 0
                    face_count_stats[num_faces] += 1

                    # Update progress
                    progress = (i + 1) / len(files_to_process) * 100
                    self.update_progress(progress)

                except Exception as e:
                    self.log(f"Error processing {image_file.name}: {str(e)}")
                    continue

            # Log summary statistics
            self.log(f"\n=== Face Detection Summary for {char_name} ===")
            self.log(f"Total images processed: {processed_count}")
            self.log(f"Total images skipped (already exist): {skipped_count}")

            for face_count in sorted(face_count_stats.keys()):
                count = face_count_stats[face_count]
                if face_count == 0:
                    folder_name = "0_people"
                elif face_count == 1:
                    folder_name = "1_person"
                else:
                    folder_name = f"{face_count}_people"

                self.log(f"  {folder_name}: {count} images")

            self.log(f"Images sorted into: {target_dir}")
            self.log(f"Character {char_name} processing completed successfully!")

            return True

        except Exception as e:
            self.log(f"Error processing character {char_name}: {str(e)}")
            return False
