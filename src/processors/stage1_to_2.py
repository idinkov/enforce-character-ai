"""
Stage 1 to 2: Duplicate Filter Processor
Removes duplicate and invalid images from raw images and converts all to PNG format
"""
import hashlib
import threading
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from PIL import Image
import cv2
from .base_processor import BaseProcessor
from ..detections.face_detection import FaceDetection
from ..detections.border_detection import BorderDetection
from ..detections.background_removal import BackgroundRemoval

# Constants
MAX_CONVERSION_THREADS = 32
MAX_BORDERS_THREADS = 16
MAX_FACE_DETECTION_THREADS = 1
MAX_REMBG_THREADS = 4
AUTOFILTER_0_FACES = True
AUTOFIX_BORDERS = True
AUTO_REMOVE_BACKGROUND = True  # New constant for background removal

class DuplicateFilterProcessor(BaseProcessor):
    def __init__(self, characters_path, log_callback=None, progress_callback=None, provider_manager=None):
        """Initialize the processor with face detection capability"""
        super().__init__(characters_path, log_callback, progress_callback, provider_manager)
        self.face_detector = None
        self._conversion_lock = threading.Lock()
        self._conversion_progress = 0
        self._border_detector = None  # Border detection module
        self._background_remover = None  # Background removal module

    def get_source_stage(self):
        return "1_raw"

    def get_target_stage(self):
        return "2_raw_filtered"

    def _get_processed_yaml_path(self, char_name):
        """Get the path to the processed.yaml file for a character"""
        return self.characters_path / char_name / "processed.yaml"

    def _load_processed_data(self, char_name):
        """Load processed data from processed.yaml"""
        processed_yaml_path = self._get_processed_yaml_path(char_name)
        if processed_yaml_path.exists():
            try:
                with open(processed_yaml_path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f) or {}
                self.log(f"Loaded processed.yaml with {len(data.get('processed_images', {}))} entries")
                return data
            except Exception as e:
                self.log(f"Error loading processed.yaml: {e}")
                return {}
        else:
            self.log(f"processed.yaml does not exist at {processed_yaml_path}")
        return {}

    def _save_processed_data(self, char_name, processed_data):
        """Save processed data to processed.yaml"""
        processed_yaml_path = self._get_processed_yaml_path(char_name)
        try:
            # Ensure the directory exists
            processed_yaml_path.parent.mkdir(parents=True, exist_ok=True)

            with open(processed_yaml_path, 'w', encoding='utf-8') as f:
                yaml.dump(processed_data, f, default_flow_style=False, allow_unicode=True)

            # Verify the file was written correctly
            if processed_yaml_path.exists():
                self.log(f"Successfully saved processed.yaml with {len(processed_data.get('processed_images', {}))} entries")
            else:
                self.log(f"ERROR: processed.yaml was not created at {processed_yaml_path}")
        except Exception as e:
            self.log(f"Error saving processed.yaml: {e}")

    def _collect_image_for_processing_mark(self, char_name, source_file_name, target_stage, processed_images_batch):
        """Collect an image to be marked as processed in batch (thread-safe)"""
        with self._conversion_lock:
            if char_name not in processed_images_batch:
                processed_images_batch[char_name] = {}

            processed_images_batch[char_name][source_file_name] = {
                'stage': target_stage,
                'processed': True
            }

    def _batch_mark_images_as_processed(self, processed_images_batch):
        """Batch mark multiple images as processed in their respective processed.yaml files"""
        for char_name, images_data in processed_images_batch.items():
            if not images_data:
                continue

            with self._conversion_lock:
                processed_data = self._load_processed_data(char_name)

                if 'processed_images' not in processed_data:
                    processed_data['processed_images'] = {}

                # Add all images for this character
                processed_data['processed_images'].update(images_data)

                self._save_processed_data(char_name, processed_data)
                self.log(f"Batch marked {len(images_data)} images as processed for {char_name}")

    def _is_image_processed(self, char_name, source_file_name, target_stage, processed_data):
        """Check if an image has been processed for the target stage"""
        if 'processed_images' not in processed_data:
            return False

        image_data = processed_data['processed_images'].get(source_file_name)
        if image_data and image_data.get('stage') == target_stage and image_data.get('processed'):
            self.log(f"Image {source_file_name} already processed for stage {target_stage}")
            return True

        return False

    def process_character(self, char_name):
        """Filter duplicate images from stage 1 to stage 2 and convert to PNG"""
        try:
            self.log(f"Starting duplicate filtering and PNG conversion for {char_name}...")

            source_dir = self.characters_path / char_name / "images" / self.get_source_stage()
            target_dir = self.characters_path / char_name / "images" / self.get_target_stage()

            if not source_dir.exists():
                self.log("Error: Source directory does not exist")
                return False

            # Physically delete images marked as deleted in deleted.yaml
            deleted_count = self.physically_delete_marked_images(char_name, source_dir)
            if deleted_count > 0:
                self.log(f"Physically deleted {deleted_count} images marked as deleted")

            # Create target directory if it doesn't exist
            target_dir.mkdir(parents=True, exist_ok=True)

            # Get all images from source directory, excluding deleted ones
            source_image_files = self.get_image_files(source_dir, char_name)

            if not source_image_files:
                self.log("No images found in source directory")
                return False

            # Count images already in target directory
            existing_target_count = 0
            if target_dir.exists():
                existing_target_count = len(list(target_dir.glob("*.png")))

            # Filter source images to only include those not already processed according to processed.yaml
            target_stage = self.get_target_stage()
            images_to_process = []
            already_processed_count = 0

            processed_data = self._load_processed_data(char_name)

            for img_file in source_image_files:
                # Load processed images
                if self._is_image_processed(char_name, img_file.name, target_stage, processed_data):
                    already_processed_count += 1
                else:
                    images_to_process.append(img_file)

            # Log comprehensive processing summary
            total_source_images = len(source_image_files)
            self.log(f"=== Processing Summary for {char_name} ===")
            self.log(f"Source directory ({self.get_source_stage()}) contains: {total_source_images} images")
            self.log(f"Target directory ({target_stage}) already contains: {existing_target_count} PNG files")
            self.log(f"Images marked as processed in processed.yaml: {already_processed_count}")
            self.log(f"New images to process: {len(images_to_process)}")

            # Skip processing if no new images to process
            if not images_to_process:
                self.log("No new images to process - skipping stage")
                self.update_progress(100)
                return True

            # Initialize face detector for this character
            character_path = self.characters_path / char_name
            self.face_detector = FaceDetection(character_path, self.log)

            self.update_progress(0)

            # Execute all phases sequentially on only the new images
            converted_files = self._phase1_duplicate_filter_and_convert(images_to_process, target_dir, char_name)
            if converted_files is None:
                return False

            if AUTOFIX_BORDERS:
                self._phase2_border_detection_and_cropping(converted_files)

            face_detection_results = self._phase3_face_detection(converted_files, char_name)

            # Initialize deletion tracking dictionaries
            zero_face_candidates = set()  # Files that have 0 faces
            bg_removal_failed_candidates = set()  # Files where background removal failed

            # Phase 4: Collect zero face candidates (but don't delete yet)
            if AUTOFILTER_0_FACES and face_detection_results:
                zero_face_candidates = self._phase4_collect_zero_face_candidates(face_detection_results, char_name, target_dir)

            # Phase 5: Process background removal and collect failed candidates
            if AUTO_REMOVE_BACKGROUND:
                bg_removal_failed_candidates = self._phase5_background_removal_with_candidates(converted_files, face_detection_results)

            # Phase 6: Make final deletion decisions based on flag combinations
            self._phase6_process_deletions(zero_face_candidates, bg_removal_failed_candidates, target_dir)

            self.log(f"Processing complete! Processed {len(converted_files)} unique images to PNG with face detection out of {len(images_to_process)}")
            self.update_progress(100)
            return True

        except Exception as e:
            self.log(f"Error during filtering: {e}")
            return False

    def _phase1_duplicate_filter_and_convert(self, image_files, target_dir, char_name):
        """Phase 1: Filter duplicates and convert to PNG with threading"""
        try:
            total_files = len(image_files)
            processed = 0
            seen_hashes = set()
            copied_files = 0
            converted_files = []
            skipped_existing = 0
            target_stage = self.get_target_stage()

            # Initialize batch processing collection
            processed_images_batch = {}

            self.log(f"Phase 1: Filtering duplicates and converting to PNG using {MAX_CONVERSION_THREADS} threads...")

            # First pass: validate images and collect unique files for conversion
            unique_conversion_tasks = []
            for img_file in image_files:
                try:
                    # Check if PNG version already exists in target directory
                    base_name = img_file.stem
                    png_filename = f"{base_name}.png"
                    target_path = target_dir / png_filename

                    if target_path.exists():
                        self.log(f"Skipping {img_file.name} - already exists as {png_filename}")
                        # Collect for batch marking as processed since it exists
                        self._collect_image_for_processing_mark(char_name, img_file.name, target_stage, processed_images_batch)
                        skipped_existing += 1
                        processed += 1
                        # Progress for validation phase (0-25%)
                        progress = (processed / total_files) * 25
                        self.update_progress(progress)
                        continue

                    # Validate image by trying to open it
                    try:
                        with Image.open(img_file) as img:
                            # Check if image is valid and has reasonable dimensions
                            if img.size[0] < 64 or img.size[1] < 64:
                                self.log(f"Skipping too small image: {img_file.name}")
                                processed += 1
                                continue

                            # Convert to RGB if necessary (handles RGBA, palette, etc.)
                            if img.mode not in ('RGB', 'L'):
                                img = img.convert('RGB')

                    except Exception as e:
                        self.log(f"Skipping invalid image {img_file.name}: {e}")
                        processed += 1
                        continue

                    # Create hash based on file content
                    file_hash = self._calculate_file_hash(img_file)

                    if file_hash not in seen_hashes:
                        seen_hashes.add(file_hash)
                        unique_conversion_tasks.append(img_file)
                        copied_files += 1

                    processed += 1
                    # Progress for validation phase (0-25%)
                    progress = (processed / total_files) * 25
                    self.update_progress(progress)

                except Exception as e:
                    self.log(f"Error processing {img_file}: {e}")
                    processed += 1

            self.log(f"Validation complete! Found {len(unique_conversion_tasks)} unique images to convert, skipped {skipped_existing} existing files")

            # Second pass: convert unique images using thread pool
            if unique_conversion_tasks:
                self.log(f"Converting {len(unique_conversion_tasks)} images using {MAX_CONVERSION_THREADS} threads...")
                self._conversion_progress = 0

                with ThreadPoolExecutor(max_workers=MAX_CONVERSION_THREADS) as executor:
                    # Submit all conversion tasks
                    future_to_file = {
                        executor.submit(self._convert_and_save_as_png_batch, img_file, target_dir, char_name, target_stage, processed_images_batch): img_file
                        for img_file in unique_conversion_tasks
                    }

                    # Process completed conversions
                    for future in as_completed(future_to_file):
                        img_file = future_to_file[future]
                        try:
                            target_path = future.result()

                            # Add to list for face detection if conversion successful
                            if target_path:
                                converted_files.append(target_path)

                            # Update progress thread-safely
                            with self._conversion_lock:
                                self._conversion_progress += 1
                                # Progress from 25% to 50%
                                progress = 25 + (self._conversion_progress / len(unique_conversion_tasks)) * 25
                                self.update_progress(progress)

                        except Exception as e:
                            self.log(f"Error converting {img_file.name}: {e}")
                            with self._conversion_lock:
                                self._conversion_progress += 1

            # Batch mark all processed images at once
            if processed_images_batch:
                self._batch_mark_images_as_processed(processed_images_batch)

            self.log(f"Phase 1 complete! Filtered and converted {len(converted_files)} unique images to PNG out of {total_files} (skipped {skipped_existing} existing files)")
            return converted_files

        except Exception as e:
            self.log(f"Error in Phase 1: {e}")
            return None

    def _phase3_face_detection(self, converted_files, char_name):
        """Phase 3: Face detection on converted files"""
        try:
            face_detection_results = {}

            if not converted_files:
                self.log("No images were converted, skipping face detection phase")
                return face_detection_results

            # Load existing face detection results from detections_face.yaml
            existing_face_detection_results = self._load_existing_face_detection_results(char_name)

            # Separate files into those that need face detection and those that already have results
            files_needing_detection = []
            files_with_existing_results = []

            for converted_file in converted_files:
                relative_path = str(converted_file.relative_to(self.characters_path / char_name))

                if relative_path in existing_face_detection_results:
                    # Use existing results
                    face_detection_results[relative_path] = existing_face_detection_results[relative_path]
                    files_with_existing_results.append(converted_file)
                    self.log(f"Using existing face detection results for {converted_file.name}")
                else:
                    # Needs new face detection
                    files_needing_detection.append(converted_file)

            self.log(f"Phase 3: Found {len(files_with_existing_results)} images with existing face detection results")
            self.log(f"Phase 3: Performing face detection on {len(files_needing_detection)} new images using {MAX_FACE_DETECTION_THREADS} threads...")

            face_detection_processed = len(files_with_existing_results)  # Start count with existing results

            # Only process files that need new face detection
            if files_needing_detection:
                # Thread-safe lock for updating results
                results_lock = threading.Lock()

                def process_face_detection(target_path):
                    """Helper function for threaded face detection"""
                    try:
                        # Perform face detection and collect results
                        relative_path = str(target_path.relative_to(self.characters_path / char_name))
                        face_boxes = self._perform_face_detection_only(target_path)

                        # Store results in memory thread-safely
                        with results_lock:
                            face_detection_results[relative_path] = {
                                'faces': face_boxes,
                                'face_count': len(face_boxes)
                            }

                        if face_boxes:
                            self.log(f"Found {len(face_boxes)} face(s) in {target_path.name}")
                        else:
                            self.log(f"No faces detected in {target_path.name}")

                        return True
                    except Exception as e:
                        self.log(f"Error during face detection for {target_path.name}: {e}")
                        return False

                # Use ThreadPoolExecutor for multithreaded face detection
                with ThreadPoolExecutor(max_workers=MAX_FACE_DETECTION_THREADS) as executor:
                    # Submit face detection tasks for files needing detection
                    future_to_file = {
                        executor.submit(process_face_detection, img_file): img_file
                        for img_file in files_needing_detection
                    }

                    # Process completed face detection results
                    for future in as_completed(future_to_file):
                        img_file = future_to_file[future]
                        try:
                            future.result()  # Wait for completion
                            face_detection_processed += 1

                            # Progress from 50% to 70%
                            progress = 50 + (face_detection_processed / len(converted_files)) * 20
                            self.update_progress(progress)

                        except Exception as e:
                            self.log(f"Error during face detection for {img_file.name}: {e}")
                            face_detection_processed += 1

                # Write only NEW face detection results to detections_face.yaml
                if files_needing_detection:
                    new_results = {k: v for k, v in face_detection_results.items()
                                 if str(self.characters_path / char_name / k) in [str(f) for f in files_needing_detection]}
                    if new_results:
                        self.log(f"Writing {len(new_results)} new face detection results to detections_face.yaml...")
                        self._save_all_face_detection_results(new_results)
            else:
                # Update progress if no new detections needed
                progress = 50 + 20  # Skip to 70%
                self.update_progress(progress)

            self.log(f"Phase 3 complete! Used {len(files_with_existing_results)} existing results, performed face detection on {len(files_needing_detection)} new images")
            return face_detection_results

        except Exception as e:
            self.log(f"Error in Phase 3: {e}")
            return {}

    def _phase2_border_detection_and_cropping(self, converted_files):
        """Phase 2: Detect and crop image borders"""
        try:
            self.log("Phase 2: AUTOFIX_BORDERS is enabled. Processing images for border detection and cropping...")
            border_processed_files = 0

            if not converted_files:
                self.log("No converted files to process for border detection")
                return

            # Initialize border detector if not already loaded
            if not self._border_detector:
                self._border_detector = BorderDetection(self.log)

            # Use ThreadPoolExecutor for multithreaded border detection and cropping
            with ThreadPoolExecutor(max_workers=MAX_BORDERS_THREADS) as executor:
                # Submit border processing tasks for all converted files
                future_to_file = {
                    executor.submit(self._process_borders_for_image, image_path): image_path
                    for image_path in converted_files
                }

                # Process completed border processing results
                for future in as_completed(future_to_file):
                    image_path = future_to_file[future]
                    try:
                        future.result()  # We don't need the result, just wait for completion
                        border_processed_files += 1
                    except Exception as e:
                        self.log(f"Error during border processing for {image_path.name}: {e}")
                        border_processed_files += 1

            self.log(f"Phase 2 complete! Processed {border_processed_files} images for border detection and cropping")

        except Exception as e:
            self.log(f"Error in Phase 2: {e}")

    def _process_borders_for_image(self, image_path):
        """Process borders for a single image (helper for multithreading)"""
        try:
            self.log(f"Processing borders for {image_path.name}...")

            # Load image using OpenCV
            img = cv2.imread(str(image_path))
            if img is None:
                self.log(f"Could not load image for border processing: {image_path.name}")
                return

            original_size = img.shape[:2]  # (height, width)

            # Apply border detection and cropping using the new module
            cropped_img = self._border_detector.detect_and_crop_borders(img)

            if cropped_img is not None:
                new_size = cropped_img.shape[:2]

                # Only save if the image was actually cropped (size changed)
                if new_size != original_size:
                    # Save the cropped image
                    cv2.imwrite(str(image_path), cropped_img)
                    self.log(
                        f"Border cropped {image_path.name}: {original_size[1]}x{original_size[0]} → {new_size[1]}x{new_size[0]}")
                else:
                    self.log(f"No border cropping needed for {image_path.name}")
            else:
                self.log(f"Border detection failed for {image_path.name}")

        except Exception as e:
            self.log(f"Error during border processing for {image_path.name}: {e}")

    def _phase4_collect_zero_face_candidates(self, face_detection_results, char_name, target_dir):
        """Phase 4: Collect images with 0 faces detected for potential deletion"""
        try:
            self.log("Phase 4: AUTOFILTER_0_FACES is enabled. Collecting images with 0 faces detected...")
            zero_face_candidates = set()

            for relative_path, result in face_detection_results.items():
                if result['face_count'] == 0:
                    try:
                        # Use Path operations instead of string splitting for cross-platform compatibility
                        relative_path_obj = Path(relative_path)
                        filename = relative_path_obj.name

                        # Construct the full image path in the target directory
                        image_path = target_dir / filename

                        # Add to candidates for deletion
                        zero_face_candidates.add(image_path)

                    except Exception as e:
                        self.log(f"Error processing image {relative_path}: {e}")

            self.log(f"Phase 4 complete! Collected {len(zero_face_candidates)} images with 0 faces detected for deletion")
            return zero_face_candidates

        except Exception as e:
            self.log(f"Error in Phase 4: {e}")
            return set()

    def _phase5_background_removal_with_candidates(self, converted_files, face_detection_results):
        """Phase 5: Remove backgrounds and collect candidates for deletion"""
        try:
            self.log("Phase 5: AUTO_REMOVE_BACKGROUND is enabled. Processing images for background removal...")
            bg_processed_files = 0
            bg_removed_files = 0
            bg_removal_failed_candidates = set()

            if not converted_files:
                self.log("No converted files to process for background removal")
                return bg_removal_failed_candidates

            # Initialize background remover if not already loaded
            if not self._background_remover:
                self._background_remover = BackgroundRemoval(self.characters_path.parent / "models", self.log)

            # Thread-safe lock for updating candidates
            candidates_lock = threading.Lock()

            def process_background_removal_with_candidate_collection(image_path):
                """Helper function for threaded background removal with candidate collection"""
                try:
                    self.log(f"Processing background removal for {image_path.name}...")

                    # Find face detection results for this image
                    face_center_x = None
                    face_center_y = None

                    # Look for the image in face detection results
                    for relative_path, result in face_detection_results.items():
                        if Path(relative_path).name == image_path.name:
                            # Check if we have highest similarity coordinates
                            if 'highest_similarity_center_x' in result and 'highest_similarity_center_y' in result:
                                face_center_x = result['highest_similarity_center_x']
                                face_center_y = result['highest_similarity_center_y']
                                self.log(f"Using highest similarity face center for {image_path.name}: ({face_center_x}, {face_center_y})")
                            elif result.get('faces') and len(result['faces']) > 0:
                                # Fallback to first detected face if no similarity data
                                first_face = result['faces'][0]
                                face_center_x = first_face.get('center_x')
                                face_center_y = first_face.get('center_y')
                                if face_center_x is not None and face_center_y is not None:
                                    self.log(f"Using first detected face center for {image_path.name}: ({face_center_x}, {face_center_y})")
                            break

                    # Load image using PIL
                    with Image.open(image_path) as img:
                        # Convert to RGB if needed
                        if img.mode != 'RGB':
                            img = img.convert('RGB')

                        # Remove background with optional face guidance
                        if face_center_x is not None and face_center_y is not None:
                            rgba_result, mask, bounding_box = self._background_remover.remove_background(
                                img,
                                foreground_threshold=240,  # Pixels > 240 are foreground
                                background_threshold=15,   # Pixels < 15 are background
                                face_center_x=face_center_x,
                                face_center_y=face_center_y,
                                crop_to_1024=True,
                            )
                        else:
                            self.log(f"No face center coordinates available for {image_path.name}, using default background removal")
                            rgba_result, mask, bounding_box = self._background_remover.remove_background(
                                img,
                                foreground_threshold=240,  # Pixels > 240 are foreground
                                background_threshold=15,   # Pixels < 15 are background
                                crop_to_1024=True,
                            )

                        if rgba_result is not None:
                            # Save the RGBA result with transparent background
                            rgba_result.save(str(image_path), 'PNG', optimize=True)
                            self.log(f"Background removed for {image_path.name}")
                            return True  # Return success status
                        else:
                            # Background removal returned None - add to candidates for deletion (96% black mask detected)
                            self.log(f"Background removal returned None for {image_path.name} - adding to deletion candidates (96% black mask)")

                            with candidates_lock:
                                bg_removal_failed_candidates.add(image_path)

                            return False  # Return False to indicate candidate for deletion

                except Exception as e:
                    self.log(f"Error during background removal for {image_path.name}: {e}")
                    return False

            # Use ThreadPoolExecutor for multithreaded background removal
            with ThreadPoolExecutor(max_workers=MAX_REMBG_THREADS) as executor:
                # Submit background removal tasks for all converted files
                future_to_file = {
                    executor.submit(process_background_removal_with_candidate_collection, image_path): image_path
                    for image_path in converted_files
                }

                # Process completed background removal results
                for future in as_completed(future_to_file):
                    image_path = future_to_file[future]
                    try:
                        success = future.result()
                        if success:
                            bg_removed_files += 1
                        bg_processed_files += 1

                        # Progress from 70% to 90%
                        progress = 70 + (bg_processed_files / len(converted_files)) * 20
                        self.update_progress(progress)

                    except Exception as e:
                        self.log(f"Error during background removal for {image_path.name}: {e}")
                        bg_processed_files += 1

            self.log(f"Phase 5 complete! Processed {bg_processed_files} images, removed backgrounds from {bg_removed_files} images, collected {len(bg_removal_failed_candidates)} candidates for deletion")
            return bg_removal_failed_candidates

        except Exception as e:
            self.log(f"Error in Phase 5: {e}")
            return set()

    def _phase6_process_deletions(self, zero_face_candidates, bg_removal_failed_candidates, target_dir):
        """Phase 6: Process final deletions based on candidate sets and flag combinations"""
        try:
            self.log("Phase 6: Processing final deletions based on candidate sets and flag combinations...")

            files_to_delete = set()

            # Always delete files that failed background removal (regardless of other flags)
            if bg_removal_failed_candidates:
                files_to_delete.update(bg_removal_failed_candidates)
                self.log(f"Adding {len(bg_removal_failed_candidates)} files that failed background removal to deletion list (always deleted)")

            # Handle zero face candidates based on flag status
            if AUTOFILTER_0_FACES and zero_face_candidates:
                files_to_delete.update(zero_face_candidates)
                self.log(f"Adding {len(zero_face_candidates)} files with 0 faces to deletion list (AUTOFILTER_0_FACES enabled)")
            elif AUTOFILTER_0_FACES:
                self.log("AUTOFILTER_0_FACES is enabled but no zero face candidates found")
            else:
                self.log("AUTOFILTER_0_FACES is disabled - not deleting files based on face count")

            if not AUTO_REMOVE_BACKGROUND:
                self.log("AUTO_REMOVE_BACKGROUND is disabled - no background removal was performed")

            # Perform the actual deletion
            total_deleted = 0
            for image_path in files_to_delete:
                try:
                    if image_path.exists():
                        image_path.unlink()
                        self.log(f"Deleted image: {image_path.name}")
                        total_deleted += 1
                    else:
                        self.log(f"Image already deleted or doesn't exist: {image_path.name}")

                except Exception as e:
                    self.log(f"Error deleting image {image_path.name}: {e}")

            # Log summary statistics
            self.log(f"Deletion summary:")
            self.log(f"  - Zero face candidates: {len(zero_face_candidates)}")
            self.log(f"  - Background removal failed candidates: {len(bg_removal_failed_candidates)}")
            self.log(f"  - Total unique files marked for deletion: {len(files_to_delete)}")
            self.log(f"  - Files actually deleted: {total_deleted}")
            self.log(f"Phase 6 complete! Deleted {total_deleted} images based on filtering criteria")

        except Exception as e:
            self.log(f"Error in Phase 6: {e}")

    def _remove_background_from_image(self, image_path, face_detection_results):
        """Remove background from a single image (helper for multithreading)"""
        try:
            self.log(f"Processing background removal for {image_path.name}...")

            # Find face detection results for this image
            face_center_x = None
            face_center_y = None

            # Look for the image in face detection results
            for relative_path, result in face_detection_results.items():
                if Path(relative_path).name == image_path.name:
                    # Check if we have highest similarity coordinates
                    if 'highest_similarity_center_x' in result and 'highest_similarity_center_y' in result:
                        face_center_x = result['highest_similarity_center_x']
                        face_center_y = result['highest_similarity_center_y']
                        self.log(f"Using highest similarity face center for {image_path.name}: ({face_center_x}, {face_center_y})")
                    elif result.get('faces') and len(result['faces']) > 0:
                        # Fallback to first detected face if no similarity data
                        first_face = result['faces'][0]
                        face_center_x = first_face.get('center_x')
                        face_center_y = first_face.get('center_y')
                        if face_center_x is not None and face_center_y is not None:
                            self.log(f"Using first detected face center for {image_path.name}: ({face_center_x}, {face_center_y})")
                    break

            # Load image using PIL
            with Image.open(image_path) as img:
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                # Remove background with optional face guidance
                if face_center_x is not None and face_center_y is not None:
                    rgba_result, mask, bounding_box = self._background_remover.remove_background(
                        img,
                        foreground_threshold=240,  # Pixels > 240 are foreground
                        background_threshold=15,   # Pixels < 15 are background
                        face_center_x=face_center_x,
                        face_center_y=face_center_y,
                        crop_to_1024=True,
                    )
                else:
                    self.log(f"No face center coordinates available for {image_path.name}, using default background removal")
                    rgba_result, mask, bounding_box = self._background_remover.remove_background(
                        img,
                        foreground_threshold=240,  # Pixels > 240 are foreground
                        background_threshold=15,   # Pixels < 15 are background
                        crop_to_1024=True,
                    )

                if rgba_result is not None:
                    # Save the RGBA result with transparent background
                    rgba_result.save(str(image_path), 'PNG', optimize=True)

                    # Save the mask as a separate PNG file
                    if mask is not None and False:
                        # Create mask filename: {original_name}_mask.png
                        mask_filename = f"{image_path.stem}_mask.png"
                        mask_path = image_path.parent / mask_filename

                        # Convert mask to PIL Image if it's not already
                        if hasattr(mask, 'save'):
                            # mask is already a PIL Image
                            mask.save(str(mask_path), 'PNG', optimize=True)
                        else:
                            # mask might be a numpy array, convert to PIL Image
                            from PIL import Image as PILImage
                            if mask.dtype != 'uint8':
                                mask = (mask * 255).astype('uint8')
                            mask_img = PILImage.fromarray(mask, mode='L')
                            mask_img.save(str(mask_path), 'PNG', optimize=True)

                        self.log(f"Background removed and mask saved: {image_path.name} -> {mask_filename}")
                    else:
                        self.log(f"Background removed for {image_path.name} (no mask available)")

                    return True  # Return success status instead of incrementing counter
                else:
                    # Background removal returned None - delete the image (96% black mask detected)
                    self.log(f"Background removal returned None for {image_path.name} - deleting image (96% black mask)")
                    try:
                        image_path.unlink()  # Delete the file
                        self.log(f"Successfully deleted {image_path.name}")
                        return False  # Return False to indicate file was deleted
                    except Exception as delete_error:
                        self.log(f"Error deleting {image_path.name}: {delete_error}")
                        return False

        except Exception as e:
            self.log(f"Error during background removal for {image_path.name}: {e}")
            return False

    def _convert_and_save_as_png_batch(self, source_file, target_dir, char_name, target_stage, processed_images_batch):
        """Convert image to PNG format and save to target directory (batch version)"""
        try:
            # Create PNG filename (change extension to .png)
            base_name = source_file.stem  # filename without extension
            png_filename = f"{base_name}.png"
            target_path = target_dir / png_filename

            with Image.open(source_file) as img:
                # Convert to RGB if necessary (handles RGBA, palette, etc.)
                if img.mode not in ('RGB', 'L'):
                    img = img.convert('RGB')

                # Handle naming conflicts for new files
                counter = 1
                while target_path.exists():
                    png_filename = f"{base_name}_{counter}.png"
                    target_path = target_dir / png_filename
                    counter += 1

                # Save as PNG with good compression
                img.save(target_path, 'PNG', optimize=True)
                self.log(f"Converted {source_file.name} → {png_filename}")

                # Collect the original image for batch marking as processed
                self._collect_image_for_processing_mark(char_name, source_file.name, target_stage, processed_images_batch)

                return target_path  # Return the path of the saved PNG

        except Exception as e:
            self.log(f"Error converting {source_file.name} to PNG: {e}")
            return None

    def _save_face_detection_result(self, relative_image_path, face_boxes):
        """Save face detection result for a single image to character.yaml"""
        try:
            import yaml
            from pathlib import Path

            character_yaml_path = self.face_detector.character_path / "character.yaml"

            # Load existing character data or create new
            character_data = {}
            if character_yaml_path.exists():
                try:
                    with open(character_yaml_path, 'r', encoding='utf-8') as f:
                        character_data = yaml.safe_load(f) or {}
                except Exception as e:
                    self.log(f"Error loading character.yaml: {str(e)}")
                    character_data = {}

            # Initialize face_detections section if not exists
            if 'face_detections' not in character_data:
                character_data['face_detections'] = {}

            # Store the face detection result
            character_data['face_detections'][relative_image_path] = {
                'faces': face_boxes,
                'face_count': len(face_boxes)
            }

            # Save updated character data
            with open(character_yaml_path, 'w', encoding='utf-8') as f:
                yaml.dump(character_data, f, default_flow_style=False, allow_unicode=True)

        except Exception as e:
            self.log(f"Error saving face detection result to character.yaml: {str(e)}")

    def _calculate_file_hash(self, file_path):
        """Calculate MD5 hash of file content"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def _perform_face_detection_only(self, image_path):
        """Perform face detection on a single image and return results without saving"""
        try:
            if not self.face_detector:
                return []

            self.log(f"Performing face detection on {image_path.name}...")

            # Detect faces in the image
            face_boxes = self.face_detector.detect_faces_in_image(image_path)

            # Check if reference face is available for similarity computation
            has_reference_face = self.face_detector.reference_face_encoding is not None

            if has_reference_face and face_boxes:
                self.log(f"Computing similarity scores for {len(face_boxes)} face(s) in {image_path.name}...")

                # Compute similarity scores for each face
                for face_box in face_boxes:
                    similarity = self.face_detector.compare_face_with_reference(image_path, face_box)
                    face_box['similarity_score'] = round(similarity, 4)

                self.log(f"Similarity scores computed for {image_path.name}")
            elif not has_reference_face:
                self.log(f"No reference face available - skipping similarity computation for {image_path.name}")

            return face_boxes

        except Exception as e:
            self.log(f"Error during face detection for {image_path.name}: {e}")
            return []

    def _save_all_face_detection_results(self, face_detection_results):
        """Save all face detection results to detections_face.yaml in a single batch operation"""
        try:
            import yaml
            from pathlib import Path

            # Change to use detections_face.yaml instead of character.yaml
            detections_face_yaml_path = self.face_detector.character_path / "detections_face.yaml"

            # Load existing face detection data or create new
            face_detection_data = {}
            if detections_face_yaml_path.exists():
                try:
                    with open(detections_face_yaml_path, 'r', encoding='utf-8') as f:
                        face_detection_data = yaml.safe_load(f) or {}
                except Exception as e:
                    self.log(f"Error loading detections_face.yaml: {str(e)}")
                    face_detection_data = {}

            # Check if reference face is available for similarity computation
            has_reference_face = self.face_detector.reference_face_encoding is not None

            # Process each face detection result to add similarity data
            for relative_path, result in face_detection_results.items():
                detection_data = {
                    'faces': result['faces'],
                    'face_count': result['face_count']
                }

                # Add similarity data if reference face is available and faces were detected
                if has_reference_face and result['faces']:
                    highest_similarity = 0.0
                    highest_similarity_center_x = None
                    highest_similarity_center_y = None

                    # Find the highest similarity score among all faces
                    for face_box in result['faces']:
                        if 'similarity_score' in face_box:
                            similarity = face_box['similarity_score']
                            if similarity > highest_similarity:
                                highest_similarity = similarity
                                highest_similarity_center_x = face_box.get('center_x')
                                highest_similarity_center_y = face_box.get('center_y')

                    # Add highest similarity data to detection_data
                    if highest_similarity > 0:
                        detection_data['highest_similarity'] = round(highest_similarity, 4)
                        if highest_similarity_center_x is not None:
                            detection_data['highest_similarity_center_x'] = highest_similarity_center_x
                            detection_data['highest_similarity_center_y'] = highest_similarity_center_y

                # Add to face detection data
                face_detection_data[relative_path] = detection_data

            # Save updated face detection data to detections_face.yaml
            with open(detections_face_yaml_path, 'w', encoding='utf-8') as f:
                yaml.dump(face_detection_data, f, default_flow_style=False, allow_unicode=True)

            if has_reference_face:
                self.log(f"Successfully saved {len(face_detection_results)} face detection results with similarity scores to detections_face.yaml")
            else:
                self.log(f"Successfully saved {len(face_detection_results)} face detection results to detections_face.yaml")

        except Exception as e:
            self.log(f"Error saving all face detection results to detections_face.yaml: {str(e)}")

    def _load_existing_face_detection_results(self, char_name):
        """Load existing face detection results from detections_face.yaml"""
        try:
            import yaml
            from pathlib import Path

            detections_face_yaml_path = self.characters_path / char_name / "detections_face.yaml"

            if not detections_face_yaml_path.exists():
                self.log(f"No existing detections_face.yaml found for {char_name}")
                return {}

            with open(detections_face_yaml_path, 'r', encoding='utf-8') as f:
                face_detection_data = yaml.safe_load(f) or {}

            self.log(f"Loaded existing face detection results for {len(face_detection_data)} images from detections_face.yaml")
            return face_detection_data

        except Exception as e:
            self.log(f"Error loading existing face detection results: {e}")
            return {}

