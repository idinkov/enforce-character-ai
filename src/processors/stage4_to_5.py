"""
Stage 4 to 5: Inpainting Processor
Applies inpainting and fixes to processed 1024x1024 images
"""
import os

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import tempfile
from .base_processor import BaseProcessor
from ..utils.img2img_sdxl_inpaint import InpaintingPipeline
from ..services.gpu_service import get_gpu_service


class InpaintingProcessor(BaseProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gpu_service = get_gpu_service()
        self.inpainting_pipeline = None
        self._initialize_pipeline()

    def _initialize_pipeline(self):
        """Initialize the inpainting pipeline with current GPU selection."""
        try:
            # Unload existing pipeline if it exists
            if self.inpainting_pipeline is not None:
                try:
                    self.inpainting_pipeline.unload_model()
                    self.log("Unloaded previous inpainting pipeline")
                except Exception as e:
                    self.log(f"Error unloading previous pipeline: {e}")

            device_info = self.gpu_service.get_selected_device()
            if device_info['type'] == 'gpu':
                device = f"cuda:{device_info['index']}"
                self.log(f"Initializing inpainting pipeline with GPU {device_info['index']}: {device_info['name']}")
            else:
                device = 'cpu'
                self.log("Initializing inpainting pipeline with CPU")

            # Create the pipeline with the selected device
            self.inpainting_pipeline = InpaintingPipeline(device=device)
            # Don't load the model here - it will be loaded when needed
            self.log(f"Inpainting pipeline created for device: {device}")
        except Exception as e:
            self.log(f"Error initializing inpainting pipeline: {e}")
            # Fallback to CPU
            try:
                self.inpainting_pipeline = InpaintingPipeline(device='cpu')
                self.log("Fallback to CPU inpainting pipeline successful")
            except Exception as fallback_error:
                self.log(f"Error with CPU fallback: {fallback_error}")
                self.inpainting_pipeline = None

    def reinitialize_with_selected_gpu(self):
        """Reinitialize inpainting pipeline with currently selected GPU."""
        self._initialize_pipeline()

    def get_source_stage(self):
        return "4_processed_1024"

    def get_target_stage(self):
        return "5_processed_fixed_1024"

    def _detect_black_borders(self, img, threshold=60, edge_percentage=0.7):
        """Detect black borders in the image with improved sensitivity for non-perfect black colors

        Args:
            img: OpenCV image (BGR format)
            threshold: Pixel intensity threshold for considering pixels as "dark" (increased from 30 to 60)
            edge_percentage: Percentage of edge pixels that need to be dark to consider it a border

        Returns:
            bool: True if black borders are detected, False otherwise
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # Check edges for dark pixels with more flexible criteria
        edge_width = 10  # Check a wider edge area

        # Top edge - check if enough pixels are dark
        top_pixels = gray[0:edge_width, :]
        top_dark_ratio = np.sum(top_pixels < threshold) / top_pixels.size
        top_edge = top_dark_ratio > edge_percentage

        # Bottom edge
        bottom_pixels = gray[h-edge_width:h, :]
        bottom_dark_ratio = np.sum(bottom_pixels < threshold) / bottom_pixels.size
        bottom_edge = bottom_dark_ratio > edge_percentage

        # Left edge
        left_pixels = gray[:, 0:edge_width]
        left_dark_ratio = np.sum(left_pixels < threshold) / left_pixels.size
        left_edge = left_dark_ratio > edge_percentage

        # Right edge
        right_pixels = gray[:, w-edge_width:w]
        right_dark_ratio = np.sum(right_pixels < threshold) / right_pixels.size
        right_edge = right_dark_ratio > edge_percentage

        return any([top_edge, bottom_edge, left_edge, right_edge])

    def _create_border_mask(self, img, threshold=10, max_border_width=800, gradient_threshold=0.9):
        """Create a mask for black border areas with dynamic border width detection

        Args:
            img: OpenCV image (BGR format)
            threshold: Pixel intensity threshold for considering pixels as "dark" (increased from 30 to 60)
            max_border_width: Maximum border width to check (safety limit)
            gradient_threshold: Ratio of dark pixels needed in a row/column to include it in mask

        Returns:
            PIL Image: Mask image where white pixels indicate areas to inpaint
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # Create mask - start with all black (no inpainting)
        mask = np.zeros((h, w), dtype=np.uint8)

        # Detect dark borders and create mask (white where inpainting needed)
        # Dynamically determine border width based on actual dark content

        # Top border - scan from top until we find non-dark rows
        for y in range(min(max_border_width, h)):
            row_pixels = gray[y, :]
            dark_ratio = np.sum(row_pixels < threshold) / len(row_pixels)
            if dark_ratio > gradient_threshold:
                mask[y, :] = 255
            elif y > 0 and dark_ratio > 0.3:
                if np.any(mask[y-1, :] > 0):
                    mask[y, :] = 255
                else:
                    break  # Found the end of the border
            else:
                break  # Found the end of the border

        # Bottom border - scan from bottom until we find non-dark rows
        for y in range(h-1, max(h-max_border_width-1, -1), -1):
            row_pixels = gray[y, :]
            dark_ratio = np.sum(row_pixels < threshold) / len(row_pixels)
            if dark_ratio > gradient_threshold:
                mask[y, :] = 255
            elif y < h-1 and dark_ratio > 0.3:
                if np.any(mask[y+1, :] > 0):
                    mask[y, :] = 255
                else:
                    break  # Found the end of the border
            else:
                break  # Found the end of the border

        # Left border - scan from left until we find non-dark columns
        for x in range(min(max_border_width, w)):
            col_pixels = gray[:, x]
            dark_ratio = np.sum(col_pixels < threshold) / len(col_pixels)
            if dark_ratio > gradient_threshold:
                mask[:, x] = 255
            elif x > 0 and dark_ratio > 0.3:
                if np.any(mask[:, x-1] > 0):
                    mask[:, x] = 255
                else:
                    break  # Found the end of the border
            else:
                break  # Found the end of the border

        # Right border - scan from right until we find non-dark columns
        for x in range(w-1, max(w-max_border_width-1, -1), -1):
            col_pixels = gray[:, x]
            dark_ratio = np.sum(col_pixels < threshold) / len(col_pixels)
            if dark_ratio > gradient_threshold:
                mask[:, x] = 255
            elif x < w-1 and dark_ratio > 0.3:
                if np.any(mask[:, x+1] > 0):
                    mask[:, x] = 255
                else:
                    break  # Found the end of the border
            else:
                break  # Found the end of the border

        # --- Ensure symmetry: if border detected on one side but not the other, apply to both ---
        # Top border width
        top_border = 0
        for y in range(min(max_border_width, h)):
            if np.all(mask[y, :] == 255):
                top_border += 1
            else:
                break
        # Bottom border width
        bottom_border = 0
        for y in range(h-1, h-1-min(max_border_width, h), -1):
            if np.all(mask[y, :] == 255):
                bottom_border += 1
            else:
                break
        # If only one of top/bottom has border, mirror it
        if top_border > 0 and bottom_border == 0:
            mask[h-top_border:h, :] = 255
        elif bottom_border > 0 and top_border == 0:
            mask[0:bottom_border, :] = 255

        # Left border width
        left_border = 0
        for x in range(min(max_border_width, w)):
            if np.all(mask[:, x] == 255):
                left_border += 1
            else:
                break
        # Right border width
        right_border = 0
        for x in range(w-1, w-1-min(max_border_width, w), -1):
            if np.all(mask[:, x] == 255):
                right_border += 1
            else:
                break
        # If only one of left/right has border, mirror it
        if left_border > 0 and right_border == 0:
            mask[:, w-left_border:w] = 255
        elif right_border > 0 and left_border == 0:
            mask[:, 0:right_border] = 255

        # Apply morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Fill small gaps
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # Remove small noise

        # Convert to PIL Image
        return Image.fromarray(mask, mode='L')

    def process_character(self, char_name):
        """Apply quality control and prepare for training"""
        try:
            self.log(f"Starting quality control for {char_name}...")

            source_dir = self.characters_path / char_name / "images" / self.get_source_stage()
            target_dir = self.characters_path / char_name / "images" / self.get_target_stage()

            if not source_dir.exists():
                self.log(f"Error: Source directory {self.get_source_stage()} does not exist")
                return False

            # Physically delete images marked as deleted in deleted.yaml
            deleted_count = self.physically_delete_marked_images(char_name, source_dir)
            if deleted_count > 0:
                self.log(f"Physically deleted {deleted_count} images marked as deleted")

            # Create target directory if it doesn't exist
            target_dir.mkdir(parents=True, exist_ok=True)

            # Normalize any existing target filenames to remove prefixes/suffixes
            self._normalize_target_filenames(char_name, source_dir, target_dir)

            image_files = self.get_image_files(source_dir, char_name)

            if not image_files:
                self.log(f"No images found in {self.get_source_stage()}")
                return False

            total_files = len(image_files)
            processed = 0
            successful = 0
            rejected = 0
            skipped = 0

            self.update_progress(0)


            # Filter out files that already exist in target directory
            files_to_process = []
            for img_file in image_files:
                target_path = target_dir / img_file.name
                if target_path.exists():
                    self.log(f"Skipping {img_file.name} - already exists in target directory")
                    skipped += 1
                    processed += 1
                else:
                    files_to_process.append(img_file)

            if not files_to_process:
                self.log(f"All {len(image_files)} files already exist in target directory, nothing to process")
                self.update_progress(100)
                return True

            self.log(f"Processing {len(files_to_process)} files, skipped {skipped} existing files")

            # Initialize inpainting pipeline once for all images that need it
            inpainting_pipeline = None
            images_needing_inpainting = []

            # First pass: identify which images need inpainting (only for files_to_process)
            for img_file in files_to_process:
                try:
                    # Load image
                    img = cv2.imread(str(img_file))
                    if img is None:
                        self.log(f"Could not load image: {img_file.name}")
                        continue

                    # Check if image is 1024x1024 (required for SDXL inpainting)
                    h, w = img.shape[:2]
                    if h != 1024 or w != 1024:
                        continue  # Will be handled in main loop

                    # Detect black borders
                    has_black_borders = self._detect_black_borders(img)
                    if has_black_borders:
                        # Create mask for inpainting
                        mask = self._create_border_mask(img)
                        mask_array = np.array(mask)
                        if np.any(mask_array > 0):
                            images_needing_inpainting.append((img_file, img, mask))

                except Exception as e:
                    self.log(f"Error checking {img_file.name}: {e}")
                    continue

            # Load inpainting model only if needed
            if images_needing_inpainting:
                self.log(f"Found {len(images_needing_inpainting)} images requiring inpainting. Loading model...")
                # Use the properly configured pipeline from the processor
                if self.inpainting_pipeline is not None:
                    inpainting_pipeline = self.inpainting_pipeline
                    inpainting_pipeline.load_model()
                    self.log(f"Using processor's inpainting pipeline on device: {inpainting_pipeline.device}")
                else:
                    self.log("Error: No inpainting pipeline available")
                    return False

            # Main processing loop (only for files_to_process)
            for img_file in files_to_process:
                try:
                    # Load image
                    img = cv2.imread(str(img_file))
                    if img is None:
                        self.log(f"Could not load image: {img_file.name}")
                        processed += 1
                        continue

                    # Check if image is 1024x1024 (required for SDXL inpainting)
                    h, w = img.shape[:2]
                    if h != 1024 or w != 1024:
                        self.log(f"Image {img_file.name} is not 1024x1024 ({w}x{h}), skipping inpainting")
                        # Copy as is
                        target_path = target_dir / img_file.name
                        cv2.imwrite(str(target_path), img)
                        successful += 1
                        processed += 1
                        continue

                    # Check if this image needs inpainting
                    needs_inpainting = False
                    mask = None
                    for inpaint_img_file, inpaint_img, inpaint_mask in images_needing_inpainting:
                        if inpaint_img_file == img_file:
                            needs_inpainting = True
                            mask = inpaint_mask
                            break

                    if needs_inpainting and inpainting_pipeline is not None:
                        self.log(f"Black borders detected in {img_file.name}, applying inpainting...")

                        # Save the mask for debugging/inspection
                        mask_save_path = target_dir / f"mask_{img_file.stem}.png"
                        mask.save(mask_save_path)
                        self.log(f"Saved mask for {img_file.name} at {mask_save_path.name}")

                        # Create temporary files for inpainting
                        with tempfile.TemporaryDirectory() as temp_dir:
                            temp_input = Path(temp_dir) / f"input_{img_file.name}"
                            temp_mask = Path(temp_dir) / f"mask_{img_file.stem}.png"
                            temp_output = Path(temp_dir) / f"output_{img_file.name}"

                            # Save input image as RGB
                            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            Image.fromarray(img_rgb).save(temp_input)

                            # Save mask
                            mask.save(temp_mask)

                            # Generate prompt based on character name
                            prompt = f"high quality portrait of {char_name}, professional photography, detailed face, natural lighting"

                            try:
                                # Perform inpainting using the loaded pipeline
                                output_path = inpainting_pipeline.process_image(
                                    input_image_path=str(temp_input),
                                    mask_image_path=str(temp_mask),
                                    prompt=prompt,
                                    output_path=str(temp_output),
                                    num_inference_steps=40,
                                    guidance_scale=7.0
                                )

                                # Check if inpainting was successful
                                if output_path is None:
                                    raise ValueError("Inpainting function returned None - inpainting failed")

                                if not Path(output_path).exists():
                                    raise FileNotFoundError(f"Inpainting output file not created: {output_path}")

                                # Load the inpainted result and convert back to BGR
                                inpainted_pil = Image.open(output_path).convert('RGB')
                                inpainted_img = cv2.cvtColor(np.array(inpainted_pil), cv2.COLOR_RGB2BGR)

                                # Save to target directory
                                target_path = target_dir / img_file.name
                                cv2.imwrite(str(target_path), inpainted_img)
                                # Remove temporary mask file
                                os.remove(mask_save_path)
                                self.log(f"Successfully inpainted {img_file.name}")
                                successful += 1

                            except Exception as inpaint_error:
                                self.log(f"Inpainting failed for {img_file.name}: {str(inpaint_error)}")
                                self.log(f"Error type: {type(inpaint_error).__name__}")
                                # Fall back to copying original
                                target_path = target_dir / img_file.name
                                cv2.imwrite(str(target_path), img)
                                successful += 1
                    else:
                        # No black borders detected or no inpainting needed, copy image as is
                        target_path = target_dir / img_file.name
                        cv2.imwrite(str(target_path), img)
                        successful += 1

                    processed += 1
                    progress = ((processed + skipped) / total_files) * 100
                    self.update_progress(progress)

                except Exception as e:
                    self.log(f"Error processing {img_file.name}: {e}")
                    processed += 1
                    rejected += 1

            # Clean up: unload the inpainting model
            if inpainting_pipeline is not None:
                self.log("Unloading inpainting model...")
                inpainting_pipeline.unload_model()

            self.log(f"Quality control complete! Successfully processed {successful} images, skipped {skipped} existing files, rejected {rejected}")
            self.update_progress(100)
            return True

        except Exception as e:
            self.log(f"Error during quality control: {e}")
            return False
