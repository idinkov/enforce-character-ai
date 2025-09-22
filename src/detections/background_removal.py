"""
Background Removal Module
Handles automatic background removal using U2Net model
"""
import numpy as np
import cv2
import onnxruntime as ort
from PIL import Image
from pathlib import Path
from ..models import get_model_manager
from ..services.gpu_service import get_gpu_service


class BackgroundRemoval:
    """U2Net-based background removal using ONNX runtime with GPU acceleration support"""

    def __init__(self, models_path=None, log_callback=None):
        """Initialize the U2Net background remover

        Args:
            models_path: Path to the models directory (deprecated, use model manager)
            log_callback: Optional logging callback function
        """
        self.log = log_callback if log_callback else lambda x: print(x)
        self.session = None
        self.input_size = 320  # U2Net standard input size
        self.gpu_service = get_gpu_service()
        self._load_model()

    def _load_model(self):
        """Load the U2Net ONNX model with GPU acceleration if available"""
        try:
            # Use the model manager to get the model path
            model_manager = get_model_manager()
            model_path = model_manager.get_model_path('u2net_human_seg')

            if not model_path or not model_path.exists():
                # Try to download the model if it's missing
                if not model_manager.is_model_available('u2net_human_seg'):
                    self.log("U2Net model not found, attempting to download...")
                    success = model_manager.download_model('u2net_human_seg')
                    if success:
                        model_path = model_manager.get_model_path('u2net_human_seg')
                    else:
                        raise FileNotFoundError("Failed to download U2Net model")
                else:
                    raise FileNotFoundError("U2Net model not found")

            # Get ONNX providers from GPU service
            providers = self.gpu_service.get_onnx_providers()

            device_info = self.gpu_service.get_selected_device()
            if device_info['type'] == 'gpu':
                self.log(f"Loading U2Net model with GPU {device_info['index']}: {device_info['name']}")
            else:
                self.log("Loading U2Net model with CPU")


            # Create ONNX Runtime session
            self.session = ort.InferenceSession(
                str(model_path), providers=providers
            )

            # Get model input/output info
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name

            self.log(
                f"U2Net background removal model loaded successfully from {model_path}"
            )
            self.log(f"Using providers: {self.session.get_providers()}")

        except Exception as e:
            self.log(f"Error loading U2Net model: {e}")
            self.session = None

    def get_h_w_c(self, img):
        """Get height, width, and channels from image array"""
        if len(img.shape) == 3:
            return img.shape[0], img.shape[1], img.shape[2]
        elif len(img.shape) == 2:
            return img.shape[0], img.shape[1], 1
        else:
            raise ValueError(f"Unsupported image shape: {img.shape}")

    def crop_to_content_fixed_size(
        self, img, img_orig, thresh_val=50.0, fixed_input=1024, padding=20
    ):
        """Crop image to content and resize to maximum fixed size (1024x1024)

        Args:
            img: RGBA image as numpy array for detection
            img_orig: Original image as numpy array to crop from
            thresh_val: Threshold value (0-100) for alpha detection
            fixed_input: Maximum target size (default 1024) - will not upscale beyond crop size
            padding: Padding around detected content

        Returns:
            tuple: (cropped_resized_image, mask) both as numpy arrays
        """
        try:
            # Threshold value 100 guarantees an empty image, so make sure the max
            # is just below that.
            thresh_val = min(thresh_val / 100, 0.99999)

            # Valid alpha is greater than threshold, else impossible to crop 0 alpha only
            alpha = img[:, :, 3]
            r = np.any(alpha > thresh_val, 1)
            if r.any():
                h, w, _ = self.get_h_w_c(img)
                c = np.any(alpha > thresh_val, 0)
                bounding_box_x = np.where(c)[0]
                bounding_box_y = np.where(r)[0]

                x = max(0, int(bounding_box_x[0]) - padding)
                y = max(0, int(bounding_box_y[0]) - padding)
                x_end = min(w, int(bounding_box_x[-1]) + padding)
                y_end = min(h, int(bounding_box_y[-1]) + padding)
                width = x_end - x
                height = y_end - y

                max_width = w
                max_height = h

                square_max = max(width, height)
                add_x = (square_max - width) // 2
                add_y = (square_max - height) // 2

                # If add_x is over 0 try to add padding to each side
                if add_x > 0:
                    x = max(0, x - add_x)
                    x_end = min(max_width, x_end + add_x)
                    width = x_end - x
                # If add_y is over 0 try to add padding to each side
                if add_y > 0:
                    y = max(0, y - add_y)
                    y_end = min(max_height, y_end + add_y)
                    height = y_end - y

                square_max = max(width, height)
                add_x = (square_max - width)
                add_y = (square_max - height)

                # If add_x is over 0 try to add padding to left
                if add_x > 0:
                    x = max(0, x - add_x)
                    width = x_end - x
                # If add_y is over 0 try to add padding to top
                if add_y > 0:
                    y = max(0, y - add_y)
                    height = y_end - y

                square_max = max(width, height)
                add_x = (square_max - width)
                add_y = (square_max - height)

                # If add_x is over 0 try to add padding to right
                if add_x > 0:
                    x_end = min(max_width, x_end + add_x)
                    width = x_end - x
                # If add_y is over 0 try to add padding to bottom
                if add_y > 0:
                    y_end = min(max_height, y_end + add_y)
                    height = y_end - y

                square_max = max(width, height)
                add_x = (square_max - width)
                add_y = (square_max - height)

                x = max(0, x - add_x)
                y = max(0, y - add_y)
                x_end = min(max_width, x_end + add_x)
                y_end = min(max_height, y_end + add_y)

                imgout = img_orig[y:y_end, x:x_end]
                # Add black pixels padding to make it square
                imgout_width = imgout.shape[1]
                imgout_height = imgout.shape[0]

                # Initialize mask
                mask = np.zeros((imgout.shape[0], imgout.shape[1]), dtype=np.uint8)
                # Pad the image to make it square
                if imgout_width > imgout_height:
                    diff = imgout_width - imgout_height
                    imgout = np.pad(
                        imgout,
                        ((diff // 2, diff - diff // 2), (0, 0), (0, 0)),
                        mode='constant',
                        constant_values=0,
                    )
                    mask = np.pad(
                        mask,
                        ((diff // 2, diff - diff // 2), (0, 0)),
                        mode='constant',
                        constant_values=255,
                    )
                    # Update mask
                elif imgout_height > imgout_width:
                    diff = imgout_height - imgout_width
                    imgout = np.pad(
                        imgout,
                        ((0, 0), (diff // 2, diff - diff // 2), (0, 0)),
                        mode='constant',
                        constant_values=0,
                    )
                    mask = np.pad(
                        mask,
                        ((0, 0), (diff // 2, diff - diff // 2)),
                        mode='constant',
                        constant_values=255,
                    )

                # Get the current square size after padding
                current_size = max(imgout.shape[0], imgout.shape[1])

                # Only resize if the current size is larger than fixed_input (downscaling)
                # If current size is smaller than fixed_input, keep it at actual size (no upscaling)
                if current_size > fixed_input:
                    # Downscale to fixed_input
                    imgout = cv2.resize(
                        imgout, (fixed_input, fixed_input), interpolation=cv2.INTER_AREA
                    )
                    mask = cv2.resize(
                        mask, (fixed_input, fixed_input), interpolation=cv2.INTER_AREA
                    )
                    self.log(
                        f"Downscaled image from {current_size}x{current_size} to {fixed_input}x{fixed_input}"
                    )
                else:
                    # Keep at actual size (no upscaling)
                    self.log(
                        f"Keeping image at actual size {current_size}x{current_size} (no upscaling to {fixed_input}x{fixed_input})"
                    )

                return (imgout, mask)

            # Add padding to the image to the fixed size
            threshold = 1.0
            threshold /= 100
            max_value = 1
            imgout = np.zeros((fixed_input, fixed_input, img_orig.shape[2]), dtype=img_orig.dtype)
            _, result = cv2.threshold(imgout, threshold, max_value, cv2.THRESH_BINARY)
            return (imgout, np.zeros((fixed_input, fixed_input), dtype=np.uint8))

        except Exception as e:
            self.log(f"Error in crop_to_content_fixed_size: {e}")
            # Return original image resized as fallback
            try:
                imgout = cv2.resize(img_orig, (fixed_input, fixed_input), interpolation=cv2.INTER_AREA)
                mask = np.zeros((fixed_input, fixed_input), dtype=np.uint8)
                return (imgout, mask)
            except:
                return (None, None)

    def remove_background(
        self,
        image_input,
        foreground_threshold=240,
        background_threshold=15,
        apply_post_processing=True,
        face_center_x=None,
        face_center_y=None,
        crop_to_1024=False,
    ):
        """Remove background from image using U2Net model

        Args:
            image_input: PIL Image, numpy array, or path to image file
            foreground_threshold: Pixels > this value (1-255) are considered foreground
            background_threshold: Pixels < this value (0-254) are considered background
            apply_post_processing: Whether to apply morphological post-processing
            face_center_x: Optional X coordinate of face center for focused processing
            face_center_y: Optional Y coordinate of face center for focused processing
            crop_to_1024: Whether to crop image to content and resize to fixed size (1024x1024)

        Returns:
            tuple: (rgba_result, grayscale_mask, bounding_box) or (None, None, None) if failed
            bounding_box: (x, y, width, height) of the detected foreground
        """
        if not self.session:
            self.log("U2Net model not loaded, cannot remove background")
            return None, None, None

        try:
            # Load and convert image to RGB PIL Image
            rgb_image = self._load_image_as_pil_rgb(image_input)
            original_image_array = np.array(rgb_image)
            if rgb_image is None:
                return None, None, None

            # Store original dimensions
            original_width, original_height = rgb_image.size

            # Apply YOLOv8n-seg person detection if face center coordinates are provided
            if face_center_x is not None and face_center_y is not None:
                self.log(f"Using face center coordinates for YOLOv8n-seg person segmentation: ({face_center_x}, {face_center_y})")

                # Apply YOLOv8n-seg person segmentation to focus on the person containing the face
                person_focused_image = self._keep_person_segmentation(rgb_image, face_center_x, face_center_y)
                if person_focused_image is not None:
                    # Convert RGBA result to RGB with black background for U2Net input
                    rgb_image = self._rgba_to_rgb_with_black_background(person_focused_image)
                    self.log("Successfully applied YOLOv8n-seg person segmentation for focused background removal")
                else:
                    self.log("YOLOv8n-seg person segmentation failed, proceeding with original image")
            else:
                self.log("No face center coordinates provided, using standard background removal")

            # Preprocess image for U2Net
            input_tensor = self._preprocess_image(rgb_image)

            # Run inference
            outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
            mask_output = outputs[0][0][0]  # Remove batch and channel dimensions

            # Post-process mask with optional face center coordinates
            mask = self._postprocess_mask(mask_output, original_width, original_height,
                                        foreground_threshold, background_threshold, apply_post_processing,
                                        center_x=face_center_x, center_y=face_center_y)

            # Calculate bounding box of detected foreground
            bounding_box = self._calculate_bounding_box(mask)

            # Create RGBA result
            rgba_result = self._create_rgba_result(rgb_image, mask)

            # Create grayscale mask for return
            grayscale_mask = Image.fromarray((mask * 255).astype(np.uint8), mode='L')

            # Apply crop to 1024x1024 if requested
            if crop_to_1024:
                try:
                    self.log("Applying crop to 1024x1024 to RGBA result and grayscale mask")

                    # Convert PIL images to numpy arrays
                    rgba_array = np.array(rgba_result)
                    grayscale_array = np.array(grayscale_mask)

                    # Apply cropping to RGBA result
                    rgba_cropped, _ = self.crop_to_content_fixed_size(
                        rgba_array, original_image_array, thresh_val=96.0, fixed_input=1024, padding=20
                    )

                    # Apply cropping to grayscale mask (convert to RGBA for processing, then back)
                    grayscale_rgba = cv2.cvtColor(grayscale_array, cv2.COLOR_GRAY2RGBA)
                    grayscale_rgba[:, :, 3] = grayscale_array  # Use grayscale as alpha channel
                    grayscale_cropped, _ = self.crop_to_content_fixed_size(
                        grayscale_rgba, grayscale_rgba, thresh_val=96.0, fixed_input=1024, padding=20
                    )

                    if rgba_cropped is not None and grayscale_cropped is not None:
                        rgba_result = Image.fromarray(rgba_cropped, mode='RGB')
                        # Extract alpha channel from cropped grayscale RGBA and use as grayscale mask
                        grayscale_mask = Image.fromarray(grayscale_cropped[:, :, 3], mode='L')
                        self.log("Successfully applied crop to 1024x1024")
                    else:
                        self.log("Cropping to 1024x1024 failed, using original results")

                except Exception as e:
                    self.log(f"Error applying crop to 1024x1024: {e}")

            # If mask is 96% black, return None to indicate deletion
            if grayscale_mask is not None and self._is_mask_mostly_black(grayscale_mask, threshold_percent=96.0):
                self.log(f"Mask is 96% black, indicating image should be deleted")
                return None, None, None

            return rgba_result, grayscale_mask, bounding_box

        except Exception as e:
            self.log(f"Error during background removal: {e}")
            return None, None, None

    def _keep_person_segmentation(self, image_pil, position_x, position_y):
        """Use YOLOv8n-seg to detect and segment person containing the face center coordinates

        Args:
            image_pil: PIL RGB Image
            position_x: X coordinate of face center
            position_y: Y coordinate of face center

        Returns:
            PIL RGBA Image with person segmentation mask kept, rest transparent, or None if failed
        """
        try:
            from ultralytics import YOLO
            import cv2

            # Convert PIL to numpy array (RGB format)
            image_np = np.array(image_pil)

            # Use the model manager to get the YOLOv8n-seg model path
            model_manager = get_model_manager()
            model_path = model_manager.get_model_path('yolov8n_seg')

            if not model_path or not model_path.exists():
                # Try to download the model if it's missing
                if not model_manager.is_model_available('yolov8n_seg'):
                    self.log("YOLOv8n-seg model not found, attempting to download...")
                    success = model_manager.download_model('yolov8n_seg')
                    if success:
                        model_path = model_manager.get_model_path('yolov8n_seg')
                    else:
                        self.log("Failed to download YOLOv8n-seg model")
                        return None
                else:
                    self.log("YOLOv8n-seg model not found")
                    return None

            # Load YOLOv8n-seg model - force CPU usage
            model = YOLO(str(model_path))

            # Force CPU usage for YOLOv8n-seg
            device = 'cpu'
            self.log("Using CPU for YOLOv8n-seg (forced CPU mode)")
            model.to(device)

            # Perform inference with CPU
            results = model(image_np, verbose=False, device=device)

            # Extract person segmentations (class 0 = person in COCO dataset)
            person_segments = []
            for result in results:
                if result.masks is not None and result.boxes is not None:
                    for mask, box in zip(result.masks.data, result.boxes):
                        # Check if detection is a person (class 0)
                        if int(box.cls[0]) == 0:  # Person class
                            confidence = float(box.conf[0])

                            # Only consider detections with reasonable confidence
                            if confidence > 0.3:
                                # Get segmentation mask as numpy array
                                mask_np = mask.cpu().numpy()

                                # Get bounding box coordinates for checking if point is inside
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                                person_segments.append({
                                    'mask': mask_np,
                                    'bbox': (x1, y1, x2, y2),
                                    'confidence': confidence
                                })

            if not person_segments:
                self.log("No person detected by YOLOv8n-seg")
                return None

            # Identify the segmentation that contains the (position_x, position_y)
            selected_segment = None
            best_confidence = 0

            for segment in person_segments:
                x1, y1, x2, y2 = segment['bbox']

                # First check if point is within bounding box
                if x1 <= position_x <= x2 and y1 <= position_y <= y2:
                    # Then check if point is within the actual segmentation mask
                    mask = segment['mask']

                    # Resize mask to match original image dimensions if needed
                    if mask.shape != image_np.shape[:2]:
                        mask_resized = cv2.resize(mask, (image_np.shape[1], image_np.shape[0]), interpolation=cv2.INTER_NEAREST)
                    else:
                        mask_resized = mask

                    # Check if the point is within the segmentation mask (mask value > 0.5)
                    if mask_resized[int(position_y), int(position_x)] > 0.5:
                        # If multiple segments contain the point, choose the one with highest confidence
                        if segment['confidence'] > best_confidence:
                            selected_segment = segment
                            best_confidence = segment['confidence']

            if selected_segment is None:
                self.log(f"No person segmentation found containing point ({position_x}, {position_y})")
                # Create transparent image with original dimensions
                height, width = image_np.shape[:2]
                rgba_transparent = np.zeros((height, width, 4), dtype=np.uint8)
                return Image.fromarray(rgba_transparent, mode='RGBA')

            # Use the selected segmentation mask
            mask = selected_segment['mask']

            # Resize mask to match original image dimensions if needed
            if mask.shape != image_np.shape[:2]:
                mask_resized = cv2.resize(mask, (image_np.shape[1], image_np.shape[0]), interpolation=cv2.INTER_NEAREST)
            else:
                mask_resized = mask

            # Create RGBA image
            height, width = image_np.shape[:2]
            rgba_image = np.zeros((height, width, 4), dtype=np.uint8)

            # Copy RGB channels where mask is positive
            mask_bool = mask_resized > 0.5
            rgba_image[mask_bool, :3] = image_np[mask_bool]
            rgba_image[mask_bool, 3] = 255  # Full opacity for person areas
            # Transparent areas remain with alpha = 0

            # Convert back to PIL Image
            rgba_pil = Image.fromarray(rgba_image, mode='RGBA')

            self.log(f"YOLOv8n-seg detected person segmentation with confidence: {best_confidence:.3f} using {device}")
            return rgba_pil

        except Exception as e:
            self.log(f"Error in YOLOv8n-seg person segmentation: {e}")
            return None

    def _rgba_to_rgb_with_black_background(self, rgba_image):
        """Convert RGBA image to RGB with black background

        Args:
            rgba_image: PIL RGBA Image

        Returns:
            PIL RGB Image with transparent areas filled with black
        """
        try:
            # Create a black background
            rgb_with_black = Image.new('RGB', rgba_image.size, (0, 0, 0))

            # Paste the RGBA image onto the black background
            # The alpha channel will be used for blending
            rgb_with_black.paste(rgba_image, mask=rgba_image.split()[3])  # Use alpha channel as mask

            return rgb_with_black

        except Exception as e:
            self.log(f"Error converting RGBA to RGB with black background: {e}")
            return rgba_image.convert('RGB')  # Fallback conversion

    def _calculate_bounding_box(self, mask, padding=0):
        """Calculate bounding box of the foreground in the mask

        Args:
            mask: 2D numpy array with values between 0 and 1
            padding: Number of pixels to add around the detected area

        Returns:
            tuple: (x, y, width, height) or None if no foreground detected
        """
        try:
            # Convert to binary mask for contour detection
            binary_mask = (mask > 0.1).astype(np.uint8)

            # Find all non-zero points
            coords = np.column_stack(np.where(binary_mask > 0))

            if len(coords) == 0:
                return None

            # Get bounding box coordinates
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)

            # Apply padding
            height, width = mask.shape
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(width - 1, x_max + padding)
            y_max = min(height - 1, y_max + padding)

            # Calculate width and height
            bbox_width = x_max - x_min + 1
            bbox_height = y_max - y_min + 1

            return (x_min, y_min, bbox_width, bbox_height)

        except Exception as e:
            self.log(f"Error calculating bounding box: {e}")
            return None

    def _is_mask_mostly_black(self, mask, threshold_percent=96.0):
        """Check if mask is mostly black (transparent)

        Args:
            mask: PIL Image in grayscale mode or numpy array
            threshold_percent: Percentage threshold for considering mask as mostly black

        Returns:
            bool: True if mask is mostly black (above threshold), False otherwise
        """
        try:
            # Convert PIL Image to numpy array if needed
            if isinstance(mask, Image.Image):
                mask_array = np.array(mask)
            else:
                mask_array = mask

            # Calculate the percentage of black (or near-black) pixels
            # Consider pixels with value <= 25 as black (to account for slight variations)
            black_pixels = np.sum(mask_array <= 25)
            total_pixels = mask_array.size
            black_percentage = (black_pixels / total_pixels) * 100

            self.log(f"Mask analysis: {black_percentage:.2f}% black pixels")
            return black_percentage >= threshold_percent

        except Exception as e:
            self.log(f"Error analyzing mask blackness: {e}")
            return False

    def process_image_file(
        self,
        input_path,
        output_path=None,
        foreground_threshold=240,
        background_threshold=15,
        apply_post_processing=True,
    ):
        """Process a single image file for background removal

        Args:
            input_path: Path to input image file
            output_path: Optional path for output. If None, overwrites input
            foreground_threshold: Pixels > this value (1-255) are considered foreground
            background_threshold: Pixels < this value (0-254) are considered background
            apply_post_processing: Whether to apply morphological post-processing

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            input_path = Path(input_path)
            if not input_path.exists():
                self.log(f"Input file does not exist: {input_path}")
                return False

            # Remove background
            rgba_result, mask, bounding_box = self.remove_background(
                input_path,
                foreground_threshold=foreground_threshold,
                background_threshold=background_threshold,
                apply_post_processing=apply_post_processing
            )

            if rgba_result is not None:
                # Check if mask is 96% black - if so, delete the image
                if mask is not None and self._is_mask_mostly_black(mask, threshold_percent=96.0):
                    self.log(f"Mask is 96% black, deleting image: {input_path.name}")
                    try:
                        input_path.unlink()  # Delete the original file
                        self.log(f"Successfully deleted {input_path.name}")
                        return True
                    except Exception as e:
                        self.log(f"Error deleting file {input_path.name}: {e}")
                        return False

                # Save the RGBA result with transparent background
                output_file = output_path if output_path else input_path
                rgba_result.save(str(output_file), 'PNG', optimize=True)
                self.log(f"Background removed for {input_path.name}")
                return True
            else:
                self.log(f"Background removal failed for {input_path.name}")
                return False

        except Exception as e:
            self.log(f"Error processing image file {input_path}: {e}")
            return False

    def _load_image_as_pil_rgb(self, image_input):
        """Load image input as PIL RGB Image"""
        try:
            if isinstance(image_input, str) or isinstance(image_input, Path):
                # Load from file path
                with Image.open(image_input) as img:
                    return img.convert('RGB')
            elif isinstance(image_input, Image.Image):
                # Convert to RGB if needed
                return image_input.convert('RGB')
            elif isinstance(image_input, np.ndarray):
                # Convert numpy array to PIL Image
                if len(image_input.shape) == 3:
                    if image_input.shape[2] == 3:  # BGR or RGB
                        # Assume BGR (OpenCV format) and convert to RGB
                        rgb_array = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
                        return Image.fromarray(rgb_array, mode='RGB')
                    elif image_input.shape[2] == 4:  # BGRA or RGBA
                        # Assume BGRA and convert to RGB
                        rgb_array = cv2.cvtColor(image_input, cv2.COLOR_BGRA2RGB)
                        return Image.fromarray(rgb_array, mode='RGB')
                elif len(image_input.shape) == 2:  # Grayscale
                    return Image.fromarray(image_input, mode='L').convert('RGB')
                else:
                    self.log(f"Unsupported numpy array shape: {image_input.shape}")
                    return None
            else:
                self.log(f"Unsupported image input type: {type(image_input)}")
                return None
        except Exception as e:
            self.log(f"Error loading image as PIL RGB: {e}")
            return None

    def _preprocess_image(self, rgb_image):
        """Preprocess RGB image for U2Net model input"""
        # Resize to model input size
        resized = rgb_image.resize((self.input_size, self.input_size), Image.Resampling.LANCZOS)

        # Convert to numpy array and normalize to [0, 1]
        img_array = np.array(resized, dtype=np.float32) / 255.0

        # Convert to CHW format and add batch dimension
        img_array = img_array.transpose(2, 0, 1)  # HWC to CHW
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        return img_array

    def _postprocess_mask(
        self, mask_output, target_width, target_height,
        foreground_threshold, background_threshold, apply_post_processing, center_x=None, center_y=None
    ):
        """Post-process the model output mask"""
        # Normalize mask to [0, 1]
        mask = (mask_output - mask_output.min()) / (mask_output.max() - mask_output.min() + 1e-8)

        # Resize back to original dimensions
        mask_pil = Image.fromarray((mask * 255).astype(np.uint8), mode='L')
        mask_pil = mask_pil.resize((target_width, target_height), Image.Resampling.LANCZOS)
        mask = np.array(mask_pil, dtype=np.float32) / 255.0

        # Apply thresholding for alpha matting
        # Normalize thresholds to [0, 1] range
        fg_thresh = foreground_threshold / 255.0
        bg_thresh = background_threshold / 255.0

        # Create alpha channel with smooth transitions
        alpha = np.zeros_like(mask)

        # Foreground (opaque)
        alpha[mask >= fg_thresh] = 1.0

        # Background (transparent)
        alpha[mask <= bg_thresh] = 0.0

        # Transition zone (semi-transparent)
        transition_mask = (mask > bg_thresh) & (mask < fg_thresh)
        if np.any(transition_mask):
            # Linear interpolation in transition zone
            transition_range = fg_thresh - bg_thresh
            alpha[transition_mask] = (mask[transition_mask] - bg_thresh) / transition_range

        # Apply morphological post-processing if requested
        if apply_post_processing:
            alpha = self._apply_morphological_operations(alpha)

        return alpha

    def _apply_morphological_operations(self, alpha):
        """Apply morphological operations to clean up the mask"""
        try:
            # Convert to uint8 for morphological operations
            alpha_uint8 = (alpha * 255).astype(np.uint8)

            # Small kernel for noise removal
            kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

            # Remove small noise
            alpha_uint8 = cv2.morphologyEx(alpha_uint8, cv2.MORPH_OPEN, kernel_small)

            # Fill small holes
            alpha_uint8 = cv2.morphologyEx(alpha_uint8, cv2.MORPH_CLOSE, kernel_small)

            # Slightly larger kernel for smoothing
            kernel_med = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

            # Smooth edges
            alpha_uint8 = cv2.morphologyEx(alpha_uint8, cv2.MORPH_CLOSE, kernel_med)

            # Apply Gaussian blur for smooth edges
            alpha_uint8 = cv2.GaussianBlur(alpha_uint8, (3, 3), 0)

            # Convert back to float32
            alpha = alpha_uint8.astype(np.float32) / 255.0

            return alpha

        except Exception as e:
            self.log(f"Error in morphological operations: {e}")
            return alpha  # Return original if post-processing fails

    def _create_rgba_result(self, rgb_image, alpha_mask):
        """Create RGBA image from RGB image and alpha mask"""
        try:
            # Convert RGB to numpy array
            rgb_array = np.array(rgb_image)

            # Ensure alpha mask has same spatial dimensions as RGB
            if alpha_mask.shape != rgb_array.shape[:2]:
                alpha_pil = Image.fromarray((alpha_mask * 255).astype(np.uint8), mode='L')
                alpha_pil = alpha_pil.resize(rgb_image.size, Image.Resampling.LANCZOS)
                alpha_mask = np.array(alpha_pil, dtype=np.float32) / 255.0

            # Create alpha channel (0-255)
            alpha_channel = (alpha_mask * 255).astype(np.uint8)

            # Combine RGB and alpha
            rgba_array = np.dstack([rgb_array, alpha_channel])

            # Create PIL Image
            rgba_result = Image.fromarray(rgba_array, mode='RGBA')

            return rgba_result

        except Exception as e:
            self.log(f"Error creating RGBA result: {e}")
            return None
