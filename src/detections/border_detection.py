"""
Border Detection Module
Handles automatic detection and cropping of image borders
"""
import cv2
import numpy as np
from PIL import Image
from pathlib import Path


class BorderDetection:
    """Border detection and cropping functionality"""

    def __init__(self, log_callback=None):
        """Initialize border detection

        Args:
            log_callback: Optional logging callback function
        """
        self.log = log_callback if log_callback else lambda x: print(x)

    def detect_and_crop_borders(self, image_input):
        """Detect and crop image borders

        Args:
            image_input: PIL Image, numpy array, or path to image file

        Returns:
            numpy.ndarray: Cropped image as BGR array, or None if failed
        """
        try:
            # Convert input to OpenCV format
            img = self._load_image_as_cv2(image_input)
            if img is None:
                return None

            # Convert to uint8 if needed
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)

            # Try normal image
            result = self._crop_to_content(self._detect_borders(img), 98, img)

            # Try inverted image
            img_inverted = cv2.bitwise_not(img)
            result_inverted = self._crop_to_content(self._detect_borders(img_inverted), 98, img)

            # Choose the smaller result (more aggressive cropping)
            if result.size > result_inverted.size:
                result = result_inverted

            return result

        except Exception as e:
            self.log(f"Error in border detection: {e}")
            return None

    def process_image_file(self, input_path, output_path=None):
        """Process a single image file for border detection and cropping

        Args:
            input_path: Path to input image file
            output_path: Optional path for output. If None, overwrites input

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            input_path = Path(input_path)
            if not input_path.exists():
                self.log(f"Input file does not exist: {input_path}")
                return False

            # Load and process image
            img = cv2.imread(str(input_path))
            if img is None:
                self.log(f"Could not load image: {input_path}")
                return False

            original_size = img.shape[:2]  # (height, width)

            # Apply border detection and cropping
            cropped_img = self.detect_and_crop_borders(img)

            if cropped_img is not None:
                new_size = cropped_img.shape[:2]

                # Only save if the image was actually cropped (size changed)
                if new_size != original_size:
                    output_file = output_path if output_path else input_path
                    cv2.imwrite(str(output_file), cropped_img)
                    self.log(f"Border cropped {input_path.name}: {original_size[1]}x{original_size[0]} → {new_size[1]}x{new_size[0]}")
                    return True
                else:
                    self.log(f"No border cropping needed for {input_path.name}")
                    return True
            else:
                self.log(f"Border detection failed for {input_path.name}")
                return False

        except Exception as e:
            self.log(f"Error processing image file {input_path}: {e}")
            return False

    def _load_image_as_cv2(self, image_input):
        """Load image input as OpenCV BGR array"""
        try:
            if isinstance(image_input, str) or isinstance(image_input, Path):
                # Load from file path
                return cv2.imread(str(image_input))
            elif isinstance(image_input, Image.Image):
                # Convert PIL Image to OpenCV
                if image_input.mode == 'RGB':
                    return cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
                elif image_input.mode == 'RGBA':
                    return cv2.cvtColor(np.array(image_input), cv2.COLOR_RGBA2BGR)
                else:
                    # Convert to RGB first
                    rgb_img = image_input.convert('RGB')
                    return cv2.cvtColor(np.array(rgb_img), cv2.COLOR_RGB2BGR)
            elif isinstance(image_input, np.ndarray):
                # Assume it's already in the right format
                return image_input
            else:
                self.log(f"Unsupported image input type: {type(image_input)}")
                return None
        except Exception as e:
            self.log(f"Error loading image: {e}")
            return None

    def _detect_borders(self, img):
        """Detect borders using Otsu thresholding and morphological operations"""
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Threshold using Otsu method
            ret, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Morphological operations to refine the mask
            kernel = np.ones((9, 9), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            # Create result with alpha channel
            result = img.copy()
            result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
            result[:, :, 3] = mask

            return result

        except Exception as e:
            self.log(f"Error in border detection algorithm: {e}")
            return None

    def _crop_to_content(self, img, thresh_val, img_orig=None):
        """Crop image to content based on alpha channel threshold"""
        try:
            if img is None or len(img.shape) < 3 or img.shape[2] < 4:
                return img_orig if img_orig is not None else img

            # Threshold value processing
            thresh_val = min(thresh_val / 100, 0.99999)

            # Get alpha channel
            alpha = img[:, :, 3]

            # Find rows and columns with valid alpha
            r = np.any(alpha > thresh_val, 1)
            if not r.any():
                return img_orig if img_orig is not None else img

            c = np.any(alpha > thresh_val, 0)
            if not c.any():
                return img_orig if img_orig is not None else img

            h, w = alpha.shape

            # Calculate crop boundaries
            top = r.argmax()
            bottom = h - r[::-1].argmax()
            left = c.argmax()
            right = w - c[::-1].argmax()

            # Use original image for cropping if available
            source_img = img_orig if img_orig is not None else img[:, :, :3]

            # Perform the crop
            cropped = source_img[top:bottom, left:right]

            return cropped

        except Exception as e:
            self.log(f"Error in crop to content: {e}")
            return img_orig if img_orig is not None else img
