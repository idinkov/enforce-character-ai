"""
Face processing utilities for cropping and saving face images.
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from PIL import Image
import math


class FaceProcessor:
    """Utility class for processing face images."""

    @staticmethod
    def calculate_square_crop_box(face: Dict, image_size: Tuple[int, int], max_size: int = 1024) -> Dict:
        """
        Calculate a square crop box around a detected face.

        Args:
            face: Face dictionary with x, y, width, height
            image_size: Tuple of (width, height) of the original image
            max_size: Maximum size for the square crop (default 1024)

        Returns:
            Dictionary with crop coordinates and size
        """
        img_width, img_height = image_size

        # Get face center
        face_center_x = face['x'] + face['width'] // 2
        face_center_y = face['y'] + face['height'] // 2

        # Calculate the size of the square crop
        # Use the larger dimension of the face and add some padding
        face_size = max(face['width'], face['height'])

        # Add padding (50% extra space around the face)
        padding_factor = 1.5
        crop_size = int(face_size * padding_factor)

        # Ensure crop size doesn't exceed max_size
        crop_size = min(crop_size, max_size)

        # Calculate crop coordinates (centered on face)
        half_crop = crop_size // 2
        crop_x = face_center_x - half_crop
        crop_y = face_center_y - half_crop

        # Adjust crop coordinates to stay within image bounds
        if crop_x < 0:
            crop_x = 0
        if crop_y < 0:
            crop_y = 0
        if crop_x + crop_size > img_width:
            crop_x = img_width - crop_size
        if crop_y + crop_size > img_height:
            crop_y = img_height - crop_size

        # If the image is smaller than the crop size, adjust crop size
        if crop_size > img_width or crop_size > img_height:
            crop_size = min(img_width, img_height)
            half_crop = crop_size // 2
            crop_x = max(0, face_center_x - half_crop)
            crop_y = max(0, face_center_y - half_crop)

            # Ensure we don't go out of bounds
            if crop_x + crop_size > img_width:
                crop_x = img_width - crop_size
            if crop_y + crop_size > img_height:
                crop_y = img_height - crop_size

        return {
            'x': max(0, crop_x),
            'y': max(0, crop_y),
            'size': crop_size,
            'x2': min(img_width, crop_x + crop_size),
            'y2': min(img_height, crop_y + crop_size)
        }

    @staticmethod
    def crop_and_save_face(image_path: Path, face: Dict, output_path: Path, max_size: int = 1024, status_callback=None) -> bool:
        """
        Crop a face from an image and save it as a square image.

        Args:
            image_path: Path to the source image
            face: Face dictionary with bounding box information
            output_path: Path where to save the cropped face
            max_size: Maximum size for the output image
            status_callback: Optional callback function for status updates

        Returns:
            True if successful, False otherwise
        """
        try:
            if status_callback:
                status_callback("Loading source image...")

            # Load the image
            with Image.open(image_path) as img:
                if status_callback:
                    status_callback("Calculating crop area...")

                # Calculate crop box
                crop_info = FaceProcessor.calculate_square_crop_box(face, img.size, max_size)

                if status_callback:
                    status_callback("Cropping face from image...")

                # Crop the image
                crop_box = (crop_info['x'], crop_info['y'], crop_info['x2'], crop_info['y2'])
                cropped_img = img.crop(crop_box)

                # Make it square by padding if needed
                width, height = cropped_img.size
                if width != height:
                    if status_callback:
                        status_callback("Making image square...")

                    # Create a square image by padding with the background color
                    square_size = max(width, height)
                    square_img = Image.new('RGB', (square_size, square_size), (255, 255, 255))

                    # Paste the cropped image in the center
                    offset_x = (square_size - width) // 2
                    offset_y = (square_size - height) // 2
                    square_img.paste(cropped_img, (offset_x, offset_y))
                    cropped_img = square_img

                # Resize if larger than max_size
                if cropped_img.size[0] > max_size:
                    if status_callback:
                        status_callback(f"Resizing image to {max_size}x{max_size}...")
                    cropped_img = cropped_img.resize((max_size, max_size), Image.Resampling.LANCZOS)

                # Ensure output directory exists
                output_path.parent.mkdir(parents=True, exist_ok=True)

                if status_callback:
                    status_callback(f"Saving face image to {output_path.name}...")

                # Save the image
                cropped_img.save(output_path, 'PNG', quality=95)

                if status_callback:
                    final_size = cropped_img.size
                    status_callback(f"Face image saved successfully ({final_size[0]}x{final_size[1]})")

                return True

        except Exception as e:
            error_msg = f"Error cropping and saving face: {e}"
            if status_callback:
                status_callback(error_msg)
            print(error_msg)
            return False

    @staticmethod
    def detect_faces_in_image(image_path: Path, character_path: Path, status_callback=None) -> List[Dict]:
        """
        Detect faces in an image using the existing face detection system.

        Args:
            image_path: Path to the image to analyze
            character_path: Path to the character directory
            status_callback: Optional callback function for status updates

        Returns:
            List of face dictionaries with bounding box information
        """
        try:
            from src.detections.face_detection import FaceDetection

            if status_callback:
                status_callback("Initializing face detection...")

            # Initialize face detection
            face_detector = FaceDetection(character_path, log_callback=status_callback)

            if status_callback:
                status_callback("Analyzing image for faces...")

            # Detect faces
            faces = face_detector.detect_faces_in_image(image_path)

            if status_callback:
                if faces:
                    status_callback(f"Found {len(faces)} face(s) in the image")
                else:
                    status_callback("No faces detected in the image")

            return faces

        except Exception as e:
            error_msg = f"Error detecting faces: {e}"
            if status_callback:
                status_callback(error_msg)
            print(error_msg)
            return []

    @staticmethod
    def update_character_face_image(character_repo, character_name: str, face_image_path: Path, status_callback=None) -> bool:
        """
        Update the character's face_image field in character.yaml.

        Args:
            character_repo: Character repository instance
            character_name: Name of the character
            face_image_path: Path to the new face image (relative to character directory)
            status_callback: Optional callback function for status updates

        Returns:
            True if successful, False otherwise
        """
        try:
            if status_callback:
                status_callback("Loading character data...")

            # Load character data
            character = character_repo.load_character(character_name)
            if not character:
                error_msg = f"Character '{character_name}' not found"
                if status_callback:
                    status_callback(error_msg)
                return False

            if status_callback:
                status_callback("Updating character face image reference...")

            # Update face image path (use relative path)
            character.face_image = str(face_image_path.name)

            if status_callback:
                status_callback("Saving character configuration...")

            # Save character data (only pass the character object)
            success = character_repo.save_character(character)

            if success and status_callback:
                status_callback("Character configuration updated successfully")
            elif not success and status_callback:
                status_callback("Failed to save character configuration")

            return success

        except Exception as e:
            error_msg = f"Error updating character face image: {e}"
            if status_callback:
                status_callback(error_msg)
            print(error_msg)
            return False
