import yaml
from pathlib import Path
import numpy as np
from facexlib.utils.face_restoration_helper import FaceRestoreHelper
import face_recognition
from ..services.gpu_service import get_gpu_service

class FaceDetection:
    """Face detection class using facexlib library for detecting faces in character images."""

    def __init__(self, character_path, log_callback=None):
        """
        Initialize FaceDetection with character path and optional logging callback.

        Args:
            character_path (Path or str): Path to the character directory
            log_callback (callable, optional): Callback function for logging messages
        """
        self.character_path = Path(character_path)
        self.log_callback = log_callback
        self.gpu_service = get_gpu_service()

        # Initialize face helper
        self.face_helper = None
        self._init_face_helper()

        # Initialize reference face encoding for character
        self.reference_face_encoding = None
        self._load_reference_face()

    def _init_face_helper(self):
        """Initialize the face helper from facexlib using selected GPU."""
        try:
            # Get device from GPU service
            device_info = self.gpu_service.get_selected_device()
            if device_info['type'] == 'gpu':
                device = f"cuda:{device_info['index']}"
                self.log(f"Attempting to initialize face helper with GPU {device_info['index']}: {device_info['name']}")
            else:
                device = 'cpu'
                self.log("Attempting to initialize face helper with CPU...")

            try:
                self.face_helper = FaceRestoreHelper(
                    upscale_factor=1,
                    face_size=512,
                    crop_ratio=(1, 1),
                    det_model='retinaface_resnet50',
                    device=device
                )
                self.log(f"Face helper initialized successfully with {device}")
            except Exception as gpu_error:
                self.log(f"Device initialization failed: {str(gpu_error)}")
                if device != 'cpu':
                    self.log("Falling back to CPU...")
                    device = 'cpu'
                    self.face_helper = FaceRestoreHelper(
                        upscale_factor=1,
                        face_size=512,
                        crop_ratio=(1, 1),
                        det_model='retinaface_resnet50',
                        device=device
                    )
                    self.log("Face helper initialized successfully with CPU")
                else:
                    raise

        except Exception as e:
            self.log(f"Error initializing face helper: {str(e)}")
            self.face_helper = None

    def reinitialize_with_selected_gpu(self):
        """Reinitialize face helper with currently selected GPU."""
        self._init_face_helper()

    def _load_reference_face(self):
        """Load and encode the character's reference face image."""
        try:
            # Load character data to get face_image path
            character_yaml_path = self.character_path / "character.yaml"
            if not character_yaml_path.exists():
                return

            with open(character_yaml_path, 'r', encoding='utf-8') as f:
                character_data = yaml.safe_load(f) or {}

            face_image_path = character_data.get('face_image', '')
            if not face_image_path:
                self.log("No face_image specified for character")
                return

            face_image_path = Path(face_image_path)
            if not face_image_path.exists():
                self.log(f"Reference face image not found: {face_image_path}")
                return

            # Load and encode the reference face
            self.log(f"Loading reference face from: {face_image_path}")
            reference_image = face_recognition.load_image_file(str(face_image_path))
            face_encodings = face_recognition.face_encodings(reference_image)

            if face_encodings:
                self.reference_face_encoding = face_encodings[0]
                self.log("Reference face encoding loaded successfully")
            else:
                self.log("No face found in reference image")

        except Exception as e:
            self.log(f"Error loading reference face: {str(e)}")
            self.reference_face_encoding = None

    def compare_face_with_reference(self, image_path, face_box):
        """
        Compare a detected face with the reference face image.

        Args:
            image_path (Path): Path to the image containing the face
            face_box (dict): Face bounding box with x, y, width, height

        Returns:
            float: Similarity score (0.0 to 1.0), or 0.0 if comparison fails
        """
        if self.reference_face_encoding is None:
            return 0.0

        try:
            # Load the image
            image = face_recognition.load_image_file(str(image_path))

            # Convert face_box to face_recognition format (top, right, bottom, left)
            x, y, width, height = face_box['x'], face_box['y'], face_box['width'], face_box['height']
            face_location = (y, x + width, y + height, x)

            # Get face encoding for the detected face
            face_encoding = face_recognition.face_encodings(image, [face_location])

            if not face_encoding:
                return 0.0

            # Calculate face distance (lower is more similar)
            face_distances = face_recognition.face_distance([self.reference_face_encoding], face_encoding[0])

            # Convert distance to similarity score (0.0 to 1.0)
            # face_recognition typically considers distances < 0.6 as matches
            # We'll convert to a 0-1 similarity score where 1.0 is perfect match
            similarity = max(0.0, 1.0 - face_distances[0])

            # Convert numpy scalar to regular Python float to avoid YAML serialization issues
            return float(similarity)

        except Exception as e:
            self.log(f"Error comparing face: {str(e)}")
            return 0.0

    def log(self, message):
        """Log a message using the callback if available."""
        if self.log_callback:
            self.log_callback(message)
        else:
            print(message)

    def detect_faces_in_image(self, image_path):
        """
        Detect faces in a single image and return bounding boxes.

        Args:
            image_path (Path or str): Path to the image file

        Returns:
            list: List of face bounding boxes as [x, y, width, height]
        """
        if not self.face_helper:
            self.log("Face helper not initialized")
            return []

        try:
            # Load image using face_helper
            self.face_helper.read_image(str(image_path))

            # Get face landmarks - this will detect faces and store them in face_helper
            self.face_helper.get_face_landmarks_5(
                only_center_face=False,
                eye_dist_threshold=5
            )

            # Extract bounding boxes from detected faces
            face_boxes = []
            if hasattr(self.face_helper, 'all_landmarks_5') and self.face_helper.all_landmarks_5:
                for i, landmarks in enumerate(self.face_helper.all_landmarks_5):
                    # Calculate bounding box from landmarks
                    landmarks_np = np.array(landmarks)
                    x_min = int(np.min(landmarks_np[:, 0]))
                    y_min = int(np.min(landmarks_np[:, 1]))
                    x_max = int(np.max(landmarks_np[:, 0]))
                    y_max = int(np.max(landmarks_np[:, 1]))

                    # Add some padding to the bounding box
                    padding = 20
                    x_min = max(0, x_min - padding)
                    y_min = max(0, y_min - padding)
                    width = x_max - x_min + padding * 2
                    height = y_max - y_min + padding * 2

                    # Calculate center coordinates
                    center_x = x_min + width // 2
                    center_y = y_min + height // 2

                    face_boxes.append({
                        'x': x_min,
                        'y': y_min,
                        'width': width,
                        'height': height,
                        'center_x': center_x,
                        'center_y': center_y,
                        'confidence': 1.0,  # facexlib doesn't provide confidence scores directly
                        'landmarks': landmarks_np  # Include landmarks for face rotation
                    })

            # Clean up for next image
            self.face_helper.clean_all()

            return face_boxes

        except Exception as e:
            self.log(f"Error detecting faces in {image_path}: {str(e)}")
            # Clean up on error
            if self.face_helper:
                self.face_helper.clean_all()
            return []