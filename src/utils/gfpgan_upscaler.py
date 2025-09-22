"""
GFPGAN-based image upscaling utility
"""
import torch
import cv2
from pathlib import Path
from typing import Optional, Union, Tuple
import logging

# Monkey patch for torchvision compatibility with GFPGAN
def _apply_torchvision_compatibility_patch():
    """Apply monkey patch to fix torchvision.transforms.functional_tensor compatibility"""
    try:
        import sys
        import torchvision.transforms.functional as F

        # Check if the problematic module already exists
        if 'torchvision.transforms.functional_tensor' in sys.modules:
            return True

        # Create a mock functional_tensor module that aliases to functional
        class _FunctionalTensorMock:
            """Mock module to provide backward compatibility for torchvision.transforms.functional_tensor"""

            @staticmethod
            def rgb_to_grayscale(img, num_output_channels=1):
                return F.rgb_to_grayscale(img, num_output_channels)

            @staticmethod
            def normalize(tensor, mean, std, inplace=False):
                return F.normalize(tensor, mean, std, inplace)

            @staticmethod
            def resize(img, size, interpolation=2, max_size=None, antialias=None):
                return F.resize(img, size, interpolation, max_size, antialias)

            @staticmethod
            def center_crop(img, output_size):
                return F.center_crop(img, output_size)

            @staticmethod
            def crop(img, top, left, height, width):
                return F.crop(img, top, left, height, width)

            @staticmethod
            def to_tensor(pic):
                return F.to_tensor(pic)

            @staticmethod
            def to_pil_image(pic, mode=None):
                return F.to_pil_image(pic, mode)

            # Add any other functions that might be needed
            def __getattr__(self, name):
                # Fallback: if function exists in functional, use it
                if hasattr(F, name):
                    return getattr(F, name)
                raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

        # Inject the mock module into sys.modules
        sys.modules['torchvision.transforms.functional_tensor'] = _FunctionalTensorMock()
        return True

    except Exception as e:
        # If patching fails, log the error but don't crash
        logging.getLogger(__name__).warning(f"Failed to apply torchvision compatibility patch: {e}")
        return False

# Apply the compatibility patch before importing GFPGAN
_apply_torchvision_compatibility_patch()

# Try to import GFPGAN with better error handling
try:
    # First try basic import
    import gfpgan
    from gfpgan import GFPGANer
    GFPGAN_AVAILABLE = True
    GFPGAN_ERROR = None
except ImportError as e:
    # If basic import fails, capture the error
    gfpgan = None
    GFPGANer = None
    GFPGAN_AVAILABLE = False
    GFPGAN_ERROR = str(e)
except Exception as e:
    # If any other error occurs during import (like the torchvision issue)
    gfpgan = None
    GFPGANer = None
    GFPGAN_AVAILABLE = False
    GFPGAN_ERROR = f"GFPGAN import failed due to dependency issue: {str(e)}"

from ..models.model_manager import ModelManager


class GFPGANUpscaler:
    """GFPGAN-based image upscaler for face restoration and enhancement."""

    def __init__(self, model_manager: ModelManager, scale_factor: float = 4.0, quality_preset: str = "balanced"):
        """Initialize the GFPGAN upscaler.

        Args:
            model_manager: Model manager instance for loading GFPGAN model
            scale_factor: Upscaling factor (2.0, 4.0, etc.)
            quality_preset: Quality preset - "fast", "balanced", or "quality"
        """
        self.logger = logging.getLogger(__name__)
        self.model_manager = model_manager
        self.scale_factor = scale_factor
        self.quality_preset = quality_preset
        self.gfpgan = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Quality presets - using only parameters supported by GFPGAN 1.3.8
        self.quality_settings = {
            "fast": {},  # Use default settings for fast mode
            "balanced": {},  # Use default settings for balanced mode
            "quality": {}  # Use default settings for quality mode
        }

    def _initialize_model(self) -> bool:
        """Initialize the GFPGAN model."""
        if not GFPGAN_AVAILABLE or GFPGANer is None:
            if GFPGAN_ERROR:
                self.logger.error(f"GFPGAN is not available: {GFPGAN_ERROR}")
                if "torchvision.transforms.functional_tensor" in GFPGAN_ERROR:
                    self.logger.error("This is a known compatibility issue with newer torchvision versions.")
                elif "basicsr" in GFPGAN_ERROR:
                    self.logger.error("BasicSR dependency issue detected.")
                    self.logger.error("Try reinstalling: pip uninstall gfpgan basicsr && pip install gfpgan")
            else:
                self.logger.error("GFPGAN is not available. Please install gfpgan package: pip install gfpgan")
            return False

        # Set up proper directory for GFPGAN auxiliary models
        self._setup_gfpgan_directories()

        # Get GFPGAN model path
        model_path = self.model_manager.get_model_path("gfpgan")
        if not model_path or not model_path.exists():
            self.logger.error("GFPGAN model not found. Please download the model first.")
            return False

        try:
            # Get quality settings
            settings = self.quality_settings.get(self.quality_preset, self.quality_settings["balanced"])

            # Initialize GFPGAN
            self.gfpgan = GFPGANer(
                model_path=str(model_path),
                upscale=self.scale_factor,
                arch='clean',
                channel_multiplier=2,
                bg_upsampler=None,  # We can add Real-ESRGAN later if needed
                device=self.device,
                **settings
            )

            self.logger.info(f"GFPGAN initialized successfully on {self.device}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize GFPGAN: {e}")
            return False

    def _setup_gfpgan_directories(self):
        """Set up proper directories for GFPGAN auxiliary models"""
        try:
            import os

            # Get the project root from model manager
            project_root = self.model_manager.project_root if hasattr(self.model_manager, 'project_root') else Path.cwd()

            # Create models/gfpgan_weights directory for auxiliary models
            gfpgan_weights_dir = project_root / "models" / "gfpgan_weights"
            gfpgan_weights_dir.mkdir(parents=True, exist_ok=True)

            # Set environment variables to redirect GFPGAN auxiliary model downloads
            # These environment variables control where facexlib downloads its models
            os.environ['FACEXLIB_CACHE_DIR'] = str(gfpgan_weights_dir)
            os.environ['TORCH_HOME'] = str(project_root / "models" / "torch_cache")

            # Also try setting the BasicSR cache directory
            os.environ['BASICSR_CACHE_DIR'] = str(project_root / "models" / "basicsr_cache")

            # Create torch cache directory
            torch_cache_dir = project_root / "models" / "torch_cache"
            torch_cache_dir.mkdir(parents=True, exist_ok=True)

            # Create basicsr cache directory
            basicsr_cache_dir = project_root / "models" / "basicsr_cache"
            basicsr_cache_dir.mkdir(parents=True, exist_ok=True)

            # Monkey patch facexlib's download function to use our directory
            self._patch_facexlib_download(gfpgan_weights_dir)

            self.logger.info(f"GFPGAN auxiliary models will be cached in: {gfpgan_weights_dir}")

        except Exception as e:
            self.logger.warning(f"Failed to setup GFPGAN directories: {e}")
            # Continue anyway - this is not critical for functionality

    def _patch_facexlib_download(self, target_dir: Path):
        """Monkey patch facexlib's load_file_from_url function to use our target directory"""
        try:
            import facexlib.utils.misc as facexlib_misc
            from urllib.parse import urlparse
            import os

            # Store original function
            original_load_file_from_url = facexlib_misc.load_file_from_url

            def patched_load_file_from_url(url, model_dir=None, progress=True, file_name=None, save_dir=None):
                """Patched version that downloads to our models directory"""
                # Always use our target directory
                save_dir = str(target_dir)
                os.makedirs(save_dir, exist_ok=True)

                # Get filename from URL or use provided file_name
                parts = urlparse(url)
                filename = os.path.basename(parts.path)
                if file_name is not None:
                    filename = file_name

                cached_file = os.path.abspath(os.path.join(save_dir, filename))

                # Only download if file doesn't exist
                if not os.path.exists(cached_file):
                    print(f'Downloading: "{url}" to {cached_file}\n')
                    # Import download function from the same module
                    from torch.hub import download_url_to_file
                    download_url_to_file(url, cached_file, hash_prefix=None, progress=progress)

                return cached_file

            # Replace the function in the module
            facexlib_misc.load_file_from_url = patched_load_file_from_url

            self.logger.info("Successfully patched facexlib download function")

        except Exception as e:
            self.logger.warning(f"Failed to patch facexlib download function: {e}")
            # Continue anyway - this is not critical for functionality

    def upscale_image(self, input_path: Union[str, Path], output_path: Union[str, Path]) -> bool:
        """Upscale a single image using GFPGAN.

        Args:
            input_path: Path to input image
            output_path: Path to save upscaled image

        Returns:
            True if successful, False otherwise
        """
        if self.gfpgan is None:
            if not self._initialize_model():
                return False

        try:
            input_path = Path(input_path)
            output_path = Path(output_path)

            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Read input image
            input_img = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
            if input_img is None:
                self.logger.error(f"Failed to load image: {input_path}")
                return False

            # Apply GFPGAN enhancement
            cropped_faces, restored_faces, restored_img = self.gfpgan.enhance(
                input_img,
                has_aligned=False,
                only_center_face=False,
                paste_back=True
            )

            # Use restored image if available, otherwise use original upscaled
            if restored_img is not None:
                result_img = restored_img
            else:
                # Fallback to simple upscaling if GFPGAN fails
                height, width = input_img.shape[:2]
                new_height = int(height * self.scale_factor)
                new_width = int(width * self.scale_factor)
                result_img = cv2.resize(input_img, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)

            # Save result
            success = cv2.imwrite(str(output_path), result_img)
            if success:
                self.logger.debug(f"Successfully upscaled: {input_path.name} -> {output_path.name}")
                return True
            else:
                self.logger.error(f"Failed to save upscaled image: {output_path}")
                return False

        except Exception as e:
            self.logger.error(f"Error upscaling image {input_path}: {e}")
            return False

    def upscale_batch(self, input_dir: Union[str, Path], output_dir: Union[str, Path],
                     progress_callback: Optional[callable] = None) -> Tuple[int, int]:
        """Upscale all images in a directory.

        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save upscaled images
            progress_callback: Optional callback for progress updates (current, total, filename)

        Returns:
            Tuple of (successful_count, total_count)
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)

        if not input_dir.exists():
            self.logger.error(f"Input directory does not exist: {input_dir}")
            return 0, 0

        # Get list of image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp']
        image_files = []

        for ext in image_extensions:
            image_files.extend(input_dir.glob(f"*{ext}"))
            image_files.extend(input_dir.glob(f"*{ext.upper()}"))

        total_count = len(image_files)
        successful_count = 0

        if total_count == 0:
            self.logger.warning(f"No image files found in: {input_dir}")
            return 0, 0

        self.logger.info(f"Starting batch upscaling of {total_count} images")

        for i, input_file in enumerate(image_files):
            # Determine output filename (keep original extension)
            output_file = output_dir / input_file.name

            # Skip if output already exists
            if output_file.exists():
                self.logger.debug(f"Skipping existing file: {output_file.name}")
                successful_count += 1
                if progress_callback:
                    progress_callback(i + 1, total_count, input_file.name)
                continue

            # Upscale the image
            if self.upscale_image(input_file, output_file):
                successful_count += 1

            # Update progress
            if progress_callback:
                progress_callback(i + 1, total_count, input_file.name)

        self.logger.info(f"Batch upscaling completed: {successful_count}/{total_count} successful")
        return successful_count, total_count

    def cleanup(self):
        """Clean up resources."""
        if self.gfpgan is not None:
            # Clear CUDA cache if using GPU
            if self.device == "cuda":
                torch.cuda.empty_cache()
            self.gfpgan = None
            self.logger.debug("GFPGAN resources cleaned up")

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()
