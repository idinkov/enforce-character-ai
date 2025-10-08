"""
Application configuration management.
"""
from dataclasses import dataclass
from typing import List, Tuple


# Directory name constants
PROVIDER_IMPORTS_DIR = "0_providers"


@dataclass
class AppConfig:
    """Application configuration settings."""

    # Application info
    APP_NAME: str = "Enforce Character AI"
    APP_VERSION: str = "0.1.2"

    # Window settings
    WINDOW_TITLE: str = "Enforce Character AI"
    WINDOW_GEOMETRY: str = "1200x1000"

    # Image settings - optimized for performance and visibility
    IMAGE_EXTENSIONS: List[str] = None
    THUMBNAIL_SIZE: Tuple[int, int] = (200, 200)  # Increased from 120x120 for better visibility
    IMAGE_WIDGET_WIDTH: int = 220  # Increased to accommodate larger thumbnails
    MIN_GRID_COLUMNS: int = 2  # Keep minimum for better space usage

    # Stage settings
    STAGE_FOLDERS: List[str] = None
    STAGES: List[Tuple[str, str, str]] = None

    # Processing settings
    STAGE_PROCESSING: dict = None
    STAGE_NAME_MAPPINGS: dict = None

    # Upscaling settings
    AUTOMATIC_UPSCALING: bool = False  # Enable automatic upscaling with GFPGAN
    UPSCALING_SCALE_FACTOR: float = 4.0  # Scale factor for upscaling (1x, 2x, 4x, etc.)
    UPSCALING_QUALITY_PRESET: str = "balanced"  # "fast", "balanced", "quality"

    # UI settings
    FACE_IMAGE_SIZE: Tuple[int, int] = (64, 64)
    PROGRESS_BAR_LENGTH: int = 150
    DEBOUNCE_DELAY_MS: int = 50  # Faster response for better UX

    # Performance optimization settings
    THUMBNAIL_CACHE_SIZE: int = 20000  # Number of thumbnails to cache
    THUMBNAIL_CACHE_MEMORY_MB: int = 8000  # Memory limit for thumbnail cache
    FACE_CACHE_SIZE: int = 100  # Number of face thumbnails to cache
    FACE_CACHE_MEMORY_MB: int = 20  # Memory limit for face cache
    THUMBNAIL_WORKER_THREADS: int = 12  # Number of background threads for thumbnail loading

    # Threading settings
    FILENAME_NORMALIZATION_THREADS: int = 1  # Number of threads for filename normalization

    def __post_init__(self):
        """Initialize default values that depend on other values."""
        if self.IMAGE_EXTENSIONS is None:
            self.IMAGE_EXTENSIONS = [
                '*.jpg', '*.jpeg', '*.png', '*.bmp',
                '*.gif', '*.tiff', '*.webp'
            ]

        if self.STAGE_FOLDERS is None:
            self.STAGE_FOLDERS = [
                PROVIDER_IMPORTS_DIR, "1_raw", "2_raw_filtered",
                "3_raw_upscaled", "4_processed_1024",
                "5_processed_fixed_1024", "6_rtt_1024", "7_final_dataset",
                "8_creations"
            ]

        if self.STAGES is None:
            self.STAGES = [
                (PROVIDER_IMPORTS_DIR, "0", "Provider Imports"),
                ("1_raw", "1", "Raw Images"),
                ("2_raw_filtered", "2", "Filtered Images (PNG)"),
                ("3_raw_upscaled", "3", "Upscaled Images"),
                ("4_processed_1024", "4", "Processed 1024x1024"),
                ("5_processed_fixed_1024", "5", "Fixed & Inpainted"),
                ("6_rtt_1024", "6", "Ready-to-Train"),
                ("7_final_dataset", "7", "Final Dataset"),
                ("8_creations", "8", "AI Creations")
            ]

        if self.STAGE_PROCESSING is None:
            self.STAGE_PROCESSING = {
                PROVIDER_IMPORTS_DIR: (
                    "0_to_1", "Copy to Raw (0→1)", "1_raw"
                ),
                "1_raw": (
                    "1_to_2", "Clean & Filter (1→2)", "2_raw_filtered"
                ),
                "2_raw_filtered": (
                    "2_to_3", "Upscale (2→3)", "3_raw_upscaled"
                ),
                "3_raw_upscaled": (
                    "3_to_4", "Resize to 1024px (3→4)", "4_processed_1024"
                ),
                "4_processed_1024": (
                    "4_to_5", "AI Inpaint (4→5)", "5_processed_fixed_1024"
                ),
                "5_processed_fixed_1024": (
                    "5_to_6", "Sort by Face Count (5→6)", "6_rtt_1024"
                ),
                "6_rtt_1024": (
                    "6_to_7", "Create Dataset (6→7)", "7_final_dataset"
                ),
                "7_final_dataset": (None, None, None),
                "8_creations": (None, None, None)
            }

        if self.STAGE_NAME_MAPPINGS is None:
            self.STAGE_NAME_MAPPINGS = {
                f"{PROVIDER_IMPORTS_DIR}_to_1_raw": "Copy to Raw (0→1)",
                "1_raw_to_2_raw_filtered": "Clean & Filter (1→2)",
                "2_raw_filtered_to_3_raw_upscaled": "Upscale (2→3)",
                "3_raw_upscaled_to_4_processed_1024": (
                    "Resize to 1024px (3→4)"
                ),
                "4_processed_1024_to_5_processed_fixed_1024": (
                    "AI Inpaint (4→5)"
                ),
                "5_processed_fixed_1024_to_6_rtt_1024": (
                    "Sort by Face Count (5→6)"
                ),
                "6_rtt_1024_to_7_final_dataset": (
                    "Create Dataset (6→7)"
                )
            }

    def get_processor_keys_in_order(self):
        """Get processor keys in the correct processing order."""
        processor_keys = []
        for stage_folder in self.STAGE_FOLDERS:
            if stage_folder in self.STAGE_PROCESSING:
                processor_key, _, _ = self.STAGE_PROCESSING[stage_folder]
                if processor_key:  # Skip stages without processors (like final stage)
                    processor_keys.append(processor_key)
        return processor_keys

    def get_all_processor_keys(self):
        """Get all valid processor keys from configuration."""
        return [key for key, (processor_key, _, _) in self.STAGE_PROCESSING.items()
                if processor_key is not None]


# Global configuration instance
config = AppConfig()
