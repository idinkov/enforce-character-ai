"""
Model Manager for handling AI model loading, downloading, and management.
"""
import requests
from pathlib import Path
from typing import Dict, Optional, Union, Any
import yaml
import logging
import shutil
from dataclasses import dataclass

# Optional imports
try:
    from huggingface_hub import hf_hub_download
    HF_HUB_AVAILABLE = True
except ImportError:
    hf_hub_download = None
    HF_HUB_AVAILABLE = False


@dataclass
class ModelInfo:
    """Information about a model."""
    name: str
    version: str
    filename: str
    download_url: str
    local_path: str
    file_size_mb: float
    description: str
    model_type: str
    framework: str
    required: bool
    is_trainable: bool = False
    train_suffix: Optional[str] = None
    # New fields for Hugging Face support
    huggingface_repo: Optional[str] = None
    huggingface_filename: Optional[str] = None
    download_method: str = "url"  # "url" or "huggingface"
    # Add project root reference
    _project_root: Optional[Path] = None

    def set_project_root(self, project_root: Path):
        """Set the project root directory for path resolution."""
        self._project_root = project_root

    @property
    def full_path(self) -> Path:
        """Get the full path to the model file."""
        if self._project_root:
            # Resolve path relative to project root
            return self._project_root / self.local_path
        else:
            # Fallback to relative path
            return Path(self.local_path)

    @property
    def exists(self) -> bool:
        """Check if the model file exists."""
        return self.full_path.exists()

    @property
    def size_bytes(self) -> int:
        """Get file size in bytes."""
        return int(self.file_size_mb * 1024 * 1024)

    @property
    def is_huggingface_model(self) -> bool:
        """Check if this is a Hugging Face model."""
        return self.download_method == "huggingface" and self.huggingface_repo is not None


class ModelManager:
    """Manager for AI models used in the application."""

    def __init__(self, config_path: Optional[Union[str, Path]] = None, project_root: Optional[Union[str, Path]] = None):
        """Initialize the model manager.

        Args:
            config_path: Path to the models.yaml configuration file.
                        If None, defaults to src/models/models.yaml
            project_root: Path to the project root directory.
                         If None, automatically determined from this file's location
        """
        self.logger = logging.getLogger(__name__)

        if config_path is None:
            config_path = Path(__file__).parent / "models.yaml"

        # Determine project root directory
        if project_root is None:
            # Go up from src/models/model_manager.py to project root
            self.project_root = Path(__file__).parent.parent.parent
        else:
            self.project_root = Path(project_root)

        self.config_path = Path(config_path)
        self.models: Dict[str, ModelInfo] = {}
        self.config: Dict[str, Any] = {}

        self._load_config()

    def _load_config(self) -> None:
        """Load model configuration from YAML file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)

            self.config = data.get('config', {})
            models_data = data.get('models', {})

            # Parse models
            for model_id, model_data in models_data.items():
                model_info = ModelInfo(
                    name=model_data['name'],
                    version=model_data['version'],
                    filename=model_data['filename'],
                    download_url=model_data['download_url'],
                    local_path=model_data['local_path'],
                    file_size_mb=model_data['file_size_mb'],
                    description=model_data['description'],
                    model_type=model_data['model_type'],
                    framework=model_data['framework'],
                    required=model_data['required'],
                    is_trainable=model_data.get('is_trainable', False),
                    train_suffix=model_data.get('train_suffix'),
                    # New fields for Hugging Face support
                    huggingface_repo=model_data.get('huggingface_repo'),
                    huggingface_filename=model_data.get('huggingface_filename'),
                    download_method=model_data.get('download_method', 'url')
                )
                # Set the project root for the model
                model_info.set_project_root(self.project_root)

                self.models[model_id] = model_info

        except Exception as e:
            self.logger.error(f"Failed to load model configuration: {e}")
            raise

    def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
        """Get information about a specific model.

        Args:
            model_id: The model identifier

        Returns:
            ModelInfo object or None if not found
        """
        return self.models.get(model_id)

    def get_model_path(self, model_id: str) -> Optional[Path]:
        """Get the local path to a model file.

        Args:
            model_id: The model identifier

        Returns:
            Path to the model file or None if not found/available
        """
        model = self.get_model_info(model_id)
        if model and model.exists:
            return model.full_path
        return None

    def is_model_available(self, model_id: str) -> bool:
        """Check if a model is available locally.

        Args:
            model_id: The model identifier

        Returns:
            True if model exists locally, False otherwise
        """
        model = self.get_model_info(model_id)
        return model.exists if model else False

    def download_model(self, model_id: str, progress_callback: Optional[callable] = None) -> bool:
        """Download a model if it doesn't exist locally.

        Args:
            model_id: The model identifier
            progress_callback: Optional callback function for progress updates
                              Signature: callback(downloaded_bytes, total_bytes, model_name)

        Returns:
            True if download successful, False otherwise
        """
        model = self.get_model_info(model_id)
        if not model:
            self.logger.error(f"Model '{model_id}' not found in configuration")
            return False

        if model.exists:
            self.logger.info(f"Model '{model.name}' already exists at {model.full_path}")
            return True

        # Create directory if it doesn't exist
        model.full_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            if model.is_huggingface_model:
                self.logger.info(f"Downloading {model.name} from Hugging Face repo {model.huggingface_repo}")

                # Ensure the huggingface_hub is installed
                if hf_hub_download is None:
                    self.logger.error("huggingface_hub package is not installed. Please install it to download Hugging Face models.")
                    return False

                # Download the model file
                downloaded_path = hf_hub_download(
                    repo_id=model.huggingface_repo,
                    filename=model.huggingface_filename or model.filename,
                    local_dir=str(model.full_path.parent),
                    local_dir_use_symlinks=False
                )

                # Move/copy to the expected location if needed
                downloaded_file = Path(downloaded_path)
                if downloaded_file != model.full_path:
                    shutil.move(str(downloaded_file), str(model.full_path))

            else:
                self.logger.info(f"Downloading {model.name} from {model.download_url}")

                response = requests.get(
                    model.download_url,
                    stream=True,
                    timeout=self.config.get('download_timeout', 300)
                )
                response.raise_for_status()

                total_size = int(response.headers.get('content-length', 0))
                downloaded_size = 0

                with open(model.full_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded_size += len(chunk)

                            if progress_callback:
                                progress_callback(downloaded_size, total_size, model.name)

            self.logger.info(f"Successfully downloaded {model.name} to {model.full_path}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to download {model.name}: {e}")
            # Clean up partial download
            if model.full_path.exists():
                model.full_path.unlink()
            return False

    def download_all_required_models(self, progress_callback: Optional[callable] = None) -> bool:
        """Download all required models that are missing.

        Args:
            progress_callback: Optional callback for progress updates

        Returns:
            True if all required models are available, False otherwise
        """
        required_models = [
            model_id for model_id, model in self.models.items()
            if model.required and not model.exists
        ]

        if not required_models:
            self.logger.info("All required models are already available")
            return True

        success = True
        for model_id in required_models:
            if not self.download_model(model_id, progress_callback):
                success = False

        return success

    def list_models(self, model_type: Optional[str] = None,
                   framework: Optional[str] = None) -> Dict[str, ModelInfo]:
        """List available models with optional filtering.

        Args:
            model_type: Filter by model type (detection, enhancement, segmentation, etc.)
            framework: Filter by framework (pytorch, onnx, tensorflow, etc.)

        Returns:
            Dictionary of filtered models
        """
        filtered_models = {}

        for model_id, model in self.models.items():
            if model_type and model.model_type != model_type:
                continue
            if framework and model.framework != framework:
                continue

            filtered_models[model_id] = model

        return filtered_models

    def get_missing_models(self) -> Dict[str, ModelInfo]:
        """Get list of models that are not available locally.

        Returns:
            Dictionary of missing models
        """
        return {
            model_id: model for model_id, model in self.models.items()
            if not model.exists
        }

    def validate_models(self) -> Dict[str, bool]:
        """Validate that all model files exist and are accessible.

        Returns:
            Dictionary mapping model IDs to validation status
        """
        validation_results = {}

        for model_id, model in self.models.items():
            try:
                validation_results[model_id] = (
                    model.exists and
                    model.full_path.is_file() and
                    model.full_path.stat().st_size > 0
                )
            except Exception as e:
                self.logger.error(f"Error validating model {model_id}: {e}")
                validation_results[model_id] = False

        return validation_results

    def get_total_download_size(self, only_missing: bool = True) -> float:
        """Get total download size in MB.

        Args:
            only_missing: If True, only count missing models

        Returns:
            Total size in MB
        """
        models_to_count = self.get_missing_models() if only_missing else self.models
        return sum(model.file_size_mb for model in models_to_count.values())

    def get_trainable_models(self) -> Dict[str, ModelInfo]:
        """Get all trainable models.

        Returns:
            Dictionary of trainable models
        """
        return {
            model_id: model for model_id, model in self.models.items()
            if model.is_trainable
        }

    def get_available_trainable_models(self) -> Dict[str, ModelInfo]:
        """Get all trainable models that are available locally.

        Returns:
            Dictionary of available trainable models
        """
        return {
            model_id: model for model_id, model in self.models.items()
            if model.is_trainable and model.exists
        }

    def get_model_for_training(self, model_id: str) -> Optional[ModelInfo]:
        """Get a specific model for training if it's trainable and available.

        Args:
            model_id: The model identifier

        Returns:
            ModelInfo object if model is trainable and available, None otherwise
        """
        model = self.get_model_info(model_id)
        if model and model.is_trainable and model.exists:
            return model
        return None

    def list_trainable_model_options(self) -> Dict[str, str]:
        """Get a dictionary of trainable model options for UI selection.

        Returns:
            Dictionary mapping model IDs to display names (name + train_suffix)
        """
        options = {}
        for model_id, model in self.get_trainable_models().items():
            display_name = model.name
            if model.train_suffix:
                display_name += f" ({model.train_suffix})"
            options[model_id] = display_name
        return options


# Global instance
_model_manager: Optional[ModelManager] = None


def get_model_manager() -> ModelManager:
    """Get the global model manager instance."""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager


def get_model_path(model_id: str) -> Optional[Path]:
    """Convenience function to get a model path."""
    return get_model_manager().get_model_path(model_id)


def is_model_available(model_id: str) -> bool:
    """Convenience function to check if a model is available."""
    return get_model_manager().is_model_available(model_id)


def get_trainable_models() -> Dict[str, ModelInfo]:
    """Convenience function to get all trainable models."""
    return get_model_manager().get_trainable_models()


def get_available_trainable_models() -> Dict[str, ModelInfo]:
    """Convenience function to get available trainable models."""
    return get_model_manager().get_available_trainable_models()


def get_model_for_training(model_id: str) -> Optional[ModelInfo]:
    """Convenience function to get a model for training."""
    return get_model_manager().get_model_for_training(model_id)


def list_trainable_model_options() -> Dict[str, str]:
    """Convenience function to get trainable model options for UI."""
    return get_model_manager().list_trainable_model_options()
