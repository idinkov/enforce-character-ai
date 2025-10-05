"""
Base provider class that all media providers inherit from.
"""

import os
import yaml
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Any, Optional
from ..config.app_config import PROVIDER_IMPORTS_DIR


class BaseProvider(ABC):
    """Base class for all media providers."""

    def __init__(self, character_name: str = None, progress_callback=None, log_callback=None):
        self.character_name = character_name
        self.provider_type = self.__class__.__name__.lower().replace('provider', '')
        self.history = []
        self.last_checked = None
        self.provider_id = None  # Will be set when saving/loading config
        self.progress_callback = progress_callback
        self.log_callback = log_callback

    def log(self, message: str):
        """Log a message if log callback is available."""
        if self.log_callback:
            self.log_callback(message)
        else:
            print(message)

    def update_progress(self, current: int, total: int, message: str = ""):
        """Update progress if progress callback is available."""
        if self.progress_callback:
            self.progress_callback(current, total, message)

    @abstractmethod
    def download(self, output_dir: str, **params) -> List[str]:
        """
        Download media from the provider source.

        Args:
            output_dir: Directory to save downloaded media
            **params: Provider-specific parameters

        Returns:
            List of downloaded file paths
        """
        pass

    @abstractmethod
    def get_required_params(self) -> List[str]:
        """Return list of required parameters for this provider."""
        pass

    @abstractmethod
    def get_optional_params(self) -> List[str]:
        """Return list of optional parameters for this provider."""
        pass

    @classmethod
    @abstractmethod
    def can_handle_url(cls, url: str) -> bool:
        """Check if this provider can handle the given URL."""
        pass

    @classmethod
    @abstractmethod
    def extract_params_from_url(cls, url: str) -> Dict[str, Any]:
        """Extract provider parameters from URL."""
        pass

    def update_history(self, downloaded_items: List[str]):
        """Update download history to avoid duplicates."""
        self.history.extend(downloaded_items)
        self.last_checked = datetime.now().isoformat()

    def is_duplicate(self, item_id: str, target_dir: str = None) -> bool:
        """
        Check if item was already downloaded.

        Args:
            item_id: The item identifier (can be a filename or ID)
            target_dir: Directory to check for existing files (optional)

        Returns:
            True if item is duplicate (exists in history or as file), False otherwise
        """
        # Check history first for backwards compatibility
        if item_id in self.history:
            return True

        # If target_dir is provided, check if file exists
        if target_dir and os.path.exists(target_dir):
            # Check if item_id is directly a filename that exists
            if os.path.exists(os.path.join(target_dir, item_id)):
                return True

            # For cases where item_id might be used to construct a filename
            # Check for common file extensions
            common_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.mp4', '.webp', '.bmp']
            for ext in common_extensions:
                if os.path.exists(os.path.join(target_dir, f"{item_id}{ext}")):
                    return True

        return False

    def save_provider_config(self, character_dir: str, config: Dict[str, Any], provider_id: str = None):
        """Save provider configuration to character's provider.yaml file."""
        provider_file = os.path.join(character_dir, 'provider.yaml')

        # Load existing config or create new one
        if os.path.exists(provider_file):
            with open(provider_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f) or {}
        else:
            data = {'providers': []}

        # Generate unique provider ID if not provided
        if provider_id is None:
            existing_ids = [p.get('id', '') for p in data.get('providers', [])]
            counter = 1
            while f"{self.provider_type}_{counter}" in existing_ids:
                counter += 1
            provider_id = f"{self.provider_type}_{counter}"

        self.provider_id = provider_id

        # Create provider config with unique ID
        provider_config = {
            'id': provider_id,
            'name': self.__class__.__name__,
            'type': self.provider_type,
            'params': config.get('params', {}),
            'download_now': config.get('download_now', True),
            'auto_check': config.get('auto_check', True),
            'history': {
                'last_checked': self.last_checked,
                'downloaded_items': self.history
            }
        }

        # Find existing provider by ID or add new one
        existing_index = None
        for i, provider in enumerate(data['providers']):
            if provider.get('id') == provider_id:
                existing_index = i
                break

        if existing_index is not None:
            data['providers'][existing_index] = provider_config
        else:
            data['providers'].append(provider_config)

        # Save updated config
        with open(provider_file, 'w', encoding='utf-8') as f:
            yaml.safe_dump(data, f, default_flow_style=False, allow_unicode=True)

    def load_provider_config(self, character_dir: str, provider_id: str = None) -> Optional[Dict[str, Any]]:
        """Load provider configuration from character's provider.yaml file."""
        provider_file = os.path.join(character_dir, 'provider.yaml')

        if not os.path.exists(provider_file):
            return None

        with open(provider_file, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}

        # Find provider by ID (preferred) or by class name (backward compatibility)
        for provider in data.get('providers', []):
            if provider_id and provider.get('id') == provider_id:
                # Load history
                history_data = provider.get('history', {})
                self.history = history_data.get('downloaded_items', [])
                self.last_checked = history_data.get('last_checked')
                self.provider_id = provider_id
                return provider
            elif not provider_id and provider.get('name') == self.__class__.__name__:
                # Backward compatibility - find first matching provider
                history_data = provider.get('history', {})
                self.history = history_data.get('downloaded_items', [])
                self.last_checked = history_data.get('last_checked')
                self.provider_id = provider.get('id')
                return provider

        return None

    def create_output_directory(self, base_dir: str) -> str:
        """Create provider-specific output directory with unique identifier."""
        if self.provider_id:
            provider_dir = os.path.join(base_dir, 'images', PROVIDER_IMPORTS_DIR, f"{self.provider_type}_{self.provider_id}")
        else:
            provider_dir = os.path.join(base_dir, 'images', PROVIDER_IMPORTS_DIR, self.provider_type)
        os.makedirs(provider_dir, exist_ok=True)
        return provider_dir

    def get_valid_filename(self, filename: str) -> str:
        """Convert filename to valid filesystem name."""
        # Remove invalid characters
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, '_')
        return filename[:255]  # Limit filename length
