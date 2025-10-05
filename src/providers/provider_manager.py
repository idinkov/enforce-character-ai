"""
Provider manager for handling media provider operations.
"""

import os
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Type
from .base_provider import BaseProvider
from . import (
    InstagramProvider, TwitterProvider, RedditProvider,
    GoogleImagesProvider, BingImagesProvider, FlickrProvider,
    YoutubeProvider, VimeoProvider, CustomProvider, FolderProvider,
    VideoFileProvider
)


class ProviderManager:
    """Manages all media providers for the application."""

    def __init__(self):
        self.providers = {
            'instagram': InstagramProvider,
            'twitter': TwitterProvider,
            'reddit': RedditProvider,
            'google_images': GoogleImagesProvider,
            'bing_images': BingImagesProvider,
            'flickr': FlickrProvider,
            'youtube': YoutubeProvider,
            'vimeo': VimeoProvider,
            'custom': CustomProvider,
            'folder': FolderProvider,
            'video_file': VideoFileProvider
        }
        self.auto_check_thread = None
        self.auto_check_running = False

    def get_provider_by_name(self, provider_name: str) -> Optional[Type[BaseProvider]]:
        """Get provider class by name."""
        return self.providers.get(provider_name.lower())

    def get_all_provider_names(self) -> List[str]:
        """Get list of all available provider names."""
        return list(self.providers.keys())

    def detect_provider_from_url(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Auto-detect provider and extract parameters from URL.

        Returns:
            Dict containing provider type, class, and extracted parameters
        """
        for provider_type, provider_class in self.providers.items():
            if provider_class.can_handle_url(url):
                params = provider_class.extract_params_from_url(url)
                return {
                    'type': provider_type,
                    'class': provider_class,
                    'params': params
                }
        return None

    def create_provider_instance(self, provider_type: str, character_name: str = None,
                                progress_callback=None, log_callback=None) -> Optional[BaseProvider]:
        """Create an instance of the specified provider."""
        provider_class = self.get_provider_by_name(provider_type)
        if provider_class:
            return provider_class(character_name, progress_callback, log_callback)
        return None

    def add_provider_to_character(self, character_dir: str, provider_type: str,
                                 params: Dict[str, Any], download_now: bool = True,
                                 auto_check: bool = True, provider_id: str = None,
                                 progress_callback=None, log_callback=None) -> bool:
        """
        Add a provider to a character.

        Args:
            character_dir: Path to character directory
            provider_type: Type of provider to add
            params: Provider parameters
            download_now: Whether to download immediately
            auto_check: Whether to enable automatic checking
            provider_id: Optional unique identifier for the provider instance
            progress_callback: Callback for progress updates
            log_callback: Callback for log messages

        Returns:
            True if successful, False otherwise
        """
        try:
            character_name = os.path.basename(character_dir)
            provider = self.create_provider_instance(provider_type, character_name, progress_callback, log_callback)

            if not provider:
                if log_callback:
                    log_callback(f"Unknown provider type: {provider_type}")
                else:
                    print(f"Unknown provider type: {provider_type}")
                return False

            # Save provider configuration
            config = {
                'params': params,
                'download_now': download_now,
                'auto_check': auto_check
            }
            provider.save_provider_config(character_dir, config, provider_id)

            # Download immediately if requested
            if download_now:
                output_dir = provider.create_output_directory(character_dir)
                downloaded_files = provider.download(output_dir, **params)
                message = f"Downloaded {len(downloaded_files)} files from {provider_type}"
                if log_callback:
                    log_callback(message)
                else:
                    print(message)

            return True

        except Exception as e:
            error_msg = f"Error adding provider to character: {e}"
            if log_callback:
                log_callback(error_msg)
            else:
                print(error_msg)
            return False

    def remove_provider_from_character(self, character_dir: str, provider_id: str) -> bool:
        """Remove a provider from a character by ID."""
        try:
            import yaml
            provider_file = os.path.join(character_dir, 'provider.yaml')

            if not os.path.exists(provider_file):
                return False

            with open(provider_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f) or {}

            # Remove provider from list by ID
            providers = data.get('providers', [])
            original_count = len(providers)
            providers = [p for p in providers if p.get('id') != provider_id]

            if len(providers) == original_count:
                print(f"Provider with ID {provider_id} not found")
                return False

            data['providers'] = providers
            print(f"Removed provider with ID: {provider_id}")

            # Save updated config
            with open(provider_file, 'w', encoding='utf-8') as f:
                yaml.safe_dump(data, f, default_flow_style=False, allow_unicode=True)

            return True

        except Exception as e:
            print(f"Error removing provider: {e}")
            return False

    def get_character_providers(self, character_dir: str) -> List[Dict[str, Any]]:
        """Get list of providers for a character."""
        try:
            import yaml
            provider_file = os.path.join(character_dir, 'provider.yaml')

            if not os.path.exists(provider_file):
                return []

            with open(provider_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f) or {}

            return data.get('providers', [])

        except Exception as e:
            print(f"Error getting character providers: {e}")
            return []

    def check_provider_now_threaded(self, character_dir: str, provider_id: str,
                                   progress_callback=None, log_callback=None,
                                   completion_callback=None) -> threading.Thread:
        """
        Run provider check in a separate thread to prevent UI freezing.

        Args:
            character_dir: Path to character directory
            provider_id: Provider ID to check
            progress_callback: Callback for progress updates
            log_callback: Callback for log messages
            completion_callback: Callback when operation completes (receives result count)

        Returns:
            Thread object that can be used to monitor the operation
        """
        def threaded_check():
            try:
                if log_callback:
                    log_callback("Starting threaded provider check...")

                result = self.check_provider_now(
                    character_dir, provider_id,
                    progress_callback, log_callback
                )

                if completion_callback:
                    completion_callback(result)

            except Exception as e:
                error_msg = f"Error in threaded provider check: {e}"
                if log_callback:
                    log_callback(error_msg)
                if completion_callback:
                    completion_callback(-1)  # Indicate error

        thread = threading.Thread(target=threaded_check, daemon=True)
        thread.start()
        return thread

    def check_provider_now(self, character_dir: str, provider_id: str,
                           progress_callback=None, log_callback=None) -> int:
        """Manually trigger a check for new media from a specific provider by ID."""
        try:
            providers = self.get_character_providers(character_dir)

            # Find provider by ID
            provider_config = None
            provider_index = -1
            for i, config in enumerate(providers):
                if config.get('id') == provider_id:
                    provider_config = config
                    provider_index = i
                    break

            if provider_config is None:
                error_msg = f"Provider with ID {provider_id} not found"
                if log_callback:
                    log_callback(error_msg)
                else:
                    print(error_msg)
                return 0

            provider_type = provider_config.get('type')
            params = provider_config.get('params', {})

            character_name = os.path.basename(character_dir)
            provider = self.create_provider_instance(provider_type, character_name, progress_callback, log_callback)

            if provider:
                # Load existing history with provider ID
                provider.load_provider_config(character_dir, provider_id)

                # Download new media
                output_dir = provider.create_output_directory(character_dir)
                downloaded_files = provider.download(output_dir, **params)

                # Update configuration
                provider_config['history'] = {
                    'last_checked': provider.last_checked,
                    'downloaded_items': provider.history
                }

                # Save the updated provider configuration back to the file
                self._update_provider_config_in_file(character_dir, provider_index, provider_config)

                success_msg = f"Provider check completed: {len(downloaded_files)} new files downloaded"
                if log_callback:
                    log_callback(success_msg)
                else:
                    print(success_msg)
                return len(downloaded_files)
            else:
                error_msg = f"Failed to create provider instance for type: {provider_type}"
                if log_callback:
                    log_callback(error_msg)
                else:
                    print(error_msg)
                return 0

        except Exception as e:
            error_msg = f"Error checking provider: {e}"
            if log_callback:
                log_callback(error_msg)
            else:
                print(error_msg)
            return 0

    def _update_provider_config_in_file(self, character_dir: str, provider_index: int, updated_config: dict):
        """Update a specific provider's configuration in the provider.yaml file."""
        try:
            import yaml
            provider_file = os.path.join(character_dir, 'provider.yaml')

            if not os.path.exists(provider_file):
                return False

            with open(provider_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f) or {}

            # Update the specific provider
            if 'providers' in data and provider_index < len(data['providers']):
                data['providers'][provider_index] = updated_config

                # Save updated config
                with open(provider_file, 'w', encoding='utf-8') as f:
                    yaml.safe_dump(data, f, default_flow_style=False, allow_unicode=True)

                return True
            else:
                print(f"Provider index {provider_index} out of range")
                return False

        except Exception as e:
            print(f"Error updating provider config: {e}")
            return False

    def start_auto_check_service(self, characters_dir: str, check_interval_hours: int = 24):
        """Start the automatic check service for all characters with auto_check enabled."""
        if self.auto_check_running:
            return

        self.auto_check_running = True
        self.auto_check_thread = threading.Thread(
            target=self._auto_check_loop,
            args=(characters_dir, check_interval_hours),
            daemon=True
        )
        self.auto_check_thread.start()
        print("Auto-check service started")

    def stop_auto_check_service(self):
        """Stop the automatic check service."""
        self.auto_check_running = False
        if self.auto_check_thread:
            self.auto_check_thread.join()
        print("Auto-check service stopped")

    def _auto_check_loop(self, characters_dir: str, check_interval_hours: int):
        """Main loop for automatic checking."""
        while self.auto_check_running:
            try:
                self._check_all_auto_providers(characters_dir)
                # Sleep for the specified interval
                time.sleep(check_interval_hours * 3600)
            except Exception as e:
                print(f"Error in auto-check loop: {e}")
                time.sleep(3600)  # Wait 1 hour before retrying

    def _check_all_auto_providers(self, characters_dir: str):
        """Check all providers with auto_check enabled."""
        if not os.path.exists(characters_dir):
            return

        for character_name in os.listdir(characters_dir):
            character_dir = os.path.join(characters_dir, character_name)
            if not os.path.isdir(character_dir):
                continue

            try:
                providers = self.get_character_providers(character_dir)

                for provider_config in providers:
                    if not provider_config.get('auto_check', False):
                        continue

                    # Check if enough time has passed since last check
                    last_checked = provider_config.get('history', {}).get('last_checked')
                    if last_checked:
                        last_check_time = datetime.fromisoformat(last_checked)
                        if datetime.now() - last_check_time < timedelta(hours=23):
                            continue  # Too soon since last check

                    # Perform the check using provider ID
                    provider_id = provider_config.get('id')
                    if provider_id:
                        downloaded_count = self.check_provider_now(character_dir, provider_id)

                        if downloaded_count > 0:
                            provider_type = provider_config.get('type', 'unknown')
                            print(f"Auto-check: Downloaded {downloaded_count} new files for {character_name} from {provider_type} ({provider_id})")

            except Exception as e:
                print(f"Error checking providers for {character_name}: {e}")

    def get_provider_info(self, provider_type: str) -> Dict[str, Any]:
        """Get information about a provider."""
        provider_class = self.get_provider_by_name(provider_type)
        if not provider_class:
            return {}

        # Create temporary instance to get parameter info
        temp_provider = provider_class()

        return {
            'name': provider_class.__name__,
            'type': provider_type,
            'required_params': temp_provider.get_required_params(),
            'optional_params': temp_provider.get_optional_params(),
            'description': provider_class.__doc__ or f"Provider for {provider_type}"
        }

    def get_provider_display_name(self, provider_config: Dict[str, Any]) -> str:
        """Get a human-readable display name for a provider configuration."""
        provider_type = provider_config.get('type', 'unknown')
        provider_id = provider_config.get('id', '')
        params = provider_config.get('params', {})

        # Try to get a descriptive parameter for display
        if provider_type == 'folder' and 'folder_path' in params:
            folder_path = params['folder_path']
            folder_name = os.path.basename(folder_path) or folder_path
            return f"{provider_type} - {folder_name}"
        elif provider_type == 'instagram' and 'username' in params:
            return f"{provider_type} - @{params['username']}"
        elif provider_type == 'youtube' and 'channel_id' in params:
            return f"{provider_type} - {params['channel_id']}"
        elif provider_type == 'custom' and 'url' in params:
            return f"{provider_type} - {params['url'][:50]}..."
        else:
            # Fallback to provider type and ID
            return f"{provider_type} - {provider_id}"
