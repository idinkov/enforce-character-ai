"""
Bing Images provider for downloading images from Bing Images search results.
Uses bing-image-downloader library.
"""

import os
from typing import Dict, List, Any
from .base_provider import BaseProvider
import urllib.parse

try:
    from bing_image_downloader import downloader
    BING_DOWNLOADER_AVAILABLE = True
except ImportError:
    downloader = None
    BING_DOWNLOADER_AVAILABLE = False


class BingImagesProvider(BaseProvider):
    """Provider for downloading images from Bing Images."""

    def __init__(self, character_name: str = None):
        super().__init__(character_name)

    def download(self, output_dir: str, **params) -> List[str]:
        """Download images from Bing Images."""
        if not BING_DOWNLOADER_AVAILABLE:
            raise ImportError("bing-image-downloader library is required. Install with: pip install bing-image-downloader")

        search_query = params.get('search_query', '')
        max_count = params.get('max_count', 100)
        adult_filter_off = params.get('adult_filter_off', True)
        force_replace = params.get('force_replace', False)
        timeout = params.get('timeout', 60)

        if not search_query:
            raise ValueError("search_query parameter is required")

        downloaded_files = []

        try:
            # Create Bing downloader instance
            bing_downloader = downloader

            # Download images
            bing_downloader.download(
                search_query,
                limit=max_count,
                output_dir=output_dir,
                adult_filter_off=adult_filter_off,
                force_replace=force_replace,
                timeout=timeout
            )

            # Find downloaded files
            query_dir = os.path.join(output_dir, search_query)
            if os.path.exists(query_dir):
                for file in os.listdir(query_dir):
                    if file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp')):
                        file_path = os.path.join(query_dir, file)
                        if not self.is_duplicate(file, query_dir):
                            downloaded_files.append(file_path)

        except Exception as e:
            print(f"Error downloading from Bing Images: {e}")

        # Update history with downloaded items
        item_ids = [os.path.basename(f) for f in downloaded_files]
        self.update_history(item_ids)

        return downloaded_files

    def get_required_params(self) -> List[str]:
        """Return list of required parameters."""
        return ['search_query']

    def get_optional_params(self) -> List[str]:
        """Return list of optional parameters."""
        return ['max_count', 'adult_filter_off', 'force_replace', 'timeout']

    @classmethod
    def can_handle_url(cls, url: str) -> bool:
        """Check if this provider can handle the given URL."""
        return 'bing.com' in url.lower() and '/images/' in url.lower()

    @classmethod
    def extract_params_from_url(cls, url: str) -> Dict[str, Any]:
        """Extract provider parameters from Bing Images URL."""
        params = {}
        parsed = urllib.parse.urlparse(url)
        query_params = urllib.parse.parse_qs(parsed.query)

        if 'q' in query_params:
            params['search_query'] = query_params['q'][0]

        return params
