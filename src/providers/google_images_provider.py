"""
Google Images provider for downloading images from Google Images search results.
Uses google-images-download library or web scraping.
"""

import os
import urllib.parse
from typing import Dict, List, Any
from .base_provider import BaseProvider

try:
    from google_images_download import google_images_download
    GOOGLE_IMAGES_AVAILABLE = True
except ImportError:
    google_images_download = None
    GOOGLE_IMAGES_AVAILABLE = False


class GoogleImagesProvider(BaseProvider):
    """Provider for downloading images from Google Images."""

    def __init__(self, character_name: str = None):
        super().__init__(character_name)

    def download(self, output_dir: str, **params) -> List[str]:
        """Download images from Google Images."""
        search_query = params.get('search_query', '')
        max_count = params.get('max_count', 100)
        image_size = params.get('image_size', 'medium')  # large, medium, icon
        image_type = params.get('image_type', '')  # clipart, face, lineart, news, photo
        color = params.get('color', '')  # red, orange, yellow, green, teal, blue, purple, pink, white, gray, black, brown

        if not search_query:
            raise ValueError("search_query parameter is required")

        downloaded_files = []

        try:
            if GOOGLE_IMAGES_AVAILABLE:
                downloaded_files.extend(self._download_with_library(search_query, output_dir, max_count, image_size, image_type, color))
            else:
                downloaded_files.extend(self._download_with_scraping(search_query, output_dir, max_count))
        except Exception as e:
            print(f"Error downloading from Google Images: {e}")

        # Update history with downloaded items
        item_ids = [os.path.basename(f) for f in downloaded_files]
        self.update_history(item_ids)

        return downloaded_files

    def _download_with_library(self, query: str, output_dir: str, max_count: int,
                              image_size: str, image_type: str, color: str) -> List[str]:
        """Download using google-images-download library."""
        response = google_images_download.googleimagesdownload()

        arguments = {
            "keywords": query,
            "limit": max_count,
            "print_urls": False,
            "output_directory": output_dir,
            "image_directory": "google_images",
            "size": image_size,
            "format": "jpg",
            "no_numbering": True
        }

        if image_type:
            arguments["type"] = image_type
        if color:
            arguments["color"] = color

        try:
            paths = response.download(arguments)
            downloaded_files = []

            # Handle different response formats from google-images-download
            if isinstance(paths, dict):
                # Normal case: paths is a dictionary
                for key, file_list in paths.items():
                    if isinstance(file_list, list):
                        downloaded_files.extend(file_list)
                    elif isinstance(file_list, str):
                        downloaded_files.append(file_list)
            elif isinstance(paths, tuple):
                # Some versions return a tuple (paths_dict, errors_dict)
                if len(paths) >= 1 and isinstance(paths[0], dict):
                    for key, file_list in paths[0].items():
                        if isinstance(file_list, list):
                            downloaded_files.extend(file_list)
                        elif isinstance(file_list, str):
                            downloaded_files.append(file_list)
            elif isinstance(paths, list):
                # Sometimes it returns a list directly
                downloaded_files.extend(paths)

            return downloaded_files
        except Exception as e:
            print(f"Error with google-images-download: {e}")
            return []

    def _download_with_scraping(self, query: str, output_dir: str, max_count: int) -> List[str]:
        """Download using basic web scraping (fallback method)."""
        # This is a simplified implementation
        # In a real scenario, you'd need to handle Google's anti-bot measures
        downloaded_files = []

        search_url = f"https://www.google.com/search?q={urllib.parse.quote(query)}&tbm=isch"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

        try:
            # Note: This is a basic example and may not work reliably
            # Google has sophisticated anti-scraping measures
            print("Google Images download library not available. Using basic scraping (limited functionality).")
            print("For better results, install: pip install google-images-download")
        except Exception as e:
            print(f"Error with web scraping: {e}")

        return downloaded_files

    def get_required_params(self) -> List[str]:
        """Return list of required parameters."""
        return ['search_query']

    def get_optional_params(self) -> List[str]:
        """Return list of optional parameters."""
        return ['max_count', 'image_size', 'image_type', 'color']

    @classmethod
    def can_handle_url(cls, url: str) -> bool:
        """Check if this provider can handle the given URL."""
        return 'google.com' in url.lower() and 'tbm=isch' in url.lower()

    @classmethod
    def extract_params_from_url(cls, url: str) -> Dict[str, Any]:
        """Extract provider parameters from Google Images URL."""
        params = {}
        parsed = urllib.parse.urlparse(url)
        query_params = urllib.parse.parse_qs(parsed.query)

        if 'q' in query_params:
            params['search_query'] = query_params['q'][0]

        return params
