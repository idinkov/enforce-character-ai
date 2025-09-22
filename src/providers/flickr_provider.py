"""
Flickr provider for downloading images from Flickr albums or users.
Uses flickrapi library.
"""

import os
import re
import requests
from typing import Dict, List, Any, Optional
from .base_provider import BaseProvider

try:
    import flickrapi
    FLICKRAPI_AVAILABLE = True
except ImportError:
    flickrapi = None
    FLICKRAPI_AVAILABLE = False


class FlickrProvider(BaseProvider):
    """Provider for downloading images from Flickr."""

    def __init__(self, character_name: str = None):
        super().__init__(character_name)
        # You'll need to get API keys from Flickr
        self.api_key = "your_flickr_api_key"
        self.api_secret = "your_flickr_api_secret"

    def download(self, output_dir: str, **params) -> List[str]:
        """Download images from Flickr."""
        if not FLICKRAPI_AVAILABLE:
            raise ImportError("flickrapi library is required. Install with: pip install flickrapi")

        user_id = params.get('user_id')
        username = params.get('username')
        album_id = params.get('album_id')
        search_query = params.get('search_query')
        max_count = params.get('max_count', 100)

        downloaded_files = []

        try:
            flickr = flickrapi.FlickrAPI(self.api_key, self.api_secret, format='parsed-json')

            if user_id or username:
                downloaded_files.extend(self._download_user_photos(flickr, user_id, username, output_dir, max_count))
            elif album_id:
                downloaded_files.extend(self._download_album_photos(flickr, album_id, output_dir, max_count))
            elif search_query:
                downloaded_files.extend(self._download_search_results(flickr, search_query, output_dir, max_count))
            else:
                raise ValueError("Must specify user_id/username, album_id, or search_query")

        except Exception as e:
            print(f"Error downloading from Flickr: {e}")

        # Update history with downloaded items
        item_ids = [os.path.basename(f) for f in downloaded_files]
        self.update_history(item_ids)

        return downloaded_files

    def _download_user_photos(self, flickr, user_id: str, username: str, output_dir: str, max_count: int) -> List[str]:
        """Download photos from a Flickr user."""
        downloaded_files = []

        try:
            # Get user ID from username if needed
            if username and not user_id:
                user_info = flickr.people.findByUsername(username=username)
                user_id = user_info['user']['nsid']

            # Get user's photos
            photos = flickr.people.getPublicPhotos(user_id=user_id, per_page=min(max_count, 500))

            for photo in photos['photos']['photo'][:max_count]:
                if not self.is_duplicate(photo['id'], output_dir):
                    file_path = self._download_photo(flickr, photo, output_dir, f"user_{user_id}")
                    if file_path:
                        downloaded_files.append(file_path)

        except Exception as e:
            print(f"Error downloading user photos: {e}")

        return downloaded_files

    def _download_album_photos(self, flickr, album_id: str, output_dir: str, max_count: int) -> List[str]:
        """Download photos from a Flickr album."""
        downloaded_files = []

        try:
            photos = flickr.photosets.getPhotos(photoset_id=album_id, per_page=min(max_count, 500))

            for photo in photos['photoset']['photo'][:max_count]:
                if not self.is_duplicate(photo['id'], output_dir):
                    file_path = self._download_photo(flickr, photo, output_dir, f"album_{album_id}")
                    if file_path:
                        downloaded_files.append(file_path)

        except Exception as e:
            print(f"Error downloading album photos: {e}")

        return downloaded_files

    def _download_search_results(self, flickr, query: str, output_dir: str, max_count: int) -> List[str]:
        """Download photos from search results."""
        downloaded_files = []

        try:
            photos = flickr.photos.search(text=query, per_page=min(max_count, 500))

            for photo in photos['photos']['photo'][:max_count]:
                if not self.is_duplicate(photo['id'], output_dir):
                    file_path = self._download_photo(flickr, photo, output_dir, "search")
                    if file_path:
                        downloaded_files.append(file_path)

        except Exception as e:
            print(f"Error downloading search results: {e}")

        return downloaded_files

    def _download_photo(self, flickr, photo: dict, output_dir: str, prefix: str) -> Optional[str]:
        """Download a single photo."""
        try:
            # Get photo info to find the best size
            sizes = flickr.photos.getSizes(photo_id=photo['id'])

            # Find the largest available size
            best_size = None
            for size in sizes['sizes']['size']:
                if size['label'] in ['Large', 'Medium 800', 'Medium 640', 'Medium', 'Small 320']:
                    best_size = size
                    break

            if not best_size:
                best_size = sizes['sizes']['size'][-1]  # Take the last available size

            # Download the photo
            url = best_size['source']
            filename = f"{prefix}_{photo['id']}.jpg"
            filepath = os.path.join(output_dir, self.get_valid_filename(filename))

            response = requests.get(url, stream=True)
            response.raise_for_status()

            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            return filepath

        except Exception as e:
            print(f"Error downloading photo {photo['id']}: {e}")
            return None

    def get_required_params(self) -> List[str]:
        """Return list of required parameters."""
        return []  # At least one of user_id/username, album_id, or search_query is required

    def get_optional_params(self) -> List[str]:
        """Return list of optional parameters."""
        return ['user_id', 'username', 'album_id', 'search_query', 'max_count']

    @classmethod
    def can_handle_url(cls, url: str) -> bool:
        """Check if this provider can handle the given URL."""
        return 'flickr.com' in url.lower()

    @classmethod
    def extract_params_from_url(cls, url: str) -> Dict[str, Any]:
        """Extract provider parameters from Flickr URL."""
        params = {}

        if '/photos/' in url:
            # User or photo URL
            if '/sets/' in url:
                # Album URL
                match = re.search(r'/sets/(\d+)', url)
                if match:
                    params['album_id'] = match.group(1)
            else:
                # User URL
                match = re.search(r'/photos/([^/]+)', url)
                if match:
                    params['username'] = match.group(1)

        return params
