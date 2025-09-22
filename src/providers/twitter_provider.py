"""
Twitter provider for downloading images and videos from Twitter profiles or tweets.
Uses the snscrape library.
"""

import os
import re
import urllib.parse
import requests
from typing import Dict, List, Any, Optional
from .base_provider import BaseProvider

try:
    import snscrape.modules.twitter as sntwitter
    SNSCRAPE_AVAILABLE = True
except ImportError:
    SNSCRAPE_AVAILABLE = False


class TwitterProvider(BaseProvider):
    """Provider for downloading media from Twitter."""

    def __init__(self, character_name: Optional[str] = None):
        super().__init__(character_name)

    def download(self, output_dir: str, **params) -> List[str]:
        """Download media from Twitter."""
        if not SNSCRAPE_AVAILABLE:
            raise ImportError("snscrape library is required. Install with: pip install snscrape")

        username = params.get('username')
        search_query = params.get('search_query')
        tweet_url = params.get('tweet_url')
        max_count = params.get('max_count', 100)

        downloaded_files = []

        try:
            if username:
                downloaded_files.extend(self._download_user_tweets(username, output_dir, max_count))
            elif search_query:
                downloaded_files.extend(self._download_search_results(search_query, output_dir, max_count))
            elif tweet_url:
                downloaded_files.extend(self._download_single_tweet(tweet_url, output_dir))
            else:
                raise ValueError("Must specify either username, search_query, or tweet_url")
        except Exception as e:
            print(f"Error downloading from Twitter: {e}")

        # Update history with downloaded items
        item_ids = [os.path.basename(f) for f in downloaded_files]
        self.update_history(item_ids)

        return downloaded_files

    def _download_user_tweets(self, username: str, output_dir: str, max_count: int) -> List[str]:
        """Download media from user tweets."""
        downloaded_files = []
        count = 0

        for tweet in sntwitter.TwitterUserScraper(username).get_items():
            if count >= max_count:
                break

            if hasattr(tweet, 'media') and tweet.media:
                for media in tweet.media:
                    if not self.is_duplicate(f"{tweet.id}_{media.id}", output_dir):
                        file_path = self._download_media(media, output_dir, f"{username}_{tweet.id}")
                        if file_path:
                            downloaded_files.append(file_path)
                count += 1

        return downloaded_files

    def _download_search_results(self, query: str, output_dir: str, max_count: int) -> List[str]:
        """Download media from search results."""
        downloaded_files = []
        count = 0

        for tweet in sntwitter.TwitterSearchScraper(query).get_items():
            if count >= max_count:
                break

            if hasattr(tweet, 'media') and tweet.media:
                for media in tweet.media:
                    if not self.is_duplicate(f"{tweet.id}_{media.id}", output_dir):
                        file_path = self._download_media(media, output_dir, f"search_{tweet.id}")
                        if file_path:
                            downloaded_files.append(file_path)
                count += 1

        return downloaded_files

    def _download_single_tweet(self, tweet_url: str, output_dir: str) -> List[str]:
        """Download media from a single tweet."""
        downloaded_files = []
        tweet_id = self._extract_tweet_id_from_url(tweet_url)

        if tweet_id:
            try:
                tweet = next(sntwitter.TwitterTweetScraper(tweet_id).get_items())
                if hasattr(tweet, 'media') and tweet.media:
                    for media in tweet.media:
                        if not self.is_duplicate(f"{tweet.id}_{media.id}", output_dir):
                            file_path = self._download_media(media, output_dir, f"tweet_{tweet.id}")
                            if file_path:
                                downloaded_files.append(file_path)
            except (StopIteration, Exception) as e:
                print(f"Error downloading tweet {tweet_id}: {e}")

        return downloaded_files

    def _download_media(self, media: Any, output_dir: str, prefix: str) -> Optional[str]:
        """Download a single media file."""
        try:
            if hasattr(media, 'fullUrl'):
                url = media.fullUrl
            elif hasattr(media, 'url'):
                url = media.url
            else:
                return None

            # Determine file extension
            if 'video' in str(type(media)).lower():
                ext = '.mp4'
            else:
                ext = '.jpg'

            filename = f"{prefix}_{media.id}{ext}"
            filepath = os.path.join(output_dir, self.get_valid_filename(filename))

            response = requests.get(url, stream=True)
            response.raise_for_status()

            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            return filepath
        except Exception as e:
            print(f"Error downloading media: {e}")
            return None

    def _extract_tweet_id_from_url(self, url: str) -> Optional[str]:
        """Extract tweet ID from Twitter URL."""
        match = re.search(r'/status/(\d+)', url)
        return match.group(1) if match else None

    def get_required_params(self) -> List[str]:
        """Return list of required parameters."""
        return []  # At least one of username, search_query, or tweet_url is required

    def get_optional_params(self) -> List[str]:
        """Return list of optional parameters."""
        return ['username', 'search_query', 'tweet_url', 'max_count']

    @classmethod
    def can_handle_url(cls, url: str) -> bool:
        """Check if this provider can handle the given URL."""
        return 'twitter.com' in url.lower() or 'x.com' in url.lower()

    @classmethod
    def extract_params_from_url(cls, url: str) -> Dict[str, Any]:
        """Extract provider parameters from Twitter URL."""
        params = {}

        if '/status/' in url:
            # Single tweet URL
            params['tweet_url'] = url
        else:
            # Profile URL
            parsed = urllib.parse.urlparse(url)
            path_parts = parsed.path.strip('/').split('/')
            if path_parts and path_parts[0]:
                params['username'] = path_parts[0]

        return params
