"""
Instagram provider for downloading images and videos from Instagram profiles, hashtags, or posts.
Uses the instaloader library.
"""

import os
import re
import urllib.parse
from typing import Dict, List, Any
from .base_provider import BaseProvider

try:
    import instaloader
    INSTALOADER_AVAILABLE = True
except ImportError:
    instaloader = None
    INSTALOADER_AVAILABLE = False


class InstagramProvider(BaseProvider):
    """Provider for downloading media from Instagram."""

    def __init__(self, character_name: str = None, progress_callback=None, log_callback=None):
        super().__init__(character_name, progress_callback, log_callback)

    def download(self, output_dir: str, **params) -> List[str]:
        """Download media from Instagram."""
        if not INSTALOADER_AVAILABLE:
            raise ImportError("instaloader library is required. Install with: pip install instaloader")

        profile = params.get('profile')
        hashtag = params.get('hashtag')
        post_url = params.get('post_url')
        max_count = params.get('max_count', 100)

        downloaded_files = []
        L = instaloader.Instaloader(dirname_pattern=output_dir)

        try:
            if profile:
                downloaded_files.extend(self._download_profile(L, profile, max_count))
            elif hashtag:
                downloaded_files.extend(self._download_hashtag(L, hashtag, max_count))
            elif post_url:
                downloaded_files.extend(self._download_post(L, post_url))
            else:
                raise ValueError("Must specify either profile, hashtag, or post_url")
        except Exception as e:
            print(f"Error downloading from Instagram: {e}")

        # Update history with downloaded items
        item_ids = [os.path.basename(f) for f in downloaded_files]
        self.update_history(item_ids)

        return downloaded_files

    def _download_profile(self, loader: 'instaloader.Instaloader', profile_name: str, max_count: int) -> List[str]:
        """Download posts from a profile."""
        downloaded_files = []
        try:
            profile = instaloader.Profile.from_username(loader.context, profile_name)
            posts = profile.get_posts()

            # Calculate target directory for this profile
            profile_dir = os.path.join(loader.dirname_pattern, profile_name)

            count = 0
            for post in posts:
                if count >= max_count:
                    break
                if not self.is_duplicate(post.shortcode, profile_dir):
                    loader.download_post(post, target=profile_name)
                    downloaded_files.append(f"{profile_name}/{post.shortcode}")
                    count += 1
        except Exception as e:
            print(f"Error downloading profile {profile_name}: {e}")

        return downloaded_files

    def _download_hashtag(self, loader: 'instaloader.Instaloader', hashtag: str, max_count: int) -> List[str]:
        """Download posts from a hashtag."""
        downloaded_files = []
        try:
            posts = instaloader.Hashtag.from_name(loader.context, hashtag).get_posts()

            # Calculate target directory for this hashtag
            hashtag_dir = os.path.join(loader.dirname_pattern, f"hashtag_{hashtag}")

            count = 0
            for post in posts:
                if count >= max_count:
                    break
                if not self.is_duplicate(post.shortcode, hashtag_dir):
                    loader.download_post(post, target=f"hashtag_{hashtag}")
                    downloaded_files.append(f"hashtag_{hashtag}/{post.shortcode}")
                    count += 1
        except Exception as e:
            print(f"Error downloading hashtag {hashtag}: {e}")

        return downloaded_files

    def _download_post(self, loader: 'instaloader.Instaloader', post_url: str) -> List[str]:
        """Download a single post."""
        downloaded_files = []
        try:
            shortcode = self._extract_shortcode_from_url(post_url)

            # Calculate target directory for single posts
            single_posts_dir = os.path.join(loader.dirname_pattern, "single_posts")

            if shortcode and not self.is_duplicate(shortcode, single_posts_dir):
                post = instaloader.Post.from_shortcode(loader.context, shortcode)
                loader.download_post(post, target="single_posts")
                downloaded_files.append(f"single_posts/{shortcode}")
        except Exception as e:
            print(f"Error downloading post {post_url}: {e}")

        return downloaded_files

    def _extract_shortcode_from_url(self, url: str) -> str:
        """Extract shortcode from Instagram URL."""
        match = re.search(r'/p/([A-Za-z0-9_-]+)/', url)
        return match.group(1) if match else None

    def get_required_params(self) -> List[str]:
        """Return list of required parameters."""
        return []  # At least one of profile, hashtag, or post_url is required

    def get_optional_params(self) -> List[str]:
        """Return list of optional parameters."""
        return ['profile', 'hashtag', 'post_url', 'max_count']

    @classmethod
    def can_handle_url(cls, url: str) -> bool:
        """Check if this provider can handle the given URL."""
        return 'instagram.com' in url.lower()

    @classmethod
    def extract_params_from_url(cls, url: str) -> Dict[str, Any]:
        """Extract provider parameters from Instagram URL."""
        params = {}

        if '/p/' in url:
            # Single post URL
            params['post_url'] = url
        elif '/hashtag/' in url or '/explore/tags/' in url:
            # Hashtag URL
            match = re.search(r'(?:hashtag|tags)/([^/]+)', url)
            if match:
                params['hashtag'] = match.group(1)
        else:
            # Profile URL
            parsed = urllib.parse.urlparse(url)
            path_parts = parsed.path.strip('/').split('/')
            if path_parts and path_parts[0]:
                params['profile'] = path_parts[0]

        return params
