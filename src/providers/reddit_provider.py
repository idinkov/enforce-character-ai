"""
Reddit provider for downloading images and videos from Reddit posts or subreddits.
Uses the PRAW library.
"""

import os
import re
import urllib.parse
import requests
from typing import Dict, List, Any, Optional
from .base_provider import BaseProvider

try:
    import praw
    PRAW_AVAILABLE = True
except ImportError:
    praw = None
    PRAW_AVAILABLE = False


class RedditProvider(BaseProvider):
    """Provider for downloading media from Reddit."""

    def __init__(self, character_name: str = None):
        super().__init__(character_name)

    def download(self, output_dir: str, **params) -> List[str]:
        """Download media from Reddit."""
        if not PRAW_AVAILABLE:
            raise ImportError("praw library is required. Install with: pip install praw")

        subreddit_name = params.get('subreddit')
        post_url = params.get('post_url')
        search_query = params.get('search_query')
        max_count = params.get('max_count', 100)
        sort_by = params.get('sort_by', 'hot')  # hot, new, top, rising

        downloaded_files = []

        try:
            # Initialize Reddit instance (read-only)
            reddit = praw.Reddit(
                client_id="dummy",
                client_secret="dummy",
            )
            reddit.read_only = True

            if subreddit_name:
                downloaded_files.extend(self._download_subreddit(reddit, subreddit_name, output_dir, max_count, sort_by))
            elif post_url:
                downloaded_files.extend(self._download_single_post(reddit, post_url, output_dir))
            elif search_query:
                downloaded_files.extend(self._download_search_results(reddit, search_query, output_dir, max_count))
            else:
                raise ValueError("Must specify either subreddit, post_url, or search_query")
        except Exception as e:
            print(f"Error downloading from Reddit: {e}")

        # Update history with downloaded items
        item_ids = [os.path.basename(f) for f in downloaded_files]
        self.update_history(item_ids)

        return downloaded_files

    def _download_subreddit(self, reddit, subreddit_name: str, output_dir: str, max_count: int, sort_by: str) -> List[str]:
        """Download media from subreddit posts."""
        downloaded_files = []
        subreddit = reddit.subreddit(subreddit_name)

        # Get posts based on sort method
        if sort_by == 'hot':
            posts = subreddit.hot(limit=max_count)
        elif sort_by == 'new':
            posts = subreddit.new(limit=max_count)
        elif sort_by == 'top':
            posts = subreddit.top(limit=max_count)
        elif sort_by == 'rising':
            posts = subreddit.rising(limit=max_count)
        else:
            posts = subreddit.hot(limit=max_count)

        for post in posts:
            if not self.is_duplicate(post.id, output_dir):
                file_path = self._download_post_media(post, output_dir, f"{subreddit_name}_{post.id}")
                if file_path:
                    downloaded_files.append(file_path)

        return downloaded_files

    def _download_single_post(self, reddit, post_url: str, output_dir: str) -> List[str]:
        """Download media from a single Reddit post."""
        downloaded_files = []
        try:
            submission = reddit.submission(url=post_url)
            if not self.is_duplicate(submission.id, output_dir):
                file_path = self._download_post_media(submission, output_dir, f"post_{submission.id}")
                if file_path:
                    downloaded_files.append(file_path)
        except Exception as e:
            print(f"Error downloading post {post_url}: {e}")

        return downloaded_files

    def _download_search_results(self, reddit, query: str, output_dir: str, max_count: int) -> List[str]:
        """Download media from search results."""
        downloaded_files = []
        try:
            for post in reddit.subreddit("all").search(query, limit=max_count):
                if not self.is_duplicate(post.id, output_dir):
                    file_path = self._download_post_media(post, output_dir, f"search_{post.id}")
                    if file_path:
                        downloaded_files.append(file_path)
        except Exception as e:
            print(f"Error downloading search results for '{query}': {e}")

        return downloaded_files

    def _download_post_media(self, post, output_dir: str, prefix: str) -> Optional[str]:
        """Download media from a Reddit post."""
        try:
            url = post.url

            # Check if it's a direct image/video link
            if any(url.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.mp4', '.webm']):
                return self._download_direct_media(url, output_dir, prefix)

            # Check for Reddit gallery
            if hasattr(post, 'is_gallery') and post.is_gallery:
                return self._download_gallery(post, output_dir, prefix)

            # Check for Reddit video
            if hasattr(post, 'is_video') and post.is_video:
                return self._download_reddit_video(post, output_dir, prefix)

            return ""
        except Exception as e:
            print(f"Error downloading post media: {e}")
            return ""

    def _download_direct_media(self, url: str, output_dir: str, prefix: str) -> Optional[str]:
        """Download direct media file."""
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()

            # Get file extension from URL
            ext = os.path.splitext(urllib.parse.urlparse(url).path)[1] or '.jpg'
            filename = f"{prefix}{ext}"
            filepath = os.path.join(output_dir, self.get_valid_filename(filename))

            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            return filepath
        except Exception as e:
            print(f"Error downloading direct media: {e}")
            return ""

    def _download_gallery(self, post, output_dir: str, prefix: str) -> Optional[str]:
        """Download Reddit gallery images."""
        # This would require additional implementation for gallery handling
        # For now, return empty string
        return ""

    def _download_reddit_video(self, post, output_dir: str, prefix: str) -> Optional[str]:
        """Download Reddit video."""
        try:
            if hasattr(post, 'media') and post.media and 'reddit_video' in post.media:
                video_url = post.media['reddit_video']['fallback_url']
                return self._download_direct_media(video_url, output_dir, prefix)
        except Exception as e:
            print(f"Error downloading Reddit video: {e}")
        return ""

    def get_required_params(self) -> List[str]:
        """Return list of required parameters."""
        return []  # At least one of subreddit, post_url, or search_query is required

    def get_optional_params(self) -> List[str]:
        """Return list of optional parameters."""
        return ['subreddit', 'post_url', 'search_query', 'max_count', 'sort_by']

    @classmethod
    def can_handle_url(cls, url: str) -> bool:
        """Check if this provider can handle the given URL."""
        return 'reddit.com' in url.lower()

    @classmethod
    def extract_params_from_url(cls, url: str) -> Dict[str, Any]:
        """Extract provider parameters from Reddit URL."""
        params = {}

        if '/r/' in url and '/comments/' in url:
            # Single post URL
            params['post_url'] = url
        elif '/r/' in url:
            # Subreddit URL
            match = re.search(r'/r/([^/]+)', url)
            if match:
                params['subreddit'] = match.group(1)

        return params
