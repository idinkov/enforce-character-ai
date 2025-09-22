"""
Providers package for downloading media from various sources.
"""

from .base_provider import BaseProvider
from .instagram_provider import InstagramProvider
from .twitter_provider import TwitterProvider
from .reddit_provider import RedditProvider
from .google_images_provider import GoogleImagesProvider
from .bing_images_provider import BingImagesProvider
from .flickr_provider import FlickrProvider
from .youtube_provider import YoutubeProvider
from .vimeo_provider import VimeoProvider
from .custom_provider import CustomProvider
from .folder_provider import FolderProvider
from .video_file_provider import VideoFileProvider
from .provider_manager import ProviderManager

__all__ = [
    'BaseProvider',
    'InstagramProvider',
    'TwitterProvider',
    'RedditProvider',
    'GoogleImagesProvider',
    'BingImagesProvider',
    'FlickrProvider',
    'YoutubeProvider',
    'VimeoProvider',
    'CustomProvider',
    'FolderProvider',
    'VideoFileProvider',
    'ProviderManager'
]
