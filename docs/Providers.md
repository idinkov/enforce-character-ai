# Providers

Providers provide a way to download and upkeep media assets from various sources. They can be used to download images, videos, or other types of media.

## Available Providers

- InstagramProvider (`src/providers/instagram_provider.py`): Downloads images and videos and stories from Instagram profiles, hashtags, or posts. Uses instaloader library.
- FacebookProvider (`src/providers/facebook_provider.py`): Downloads images and videos from Facebook profiles or pages. Uses facebook-scraper library.
- TwitterProvider (`src/providers/twitter_provider.py`): Downloads images and videos from Twitter profiles or tweets. Uses snscrape library.
- RedditProvider (`src/providers/reddit_provider.py`): Downloads images and videos from Reddit posts or subreddits. Uses PRAW library.
- GoogleImagesProvider (`src/providers/google_images_provider.py`): Downloads images from Google Images search results. Uses google-images-download library.
- BingImagesProvider (`src/providers/bing_images_provider.py`): Downloads images from Bing Images search results. Uses bing-image-downloader library.
- FlickrProvider (`src/providers/flickr_provider.py`): Downloads images from Flickr albums or users. Uses flickrapi library.
- YoutubeProvider (`src/providers/youtube_provider.py`): Downloads videos from YouTube channels or playlists. Uses pytube library.
- VimeoProvider (`src/providers/vimeo_provider.py`): Downloads videos from Vimeo users or albums. Uses vimeo library.
- CustomProvider (`src/providers/custom_provider.py`): A template for creating custom providers for other sources.

## Installation

Some providers require additional libraries to be installed. You can install them using pip:

```bash
pip install instaloader snscrape praw google-images-download bing-image-downloader flickrapi pytube vimeo
```

## How Providers Interact with the System

Every character can have multiple providers associated with it. Each provider is responsible for downloading media from a specific source. The downloaded media is then stored in the character's image directories (e.g., `0_providers/{providerName}`).
You can see attached providers in the Character Tab. And via a button Edit Providers you can add/remove providers for the character. Usually each provider has its own set of parameters that you can configure (e.g., profile name, hashtag, search query, etc...).
When you attach a provider you have two checkboxes preselected by default: "Download now" and "Auto Check".
- **Download Now**: When checked, the provider will immediately download media when added.
- **Auto Check**: When checked, the provider will periodically check for new media and download them automatically.
- **Manual Check**: You can also manually trigger a check for new media by clicking the "Check Now" button next to the provider in the Character Tab.
- **Provider History**: Each time a provider downloads new media, it updates its history to avoid downloading duplicates in the future. The history is stored in the character's `provider.yaml` file under the `history` section.

## Provider.Yaml File

Each character has a `provider.yaml` file that stores the configuration and history of its providers. The file is located in the character's main directory (e.g., `characters/{characterName}/provider.yaml`).
The structure of the `provider.yaml` file is as follows:

```yaml
providers:
    - name: InstagramProvider
    - type: instagram
    - params:
        - profile: example_profile
        - hashtag: example_hashtag
        - search_query: example_query
    - download_now: true
    - auto_check: true
    - history:
        last_checked: 2023-10-01T12:00:00Z
        downloaded_items:
            - item1_id
            - item2_id
            - item3_id
```

## Auto Detect Provider via URL

Each provider has a way to auto detect via URL if its the provider needed. For example if you provide an Instagram post URL, the system will detect that it is an InstagramProvider and will use it to download the media and display parameters accordingly. The user can enter only raw URL in the Add Provider dialog and the system will auto detect the provider and fill in the parameters.

## Usage

To use a provider, you need to create an instance of the provider class and call the `download` method with the appropriate parameters. For example, to download images from an Instagram profile:

```python
from src.providers.instagram_provider import InstagramProvider
provider = InstagramProvider()
provider.download(profile='example_profile', max_count=100, output_dir='downloads/instagram')
```

Each provider has its own set of parameters for the `download` method. Refer to the documentation in the respective provider files for more details.

## Adding New Providers
To add a new provider, create a new class that inherits from `BaseProvider` in the `src/providers` directory. Implement the `download` method to handle the downloading logic for the new source. Make sure to handle any necessary authentication or API requirements.