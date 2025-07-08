"""
Refactored YouTube API utilities for video processing.

This package contains the refactored utility components that were previously
in a single youtube_api.py file. Each module focuses on a specific aspect of
YouTube API interaction:

- transcript_fetcher.py: Core transcript extraction functionality
- video_metadata.py: Video metadata extraction and processing
- url_validator.py: YouTube URL validation and video ID extraction
- language_handler.py: Language detection and processing utilities
- youtube_errors.py: YouTube-specific error classes and handling

All components maintain compatibility with the original interface.
"""

from .transcript_fetcher import YouTubeTranscriptFetcher
from .video_metadata import YouTubeVideoMetadataExtractor
from .url_validator import YouTubeURLValidator
from .youtube_errors import (
    YouTubeTranscriptError,
    UnsupportedVideoTypeError,
    PrivateVideoError,
    LiveVideoError,
    NoTranscriptAvailableError,
    VideoTooLongError,
    VideoNotFoundError,
    NetworkTimeoutError,
    RateLimitError,
    AgeRestrictedVideoError,
    RegionBlockedVideoError,
    TranscriptProcessingError,
    TranscriptValidationError
)

__all__ = [
    'YouTubeTranscriptFetcher',
    'YouTubeVideoMetadataExtractor',
    'YouTubeURLValidator',
    'YouTubeTranscriptError',
    'UnsupportedVideoTypeError',
    'PrivateVideoError',
    'LiveVideoError',
    'NoTranscriptAvailableError',
    'VideoTooLongError',
    'VideoNotFoundError',
    'NetworkTimeoutError',
    'RateLimitError',
    'AgeRestrictedVideoError',
    'RegionBlockedVideoError',
    'TranscriptProcessingError',
    'TranscriptValidationError'
]