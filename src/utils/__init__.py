"""
Utilities package for the YouTube summarizer.

This package provides various utility functions and classes for YouTube API integration,
language detection, validation, and other common operations.
"""

# Import main functionality for backward compatibility
from .youtube_fetcher import YouTubeUnifiedFetcher, YouTubeTranscriptFetcher
from .youtube_metadata import YouTubeVideoMetadataExtractor
from .youtube_transcripts import ThreeTierTranscriptStrategy, ProxyAwareTranscriptApi
from .youtube_core import (
    YouTubeTranscriptError, NoTranscriptAvailableError, RateLimitError,
    VideoNotFoundError, PrivateVideoError, LiveVideoError, VideoTooLongError
)
from .youtube_utils import (
    get_youtube_video_metadata,
    get_youtube_video_metadata_from_url,
    get_available_youtube_transcripts,
    fetch_youtube_transcript,
    fetch_youtube_transcript_from_url,
    fetch_youtube_data_unified,
    is_youtube_video_supported,
    get_youtube_transcript_tier_summary,
    diagnose_youtube_video,
    analyze_youtube_error,
    get_youtube_error_statistics
)

# Import other utilities
from .validators import YouTubeURLValidator
from .language_detector import YouTubeLanguageDetector

__all__ = [
    # Main classes
    'YouTubeUnifiedFetcher',
    'YouTubeTranscriptFetcher',  # Backward compatibility
    'YouTubeVideoMetadataExtractor',
    'ThreeTierTranscriptStrategy',
    'ProxyAwareTranscriptApi',
    
    # Exceptions
    'YouTubeTranscriptError',
    'NoTranscriptAvailableError',
    'RateLimitError',
    'VideoNotFoundError',
    'PrivateVideoError',
    'LiveVideoError',
    'VideoTooLongError',
    
    # Utility functions
    'get_youtube_video_metadata',
    'get_youtube_video_metadata_from_url',
    'get_available_youtube_transcripts',
    'fetch_youtube_transcript',
    'fetch_youtube_transcript_from_url',
    'fetch_youtube_data_unified',
    'is_youtube_video_supported',
    'get_youtube_transcript_tier_summary',
    'diagnose_youtube_video',
    'analyze_youtube_error',
    'get_youtube_error_statistics',
    
    # Other utilities
    'YouTubeURLValidator',
    'YouTubeLanguageDetector'
]