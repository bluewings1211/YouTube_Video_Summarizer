"""
Legacy YouTube API module - DEPRECATED

This file maintains backward compatibility with the original youtube_api.py.
All functionality has been moved to the new modular structure:

- youtube_core.py: Core classes and exceptions
- youtube_metadata.py: Video metadata extraction  
- youtube_transcripts.py: Transcript processing and strategies
- youtube_fetcher.py: Unified data fetcher (optimized)
- youtube_utils.py: Utility functions

New code should use the modular imports or the unified fetcher directly.
"""

import warnings

# Issue deprecation warning
warnings.warn(
    "The youtube_api module has been refactored into smaller, focused modules. "
    "Please update your imports to use the new modular structure for better performance and maintainability. "
    "See youtube_fetcher.YouTubeUnifiedFetcher for the optimized API.",
    DeprecationWarning,
    stacklevel=2
)

# Import everything for backward compatibility
from .youtube_core import *
from .youtube_metadata import *
from .youtube_transcripts import *
from .youtube_fetcher import *
from .youtube_utils import *

# Re-export main classes for backward compatibility
YouTubeTranscriptFetcher = YouTubeUnifiedFetcher  # Main compatibility alias

# Legacy function aliases
get_youtube_video_metadata = get_youtube_video_metadata
get_youtube_video_metadata_from_url = get_youtube_video_metadata_from_url
get_available_youtube_transcripts = get_available_youtube_transcripts
fetch_youtube_transcript = fetch_youtube_transcript
fetch_youtube_transcript_from_url = fetch_youtube_transcript_from_url

# New optimized function (recommended)
fetch_youtube_data_unified = fetch_youtube_data_unified