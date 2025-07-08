"""
YouTube API utilities - REFACTORED AND OPTIMIZED

This module has been completely refactored to improve maintainability and reduce API calls.
The original 3135-line file has been split into focused modules:

NEW MODULAR STRUCTURE:
- youtube_core.py: Core classes, exceptions, and logging
- youtube_metadata.py: Video metadata extraction
- youtube_transcripts.py: Transcript processing and strategies  
- youtube_fetcher.py: UNIFIED data fetcher (RECOMMENDED - reduces API calls from 3 to 2)
- youtube_utils.py: Utility functions and error handling

PERFORMANCE IMPROVEMENTS:
- API calls reduced from 3 to 2 (33% reduction)
- Unified retry logic eliminates redundant calls
- Better error handling and rate limit management

MIGRATION GUIDE:
Old code:
    from src.utils.youtube_api import YouTubeTranscriptFetcher
    fetcher = YouTubeTranscriptFetcher()
    metadata = fetcher.get_video_metadata(video_id)
    transcripts = fetcher.get_available_transcripts(video_id) 
    data = fetcher.fetch_transcript(video_id)

New optimized code:
    from src.utils.youtube_fetcher import YouTubeUnifiedFetcher
    fetcher = YouTubeUnifiedFetcher()
    all_data = fetcher.fetch_all_data(video_id)  # Single call gets everything!

For backward compatibility, the old interface is maintained below.
"""

import warnings

# Issue deprecation warning for the old interface
warnings.warn(
    "The youtube_api module has been refactored for better performance. "
    "The new YouTubeUnifiedFetcher reduces API calls from 3 to 2. "
    "Consider migrating to: from src.utils.youtube_fetcher import YouTubeUnifiedFetcher",
    DeprecationWarning,
    stacklevel=2
)

# Import everything from the new modular structure for backward compatibility
from .youtube_core import (
    YouTubeTranscriptError, NoTranscriptAvailableError, RateLimitError,
    VideoNotFoundError, PrivateVideoError, LiveVideoError, VideoTooLongError,
    UnsupportedVideoTypeError, NetworkTimeoutError, AgeRestrictedVideoError,
    RegionBlockedVideoError, TranscriptProcessingError, TranscriptValidationError,
    TierStrategyError, TranscriptAcquisitionLogger, setup_transcript_logging
)

from .youtube_metadata import YouTubeVideoMetadataExtractor

from .youtube_transcripts import (
    TranscriptTier, TranscriptInfo, ThreeTierTranscriptStrategy, 
    ProxyAwareTranscriptApi, handle_youtube_api_exceptions,
    detect_transcript_language, calculate_transcript_duration
)

from .youtube_fetcher import YouTubeUnifiedFetcher

from .youtube_utils import (
    get_youtube_video_metadata, get_youtube_video_metadata_from_url,
    get_available_youtube_transcripts, fetch_youtube_transcript,
    fetch_youtube_transcript_from_url, fetch_youtube_data_unified,
    is_youtube_video_supported, get_youtube_transcript_tier_summary,
    diagnose_youtube_video, analyze_youtube_error, get_youtube_error_statistics,
    YouTubeAPIErrorHandler
)

# Backward compatibility aliases
YouTubeTranscriptFetcher = YouTubeUnifiedFetcher

# Legacy function aliases for complete backward compatibility
def get_available_youtube_transcripts_legacy(video_id: str):
    """Legacy function - use get_available_youtube_transcripts instead."""
    return get_available_youtube_transcripts(video_id)

def get_youtube_video_metadata_legacy(video_id: str):
    """Legacy function - use get_youtube_video_metadata instead.""" 
    return get_youtube_video_metadata(video_id)

def fetch_youtube_transcript_legacy(video_id: str, **kwargs):
    """Legacy function - use fetch_youtube_transcript instead."""
    return fetch_youtube_transcript(video_id, **kwargs)

# Export all symbols for backward compatibility
__all__ = [
    # Main classes (with backward compatibility)
    'YouTubeTranscriptFetcher',  # Now points to YouTubeUnifiedFetcher
    'YouTubeUnifiedFetcher',     # New optimized fetcher
    'YouTubeVideoMetadataExtractor',
    'ThreeTierTranscriptStrategy',
    'ProxyAwareTranscriptApi',
    'TranscriptAcquisitionLogger',
    'YouTubeAPIErrorHandler',
    
    # Transcript info classes
    'TranscriptTier',
    'TranscriptInfo',
    
    # Exception classes
    'YouTubeTranscriptError',
    'NoTranscriptAvailableError', 
    'RateLimitError',
    'VideoNotFoundError',
    'PrivateVideoError',
    'LiveVideoError',
    'VideoTooLongError',
    'UnsupportedVideoTypeError',
    'NetworkTimeoutError',
    'AgeRestrictedVideoError',
    'RegionBlockedVideoError',
    'TranscriptProcessingError',
    'TranscriptValidationError',
    'TierStrategyError',
    
    # Utility functions
    'get_youtube_video_metadata',
    'get_youtube_video_metadata_from_url',
    'get_available_youtube_transcripts',
    'fetch_youtube_transcript',
    'fetch_youtube_transcript_from_url',
    'fetch_youtube_data_unified',  # NEW: Optimized single-call function
    'is_youtube_video_supported',
    'get_youtube_transcript_tier_summary',
    'diagnose_youtube_video',
    'analyze_youtube_error',
    'get_youtube_error_statistics',
    
    # Helper functions
    'setup_transcript_logging',
    'handle_youtube_api_exceptions',
    'detect_transcript_language',
    'calculate_transcript_duration',
    
    # Legacy compatibility functions
    'get_available_youtube_transcripts_legacy',
    'get_youtube_video_metadata_legacy', 
    'fetch_youtube_transcript_legacy'
]


# Module-level convenience instance for quick access (backward compatibility)
_default_fetcher = None

def get_default_fetcher():
    """Get default YouTube fetcher instance."""
    global _default_fetcher
    if _default_fetcher is None:
        _default_fetcher = YouTubeUnifiedFetcher(enable_detailed_logging=True)
    return _default_fetcher

# Quick access functions using default fetcher (for convenience)
def quick_fetch_metadata(video_id: str):
    """Quick metadata fetch using default fetcher."""
    return get_default_fetcher().get_video_metadata(video_id)

def quick_fetch_transcript(video_id: str, **kwargs):
    """Quick transcript fetch using default fetcher."""
    return get_default_fetcher().fetch_transcript(video_id, **kwargs)

def quick_fetch_all_data(video_id: str, **kwargs):
    """Quick unified data fetch using default fetcher (RECOMMENDED)."""
    return get_default_fetcher().fetch_all_data(video_id, **kwargs)