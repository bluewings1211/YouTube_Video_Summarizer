"""
YouTube-specific error classes and exception handling.

This module contains all error classes and exception handling utilities
for YouTube API operations, providing detailed error context and recovery
suggestions for different failure scenarios.
"""

import logging
from typing import Optional, List
from datetime import datetime
from contextlib import contextmanager

try:
    from youtube_transcript_api._errors import (
        TranscriptsDisabled, 
        NoTranscriptFound, 
        VideoUnavailable
    )
    # TooManyRequests might not be available in all versions
    try:
        from youtube_transcript_api._errors import TooManyRequests
    except ImportError:
        TooManyRequests = Exception
except ImportError:
    # Fallback error classes for development
    class TranscriptsDisabled(Exception): pass
    class NoTranscriptFound(Exception): pass
    class VideoUnavailable(Exception): pass
    class TooManyRequests(Exception): pass

logger = logging.getLogger(__name__)


class YouTubeTranscriptError(Exception):
    """Base exception for YouTube transcript operations."""
    
    def __init__(self, message: str, video_id: str = "", error_code: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.video_id = video_id
        self.error_code = error_code
        self.timestamp = datetime.utcnow().isoformat()


class UnsupportedVideoTypeError(YouTubeTranscriptError):
    """Raised when video type is not supported for transcript extraction."""
    pass


class PrivateVideoError(UnsupportedVideoTypeError):
    """Raised when video is private."""
    
    def __init__(self, video_id: str = ""):
        message = f"Video {video_id} is private and cannot be accessed"
        super().__init__(message, video_id, "PRIVATE_VIDEO")


class LiveVideoError(UnsupportedVideoTypeError):
    """Raised when video is live content."""
    
    def __init__(self, video_id: str = ""):
        message = f"Video {video_id} is a live stream or premiere and cannot be processed"
        super().__init__(message, video_id, "LIVE_VIDEO")


class NoTranscriptAvailableError(UnsupportedVideoTypeError):
    """Raised when no transcripts are available for the video."""
    
    def __init__(self, video_id: str = "", available_languages: List[str] = None):
        available_langs = ", ".join(available_languages) if available_languages else "none"
        message = f"No transcript available for video {video_id}. Available languages: {available_langs}"
        super().__init__(message, video_id, "NO_TRANSCRIPT")
        self.available_languages = available_languages or []


class VideoTooLongError(UnsupportedVideoTypeError):
    """Raised when video duration exceeds the maximum allowed limit."""
    
    def __init__(self, video_id: str = "", duration: int = 0, max_duration: int = 1800):
        message = f"Video {video_id} duration ({duration}s) exceeds maximum limit ({max_duration}s)"
        super().__init__(message, video_id, "VIDEO_TOO_LONG")
        self.duration = duration
        self.max_duration = max_duration


class VideoNotFoundError(YouTubeTranscriptError):
    """Raised when video cannot be found."""
    
    def __init__(self, video_id: str = ""):
        message = f"Video {video_id} not found or unavailable"
        super().__init__(message, video_id, "VIDEO_NOT_FOUND")


class NetworkTimeoutError(YouTubeTranscriptError):
    """Raised when network operations timeout."""
    
    def __init__(self, operation: str = "YouTube API request", timeout: int = 30):
        message = f"{operation} timed out after {timeout} seconds"
        super().__init__(message, "", "NETWORK_TIMEOUT")
        self.operation = operation
        self.timeout = timeout


class RateLimitError(YouTubeTranscriptError):
    """Raised when rate limits are exceeded."""
    
    def __init__(self, retry_after: int = 60):
        message = f"Rate limit exceeded. Please try again after {retry_after} seconds"
        super().__init__(message, "", "RATE_LIMIT")
        self.retry_after = retry_after


class AgeRestrictedVideoError(UnsupportedVideoTypeError):
    """Raised when video is age restricted."""
    
    def __init__(self, video_id: str = ""):
        message = f"Video {video_id} is age restricted and cannot be processed"
        super().__init__(message, video_id, "AGE_RESTRICTED")


class RegionBlockedVideoError(UnsupportedVideoTypeError):
    """Raised when video is region blocked."""
    
    def __init__(self, video_id: str = ""):
        message = f"Video {video_id} is region blocked and cannot be accessed"
        super().__init__(message, video_id, "REGION_BLOCKED")


class TranscriptProcessingError(YouTubeTranscriptError):
    """Raised when transcript processing fails."""
    
    def __init__(self, message: str, video_id: str = "", processing_stage: str = ""):
        super().__init__(message, video_id, "TRANSCRIPT_PROCESSING_ERROR")
        self.processing_stage = processing_stage


class TranscriptValidationError(YouTubeTranscriptError):
    """Raised when transcript validation fails."""
    
    def __init__(self, message: str, video_id: str = "", validation_type: str = ""):
        super().__init__(message, video_id, "TRANSCRIPT_VALIDATION_ERROR")
        self.validation_type = validation_type


class TierStrategyError(YouTubeTranscriptError):
    """Raised when three-tier strategy execution fails."""
    
    def __init__(self, message: str, video_id: str = "", tier_attempted: str = ""):
        super().__init__(message, video_id, "TIER_STRATEGY_ERROR")
        self.tier_attempted = tier_attempted


@contextmanager
def timeout_handler(operation: str, timeout_seconds: int = 30):
    """Context manager for handling timeouts in network operations."""
    import signal
    
    def timeout_handler_func(signum, frame):
        raise NetworkTimeoutError(operation, timeout_seconds)
    
    # Set the signal handler
    old_handler = signal.signal(signal.SIGALRM, timeout_handler_func)
    signal.alarm(timeout_seconds)
    
    try:
        yield
    finally:
        # Restore the old signal handler
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def handle_youtube_api_exceptions(func):
    """
    Decorator to handle common YouTube API exceptions with enhanced error categorization.
    
    This decorator provides comprehensive exception handling for YouTube API operations,
    converting YouTube API specific exceptions into our custom exception hierarchy
    with detailed error context and recovery suggestions.
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except TranscriptsDisabled as e:
            video_id = getattr(args[0], 'video_id', '') if args else ''
            logger.error(f"Transcripts disabled for video {video_id}: {str(e)}")
            raise NoTranscriptAvailableError(
                video_id, 
                available_languages=[]
            ) from e
        except NoTranscriptFound as e:
            video_id = getattr(args[0], 'video_id', '') if args else ''
            logger.error(f"No transcript found for video {video_id}: {str(e)}")
            raise NoTranscriptAvailableError(
                video_id, 
                available_languages=[]
            ) from e
        except VideoUnavailable as e:
            video_id = getattr(args[0], 'video_id', '') if args else ''
            error_msg = str(e).lower()
            
            if 'private' in error_msg:
                logger.error(f"Video {video_id} is private: {str(e)}")
                raise PrivateVideoError(video_id) from e
            elif 'live' in error_msg or 'premiere' in error_msg:
                logger.error(f"Video {video_id} is live/premiere: {str(e)}")
                raise LiveVideoError(video_id) from e
            elif 'deleted' in error_msg or 'removed' in error_msg:
                logger.error(f"Video {video_id} has been deleted: {str(e)}")
                raise VideoNotFoundError(video_id) from e
            elif 'age' in error_msg and 'restricted' in error_msg:
                logger.error(f"Video {video_id} is age restricted: {str(e)}")
                raise AgeRestrictedVideoError(video_id) from e
            elif 'region' in error_msg or 'country' in error_msg:
                logger.error(f"Video {video_id} is region blocked: {str(e)}")
                raise RegionBlockedVideoError(video_id) from e
            else:
                logger.error(f"Video {video_id} is unavailable: {str(e)}")
                raise VideoNotFoundError(video_id) from e
        except TooManyRequests as e:
            logger.error(f"Rate limit exceeded: {str(e)}")
            # Extract retry-after header if available
            retry_after = 60  # Default retry after 60 seconds
            if hasattr(e, 'response') and hasattr(e.response, 'headers'):
                retry_after = int(e.response.headers.get('Retry-After', retry_after))
            raise RateLimitError(retry_after) from e
        except Exception as e:
            logger.error(f"Unexpected YouTube API error: {str(e)}", exc_info=True)
            video_id = getattr(args[0], 'video_id', '') if args else ''
            raise YouTubeTranscriptError(f"Unexpected error: {str(e)}", video_id) from e
    return wrapper


def categorize_error(error: Exception, video_id: str = "") -> YouTubeTranscriptError:
    """
    Categorize and convert generic exceptions into specific YouTube error types.
    
    Args:
        error: The original exception
        video_id: Video ID for context
        
    Returns:
        YouTubeTranscriptError: Appropriately categorized error
    """
    error_message = str(error).lower()
    
    # Network-related errors
    if any(keyword in error_message for keyword in ['timeout', 'connection', 'network']):
        return NetworkTimeoutError(f"Network error: {str(error)}")
    
    # Rate limiting
    if any(keyword in error_message for keyword in ['rate limit', 'too many requests', '429']):
        return RateLimitError()
    
    # Video access errors
    if 'private' in error_message:
        return PrivateVideoError(video_id)
    elif 'live' in error_message or 'premiere' in error_message:
        return LiveVideoError(video_id)
    elif 'not found' in error_message or '404' in error_message:
        return VideoNotFoundError(video_id)
    elif 'age' in error_message and 'restricted' in error_message:
        return AgeRestrictedVideoError(video_id)
    elif 'region' in error_message or 'country' in error_message:
        return RegionBlockedVideoError(video_id)
    
    # Generic transcript error
    return YouTubeTranscriptError(str(error), video_id)