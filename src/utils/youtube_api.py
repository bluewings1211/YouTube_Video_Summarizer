"""
YouTube API utilities for transcript extraction and video information.

This module provides functionality to fetch YouTube video transcripts,
extract video metadata, and handle various YouTube-specific operations
using the youtube-transcript-api library.
"""

import logging
from typing import Optional, Dict, List, Any, Tuple
from datetime import datetime
import re
import json
import urllib.request
import urllib.parse
import time
import requests
from contextlib import contextmanager

try:
    from youtube_transcript_api import YouTubeTranscriptApi
    from youtube_transcript_api.formatters import TextFormatter
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
    raise ImportError(
        "youtube-transcript-api is required. Install with: pip install youtube-transcript-api"
    )

from src.utils.validators import YouTubeURLValidator
from src.utils.language_detector import YouTubeLanguageDetector, LanguageDetectionResult
from src.utils.proxy_manager import get_proxy_manager, get_retry_manager

# Configure logging
logger = logging.getLogger(__name__)

# Configure detailed logging for transcript acquisition
def setup_transcript_logging(log_level: str = "INFO", log_file: str = None) -> logging.Logger:
    """
    Set up comprehensive logging for transcript acquisition operations.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path for file logging
        
    Returns:
        Configured logger instance
    """
    transcript_logger = logging.getLogger(f"{__name__}.transcript_acquisition")
    transcript_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    
    # Console handler
    if not transcript_logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        transcript_logger.addHandler(console_handler)
        
        # File handler if specified
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            transcript_logger.addHandler(file_handler)
    
    return transcript_logger

# Create specialized logger for transcript operations
transcript_logger = setup_transcript_logging()


class TranscriptAcquisitionLogger:
    """
    Comprehensive logging utility for transcript acquisition operations.
    
    This class provides structured logging for all phases of transcript acquisition,
    including attempt tracking, success/failure analysis, and performance metrics.
    """
    
    def __init__(self, logger_instance: logging.Logger = None):
        self.logger = logger_instance or transcript_logger
        self.session_stats = {
            'total_attempts': 0,
            'successful_attempts': 0,
            'failed_attempts': 0,
            'videos_processed': set(),
            'tiers_used': {},
            'error_counts': {},
            'session_start': datetime.utcnow().isoformat()
        }
    
    def log_acquisition_start(self, video_id: str, strategy: str = "three_tier", 
                            preferred_languages: List[str] = None) -> None:
        """Log the start of transcript acquisition process."""
        self.logger.info(
            f"Starting transcript acquisition for video {video_id} "
            f"using {strategy} strategy with languages: {preferred_languages or 'auto-detect'}"
        )
        self.session_stats['videos_processed'].add(video_id)
    
    def log_video_metadata(self, video_id: str, metadata: Dict[str, Any]) -> None:
        """Log extracted video metadata."""
        if metadata:
            duration = metadata.get('duration_seconds', 'unknown')
            title = metadata.get('title', 'unknown')[:50] + '...' if metadata.get('title') else 'unknown'
            language = metadata.get('language', 'unknown')
            
            self.logger.debug(
                f"Video {video_id} metadata - Title: {title}, "
                f"Duration: {duration}s, Language: {language}"
            )
        else:
            self.logger.warning(f"No metadata extracted for video {video_id}")
    
    def log_strategy_planning(self, video_id: str, strategy_order: List[Any], 
                            total_options: int) -> None:
        """Log three-tier strategy planning details."""
        self.logger.info(
            f"Three-tier strategy planned for {video_id}: {total_options} transcript options available"
        )
        
        # Log tier breakdown
        tier_counts = {}
        for option in strategy_order[:10]:  # Log first 10 options
            tier = getattr(option, 'tier', 'unknown')
            lang = getattr(option, 'language_code', 'unknown')
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
            
            self.logger.debug(
                f"Strategy option: {lang} ({tier}) - Quality score: {getattr(option, 'quality_score', 'unknown')}"
            )
        
        for tier, count in tier_counts.items():
            self.logger.debug(f"Tier {tier}: {count} options available")
    
    def log_tier_attempt_start(self, video_id: str, tier: str, language: str, 
                              attempt_number: int, tier_attempt_number: int) -> None:
        """Log the start of a specific tier attempt."""
        self.logger.debug(
            f"Video {video_id} - Tier {tier} attempt {tier_attempt_number} "
            f"(overall attempt {attempt_number}): Trying language {language}"
        )
        self.session_stats['total_attempts'] += 1
        
        if tier not in self.session_stats['tiers_used']:
            self.session_stats['tiers_used'][tier] = 0
        self.session_stats['tiers_used'][tier] += 1
    
    def log_tier_attempt_success(self, video_id: str, tier: str, language: str, 
                                attempt_number: int, word_count: int, duration: float) -> None:
        """Log successful tier attempt."""
        self.logger.info(
            f"Video {video_id} - SUCCESS: Tier {tier} ({language}) "
            f"after {attempt_number} attempts. Transcript: {word_count} words, {duration:.1f}s"
        )
        self.session_stats['successful_attempts'] += 1
    
    def log_tier_attempt_failure(self, video_id: str, tier: str, language: str, 
                                attempt_number: int, error: Exception) -> None:
        """Log failed tier attempt."""
        error_type = type(error).__name__
        self.logger.warning(
            f"Video {video_id} - FAILED: Tier {tier} ({language}) "
            f"attempt {attempt_number} failed with {error_type}: {str(error)}"
        )
        self.session_stats['failed_attempts'] += 1
        
        if error_type not in self.session_stats['error_counts']:
            self.session_stats['error_counts'][error_type] = 0
        self.session_stats['error_counts'][error_type] += 1
    
    def log_tier_fallback(self, video_id: str, from_tier: str, to_tier: str, 
                         reason: str = "") -> None:
        """Log fallback from one tier to another."""
        self.logger.info(
            f"Video {video_id} - FALLBACK: Moving from tier {from_tier} to {to_tier}"
            f"{f' ({reason})' if reason else ''}"
        )
    
    def log_unsupported_video(self, video_id: str, reason: str, issues: List[Dict]) -> None:
        """Log when video is determined to be unsupported."""
        self.logger.warning(
            f"Video {video_id} is unsupported: {reason}"
        )
        for issue in issues:
            self.logger.debug(
                f"Video {video_id} issue: {issue.get('type', 'unknown')} - "
                f"{issue.get('message', 'No details')}"
            )
    
    def log_acquisition_complete(self, video_id: str, success: bool, 
                               final_tier: str = None, total_attempts: int = 0,
                               processing_time: float = 0) -> None:
        """Log completion of transcript acquisition process."""
        if success:
            self.logger.info(
                f"Video {video_id} - COMPLETED: Successfully acquired transcript "
                f"from tier {final_tier} after {total_attempts} attempts "
                f"in {processing_time:.2f}s"
            )
        else:
            self.logger.error(
                f"Video {video_id} - FAILED: Could not acquire transcript "
                f"after {total_attempts} attempts in {processing_time:.2f}s"
            )
    
    def log_performance_metrics(self, video_id: str, metrics: Dict[str, Any]) -> None:
        """Log performance metrics for the acquisition process."""
        self.logger.debug(
            f"Video {video_id} performance metrics: "
            f"Total time: {metrics.get('total_time', 0):.2f}s, "
            f"API calls: {metrics.get('api_calls', 0)}, "
            f"Retry count: {metrics.get('retries', 0)}"
        )
    
    def log_error_analysis(self, video_id: str, error_report: Dict[str, Any]) -> None:
        """Log detailed error analysis."""
        self.logger.debug(
            f"Video {video_id} error analysis: "
            f"Total errors: {error_report.get('total_errors', 0)}, "
            f"Categories: {list(error_report.get('error_categories', {}).keys())}, "
            f"Recoverable: {error_report.get('recovery_analysis', {}).get('recoverable_errors', 0)}"
        )
    
    def log_session_summary(self) -> Dict[str, Any]:
        """Log and return session summary statistics."""
        summary = {
            'session_duration': (datetime.utcnow() - 
                                datetime.fromisoformat(self.session_stats['session_start'])).total_seconds(),
            'videos_processed': len(self.session_stats['videos_processed']),
            'total_attempts': self.session_stats['total_attempts'],
            'successful_attempts': self.session_stats['successful_attempts'],
            'failed_attempts': self.session_stats['failed_attempts'],
            'success_rate': (self.session_stats['successful_attempts'] / 
                           max(1, self.session_stats['total_attempts'])) * 100,
            'tiers_used': dict(self.session_stats['tiers_used']),
            'error_counts': dict(self.session_stats['error_counts'])
        }
        
        self.logger.info(
            f"Session summary: {summary['videos_processed']} videos, "
            f"{summary['total_attempts']} attempts, "
            f"{summary['success_rate']:.1f}% success rate"
        )
        
        return summary
    
    def reset_session_stats(self) -> None:
        """Reset session statistics."""
        self.session_stats = {
            'total_attempts': 0,
            'successful_attempts': 0,
            'failed_attempts': 0,
            'videos_processed': set(),
            'tiers_used': {},
            'error_counts': {},
            'session_start': datetime.utcnow().isoformat()
        }
        self.logger.info("Session statistics reset")


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


class YouTubeVideoMetadataExtractor:
    """Extracts video metadata from YouTube."""
    
    def __init__(self):
        self.user_agent = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        )
    
    def extract_video_metadata(self, video_id: str) -> Dict[str, Any]:
        """
        Extract video metadata including title, duration, and language.
        
        Args:
            video_id: YouTube video ID
            
        Returns:
            Dictionary containing video metadata
            
        Raises:
            YouTubeTranscriptError: If metadata cannot be extracted
        """
        if not video_id:
            raise YouTubeTranscriptError("Video ID is required")
            
        if not YouTubeURLValidator._is_valid_video_id(video_id):
            raise YouTubeTranscriptError(f"Invalid video ID format: {video_id}")
        
        try:
            # Fetch video page to extract metadata using proxy manager
            url = f"https://www.youtube.com/watch?v={video_id}"
            
            # Use proxy manager with retry logic
            proxy_manager = get_proxy_manager()
            retry_manager = get_retry_manager()
            
            with retry_manager.retry_context("metadata_extraction"):
                with proxy_manager.request_context() as (session, proxy):
                    # Configure headers
                    headers = {
                        'User-Agent': self.user_agent,
                        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                        'Accept-Language': 'en-US,en;q=0.5',
                        'Accept-Encoding': 'gzip, deflate',
                        'Connection': 'keep-alive',
                        'Upgrade-Insecure-Requests': '1'
                    }
                    
                    # Make request with proxy support
                    logger.debug(f"Fetching metadata for {video_id} via proxy: {proxy.url if proxy else 'direct'}")
                    
                    response = session.get(url, headers=headers, timeout=30)
                    response.raise_for_status()
                    
                    page_content = response.text
                    
                    # Log successful metadata retrieval
                    logger.info(f"Successfully fetched metadata for {video_id}")
            
            # Extract metadata from page
            metadata = self._parse_metadata_from_page(page_content, video_id)
            
            return metadata
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                raise YouTubeTranscriptError("Video not found")
            elif e.response.status_code == 403:
                raise YouTubeTranscriptError("Access denied to video")
            else:
                raise YouTubeTranscriptError(f"HTTP error {e.response.status_code}: {e.response.reason}")
                
        except Exception as e:
            logger.error(f"Error extracting metadata for {video_id}: {str(e)}")
            raise YouTubeTranscriptError(f"Failed to extract video metadata: {str(e)}")
    
    def _parse_metadata_from_page(self, page_content: str, video_id: str) -> Dict[str, Any]:
        """
        Parse metadata from YouTube page content.
        
        Args:
            page_content: HTML content of YouTube page
            video_id: Video ID for reference
            
        Returns:
            Dictionary containing parsed metadata
        """
        metadata = {
            'video_id': video_id,
            'title': None,
            'duration_seconds': None,
            'language': None,
            'view_count': None,
            'upload_date': None,
            'channel_name': None,
            'description': None,
            'is_live': False,
            'is_private': False,
            'extraction_timestamp': datetime.utcnow().isoformat()
        }
        
        try:
            # Extract title
            title_match = re.search(r'"title":"([^"]+)"', page_content)
            if title_match:
                metadata['title'] = title_match.group(1).replace('\\u0026', '&')
            
            # Extract duration (in seconds)
            duration_match = re.search(r'"lengthSeconds":"(\d+)"', page_content)
            if duration_match:
                metadata['duration_seconds'] = int(duration_match.group(1))
            
            # Extract language
            lang_match = re.search(r'"defaultAudioLanguage":"([^"]+)"', page_content)
            if lang_match:
                metadata['language'] = lang_match.group(1)
            else:
                # Try alternative language extraction
                lang_match = re.search(r'"lang":"([^"]+)"', page_content)
                if lang_match:
                    metadata['language'] = lang_match.group(1)
            
            # Extract view count
            view_match = re.search(r'"viewCount":"(\d+)"', page_content)
            if view_match:
                metadata['view_count'] = int(view_match.group(1))
            
            # Extract channel name
            channel_match = re.search(r'"author":"([^"]+)"', page_content)
            if channel_match:
                metadata['channel_name'] = channel_match.group(1)
            
            # Check if video is live
            if '"isLiveContent":true' in page_content:
                metadata['is_live'] = True
            
            # Check if video is private (basic check)
            if 'Private video' in page_content or 'This video is private' in page_content:
                metadata['is_private'] = True
            
            # Extract description (first part)
            desc_match = re.search(r'"shortDescription":"([^"]+)"', page_content)
            if desc_match:
                metadata['description'] = desc_match.group(1)[:500]  # Limit to 500 chars
            
        except Exception as e:
            logger.warning(f"Error parsing some metadata fields for {video_id}: {str(e)}")
        
        return metadata
    
    def get_video_duration_formatted(self, duration_seconds: int) -> str:
        """
        Format duration in seconds to human-readable format.
        
        Args:
            duration_seconds: Duration in seconds
            
        Returns:
            Formatted duration string (e.g., "5:30", "1:02:30")
        """
        if duration_seconds is None:
            return "Unknown"
            
        hours = duration_seconds // 3600
        minutes = (duration_seconds % 3600) // 60
        seconds = duration_seconds % 60
        
        if hours > 0:
            return f"{hours}:{minutes:02d}:{seconds:02d}"
        else:
            return f"{minutes}:{seconds:02d}"
    
    def detect_unsupported_video_type(
        self, 
        video_id: str, 
        max_duration_seconds: int = 1800
    ) -> Dict[str, Any]:
        """
        Detect if a video has unsupported characteristics for transcript extraction.
        
        Args:
            video_id: YouTube video ID
            max_duration_seconds: Maximum allowed duration in seconds (default: 1800 = 30 minutes)
            
        Returns:
            Dictionary containing video support status and details
            
        Raises:
            YouTubeTranscriptError: If video cannot be analyzed
        """
        try:
            # Extract metadata to check video characteristics
            metadata = self.extract_video_metadata(video_id)
            
            issues = []
            is_supported = True
            
            # Check if video is private
            if metadata.get('is_private', False):
                issues.append({
                    'type': 'private',
                    'severity': 'critical',
                    'message': 'Video is private and cannot be accessed'
                })
                is_supported = False
            
            # Check if video is live content
            if metadata.get('is_live', False):
                issues.append({
                    'type': 'live',
                    'severity': 'critical',
                    'message': 'Live videos do not have transcripts available'
                })
                is_supported = False
            
            # Check video duration
            duration_seconds = metadata.get('duration_seconds')
            if duration_seconds is not None:
                if duration_seconds > max_duration_seconds:
                    duration_formatted = self.get_video_duration_formatted(duration_seconds)
                    max_duration_formatted = self.get_video_duration_formatted(max_duration_seconds)
                    issues.append({
                        'type': 'too_long',
                        'severity': 'critical',
                        'message': f'Video duration ({duration_formatted}) exceeds maximum allowed duration ({max_duration_formatted})'
                    })
                    is_supported = False
            else:
                issues.append({
                    'type': 'no_duration',
                    'severity': 'warning',
                    'message': 'Video duration could not be determined'
                })
            
            # Check if video has no title (might be unavailable)
            if not metadata.get('title'):
                issues.append({
                    'type': 'unavailable',
                    'severity': 'warning',
                    'message': 'Video title could not be extracted, may be unavailable'
                })
            
            return {
                'video_id': video_id,
                'is_supported': is_supported,
                'issues': issues,
                'metadata': metadata,
                'max_duration_seconds': max_duration_seconds,
                'check_timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error detecting video type for {video_id}: {str(e)}")
            raise YouTubeTranscriptError(f"Failed to analyze video: {str(e)}")
    
    def check_transcript_availability(self, video_id: str) -> Dict[str, Any]:
        """
        Check if transcripts are available for a video without fetching them.
        
        Args:
            video_id: YouTube video ID
            
        Returns:
            Dictionary containing transcript availability information
            
        Raises:
            YouTubeTranscriptError: If video cannot be analyzed
        """
        try:
            # Use proxy-aware transcript API
            proxy_api = ProxyAwareTranscriptApi()
            
            # Try to list available transcripts
            transcript_list = proxy_api.list_transcripts(video_id)
            
            available_transcripts = []
            for transcript in transcript_list:
                available_transcripts.append({
                    'language': transcript.language,
                    'language_code': transcript.language_code,
                    'is_generated': transcript.is_generated,
                    'is_translatable': transcript.is_translatable
                })
            
            return {
                'video_id': video_id,
                'has_transcripts': len(available_transcripts) > 0,
                'transcript_count': len(available_transcripts),
                'available_transcripts': available_transcripts,
                'check_timestamp': datetime.utcnow().isoformat()
            }
            
        except VideoUnavailable:
            raise YouTubeTranscriptError("Video is unavailable")
        except Exception as e:
            logger.error(f"Error checking transcript availability for {video_id}: {str(e)}")
            raise YouTubeTranscriptError(f"Failed to check transcript availability: {str(e)}")
    
    def validate_video_duration(
        self, 
        video_id: str, 
        max_duration_seconds: int = 1800
    ) -> Dict[str, Any]:
        """
        Validate that video duration is within acceptable limits.
        
        Args:
            video_id: YouTube video ID
            max_duration_seconds: Maximum allowed duration in seconds (default: 1800 = 30 minutes)
            
        Returns:
            Dictionary containing duration validation results
            
        Raises:
            YouTubeTranscriptError: If video cannot be analyzed
            VideoTooLongError: If video exceeds duration limit
        """
        try:
            # Extract metadata to get duration
            metadata = self.extract_video_metadata(video_id)
            duration_seconds = metadata.get('duration_seconds')
            
            if duration_seconds is None:
                return {
                    'video_id': video_id,
                    'is_valid_duration': False,
                    'duration_seconds': None,
                    'max_duration_seconds': max_duration_seconds,
                    'duration_formatted': 'Unknown',
                    'error': 'Could not determine video duration',
                    'check_timestamp': datetime.utcnow().isoformat()
                }
            
            is_valid = duration_seconds <= max_duration_seconds
            duration_formatted = self.get_video_duration_formatted(duration_seconds)
            max_duration_formatted = self.get_video_duration_formatted(max_duration_seconds)
            
            result = {
                'video_id': video_id,
                'is_valid_duration': is_valid,
                'duration_seconds': duration_seconds,
                'max_duration_seconds': max_duration_seconds,
                'duration_formatted': duration_formatted,
                'max_duration_formatted': max_duration_formatted,
                'check_timestamp': datetime.utcnow().isoformat()
            }
            
            if not is_valid:
                result['error'] = f"Video duration ({duration_formatted}) exceeds maximum allowed duration ({max_duration_formatted})"
                
            return result
            
        except Exception as e:
            logger.error(f"Error validating duration for {video_id}: {str(e)}")
            raise YouTubeTranscriptError(f"Failed to validate video duration: {str(e)}")
    
    def check_duration_limit(
        self, 
        video_id: str, 
        max_duration_seconds: int = 1800,
        raise_on_exceeded: bool = True
    ) -> bool:
        """
        Check if video duration is within limits.
        
        Args:
            video_id: YouTube video ID
            max_duration_seconds: Maximum allowed duration in seconds (default: 1800 = 30 minutes)
            raise_on_exceeded: Whether to raise exception if duration is exceeded
            
        Returns:
            True if duration is within limits, False otherwise
            
        Raises:
            VideoTooLongError: If video exceeds duration and raise_on_exceeded is True
            YouTubeTranscriptError: If video cannot be analyzed
        """
        validation_result = self.validate_video_duration(video_id, max_duration_seconds)
        
        if not validation_result['is_valid_duration'] and raise_on_exceeded:
            duration_formatted = validation_result.get('duration_formatted', 'Unknown')
            max_duration_formatted = validation_result.get('max_duration_formatted', 'Unknown')
            
            raise VideoTooLongError(
                f"Video duration ({duration_formatted}) exceeds maximum allowed duration ({max_duration_formatted})"
            )
        
        return validation_result['is_valid_duration']


class TranscriptTier:
    """Enumeration of transcript quality tiers."""
    MANUAL = "manual"           # Tier 1: Manually created transcripts (highest quality)
    AUTO_GENERATED = "auto"     # Tier 2: Auto-generated transcripts (medium quality)  
    TRANSLATED = "translated"   # Tier 3: Translated transcripts (lowest quality)


class TranscriptInfo:
    """Information about a transcript including its tier and metadata."""
    
    def __init__(self, language_code: str, is_generated: bool, is_translatable: bool, 
                 video_id: str = "", original_transcript=None):
        self.language_code = language_code
        self.is_generated = is_generated
        self.is_translatable = is_translatable
        self.video_id = video_id
        self.original_transcript = original_transcript
        
        # Determine transcript tier
        if not is_generated:
            self.tier = TranscriptTier.MANUAL
        elif is_translatable:
            self.tier = TranscriptTier.TRANSLATED
        else:
            self.tier = TranscriptTier.AUTO_GENERATED
        
        # Quality score (higher is better)
        self.quality_score = self._calculate_quality_score()
    
    def _calculate_quality_score(self) -> int:
        """Calculate quality score for transcript comparison."""
        if self.tier == TranscriptTier.MANUAL:
            return 100
        elif self.tier == TranscriptTier.AUTO_GENERATED:
            return 50
        else:  # TRANSLATED
            return 25
    
    def __repr__(self):
        return f"TranscriptInfo(lang={self.language_code}, tier={self.tier}, score={self.quality_score})"


class ThreeTierTranscriptStrategy:
    """
    Implements three-tier transcript acquisition strategy.
    
    Tier 1: Manual transcripts (highest quality, human-created)
    Tier 2: Auto-generated transcripts (medium quality, AI-created)
    Tier 3: Translated transcripts (lowest quality, translated from other languages)
    """
    
    def __init__(self, language_detector: Optional[YouTubeLanguageDetector] = None):
        self.language_detector = language_detector or YouTubeLanguageDetector()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def get_transcript_strategy(self, video_id: str, preferred_languages: Optional[List[str]] = None,
                              video_metadata: Optional[Dict[str, Any]] = None) -> List[TranscriptInfo]:
        """
        Get ordered list of transcript acquisition attempts based on three-tier strategy.
        
        Args:
            video_id: YouTube video ID
            preferred_languages: List of preferred language codes
            video_metadata: Optional video metadata for language detection
            
        Returns:
            List of TranscriptInfo objects ordered by preference (highest quality first)
            
        Raises:
            YouTubeTranscriptError: If unable to analyze transcript options
        """
        try:
            self.logger.debug(f"Building three-tier strategy for video {video_id}")
            
            # Get available transcripts
            available_transcripts = self._get_available_transcripts(video_id)
            
            if not available_transcripts:
                raise NoTranscriptAvailableError(video_id, [])
            
            # Determine preferred languages from video metadata if not provided
            if not preferred_languages and video_metadata:
                try:
                    detection_result = self.language_detector.detect_language_from_metadata(video_metadata)
                    preferred_languages = self.language_detector.get_preferred_transcript_languages(detection_result)
                    self.logger.debug(f"Detected preferred languages: {preferred_languages}")
                except Exception as e:
                    self.logger.warning(f"Language detection failed, using default languages: {str(e)}")
                    preferred_languages = ['en', 'zh-CN', 'zh-TW', 'zh', 'ja', 'ko']
            elif not preferred_languages:
                preferred_languages = ['en', 'zh-CN', 'zh-TW', 'zh', 'ja', 'ko']
            
            # Create transcript info objects
            transcript_infos = []
            for transcript in available_transcripts:
                try:
                    info = TranscriptInfo(
                        language_code=transcript.language_code,
                        is_generated=transcript.is_generated,
                        is_translatable=transcript.is_translatable,
                        video_id=video_id,
                        original_transcript=transcript
                    )
                    transcript_infos.append(info)
                except Exception as e:
                    self.logger.warning(f"Failed to create TranscriptInfo for {transcript.language_code}: {str(e)}")
            
            # Sort transcripts by three-tier strategy
            strategy_order = self._sort_by_strategy(transcript_infos, preferred_languages)
            
            self.logger.info(f"Three-tier strategy for {video_id}: {len(strategy_order)} options available")
            for i, info in enumerate(strategy_order[:5]):  # Log first 5
                self.logger.debug(f"  {i+1}. {info.language_code} ({info.tier}, score: {info.quality_score})")
            
            return strategy_order
            
        except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable) as e:
            # Re-raise known transcript errors
            raise NoTranscriptAvailableError(video_id, [])
        except Exception as e:
            self.logger.error(f"Failed to build transcript strategy for {video_id}: {str(e)}")
            raise YouTubeTranscriptError(f"Failed to analyze transcript options: {str(e)}", video_id)
    
    def _get_available_transcripts(self, video_id: str):
        """Get available transcripts from YouTube API using proxy support."""
        try:
            proxy_api = ProxyAwareTranscriptApi()
            return proxy_api.list_transcripts(video_id)
        except Exception as e:
            self.logger.debug(f"Failed to list transcripts for {video_id}: {str(e)}")
            raise
    
    def _sort_by_strategy(self, transcript_infos: List[TranscriptInfo], 
                         preferred_languages: List[str]) -> List[TranscriptInfo]:
        """
        Sort transcripts by three-tier strategy preference.
        
        Priority order:
        1. Manual transcripts in preferred languages (Tier 1)
        2. Auto-generated transcripts in preferred languages (Tier 2)
        3. Manual transcripts in other languages (Tier 1 fallback)
        4. Auto-generated transcripts in other languages (Tier 2 fallback)
        5. Translated transcripts in preferred languages (Tier 3)
        6. Translated transcripts in other languages (Tier 3 fallback)
        """
        
        def sort_key(transcript_info: TranscriptInfo):
            # Language preference score (higher for preferred languages)
            try:
                lang_score = len(preferred_languages) - preferred_languages.index(transcript_info.language_code)
            except ValueError:
                # Check for partial matches (e.g., 'en' matches 'en-US')
                lang_score = 0
                for i, lang in enumerate(preferred_languages):
                    if (transcript_info.language_code.startswith(lang) or 
                        lang.startswith(transcript_info.language_code)):
                        lang_score = len(preferred_languages) - i
                        break
            
            # Tier priority score
            if transcript_info.tier == TranscriptTier.MANUAL:
                tier_score = 1000
            elif transcript_info.tier == TranscriptTier.AUTO_GENERATED:
                tier_score = 500  
            else:  # TRANSLATED
                tier_score = 100
            
            # Combined score (tier is most important, then language preference)
            total_score = tier_score + lang_score
            
            # Log for debugging
            self.logger.debug(f"Score for {transcript_info.language_code} ({transcript_info.tier}): {total_score}")
            
            return total_score
        
        # Sort by combined score (highest first)
        sorted_transcripts = sorted(transcript_infos, key=sort_key, reverse=True)
        
        return sorted_transcripts
    
    def get_best_transcript_option(self, video_id: str, preferred_languages: Optional[List[str]] = None,
                                  video_metadata: Optional[Dict[str, Any]] = None) -> Optional[TranscriptInfo]:
        """
        Get the single best transcript option based on three-tier strategy.
        
        Args:
            video_id: YouTube video ID
            preferred_languages: List of preferred language codes
            video_metadata: Optional video metadata for language detection
            
        Returns:
            Best TranscriptInfo option or None if no transcripts available
        """
        try:
            strategy_order = self.get_transcript_strategy(video_id, preferred_languages, video_metadata)
            return strategy_order[0] if strategy_order else None
        except Exception:
            return None
    
    def categorize_transcripts_by_tier(self, transcript_infos: List[TranscriptInfo]) -> Dict[str, List[TranscriptInfo]]:
        """
        Categorize transcripts by their tier.
        
        Args:
            transcript_infos: List of transcript information objects
            
        Returns:
            Dictionary with tiers as keys and lists of transcripts as values
        """
        categorized = {
            TranscriptTier.MANUAL: [],
            TranscriptTier.AUTO_GENERATED: [],
            TranscriptTier.TRANSLATED: []
        }
        
        for info in transcript_infos:
            categorized[info.tier].append(info)
        
        return categorized
    
    def get_transcript_tier_summary(self, video_id: str, preferred_languages: Optional[List[str]] = None,
                                   video_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Get summary of available transcripts organized by tier.
        
        Args:
            video_id: YouTube video ID
            preferred_languages: List of preferred language codes
            video_metadata: Optional video metadata for language detection
            
        Returns:
            Dictionary containing transcript tier summary
        """
        try:
            strategy_order = self.get_transcript_strategy(video_id, preferred_languages, video_metadata)
            categorized = self.categorize_transcripts_by_tier(strategy_order)
            
            summary = {
                'video_id': video_id,
                'total_transcripts': len(strategy_order),
                'tiers': {
                    'manual': {
                        'count': len(categorized[TranscriptTier.MANUAL]),
                        'languages': [t.language_code for t in categorized[TranscriptTier.MANUAL]]
                    },
                    'auto_generated': {
                        'count': len(categorized[TranscriptTier.AUTO_GENERATED]),
                        'languages': [t.language_code for t in categorized[TranscriptTier.AUTO_GENERATED]]
                    },
                    'translated': {
                        'count': len(categorized[TranscriptTier.TRANSLATED]),
                        'languages': [t.language_code for t in categorized[TranscriptTier.TRANSLATED]]
                    }
                },
                'best_option': {
                    'language': strategy_order[0].language_code if strategy_order else None,
                    'tier': strategy_order[0].tier if strategy_order else None,
                    'quality_score': strategy_order[0].quality_score if strategy_order else None
                },
                'preferred_languages': preferred_languages,
                'strategy_order': [
                    {
                        'language': t.language_code,
                        'tier': t.tier,
                        'quality_score': t.quality_score
                    } for t in strategy_order
                ]
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Failed to generate transcript tier summary for {video_id}: {str(e)}")
            return {
                'video_id': video_id,
                'error': str(e),
                'total_transcripts': 0,
                'tiers': {'manual': {'count': 0, 'languages': []}, 
                         'auto_generated': {'count': 0, 'languages': []},
                         'translated': {'count': 0, 'languages': []}},
                'best_option': {'language': None, 'tier': None, 'quality_score': None}
            }


class ProxyAwareTranscriptApi:
    """
    Proxy-aware wrapper for YouTubeTranscriptApi.
    
    This class provides proxy support for YouTube transcript API calls by 
    monkey-patching the underlying HTTP requests.
    """
    
    def __init__(self):
        self.proxy_manager = get_proxy_manager()
        self.retry_manager = get_retry_manager()
        
    def _patch_transcript_api(self, proxy_info):
        """
        Temporarily patch the YouTubeTranscriptApi to use proxy.
        
        Args:
            proxy_info: Proxy information from proxy manager
        """
        try:
            if proxy_info and hasattr(urllib.request, 'ProxyHandler'):
                # Create proxy handler
                proxy_dict = {
                    'http': proxy_info.url,
                    'https': proxy_info.url
                }
                
                proxy_handler = urllib.request.ProxyHandler(proxy_dict)
                opener = urllib.request.build_opener(proxy_handler)
                
                # Install the proxy opener
                urllib.request.install_opener(opener)
                
                logger.debug(f"Successfully installed proxy handler for transcript API: {proxy_info.url}")
            else:
                logger.debug("No proxy info provided or ProxyHandler not available, using direct connection")
                
        except Exception as e:
            logger.error(f"Failed to install proxy handler for {proxy_info.url if proxy_info else 'None'}: {e}")
            # Continue without proxy
            pass
    
    def _restore_transcript_api(self):
        """Restore original YouTubeTranscriptApi behavior."""
        try:
            # Create a new opener without proxy
            opener = urllib.request.build_opener()
            urllib.request.install_opener(opener)
            
            logger.debug("Successfully restored original transcript API behavior")
        except Exception as e:
            logger.error(f"Failed to restore original transcript API behavior: {e}")
            # This is not critical, continue
    
    def get_transcript(self, video_id: str, languages: Optional[List[str]] = None):
        """
        Get transcript with proxy support.
        
        Args:
            video_id: YouTube video ID
            languages: Preferred languages for transcript
            
        Returns:
            Transcript data
        """
        operation_name = f"transcript_fetch_{video_id}"
        
        with self.retry_manager.retry_context(operation_name):
            with self.proxy_manager.request_context() as (session, proxy):
                try:
                    # Apply proxy to transcript API
                    self._patch_transcript_api(proxy)
                    
                    # Log the request with detailed information
                    lang_str = f"languages={languages}" if languages else "default_languages"
                    proxy_str = proxy.url if proxy else 'direct'
                    logger.debug(f"Fetching transcript for {video_id} ({lang_str}) via {proxy_str}")
                    
                    # Make the transcript request
                    start_time = time.time()
                    
                    if languages:
                        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=languages)
                    else:
                        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
                    
                    fetch_time = time.time() - start_time
                    
                    # Validate transcript data
                    if not transcript_list:
                        raise YouTubeTranscriptError(f"Empty transcript returned for video {video_id}")
                    
                    # Log successful retrieval with metrics
                    transcript_length = len(transcript_list)
                    logger.info(f"Successfully fetched transcript for {video_id} "
                              f"(length: {transcript_length} segments, "
                              f"fetch_time: {fetch_time:.2f}s, "
                              f"proxy: {proxy_str})")
                    
                    return transcript_list
                    
                except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable) as e:
                    # These are expected YouTube API errors, don't retry
                    logger.warning(f"YouTube API error for {video_id}: {type(e).__name__}: {e}")
                    raise
                except TooManyRequests as e:
                    # Rate limiting error, should be retried with backoff
                    logger.warning(f"Rate limit hit for {video_id}: {e}")
                    raise
                except Exception as e:
                    # Unexpected error, log details and reraise for retry logic
                    logger.error(f"Unexpected error fetching transcript for {video_id}: {type(e).__name__}: {e}")
                    raise
                    
                finally:
                    # Always restore original behavior
                    self._restore_transcript_api()
    
    def list_transcripts(self, video_id: str):
        """
        List available transcripts with proxy support.
        
        Args:
            video_id: YouTube video ID
            
        Returns:
            Transcript list
        """
        operation_name = f"transcript_list_{video_id}"
        
        with self.retry_manager.retry_context(operation_name):
            with self.proxy_manager.request_context() as (session, proxy):
                try:
                    # Apply proxy to transcript API
                    self._patch_transcript_api(proxy)
                    
                    # Log the request with detailed information
                    proxy_str = proxy.url if proxy else 'direct'
                    logger.debug(f"Listing transcripts for {video_id} via {proxy_str}")
                    
                    # Make the transcript list request
                    start_time = time.time()
                    transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
                    list_time = time.time() - start_time
                    
                    # Validate response
                    if transcript_list is None:
                        raise YouTubeTranscriptError(f"No transcript list returned for video {video_id}")
                    
                    # Count available transcripts
                    transcript_count = len(list(transcript_list))
                    
                    # Log successful retrieval with metrics
                    logger.info(f"Successfully listed transcripts for {video_id} "
                              f"(found: {transcript_count} transcripts, "
                              f"list_time: {list_time:.2f}s, "
                              f"proxy: {proxy_str})")
                    
                    return transcript_list
                    
                except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable) as e:
                    # These are expected YouTube API errors, don't retry
                    logger.warning(f"YouTube API error listing transcripts for {video_id}: {type(e).__name__}: {e}")
                    raise
                except TooManyRequests as e:
                    # Rate limiting error, should be retried with backoff
                    logger.warning(f"Rate limit hit listing transcripts for {video_id}: {e}")
                    raise
                except Exception as e:
                    # Unexpected error, log details and reraise for retry logic
                    logger.error(f"Unexpected error listing transcripts for {video_id}: {type(e).__name__}: {e}")
                    raise
                    
                finally:
                    # Always restore original behavior
                    self._restore_transcript_api()


class YouTubeTranscriptFetcher:
    """Handles YouTube transcript fetching and processing with comprehensive logging."""
    
    def __init__(self, enable_detailed_logging: bool = True, log_file: str = None):
        self.formatter = TextFormatter()
        self.api = ProxyAwareTranscriptApi()  # Use proxy-aware API
        self.metadata_extractor = YouTubeVideoMetadataExtractor()
        self.three_tier_strategy = ThreeTierTranscriptStrategy()
        
        # Initialize comprehensive logging
        if enable_detailed_logging:
            self.acquisition_logger = TranscriptAcquisitionLogger(
                setup_transcript_logging(log_file=log_file)
            )
        else:
            self.acquisition_logger = None
    
    @handle_youtube_api_exceptions
    def fetch_transcript(
        self, 
        video_id: str, 
        languages: Optional[List[str]] = None,
        include_metadata: bool = True,
        check_unsupported: bool = True,
        max_duration_seconds: int = 1800
    ) -> Dict[str, Any]:
        """
        Fetch transcript for a YouTube video.
        
        Args:
            video_id: YouTube video ID
            languages: Preferred languages for transcript (default: ['en'])
            include_metadata: Whether to include video metadata (default: True)
            check_unsupported: Whether to check for unsupported video types (default: True)
            max_duration_seconds: Maximum allowed duration in seconds (default: 1800 = 30 minutes)
            
        Returns:
            Dictionary containing transcript data and metadata
            
        Raises:
            YouTubeTranscriptError: If transcript cannot be fetched
            UnsupportedVideoTypeError: If video type is not supported
            VideoTooLongError: If video exceeds duration limit
        """
        if not video_id:
            raise YouTubeTranscriptError("Video ID is required")
            
        if not YouTubeURLValidator._is_valid_video_id(video_id):
            raise YouTubeTranscriptError(f"Invalid video ID format: {video_id}")
            
        if languages is None:
            languages = ['en', 'zh-TW', 'zh-CN', 'zh', 'ja', 'ko', 'fr', 'de', 'es', 'it', 'pt', 'ru']
        
        # Check for unsupported video types first
        if check_unsupported:
            try:
                support_status = self.metadata_extractor.detect_unsupported_video_type(
                    video_id, max_duration_seconds
                )
                
                if not support_status['is_supported']:
                    for issue in support_status['issues']:
                        if issue['type'] == 'private':
                            raise PrivateVideoError("Video is private and cannot be accessed")
                        elif issue['type'] == 'live':
                            raise LiveVideoError("Live videos do not have transcripts available")
                        elif issue['type'] == 'too_long':
                            raise VideoTooLongError(issue['message'])
                        
            except (PrivateVideoError, LiveVideoError, VideoTooLongError):
                raise  # Re-raise these specific errors
            except Exception as e:
                logger.warning(f"Could not check video support status for {video_id}: {str(e)}")
                # Continue with transcript fetch attempt
            
        try:
            # Fetch transcript
            transcript_list = self.api.get_transcript(video_id, languages=languages)
            
            # Format transcript
            formatted_transcript = self.formatter.format_transcript(transcript_list)
            
            # Extract additional metadata from transcript
            duration = self._calculate_duration(transcript_list)
            word_count = len(formatted_transcript.split())
            
            result = {
                'video_id': video_id,
                'transcript': formatted_transcript,
                'raw_transcript': transcript_list,
                'language': self._detect_language(transcript_list),
                'duration_seconds': duration,
                'word_count': word_count,
                'fetch_timestamp': datetime.utcnow().isoformat(),
                'success': True
            }
            
            # Include video metadata if requested
            if include_metadata:
                try:
                    video_metadata = self.metadata_extractor.extract_video_metadata(video_id)
                    result['video_metadata'] = video_metadata
                    
                    # Update duration if available from metadata (more accurate)
                    if video_metadata.get('duration_seconds'):
                        result['duration_seconds'] = video_metadata['duration_seconds']
                        
                except Exception as e:
                    logger.warning(f"Failed to extract video metadata for {video_id}: {str(e)}")
                    result['video_metadata'] = None
            
            return result
            
        except TranscriptsDisabled:
            logger.error(f"Transcripts disabled for video {video_id}")
            raise NoTranscriptAvailableError("Transcripts are disabled for this video")
            
        except NoTranscriptFound:
            logger.error(f"No transcript found for video {video_id}")
            raise NoTranscriptAvailableError(
                f"No transcript found for this video in languages: {languages}"
            )
            
        except VideoUnavailable:
            logger.error(f"Video {video_id} is unavailable")
            raise YouTubeTranscriptError("Video is unavailable")
            
        except TooManyRequests:
            logger.error(f"Rate limit exceeded for video {video_id}")
            raise YouTubeTranscriptError("Rate limit exceeded. Please try again later")
            
        except Exception as e:
            logger.error(f"Unexpected error fetching transcript for {video_id}: {str(e)}")
            raise YouTubeTranscriptError(f"Failed to fetch transcript: {str(e)}")
    
    def fetch_transcript_from_url(
        self, 
        url: str, 
        languages: Optional[List[str]] = None,
        include_metadata: bool = True,
        check_unsupported: bool = True,
        max_duration_seconds: int = 1800
    ) -> Dict[str, Any]:
        """
        Fetch transcript from a YouTube URL.
        
        Args:
            url: YouTube URL
            languages: Preferred languages for transcript
            include_metadata: Whether to include video metadata
            check_unsupported: Whether to check for unsupported video types
            max_duration_seconds: Maximum allowed duration in seconds (default: 1800 = 30 minutes)
            
        Returns:
            Dictionary containing transcript data and metadata
            
        Raises:
            YouTubeTranscriptError: If URL is invalid or transcript cannot be fetched
            UnsupportedVideoTypeError: If video type is not supported
            VideoTooLongError: If video exceeds duration limit
        """
        # Validate URL and extract video ID
        is_valid, video_id = YouTubeURLValidator.validate_and_extract(url)
        
        if not is_valid or not video_id:
            raise YouTubeTranscriptError(f"Invalid YouTube URL: {url}")
            
        return self.fetch_transcript(video_id, languages, include_metadata, check_unsupported, max_duration_seconds)
    
    @handle_youtube_api_exceptions
    def get_available_transcripts(self, video_id: str) -> List[Dict[str, Any]]:
        """
        Get list of available transcripts for a video.
        
        Args:
            video_id: YouTube video ID
            
        Returns:
            List of available transcript metadata
            
        Raises:
            YouTubeTranscriptError: If video is unavailable
        """
        if not video_id:
            raise YouTubeTranscriptError("Video ID is required")
            
        if not YouTubeURLValidator._is_valid_video_id(video_id):
            raise YouTubeTranscriptError(f"Invalid video ID format: {video_id}")
            
        try:
            transcript_list = self.api.list_transcripts(video_id)
            available_transcripts = []
            
            for transcript in transcript_list:
                available_transcripts.append({
                    'language': transcript.language,
                    'language_code': transcript.language_code,
                    'is_generated': transcript.is_generated,
                    'is_translatable': transcript.is_translatable
                })
                
            return available_transcripts
            
        except VideoUnavailable:
            logger.error(f"Video {video_id} is unavailable")
            raise YouTubeTranscriptError("Video is unavailable")
            
        except Exception as e:
            logger.error(f"Error getting available transcripts for {video_id}: {str(e)}")
            raise YouTubeTranscriptError(f"Failed to get available transcripts: {str(e)}")
    
    def get_video_metadata(self, video_id: str) -> Dict[str, Any]:
        """
        Get video metadata only (without transcript).
        
        Args:
            video_id: YouTube video ID
            
        Returns:
            Dictionary containing video metadata
            
        Raises:
            YouTubeTranscriptError: If metadata cannot be extracted
        """
        return self.metadata_extractor.extract_video_metadata(video_id)
    
    def get_video_metadata_from_url(self, url: str) -> Dict[str, Any]:
        """
        Get video metadata from a YouTube URL.
        
        Args:
            url: YouTube URL
            
        Returns:
            Dictionary containing video metadata
            
        Raises:
            YouTubeTranscriptError: If URL is invalid or metadata cannot be extracted
        """
        # Validate URL and extract video ID
        is_valid, video_id = YouTubeURLValidator.validate_and_extract(url)
        
        if not is_valid or not video_id:
            raise YouTubeTranscriptError(f"Invalid YouTube URL: {url}")
            
        return self.get_video_metadata(video_id)
    
    def check_video_support(self, video_id: str, max_duration_seconds: int = 1800) -> Dict[str, Any]:
        """
        Check if a video is supported for transcript extraction.
        
        Args:
            video_id: YouTube video ID
            max_duration_seconds: Maximum allowed duration in seconds (default: 1800 = 30 minutes)
            
        Returns:
            Dictionary containing video support status
            
        Raises:
            YouTubeTranscriptError: If video cannot be analyzed
        """
        return self.metadata_extractor.detect_unsupported_video_type(video_id, max_duration_seconds)
    
    def check_video_support_from_url(self, url: str, max_duration_seconds: int = 1800) -> Dict[str, Any]:
        """
        Check if a video is supported for transcript extraction from URL.
        
        Args:
            url: YouTube URL
            max_duration_seconds: Maximum allowed duration in seconds (default: 1800 = 30 minutes)
            
        Returns:
            Dictionary containing video support status
            
        Raises:
            YouTubeTranscriptError: If URL is invalid or video cannot be analyzed
        """
        # Validate URL and extract video ID
        is_valid, video_id = YouTubeURLValidator.validate_and_extract(url)
        
        if not is_valid or not video_id:
            raise YouTubeTranscriptError(f"Invalid YouTube URL: {url}")
            
        return self.check_video_support(video_id, max_duration_seconds)
    
    def check_transcript_availability(self, video_id: str) -> Dict[str, Any]:
        """
        Check transcript availability for a video.
        
        Args:
            video_id: YouTube video ID
            
        Returns:
            Dictionary containing transcript availability info
            
        Raises:
            YouTubeTranscriptError: If video cannot be analyzed
        """
        return self.metadata_extractor.check_transcript_availability(video_id)
    
    def check_transcript_availability_from_url(self, url: str) -> Dict[str, Any]:
        """
        Check transcript availability for a video from URL.
        
        Args:
            url: YouTube URL
            
        Returns:
            Dictionary containing transcript availability info
            
        Raises:
            YouTubeTranscriptError: If URL is invalid or video cannot be analyzed
        """
        # Validate URL and extract video ID
        is_valid, video_id = YouTubeURLValidator.validate_and_extract(url)
        
        if not is_valid or not video_id:
            raise YouTubeTranscriptError(f"Invalid YouTube URL: {url}")
            
        return self.check_transcript_availability(video_id)
    
    def validate_video_duration(
        self, 
        video_id: str, 
        max_duration_seconds: int = 1800
    ) -> Dict[str, Any]:
        """
        Validate video duration against maximum limit.
        
        Args:
            video_id: YouTube video ID
            max_duration_seconds: Maximum allowed duration in seconds (default: 1800 = 30 minutes)
            
        Returns:
            Dictionary containing duration validation results
            
        Raises:
            YouTubeTranscriptError: If video cannot be analyzed
        """
        return self.metadata_extractor.validate_video_duration(video_id, max_duration_seconds)
    
    def validate_video_duration_from_url(
        self, 
        url: str, 
        max_duration_seconds: int = 1800
    ) -> Dict[str, Any]:
        """
        Validate video duration from URL against maximum limit.
        
        Args:
            url: YouTube URL
            max_duration_seconds: Maximum allowed duration in seconds (default: 1800 = 30 minutes)
            
        Returns:
            Dictionary containing duration validation results
            
        Raises:
            YouTubeTranscriptError: If URL is invalid or video cannot be analyzed
        """
        # Validate URL and extract video ID
        is_valid, video_id = YouTubeURLValidator.validate_and_extract(url)
        
        if not is_valid or not video_id:
            raise YouTubeTranscriptError(f"Invalid YouTube URL: {url}")
            
        return self.validate_video_duration(video_id, max_duration_seconds)
    
    def check_duration_limit(
        self, 
        video_id: str, 
        max_duration_seconds: int = 1800,
        raise_on_exceeded: bool = True
    ) -> bool:
        """
        Check if video duration is within limits.
        
        Args:
            video_id: YouTube video ID
            max_duration_seconds: Maximum allowed duration in seconds (default: 1800 = 30 minutes)
            raise_on_exceeded: Whether to raise exception if duration is exceeded
            
        Returns:
            True if duration is within limits, False otherwise
            
        Raises:
            VideoTooLongError: If video exceeds duration and raise_on_exceeded is True
            YouTubeTranscriptError: If video cannot be analyzed
        """
        return self.metadata_extractor.check_duration_limit(video_id, max_duration_seconds, raise_on_exceeded)
    
    def check_duration_limit_from_url(
        self, 
        url: str, 
        max_duration_seconds: int = 1800,
        raise_on_exceeded: bool = True
    ) -> bool:
        """
        Check if video duration is within limits from URL.
        
        Args:
            url: YouTube URL
            max_duration_seconds: Maximum allowed duration in seconds (default: 1800 = 30 minutes)
            raise_on_exceeded: Whether to raise exception if duration is exceeded
            
        Returns:
            True if duration is within limits, False otherwise
            
        Raises:
            VideoTooLongError: If video exceeds duration and raise_on_exceeded is True
            YouTubeTranscriptError: If URL is invalid or video cannot be analyzed
        """
        # Validate URL and extract video ID
        is_valid, video_id = YouTubeURLValidator.validate_and_extract(url)
        
        if not is_valid or not video_id:
            raise YouTubeTranscriptError(f"Invalid YouTube URL: {url}")
            
        return self.check_duration_limit(video_id, max_duration_seconds, raise_on_exceeded)
    
    def _calculate_duration(self, transcript_list: List[Dict[str, Any]]) -> float:
        """
        Calculate video duration from transcript data.
        
        Args:
            transcript_list: Raw transcript data
            
        Returns:
            Duration in seconds
        """
        if not transcript_list:
            return 0.0
            
        try:
            # Get the last transcript entry
            last_entry = transcript_list[-1]
            return last_entry.get('start', 0.0) + last_entry.get('duration', 0.0)
        except (IndexError, KeyError):
            return 0.0
    
    def _detect_language(self, transcript_list: List[Dict[str, Any]]) -> str:
        """
        Detect language from transcript data.
        
        Args:
            transcript_list: Raw transcript data
            
        Returns:
            Language code or 'unknown'
        """
        # This is a simplified language detection
        # In a real implementation, you might use a language detection library
        if not transcript_list:
            return 'unknown'
            
        # Check for English patterns
        text_sample = ' '.join([entry.get('text', '') for entry in transcript_list[:10]])
        
        # Simple heuristic for English detection
        english_indicators = ['the', 'and', 'is', 'in', 'to', 'of', 'a', 'that', 'it', 'with']
        text_lower = text_sample.lower()
        
        english_count = sum(1 for word in english_indicators if word in text_lower)
        
        if english_count >= 3:
            return 'en'
        
        return 'unknown'
    
    def fetch_transcript_with_three_tier_strategy(
        self, 
        video_id: str, 
        preferred_languages: Optional[List[str]] = None,
        include_metadata: bool = True,
        check_unsupported: bool = True,
        max_duration_seconds: int = 1800,
        max_tier_attempts: int = 3,
        enable_intelligent_fallback: bool = True,
        fallback_retry_delay: float = 1.0
    ) -> Dict[str, Any]:
        """
        Fetch transcript using three-tier strategy with intelligent fallback logic.
        
        This method implements the enhanced transcript acquisition strategy that tries
        transcripts in order of quality preference, falling back to lower tiers if needed.
        The intelligent fallback logic automatically tries the next tier when the current 
        tier fails, with configurable retry delays and tier-specific retry strategies.
        
        Args:
            video_id: YouTube video ID
            preferred_languages: List of preferred language codes (auto-detected if None)
            include_metadata: Whether to include video metadata
            check_unsupported: Whether to check for unsupported video types
            max_duration_seconds: Maximum allowed duration in seconds
            max_tier_attempts: Maximum number of transcript options to try per tier
            enable_intelligent_fallback: Whether to use intelligent fallback logic
            fallback_retry_delay: Delay in seconds between fallback attempts
            
        Returns:
            Dictionary containing transcript data and three-tier metadata
            
        Raises:
            YouTubeTranscriptError: If transcript cannot be fetched
            UnsupportedVideoTypeError: If video type is not supported
            VideoTooLongError: If video exceeds duration limit
        """
        if not video_id:
            raise YouTubeTranscriptError("Video ID is required")
            
        if not YouTubeURLValidator._is_valid_video_id(video_id):
            raise YouTubeTranscriptError(f"Invalid video ID format: {video_id}")
        
        start_time = time.time()
        
        # Log acquisition start
        if self.acquisition_logger:
            self.acquisition_logger.log_acquisition_start(
                video_id, "intelligent_fallback" if enable_intelligent_fallback else "basic_fallback", 
                preferred_languages
            )
        
        logger.debug(f"Starting three-tier transcript acquisition for video {video_id}")
        
        # Check for unsupported video types first
        if check_unsupported:
            try:
                support_status = self.metadata_extractor.detect_unsupported_video_type(
                    video_id, max_duration_seconds
                )
                
                if not support_status['is_supported']:
                    # Log unsupported video details
                    if self.acquisition_logger:
                        self.acquisition_logger.log_unsupported_video(
                            video_id, "Video type not supported", support_status['issues']
                        )
                    
                    for issue in support_status['issues']:
                        if issue['type'] == 'private':
                            raise PrivateVideoError(video_id)
                        elif issue['type'] == 'live':
                            raise LiveVideoError(video_id)
                        elif issue['type'] == 'too_long':
                            raise VideoTooLongError(video_id)
                        
            except (PrivateVideoError, LiveVideoError, VideoTooLongError):
                raise  # Re-raise these specific errors
            except Exception as e:
                logger.warning(f"Could not check video support status for {video_id}: {str(e)}")
                # Continue with transcript fetch attempt
        
        # Get video metadata for language detection
        video_metadata = None
        if include_metadata:
            try:
                video_metadata = self.metadata_extractor.extract_video_metadata(video_id)
                if self.acquisition_logger:
                    self.acquisition_logger.log_video_metadata(video_id, video_metadata)
            except Exception as e:
                logger.warning(f"Failed to extract video metadata for {video_id}: {str(e)}")
                if self.acquisition_logger:
                    self.acquisition_logger.log_video_metadata(video_id, None)
        
        # Build three-tier strategy
        try:
            strategy_order = self.three_tier_strategy.get_transcript_strategy(
                video_id, preferred_languages, video_metadata
            )
            
            if not strategy_order:
                raise NoTranscriptAvailableError(video_id, [])
            
            # Log strategy planning
            if self.acquisition_logger:
                self.acquisition_logger.log_strategy_planning(
                    video_id, strategy_order, len(strategy_order)
                )
            
            logger.info(f"Three-tier strategy built: {len(strategy_order)} transcript options available")
            
        except Exception as e:
            logger.error(f"Failed to build three-tier strategy for {video_id}: {str(e)}")
            raise YouTubeTranscriptError(f"Failed to analyze transcript options: {str(e)}", video_id)
        
        # Intelligent fallback logic: Try transcripts in three-tier order with intelligent retry
        try:
            if enable_intelligent_fallback:
                result = self._execute_intelligent_fallback_strategy(
                    video_id, strategy_order, video_metadata, preferred_languages, 
                    max_tier_attempts, fallback_retry_delay
                )
            else:
                # Original fallback logic for backward compatibility
                result = self._execute_basic_fallback_strategy(
                    video_id, strategy_order, video_metadata, preferred_languages, 
                    max_tier_attempts
                )
            
            # Log successful completion
            if self.acquisition_logger:
                processing_time = time.time() - start_time
                tier_metadata = result.get('three_tier_metadata', {})
                self.acquisition_logger.log_acquisition_complete(
                    video_id, True, 
                    tier_metadata.get('selected_tier'),
                    tier_metadata.get('total_attempts', 0),
                    processing_time
                )
            
            return result
            
        except Exception as e:
            # Log failed completion
            if self.acquisition_logger:
                processing_time = time.time() - start_time
                # Extract attempt count from exception or metadata
                total_attempts = getattr(e, 'total_attempts', 0)
                self.acquisition_logger.log_acquisition_complete(
                    video_id, False, None, total_attempts, processing_time
                )
            raise
    
    def _execute_intelligent_fallback_strategy(
        self,
        video_id: str,
        strategy_order: List[TranscriptInfo],
        video_metadata: Optional[Dict[str, Any]],
        preferred_languages: Optional[List[str]],
        max_tier_attempts: int,
        fallback_retry_delay: float
    ) -> Dict[str, Any]:
        """
        Execute intelligent fallback strategy with tier-based retry logic.
        
        This method implements intelligent fallback logic that:
        1. Groups transcript options by tier
        2. Tries all options in a tier before moving to the next
        3. Implements tier-specific retry strategies
        4. Provides detailed fallback analytics
        
        Args:
            video_id: YouTube video ID
            strategy_order: Ordered list of transcript options
            video_metadata: Video metadata for analysis
            preferred_languages: Preferred language codes
            max_tier_attempts: Maximum attempts per tier
            fallback_retry_delay: Delay between fallback attempts
            
        Returns:
            Dictionary containing transcript data and fallback metadata
        """
        logger.info(f"Executing intelligent fallback strategy for {video_id}")
        
        # Group transcripts by tier for intelligent fallback
        tier_groups = self._group_transcripts_by_tier(strategy_order)
        
        # Track overall attempt statistics
        total_attempts = 0
        tier_attempts = {}
        all_attempts_made = []
        last_error = None
        
        # Try each tier in order: MANUAL -> AUTO_GENERATED -> TRANSLATED
        tier_order = [TranscriptTier.MANUAL, TranscriptTier.AUTO_GENERATED, TranscriptTier.TRANSLATED]
        
        for tier_index, tier in enumerate(tier_order):
            if tier not in tier_groups or not tier_groups[tier]:
                logger.debug(f"No transcripts available for tier {tier}")
                continue
                
            tier_transcripts = tier_groups[tier][:max_tier_attempts]
            tier_attempts[tier] = {'attempted': 0, 'failed': 0, 'errors': []}
            
            logger.info(f"Trying tier {tier} with {len(tier_transcripts)} options")
            
            for attempt_num, transcript_info in enumerate(tier_transcripts):
                total_attempts += 1
                tier_attempts[tier]['attempted'] += 1
                
                # Log attempt start
                if hasattr(self, 'acquisition_logger') and self.acquisition_logger:
                    self.acquisition_logger.log_tier_attempt_start(
                        video_id, tier, transcript_info.language_code,
                        total_attempts, tier_attempts[tier]['attempted']
                    )
                
                try:
                    logger.debug(f"Tier {tier} attempt {attempt_num + 1}: "
                               f"Trying {transcript_info.language_code}")
                    
                    # Add delay between attempts if not the first attempt
                    if total_attempts > 1 and fallback_retry_delay > 0:
                        time.sleep(fallback_retry_delay)
                    
                    # Fetch the transcript using the original transcript object
                    if transcript_info.original_transcript:
                        transcript_data = transcript_info.original_transcript.fetch()
                    else:
                        # Fallback to API call
                        transcript_data = self.api.get_transcript(video_id, [transcript_info.language_code])
                    
                    # Validate transcript data
                    if not transcript_data or len(transcript_data) == 0:
                        raise TranscriptValidationError(
                            "Empty transcript data received", 
                            video_id, 
                            "raw_data_empty"
                        )
                    
                    # Format transcript
                    try:
                        formatted_transcript = self.formatter.format_transcript(transcript_data)
                    except Exception as format_error:
                        raise TranscriptProcessingError(
                            f"Failed to format transcript: {str(format_error)}", 
                            video_id, 
                            "formatting"
                        ) from format_error
                    
                    # Validate formatted transcript
                    if not formatted_transcript or len(formatted_transcript.strip()) == 0:
                        raise TranscriptValidationError(
                            "Empty formatted transcript", 
                            video_id, 
                            "formatted_text_empty"
                        )
                    
                    # Extract additional metadata from transcript
                    duration = self._calculate_duration(transcript_data)
                    word_count = len(formatted_transcript.split())
                    
                    # Log successful attempt
                    if hasattr(self, 'acquisition_logger') and self.acquisition_logger:
                        self.acquisition_logger.log_tier_attempt_success(
                            video_id, tier, transcript_info.language_code,
                            total_attempts, word_count, duration
                        )
                    
                    # Success - build result with intelligent fallback metadata
                    result = {
                        'video_id': video_id,
                        'transcript': formatted_transcript,
                        'raw_transcript': transcript_data,
                        'language': transcript_info.language_code,
                        'duration_seconds': duration,
                        'word_count': word_count,
                        'fetch_timestamp': datetime.utcnow().isoformat(),
                        'success': True,
                        'three_tier_metadata': {
                            'selected_tier': transcript_info.tier,
                            'selected_language': transcript_info.language_code,
                            'quality_score': transcript_info.quality_score,
                            'attempt_number': total_attempts,
                            'tier_attempt_number': tier_attempts[tier]['attempted'],
                            'total_attempts': total_attempts,
                            'strategy_options': len(strategy_order),
                            'tier_summary': self.three_tier_strategy.get_transcript_tier_summary(
                                video_id, preferred_languages, video_metadata
                            ),
                            'intelligent_fallback_metadata': {
                                'fallback_enabled': True,
                                'tiers_tried': list(tier_attempts.keys()),
                                'tier_statistics': tier_attempts,
                                'successful_tier': tier,
                                'successful_tier_index': tier_index,
                                'total_tiers_available': len(tier_groups),
                                'fallback_retry_delay': fallback_retry_delay
                            },
                            'attempts_made': all_attempts_made + [{
                                'language': transcript_info.language_code,
                                'tier': transcript_info.tier,
                                'quality_score': transcript_info.quality_score,
                                'success': True,
                                'tier_attempt_number': tier_attempts[tier]['attempted']
                            }]
                        }
                    }
                    
                    # Include video metadata if available
                    if video_metadata:
                        result['video_metadata'] = video_metadata
                        
                        # Update duration if available from metadata (more accurate)
                        if video_metadata.get('duration_seconds'):
                            result['duration_seconds'] = video_metadata['duration_seconds']
                    
                    logger.info(f"Intelligent fallback successful for {video_id}: "
                               f"{transcript_info.language_code} ({transcript_info.tier}) "
                               f"after {total_attempts} attempts")
                    
                    return result
                    
                except Exception as e:
                    # Log failed attempt
                    if hasattr(self, 'acquisition_logger') and self.acquisition_logger:
                        self.acquisition_logger.log_tier_attempt_failure(
                            video_id, tier, transcript_info.language_code,
                            total_attempts, e
                        )
                    
                    # Record this attempt
                    attempt_record = {
                        'language': transcript_info.language_code,
                        'tier': transcript_info.tier,
                        'quality_score': transcript_info.quality_score,
                        'success': False,
                        'error': str(e),
                        'error_type': type(e).__name__,
                        'tier_attempt_number': tier_attempts[tier]['attempted']
                    }
                    all_attempts_made.append(attempt_record)
                    
                    # Track tier-specific errors
                    tier_attempts[tier]['failed'] += 1
                    tier_attempts[tier]['errors'].append({
                        'error': str(e),
                        'error_type': type(e).__name__,
                        'language': transcript_info.language_code
                    })
                    
                    last_error = e
                    logger.warning(f"Tier {tier} attempt {attempt_num + 1} failed for "
                                 f"{transcript_info.language_code}: {str(e)}")
                    
                    # Continue to next transcript option in current tier
                    continue
            
            # All attempts in this tier failed
            logger.warning(f"All attempts in tier {tier} failed, moving to next tier")
            
            # Log tier fallback if there's a next tier
            next_tier_index = tier_index + 1
            if next_tier_index < len(tier_order) and hasattr(self, 'acquisition_logger') and self.acquisition_logger:
                next_tier = tier_order[next_tier_index]
                if next_tier in tier_groups and tier_groups[next_tier]:
                    self.acquisition_logger.log_tier_fallback(
                        video_id, tier, next_tier, 
                        f"All {tier_attempts[tier]['attempted']} attempts failed"
                    )
        
        # All tiers and attempts failed
        available_languages = [info.language_code for info in strategy_order]
        
        logger.error(f"All intelligent fallback attempts failed for video {video_id}")
        
        # Provide detailed error information with intelligent fallback metadata
        error_metadata = {
            'total_attempts': total_attempts,
            'available_languages': available_languages,
            'attempts_made': all_attempts_made,
            'tier_summary': self.three_tier_strategy.get_transcript_tier_summary(
                video_id, preferred_languages, video_metadata
            ) if video_metadata else None,
            'intelligent_fallback_metadata': {
                'fallback_enabled': True,
                'tiers_tried': list(tier_attempts.keys()),
                'tier_statistics': tier_attempts,
                'successful_tier': None,
                'total_tiers_available': len(tier_groups),
                'fallback_retry_delay': fallback_retry_delay
            }
        }
        
        # Determine the most appropriate error to raise based on the last error encountered
        if isinstance(last_error, (TranscriptsDisabled, NoTranscriptFound, NoTranscriptAvailableError)):
            logger.error(f"No transcripts available for video {video_id}")
            raise NoTranscriptAvailableError(video_id, available_languages)
        elif isinstance(last_error, (PrivateVideoError, LiveVideoError, VideoTooLongError, 
                                   AgeRestrictedVideoError, RegionBlockedVideoError)):
            logger.error(f"Video {video_id} is unsupported: {str(last_error)}")
            raise last_error  # Re-raise the specific unsupported video error
        elif isinstance(last_error, VideoUnavailable):
            logger.error(f"Video {video_id} is unavailable: {str(last_error)}")
            raise VideoNotFoundError(video_id) from last_error
        elif isinstance(last_error, (TooManyRequests, RateLimitError)):
            logger.error(f"Rate limit exceeded for video {video_id}: {str(last_error)}")
            retry_after = getattr(last_error, 'retry_after', 60)
            raise RateLimitError(retry_after) from last_error
        elif isinstance(last_error, (TranscriptValidationError, TranscriptProcessingError)):
            logger.error(f"Transcript processing failed for video {video_id}: {str(last_error)}")
            raise last_error  # Re-raise the specific processing error
        elif isinstance(last_error, NetworkTimeoutError):
            logger.error(f"Network timeout for video {video_id}: {str(last_error)}")
            raise last_error  # Re-raise the timeout error
        else:
            logger.error(f"Unexpected error during intelligent fallback for video {video_id}: {str(last_error)}")
            raise TierStrategyError(
                f"Intelligent fallback failed after {total_attempts} attempts across "
                f"{len(tier_attempts)} tiers: {str(last_error)}", 
                video_id,
                "intelligent_fallback"
            ) from last_error
    
    def _execute_basic_fallback_strategy(
        self,
        video_id: str,
        strategy_order: List[TranscriptInfo],
        video_metadata: Optional[Dict[str, Any]],
        preferred_languages: Optional[List[str]],
        max_tier_attempts: int
    ) -> Dict[str, Any]:
        """
        Execute basic fallback strategy (original logic for backward compatibility).
        
        Args:
            video_id: YouTube video ID
            strategy_order: Ordered list of transcript options
            video_metadata: Video metadata for analysis
            preferred_languages: Preferred language codes
            max_tier_attempts: Maximum attempts per tier
            
        Returns:
            Dictionary containing transcript data and metadata
        """
        logger.info(f"Executing basic fallback strategy for {video_id}")
        
        # Try transcripts in three-tier order
        last_error = None
        attempts_made = []
        
        for attempt_num, transcript_info in enumerate(strategy_order[:max_tier_attempts * 3]):
            try:
                logger.debug(f"Attempt {attempt_num + 1}: Trying {transcript_info.language_code} ({transcript_info.tier})")
                
                # Fetch the transcript using the original transcript object
                if transcript_info.original_transcript:
                    transcript_data = transcript_info.original_transcript.fetch()
                else:
                    # Fallback to API call
                    transcript_data = self.api.get_transcript(video_id, [transcript_info.language_code])
                
                # Format transcript
                formatted_transcript = self.formatter.format_transcript(transcript_data)
                
                # Extract additional metadata from transcript
                duration = self._calculate_duration(transcript_data)
                word_count = len(formatted_transcript.split())
                
                # Success - build result
                result = {
                    'video_id': video_id,
                    'transcript': formatted_transcript,
                    'raw_transcript': transcript_data,
                    'language': transcript_info.language_code,
                    'duration_seconds': duration,
                    'word_count': word_count,
                    'fetch_timestamp': datetime.utcnow().isoformat(),
                    'success': True,
                    'three_tier_metadata': {
                        'selected_tier': transcript_info.tier,
                        'selected_language': transcript_info.language_code,
                        'quality_score': transcript_info.quality_score,
                        'attempt_number': attempt_num + 1,
                        'total_attempts': len(attempts_made) + 1,
                        'strategy_options': len(strategy_order),
                        'tier_summary': self.three_tier_strategy.get_transcript_tier_summary(
                            video_id, preferred_languages, video_metadata
                        ),
                        'intelligent_fallback_metadata': {
                            'fallback_enabled': False,
                            'strategy_type': 'basic'
                        },
                        'attempts_made': attempts_made + [{
                            'language': transcript_info.language_code,
                            'tier': transcript_info.tier,
                            'quality_score': transcript_info.quality_score,
                            'success': True
                        }]
                    }
                }
                
                # Include video metadata if available
                if video_metadata:
                    result['video_metadata'] = video_metadata
                    
                    # Update duration if available from metadata (more accurate)
                    if video_metadata.get('duration_seconds'):
                        result['duration_seconds'] = video_metadata['duration_seconds']
                
                logger.info(f"Basic fallback successful for {video_id}: "
                           f"{transcript_info.language_code} ({transcript_info.tier})")
                
                return result
                
            except Exception as e:
                # Record this attempt
                attempt_record = {
                    'language': transcript_info.language_code,
                    'tier': transcript_info.tier,
                    'quality_score': transcript_info.quality_score,
                    'success': False,
                    'error': str(e),
                    'error_type': type(e).__name__
                }
                attempts_made.append(attempt_record)
                
                last_error = e
                logger.warning(f"Attempt {attempt_num + 1} failed for {transcript_info.language_code} "
                              f"({transcript_info.tier}): {str(e)}")
                
                # Continue to next transcript option
                continue
        
        # All attempts failed
        available_languages = [info.language_code for info in strategy_order]
        
        logger.error(f"All basic fallback attempts failed for video {video_id}")
        
        # Provide detailed error information
        error_summary = {
            'total_attempts': len(attempts_made),
            'available_languages': available_languages,
            'attempts_made': attempts_made,
            'tier_summary': self.three_tier_strategy.get_transcript_tier_summary(
                video_id, preferred_languages, video_metadata
            ) if video_metadata else None,
            'intelligent_fallback_metadata': {
                'fallback_enabled': False,
                'strategy_type': 'basic'
            }
        }
        
        # Determine the most appropriate error to raise based on the last error encountered
        if isinstance(last_error, (TranscriptsDisabled, NoTranscriptFound, NoTranscriptAvailableError)):
            logger.error(f"No transcripts available for video {video_id}")
            raise NoTranscriptAvailableError(video_id, available_languages)
        elif isinstance(last_error, (PrivateVideoError, LiveVideoError, VideoTooLongError, 
                                   AgeRestrictedVideoError, RegionBlockedVideoError)):
            logger.error(f"Video {video_id} is unsupported: {str(last_error)}")
            raise last_error  # Re-raise the specific unsupported video error
        elif isinstance(last_error, VideoUnavailable):
            logger.error(f"Video {video_id} is unavailable: {str(last_error)}")
            raise VideoNotFoundError(video_id) from last_error
        elif isinstance(last_error, (TooManyRequests, RateLimitError)):
            logger.error(f"Rate limit exceeded for video {video_id}: {str(last_error)}")
            retry_after = getattr(last_error, 'retry_after', 60)
            raise RateLimitError(retry_after) from last_error
        elif isinstance(last_error, (TranscriptValidationError, TranscriptProcessingError)):
            logger.error(f"Transcript processing failed for video {video_id}: {str(last_error)}")
            raise last_error  # Re-raise the specific processing error
        elif isinstance(last_error, NetworkTimeoutError):
            logger.error(f"Network timeout for video {video_id}: {str(last_error)}")
            raise last_error  # Re-raise the timeout error
        else:
            logger.error(f"Unexpected error during basic fallback for video {video_id}: {str(last_error)}")
            raise TierStrategyError(
                f"Basic fallback failed after {len(attempts_made)} attempts: {str(last_error)}", 
                video_id,
                "basic_fallback"
            ) from last_error
    
    def _group_transcripts_by_tier(self, strategy_order: List[TranscriptInfo]) -> Dict[str, List[TranscriptInfo]]:
        """
        Group transcript options by tier for intelligent fallback.
        
        Args:
            strategy_order: Ordered list of transcript options
            
        Returns:
            Dictionary with tiers as keys and lists of transcript options as values
        """
        tier_groups = {
            TranscriptTier.MANUAL: [],
            TranscriptTier.AUTO_GENERATED: [],
            TranscriptTier.TRANSLATED: []
        }
        
        for transcript_info in strategy_order:
            tier_groups[transcript_info.tier].append(transcript_info)
        
        return tier_groups


class YouTubeAPIErrorHandler:
    """
    Centralized error handling and reporting for YouTube API operations.
    
    This class provides utilities for error categorization, recovery suggestions,
    and detailed error reporting for monitoring and debugging purposes.
    """
    
    def __init__(self):
        self.error_statistics = {
            'total_errors': 0,
            'error_types': {},
            'video_errors': {},
            'recovery_attempts': 0,
            'successful_recoveries': 0
        }
    
    def categorize_error(self, error: Exception, video_id: str = "") -> Dict[str, Any]:
        """
        Categorize a YouTube API error and provide recovery suggestions.
        
        Args:
            error: The exception that was raised
            video_id: YouTube video ID (if available)
            
        Returns:
            Dictionary containing error category and recovery information
        """
        error_category = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'video_id': video_id,
            'severity': 'unknown',
            'category': 'unknown',
            'recoverable': False,
            'recovery_suggestions': [],
            'retry_recommended': False,
            'retry_delay': 0,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Categorize by error type
        if isinstance(error, (TranscriptsDisabled, NoTranscriptFound, NoTranscriptAvailableError)):
            error_category.update({
                'severity': 'high',
                'category': 'transcript_unavailable',
                'recoverable': False,
                'recovery_suggestions': [
                    'Video may not have transcripts enabled',
                    'Try alternative videos from the same channel',
                    'Check if manual captions are available'
                ]
            })
        elif isinstance(error, (PrivateVideoError, AgeRestrictedVideoError, RegionBlockedVideoError)):
            error_category.update({
                'severity': 'high',
                'category': 'access_restricted',
                'recoverable': False,
                'recovery_suggestions': [
                    'Video access is restricted',
                    'Try different video sources',
                    'Check video availability in different regions'
                ]
            })
        elif isinstance(error, (LiveVideoError, VideoTooLongError)):
            error_category.update({
                'severity': 'medium',
                'category': 'unsupported_content',
                'recoverable': False,
                'recovery_suggestions': [
                    'Video type is not supported for transcript extraction',
                    'Try videos with standard duration limits',
                    'Avoid live streams and premieres'
                ]
            })
        elif isinstance(error, (RateLimitError, TooManyRequests)):
            retry_after = getattr(error, 'retry_after', 60)
            error_category.update({
                'severity': 'medium',
                'category': 'rate_limiting',
                'recoverable': True,
                'recovery_suggestions': [
                    f'Wait {retry_after} seconds before retrying',
                    'Implement exponential backoff',
                    'Use proxy rotation if available'
                ],
                'retry_recommended': True,
                'retry_delay': retry_after
            })
        elif isinstance(error, NetworkTimeoutError):
            error_category.update({
                'severity': 'low',
                'category': 'network_issue',
                'recoverable': True,
                'recovery_suggestions': [
                    'Retry with longer timeout',
                    'Check network connectivity',
                    'Try alternative network routes'
                ],
                'retry_recommended': True,
                'retry_delay': 5
            })
        elif isinstance(error, (TranscriptValidationError, TranscriptProcessingError)):
            error_category.update({
                'severity': 'medium',
                'category': 'processing_error',
                'recoverable': True,
                'recovery_suggestions': [
                    'Try different transcript language',
                    'Validate transcript format',
                    'Check for corrupt transcript data'
                ],
                'retry_recommended': True,
                'retry_delay': 2
            })
        elif isinstance(error, VideoNotFoundError):
            error_category.update({
                'severity': 'high',
                'category': 'video_unavailable',
                'recoverable': False,
                'recovery_suggestions': [
                    'Video may have been deleted or made private',
                    'Verify video ID is correct',
                    'Try alternative video sources'
                ]
            })
        else:
            error_category.update({
                'severity': 'unknown',
                'category': 'unexpected_error',
                'recoverable': True,
                'recovery_suggestions': [
                    'Review error details for specific issues',
                    'Check API documentation for updates',
                    'Consider reporting the error for investigation'
                ],
                'retry_recommended': True,
                'retry_delay': 10
            })
        
        # Update statistics
        self._update_error_statistics(error_category)
        
        return error_category
    
    def _update_error_statistics(self, error_category: Dict[str, Any]) -> None:
        """Update internal error statistics."""
        self.error_statistics['total_errors'] += 1
        
        error_type = error_category['error_type']
        if error_type not in self.error_statistics['error_types']:
            self.error_statistics['error_types'][error_type] = 0
        self.error_statistics['error_types'][error_type] += 1
        
        video_id = error_category.get('video_id', 'unknown')
        if video_id and video_id != 'unknown':
            if video_id not in self.error_statistics['video_errors']:
                self.error_statistics['video_errors'][video_id] = 0
            self.error_statistics['video_errors'][video_id] += 1
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get current error statistics."""
        return self.error_statistics.copy()
    
    def reset_statistics(self) -> None:
        """Reset error statistics."""
        self.error_statistics = {
            'total_errors': 0,
            'error_types': {},
            'video_errors': {},
            'recovery_attempts': 0,
            'successful_recoveries': 0
        }
    
    def should_retry_error(self, error: Exception) -> Tuple[bool, int]:
        """
        Determine if an error should be retried and suggest delay.
        
        Args:
            error: The exception to analyze
            
        Returns:
            Tuple of (should_retry, delay_seconds)
        """
        error_info = self.categorize_error(error)
        return error_info['retry_recommended'], error_info['retry_delay']
    
    def generate_error_report(self, errors: List[Exception], video_id: str = "") -> Dict[str, Any]:
        """
        Generate comprehensive error report for multiple errors.
        
        Args:
            errors: List of exceptions encountered
            video_id: YouTube video ID (if available)
            
        Returns:
            Dictionary containing comprehensive error analysis
        """
        report = {
            'video_id': video_id,
            'total_errors': len(errors),
            'error_categories': {},
            'recovery_analysis': {
                'recoverable_errors': 0,
                'unrecoverable_errors': 0,
                'retry_recommended': False,
                'suggested_retry_delay': 0
            },
            'error_details': [],
            'timestamp': datetime.utcnow().isoformat()
        }
        
        recoverable_delays = []
        
        for error in errors:
            error_info = self.categorize_error(error, video_id)
            report['error_details'].append(error_info)
            
            category = error_info['category']
            if category not in report['error_categories']:
                report['error_categories'][category] = 0
            report['error_categories'][category] += 1
            
            if error_info['recoverable']:
                report['recovery_analysis']['recoverable_errors'] += 1
                if error_info['retry_recommended']:
                    recoverable_delays.append(error_info['retry_delay'])
            else:
                report['recovery_analysis']['unrecoverable_errors'] += 1
        
        # Determine overall retry recommendation
        if recoverable_delays:
            report['recovery_analysis']['retry_recommended'] = True
            report['recovery_analysis']['suggested_retry_delay'] = max(recoverable_delays)
        
        return report


# Convenience functions
def fetch_youtube_transcript(
    video_id: str, 
    languages: Optional[List[str]] = None,
    include_metadata: bool = True,
    check_unsupported: bool = True,
    max_duration_seconds: int = 1800
) -> Dict[str, Any]:
    """
    Convenience function to fetch YouTube transcript.
    
    Args:
        video_id: YouTube video ID
        languages: Preferred languages for transcript
        include_metadata: Whether to include video metadata
        check_unsupported: Whether to check for unsupported video types
        max_duration_seconds: Maximum allowed duration in seconds (default: 1800 = 30 minutes)
        
    Returns:
        Dictionary containing transcript data and metadata
    """
    fetcher = YouTubeTranscriptFetcher()
    return fetcher.fetch_transcript(video_id, languages, include_metadata, check_unsupported, max_duration_seconds)


def fetch_youtube_transcript_from_url(
    url: str, 
    languages: Optional[List[str]] = None,
    include_metadata: bool = True,
    check_unsupported: bool = True,
    max_duration_seconds: int = 1800
) -> Dict[str, Any]:
    """
    Convenience function to fetch YouTube transcript from URL.
    
    Args:
        url: YouTube URL
        languages: Preferred languages for transcript
        include_metadata: Whether to include video metadata
        check_unsupported: Whether to check for unsupported video types
        max_duration_seconds: Maximum allowed duration in seconds (default: 1800 = 30 minutes)
        
    Returns:
        Dictionary containing transcript data and metadata
    """
    fetcher = YouTubeTranscriptFetcher()
    return fetcher.fetch_transcript_from_url(url, languages, include_metadata, check_unsupported, max_duration_seconds)


def get_available_youtube_transcripts(video_id: str) -> List[Dict[str, Any]]:
    """
    Convenience function to get available transcripts for a video.
    
    Args:
        video_id: YouTube video ID
        
    Returns:
        List of available transcript metadata
    """
    fetcher = YouTubeTranscriptFetcher()
    return fetcher.get_available_transcripts(video_id)


def get_youtube_video_metadata(video_id: str) -> Dict[str, Any]:
    """
    Convenience function to get YouTube video metadata.
    
    Args:
        video_id: YouTube video ID
        
    Returns:
        Dictionary containing video metadata
    """
    fetcher = YouTubeTranscriptFetcher()
    return fetcher.get_video_metadata(video_id)


def get_youtube_video_metadata_from_url(url: str) -> Dict[str, Any]:
    """
    Convenience function to get YouTube video metadata from URL.
    
    Args:
        url: YouTube URL
        
    Returns:
        Dictionary containing video metadata
    """
    fetcher = YouTubeTranscriptFetcher()
    return fetcher.get_video_metadata_from_url(url)


def check_youtube_video_support(video_id: str, max_duration_seconds: int = 1800) -> Dict[str, Any]:
    """
    Convenience function to check if a YouTube video is supported.
    
    Args:
        video_id: YouTube video ID
        max_duration_seconds: Maximum allowed duration in seconds (default: 1800 = 30 minutes)
        
    Returns:
        Dictionary containing video support status
    """
    fetcher = YouTubeTranscriptFetcher()
    return fetcher.check_video_support(video_id, max_duration_seconds)


def check_youtube_video_support_from_url(url: str, max_duration_seconds: int = 1800) -> Dict[str, Any]:
    """
    Convenience function to check if a YouTube video is supported from URL.
    
    Args:
        url: YouTube URL
        max_duration_seconds: Maximum allowed duration in seconds (default: 1800 = 30 minutes)
        
    Returns:
        Dictionary containing video support status
    """
    fetcher = YouTubeTranscriptFetcher()
    return fetcher.check_video_support_from_url(url, max_duration_seconds)


def check_youtube_transcript_availability(video_id: str) -> Dict[str, Any]:
    """
    Convenience function to check YouTube transcript availability.
    
    Args:
        video_id: YouTube video ID
        
    Returns:
        Dictionary containing transcript availability info
    """
    fetcher = YouTubeTranscriptFetcher()
    return fetcher.check_transcript_availability(video_id)


def check_youtube_transcript_availability_from_url(url: str) -> Dict[str, Any]:
    """
    Convenience function to check YouTube transcript availability from URL.
    
    Args:
        url: YouTube URL
        
    Returns:
        Dictionary containing transcript availability info
    """
    fetcher = YouTubeTranscriptFetcher()
    return fetcher.check_transcript_availability_from_url(url)


def validate_youtube_video_duration(video_id: str, max_duration_seconds: int = 1800) -> Dict[str, Any]:
    """
    Convenience function to validate YouTube video duration.
    
    Args:
        video_id: YouTube video ID
        max_duration_seconds: Maximum allowed duration in seconds (default: 1800 = 30 minutes)
        
    Returns:
        Dictionary containing duration validation results
    """
    fetcher = YouTubeTranscriptFetcher()
    return fetcher.validate_video_duration(video_id, max_duration_seconds)


def validate_youtube_video_duration_from_url(url: str, max_duration_seconds: int = 1800) -> Dict[str, Any]:
    """
    Convenience function to validate YouTube video duration from URL.
    
    Args:
        url: YouTube URL
        max_duration_seconds: Maximum allowed duration in seconds (default: 1800 = 30 minutes)
        
    Returns:
        Dictionary containing duration validation results
    """
    fetcher = YouTubeTranscriptFetcher()
    return fetcher.validate_video_duration_from_url(url, max_duration_seconds)


def check_youtube_duration_limit(
    video_id: str, 
    max_duration_seconds: int = 1800, 
    raise_on_exceeded: bool = True
) -> bool:
    """
    Convenience function to check YouTube video duration limit.
    
    Args:
        video_id: YouTube video ID
        max_duration_seconds: Maximum allowed duration in seconds (default: 1800 = 30 minutes)
        raise_on_exceeded: Whether to raise exception if duration is exceeded
        
    Returns:
        True if duration is within limits, False otherwise
    """
    fetcher = YouTubeTranscriptFetcher()
    return fetcher.check_duration_limit(video_id, max_duration_seconds, raise_on_exceeded)


def check_youtube_duration_limit_from_url(
    url: str, 
    max_duration_seconds: int = 1800, 
    raise_on_exceeded: bool = True
) -> bool:
    """
    Convenience function to check YouTube video duration limit from URL.
    
    Args:
        url: YouTube URL
        max_duration_seconds: Maximum allowed duration in seconds (default: 1800 = 30 minutes)
        raise_on_exceeded: Whether to raise exception if duration is exceeded
        
    Returns:
        True if duration is within limits, False otherwise
    """
    fetcher = YouTubeTranscriptFetcher()
    return fetcher.check_duration_limit_from_url(url, max_duration_seconds, raise_on_exceeded)


def fetch_youtube_transcript_with_three_tier_strategy(
    video_id: str, 
    preferred_languages: Optional[List[str]] = None,
    include_metadata: bool = True,
    check_unsupported: bool = True,
    max_duration_seconds: int = 1800,
    max_tier_attempts: int = 3,
    enable_intelligent_fallback: bool = True,
    fallback_retry_delay: float = 1.0
) -> Dict[str, Any]:
    """
    Convenience function to fetch YouTube transcript using three-tier strategy with intelligent fallback.
    
    Args:
        video_id: YouTube video ID
        preferred_languages: List of preferred language codes (auto-detected if None)
        include_metadata: Whether to include video metadata
        check_unsupported: Whether to check for unsupported video types
        max_duration_seconds: Maximum allowed duration in seconds
        max_tier_attempts: Maximum number of transcript options to try per tier
        enable_intelligent_fallback: Whether to use intelligent fallback logic
        fallback_retry_delay: Delay in seconds between fallback attempts
        
    Returns:
        Dictionary containing transcript data and three-tier metadata
    """
    fetcher = YouTubeTranscriptFetcher()
    return fetcher.fetch_transcript_with_three_tier_strategy(
        video_id, preferred_languages, include_metadata, check_unsupported, 
        max_duration_seconds, max_tier_attempts, enable_intelligent_fallback, 
        fallback_retry_delay
    )


def get_youtube_transcript_tier_summary(
    video_id: str, 
    preferred_languages: Optional[List[str]] = None,
    video_metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Convenience function to get YouTube transcript tier summary.
    
    Args:
        video_id: YouTube video ID
        preferred_languages: List of preferred language codes
        video_metadata: Optional video metadata for language detection
        
    Returns:
        Dictionary containing transcript tier summary
    """
    strategy = ThreeTierTranscriptStrategy()
    return strategy.get_transcript_tier_summary(video_id, preferred_languages, video_metadata)


def get_video_info(
    video_url_or_id: str,
    include_transcript: bool = True,
    preferred_languages: Optional[List[str]] = None,
    use_enhanced_acquisition: bool = True,
    max_duration_seconds: int = 1800,
    enable_detailed_logging: bool = True
) -> Dict[str, Any]:
    """
    Enhanced get_video_info function that uses the new enhanced acquisition system.
    
    This function provides a simple interface to extract comprehensive video information
    including metadata, transcript data, and acquisition details using the three-tier
    strategy for improved reliability and quality.
    
    Args:
        video_url_or_id: YouTube URL or video ID
        include_transcript: Whether to fetch transcript data
        preferred_languages: List of preferred language codes (auto-detected if None)
        use_enhanced_acquisition: Whether to use three-tier strategy (recommended)
        max_duration_seconds: Maximum allowed video duration
        enable_detailed_logging: Whether to enable comprehensive logging
        
    Returns:
        Dictionary containing comprehensive video information
        
    Raises:
        YouTubeTranscriptError: If video processing fails
        UnsupportedVideoTypeError: If video type is not supported
    """
    # Initialize fetcher with logging configuration
    fetcher = YouTubeTranscriptFetcher(enable_detailed_logging=enable_detailed_logging)
    
    # Extract video ID if URL provided
    if video_url_or_id.startswith(('http', 'www')):
        is_valid, video_id = YouTubeURLValidator.validate_and_extract(video_url_or_id)
        if not is_valid or not video_id:
            raise YouTubeTranscriptError(f"Invalid YouTube URL: {video_url_or_id}")
    else:
        video_id = video_url_or_id
        if not YouTubeURLValidator._is_valid_video_id(video_id):
            raise YouTubeTranscriptError(f"Invalid video ID format: {video_id}")
    
    # Start with basic video information
    result = {
        'video_id': video_id,
        'video_url': f"https://www.youtube.com/watch?v={video_id}",
        'processing_timestamp': datetime.utcnow().isoformat(),
        'enhanced_acquisition_used': use_enhanced_acquisition
    }
    
    try:
        # Get video metadata
        video_metadata = fetcher.get_video_metadata(video_id)
        result['video_metadata'] = video_metadata
        
        # Check video support
        support_status = fetcher.check_video_support(video_id, max_duration_seconds)
        result['support_status'] = support_status
        
        if not support_status['is_supported']:
            result['error'] = {
                'type': 'unsupported_video',
                'message': 'Video type is not supported',
                'issues': support_status['issues']
            }
            return result
        
        # Check transcript availability
        transcript_availability = fetcher.check_transcript_availability(video_id)
        result['transcript_availability'] = transcript_availability
        
        if not transcript_availability['has_transcripts']:
            result['error'] = {
                'type': 'no_transcripts',
                'message': 'No transcripts available for this video',
                'available_transcripts': transcript_availability['available_transcripts']
            }
            return result
        
        # Fetch transcript if requested
        if include_transcript:
            if use_enhanced_acquisition:
                # Use enhanced three-tier strategy
                transcript_data = fetcher.fetch_transcript_with_three_tier_strategy(
                    video_id=video_id,
                    preferred_languages=preferred_languages,
                    include_metadata=False,  # Already have metadata
                    check_unsupported=False,  # Already checked
                    max_duration_seconds=max_duration_seconds,
                    max_tier_attempts=3,
                    enable_intelligent_fallback=True,
                    fallback_retry_delay=1.0
                )
            else:
                # Use basic acquisition
                transcript_data = fetcher.fetch_transcript(
                    video_id=video_id,
                    languages=preferred_languages or ['en'],
                    include_metadata=False,  # Already have metadata
                    check_unsupported=False,  # Already checked
                    max_duration_seconds=max_duration_seconds
                )
            
            result['transcript_data'] = {
                'transcript_text': transcript_data['transcript'],
                'raw_transcript': transcript_data['raw_transcript'],
                'language': transcript_data['language'],
                'duration_seconds': transcript_data['duration_seconds'],
                'word_count': transcript_data['word_count'],
                'acquisition_method': 'enhanced_three_tier' if use_enhanced_acquisition else 'basic'
            }
            
            # Add three-tier metadata if available
            if use_enhanced_acquisition and 'three_tier_metadata' in transcript_data:
                three_tier_metadata = transcript_data['three_tier_metadata']
                result['three_tier_acquisition'] = {
                    'selected_tier': three_tier_metadata.get('selected_tier'),
                    'selected_language': three_tier_metadata.get('selected_language'),
                    'quality_score': three_tier_metadata.get('quality_score'),
                    'total_attempts': three_tier_metadata.get('total_attempts', 0),
                    'strategy_options': three_tier_metadata.get('strategy_options', 0),
                    'tier_summary': three_tier_metadata.get('tier_summary', {}),
                    'intelligent_fallback_used': three_tier_metadata.get('intelligent_fallback_metadata', {}).get('fallback_enabled', False),
                    'tiers_tried': three_tier_metadata.get('intelligent_fallback_metadata', {}).get('tiers_tried', []),
                    'success_tier_index': three_tier_metadata.get('intelligent_fallback_metadata', {}).get('successful_tier_index'),
                    'attempts_made': three_tier_metadata.get('attempts_made', [])
                }
        
        result['success'] = True
        return result
        
    except Exception as e:
        # Handle and categorize errors
        error_handler = YouTubeAPIErrorHandler()
        error_info = error_handler.categorize_error(e, video_id)
        
        result['success'] = False
        result['error'] = {
            'type': error_info['error_type'],
            'message': error_info['error_message'],
            'category': error_info['category'],
            'severity': error_info['severity'],
            'recoverable': error_info['recoverable'],
            'recovery_suggestions': error_info['recovery_suggestions'],
            'timestamp': error_info['timestamp']
        }
        
        # Include partial results if available
        if 'video_metadata' in result:
            result['partial_success'] = True
        
        return result


# Backward compatibility alias
def get_video_info_legacy(video_id: str, languages: List[str] = None) -> Dict[str, Any]:
    """
    Legacy get_video_info function for backward compatibility.
    
    This function maintains compatibility with existing code while internally
    using the enhanced acquisition system.
    
    Args:
        video_id: YouTube video ID
        languages: List of preferred languages (deprecated, use preferred_languages)
        
    Returns:
        Dictionary containing video information (legacy format)
    """
    return get_video_info(
        video_url_or_id=video_id,
        include_transcript=True,
        preferred_languages=languages,
        use_enhanced_acquisition=True,
        enable_detailed_logging=False  # Disable for legacy usage
    )