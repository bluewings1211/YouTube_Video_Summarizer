"""
YouTube API utility functions and helpers.

This module contains standalone utility functions and decorators
that support the YouTube API integration.
"""

import logging
from typing import Dict, Any, List, Optional
from functools import wraps

from .youtube_core import YouTubeTranscriptError, RateLimitError, VideoNotFoundError
from .youtube_fetcher import YouTubeUnifiedFetcher
from .validators import YouTubeURLValidator

logger = logging.getLogger(__name__)


# Global fetcher instance for utility functions
_global_fetcher = None


def get_global_fetcher() -> YouTubeUnifiedFetcher:
    """Get or create global YouTube fetcher instance."""
    global _global_fetcher
    if _global_fetcher is None:
        _global_fetcher = YouTubeUnifiedFetcher(enable_detailed_logging=True)
    return _global_fetcher


# Utility functions for external use
def get_youtube_video_metadata(video_id: str) -> Dict[str, Any]:
    """
    Get YouTube video metadata.
    
    Args:
        video_id: YouTube video ID
        
    Returns:
        Dictionary containing video metadata
        
    Raises:
        YouTubeTranscriptError: If metadata cannot be fetched
    """
    fetcher = get_global_fetcher()
    return fetcher.get_video_metadata(video_id)


def get_youtube_video_metadata_from_url(url: str) -> Dict[str, Any]:
    """
    Get YouTube video metadata from URL.
    
    Args:
        url: YouTube video URL
        
    Returns:
        Dictionary containing video metadata
        
    Raises:
        YouTubeTranscriptError: If URL is invalid or metadata cannot be fetched
    """
    # Validate URL and extract video ID
    is_valid, video_id = YouTubeURLValidator.validate_and_extract(url)
    
    if not is_valid or not video_id:
        raise YouTubeTranscriptError(f"Invalid YouTube URL: {url}")
    
    return get_youtube_video_metadata(video_id)


def get_available_youtube_transcripts(video_id: str) -> List[Dict[str, Any]]:
    """
    Get available transcripts for a YouTube video.
    
    Args:
        video_id: YouTube video ID
        
    Returns:
        List of available transcript information
        
    Raises:
        YouTubeTranscriptError: If transcripts cannot be retrieved
    """
    fetcher = get_global_fetcher()
    return fetcher.get_available_transcripts(video_id)


def fetch_youtube_transcript(
    video_id: str, 
    languages: Optional[List[str]] = None,
    include_metadata: bool = True,
    check_unsupported: bool = True,
    max_duration_seconds: int = 1800
) -> Dict[str, Any]:
    """
    Fetch YouTube transcript with metadata.
    
    Args:
        video_id: YouTube video ID
        languages: Preferred languages for transcript
        include_metadata: Whether to include video metadata
        check_unsupported: Whether to check for unsupported video types
        max_duration_seconds: Maximum allowed duration in seconds
        
    Returns:
        Dictionary containing transcript data and metadata
        
    Raises:
        YouTubeTranscriptError: If transcript cannot be fetched
    """
    fetcher = get_global_fetcher()
    return fetcher.fetch_transcript(
        video_id, languages, include_metadata, check_unsupported, max_duration_seconds
    )


def fetch_youtube_transcript_from_url(
    url: str,
    languages: Optional[List[str]] = None,
    include_metadata: bool = True,
    check_unsupported: bool = True,
    max_duration_seconds: int = 1800
) -> Dict[str, Any]:
    """
    Fetch YouTube transcript from URL.
    
    Args:
        url: YouTube video URL
        languages: Preferred languages for transcript
        include_metadata: Whether to include video metadata
        check_unsupported: Whether to check for unsupported video types
        max_duration_seconds: Maximum allowed duration in seconds
        
    Returns:
        Dictionary containing transcript data and metadata
        
    Raises:
        YouTubeTranscriptError: If URL is invalid or transcript cannot be fetched
    """
    # Validate URL and extract video ID
    is_valid, video_id = YouTubeURLValidator.validate_and_extract(url)
    
    if not is_valid or not video_id:
        raise YouTubeTranscriptError(f"Invalid YouTube URL: {url}")
    
    return fetch_youtube_transcript(
        video_id, languages, include_metadata, check_unsupported, max_duration_seconds
    )


def fetch_youtube_data_unified(
    video_id: str,
    languages: Optional[List[str]] = None,
    max_duration_seconds: int = 1800,
    check_unsupported: bool = True
) -> Dict[str, Any]:
    """
    Fetch all YouTube data using the unified approach (optimized for minimal API calls).
    
    This function uses the new unified fetcher to get all data in just 2 API calls
    instead of the previous 3 calls.
    
    Args:
        video_id: YouTube video ID
        languages: Preferred languages for transcript
        max_duration_seconds: Maximum allowed duration in seconds
        check_unsupported: Whether to check for unsupported video types
        
    Returns:
        Dictionary containing all YouTube data:
        - video_metadata: Video information
        - available_transcripts: List of available transcript options
        - selected_transcript: Information about the chosen transcript
        - transcript_data: Complete transcript content and metadata
        
    Raises:
        YouTubeTranscriptError: If data cannot be fetched
    """
    fetcher = get_global_fetcher()
    return fetcher.fetch_all_data(video_id, languages, max_duration_seconds, check_unsupported)


def is_youtube_video_supported(video_id: str, max_duration_seconds: int = 1800) -> bool:
    """
    Check if a YouTube video is supported for transcript extraction.
    
    Args:
        video_id: YouTube video ID
        max_duration_seconds: Maximum allowed duration in seconds
        
    Returns:
        True if video is supported, False otherwise
    """
    fetcher = get_global_fetcher()
    return fetcher.is_video_supported(video_id, max_duration_seconds)


def get_youtube_transcript_tier_summary(
    video_id: str,
    preferred_languages: Optional[List[str]] = None,
    video_metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Get a summary of available transcript tiers for a video.
    
    Args:
        video_id: YouTube video ID
        preferred_languages: List of preferred language codes
        video_metadata: Optional video metadata for language detection
        
    Returns:
        Dictionary containing tier summary information
    """
    fetcher = get_global_fetcher()
    return fetcher.three_tier_strategy.get_transcript_tier_summary(
        video_id, preferred_languages, video_metadata
    )


# Performance monitoring and diagnostics
def diagnose_youtube_video(video_id: str) -> Dict[str, Any]:
    """
    Perform comprehensive diagnosis of a YouTube video's accessibility and transcript availability.
    
    Args:
        video_id: YouTube video ID
        
    Returns:
        Comprehensive diagnostic report
    """
    diagnosis = {
        'video_id': video_id,
        'timestamp': None,
        'metadata_accessible': False,
        'transcripts_available': False,
        'supported_for_processing': False,
        'issues': [],
        'recommendations': [],
        'metadata': None,
        'transcript_summary': None,
        'performance_metrics': {}
    }
    
    import time
    start_time = time.time()
    
    try:
        # Test metadata access
        try:
            metadata = get_youtube_video_metadata(video_id)
            diagnosis['metadata_accessible'] = True
            diagnosis['metadata'] = metadata
            
            # Check for issues
            if metadata.get('is_private'):
                diagnosis['issues'].append("Video is private")
            if metadata.get('is_live'):
                diagnosis['issues'].append("Video is a live stream")
            if metadata.get('is_age_restricted'):
                diagnosis['issues'].append("Video is age-restricted")
            
            duration = metadata.get('duration_seconds', 0)
            if duration > 1800:
                diagnosis['issues'].append(f"Video is long ({duration}s > 1800s recommended)")
                
        except Exception as e:
            diagnosis['issues'].append(f"Cannot access metadata: {str(e)}")
        
        # Test transcript availability
        try:
            transcripts = get_available_youtube_transcripts(video_id)
            if transcripts:
                diagnosis['transcripts_available'] = True
                diagnosis['transcript_summary'] = {
                    'total_available': len(transcripts),
                    'languages': [t['language_code'] for t in transcripts],
                    'has_manual': any(not t['is_generated'] for t in transcripts),
                    'has_auto': any(t['is_generated'] for t in transcripts)
                }
            else:
                diagnosis['issues'].append("No transcripts available")
                
        except Exception as e:
            diagnosis['issues'].append(f"Cannot access transcripts: {str(e)}")
        
        # Determine overall support
        diagnosis['supported_for_processing'] = (
            diagnosis['metadata_accessible'] and 
            diagnosis['transcripts_available'] and 
            len([issue for issue in diagnosis['issues'] if 'private' in issue.lower() or 'live' in issue.lower()]) == 0
        )
        
        # Generate recommendations
        if not diagnosis['supported_for_processing']:
            if not diagnosis['metadata_accessible']:
                diagnosis['recommendations'].append("Check if video ID is valid and video is publicly accessible")
            if not diagnosis['transcripts_available']:
                diagnosis['recommendations'].append("Video may not have transcripts enabled or available")
            if any('private' in issue.lower() for issue in diagnosis['issues']):
                diagnosis['recommendations'].append("Video is private - request public access or different video")
            if any('live' in issue.lower() for issue in diagnosis['issues']):
                diagnosis['recommendations'].append("Live videos don't have transcripts - wait for video to end and process")
        else:
            diagnosis['recommendations'].append("Video is fully supported for processing")
        
        # Performance metrics
        diagnosis['performance_metrics'] = {
            'total_diagnosis_time': time.time() - start_time,
            'metadata_check_completed': diagnosis['metadata_accessible'],
            'transcript_check_completed': diagnosis['transcripts_available']
        }
        
    except Exception as e:
        diagnosis['issues'].append(f"Diagnosis failed: {str(e)}")
    
    diagnosis['timestamp'] = time.time()
    return diagnosis


# Error analysis utilities
class YouTubeAPIErrorHandler:
    """Centralized error handling and analysis for YouTube API operations."""
    
    def __init__(self):
        self.error_history = []
        self.error_patterns = {
            'rate_limit': ['rate limit', 'too many requests', '429'],
            'video_not_found': ['not found', '404', 'unavailable'],
            'access_denied': ['access denied', '403', 'private'],
            'network_timeout': ['timeout', 'connection', 'network'],
            'transcript_disabled': ['transcript', 'disabled', 'no transcript']
        }
    
    def analyze_error(self, error: Exception, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Analyze an error and provide categorization and recommendations.
        
        Args:
            error: The exception that occurred
            context: Additional context about the error
            
        Returns:
            Error analysis report
        """
        analysis = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'category': 'unknown',
            'severity': 'medium',
            'is_recoverable': False,
            'recommendations': [],
            'context': context or {}
        }
        
        error_msg_lower = str(error).lower()
        
        # Categorize error
        for category, patterns in self.error_patterns.items():
            if any(pattern in error_msg_lower for pattern in patterns):
                analysis['category'] = category
                break
        
        # Provide specific recommendations based on category
        if analysis['category'] == 'rate_limit':
            analysis['severity'] = 'high'
            analysis['is_recoverable'] = True
            analysis['recommendations'] = [
                "Implement exponential backoff retry logic",
                "Use proxy rotation if available",
                "Reduce request frequency",
                "Consider caching results to avoid repeated requests"
            ]
        elif analysis['category'] == 'video_not_found':
            analysis['severity'] = 'low'
            analysis['is_recoverable'] = False
            analysis['recommendations'] = [
                "Verify video ID is correct",
                "Check if video has been deleted or made private",
                "Try alternative video sources"
            ]
        elif analysis['category'] == 'access_denied':
            analysis['severity'] = 'medium'
            analysis['is_recoverable'] = False
            analysis['recommendations'] = [
                "Video may be private or region-blocked",
                "Check if authentication is required",
                "Try accessing from different region/proxy"
            ]
        elif analysis['category'] == 'network_timeout':
            analysis['severity'] = 'medium'
            analysis['is_recoverable'] = True
            analysis['recommendations'] = [
                "Retry with exponential backoff",
                "Check network connectivity",
                "Increase timeout values",
                "Use proxy if available"
            ]
        elif analysis['category'] == 'transcript_disabled':
            analysis['severity'] = 'low'
            analysis['is_recoverable'] = False
            analysis['recommendations'] = [
                "Video does not have transcripts available",
                "Check if manual transcripts exist",
                "Consider using video without transcript processing"
            ]
        
        # Record error for pattern analysis
        self.error_history.append({
            'timestamp': time.time(),
            'analysis': analysis
        })
        
        return analysis
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get statistics about recent errors."""
        if not self.error_history:
            return {'total_errors': 0, 'categories': {}, 'recent_errors': []}
        
        categories = {}
        for error_record in self.error_history:
            category = error_record['analysis']['category']
            categories[category] = categories.get(category, 0) + 1
        
        return {
            'total_errors': len(self.error_history),
            'categories': categories,
            'recent_errors': self.error_history[-5:],  # Last 5 errors
            'most_common_category': max(categories, key=categories.get) if categories else None
        }


# Global error handler instance
_global_error_handler = YouTubeAPIErrorHandler()


def analyze_youtube_error(error: Exception, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Analyze a YouTube API error and get recommendations.
    
    Args:
        error: The exception that occurred
        context: Additional context about the error
        
    Returns:
        Error analysis report
    """
    return _global_error_handler.analyze_error(error, context)


def get_youtube_error_statistics() -> Dict[str, Any]:
    """Get statistics about recent YouTube API errors."""
    return _global_error_handler.get_error_statistics()