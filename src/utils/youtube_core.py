"""
Core YouTube API functionality and base classes.

This module contains the fundamental classes, exceptions, and utilities
used throughout the YouTube API integration.
"""

import logging
from typing import Optional, Dict, List, Any
from datetime import datetime


# Configure logging
logger = logging.getLogger(__name__)


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
        """Log the start of transcript acquisition for a video."""
        self.session_stats['total_attempts'] += 1
        self.session_stats['videos_processed'].add(video_id)
        
        self.logger.info(f"🎬 Starting transcript acquisition for video: {video_id}")
        self.logger.info(f"Strategy: {strategy}, Preferred languages: {preferred_languages or 'auto-detect'}")
    
    def log_video_metadata(self, video_id: str, metadata: Dict[str, Any]) -> None:
        """Log video metadata information."""
        if metadata:
            duration = metadata.get('duration_seconds', 'unknown')
            title = metadata.get('title', 'unknown')[:50] + '...' if len(metadata.get('title', '')) > 50 else metadata.get('title', 'unknown')
            language = metadata.get('language', 'unknown')
            
            self.logger.info(f"📹 Video metadata - Title: {title}")
            self.logger.info(f"Duration: {duration}s, Language: {language}")
        else:
            self.logger.warning(f"⚠️  No metadata available for video: {video_id}")
    
    def log_strategy_planning(self, video_id: str, strategy_order: List[Any], 
                            total_options: int, detection_result: Any = None) -> None:
        """Log transcript strategy planning information."""
        self.logger.info(f"🎯 Planning transcript strategy for {video_id}")
        self.logger.info(f"Total transcript options available: {total_options}")
        
        if detection_result:
            detected_lang = getattr(detection_result, 'detected_language', 'unknown')
            confidence = getattr(detection_result, 'confidence_score', 0)
            self.logger.info(f"Language detection: {detected_lang} (confidence: {confidence:.2f})")
        
        for i, tier in enumerate(strategy_order, 1):
            tier_name = getattr(tier, 'tier_name', 'unknown')
            language = getattr(tier, 'language_code', 'unknown')
            self.logger.debug(f"  Tier {i}: {tier_name} ({language})")
    
    def log_tier_attempt_start(self, video_id: str, tier: str, language: str, 
                              attempt_number: int = 1) -> None:
        """Log the start of a tier attempt."""
        if tier not in self.session_stats['tiers_used']:
            self.session_stats['tiers_used'][tier] = 0
        self.session_stats['tiers_used'][tier] += 1
        
        self.logger.info(f"🎭 Tier {attempt_number}: Attempting {tier} transcript ({language}) for {video_id}")
    
    def log_tier_attempt_success(self, video_id: str, tier: str, language: str, 
                               word_count: int, duration: float) -> None:
        """Log successful tier attempt."""
        self.session_stats['successful_attempts'] += 1
        
        self.logger.info(f"✅ Success! {tier} transcript acquired for {video_id}")
        self.logger.info(f"Language: {language}, Words: {word_count}, Duration: {duration:.1f}s")
    
    def log_tier_attempt_failure(self, video_id: str, tier: str, language: str, 
                               error: str, error_type: str = "unknown") -> None:
        """Log failed tier attempt."""
        self.session_stats['failed_attempts'] += 1
        
        if error_type not in self.session_stats['error_counts']:
            self.session_stats['error_counts'][error_type] = 0
        self.session_stats['error_counts'][error_type] += 1
        
        self.logger.warning(f"❌ {tier} transcript failed for {video_id}")
        self.logger.warning(f"Error type: {error_type}, Details: {error}")
    
    def log_tier_fallback(self, video_id: str, from_tier: str, to_tier: str, 
                         reason: str) -> None:
        """Log tier fallback."""
        self.logger.info(f"🔄 Falling back from {from_tier} to {to_tier} for {video_id}")
        self.logger.info(f"Reason: {reason}")
    
    def log_unsupported_video(self, video_id: str, reason: str, issues: List[Dict]) -> None:
        """Log unsupported video detection."""
        self.logger.warning(f"🚫 Video {video_id} is unsupported: {reason}")
        
        for issue in issues:
            issue_type = issue.get('type', 'unknown')
            message = issue.get('message', 'No details')
            self.logger.warning(f"  - {issue_type}: {message}")
    
    def log_acquisition_complete(self, video_id: str, success: bool, 
                               final_language: str = None, processing_time: float = 0,
                               total_attempts: int = 0) -> None:
        """Log completion of transcript acquisition."""
        if success:
            self.logger.info(f"🎉 Transcript acquisition completed for {video_id}")
            self.logger.info(f"Final language: {final_language}, Time: {processing_time:.2f}s, Attempts: {total_attempts}")
        else:
            self.logger.error(f"💥 Transcript acquisition failed for {video_id}")
            self.logger.error(f"Processing time: {processing_time:.2f}s, Total attempts: {total_attempts}")
    
    def log_performance_metrics(self, video_id: str, metrics: Dict[str, Any]) -> None:
        """Log performance metrics."""
        self.logger.debug(f"📊 Performance metrics for {video_id}:")
        for metric, value in metrics.items():
            self.logger.debug(f"  {metric}: {value}")
    
    def log_error_analysis(self, video_id: str, error_report: Dict[str, Any]) -> None:
        """Log detailed error analysis."""
        self.logger.error(f"🔍 Error analysis for {video_id}:")
        for category, details in error_report.items():
            self.logger.error(f"  {category}: {details}")
    
    def log_session_summary(self) -> Dict[str, Any]:
        """Log and return session summary statistics."""
        summary = {
            'session_duration': (datetime.utcnow().isoformat()),
            'total_videos': len(self.session_stats['videos_processed']),
            'total_attempts': self.session_stats['total_attempts'],
            'successful_attempts': self.session_stats['successful_attempts'],
            'failed_attempts': self.session_stats['failed_attempts'],
            'success_rate': (self.session_stats['successful_attempts'] / 
                           max(1, self.session_stats['total_attempts'])) * 100,
            'tiers_usage': self.session_stats['tiers_used'],
            'error_breakdown': self.session_stats['error_counts']
        }
        
        self.logger.info("📈 Session Summary:")
        self.logger.info(f"  Videos processed: {summary['total_videos']}")
        self.logger.info(f"  Success rate: {summary['success_rate']:.1f}%")
        self.logger.info(f"  Most used tier: {max(summary['tiers_usage'], key=summary['tiers_usage'].get) if summary['tiers_usage'] else 'none'}")
        
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
        self.logger.info("📊 Session statistics reset")


# Exception classes
class YouTubeTranscriptError(Exception):
    """Base exception for YouTube transcript operations."""
    
    def __init__(self, message: str, video_id: str = "", error_code: Optional[str] = None):
        super().__init__(message)
        self.video_id = video_id
        self.error_code = error_code
        self.timestamp = datetime.utcnow().isoformat()


class UnsupportedVideoTypeError(YouTubeTranscriptError):
    """Raised when video type is not supported for transcript extraction."""
    
    def __init__(self, video_id: str = ""):
        super().__init__(f"Video type not supported for transcript extraction", video_id)


class PrivateVideoError(UnsupportedVideoTypeError):
    """Raised when video is private."""
    
    def __init__(self, video_id: str = ""):
        super().__init__(video_id)
        self.args = (f"Video {video_id} is private and cannot be accessed",)


class LiveVideoError(UnsupportedVideoTypeError):
    """Raised when video is a live stream."""
    
    def __init__(self, video_id: str = ""):
        super().__init__(video_id)
        self.args = (f"Live video {video_id} does not have transcripts available",)


class NoTranscriptAvailableError(UnsupportedVideoTypeError):
    """Raised when no transcript is available for the video."""
    
    def __init__(self, video_id: str = "", available_languages: List[str] = None):
        super().__init__(video_id)
        available = ", ".join(available_languages) if available_languages else "none"
        self.args = (f"No transcript available for video {video_id}. Available languages: {available}",)


class VideoTooLongError(UnsupportedVideoTypeError):
    """Raised when video is too long for processing."""
    
    def __init__(self, video_id: str = "", duration: int = 0, max_duration: int = 1800):
        super().__init__(video_id)
        self.duration = duration
        self.max_duration = max_duration
        self.args = (f"Video {video_id} is too long ({duration}s > {max_duration}s maximum)",)


class VideoNotFoundError(YouTubeTranscriptError):
    """Raised when video is not found."""
    
    def __init__(self, video_id: str = ""):
        super().__init__(f"Video {video_id} not found or unavailable", video_id)


class NetworkTimeoutError(YouTubeTranscriptError):
    """Raised when network request times out."""
    
    def __init__(self, video_id: str = "", timeout: int = 30):
        super().__init__(f"Network timeout ({timeout}s) for video {video_id}", video_id)
        self.timeout = timeout


class RateLimitError(YouTubeTranscriptError):
    """Raised when rate limit is exceeded."""
    
    def __init__(self, video_id: str = ""):
        super().__init__(f"Rate limit exceeded for video {video_id}", video_id)
        self.error_code = "RATE_LIMIT_EXCEEDED"


class AgeRestrictedVideoError(UnsupportedVideoTypeError):
    """Raised when video is age-restricted."""
    
    def __init__(self, video_id: str = ""):
        super().__init__(video_id)
        self.args = (f"Video {video_id} is age-restricted and cannot be accessed",)


class RegionBlockedVideoError(UnsupportedVideoTypeError):
    """Raised when video is blocked in the current region."""
    
    def __init__(self, video_id: str = ""):
        super().__init__(video_id)
        self.args = (f"Video {video_id} is blocked in the current region",)


class TranscriptProcessingError(YouTubeTranscriptError):
    """Raised when transcript processing fails."""
    
    def __init__(self, video_id: str = "", processing_stage: str = "unknown"):
        super().__init__(f"Transcript processing failed at stage: {processing_stage}", video_id)


class TranscriptValidationError(YouTubeTranscriptError):
    """Raised when transcript validation fails."""
    
    def __init__(self, video_id: str = "", validation_issue: str = "unknown"):
        super().__init__(f"Transcript validation failed: {validation_issue}", video_id)


class TierStrategyError(YouTubeTranscriptError):
    """Raised when tier strategy fails."""
    
    def __init__(self, video_id: str = "", strategy_issue: str = "unknown"):
        super().__init__(f"Tier strategy failed: {strategy_issue}", video_id)