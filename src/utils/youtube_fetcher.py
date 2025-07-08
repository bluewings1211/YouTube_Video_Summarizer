"""
Unified YouTube data fetcher to minimize API calls.

This module provides a unified approach to fetching all YouTube data
in a single operation, reducing the number of API calls from 3 to 2.
"""

import logging
from typing import Optional, Dict, List, Any
from datetime import datetime

try:
    from youtube_transcript_api.formatters import TextFormatter
    from youtube_transcript_api._errors import (
        TranscriptsDisabled, 
        NoTranscriptFound, 
        VideoUnavailable,
        TooManyRequests
    )
except ImportError:
    # Fallback for development
    class TextFormatter:
        def format_transcript(self, transcript): return ""
    TranscriptsDisabled = Exception
    NoTranscriptFound = Exception
    VideoUnavailable = Exception
    TooManyRequests = Exception

from .youtube_core import (
    YouTubeTranscriptError, NoTranscriptAvailableError, RateLimitError,
    VideoNotFoundError, PrivateVideoError, LiveVideoError, VideoTooLongError,
    TranscriptAcquisitionLogger
)
from .youtube_metadata import YouTubeVideoMetadataExtractor
from .youtube_transcripts import (
    ThreeTierTranscriptStrategy, ProxyAwareTranscriptApi, TranscriptInfo,
    handle_youtube_api_exceptions, detect_transcript_language, calculate_transcript_duration
)
from .validators import YouTubeURLValidator

logger = logging.getLogger(__name__)


class YouTubeUnifiedFetcher:
    """
    Unified YouTube data fetcher that minimizes API calls.
    
    This class fetches all YouTube data (metadata, transcripts, content) 
    in an optimized manner, reducing API calls from 3 to 2:
    1. Video metadata (via web scraping) 
    2. Transcript data (via YouTube Transcript API - includes listing + content)
    """
    
    def __init__(self, enable_detailed_logging: bool = True, log_file: str = None):
        self.formatter = TextFormatter()
        self.api = ProxyAwareTranscriptApi()
        self.metadata_extractor = YouTubeVideoMetadataExtractor()
        self.three_tier_strategy = ThreeTierTranscriptStrategy()
        
        # Initialize comprehensive logging
        if enable_detailed_logging:
            from .youtube_core import setup_transcript_logging
            self.acquisition_logger = TranscriptAcquisitionLogger(
                setup_transcript_logging(log_file=log_file)
            )
        else:
            self.acquisition_logger = None
    
    @handle_youtube_api_exceptions
    def fetch_all_data(
        self, 
        video_id: str, 
        languages: Optional[List[str]] = None,
        max_duration_seconds: int = 1800,
        check_unsupported: bool = True
    ) -> Dict[str, Any]:
        """
        Fetch all YouTube data in a unified operation with minimal API calls.
        
        This method optimizes API usage by:
        1. Fetching video metadata once (web scraping)
        2. Using a single transcript API call to get both available transcripts 
           and selected transcript content
        
        Args:
            video_id: YouTube video ID
            languages: Preferred languages for transcript (default: ['en'])
            max_duration_seconds: Maximum allowed duration in seconds (default: 1800 = 30 minutes)
            check_unsupported: Whether to check for unsupported video types (default: True)
            
        Returns:
            Dictionary containing all YouTube data:
            - video_metadata: Video information
            - available_transcripts: List of available transcript options
            - selected_transcript: Information about the chosen transcript
            - transcript_data: Complete transcript content and metadata
            
        Raises:
            YouTubeTranscriptError: If data cannot be fetched
            UnsupportedVideoTypeError: If video type is not supported
            VideoTooLongError: If video exceeds duration limit
        """
        if not video_id:
            raise YouTubeTranscriptError("Video ID is required")
            
        if not YouTubeURLValidator._is_valid_video_id(video_id):
            raise YouTubeTranscriptError(f"Invalid video ID format: {video_id}")
            
        if languages is None:
            languages = ['en', 'zh-TW', 'zh-CN', 'zh', 'ja', 'ko', 'fr', 'de', 'es', 'it', 'pt', 'ru']
        
        # Log acquisition start
        if self.acquisition_logger:
            self.acquisition_logger.log_acquisition_start(video_id, "unified", languages)
        
        start_time = datetime.utcnow()
        
        try:
            # STEP 1: Get video metadata (1 API call via web scraping)
            logger.debug(f"Fetching video metadata for {video_id}")
            video_metadata = self.metadata_extractor.extract_video_metadata(video_id)
            
            if self.acquisition_logger:
                self.acquisition_logger.log_video_metadata(video_id, video_metadata)
            
            # Check for unsupported video types if requested
            if check_unsupported:
                self._validate_video_support(video_metadata, video_id, max_duration_seconds)
            
            # STEP 2: Get transcript data using unified strategy (1 API call)
            logger.debug(f"Fetching transcript data for {video_id}")
            transcript_result = self._fetch_transcript_unified(video_id, languages, video_metadata)
            
            # Combine all data
            unified_result = {
                'video_metadata': video_metadata,
                'available_transcripts': transcript_result['available_transcripts'],
                'selected_transcript': transcript_result['selected_transcript'],
                'transcript_data': transcript_result['transcript_data'],
                'fetch_timestamp': datetime.utcnow().isoformat(),
                'success': True,
                'api_calls_made': 2,  # metadata + transcript
                'optimization_applied': True
            }
            
            # Log successful completion
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            if self.acquisition_logger:
                final_language = transcript_result['transcript_data'].get('language', 'unknown')
                self.acquisition_logger.log_acquisition_complete(
                    video_id, True, final_language, processing_time, 1
                )
            
            logger.info(f"Successfully fetched all data for {video_id} in {processing_time:.2f}s")
            return unified_result
            
        except (PrivateVideoError, LiveVideoError, VideoTooLongError):
            raise  # Re-raise these specific errors
        except Exception as e:
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            if self.acquisition_logger:
                self.acquisition_logger.log_acquisition_complete(
                    video_id, False, None, processing_time, 1
                )
            logger.error(f"Failed to fetch data for {video_id}: {str(e)}")
            raise
    
    def _validate_video_support(self, video_metadata: Dict[str, Any], video_id: str, max_duration_seconds: int):
        """Validate that the video is supported for transcript extraction."""
        # Check if video is private
        if video_metadata.get('is_private', False):
            raise PrivateVideoError(video_id)
        
        # Check if video is live
        if video_metadata.get('is_live', False):
            raise LiveVideoError(video_id)
        
        # Check video duration
        duration = video_metadata.get('duration_seconds')
        if duration and duration > max_duration_seconds:
            raise VideoTooLongError(video_id, duration, max_duration_seconds)
    
    def _fetch_transcript_unified(self, video_id: str, languages: List[str], 
                                video_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fetch transcript data using unified approach.
        
        This method uses the three-tier strategy to get both available transcripts
        and the best transcript content in a single optimized operation.
        """
        try:
            # Get transcript strategy (includes available transcripts discovery)
            strategy_order = self.three_tier_strategy.get_transcript_strategy(
                video_id, languages, video_metadata
            )
            
            if not strategy_order:
                raise NoTranscriptAvailableError(video_id, [])
            
            # Log strategy planning
            if self.acquisition_logger:
                self.acquisition_logger.log_strategy_planning(
                    video_id, strategy_order, len(strategy_order)
                )
            
            # Convert strategy order to available transcripts list
            available_transcripts = []
            for info in strategy_order:
                available_transcripts.append({
                    'language': getattr(info.original_transcript, 'language', info.language_code),
                    'language_code': info.language_code,
                    'is_generated': info.is_generated,
                    'is_translatable': info.is_translatable
                })
            
            # Try to fetch transcript content using the strategy order
            selected_transcript = None
            transcript_data = None
            
            for attempt, transcript_info in enumerate(strategy_order, 1):
                if self.acquisition_logger:
                    self.acquisition_logger.log_tier_attempt_start(
                        video_id, transcript_info.tier, transcript_info.language_code, attempt
                    )
                
                try:
                    # Fetch transcript content
                    transcript_list = self.api.get_transcript(
                        video_id, [transcript_info.language_code]
                    )
                    
                    # Format transcript
                    formatted_transcript = self.formatter.format_transcript(transcript_list)
                    
                    # Fallback formatting if formatter fails
                    if not formatted_transcript or not formatted_transcript.strip():
                        # Manual formatting from segments
                        if transcript_list:
                            formatted_transcript = ' '.join([
                                segment.get('text', '') if isinstance(segment, dict) 
                                else str(segment) 
                                for segment in transcript_list
                            ]).strip()
                    
                    # Calculate metrics
                    duration = calculate_transcript_duration(transcript_list)
                    word_count = len(formatted_transcript.split()) if formatted_transcript else 0
                    detected_language = detect_transcript_language(transcript_list)
                    
                    # Success - build transcript data
                    transcript_data = {
                        'video_id': video_id,
                        'transcript': formatted_transcript,
                        'raw_transcript': transcript_list,
                        'language': detected_language,
                        'duration_seconds': duration,
                        'word_count': word_count,
                        'fetch_timestamp': datetime.utcnow().isoformat(),
                        'success': True,
                        'video_metadata': video_metadata  # Include metadata for compatibility
                    }
                    
                    selected_transcript = {
                        'language_code': transcript_info.language_code,
                        'tier': transcript_info.tier,
                        'quality_score': transcript_info.quality_score,
                        'is_generated': transcript_info.is_generated,
                        'is_translatable': transcript_info.is_translatable
                    }
                    
                    # Log success
                    if self.acquisition_logger:
                        self.acquisition_logger.log_tier_attempt_success(
                            video_id, transcript_info.tier, transcript_info.language_code,
                            word_count, duration
                        )
                    
                    logger.info(f"Successfully fetched {transcript_info.tier} transcript for {video_id} " +
                               f"({transcript_info.language_code}, {word_count} words)")
                    break
                    
                except Exception as e:
                    # Log failure and try next option
                    if self.acquisition_logger:
                        self.acquisition_logger.log_tier_attempt_failure(
                            video_id, transcript_info.tier, transcript_info.language_code,
                            str(e), type(e).__name__
                        )
                    
                    logger.warning(f"Failed to fetch {transcript_info.tier} transcript " +
                                 f"for {video_id} ({transcript_info.language_code}): {str(e)}")
                    
                    # If this was not the last attempt, log fallback
                    if attempt < len(strategy_order):
                        next_info = strategy_order[attempt]
                        if self.acquisition_logger:
                            self.acquisition_logger.log_tier_fallback(
                                video_id, transcript_info.tier, next_info.tier, str(e)
                            )
                    
                    continue
            
            # Check if we got transcript data
            if not transcript_data:
                available_langs = [t['language_code'] for t in available_transcripts]
                raise NoTranscriptAvailableError(video_id, available_langs)
            
            return {
                'available_transcripts': available_transcripts,
                'selected_transcript': selected_transcript,
                'transcript_data': transcript_data
            }
            
        except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable):
            raise NoTranscriptAvailableError(video_id, [])
        except TooManyRequests:
            raise RateLimitError(video_id)
        except Exception as e:
            logger.error(f"Unexpected error fetching transcript for {video_id}: {str(e)}")
            raise YouTubeTranscriptError(f"Failed to fetch transcript: {str(e)}", video_id)
    
    # Compatibility methods for existing code
    def get_video_metadata(self, video_id: str) -> Dict[str, Any]:
        """
        Get video metadata only (for compatibility with existing code).
        
        Args:
            video_id: YouTube video ID
            
        Returns:
            Dictionary containing video metadata
        """
        return self.metadata_extractor.extract_video_metadata(video_id)
    
    def get_available_transcripts(self, video_id: str) -> List[Dict[str, Any]]:
        """
        Get list of available transcripts for a video (for compatibility).
        
        Args:
            video_id: YouTube video ID
            
        Returns:
            List of available transcript metadata
        """
        try:
            strategy_order = self.three_tier_strategy.get_transcript_strategy(video_id)
            
            available_transcripts = []
            for info in strategy_order:
                available_transcripts.append({
                    'language': getattr(info.original_transcript, 'language', info.language_code),
                    'language_code': info.language_code,
                    'is_generated': info.is_generated,
                    'is_translatable': info.is_translatable
                })
            
            return available_transcripts
            
        except Exception as e:
            logger.error(f"Error getting available transcripts for {video_id}: {str(e)}")
            raise YouTubeTranscriptError(f"Failed to get available transcripts: {str(e)}", video_id)
    
    def fetch_transcript(
        self, 
        video_id: str, 
        languages: Optional[List[str]] = None,
        include_metadata: bool = True,
        check_unsupported: bool = True,
        max_duration_seconds: int = 1800
    ) -> Dict[str, Any]:
        """
        Fetch transcript for a YouTube video (for compatibility with existing code).
        
        This method maintains compatibility with the old interface while using
        the optimized unified fetcher internally.
        
        Args:
            video_id: YouTube video ID
            languages: Preferred languages for transcript (default: ['en'])
            include_metadata: Whether to include video metadata (default: True)
            check_unsupported: Whether to check for unsupported video types (default: True)
            max_duration_seconds: Maximum allowed duration in seconds (default: 1800 = 30 minutes)
            
        Returns:
            Dictionary containing transcript data and metadata (compatible format)
        """
        try:
            # Use unified fetcher but return in compatible format
            unified_data = self.fetch_all_data(
                video_id, languages, max_duration_seconds, check_unsupported
            )
            
            # Extract transcript data for compatibility
            transcript_data = unified_data['transcript_data'].copy()
            
            # Add metadata if requested (it's already included in unified approach)
            if include_metadata:
                transcript_data['video_metadata'] = unified_data['video_metadata']
            
            return transcript_data
            
        except Exception as e:
            logger.error(f"Error in compatibility fetch_transcript for {video_id}: {str(e)}")
            raise
    
    def is_video_supported(self, video_id: str, max_duration_seconds: int = 1800) -> bool:
        """
        Check if a video is supported for transcript extraction.
        
        Args:
            video_id: YouTube video ID
            max_duration_seconds: Maximum allowed duration in seconds
            
        Returns:
            True if video is supported, False otherwise
        """
        try:
            support_result = self.metadata_extractor.detect_unsupported_video_type(
                video_id, max_duration_seconds
            )
            return support_result['is_supported']
        except Exception:
            return False


# Legacy compatibility - create instance for backward compatibility
class YouTubeTranscriptFetcher(YouTubeUnifiedFetcher):
    """
    Legacy compatibility class that extends YouTubeUnifiedFetcher.
    
    This maintains backward compatibility with existing code while
    providing the optimized unified fetching capabilities.
    """
    pass