"""
YouTube transcript acquisition and processing functionality.

This module handles transcript fetching, tier strategies, and transcript
processing using the YouTube Transcript API.
"""

import logging
from typing import Optional, Dict, List, Any, Union
from datetime import datetime
import time
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

from .youtube_core import (
    YouTubeTranscriptError, NoTranscriptAvailableError, RateLimitError,
    NetworkTimeoutError, VideoNotFoundError
)
from .validators import YouTubeURLValidator
from .proxy_manager import get_proxy_manager, get_retry_manager

try:
    from .language_detector import YouTubeLanguageDetector
except ImportError:
    # If we still get ImportError, there's a configuration issue
    raise ImportError("YouTubeLanguageDetector not found. Please check your environment setup and ensure all dependencies are installed.")

logger = logging.getLogger(__name__)


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
                lang_score = 0  # Not in preferred languages
            
            # Tier priority score (higher is better)
            tier_score = transcript_info.quality_score
            
            # Combine scores: tier is primary, language is secondary
            return (tier_score, lang_score)
        
        # Sort by combined score (descending order - highest score first)
        return sorted(transcript_infos, key=sort_key, reverse=True)
    
    def get_best_transcript_option(self, video_id: str, preferred_languages: Optional[List[str]] = None,
                                 video_metadata: Optional[Dict[str, Any]] = None) -> Optional[TranscriptInfo]:
        """
        Get the best transcript option based on three-tier strategy.
        
        Args:
            video_id: YouTube video ID
            preferred_languages: List of preferred language codes
            video_metadata: Optional video metadata for language detection
            
        Returns:
            Best TranscriptInfo object or None if no transcripts available
        """
        try:
            strategy_order = self.get_transcript_strategy(video_id, preferred_languages, video_metadata)
            return strategy_order[0] if strategy_order else None
        except Exception as e:
            self.logger.error(f"Failed to get best transcript option for {video_id}: {str(e)}")
            return None
    
    def get_transcript_tier_summary(self, video_id: str, preferred_languages: Optional[List[str]] = None,
                                   video_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Get a summary of available transcript tiers for a video.
        
        Args:
            video_id: YouTube video ID
            preferred_languages: List of preferred language codes
            video_metadata: Optional video metadata for language detection
            
        Returns:
            Dictionary containing tier summary information
        """
        try:
            strategy_order = self.get_transcript_strategy(video_id, preferred_languages, video_metadata)
            
            # Group by tier
            tiers = {
                TranscriptTier.MANUAL: [],
                TranscriptTier.AUTO_GENERATED: [],
                TranscriptTier.TRANSLATED: []
            }
            
            for info in strategy_order:
                tiers[info.tier].append({
                    'language_code': info.language_code,
                    'quality_score': info.quality_score
                })
            
            return {
                'video_id': video_id,
                'total_transcripts': len(strategy_order),
                'best_option': {
                    'language_code': strategy_order[0].language_code,
                    'tier': strategy_order[0].tier,
                    'quality_score': strategy_order[0].quality_score
                } if strategy_order else None,
                'tiers': {
                    'manual': {
                        'count': len(tiers[TranscriptTier.MANUAL]),
                        'languages': [t['language_code'] for t in tiers[TranscriptTier.MANUAL]]
                    },
                    'auto_generated': {
                        'count': len(tiers[TranscriptTier.AUTO_GENERATED]),
                        'languages': [t['language_code'] for t in tiers[TranscriptTier.AUTO_GENERATED]]
                    },
                    'translated': {
                        'count': len(tiers[TranscriptTier.TRANSLATED]),
                        'languages': [t['language_code'] for t in tiers[TranscriptTier.TRANSLATED]]
                    }
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get transcript tier summary for {video_id}: {str(e)}")
            return {
                'video_id': video_id,
                'error': str(e),
                'total_transcripts': 0,
                'best_option': None,
                'tiers': {}
            }


class ProxyAwareTranscriptApi:
    """
    YouTube Transcript API wrapper with proxy support and error handling.
    """
    
    def __init__(self):
        self.proxy_manager = get_proxy_manager()
        self.retry_manager = get_retry_manager()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def list_transcripts(self, video_id: str):
        """List available transcripts for a video with proxy support."""
        try:
            with self.retry_manager.retry_context("transcript_list"):
                with self.proxy_manager.request_context() as (session, proxy):
                    if proxy:
                        self.logger.debug(f"Listing transcripts for {video_id} via proxy: {proxy.url}")
                        # Configure proxy for YouTubeTranscriptApi if possible
                        # Note: The youtube-transcript-api doesn't directly support proxy configuration
                        # This is a placeholder for potential future proxy integration
                    
                    return YouTubeTranscriptApi.list_transcripts(video_id)
                    
        except Exception as e:
            self.logger.error(f"Failed to list transcripts for {video_id}: {str(e)}")
            raise
    
    def get_transcript(self, video_id: str, languages: Optional[List[str]] = None):
        """Get transcript content with proxy support and retry logic."""
        try:
            with self.retry_manager.retry_context("transcript_fetch"):
                with self.proxy_manager.request_context() as (session, proxy):
                    if proxy:
                        self.logger.debug(f"Fetching transcript for {video_id} via proxy: {proxy.url}")
                    
                    # Attempt to get transcript
                    if languages:
                        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=languages)
                    else:
                        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
                    
                    return transcript_list
                    
        except TooManyRequests as e:
            self.logger.error(f"Rate limit exceeded for {video_id}")
            raise RateLimitError(video_id)
        except TranscriptsDisabled:
            self.logger.error(f"Transcripts disabled for {video_id}")
            raise NoTranscriptAvailableError(video_id)
        except NoTranscriptFound:
            self.logger.error(f"No transcript found for {video_id}")
            raise NoTranscriptAvailableError(video_id)
        except VideoUnavailable:
            self.logger.error(f"Video {video_id} unavailable")
            raise VideoNotFoundError(video_id)
        except Exception as e:
            self.logger.error(f"Unexpected error fetching transcript for {video_id}: {str(e)}")
            raise YouTubeTranscriptError(f"Failed to fetch transcript: {str(e)}", video_id)


# Error handling decorator
def handle_youtube_api_exceptions(func):
    """Decorator to handle YouTube API exceptions consistently."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except TooManyRequests as e:
            video_id = kwargs.get('video_id', args[1] if len(args) > 1 else 'unknown')
            raise RateLimitError(video_id)
        except TranscriptsDisabled as e:
            video_id = kwargs.get('video_id', args[1] if len(args) > 1 else 'unknown')
            raise NoTranscriptAvailableError(video_id)
        except NoTranscriptFound as e:
            video_id = kwargs.get('video_id', args[1] if len(args) > 1 else 'unknown')
            raise NoTranscriptAvailableError(video_id)
        except VideoUnavailable as e:
            video_id = kwargs.get('video_id', args[1] if len(args) > 1 else 'unknown')
            raise VideoNotFoundError(video_id)
        except Exception as e:
            video_id = kwargs.get('video_id', args[1] if len(args) > 1 else 'unknown')
            logger.error(f"Unexpected error in {func.__name__} for {video_id}: {str(e)}")
            raise YouTubeTranscriptError(f"Failed in {func.__name__}: {str(e)}", video_id)
    
    return wrapper


# Utility functions for transcript processing
def detect_transcript_language(transcript_list: List[Dict[str, Any]]) -> str:
    """
    Detect the language of a transcript from its content.
    
    Args:
        transcript_list: List of transcript segments
        
    Returns:
        Detected language code
    """
    if not transcript_list:
        return 'unknown'
    
    # Get text sample from first few segments
    def get_text(entry):
        if isinstance(entry, dict):
            return entry.get('text', '')
        return str(entry)
    
    text_sample = ' '.join([get_text(entry) for entry in transcript_list[:10]])
    
    # Simple language detection based on character patterns
    if any('\u4e00' <= char <= '\u9fff' for char in text_sample):
        return 'zh'  # Chinese characters detected
    elif any('\u3040' <= char <= '\u309f' or '\u30a0' <= char <= '\u30ff' for char in text_sample):
        return 'ja'  # Japanese characters detected
    elif any('\uac00' <= char <= '\ud7af' for char in text_sample):
        return 'ko'  # Korean characters detected
    else:
        return 'en'  # Default to English


def calculate_transcript_duration(transcript_list: List[Dict[str, Any]]) -> float:
    """
    Calculate total duration of transcript in seconds.
    
    Args:
        transcript_list: List of transcript segments
        
    Returns:
        Duration in seconds
    """
    if not transcript_list:
        return 0.0
    
    try:
        # Get the last segment's start time and duration
        last_segment = transcript_list[-1]
        if isinstance(last_segment, dict):
            start = last_segment.get('start', 0)
            duration = last_segment.get('duration', 0)
            return start + duration
        
        # Fallback: estimate based on number of segments
        return len(transcript_list) * 3.0  # Assume 3 seconds per segment
        
    except Exception:
        # Fallback calculation
        return len(transcript_list) * 3.0