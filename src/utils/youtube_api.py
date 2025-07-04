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

try:
    from youtube_transcript_api import YouTubeTranscriptApi
    from youtube_transcript_api.formatters import TextFormatter
    from youtube_transcript_api._errors import (
        TranscriptsDisabled, 
        NoTranscriptFound, 
        VideoUnavailable,
        TooManyRequests
    )
except ImportError:
    raise ImportError(
        "youtube-transcript-api is required. Install with: pip install youtube-transcript-api"
    )

from .validators import YouTubeURLValidator

# Configure logging
logger = logging.getLogger(__name__)


class YouTubeTranscriptError(Exception):
    """Base exception for YouTube transcript operations."""
    pass


class YouTubeTranscriptFetcher:
    """Handles YouTube transcript fetching and processing."""
    
    def __init__(self):
        self.formatter = TextFormatter()
        self.api = YouTubeTranscriptApi
    
    def fetch_transcript(
        self, 
        video_id: str, 
        languages: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Fetch transcript for a YouTube video.
        
        Args:
            video_id: YouTube video ID
            languages: Preferred languages for transcript (default: ['en'])
            
        Returns:
            Dictionary containing transcript data and metadata
            
        Raises:
            YouTubeTranscriptError: If transcript cannot be fetched
        """
        if not video_id:
            raise YouTubeTranscriptError("Video ID is required")
            
        if not YouTubeURLValidator._is_valid_video_id(video_id):
            raise YouTubeTranscriptError(f"Invalid video ID format: {video_id}")
            
        if languages is None:
            languages = ['en']
            
        try:
            # Fetch transcript
            transcript_list = self.api.get_transcript(video_id, languages=languages)
            
            # Format transcript
            formatted_transcript = self.formatter.format_transcript(transcript_list)
            
            # Extract additional metadata from transcript
            duration = self._calculate_duration(transcript_list)
            word_count = len(formatted_transcript.split())
            
            return {
                'video_id': video_id,
                'transcript': formatted_transcript,
                'raw_transcript': transcript_list,
                'language': self._detect_language(transcript_list),
                'duration_seconds': duration,
                'word_count': word_count,
                'fetch_timestamp': datetime.utcnow().isoformat(),
                'success': True
            }
            
        except TranscriptsDisabled:
            logger.error(f"Transcripts disabled for video {video_id}")
            raise YouTubeTranscriptError("Transcripts are disabled for this video")
            
        except NoTranscriptFound:
            logger.error(f"No transcript found for video {video_id}")
            raise YouTubeTranscriptError(
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
        languages: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Fetch transcript from a YouTube URL.
        
        Args:
            url: YouTube URL
            languages: Preferred languages for transcript
            
        Returns:
            Dictionary containing transcript data and metadata
            
        Raises:
            YouTubeTranscriptError: If URL is invalid or transcript cannot be fetched
        """
        # Validate URL and extract video ID
        is_valid, video_id = YouTubeURLValidator.validate_and_extract(url)
        
        if not is_valid or not video_id:
            raise YouTubeTranscriptError(f"Invalid YouTube URL: {url}")
            
        return self.fetch_transcript(video_id, languages)
    
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


# Convenience functions
def fetch_youtube_transcript(
    video_id: str, 
    languages: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Convenience function to fetch YouTube transcript.
    
    Args:
        video_id: YouTube video ID
        languages: Preferred languages for transcript
        
    Returns:
        Dictionary containing transcript data and metadata
    """
    fetcher = YouTubeTranscriptFetcher()
    return fetcher.fetch_transcript(video_id, languages)


def fetch_youtube_transcript_from_url(
    url: str, 
    languages: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Convenience function to fetch YouTube transcript from URL.
    
    Args:
        url: YouTube URL
        languages: Preferred languages for transcript
        
    Returns:
        Dictionary containing transcript data and metadata
    """
    fetcher = YouTubeTranscriptFetcher()
    return fetcher.fetch_transcript_from_url(url, languages)


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