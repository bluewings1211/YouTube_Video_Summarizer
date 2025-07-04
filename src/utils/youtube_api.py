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


class UnsupportedVideoTypeError(YouTubeTranscriptError):
    """Raised when video type is not supported for transcript extraction."""
    pass


class PrivateVideoError(UnsupportedVideoTypeError):
    """Raised when video is private."""
    pass


class LiveVideoError(UnsupportedVideoTypeError):
    """Raised when video is live content."""
    pass


class NoTranscriptAvailableError(UnsupportedVideoTypeError):
    """Raised when no transcripts are available for the video."""
    pass


class VideoTooLongError(UnsupportedVideoTypeError):
    """Raised when video duration exceeds the maximum allowed limit."""
    pass


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
            # Fetch video page to extract metadata
            url = f"https://www.youtube.com/watch?v={video_id}"
            
            # Create request with user agent
            request = urllib.request.Request(url)
            request.add_header('User-Agent', self.user_agent)
            
            # Get page content
            with urllib.request.urlopen(request) as response:
                page_content = response.read().decode('utf-8')
            
            # Extract metadata from page
            metadata = self._parse_metadata_from_page(page_content, video_id)
            
            return metadata
            
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise YouTubeTranscriptError("Video not found")
            elif e.code == 403:
                raise YouTubeTranscriptError("Access denied to video")
            else:
                raise YouTubeTranscriptError(f"HTTP error {e.code}: {e.reason}")
                
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
            from youtube_transcript_api import YouTubeTranscriptApi
            
            # Try to list available transcripts
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            
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


class YouTubeTranscriptFetcher:
    """Handles YouTube transcript fetching and processing."""
    
    def __init__(self):
        self.formatter = TextFormatter()
        self.api = YouTubeTranscriptApi
        self.metadata_extractor = YouTubeVideoMetadataExtractor()
    
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
            languages = ['en']
        
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