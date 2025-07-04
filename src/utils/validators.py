"""
YouTube URL validation and video ID extraction utilities.

This module provides functionality to validate YouTube URLs and extract video IDs
from various YouTube URL formats including standard watch URLs, shortened URLs,
and embedded URLs.
"""

import re
from typing import Optional, Tuple
from urllib.parse import urlparse, parse_qs


class YouTubeURLValidator:
    """Validates YouTube URLs and extracts video IDs."""
    
    # YouTube URL patterns
    YOUTUBE_URL_PATTERNS = [
        r'(?:https?://)?(?:www\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]{11})',
        r'(?:https?://)?(?:www\.)?youtube\.com/embed/([a-zA-Z0-9_-]{11})',
        r'(?:https?://)?(?:www\.)?youtube\.com/v/([a-zA-Z0-9_-]{11})',
        r'(?:https?://)?youtu\.be/([a-zA-Z0-9_-]{11})',
        r'(?:https?://)?(?:www\.)?youtube\.com/shorts/([a-zA-Z0-9_-]{11})',
    ]
    
    @classmethod
    def validate_youtube_url(cls, url: str) -> bool:
        """
        Validate if a URL is a valid YouTube URL.
        
        Args:
            url: The URL to validate
            
        Returns:
            True if URL is a valid YouTube URL, False otherwise
        """
        if not url or not isinstance(url, str):
            return False
            
        # Check against all YouTube URL patterns
        for pattern in cls.YOUTUBE_URL_PATTERNS:
            if re.match(pattern, url.strip()):
                return True
                
        return False
    
    @classmethod
    def extract_video_id(cls, url: str) -> Optional[str]:
        """
        Extract video ID from a YouTube URL.
        
        Args:
            url: The YouTube URL to extract video ID from
            
        Returns:
            Video ID if found, None otherwise
        """
        if not url or not isinstance(url, str):
            return None
            
        url = url.strip()
        
        # Try each pattern to extract video ID
        for pattern in cls.YOUTUBE_URL_PATTERNS:
            match = re.match(pattern, url)
            if match:
                return match.group(1)
        
        # Special handling for URLs with additional parameters
        try:
            parsed_url = urlparse(url)
            if parsed_url.hostname in ['www.youtube.com', 'youtube.com']:
                # Check for v parameter in query string
                query_params = parse_qs(parsed_url.query)
                if 'v' in query_params:
                    video_id = query_params['v'][0]
                    if cls._is_valid_video_id(video_id):
                        return video_id
        except Exception:
            pass
            
        return None
    
    @classmethod
    def _is_valid_video_id(cls, video_id: str) -> bool:
        """
        Check if a video ID has the correct format.
        
        Args:
            video_id: The video ID to validate
            
        Returns:
            True if video ID format is valid, False otherwise
        """
        if not video_id or not isinstance(video_id, str):
            return False
            
        # YouTube video IDs are 11 characters long and contain only
        # letters, numbers, hyphens, and underscores
        pattern = r'^[a-zA-Z0-9_-]{11}$'
        return bool(re.match(pattern, video_id))
    
    @classmethod
    def validate_and_extract(cls, url: str) -> Tuple[bool, Optional[str]]:
        """
        Validate URL and extract video ID in one operation.
        
        Args:
            url: The YouTube URL to validate and extract from
            
        Returns:
            Tuple of (is_valid, video_id)
        """
        is_valid = cls.validate_youtube_url(url)
        video_id = cls.extract_video_id(url) if is_valid else None
        
        return is_valid, video_id


def validate_youtube_url(url: str) -> bool:
    """
    Convenience function to validate YouTube URLs.
    
    Args:
        url: The URL to validate
        
    Returns:
        True if URL is valid, False otherwise
    """
    return YouTubeURLValidator.validate_youtube_url(url)


def extract_youtube_video_id(url: str) -> Optional[str]:
    """
    Convenience function to extract video ID from YouTube URLs.
    
    Args:
        url: The YouTube URL to extract video ID from
        
    Returns:
        Video ID if found, None otherwise
    """
    return YouTubeURLValidator.extract_video_id(url)


def validate_and_extract_youtube_video_id(url: str) -> Tuple[bool, Optional[str]]:
    """
    Convenience function to validate and extract video ID from YouTube URLs.
    
    Args:
        url: The YouTube URL to validate and extract from
        
    Returns:
        Tuple of (is_valid, video_id)
    """
    return YouTubeURLValidator.validate_and_extract(url)