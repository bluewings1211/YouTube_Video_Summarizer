"""
YouTube URL validation and video ID extraction utilities.

This module provides functionality to validate YouTube URLs and extract video IDs
from various YouTube URL formats including standard watch URLs, shortened URLs,
and embedded URLs.
"""

import re
from typing import Optional, Tuple, Dict, Any
from urllib.parse import urlparse, parse_qs
from enum import Enum
from dataclasses import dataclass


class ValidationError(Exception):
    """Base exception for validation errors."""
    pass


class URLValidationError(ValidationError):
    """Exception for URL validation errors."""
    pass


class VideoIDValidationError(ValidationError):
    """Exception for video ID validation errors."""
    pass


class URLValidationResult(Enum):
    """Enumeration for URL validation results."""
    VALID = "valid"
    INVALID_FORMAT = "invalid_format"
    INVALID_DOMAIN = "invalid_domain"
    INVALID_VIDEO_ID = "invalid_video_id"
    TOO_LONG = "too_long"
    EMPTY_URL = "empty_url"
    INVALID_TYPE = "invalid_type"
    MALFORMED_URL = "malformed_url"
    SUSPICIOUS_CONTENT = "suspicious_content"


@dataclass
class ValidationResult:
    """Result of URL validation with detailed information."""
    is_valid: bool
    result_type: URLValidationResult
    video_id: Optional[str] = None
    error_message: Optional[str] = None
    warning_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class YouTubeURLValidator:
    """Validates YouTube URLs and extracts video IDs with comprehensive error handling."""
    
    # YouTube URL patterns with named groups for better error reporting
    YOUTUBE_URL_PATTERNS = [
        r'(?:https?://)?(?:www\.)?youtube\.com/watch\?.*v=([a-zA-Z0-9_-]{11})',
        r'(?:https?://)?(?:www\.)?youtube\.com/embed/([a-zA-Z0-9_-]{11})',
        r'(?:https?://)?(?:www\.)?youtube\.com/v/([a-zA-Z0-9_-]{11})',
        r'(?:https?://)?youtu\.be/([a-zA-Z0-9_-]{11})',
        r'(?:https?://)?(?:www\.)?youtube\.com/shorts/([a-zA-Z0-9_-]{11})',
    ]
    
    # Maximum URL length to prevent abuse
    MAX_URL_LENGTH = 2000
    
    # Valid YouTube domains
    VALID_DOMAINS = {
        'youtube.com',
        'www.youtube.com',
        'youtu.be',
        'm.youtube.com',
        'music.youtube.com'
    }
    
    # Suspicious patterns that might indicate malicious content
    SUSPICIOUS_PATTERNS = [
        r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]',  # Control characters
        r'javascript:',  # JavaScript URLs
        r'data:',  # Data URLs
        r'<script',  # Script tags
        r'%3Cscript',  # URL encoded script tags
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
        result = cls.validate_youtube_url_detailed(url)
        return result.is_valid
    
    @classmethod
    def validate_youtube_url_detailed(cls, url: str) -> ValidationResult:
        """
        Validate a YouTube URL with detailed error reporting.
        
        Args:
            url: The URL to validate
            
        Returns:
            ValidationResult with detailed information about the validation
        """
        # Check for None or empty URL
        if url is None:
            return ValidationResult(
                is_valid=False,
                result_type=URLValidationResult.EMPTY_URL,
                error_message="URL cannot be None"
            )
        
        # Check for correct type
        if not isinstance(url, str):
            return ValidationResult(
                is_valid=False,
                result_type=URLValidationResult.INVALID_TYPE,
                error_message=f"URL must be a string, got {type(url).__name__}"
            )
        
        # Check for empty string
        if not url.strip():
            return ValidationResult(
                is_valid=False,
                result_type=URLValidationResult.EMPTY_URL,
                error_message="URL cannot be empty"
            )
        
        # Clean and normalize URL
        url = url.strip()
        
        # Check URL length
        if len(url) > cls.MAX_URL_LENGTH:
            return ValidationResult(
                is_valid=False,
                result_type=URLValidationResult.TOO_LONG,
                error_message=f"URL too long (max {cls.MAX_URL_LENGTH} characters)"
            )
        
        # Check for suspicious content
        for pattern in cls.SUSPICIOUS_PATTERNS:
            if re.search(pattern, url, re.IGNORECASE):
                return ValidationResult(
                    is_valid=False,
                    result_type=URLValidationResult.SUSPICIOUS_CONTENT,
                    error_message="URL contains suspicious content"
                )
        
        # Try to parse URL to check basic structure
        try:
            parsed_url = urlparse(url)
            
            # Check if we have a valid scheme for URLs with scheme
            if parsed_url.scheme and parsed_url.scheme not in ['http', 'https']:
                return ValidationResult(
                    is_valid=False,
                    result_type=URLValidationResult.MALFORMED_URL,
                    error_message=f"Invalid URL scheme: {parsed_url.scheme}"
                )
            
            # Check domain if URL has a scheme
            if parsed_url.netloc:
                domain = parsed_url.netloc.lower()
                if domain not in cls.VALID_DOMAINS:
                    return ValidationResult(
                        is_valid=False,
                        result_type=URLValidationResult.INVALID_DOMAIN,
                        error_message=f"Invalid domain: {domain}"
                    )
        
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                result_type=URLValidationResult.MALFORMED_URL,
                error_message=f"Malformed URL: {str(e)}"
            )
        
        # Check against YouTube URL patterns and extract video ID
        video_id = None
        for pattern in cls.YOUTUBE_URL_PATTERNS:
            match = re.search(pattern, url, re.IGNORECASE)
            if match:
                video_id = match.group(1)
                break
        
        # If no pattern matched, try alternative extraction methods
        if not video_id:
            video_id = cls._extract_video_id_alternative(url)
        
        # Validate video ID if found
        if video_id:
            if not cls._is_valid_video_id(video_id):
                return ValidationResult(
                    is_valid=False,
                    result_type=URLValidationResult.INVALID_VIDEO_ID,
                    error_message=f"Invalid video ID format: {video_id}"
                )
            
            # Success case
            return ValidationResult(
                is_valid=True,
                result_type=URLValidationResult.VALID,
                video_id=video_id,
                metadata={
                    'normalized_url': url,
                    'domain': parsed_url.netloc if parsed_url.netloc else 'youtube.com'
                }
            )
        
        # No video ID found
        return ValidationResult(
            is_valid=False,
            result_type=URLValidationResult.INVALID_FORMAT,
            error_message="No valid YouTube video ID found in URL"
        )
    
    @classmethod
    def _extract_video_id_alternative(cls, url: str) -> Optional[str]:
        """
        Alternative video ID extraction using URL parsing.
        
        Args:
            url: The URL to extract video ID from
            
        Returns:
            Video ID if found, None otherwise
        """
        try:
            parsed_url = urlparse(url)
            
            # Handle URLs with query parameters
            if parsed_url.query:
                query_params = parse_qs(parsed_url.query)
                if 'v' in query_params and query_params['v']:
                    video_id = query_params['v'][0]
                    if cls._is_valid_video_id(video_id):
                        return video_id
            
            # Handle path-based URLs
            path_parts = parsed_url.path.strip('/').split('/')
            for i, part in enumerate(path_parts):
                if part in ['watch', 'embed', 'v', 'shorts'] and i + 1 < len(path_parts):
                    video_id = path_parts[i + 1]
                    if cls._is_valid_video_id(video_id):
                        return video_id
            
            # Handle youtu.be domain
            if 'youtu.be' in url.lower():
                path = parsed_url.path.strip('/')
                if path and cls._is_valid_video_id(path):
                    return path
                    
        except Exception:
            pass
        
        return None
    
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
    def validate_video_id(cls, video_id: str) -> ValidationResult:
        """
        Validate a video ID with detailed error reporting.
        
        Args:
            video_id: The video ID to validate
            
        Returns:
            ValidationResult with detailed information
        """
        if video_id is None:
            return ValidationResult(
                is_valid=False,
                result_type=URLValidationResult.EMPTY_URL,
                error_message="Video ID cannot be None"
            )
        
        if not isinstance(video_id, str):
            return ValidationResult(
                is_valid=False,
                result_type=URLValidationResult.INVALID_TYPE,
                error_message=f"Video ID must be a string, got {type(video_id).__name__}"
            )
        
        if not video_id.strip():
            return ValidationResult(
                is_valid=False,
                result_type=URLValidationResult.EMPTY_URL,
                error_message="Video ID cannot be empty"
            )
        
        video_id = video_id.strip()
        
        if len(video_id) != 11:
            return ValidationResult(
                is_valid=False,
                result_type=URLValidationResult.INVALID_VIDEO_ID,
                error_message=f"Video ID must be 11 characters long, got {len(video_id)}"
            )
        
        if not re.match(r'^[a-zA-Z0-9_-]{11}$', video_id):
            return ValidationResult(
                is_valid=False,
                result_type=URLValidationResult.INVALID_VIDEO_ID,
                error_message="Video ID contains invalid characters (only letters, numbers, hyphens, and underscores allowed)"
            )
        
        return ValidationResult(
            is_valid=True,
            result_type=URLValidationResult.VALID,
            video_id=video_id
        )
    
    @classmethod
    def sanitize_url(cls, url: str) -> str:
        """
        Sanitize a URL by removing potentially dangerous content.
        
        Args:
            url: The URL to sanitize
            
        Returns:
            Sanitized URL
        """
        if not url or not isinstance(url, str):
            return ""
        
        # Remove control characters
        url = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', '', url)
        
        # Remove dangerous protocols
        url = re.sub(r'^(javascript|data|vbscript):', '', url, flags=re.IGNORECASE)
        
        # Strip whitespace
        url = url.strip()
        
        return url
    
    @classmethod
    def normalize_url(cls, url: str) -> str:
        """
        Normalize a YouTube URL to a standard format.
        
        Args:
            url: The URL to normalize
            
        Returns:
            Normalized URL
        """
        result = cls.validate_youtube_url_detailed(url)
        if not result.is_valid or not result.video_id:
            return url
        
        # Return standard watch URL format
        return f"https://www.youtube.com/watch?v={result.video_id}"
    
    @classmethod
    def get_url_info(cls, url: str) -> Dict[str, Any]:
        """
        Get detailed information about a YouTube URL.
        
        Args:
            url: The URL to analyze
            
        Returns:
            Dictionary with URL information
        """
        result = cls.validate_youtube_url_detailed(url)
        
        info = {
            'is_valid': result.is_valid,
            'validation_result': result.result_type.value,
            'video_id': result.video_id,
            'error_message': result.error_message,
            'warning_message': result.warning_message
        }
        
        if result.metadata:
            info.update(result.metadata)
        
        if result.is_valid and result.video_id:
            info['normalized_url'] = cls.normalize_url(url)
            info['video_url'] = f"https://www.youtube.com/watch?v={result.video_id}"
            info['embed_url'] = f"https://www.youtube.com/embed/{result.video_id}"
            info['thumbnail_url'] = f"https://img.youtube.com/vi/{result.video_id}/maxresdefault.jpg"
        
        return info
    
    @classmethod
    def validate_and_extract(cls, url: str) -> Tuple[bool, Optional[str]]:
        """
        Validate URL and extract video ID in one operation.
        
        Args:
            url: The YouTube URL to validate and extract from
            
        Returns:
            Tuple of (is_valid, video_id)
        """
        result = cls.validate_youtube_url_detailed(url)
        return result.is_valid, result.video_id
    
    @classmethod
    def validate_and_extract_detailed(cls, url: str) -> ValidationResult:
        """
        Validate URL and extract video ID with detailed error reporting.
        
        Args:
            url: The YouTube URL to validate and extract from
            
        Returns:
            ValidationResult with detailed information
        """
        return cls.validate_youtube_url_detailed(url)
    
    @classmethod
    def batch_validate(cls, urls: list) -> Dict[str, ValidationResult]:
        """
        Validate multiple URLs at once.
        
        Args:
            urls: List of URLs to validate
            
        Returns:
            Dictionary mapping URLs to their validation results
        """
        results = {}
        
        if not isinstance(urls, list):
            return results
        
        for url in urls:
            if isinstance(url, str):
                results[url] = cls.validate_youtube_url_detailed(url)
            else:
                results[str(url)] = ValidationResult(
                    is_valid=False,
                    result_type=URLValidationResult.INVALID_TYPE,
                    error_message=f"URL must be a string, got {type(url).__name__}"
                )
        
        return results


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


def validate_youtube_url_detailed(url: str) -> ValidationResult:
    """
    Convenience function to validate YouTube URLs with detailed error reporting.
    
    Args:
        url: The URL to validate
        
    Returns:
        ValidationResult with detailed information
    """
    return YouTubeURLValidator.validate_youtube_url_detailed(url)


def validate_video_id(video_id: str) -> ValidationResult:
    """
    Convenience function to validate a YouTube video ID.
    
    Args:
        video_id: The video ID to validate
        
    Returns:
        ValidationResult with detailed information
    """
    return YouTubeURLValidator.validate_video_id(video_id)


def sanitize_url(url: str) -> str:
    """
    Convenience function to sanitize a URL.
    
    Args:
        url: The URL to sanitize
        
    Returns:
        Sanitized URL
    """
    return YouTubeURLValidator.sanitize_url(url)


def normalize_url(url: str) -> str:
    """
    Convenience function to normalize a YouTube URL.
    
    Args:
        url: The URL to normalize
        
    Returns:
        Normalized URL
    """
    return YouTubeURLValidator.normalize_url(url)


def get_url_info(url: str) -> Dict[str, Any]:
    """
    Convenience function to get detailed information about a YouTube URL.
    
    Args:
        url: The URL to analyze
        
    Returns:
        Dictionary with URL information
    """
    return YouTubeURLValidator.get_url_info(url)


def validate_and_extract_detailed(url: str) -> ValidationResult:
    """
    Convenience function to validate and extract video ID with detailed error reporting.
    
    Args:
        url: The YouTube URL to validate and extract from
        
    Returns:
        ValidationResult with detailed information
    """
    return YouTubeURLValidator.validate_and_extract_detailed(url)


def batch_validate_urls(urls: list) -> Dict[str, ValidationResult]:
    """
    Convenience function to validate multiple URLs at once.
    
    Args:
        urls: List of URLs to validate
        
    Returns:
        Dictionary mapping URLs to their validation results
    """
    return YouTubeURLValidator.batch_validate(urls)