"""
YouTube video metadata extraction functionality.

This module handles the extraction and parsing of YouTube video metadata
from video pages and API responses.
"""

import logging
import re
import json
from typing import Dict, Any, Optional
import requests
from datetime import datetime

from .youtube_core import YouTubeTranscriptError, VideoNotFoundError
from .validators import YouTubeURLValidator
from .proxy_manager import get_proxy_manager, get_retry_manager

logger = logging.getLogger(__name__)


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
                raise VideoNotFoundError(video_id)
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
            page_content: HTML content of YouTube video page
            video_id: YouTube video ID
            
        Returns:
            Dictionary containing parsed metadata
        """
        metadata = {
            'video_id': video_id,
            'title': None,
            'duration_seconds': None,
            'view_count': None,
            'like_count': None,
            'channel_name': None,
            'channel_id': None,
            'description': None,
            'upload_date': None,
            'language': None,
            'thumbnail_url': None,
            'tags': [],
            'category': None,
            'is_live': False,
            'is_private': False,
            'is_age_restricted': False,
            'extraction_timestamp': datetime.utcnow().isoformat()
        }
        
        try:
            # Extract title using multiple patterns
            metadata['title'] = self._extract_title(page_content)
            
            # Extract duration
            metadata['duration_seconds'] = self._extract_duration(page_content)
            
            # Extract view count
            metadata['view_count'] = self._extract_view_count(page_content)
            
            # Extract channel information
            channel_info = self._extract_channel_info(page_content)
            metadata.update(channel_info)
            
            # Extract language
            metadata['language'] = self._extract_language(page_content)
            
            # Extract thumbnail URL
            metadata['thumbnail_url'] = self._extract_thumbnail_url(page_content, video_id)
            
            # Extract upload date
            metadata['upload_date'] = self._extract_upload_date(page_content)
            
            # Extract description
            metadata['description'] = self._extract_description(page_content)
            
            # Check for special video types
            metadata['is_live'] = self._is_live_video(page_content)
            metadata['is_private'] = self._is_private_video(page_content)
            metadata['is_age_restricted'] = self._is_age_restricted(page_content)
            
            # Extract tags and category
            metadata['tags'] = self._extract_tags(page_content)
            metadata['category'] = self._extract_category(page_content)
            
            logger.debug(f"Successfully parsed metadata for {video_id}: {metadata['title']}")
            
        except Exception as e:
            logger.warning(f"Error parsing some metadata fields for {video_id}: {str(e)}")
        
        return metadata
    
    def _extract_title(self, page_content: str) -> Optional[str]:
        """Extract video title from page content."""
        patterns = [
            r'"title":"([^"]+)"',
            r'<title>([^<]+)</title>',
            r'property="og:title" content="([^"]+)"',
            r'name="title" content="([^"]+)"'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, page_content)
            if match:
                title = match.group(1)
                # Clean up title
                title = title.replace(' - YouTube', '').strip()
                # Decode HTML entities
                title = self._decode_html_entities(title)
                return title
        
        return None
    
    def _extract_duration(self, page_content: str) -> Optional[int]:
        """Extract video duration in seconds."""
        patterns = [
            r'"lengthSeconds":"(\d+)"',
            r'"approxDurationMs":"(\d+)"',
            r'content="PT(\d+)M(\d+)S"',  # ISO 8601 format
            r'content="PT(\d+)H(\d+)M(\d+)S"'  # Hours included
        ]
        
        # Try lengthSeconds first (most reliable)
        match = re.search(patterns[0], page_content)
        if match:
            return int(match.group(1))
        
        # Try approxDurationMs
        match = re.search(patterns[1], page_content)
        if match:
            return int(match.group(1)) // 1000
        
        # Try ISO 8601 format (minutes and seconds)
        match = re.search(patterns[2], page_content)
        if match:
            minutes = int(match.group(1))
            seconds = int(match.group(2))
            return minutes * 60 + seconds
        
        # Try ISO 8601 format (hours, minutes, seconds)
        match = re.search(patterns[3], page_content)
        if match:
            hours = int(match.group(1))
            minutes = int(match.group(2))
            seconds = int(match.group(3))
            return hours * 3600 + minutes * 60 + seconds
        
        return None
    
    def _extract_view_count(self, page_content: str) -> Optional[int]:
        """Extract view count."""
        patterns = [
            r'"viewCount":"(\d+)"',
            r'"views":{"simpleText":"([\d,]+) views"}',
            r'(\d+(?:,\d+)*) views'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, page_content)
            if match:
                view_str = match.group(1).replace(',', '')
                try:
                    return int(view_str)
                except ValueError:
                    continue
        
        return None
    
    def _extract_channel_info(self, page_content: str) -> Dict[str, Optional[str]]:
        """Extract channel name and ID."""
        channel_info = {
            'channel_name': None,
            'channel_id': None
        }
        
        # Extract channel name
        patterns = [
            r'"ownerChannelName":"([^"]+)"',
            r'"author":"([^"]+)"',
            r'<link itemprop="name" content="([^"]+)">'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, page_content)
            if match:
                channel_info['channel_name'] = self._decode_html_entities(match.group(1))
                break
        
        # Extract channel ID
        channel_id_pattern = r'"channelId":"([^"]+)"'
        match = re.search(channel_id_pattern, page_content)
        if match:
            channel_info['channel_id'] = match.group(1)
        
        return channel_info
    
    def _extract_language(self, page_content: str) -> Optional[str]:
        """Extract video language."""
        patterns = [
            r'"defaultAudioLanguage":"([^"]+)"',
            r'"defaultLanguage":"([^"]+)"',
            r'<html[^>]+lang="([^"]+)"'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, page_content)
            if match:
                return match.group(1)
        
        return None
    
    def _extract_thumbnail_url(self, page_content: str, video_id: str) -> str:
        """Extract thumbnail URL."""
        # Try to extract from page content
        patterns = [
            r'"thumbnails":\[{"url":"([^"]+)"',
            r'property="og:image" content="([^"]+)"'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, page_content)
            if match:
                return match.group(1)
        
        # Fallback to standard thumbnail URL
        return f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg"
    
    def _extract_upload_date(self, page_content: str) -> Optional[str]:
        """Extract video upload date."""
        patterns = [
            r'"uploadDate":"([^"]+)"',
            r'"publishedTimeText":{"simpleText":"([^"]+)"}'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, page_content)
            if match:
                return match.group(1)
        
        return None
    
    def _extract_description(self, page_content: str) -> Optional[str]:
        """Extract video description."""
        patterns = [
            r'"shortDescription":"([^"]+)"',
            r'property="og:description" content="([^"]+)"',
            r'name="description" content="([^"]+)"'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, page_content)
            if match:
                desc = match.group(1)
                return self._decode_html_entities(desc)
        
        return None
    
    def _extract_tags(self, page_content: str) -> list:
        """Extract video tags."""
        tags = []
        
        # Try to extract from JSON-LD
        pattern = r'"keywords":"([^"]+)"'
        match = re.search(pattern, page_content)
        if match:
            keywords_str = match.group(1)
            tags = [tag.strip() for tag in keywords_str.split(',')]
        
        return tags
    
    def _extract_category(self, page_content: str) -> Optional[str]:
        """Extract video category."""
        pattern = r'"category":"([^"]+)"'
        match = re.search(pattern, page_content)
        if match:
            return match.group(1)
        
        return None
    
    def _is_live_video(self, page_content: str) -> bool:
        """Check if video is a live stream."""
        live_indicators = [
            '"isLiveContent":true',
            '"isLive":true',
            'LIVE</span>',
            'live-badge'
        ]
        
        return any(indicator in page_content for indicator in live_indicators)
    
    def _is_private_video(self, page_content: str) -> bool:
        """Check if video is private."""
        private_indicators = [
            '"isPrivate":true',
            'This video is private',
            'Video unavailable'
        ]
        
        return any(indicator in page_content for indicator in private_indicators)
    
    def _is_age_restricted(self, page_content: str) -> bool:
        """Check if video is age-restricted."""
        age_indicators = [
            '"isAgeGated":true',
            'Sign in to confirm your age',
            'age-gate'
        ]
        
        return any(indicator in page_content for indicator in age_indicators)
    
    def _decode_html_entities(self, text: str) -> str:
        """Decode HTML entities in text."""
        import html
        return html.unescape(text)
    
    def detect_unsupported_video_type(self, video_id: str, max_duration_seconds: int = 1800) -> Dict[str, Any]:
        """
        Detect if a video is of an unsupported type for transcript extraction.
        
        Args:
            video_id: YouTube video ID
            max_duration_seconds: Maximum allowed duration in seconds
            
        Returns:
            Dictionary containing support status and issues
        """
        try:
            metadata = self.extract_video_metadata(video_id)
            
            issues = []
            
            # Check if video is private
            if metadata.get('is_private', False):
                issues.append({
                    'type': 'private',
                    'message': 'Video is private and cannot be accessed'
                })
            
            # Check if video is live
            if metadata.get('is_live', False):
                issues.append({
                    'type': 'live',
                    'message': 'Live videos do not have transcripts available'
                })
            
            # Check if video is age-restricted
            if metadata.get('is_age_restricted', False):
                issues.append({
                    'type': 'age_restricted',
                    'message': 'Age-restricted videos may not have accessible transcripts'
                })
            
            # Check video duration
            duration = metadata.get('duration_seconds')
            if duration and duration > max_duration_seconds:
                issues.append({
                    'type': 'too_long',
                    'message': f'Video is too long ({duration}s > {max_duration_seconds}s maximum)'
                })
            
            is_supported = len(issues) == 0
            
            return {
                'is_supported': is_supported,
                'issues': issues,
                'metadata': metadata
            }
            
        except Exception as e:
            logger.error(f"Error detecting video support status for {video_id}: {str(e)}")
            return {
                'is_supported': False,
                'issues': [{
                    'type': 'detection_error',
                    'message': f'Could not determine video support status: {str(e)}'
                }],
                'metadata': None
            }