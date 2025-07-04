"""
Unit tests for YouTube URL validation and video ID extraction utilities.

This module contains comprehensive tests for the validators module,
testing all URL validation patterns and edge cases.
"""

import pytest
from src.utils.validators import (
    YouTubeURLValidator,
    validate_youtube_url,
    extract_youtube_video_id,
    validate_and_extract_youtube_video_id
)


class TestYouTubeURLValidator:
    """Test cases for YouTubeURLValidator class."""
    
    def test_validate_youtube_url_standard_format(self):
        """Test validation of standard YouTube URLs."""
        valid_urls = [
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "http://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://youtube.com/watch?v=dQw4w9WgXcQ",
            "http://youtube.com/watch?v=dQw4w9WgXcQ",
            "www.youtube.com/watch?v=dQw4w9WgXcQ",
            "youtube.com/watch?v=dQw4w9WgXcQ"
        ]
        
        for url in valid_urls:
            assert YouTubeURLValidator.validate_youtube_url(url) is True
    
    def test_validate_youtube_url_shortened_format(self):
        """Test validation of shortened YouTube URLs."""
        valid_urls = [
            "https://youtu.be/dQw4w9WgXcQ",
            "http://youtu.be/dQw4w9WgXcQ",
            "youtu.be/dQw4w9WgXcQ"
        ]
        
        for url in valid_urls:
            assert YouTubeURLValidator.validate_youtube_url(url) is True
    
    def test_validate_youtube_url_embed_format(self):
        """Test validation of embedded YouTube URLs."""
        valid_urls = [
            "https://www.youtube.com/embed/dQw4w9WgXcQ",
            "http://www.youtube.com/embed/dQw4w9WgXcQ",
            "https://youtube.com/embed/dQw4w9WgXcQ",
            "www.youtube.com/embed/dQw4w9WgXcQ",
            "youtube.com/embed/dQw4w9WgXcQ"
        ]
        
        for url in valid_urls:
            assert YouTubeURLValidator.validate_youtube_url(url) is True
    
    def test_validate_youtube_url_v_format(self):
        """Test validation of /v/ YouTube URLs."""
        valid_urls = [
            "https://www.youtube.com/v/dQw4w9WgXcQ",
            "http://www.youtube.com/v/dQw4w9WgXcQ",
            "https://youtube.com/v/dQw4w9WgXcQ",
            "www.youtube.com/v/dQw4w9WgXcQ",
            "youtube.com/v/dQw4w9WgXcQ"
        ]
        
        for url in valid_urls:
            assert YouTubeURLValidator.validate_youtube_url(url) is True
    
    def test_validate_youtube_url_shorts_format(self):
        """Test validation of YouTube Shorts URLs."""
        valid_urls = [
            "https://www.youtube.com/shorts/dQw4w9WgXcQ",
            "http://www.youtube.com/shorts/dQw4w9WgXcQ",
            "https://youtube.com/shorts/dQw4w9WgXcQ",
            "www.youtube.com/shorts/dQw4w9WgXcQ",
            "youtube.com/shorts/dQw4w9WgXcQ"
        ]
        
        for url in valid_urls:
            assert YouTubeURLValidator.validate_youtube_url(url) is True
    
    def test_validate_youtube_url_invalid_formats(self):
        """Test validation fails for invalid URLs."""
        invalid_urls = [
            "https://www.google.com",
            "https://vimeo.com/123456789",
            "https://www.youtube.com/user/testuser",
            "https://www.youtube.com/channel/UC123456789",
            "not-a-url",
            "",
            None,
            123,
            [],
            {}
        ]
        
        for url in invalid_urls:
            assert YouTubeURLValidator.validate_youtube_url(url) is False
    
    def test_extract_video_id_standard_format(self):
        """Test video ID extraction from standard URLs."""
        test_cases = [
            ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("http://www.youtube.com/watch?v=abc123def45", "abc123def45"),
            ("youtube.com/watch?v=XyZ_987-654", "XyZ_987-654")
        ]
        
        for url, expected_id in test_cases:
            assert YouTubeURLValidator.extract_video_id(url) == expected_id
    
    def test_extract_video_id_shortened_format(self):
        """Test video ID extraction from shortened URLs."""
        test_cases = [
            ("https://youtu.be/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("http://youtu.be/abc123def45", "abc123def45"),
            ("youtu.be/XyZ_987-654", "XyZ_987-654")
        ]
        
        for url, expected_id in test_cases:
            assert YouTubeURLValidator.extract_video_id(url) == expected_id
    
    def test_extract_video_id_embed_format(self):
        """Test video ID extraction from embed URLs."""
        test_cases = [
            ("https://www.youtube.com/embed/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("youtube.com/embed/abc123def45", "abc123def45")
        ]
        
        for url, expected_id in test_cases:
            assert YouTubeURLValidator.extract_video_id(url) == expected_id
    
    def test_extract_video_id_shorts_format(self):
        """Test video ID extraction from Shorts URLs."""
        test_cases = [
            ("https://www.youtube.com/shorts/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("youtube.com/shorts/abc123def45", "abc123def45")
        ]
        
        for url, expected_id in test_cases:
            assert YouTubeURLValidator.extract_video_id(url) == expected_id
    
    def test_extract_video_id_with_parameters(self):
        """Test video ID extraction from URLs with additional parameters."""
        test_cases = [
            ("https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=30s", "dQw4w9WgXcQ"),
            ("https://www.youtube.com/watch?v=dQw4w9WgXcQ&list=PLhQjrBD2T380F7R4-8V6pVgQkgIlmfhFg", "dQw4w9WgXcQ"),
            ("https://www.youtube.com/watch?feature=youtu.be&v=dQw4w9WgXcQ", "dQw4w9WgXcQ")
        ]
        
        for url, expected_id in test_cases:
            assert YouTubeURLValidator.extract_video_id(url) == expected_id
    
    def test_extract_video_id_invalid_inputs(self):
        """Test video ID extraction returns None for invalid inputs."""
        invalid_inputs = [
            "https://www.google.com",
            "https://vimeo.com/123456789",
            "not-a-url",
            "",
            None,
            123,
            [],
            {}
        ]
        
        for invalid_input in invalid_inputs:
            assert YouTubeURLValidator.extract_video_id(invalid_input) is None
    
    def test_is_valid_video_id(self):
        """Test video ID format validation."""
        valid_ids = [
            "dQw4w9WgXcQ",
            "abc123def45",
            "XyZ_987-654",
            "0123456789a",
            "ABCDEFGHIJK"
        ]
        
        for video_id in valid_ids:
            assert YouTubeURLValidator._is_valid_video_id(video_id) is True
    
    def test_is_valid_video_id_invalid(self):
        """Test video ID format validation for invalid IDs."""
        invalid_ids = [
            "short",
            "toolongvideoid123",
            "invalid@char",
            "spac e",
            "",
            None,
            123,
            "dQw4w9WgXc",  # too short
            "dQw4w9WgXcQQ"  # too long
        ]
        
        for video_id in invalid_ids:
            assert YouTubeURLValidator._is_valid_video_id(video_id) is False
    
    def test_validate_and_extract(self):
        """Test combined validation and extraction."""
        # Valid URL
        is_valid, video_id = YouTubeURLValidator.validate_and_extract(
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        )
        assert is_valid is True
        assert video_id == "dQw4w9WgXcQ"
        
        # Invalid URL
        is_valid, video_id = YouTubeURLValidator.validate_and_extract(
            "https://www.google.com"
        )
        assert is_valid is False
        assert video_id is None
    
    def test_whitespace_handling(self):
        """Test that URLs with whitespace are handled correctly."""
        urls_with_whitespace = [
            "  https://www.youtube.com/watch?v=dQw4w9WgXcQ  ",
            "\thttps://youtu.be/dQw4w9WgXcQ\n",
            " youtube.com/watch?v=dQw4w9WgXcQ "
        ]
        
        for url in urls_with_whitespace:
            assert YouTubeURLValidator.validate_youtube_url(url) is True
            assert YouTubeURLValidator.extract_video_id(url) == "dQw4w9WgXcQ"


class TestConvenienceFunctions:
    """Test cases for module-level convenience functions."""
    
    def test_validate_youtube_url_function(self):
        """Test standalone validate_youtube_url function."""
        assert validate_youtube_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ") is True
        assert validate_youtube_url("https://www.google.com") is False
    
    def test_extract_youtube_video_id_function(self):
        """Test standalone extract_youtube_video_id function."""
        assert extract_youtube_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"
        assert extract_youtube_video_id("https://www.google.com") is None
    
    def test_validate_and_extract_youtube_video_id_function(self):
        """Test standalone validate_and_extract_youtube_video_id function."""
        is_valid, video_id = validate_and_extract_youtube_video_id(
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        )
        assert is_valid is True
        assert video_id == "dQw4w9WgXcQ"
        
        is_valid, video_id = validate_and_extract_youtube_video_id(
            "https://www.google.com"
        )
        assert is_valid is False
        assert video_id is None


class TestEdgeCases:
    """Test edge cases and unusual inputs."""
    
    def test_video_id_with_all_allowed_characters(self):
        """Test video ID with all allowed character types."""
        video_id = "aZ9_-012345"
        url = f"https://www.youtube.com/watch?v={video_id}"
        
        assert YouTubeURLValidator.validate_youtube_url(url) is True
        assert YouTubeURLValidator.extract_video_id(url) == video_id
    
    def test_case_sensitivity(self):
        """Test that video IDs are case-sensitive."""
        lower_case = "abcdefghijk"
        upper_case = "ABCDEFGHIJK"
        mixed_case = "AbCdEfGhIjK"
        
        for video_id in [lower_case, upper_case, mixed_case]:
            url = f"https://www.youtube.com/watch?v={video_id}"
            assert YouTubeURLValidator.validate_youtube_url(url) is True
            assert YouTubeURLValidator.extract_video_id(url) == video_id
    
    def test_malformed_urls(self):
        """Test handling of malformed URLs."""
        malformed_urls = [
            "https://youtube.com/watch?v=",
            "https://youtube.com/watch?",
            "https://youtube.com/watch",
            "https://youtube.com/",
            "youtube.com/watch?v=short",
            "youtube.com/watch?v=toolongvideoidhere"
        ]
        
        for url in malformed_urls:
            assert YouTubeURLValidator.validate_youtube_url(url) is False
            assert YouTubeURLValidator.extract_video_id(url) is None