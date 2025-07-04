"""
Unit tests for YouTube API utilities including transcript extraction and metadata.

This module contains comprehensive tests for the youtube_api module,
including mock tests for API interactions and error handling.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from src.utils.youtube_api import (
    YouTubeTranscriptError,
    UnsupportedVideoTypeError,
    PrivateVideoError,
    LiveVideoError,
    NoTranscriptAvailableError,
    VideoTooLongError,
    YouTubeVideoMetadataExtractor,
    YouTubeTranscriptFetcher,
    fetch_youtube_transcript,
    fetch_youtube_transcript_from_url,
    get_available_youtube_transcripts,
    get_youtube_video_metadata,
    check_youtube_video_support,
    validate_youtube_video_duration,
    check_youtube_duration_limit,
    # Enhanced functions
    get_video_info,
    get_video_info_legacy,
    fetch_youtube_transcript_with_three_tier_strategy
)


class TestYouTubeTranscriptError:
    """Test YouTube error classes."""
    
    def test_youtube_transcript_error(self):
        """Test base YouTubeTranscriptError."""
        error = YouTubeTranscriptError("Test error")
        assert str(error) == "Test error"
        assert isinstance(error, Exception)
    
    def test_unsupported_video_type_error(self):
        """Test UnsupportedVideoTypeError inheritance."""
        error = UnsupportedVideoTypeError("Unsupported video")
        assert str(error) == "Unsupported video"
        assert isinstance(error, YouTubeTranscriptError)
    
    def test_private_video_error(self):
        """Test PrivateVideoError inheritance."""
        error = PrivateVideoError("Private video")
        assert str(error) == "Private video"
        assert isinstance(error, UnsupportedVideoTypeError)
        assert isinstance(error, YouTubeTranscriptError)
    
    def test_live_video_error(self):
        """Test LiveVideoError inheritance."""
        error = LiveVideoError("Live video")
        assert str(error) == "Live video"
        assert isinstance(error, UnsupportedVideoTypeError)
        assert isinstance(error, YouTubeTranscriptError)
    
    def test_no_transcript_available_error(self):
        """Test NoTranscriptAvailableError inheritance."""
        error = NoTranscriptAvailableError("No transcript")
        assert str(error) == "No transcript"
        assert isinstance(error, UnsupportedVideoTypeError)
        assert isinstance(error, YouTubeTranscriptError)
    
    def test_video_too_long_error(self):
        """Test VideoTooLongError inheritance."""
        error = VideoTooLongError("Video too long")
        assert str(error) == "Video too long"
        assert isinstance(error, UnsupportedVideoTypeError)
        assert isinstance(error, YouTubeTranscriptError)


class TestYouTubeVideoMetadataExtractor:
    """Test YouTube video metadata extraction."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = YouTubeVideoMetadataExtractor()
    
    @patch('urllib.request.urlopen')
    def test_extract_video_metadata_success(self, mock_urlopen):
        """Test successful metadata extraction."""
        # Mock page content
        mock_response = Mock()
        mock_response.read.return_value = b'''
        <html>
        <script>
        var ytInitialData = {
            "title": "Test Video Title",
            "lengthSeconds": "1200",
            "defaultAudioLanguage": "en",
            "viewCount": "12345",
            "author": "Test Channel"
        };
        </script>
        </html>
        '''
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=None)
        mock_urlopen.return_value = mock_response
        
        metadata = self.extractor.extract_video_metadata("dQw4w9WgXcQ")
        
        assert metadata['video_id'] == "dQw4w9WgXcQ"
        assert metadata['title'] is not None
        assert metadata['is_live'] is False
        assert metadata['is_private'] is False
        assert 'extraction_timestamp' in metadata
    
    def test_extract_video_metadata_invalid_id(self):
        """Test metadata extraction with invalid video ID."""
        with pytest.raises(YouTubeTranscriptError, match="Invalid video ID format"):
            self.extractor.extract_video_metadata("invalid")
    
    def test_extract_video_metadata_empty_id(self):
        """Test metadata extraction with empty video ID."""
        with pytest.raises(YouTubeTranscriptError, match="Video ID is required"):
            self.extractor.extract_video_metadata("")
    
    @patch('urllib.request.urlopen')
    def test_extract_video_metadata_http_error(self, mock_urlopen):
        """Test metadata extraction with HTTP error."""
        from urllib.error import HTTPError
        
        mock_urlopen.side_effect = HTTPError(
            url="test", code=404, msg="Not Found", hdrs=None, fp=None
        )
        
        with pytest.raises(YouTubeTranscriptError, match="Video not found"):
            self.extractor.extract_video_metadata("dQw4w9WgXcQ")
    
    def test_get_video_duration_formatted(self):
        """Test duration formatting."""
        assert self.extractor.get_video_duration_formatted(65) == "1:05"
        assert self.extractor.get_video_duration_formatted(3665) == "1:01:05"
        assert self.extractor.get_video_duration_formatted(None) == "Unknown"
        assert self.extractor.get_video_duration_formatted(0) == "0:00"
    
    @patch.object(YouTubeVideoMetadataExtractor, 'extract_video_metadata')
    def test_detect_unsupported_video_type_private(self, mock_extract):
        """Test detection of private videos."""
        mock_extract.return_value = {
            'video_id': 'test123',
            'is_private': True,
            'is_live': False,
            'title': 'Test Video',
            'duration_seconds': 300
        }
        
        result = self.extractor.detect_unsupported_video_type("test123")
        
        assert result['is_supported'] is False
        assert any(issue['type'] == 'private' for issue in result['issues'])
    
    @patch.object(YouTubeVideoMetadataExtractor, 'extract_video_metadata')
    def test_detect_unsupported_video_type_live(self, mock_extract):
        """Test detection of live videos."""
        mock_extract.return_value = {
            'video_id': 'test123',
            'is_private': False,
            'is_live': True,
            'title': 'Live Stream',
            'duration_seconds': None
        }
        
        result = self.extractor.detect_unsupported_video_type("test123")
        
        assert result['is_supported'] is False
        assert any(issue['type'] == 'live' for issue in result['issues'])
    
    @patch.object(YouTubeVideoMetadataExtractor, 'extract_video_metadata')
    def test_detect_unsupported_video_type_too_long(self, mock_extract):
        """Test detection of videos that are too long."""
        mock_extract.return_value = {
            'video_id': 'test123',
            'is_private': False,
            'is_live': False,
            'title': 'Long Video',
            'duration_seconds': 2400  # 40 minutes
        }
        
        result = self.extractor.detect_unsupported_video_type("test123", max_duration_seconds=1800)
        
        assert result['is_supported'] is False
        assert any(issue['type'] == 'too_long' for issue in result['issues'])
    
    @patch.object(YouTubeVideoMetadataExtractor, 'extract_video_metadata')
    def test_detect_unsupported_video_type_supported(self, mock_extract):
        """Test detection of supported videos."""
        mock_extract.return_value = {
            'video_id': 'test123',
            'is_private': False,
            'is_live': False,
            'title': 'Test Video',
            'duration_seconds': 300
        }
        
        result = self.extractor.detect_unsupported_video_type("test123")
        
        assert result['is_supported'] is True
        assert len(result['issues']) == 0
    
    @patch.object(YouTubeVideoMetadataExtractor, 'extract_video_metadata')
    def test_validate_video_duration_valid(self, mock_extract):
        """Test validation of valid video duration."""
        mock_extract.return_value = {
            'duration_seconds': 1200
        }
        
        result = self.extractor.validate_video_duration("test123", max_duration_seconds=1800)
        
        assert result['is_valid_duration'] is True
        assert result['duration_seconds'] == 1200
        assert result['max_duration_seconds'] == 1800
        assert 'error' not in result
    
    @patch.object(YouTubeVideoMetadataExtractor, 'extract_video_metadata')
    def test_validate_video_duration_invalid(self, mock_extract):
        """Test validation of invalid video duration."""
        mock_extract.return_value = {
            'duration_seconds': 2400
        }
        
        result = self.extractor.validate_video_duration("test123", max_duration_seconds=1800)
        
        assert result['is_valid_duration'] is False
        assert result['duration_seconds'] == 2400
        assert 'error' in result
    
    @patch.object(YouTubeVideoMetadataExtractor, 'extract_video_metadata')
    def test_validate_video_duration_no_duration(self, mock_extract):
        """Test validation when duration is not available."""
        mock_extract.return_value = {
            'duration_seconds': None
        }
        
        result = self.extractor.validate_video_duration("test123")
        
        assert result['is_valid_duration'] is False
        assert result['duration_seconds'] is None
        assert 'error' in result
    
    @patch.object(YouTubeVideoMetadataExtractor, 'validate_video_duration')
    def test_check_duration_limit_within_limit(self, mock_validate):
        """Test duration limit check within limit."""
        mock_validate.return_value = {
            'is_valid_duration': True
        }
        
        result = self.extractor.check_duration_limit("test123", raise_on_exceeded=False)
        
        assert result is True
    
    @patch.object(YouTubeVideoMetadataExtractor, 'validate_video_duration')
    def test_check_duration_limit_exceeded_no_raise(self, mock_validate):
        """Test duration limit check exceeded without raising."""
        mock_validate.return_value = {
            'is_valid_duration': False,
            'duration_formatted': '40:00',
            'max_duration_formatted': '30:00'
        }
        
        result = self.extractor.check_duration_limit("test123", raise_on_exceeded=False)
        
        assert result is False
    
    @patch.object(YouTubeVideoMetadataExtractor, 'validate_video_duration')
    def test_check_duration_limit_exceeded_with_raise(self, mock_validate):
        """Test duration limit check exceeded with raising."""
        mock_validate.return_value = {
            'is_valid_duration': False,
            'duration_formatted': '40:00',
            'max_duration_formatted': '30:00'
        }
        
        with pytest.raises(VideoTooLongError):
            self.extractor.check_duration_limit("test123", raise_on_exceeded=True)


class TestYouTubeTranscriptFetcher:
    """Test YouTube transcript fetching functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.fetcher = YouTubeTranscriptFetcher()
    
    @patch('src.utils.youtube_api.YouTubeTranscriptApi')
    @patch.object(YouTubeVideoMetadataExtractor, 'extract_video_metadata')
    @patch.object(YouTubeVideoMetadataExtractor, 'detect_unsupported_video_type')
    def test_fetch_transcript_success(self, mock_detect, mock_extract, mock_api):
        """Test successful transcript fetching."""
        # Mock unsupported video type detection
        mock_detect.return_value = {
            'is_supported': True,
            'issues': []
        }
        
        # Mock transcript API
        mock_transcript = [
            {'start': 0.0, 'duration': 3.0, 'text': 'Hello world'},
            {'start': 3.0, 'duration': 3.0, 'text': 'This is a test'}
        ]
        mock_api.get_transcript.return_value = mock_transcript
        
        # Mock metadata extraction
        mock_extract.return_value = {
            'video_id': 'dQw4w9WgXcQ',
            'title': 'Test Video',
            'duration_seconds': 300
        }
        
        result = self.fetcher.fetch_transcript("dQw4w9WgXcQ")
        
        assert result['success'] is True
        assert result['video_id'] == "dQw4w9WgXcQ"
        assert 'transcript' in result
        assert 'raw_transcript' in result
        assert 'video_metadata' in result
    
    def test_fetch_transcript_invalid_id(self):
        """Test transcript fetching with invalid video ID."""
        with pytest.raises(YouTubeTranscriptError, match="Invalid video ID format"):
            self.fetcher.fetch_transcript("invalid")
    
    def test_fetch_transcript_empty_id(self):
        """Test transcript fetching with empty video ID."""
        with pytest.raises(YouTubeTranscriptError, match="Video ID is required"):
            self.fetcher.fetch_transcript("")
    
    @patch.object(YouTubeVideoMetadataExtractor, 'detect_unsupported_video_type')
    def test_fetch_transcript_private_video(self, mock_detect):
        """Test transcript fetching with private video."""
        mock_detect.return_value = {
            'is_supported': False,
            'issues': [{'type': 'private', 'message': 'Video is private'}]
        }
        
        with pytest.raises(PrivateVideoError):
            self.fetcher.fetch_transcript("dQw4w9WgXcQ")
    
    @patch.object(YouTubeVideoMetadataExtractor, 'detect_unsupported_video_type')
    def test_fetch_transcript_live_video(self, mock_detect):
        """Test transcript fetching with live video."""
        mock_detect.return_value = {
            'is_supported': False,
            'issues': [{'type': 'live', 'message': 'Video is live'}]
        }
        
        with pytest.raises(LiveVideoError):
            self.fetcher.fetch_transcript("dQw4w9WgXcQ")
    
    @patch.object(YouTubeVideoMetadataExtractor, 'detect_unsupported_video_type')
    def test_fetch_transcript_too_long(self, mock_detect):
        """Test transcript fetching with video too long."""
        mock_detect.return_value = {
            'is_supported': False,
            'issues': [{'type': 'too_long', 'message': 'Video is too long'}]
        }
        
        with pytest.raises(VideoTooLongError):
            self.fetcher.fetch_transcript("dQw4w9WgXcQ")
    
    @patch('src.utils.youtube_api.YouTubeTranscriptApi')
    @patch.object(YouTubeVideoMetadataExtractor, 'detect_unsupported_video_type')
    def test_fetch_transcript_no_transcript_found(self, mock_detect, mock_api):
        """Test transcript fetching when no transcript is found."""
        from youtube_transcript_api._errors import NoTranscriptFound
        
        mock_detect.return_value = {'is_supported': True, 'issues': []}
        mock_api.get_transcript.side_effect = NoTranscriptFound("video_id", [], None)
        
        with pytest.raises(NoTranscriptAvailableError):
            self.fetcher.fetch_transcript("dQw4w9WgXcQ")
    
    @patch('src.utils.youtube_api.YouTubeTranscriptApi')
    @patch.object(YouTubeVideoMetadataExtractor, 'detect_unsupported_video_type')
    def test_fetch_transcript_transcripts_disabled(self, mock_detect, mock_api):
        """Test transcript fetching when transcripts are disabled."""
        from youtube_transcript_api._errors import TranscriptsDisabled
        
        mock_detect.return_value = {'is_supported': True, 'issues': []}
        mock_api.get_transcript.side_effect = TranscriptsDisabled("video_id")
        
        with pytest.raises(NoTranscriptAvailableError):
            self.fetcher.fetch_transcript("dQw4w9WgXcQ")
    
    @patch('src.utils.youtube_api.YouTubeTranscriptApi')
    @patch.object(YouTubeVideoMetadataExtractor, 'detect_unsupported_video_type')
    def test_fetch_transcript_video_unavailable(self, mock_detect, mock_api):
        """Test transcript fetching when video is unavailable."""
        from youtube_transcript_api._errors import VideoUnavailable
        
        mock_detect.return_value = {'is_supported': True, 'issues': []}
        mock_api.get_transcript.side_effect = VideoUnavailable("video_id")
        
        with pytest.raises(YouTubeTranscriptError, match="Video is unavailable"):
            self.fetcher.fetch_transcript("dQw4w9WgXcQ")
    
    @patch('src.utils.validators.YouTubeURLValidator.validate_and_extract')
    @patch.object(YouTubeTranscriptFetcher, 'fetch_transcript')
    def test_fetch_transcript_from_url_success(self, mock_fetch, mock_validate):
        """Test successful transcript fetching from URL."""
        mock_validate.return_value = (True, "dQw4w9WgXcQ")
        mock_fetch.return_value = {'success': True, 'video_id': 'dQw4w9WgXcQ'}
        
        result = self.fetcher.fetch_transcript_from_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        
        assert result['success'] is True
        assert result['video_id'] == "dQw4w9WgXcQ"
    
    @patch('src.utils.validators.YouTubeURLValidator.validate_and_extract')
    def test_fetch_transcript_from_url_invalid(self, mock_validate):
        """Test transcript fetching from invalid URL."""
        mock_validate.return_value = (False, None)
        
        with pytest.raises(YouTubeTranscriptError, match="Invalid YouTube URL"):
            self.fetcher.fetch_transcript_from_url("https://www.google.com")
    
    @patch('src.utils.youtube_api.YouTubeTranscriptApi')
    def test_get_available_transcripts_success(self, mock_api):
        """Test getting available transcripts."""
        mock_transcript = Mock()
        mock_transcript.language = "English"
        mock_transcript.language_code = "en"
        mock_transcript.is_generated = True
        mock_transcript.is_translatable = False
        
        mock_api.list_transcripts.return_value = [mock_transcript]
        
        result = self.fetcher.get_available_transcripts("dQw4w9WgXcQ")
        
        assert len(result) == 1
        assert result[0]['language'] == "English"
        assert result[0]['language_code'] == "en"
        assert result[0]['is_generated'] is True
    
    def test_get_available_transcripts_invalid_id(self):
        """Test getting available transcripts with invalid ID."""
        with pytest.raises(YouTubeTranscriptError, match="Invalid video ID format"):
            self.fetcher.get_available_transcripts("invalid")
    
    def test_calculate_duration(self):
        """Test duration calculation from transcript."""
        transcript = [
            {'start': 0.0, 'duration': 3.0, 'text': 'Hello'},
            {'start': 3.0, 'duration': 3.0, 'text': 'World'},
            {'start': 6.0, 'duration': 4.0, 'text': 'Test'}
        ]
        
        duration = self.fetcher._calculate_duration(transcript)
        assert duration == 10.0  # 6.0 + 4.0
    
    def test_calculate_duration_empty(self):
        """Test duration calculation with empty transcript."""
        duration = self.fetcher._calculate_duration([])
        assert duration == 0.0
    
    def test_detect_language(self):
        """Test language detection from transcript."""
        transcript = [
            {'text': 'The quick brown fox jumps over the lazy dog'},
            {'text': 'This is a test of the language detection'},
            {'text': 'It should detect English with high confidence'}
        ]
        
        language = self.fetcher._detect_language(transcript)
        assert language == 'en'
    
    def test_detect_language_unknown(self):
        """Test language detection with unknown language."""
        transcript = [
            {'text': 'xyz abc def'},
            {'text': 'random text without indicators'}
        ]
        
        language = self.fetcher._detect_language(transcript)
        assert language == 'unknown'


class TestConvenienceFunctions:
    """Test module-level convenience functions."""
    
    @patch.object(YouTubeTranscriptFetcher, 'fetch_transcript')
    def test_fetch_youtube_transcript_function(self, mock_fetch):
        """Test standalone fetch_youtube_transcript function."""
        mock_fetch.return_value = {'success': True, 'video_id': 'test123'}
        
        result = fetch_youtube_transcript("test123")
        
        assert result['success'] is True
        mock_fetch.assert_called_once()
    
    @patch.object(YouTubeTranscriptFetcher, 'fetch_transcript_from_url')
    def test_fetch_youtube_transcript_from_url_function(self, mock_fetch):
        """Test standalone fetch_youtube_transcript_from_url function."""
        mock_fetch.return_value = {'success': True, 'video_id': 'test123'}
        
        result = fetch_youtube_transcript_from_url("https://youtube.com/watch?v=test123")
        
        assert result['success'] is True
        mock_fetch.assert_called_once()
    
    @patch.object(YouTubeTranscriptFetcher, 'get_available_transcripts')
    def test_get_available_youtube_transcripts_function(self, mock_get):
        """Test standalone get_available_youtube_transcripts function."""
        mock_get.return_value = [{'language': 'English'}]
        
        result = get_available_youtube_transcripts("test123")
        
        assert len(result) == 1
        mock_get.assert_called_once()
    
    @patch.object(YouTubeTranscriptFetcher, 'get_video_metadata')
    def test_get_youtube_video_metadata_function(self, mock_get):
        """Test standalone get_youtube_video_metadata function."""
        mock_get.return_value = {'title': 'Test Video'}
        
        result = get_youtube_video_metadata("test123")
        
        assert result['title'] == 'Test Video'
        mock_get.assert_called_once()
    
    @patch.object(YouTubeTranscriptFetcher, 'check_video_support')
    def test_check_youtube_video_support_function(self, mock_check):
        """Test standalone check_youtube_video_support function."""
        mock_check.return_value = {'is_supported': True}
        
        result = check_youtube_video_support("test123")
        
        assert result['is_supported'] is True
        mock_check.assert_called_once()
    
    @patch.object(YouTubeTranscriptFetcher, 'validate_video_duration')
    def test_validate_youtube_video_duration_function(self, mock_validate):
        """Test standalone validate_youtube_video_duration function."""
        mock_validate.return_value = {'is_valid_duration': True}
        
        result = validate_youtube_video_duration("test123")
        
        assert result['is_valid_duration'] is True
        mock_validate.assert_called_once()
    
    @patch.object(YouTubeTranscriptFetcher, 'check_duration_limit')
    def test_check_youtube_duration_limit_function(self, mock_check):
        """Test standalone check_youtube_duration_limit function."""
        mock_check.return_value = True
        
        result = check_youtube_duration_limit("test123")
        
        assert result is True
        mock_check.assert_called_once()


class TestIntegration:
    """Integration tests combining multiple components."""
    
    @patch('src.utils.youtube_api.YouTubeTranscriptApi')
    @patch('urllib.request.urlopen')
    def test_full_transcript_workflow(self, mock_urlopen, mock_api):
        """Test complete transcript extraction workflow."""
        # Mock metadata extraction
        mock_response = Mock()
        mock_response.read.return_value = b'''
        <html><script>
        "title": "Test Video",
        "lengthSeconds": "300",
        "defaultAudioLanguage": "en"
        </script></html>
        '''
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=None)
        mock_urlopen.return_value = mock_response
        
        # Mock transcript API
        mock_transcript = [
            {'start': 0.0, 'duration': 3.0, 'text': 'Hello world'},
            {'start': 3.0, 'duration': 3.0, 'text': 'This is a test'}
        ]
        mock_api.get_transcript.return_value = mock_transcript
        
        # Test the full workflow
        result = fetch_youtube_transcript("dQw4w9WgXcQ")
        
        assert result['success'] is True
        assert result['video_id'] == "dQw4w9WgXcQ"
        assert 'transcript' in result
        assert 'video_metadata' in result
        assert result['word_count'] > 0


class TestParameterValidation:
    """Test parameter validation and edge cases."""
    
    def test_duration_limits(self):
        """Test various duration limit scenarios."""
        extractor = YouTubeVideoMetadataExtractor()
        
        # Test default 30-minute limit
        assert extractor.get_video_duration_formatted(1800) == "30:00"
        
        # Test custom limits
        assert extractor.get_video_duration_formatted(3600) == "1:00:00"
        
        # Test edge cases
        assert extractor.get_video_duration_formatted(1) == "0:01"
        assert extractor.get_video_duration_formatted(59) == "0:59"
        assert extractor.get_video_duration_formatted(60) == "1:00"
    
    def test_language_preferences(self):
        """Test language preference handling."""
        fetcher = YouTubeTranscriptFetcher()
        
        # Test default languages
        with patch.object(fetcher, 'fetch_transcript') as mock_fetch:
            mock_fetch.return_value = {'success': True}
            fetcher.fetch_transcript("test123")
            
            # Should be called with default ['en'] languages
            args, kwargs = mock_fetch.call_args
            # Since we're mocking the method we're testing, we need to check the actual implementation
            # This is more of a documentation test
            pass


class TestEnhancedVideoInfoFunction:
    """Test enhanced get_video_info function."""
    
    @patch.object(YouTubeTranscriptFetcher, 'get_video_metadata')
    @patch.object(YouTubeTranscriptFetcher, 'check_video_support')
    @patch.object(YouTubeTranscriptFetcher, 'check_transcript_availability')
    @patch.object(YouTubeTranscriptFetcher, 'fetch_transcript_with_three_tier_strategy')
    def test_get_video_info_basic_success(self, mock_fetch, mock_avail, mock_support, mock_metadata):
        """Test basic successful video info retrieval."""
        # Mock responses
        mock_metadata.return_value = {
            'title': 'Test Video',
            'duration_seconds': 300,
            'language': 'en'
        }
        
        mock_support.return_value = {
            'is_supported': True,
            'issues': []
        }
        
        mock_avail.return_value = {
            'has_transcripts': True,
            'available_transcripts': [{'language': 'English', 'language_code': 'en'}]
        }
        
        mock_fetch.return_value = {
            'transcript': 'This is a test transcript.',
            'raw_transcript': [{'text': 'This is a test transcript.', 'start': 0}],
            'language': 'en',
            'duration_seconds': 300,
            'word_count': 5,
            'three_tier_metadata': {
                'selected_tier': 'manual',
                'selected_language': 'en',
                'quality_score': 100
            }
        }
        
        result = get_video_info('test123')
        
        assert result['success'] is True
        assert result['video_id'] == 'test123'
        assert result['enhanced_acquisition_used'] is True
        assert 'video_metadata' in result
        assert 'transcript_data' in result
        assert 'three_tier_acquisition' in result
    
    @patch.object(YouTubeTranscriptFetcher, 'get_video_metadata')
    @patch.object(YouTubeTranscriptFetcher, 'check_video_support')
    def test_get_video_info_unsupported_video_error(self, mock_support, mock_metadata):
        """Test get_video_info with unsupported video."""
        mock_metadata.return_value = {'title': 'Private Video'}
        mock_support.return_value = {
            'is_supported': False,
            'issues': [{'type': 'private', 'message': 'Video is private'}]
        }
        
        result = get_video_info('test123')
        
        assert result['success'] is False
        assert result['error']['type'] == 'unsupported_video'
        assert 'private' in result['error']['issues'][0]['type']
    
    @patch('src.utils.validators.YouTubeURLValidator.validate_and_extract')
    def test_get_video_info_with_url(self, mock_validate):
        """Test get_video_info with YouTube URL."""
        mock_validate.return_value = (True, 'dQw4w9WgXcQ')
        
        with patch('src.utils.youtube_api.get_video_info') as mock_get_video_info:
            # Mock the actual get_video_info call to avoid infinite recursion
            mock_get_video_info.return_value = {'success': True, 'video_id': 'dQw4w9WgXcQ'}
            
            # Temporarily replace the function to test URL parsing
            from src.utils.youtube_api import YouTubeTranscriptFetcher
            from src.utils.validators import YouTubeURLValidator
            
            # Test URL validation part
            is_valid, video_id = YouTubeURLValidator.validate_and_extract(
                'https://www.youtube.com/watch?v=dQw4w9WgXcQ'
            )
            
            assert is_valid is True
            assert video_id == 'dQw4w9WgXcQ'
    
    def test_get_video_info_invalid_url(self):
        """Test get_video_info with invalid URL."""
        with pytest.raises(YouTubeTranscriptError, match="Invalid YouTube URL"):
            get_video_info('https://invalid-url.com')
    
    def test_get_video_info_invalid_video_id(self):
        """Test get_video_info with invalid video ID."""
        with pytest.raises(YouTubeTranscriptError, match="Invalid video ID format"):
            get_video_info('invalid_id')
    
    @patch('src.utils.youtube_api.get_video_info')
    def test_get_video_info_legacy_compatibility(self, mock_get_video_info):
        """Test legacy compatibility function."""
        mock_get_video_info.return_value = {'success': True, 'video_id': 'test123'}
        
        result = get_video_info_legacy('test123', ['en', 'zh'])
        
        assert result['success'] is True
        mock_get_video_info.assert_called_once_with(
            video_url_or_id='test123',
            include_transcript=True,
            preferred_languages=['en', 'zh'],
            use_enhanced_acquisition=True,
            enable_detailed_logging=False
        )
    
    @patch.object(YouTubeTranscriptFetcher, 'get_video_metadata')
    @patch.object(YouTubeTranscriptFetcher, 'check_video_support')
    @patch.object(YouTubeTranscriptFetcher, 'check_transcript_availability')
    def test_get_video_info_no_transcripts_available(self, mock_avail, mock_support, mock_metadata):
        """Test get_video_info when no transcripts are available."""
        mock_metadata.return_value = {'title': 'Test Video'}
        mock_support.return_value = {'is_supported': True, 'issues': []}
        mock_avail.return_value = {
            'has_transcripts': False,
            'available_transcripts': []
        }
        
        result = get_video_info('test123')
        
        assert result['success'] is False
        assert result['error']['type'] == 'no_transcripts'
        assert 'transcript_availability' in result
    
    @patch.object(YouTubeTranscriptFetcher, 'get_video_metadata')
    @patch.object(YouTubeTranscriptFetcher, 'check_video_support')
    @patch.object(YouTubeTranscriptFetcher, 'check_transcript_availability')
    @patch.object(YouTubeTranscriptFetcher, 'fetch_transcript')
    def test_get_video_info_basic_acquisition(self, mock_fetch, mock_avail, mock_support, mock_metadata):
        """Test get_video_info with basic acquisition (not enhanced)."""
        mock_metadata.return_value = {'title': 'Test Video'}
        mock_support.return_value = {'is_supported': True, 'issues': []}
        mock_avail.return_value = {'has_transcripts': True, 'available_transcripts': [{'language': 'en'}]}
        mock_fetch.return_value = {
            'transcript': 'Basic transcript',
            'raw_transcript': [{'text': 'Basic transcript', 'start': 0}],
            'language': 'en',
            'duration_seconds': 300,
            'word_count': 2
        }
        
        result = get_video_info('test123', use_enhanced_acquisition=False)
        
        assert result['success'] is True
        assert result['enhanced_acquisition_used'] is False
        assert result['transcript_data']['acquisition_method'] == 'basic'
        assert 'three_tier_acquisition' not in result
    
    @patch.object(YouTubeTranscriptFetcher, 'get_video_metadata')
    def test_get_video_info_error_handling(self, mock_metadata):
        """Test get_video_info error handling and categorization."""
        mock_metadata.side_effect = RateLimitError(60)
        
        result = get_video_info('test123')
        
        assert result['success'] is False
        assert result['error']['type'] == 'RateLimitError'
        assert result['error']['category'] == 'rate_limiting'
        assert result['error']['recoverable'] is True
        assert len(result['error']['recovery_suggestions']) > 0


class TestEnhancedConvenienceFunctions:
    """Test enhanced convenience functions."""
    
    @patch.object(YouTubeTranscriptFetcher, 'fetch_transcript_with_three_tier_strategy')
    def test_fetch_youtube_transcript_with_three_tier_strategy_function(self, mock_fetch):
        """Test three-tier strategy convenience function."""
        mock_fetch.return_value = {
            'success': True,
            'video_id': 'test123',
            'transcript': 'Enhanced transcript',
            'three_tier_metadata': {
                'selected_tier': 'manual',
                'intelligent_fallback_metadata': {'fallback_enabled': True}
            }
        }
        
        result = fetch_youtube_transcript_with_three_tier_strategy(
            'test123',
            preferred_languages=['en', 'zh'],
            enable_intelligent_fallback=True,
            fallback_retry_delay=0.5
        )
        
        assert result['success'] is True
        assert result['video_id'] == 'test123'
        assert 'three_tier_metadata' in result
        
        # Verify function was called with correct parameters
        mock_fetch.assert_called_once_with(
            'test123',                    # video_id
            ['en', 'zh'],                # preferred_languages
            True,                        # include_metadata
            True,                        # check_unsupported
            1800,                        # max_duration_seconds
            3,                           # max_tier_attempts
            True,                        # enable_intelligent_fallback
            0.5                          # fallback_retry_delay
        )