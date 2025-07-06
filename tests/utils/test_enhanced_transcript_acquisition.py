"""
Comprehensive unit tests for enhanced transcript acquisition system.

This module contains tests for the three-tier strategy, intelligent fallback logic,
enhanced exception handling, comprehensive logging, and all new acquisition features.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, call
from datetime import datetime
import logging
import time

from src.utils.youtube_api import (
    # Core classes
    YouTubeTranscriptFetcher,
    ThreeTierTranscriptStrategy,
    TranscriptInfo,
    TranscriptTier,
    TranscriptAcquisitionLogger,
    YouTubeAPIErrorHandler,
    
    # Enhanced exceptions
    TranscriptProcessingError,
    TranscriptValidationError,
    TierStrategyError,
    AgeRestrictedVideoError,
    RegionBlockedVideoError,
    
    # Enhanced functions
    fetch_youtube_transcript_with_three_tier_strategy,
    get_video_info,
    get_video_info_legacy,
    get_youtube_transcript_tier_summary,
    
    # Original exceptions for comparison
    YouTubeTranscriptError,
    NoTranscriptAvailableError,
    PrivateVideoError,
    LiveVideoError,
    VideoTooLongError,
    RateLimitError,
    NetworkTimeoutError
)


class TestTranscriptTier:
    """Test transcript tier enumeration."""
    
    def test_tier_constants(self):
        """Test that tier constants have correct values."""
        assert TranscriptTier.MANUAL == "manual"
        assert TranscriptTier.AUTO_GENERATED == "auto"
        assert TranscriptTier.TRANSLATED == "translated"


class TestTranscriptInfo:
    """Test TranscriptInfo class."""
    
    def test_manual_transcript_info(self):
        """Test manual transcript info creation."""
        info = TranscriptInfo('en', False, False, 'test123')
        
        assert info.language_code == 'en'
        assert info.tier == TranscriptTier.MANUAL
        assert info.quality_score == 100
        assert info.is_generated is False
        assert info.is_translatable is False
        assert info.video_id == 'test123'
    
    def test_auto_generated_transcript_info(self):
        """Test auto-generated transcript info creation."""
        info = TranscriptInfo('en', True, False, 'test123')
        
        assert info.language_code == 'en'
        assert info.tier == TranscriptTier.AUTO_GENERATED
        assert info.quality_score == 50
        assert info.is_generated is True
        assert info.is_translatable is False
    
    def test_translated_transcript_info(self):
        """Test translated transcript info creation."""
        info = TranscriptInfo('es', True, True, 'test123')
        
        assert info.language_code == 'es'
        assert info.tier == TranscriptTier.TRANSLATED
        assert info.quality_score == 25
        assert info.is_generated is True
        assert info.is_translatable is True
    
    def test_transcript_info_repr(self):
        """Test string representation."""
        info = TranscriptInfo('en', False, False, 'test123')
        repr_str = repr(info)
        
        assert 'en' in repr_str
        assert 'manual' in repr_str
        assert '100' in repr_str


class TestThreeTierTranscriptStrategy:
    """Test three-tier transcript strategy."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.strategy = ThreeTierTranscriptStrategy()
        self.video_id = "test_video_123"
    
    @patch('src.utils.youtube_api.YouTubeTranscriptApi.list_transcripts')
    def test_get_transcript_strategy_manual_preferred(self, mock_list):
        """Test strategy with manual transcripts preferred."""
        # Mock available transcripts
        mock_transcripts = []
        
        # Manual English (highest priority)
        manual_en = Mock()
        manual_en.language_code = 'en'
        manual_en.is_generated = False
        manual_en.is_translatable = False
        mock_transcripts.append(manual_en)
        
        # Auto-generated English (second priority)
        auto_en = Mock()
        auto_en.language_code = 'en'
        auto_en.is_generated = True
        auto_en.is_translatable = False
        mock_transcripts.append(auto_en)
        
        # Translated Spanish (lower priority)
        trans_es = Mock()
        trans_es.language_code = 'es'
        trans_es.is_generated = True
        trans_es.is_translatable = True
        mock_transcripts.append(trans_es)
        
        mock_list.return_value = mock_transcripts
        
        strategy_order = self.strategy.get_transcript_strategy(
            self.video_id, ['en', 'es']
        )
        
        # First should be manual English
        assert strategy_order[0].language_code == 'en'
        assert strategy_order[0].tier == TranscriptTier.MANUAL
        assert strategy_order[0].quality_score == 100
        
        # Second should be auto-generated English
        assert strategy_order[1].language_code == 'en'
        assert strategy_order[1].tier == TranscriptTier.AUTO_GENERATED
        assert strategy_order[1].quality_score == 50
        
        # Third should be translated Spanish
        assert strategy_order[2].language_code == 'es'
        assert strategy_order[2].tier == TranscriptTier.TRANSLATED
        assert strategy_order[2].quality_score == 25
    
    @patch('src.utils.youtube_api.YouTubeTranscriptApi.list_transcripts')
    def test_get_transcript_strategy_language_preference(self, mock_list):
        """Test strategy respects language preferences."""
        # Mock available transcripts
        mock_transcripts = []
        
        # Manual Chinese (preferred language)
        manual_zh = Mock()
        manual_zh.language_code = 'zh-CN'
        manual_zh.is_generated = False
        manual_zh.is_translatable = False
        mock_transcripts.append(manual_zh)
        
        # Manual English (non-preferred)
        manual_en = Mock()
        manual_en.language_code = 'en'
        manual_en.is_generated = False
        manual_en.is_translatable = False
        mock_transcripts.append(manual_en)
        
        mock_list.return_value = mock_transcripts
        
        strategy_order = self.strategy.get_transcript_strategy(
            self.video_id, ['zh-CN', 'zh']
        )
        
        # Chinese should come first despite both being manual
        assert strategy_order[0].language_code == 'zh-CN'
        assert strategy_order[1].language_code == 'en'
    
    @patch('src.utils.youtube_api.YouTubeTranscriptApi.list_transcripts')
    def test_get_best_transcript_option(self, mock_list):
        """Test getting single best transcript option."""
        mock_transcript = Mock()
        mock_transcript.language_code = 'en'
        mock_transcript.is_generated = False
        mock_transcript.is_translatable = False
        
        mock_list.return_value = [mock_transcript]
        
        best_option = self.strategy.get_best_transcript_option(self.video_id)
        
        assert best_option.language_code == 'en'
        assert best_option.tier == TranscriptTier.MANUAL
    
    @patch('src.utils.youtube_api.YouTubeTranscriptApi.list_transcripts')
    def test_categorize_transcripts_by_tier(self, mock_list):
        """Test transcript categorization by tier."""
        mock_transcripts = []
        
        # Add one of each tier
        manual = Mock()
        manual.language_code = 'en'
        manual.is_generated = False
        manual.is_translatable = False
        mock_transcripts.append(manual)
        
        auto = Mock()
        auto.language_code = 'en'
        auto.is_generated = True
        auto.is_translatable = False
        mock_transcripts.append(auto)
        
        translated = Mock()
        translated.language_code = 'es'
        translated.is_generated = True
        translated.is_translatable = True
        mock_transcripts.append(translated)
        
        mock_list.return_value = mock_transcripts
        
        strategy_order = self.strategy.get_transcript_strategy(self.video_id)
        categorized = self.strategy.categorize_transcripts_by_tier(strategy_order)
        
        assert len(categorized[TranscriptTier.MANUAL]) == 1
        assert len(categorized[TranscriptTier.AUTO_GENERATED]) == 1
        assert len(categorized[TranscriptTier.TRANSLATED]) == 1
        
        assert categorized[TranscriptTier.MANUAL][0].language_code == 'en'
        assert categorized[TranscriptTier.AUTO_GENERATED][0].language_code == 'en'
        assert categorized[TranscriptTier.TRANSLATED][0].language_code == 'es'
    
    @patch('src.utils.youtube_api.YouTubeTranscriptApi.list_transcripts')
    def test_get_transcript_tier_summary(self, mock_list):
        """Test transcript tier summary generation."""
        mock_transcripts = []
        
        # Add transcripts for testing
        for lang, is_gen, is_trans in [('en', False, False), ('en', True, False), ('es', True, True)]:
            mock_transcript = Mock()
            mock_transcript.language_code = lang
            mock_transcript.is_generated = is_gen
            mock_transcript.is_translatable = is_trans
            mock_transcripts.append(mock_transcript)
        
        mock_list.return_value = mock_transcripts
        
        summary = self.strategy.get_transcript_tier_summary(self.video_id, ['en'])
        
        assert summary['video_id'] == self.video_id
        assert summary['total_transcripts'] == 3
        assert summary['tiers']['manual']['count'] == 1
        assert summary['tiers']['auto_generated']['count'] == 1
        assert summary['tiers']['translated']['count'] == 1
        assert summary['best_option']['language'] == 'en'
        assert summary['best_option']['tier'] == TranscriptTier.MANUAL
    
    @patch('src.utils.youtube_api.YouTubeTranscriptApi.list_transcripts')
    def test_no_transcripts_available_error(self, mock_list):
        """Test error when no transcripts are available."""
        from youtube_transcript_api._errors import NoTranscriptFound
        
        mock_list.side_effect = NoTranscriptFound("test_video", [], None)
        
        with pytest.raises(NoTranscriptAvailableError):
            self.strategy.get_transcript_strategy(self.video_id)


class TestEnhancedExceptions:
    """Test enhanced exception classes."""
    
    def test_age_restricted_video_error(self):
        """Test age restricted video error."""
        error = AgeRestrictedVideoError("test123")
        
        assert "test123" in str(error)
        assert "age restricted" in str(error).lower()
        assert error.error_code == "AGE_RESTRICTED"
        assert isinstance(error, YouTubeTranscriptError)
    
    def test_region_blocked_video_error(self):
        """Test region blocked video error."""
        error = RegionBlockedVideoError("test123")
        
        assert "test123" in str(error)
        assert "region blocked" in str(error).lower()
        assert error.error_code == "REGION_BLOCKED"
        assert isinstance(error, YouTubeTranscriptError)
    
    def test_transcript_processing_error(self):
        """Test transcript processing error."""
        error = TranscriptProcessingError("Processing failed", "test123", "formatting")
        
        assert error.message == "Processing failed"
        assert error.video_id == "test123"
        assert error.processing_stage == "formatting"
        assert error.error_code == "TRANSCRIPT_PROCESSING_ERROR"
    
    def test_transcript_validation_error(self):
        """Test transcript validation error."""
        error = TranscriptValidationError("Validation failed", "test123", "empty_data")
        
        assert error.message == "Validation failed"
        assert error.video_id == "test123"
        assert error.validation_type == "empty_data"
        assert error.error_code == "TRANSCRIPT_VALIDATION_ERROR"
    
    def test_tier_strategy_error(self):
        """Test tier strategy error."""
        error = TierStrategyError("Strategy failed", "test123", "intelligent_fallback")
        
        assert error.message == "Strategy failed"
        assert error.video_id == "test123"
        assert error.tier_attempted == "intelligent_fallback"
        assert error.error_code == "TIER_STRATEGY_ERROR"


class TestTranscriptAcquisitionLogger:
    """Test comprehensive logging functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_logger = Mock()
        self.acquisition_logger = TranscriptAcquisitionLogger(self.mock_logger)
    
    def test_log_acquisition_start(self):
        """Test logging acquisition start."""
        self.acquisition_logger.log_acquisition_start(
            "test123", "three_tier", ["en", "zh-CN"]
        )
        
        self.mock_logger.info.assert_called_once()
        call_args = self.mock_logger.info.call_args[0][0]
        assert "test123" in call_args
        assert "three_tier" in call_args
        assert "en" in call_args
        
        # Check session stats updated
        assert "test123" in self.acquisition_logger.session_stats['videos_processed']
    
    def test_log_video_metadata(self):
        """Test logging video metadata."""
        metadata = {
            'title': 'Test Video',
            'duration_seconds': 300,
            'language': 'en'
        }
        
        self.acquisition_logger.log_video_metadata("test123", metadata)
        
        self.mock_logger.debug.assert_called_once()
        call_args = self.mock_logger.debug.call_args[0][0]
        assert "Test Video" in call_args
        assert "300" in call_args
    
    def test_log_video_metadata_none(self):
        """Test logging when metadata is None."""
        self.acquisition_logger.log_video_metadata("test123", None)
        
        self.mock_logger.warning.assert_called_once()
        call_args = self.mock_logger.warning.call_args[0][0]
        assert "test123" in call_args
        assert "No metadata" in call_args
    
    def test_log_tier_attempt_start(self):
        """Test logging tier attempt start."""
        self.acquisition_logger.log_tier_attempt_start(
            "test123", "manual", "en", 1, 1
        )
        
        self.mock_logger.debug.assert_called_once()
        call_args = self.mock_logger.debug.call_args[0][0]
        assert "test123" in call_args
        assert "manual" in call_args
        assert "en" in call_args
        
        # Check session stats updated
        assert self.acquisition_logger.session_stats['total_attempts'] == 1
        assert self.acquisition_logger.session_stats['tiers_used']['manual'] == 1
    
    def test_log_tier_attempt_success(self):
        """Test logging successful tier attempt."""
        self.acquisition_logger.log_tier_attempt_success(
            "test123", "manual", "en", 1, 150, 300.5
        )
        
        self.mock_logger.info.assert_called_once()
        call_args = self.mock_logger.info.call_args[0][0]
        assert "SUCCESS" in call_args
        assert "test123" in call_args
        assert "manual" in call_args
        assert "150 words" in call_args
        
        # Check session stats updated
        assert self.acquisition_logger.session_stats['successful_attempts'] == 1
    
    def test_log_tier_attempt_failure(self):
        """Test logging failed tier attempt."""
        error = ValueError("Test error")
        
        self.acquisition_logger.log_tier_attempt_failure(
            "test123", "manual", "en", 1, error
        )
        
        self.mock_logger.warning.assert_called_once()
        call_args = self.mock_logger.warning.call_args[0][0]
        assert "FAILED" in call_args
        assert "test123" in call_args
        assert "ValueError" in call_args
        
        # Check session stats updated
        assert self.acquisition_logger.session_stats['failed_attempts'] == 1
        assert self.acquisition_logger.session_stats['error_counts']['ValueError'] == 1
    
    def test_log_tier_fallback(self):
        """Test logging tier fallback."""
        self.acquisition_logger.log_tier_fallback(
            "test123", "manual", "auto", "All manual attempts failed"
        )
        
        self.mock_logger.info.assert_called_once()
        call_args = self.mock_logger.info.call_args[0][0]
        assert "FALLBACK" in call_args
        assert "manual" in call_args
        assert "auto" in call_args
        assert "All manual attempts failed" in call_args
    
    def test_log_session_summary(self):
        """Test session summary logging."""
        # Add some test data
        self.acquisition_logger.session_stats['total_attempts'] = 5
        self.acquisition_logger.session_stats['successful_attempts'] = 3
        self.acquisition_logger.session_stats['videos_processed'] = {"test1", "test2"}
        
        summary = self.acquisition_logger.log_session_summary()
        
        assert summary['videos_processed'] == 2
        assert summary['total_attempts'] == 5
        assert summary['successful_attempts'] == 3
        assert summary['success_rate'] == 60.0
        
        self.mock_logger.info.assert_called()
    
    def test_reset_session_stats(self):
        """Test resetting session statistics."""
        # Add some data
        self.acquisition_logger.session_stats['total_attempts'] = 5
        self.acquisition_logger.session_stats['videos_processed'].add("test123")
        
        self.acquisition_logger.reset_session_stats()
        
        assert self.acquisition_logger.session_stats['total_attempts'] == 0
        assert len(self.acquisition_logger.session_stats['videos_processed']) == 0
        
        self.mock_logger.info.assert_called_with("Session statistics reset")


class TestYouTubeAPIErrorHandler:
    """Test enhanced error handling and categorization."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.error_handler = YouTubeAPIErrorHandler()
    
    def test_categorize_no_transcript_error(self):
        """Test categorization of no transcript error."""
        error = NoTranscriptAvailableError("test123", ["en"])
        
        category = self.error_handler.categorize_error(error, "test123")
        
        assert category['error_type'] == 'NoTranscriptAvailableError'
        assert category['category'] == 'transcript_unavailable'
        assert category['severity'] == 'high'
        assert category['recoverable'] is False
        assert len(category['recovery_suggestions']) > 0
    
    def test_categorize_rate_limit_error(self):
        """Test categorization of rate limit error."""
        error = RateLimitError(120)
        
        category = self.error_handler.categorize_error(error)
        
        assert category['error_type'] == 'RateLimitError'
        assert category['category'] == 'rate_limiting'
        assert category['severity'] == 'medium'
        assert category['recoverable'] is True
        assert category['retry_recommended'] is True
        assert category['retry_delay'] == 120
    
    def test_categorize_network_timeout_error(self):
        """Test categorization of network timeout error."""
        error = NetworkTimeoutError("API request", 30)
        
        category = self.error_handler.categorize_error(error)
        
        assert category['error_type'] == 'NetworkTimeoutError'
        assert category['category'] == 'network_issue'
        assert category['severity'] == 'low'
        assert category['recoverable'] is True
        assert category['retry_delay'] == 5
    
    def test_categorize_processing_error(self):
        """Test categorization of processing error."""
        error = TranscriptProcessingError("Format failed", "test123", "formatting")
        
        category = self.error_handler.categorize_error(error, "test123")
        
        assert category['error_type'] == 'TranscriptProcessingError'
        assert category['category'] == 'processing_error'
        assert category['severity'] == 'medium'
        assert category['recoverable'] is True
        assert category['retry_recommended'] is True
    
    def test_should_retry_error(self):
        """Test error retry recommendation."""
        # Recoverable error
        recoverable_error = NetworkTimeoutError()
        should_retry, delay = self.error_handler.should_retry_error(recoverable_error)
        assert should_retry is True
        assert delay == 5
        
        # Non-recoverable error
        non_recoverable_error = NoTranscriptAvailableError("test123")
        should_retry, delay = self.error_handler.should_retry_error(non_recoverable_error)
        assert should_retry is False
        assert delay == 0
    
    def test_generate_error_report(self):
        """Test comprehensive error report generation."""
        errors = [
            RateLimitError(60),
            NetworkTimeoutError(),
            NoTranscriptAvailableError("test123")
        ]
        
        report = self.error_handler.generate_error_report(errors, "test123")
        
        assert report['video_id'] == "test123"
        assert report['total_errors'] == 3
        assert 'rate_limiting' in report['error_categories']
        assert 'network_issue' in report['error_categories']
        assert 'transcript_unavailable' in report['error_categories']
        
        assert report['recovery_analysis']['recoverable_errors'] == 2
        assert report['recovery_analysis']['unrecoverable_errors'] == 1
        assert report['recovery_analysis']['retry_recommended'] is True
        assert report['recovery_analysis']['suggested_retry_delay'] == 60  # Max of delays
    
    def test_error_statistics_tracking(self):
        """Test error statistics tracking."""
        initial_stats = self.error_handler.get_error_statistics()
        assert initial_stats['total_errors'] == 0
        
        # Categorize some errors
        self.error_handler.categorize_error(RateLimitError(), "test1")
        self.error_handler.categorize_error(NetworkTimeoutError(), "test1")
        self.error_handler.categorize_error(RateLimitError(), "test2")
        
        stats = self.error_handler.get_error_statistics()
        assert stats['total_errors'] == 3
        assert stats['error_types']['RateLimitError'] == 2
        assert stats['error_types']['NetworkTimeoutError'] == 1
        assert stats['video_errors']['test1'] == 2
        assert stats['video_errors']['test2'] == 1


class TestEnhancedTranscriptFetcher:
    """Test enhanced transcript fetcher with three-tier strategy."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.fetcher = YouTubeTranscriptFetcher(enable_detailed_logging=False)
    
    @patch('src.utils.youtube_api.YouTubeTranscriptApi')
    @patch.object(YouTubeTranscriptFetcher, '_execute_intelligent_fallback_strategy')
    def test_fetch_transcript_with_three_tier_strategy_intelligent(self, mock_fallback, mock_api):
        """Test three-tier strategy with intelligent fallback enabled."""
        # Mock successful result
        mock_result = {
            'video_id': 'test123',
            'transcript': 'Test transcript',
            'language': 'en',
            'three_tier_metadata': {
                'selected_tier': 'manual',
                'total_attempts': 1
            }
        }
        mock_fallback.return_value = mock_result
        
        result = self.fetcher.fetch_transcript_with_three_tier_strategy(
            'test123', enable_intelligent_fallback=True
        )
        
        assert result['video_id'] == 'test123'
        assert result['transcript'] == 'Test transcript'
        mock_fallback.assert_called_once()
    
    @patch('src.utils.youtube_api.YouTubeTranscriptApi')
    @patch.object(YouTubeTranscriptFetcher, '_execute_basic_fallback_strategy')
    def test_fetch_transcript_with_three_tier_strategy_basic(self, mock_fallback, mock_api):
        """Test three-tier strategy with basic fallback."""
        # Mock successful result
        mock_result = {
            'video_id': 'test123',
            'transcript': 'Test transcript',
            'language': 'en'
        }
        mock_fallback.return_value = mock_result
        
        result = self.fetcher.fetch_transcript_with_three_tier_strategy(
            'test123', enable_intelligent_fallback=False
        )
        
        assert result['video_id'] == 'test123'
        mock_fallback.assert_called_once()
    
    @patch.object(ThreeTierTranscriptStrategy, 'get_transcript_strategy')
    def test_intelligent_fallback_strategy_execution(self, mock_strategy):
        """Test intelligent fallback strategy execution."""
        # Mock transcript options
        mock_transcript = Mock()
        mock_transcript.fetch.return_value = [
            {'start': 0, 'duration': 3, 'text': 'Test transcript'}
        ]
        
        manual_info = TranscriptInfo('en', False, False, 'test123', mock_transcript)
        auto_info = TranscriptInfo('en', True, False, 'test123', mock_transcript)
        
        mock_strategy.return_value = [manual_info, auto_info]
        
        # Mock formatter
        with patch.object(self.fetcher.formatter, 'format_transcript') as mock_format:
            mock_format.return_value = "Test transcript"
            
            result = self.fetcher._execute_intelligent_fallback_strategy(
                'test123', [manual_info, auto_info], None, ['en'], 3, 1.0
            )
        
        assert result['video_id'] == 'test123'
        assert result['transcript'] == 'Test transcript'
        assert result['three_tier_metadata']['selected_tier'] == 'manual'
        assert result['three_tier_metadata']['intelligent_fallback_metadata']['fallback_enabled'] is True
    
    @patch.object(ThreeTierTranscriptStrategy, 'get_transcript_strategy')
    def test_intelligent_fallback_with_failures(self, mock_strategy):
        """Test intelligent fallback when first tier fails."""
        # Mock transcripts that will fail and succeed
        failing_transcript = Mock()
        failing_transcript.fetch.side_effect = Exception("Failed to fetch")
        
        succeeding_transcript = Mock()
        succeeding_transcript.fetch.return_value = [
            {'start': 0, 'duration': 3, 'text': 'Working transcript'}
        ]
        
        manual_info = TranscriptInfo('en', False, False, 'test123', failing_transcript)
        auto_info = TranscriptInfo('en', True, False, 'test123', succeeding_transcript)
        
        mock_strategy.return_value = [manual_info, auto_info]
        
        # Mock formatter
        with patch.object(self.fetcher.formatter, 'format_transcript') as mock_format:
            mock_format.return_value = "Working transcript"
            
            result = self.fetcher._execute_intelligent_fallback_strategy(
                'test123', [manual_info, auto_info], None, ['en'], 3, 0.1
            )
        
        assert result['video_id'] == 'test123'
        assert result['transcript'] == 'Working transcript'
        assert result['three_tier_metadata']['selected_tier'] == 'auto'
        assert result['three_tier_metadata']['total_attempts'] == 2
        
        # Check that one attempt failed and one succeeded
        attempts_made = result['three_tier_metadata']['attempts_made']
        assert len(attempts_made) == 2
        assert attempts_made[0]['success'] is False
        assert attempts_made[1]['success'] is True
    
    def test_group_transcripts_by_tier(self):
        """Test grouping transcripts by tier."""
        transcripts = [
            TranscriptInfo('en', False, False, 'test123'),  # Manual
            TranscriptInfo('en', True, False, 'test123'),   # Auto-generated
            TranscriptInfo('es', True, True, 'test123'),    # Translated
            TranscriptInfo('zh', False, False, 'test123'),  # Manual
        ]
        
        grouped = self.fetcher._group_transcripts_by_tier(transcripts)
        
        assert len(grouped[TranscriptTier.MANUAL]) == 2
        assert len(grouped[TranscriptTier.AUTO_GENERATED]) == 1
        assert len(grouped[TranscriptTier.TRANSLATED]) == 1
        
        # Check languages in each tier
        manual_languages = [t.language_code for t in grouped[TranscriptTier.MANUAL]]
        assert 'en' in manual_languages
        assert 'zh' in manual_languages


class TestGetVideoInfo:
    """Test enhanced get_video_info function."""
    
    @patch.object(YouTubeTranscriptFetcher, 'get_video_metadata')
    @patch.object(YouTubeTranscriptFetcher, 'check_video_support')
    @patch.object(YouTubeTranscriptFetcher, 'check_transcript_availability')
    @patch.object(YouTubeTranscriptFetcher, 'fetch_transcript_with_three_tier_strategy')
    def test_get_video_info_success_enhanced(self, mock_fetch, mock_avail, mock_support, mock_metadata):
        """Test successful video info retrieval with enhanced acquisition."""
        # Mock responses
        mock_metadata.return_value = {
            'title': 'Test Video',
            'duration_seconds': 300
        }
        
        mock_support.return_value = {
            'is_supported': True,
            'issues': []
        }
        
        mock_avail.return_value = {
            'has_transcripts': True,
            'available_transcripts': [{'language': 'en'}]
        }
        
        mock_fetch.return_value = {
            'transcript': 'Test transcript content',
            'raw_transcript': [{'text': 'Test', 'start': 0}],
            'language': 'en',
            'duration_seconds': 300,
            'word_count': 150,
            'three_tier_metadata': {
                'selected_tier': 'manual',
                'quality_score': 100,
                'total_attempts': 1
            }
        }
        
        result = get_video_info(
            'test123',
            include_transcript=True,
            use_enhanced_acquisition=True
        )
        
        assert result['success'] is True
        assert result['video_id'] == 'test123'
        assert result['enhanced_acquisition_used'] is True
        assert 'video_metadata' in result
        assert 'transcript_data' in result
        assert 'three_tier_acquisition' in result
        
        # Check transcript data
        transcript_data = result['transcript_data']
        assert transcript_data['transcript_text'] == 'Test transcript content'
        assert transcript_data['language'] == 'en'
        assert transcript_data['acquisition_method'] == 'enhanced_three_tier'
        
        # Check three-tier metadata
        three_tier = result['three_tier_acquisition']
        assert three_tier['selected_tier'] == 'manual'
        assert three_tier['quality_score'] == 100
        assert three_tier['total_attempts'] == 1
    
    @patch.object(YouTubeTranscriptFetcher, 'get_video_metadata')
    @patch.object(YouTubeTranscriptFetcher, 'check_video_support')
    def test_get_video_info_unsupported_video(self, mock_support, mock_metadata):
        """Test video info with unsupported video."""
        mock_metadata.return_value = {'title': 'Private Video'}
        
        mock_support.return_value = {
            'is_supported': False,
            'issues': [{'type': 'private', 'message': 'Video is private'}]
        }
        
        result = get_video_info('test123')
        
        assert result['success'] is False
        assert result['error']['type'] == 'unsupported_video'
        assert result['error']['message'] == 'Video type is not supported'
        assert len(result['error']['issues']) == 1
    
    @patch.object(YouTubeTranscriptFetcher, 'get_video_metadata')
    @patch.object(YouTubeTranscriptFetcher, 'check_video_support')
    @patch.object(YouTubeTranscriptFetcher, 'check_transcript_availability')
    def test_get_video_info_no_transcripts(self, mock_avail, mock_support, mock_metadata):
        """Test video info when no transcripts are available."""
        mock_metadata.return_value = {'title': 'Test Video'}
        mock_support.return_value = {'is_supported': True, 'issues': []}
        mock_avail.return_value = {
            'has_transcripts': False,
            'available_transcripts': []
        }
        
        result = get_video_info('test123')
        
        assert result['success'] is False
        assert result['error']['type'] == 'no_transcripts'
        assert result['error']['message'] == 'No transcripts available for this video'
    
    def test_get_video_info_invalid_url(self):
        """Test video info with invalid URL."""
        with pytest.raises(YouTubeTranscriptError, match="Invalid YouTube URL"):
            get_video_info('https://invalid-url.com')
    
    def test_get_video_info_legacy_compatibility(self):
        """Test legacy compatibility function."""
        with patch('src.utils.youtube_api.get_video_info') as mock_get:
            mock_get.return_value = {'success': True, 'video_id': 'test123'}
            
            result = get_video_info_legacy('test123', ['en', 'zh'])
            
            assert result['success'] is True
            mock_get.assert_called_once_with(
                video_url_or_id='test123',
                include_transcript=True,
                preferred_languages=['en', 'zh'],
                use_enhanced_acquisition=True,
                enable_detailed_logging=False
            )


class TestConvenienceFunctions:
    """Test enhanced convenience functions."""
    
    @patch.object(YouTubeTranscriptFetcher, 'fetch_transcript_with_three_tier_strategy')
    def test_fetch_youtube_transcript_with_three_tier_strategy(self, mock_fetch):
        """Test convenience function for three-tier strategy."""
        mock_fetch.return_value = {
            'success': True,
            'video_id': 'test123',
            'transcript': 'Test content'
        }
        
        result = fetch_youtube_transcript_with_three_tier_strategy(
            'test123',
            preferred_languages=['en'],
            enable_intelligent_fallback=True,
            fallback_retry_delay=0.5
        )
        
        assert result['success'] is True
        assert result['video_id'] == 'test123'
        
        mock_fetch.assert_called_once_with(
            'test123',
            ['en'],
            True,  # include_metadata
            True,  # check_unsupported
            1800,  # max_duration_seconds
            3,     # max_tier_attempts
            True,  # enable_intelligent_fallback
            0.5    # fallback_retry_delay
        )
    
    @patch.object(ThreeTierTranscriptStrategy, 'get_transcript_tier_summary')
    def test_get_youtube_transcript_tier_summary(self, mock_summary):
        """Test convenience function for tier summary."""
        mock_summary.return_value = {
            'video_id': 'test123',
            'total_transcripts': 3,
            'tiers': {'manual': {'count': 1}}
        }
        
        result = get_youtube_transcript_tier_summary('test123', ['en'])
        
        assert result['video_id'] == 'test123'
        assert result['total_transcripts'] == 3
        
        mock_summary.assert_called_once_with('test123', ['en'], None)


class TestIntegrationScenarios:
    """Test complete integration scenarios."""
    
    @patch('src.utils.youtube_api.YouTubeTranscriptApi')
    @patch('urllib.request.urlopen')
    def test_complete_enhanced_workflow_success(self, mock_urlopen, mock_api):
        """Test complete enhanced workflow from URL to result."""
        # Mock metadata extraction
        mock_response = Mock()
        mock_response.read.return_value = b'''
        <script>
        "title": "Test Video Title",
        "lengthSeconds": "300",
        "defaultAudioLanguage": "en"
        </script>
        '''
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=None)
        mock_urlopen.return_value = mock_response
        
        # Mock transcript API for strategy building
        mock_transcript = Mock()
        mock_transcript.language_code = 'en'
        mock_transcript.is_generated = False
        mock_transcript.is_translatable = False
        mock_transcript.fetch.return_value = [
            {'start': 0.0, 'duration': 3.0, 'text': 'Hello world'},
            {'start': 3.0, 'duration': 3.0, 'text': 'This is a test'}
        ]
        
        mock_api.list_transcripts.return_value = [mock_transcript]
        
        # Test complete workflow
        result = get_video_info(
            'https://www.youtube.com/watch?v=dQw4w9WgXcQ',
            include_transcript=True,
            use_enhanced_acquisition=True
        )
        
        assert result['success'] is True
        assert result['video_id'] == 'dQw4w9WgXcQ'
        assert result['enhanced_acquisition_used'] is True
        assert 'three_tier_acquisition' in result
        assert result['three_tier_acquisition']['selected_tier'] == 'manual'
    
    @patch('src.utils.youtube_api.YouTubeTranscriptApi')
    def test_enhanced_workflow_with_fallback(self, mock_api):
        """Test enhanced workflow with fallback scenarios."""
        # Mock failing manual transcript, succeeding auto-generated
        failing_transcript = Mock()
        failing_transcript.language_code = 'en'
        failing_transcript.is_generated = False
        failing_transcript.is_translatable = False
        failing_transcript.fetch.side_effect = Exception("Manual transcript failed")
        
        succeeding_transcript = Mock()
        succeeding_transcript.language_code = 'en'
        succeeding_transcript.is_generated = True
        succeeding_transcript.is_translatable = False
        succeeding_transcript.fetch.return_value = [
            {'start': 0.0, 'duration': 3.0, 'text': 'Auto-generated content'}
        ]
        
        mock_api.list_transcripts.return_value = [failing_transcript, succeeding_transcript]
        
        fetcher = YouTubeTranscriptFetcher(enable_detailed_logging=False)
        
        with patch.object(fetcher.metadata_extractor, 'extract_video_metadata') as mock_metadata:
            mock_metadata.return_value = {'title': 'Test Video', 'duration_seconds': 300}
            
            result = fetcher.fetch_transcript_with_three_tier_strategy(
                'test123',
                enable_intelligent_fallback=True,
                fallback_retry_delay=0.1
            )
        
        assert result['success'] is True
        assert result['three_tier_metadata']['selected_tier'] == 'auto'
        assert result['three_tier_metadata']['total_attempts'] == 2
        
        # Verify fallback was used
        fallback_metadata = result['three_tier_metadata']['intelligent_fallback_metadata']
        assert fallback_metadata['fallback_enabled'] is True
        assert 'manual' in fallback_metadata['tiers_tried']
        
        # Verify attempts were recorded
        attempts = result['three_tier_metadata']['attempts_made']
        assert len(attempts) == 2
        assert attempts[0]['success'] is False
        assert attempts[0]['tier'] == 'manual'
        assert attempts[1]['success'] is True
        assert attempts[1]['tier'] == 'auto'


class TestPerformanceAndEdgeCases:
    """Test performance characteristics and edge cases."""
    
    def test_empty_transcript_handling(self):
        """Test handling of empty transcript data."""
        fetcher = YouTubeTranscriptFetcher(enable_detailed_logging=False)
        
        with pytest.raises(TranscriptValidationError, match="Empty transcript data"):
            fetcher._execute_intelligent_fallback_strategy(
                'test123', [], None, ['en'], 3, 0.1
            )
    
    def test_invalid_transcript_format(self):
        """Test handling of invalid transcript format."""
        mock_transcript = Mock()
        mock_transcript.fetch.return_value = []  # Empty list
        
        transcript_info = TranscriptInfo('en', False, False, 'test123', mock_transcript)
        
        fetcher = YouTubeTranscriptFetcher(enable_detailed_logging=False)
        
        with pytest.raises(TranscriptValidationError):
            fetcher._execute_intelligent_fallback_strategy(
                'test123', [transcript_info], None, ['en'], 3, 0.1
            )
    
    def test_large_number_of_transcripts(self):
        """Test handling of large number of transcript options."""
        # Create many transcript options
        transcript_infos = []
        for i in range(20):
            info = TranscriptInfo(f'lang_{i}', i % 2 == 0, i % 3 == 0, 'test123')
            transcript_infos.append(info)
        
        fetcher = YouTubeTranscriptFetcher(enable_detailed_logging=False)
        grouped = fetcher._group_transcripts_by_tier(transcript_infos)
        
        # Should handle large numbers without issues
        total_count = sum(len(transcripts) for transcripts in grouped.values())
        assert total_count == 20
    
    @patch('time.sleep')
    def test_retry_delay_timing(self, mock_sleep):
        """Test that retry delays are properly applied."""
        mock_transcript = Mock()
        mock_transcript.fetch.side_effect = [
            Exception("First failure"),
            Exception("Second failure"),
            [{'start': 0, 'duration': 3, 'text': 'Success on third try'}]
        ]
        
        manual_info = TranscriptInfo('en', False, False, 'test123', mock_transcript)
        
        fetcher = YouTubeTranscriptFetcher(enable_detailed_logging=False)
        
        with patch.object(fetcher.formatter, 'format_transcript') as mock_format:
            mock_format.return_value = "Success on third try"
            
            result = fetcher._execute_intelligent_fallback_strategy(
                'test123', [manual_info], None, ['en'], 3, 1.5
            )
        
        # Should have called sleep twice (between attempts 1-2 and 2-3)
        assert mock_sleep.call_count == 2
        # First call should be with 1.5 seconds delay
        mock_sleep.assert_any_call(1.5)
        
        assert result['success'] is True
        assert result['three_tier_metadata']['total_attempts'] == 3
    
    def test_logging_integration_performance(self):
        """Test that logging doesn't significantly impact performance."""
        # This is a basic performance test - in practice you'd use profiling tools
        import time
        
        # Test with logging enabled
        start_time = time.time()
        fetcher_with_logging = YouTubeTranscriptFetcher(enable_detailed_logging=True)
        for i in range(100):
            logger = fetcher_with_logging.acquisition_logger
            logger.log_tier_attempt_start('test', 'manual', 'en', i, 1)
        logged_time = time.time() - start_time
        
        # Test without logging
        start_time = time.time()
        fetcher_without_logging = YouTubeTranscriptFetcher(enable_detailed_logging=False)
        for i in range(100):
            # Simulate equivalent work
            pass
        unlogged_time = time.time() - start_time
        
        # Logging should not add excessive overhead (this is a rough test)
        assert logged_time < unlogged_time * 10  # Less than 10x slower


if __name__ == '__main__':
    pytest.main([__file__, '-v'])