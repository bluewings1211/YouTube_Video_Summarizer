"""
Unit tests for PocketFlow Node implementations.

This module provides comprehensive test coverage for all node implementations
including YouTubeTranscriptNode, SummarizationNode, TimestampNode, and KeywordExtractionNode.
"""

import pytest
import unittest.mock as mock
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime
import os
import sys

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from nodes import (
    BaseProcessingNode,
    YouTubeTranscriptNode,
    SummarizationNode,
    TimestampNode,
    KeywordExtractionNode,
    NodeError,
    Store
)


class TestBaseProcessingNode:
    """Test cases for BaseProcessingNode functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.base_node = BaseProcessingNode("TestNode")
    
    def test_init(self):
        """Test BaseProcessingNode initialization."""
        node = BaseProcessingNode("TestNode", max_retries=5, retry_delay=2.0)
        assert node.name == "TestNode"
        assert node.max_retries == 5
        assert node.retry_delay == 2.0
    
    def test_handle_error(self):
        """Test error handling and NodeError creation."""
        error = ValueError("Test error")
        error_info = self.base_node._handle_error(error, "Test context")
        
        assert isinstance(error_info, NodeError)
        assert error_info.node_name == "TestNode"
        assert error_info.error_type == "ValueError"
        assert error_info.message == "Test error"
        assert error_info.retry_count == 0
        assert error_info.is_recoverable is True
    
    def test_validate_store_data(self):
        """Test store data validation."""
        store = Store()
        store['key1'] = 'value1'
        store['key2'] = 'value2'
        
        # Test valid data
        is_valid, missing = self.base_node._validate_store_data(store, ['key1', 'key2'])
        assert is_valid is True
        assert missing == []
        
        # Test missing data
        is_valid, missing = self.base_node._validate_store_data(store, ['key1', 'key3'])
        assert is_valid is False
        assert missing == ['key3']
    
    def test_safe_store_update(self):
        """Test safe store updates."""
        store = Store()
        data = {'new_key': 'new_value'}
        
        self.base_node._safe_store_update(store, data)
        assert store['new_key'] == 'new_value'
    
    @patch('time.sleep')
    def test_retry_with_delay(self, mock_sleep):
        """Test retry delay mechanism."""
        self.base_node._retry_with_delay(1)
        mock_sleep.assert_called_once_with(1.0)
        
        mock_sleep.reset_mock()
        self.base_node._retry_with_delay(2)
        mock_sleep.assert_called_once_with(2.0)


class TestYouTubeTranscriptNode:
    """Test cases for YouTubeTranscriptNode."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.node = YouTubeTranscriptNode()
        self.store = Store()
        self.store['youtube_url'] = 'https://www.youtube.com/watch?v=dQw4w9WgXcQ'
    
    @patch('nodes.YouTubeTranscriptFetcher')
    @patch('nodes.YouTubeURLValidator')
    def test_prep_success(self, mock_validator, mock_fetcher):
        """Test successful prep phase."""
        # Mock URL validation
        mock_validator.return_value.validate_and_extract.return_value = (True, 'dQw4w9WgXcQ')
        
        # Mock video support check
        mock_fetcher_instance = mock_fetcher.return_value
        mock_fetcher_instance.check_video_support.return_value = {
            'is_supported': True,
            'issues': []
        }
        mock_fetcher_instance.check_transcript_availability.return_value = {
            'has_transcripts': True,
            'available_transcripts': [{'language': 'en'}]
        }
        
        result = self.node.prep(self.store)
        
        assert result['prep_status'] == 'success'
        assert result['video_id'] == 'dQw4w9WgXcQ'
        assert result['is_supported'] is True
        assert result['has_transcripts'] is True
    
    def test_prep_missing_url(self):
        """Test prep phase with missing URL."""
        empty_store = Store()
        result = self.node.prep(empty_store)
        
        assert result['prep_status'] == 'failed'
        assert 'error' in result
    
    @patch('nodes.YouTubeTranscriptFetcher')
    def test_exec_success(self, mock_fetcher):
        """Test successful exec phase."""
        prep_result = {
            'prep_status': 'success',
            'video_id': 'dQw4w9WgXcQ'
        }
        
        # Mock transcript fetching
        mock_fetcher_instance = mock_fetcher.return_value
        mock_fetcher_instance.fetch_transcript.return_value = {
            'transcript': 'Test transcript content',
            'raw_transcript': [{'start': 0, 'text': 'Test'}],
            'language': 'en',
            'duration_seconds': 180,
            'word_count': 50,
            'video_metadata': {'title': 'Test Video'}
        }
        
        result = self.node.exec(self.store, prep_result)
        
        assert result['exec_status'] == 'success'
        assert result['video_id'] == 'dQw4w9WgXcQ'
        assert result['transcript_text'] == 'Test transcript content'
    
    def test_exec_prep_failed(self):
        """Test exec phase when prep failed."""
        prep_result = {'prep_status': 'failed'}
        result = self.node.exec(self.store, prep_result)
        
        assert result['exec_status'] == 'failed'
        assert result['error'] == 'Prep phase failed'
    
    def test_post_success(self):
        """Test successful post phase."""
        prep_result = {'prep_timestamp': datetime.utcnow().isoformat()}
        exec_result = {
            'exec_status': 'success',
            'video_id': 'dQw4w9WgXcQ',
            'transcript_text': 'Test transcript',
            'raw_transcript': [{'start': 0, 'text': 'Test'}],
            'transcript_language': 'en',
            'transcript_duration': 180,
            'transcript_word_count': 50,
            'video_metadata': {'title': 'Test Video'},
            'exec_timestamp': datetime.utcnow().isoformat()
        }
        
        result = self.node.post(self.store, prep_result, exec_result)
        
        assert result['post_status'] == 'success'
        assert result['transcript_ready'] is True
        assert 'transcript_data' in self.store
    
    def test_calculate_transcript_stats(self):
        """Test transcript statistics calculation."""
        transcript = "This is a test transcript. It has multiple sentences! How exciting?"
        stats = self.node._calculate_transcript_stats(transcript)
        
        assert stats['word_count'] == 11
        assert stats['sentence_count'] == 3
        assert stats['char_count'] == len(transcript)


class TestSummarizationNode:
    """Test cases for SummarizationNode."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.node = SummarizationNode()
        self.store = Store()
        self.store['transcript_data'] = {
            'transcript_text': 'This is a long transcript that needs to be summarized. ' * 20,
            'video_id': 'test_video'
        }
        self.store['video_metadata'] = {'title': 'Test Video'}
    
    @patch.dict(os.environ, {'LLM_PROVIDER': 'openai', 'LLM_MODEL': 'gpt-3.5-turbo'})
    @patch('nodes.create_llm_client')
    def test_prep_success(self, mock_create_client):
        """Test successful prep phase."""
        mock_client = Mock()
        mock_create_client.return_value = mock_client
        
        result = self.node.prep(self.store)
        
        assert result['prep_status'] == 'success'
        assert result['video_id'] == 'test_video'
        assert result['target_word_count'] == 500
        assert mock_create_client.called
    
    def test_prep_missing_transcript(self):
        """Test prep phase with missing transcript."""
        empty_store = Store()
        result = self.node.prep(empty_store)
        
        assert result['prep_status'] == 'failed'
        assert 'error' in result
    
    def test_prep_short_transcript(self):
        """Test prep phase with transcript too short."""
        short_store = Store()
        short_store['transcript_data'] = {'transcript_text': 'Too short'}
        
        result = self.node.prep(short_store)
        
        assert result['prep_status'] == 'failed'
        assert 'too short' in str(result['error'])
    
    @patch('nodes.create_llm_client')
    def test_exec_success(self, mock_create_client):
        """Test successful exec phase."""
        prep_result = {
            'prep_status': 'success',
            'transcript_text': 'Long transcript content...',
            'video_title': 'Test Video',
            'target_word_count': 500,
            'original_word_count': 1000,
            'video_id': 'test_video'
        }
        
        # Mock LLM client
        mock_client = Mock()
        mock_client.summarize_text.return_value = {
            'summary': 'This is a test summary of the video content.',
            'word_count': 10,
            'compression_ratio': 100,
            'generation_metadata': {'model': 'gpt-3.5-turbo'}
        }
        self.node.llm_client = mock_client
        
        result = self.node.exec(self.store, prep_result)
        
        assert result['exec_status'] == 'success'
        assert result['summary_text'] == 'This is a test summary of the video content.'
        assert result['summary_word_count'] == 10
    
    def test_calculate_summary_stats(self):
        """Test summary statistics calculation."""
        summary = "This is a test summary. It has two sentences."
        original = "This is a much longer original text that needs to be summarized into a shorter format."
        
        stats = self.node._calculate_summary_stats(summary, original, 10)
        
        assert 'word_count' in stats
        assert 'sentence_count' in stats
        assert 'compression_ratio' in stats
        assert 'target_accuracy' in stats


class TestTimestampNode:
    """Test cases for TimestampNode."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.node = TimestampNode()
        self.store = Store()
        self.store['transcript_data'] = {
            'raw_transcript': [
                {'start': 10.5, 'text': 'This is the first segment'},
                {'start': 25.0, 'text': 'This is the second segment'},
                {'start': 40.2, 'text': 'This is the third segment'}
            ],
            'video_id': 'test_video'
        }
        self.store['video_metadata'] = {
            'title': 'Test Video',
            'duration_seconds': 120
        }
    
    @patch('nodes.create_llm_client')
    def test_prep_success(self, mock_create_client):
        """Test successful prep phase."""
        mock_client = Mock()
        mock_create_client.return_value = mock_client
        
        result = self.node.prep(self.store)
        
        assert result['prep_status'] == 'success'
        assert result['video_id'] == 'test_video'
        assert result['transcript_count'] == 3
    
    def test_prep_missing_transcript(self):
        """Test prep phase with missing transcript."""
        empty_store = Store()
        empty_store['video_metadata'] = {'title': 'Test'}
        
        result = self.node.prep(empty_store)
        
        assert result['prep_status'] == 'failed'
    
    @patch('nodes.create_llm_client')
    def test_exec_success(self, mock_create_client):
        """Test successful exec phase."""
        prep_result = {
            'prep_status': 'success',
            'video_id': 'test_video',
            'video_title': 'Test Video',
            'raw_transcript': [{'start': 10, 'text': 'Test content'}],
            'timestamp_count': 5,
            'video_duration': 120
        }
        
        # Mock LLM client
        mock_client = Mock()
        mock_client.generate_timestamps.return_value = {
            'timestamps': [
                {
                    'timestamp_seconds': 10.0,
                    'timestamp_formatted': '0:10',
                    'description': 'Important moment',
                    'importance_rating': 8,
                    'youtube_url': 'https://youtube.com/watch?v=test_video&t=10s'
                }
            ],
            'generation_metadata': {'model': 'gpt-3.5-turbo'}
        }
        self.node.llm_client = mock_client
        
        result = self.node.exec(self.store, prep_result)
        
        assert result['exec_status'] == 'success'
        assert len(result['timestamps']) == 1
        assert result['timestamps'][0]['importance_rating'] == 8
    
    def test_filter_transcript_entries(self):
        """Test transcript entry filtering."""
        raw_transcript = [
            {'start': 10.5, 'text': 'Valid entry with enough words'},
            {'start': 20.0, 'text': 'Short'},  # Too short
            {'start': None, 'text': 'No start time'},  # Invalid start
            {'start': 30.0, 'text': ''},  # Empty text
            {'start': 40.0, 'text': 'Another valid entry here'}
        ]
        
        filtered = self.node._filter_transcript_entries(raw_transcript)
        
        assert len(filtered) == 2
        assert filtered[0]['start'] == 10.5
        assert filtered[1]['start'] == 40.0
    
    def test_enhance_timestamps(self):
        """Test timestamp enhancement."""
        timestamps = [
            {
                'timestamp_seconds': 30.0,
                'description': 'Test moment',
                'importance_rating': 7
            }
        ]
        
        enhanced = self.node._enhance_timestamps(timestamps, 'Test Video', 120)
        
        assert enhanced[0]['video_title'] == 'Test Video'
        assert enhanced[0]['context_tag'] == 'important'
        assert enhanced[0]['relative_position_percent'] == 25.0


class TestKeywordExtractionNode:
    """Test cases for KeywordExtractionNode."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.node = KeywordExtractionNode()
        self.store = Store()
        self.store['transcript_data'] = {
            'transcript_text': 'This is a comprehensive transcript about machine learning and artificial intelligence.',
            'video_id': 'test_video'
        }
        self.store['summary_data'] = {
            'summary_text': 'Summary about AI and ML concepts.'
        }
        self.store['video_metadata'] = {'title': 'AI Tutorial'}
    
    @patch('nodes.create_llm_client')
    def test_prep_success_with_summary(self, mock_create_client):
        """Test successful prep phase with summary available."""
        mock_client = Mock()
        mock_create_client.return_value = mock_client
        
        result = self.node.prep(self.store)
        
        assert result['prep_status'] == 'success'
        assert result['text_source'] == 'summary'
        assert result['has_summary'] is True
    
    @patch('nodes.create_llm_client')
    def test_prep_success_without_summary(self, mock_create_client):
        """Test successful prep phase without summary."""
        store_no_summary = Store()
        store_no_summary['transcript_data'] = {
            'transcript_text': 'Long transcript about technology and innovation.',
            'video_id': 'test_video'
        }
        store_no_summary['video_metadata'] = {'title': 'Tech Video'}
        
        mock_client = Mock()
        mock_create_client.return_value = mock_client
        
        result = self.node.prep(store_no_summary)
        
        assert result['prep_status'] == 'success'
        assert result['text_source'] == 'transcript'
        assert result['has_summary'] is False
    
    @patch('nodes.create_llm_client')
    def test_exec_success(self, mock_create_client):
        """Test successful exec phase."""
        prep_result = {
            'prep_status': 'success',
            'extraction_text': 'Text about machine learning and AI',
            'video_id': 'test_video',
            'video_title': 'AI Tutorial',
            'keyword_count': 6,
            'text_source': 'summary'
        }
        
        # Mock LLM client
        mock_client = Mock()
        mock_client.extract_keywords.return_value = {
            'keywords': ['machine learning', 'artificial intelligence', 'technology', 'algorithms', 'data science', 'neural networks'],
            'generation_metadata': {'model': 'gpt-3.5-turbo'}
        }
        self.node.llm_client = mock_client
        
        result = self.node.exec(self.store, prep_result)
        
        assert result['exec_status'] == 'success'
        assert len(result['keywords']) == 6
        assert result['text_source'] == 'summary'
    
    @patch('nodes.create_llm_client')
    def test_exec_insufficient_keywords(self, mock_create_client):
        """Test exec phase with insufficient initial keywords."""
        prep_result = {
            'prep_status': 'success',
            'extraction_text': 'Short text',
            'video_id': 'test_video',
            'video_title': 'Test Video',
            'keyword_count': 6,
            'text_source': 'transcript'
        }
        
        # Mock LLM client to return few keywords initially, then more
        mock_client = Mock()
        mock_client.extract_keywords.side_effect = [
            {'keywords': ['keyword1', 'keyword2'], 'generation_metadata': {}},  # First call - insufficient
            {'keywords': ['kw1', 'kw2', 'kw3', 'kw4', 'kw5', 'kw6', 'kw7', 'kw8'], 'generation_metadata': {}}  # Second call
        ]
        self.node.llm_client = mock_client
        
        result = self.node.exec(self.store, prep_result)
        
        assert result['exec_status'] == 'success'
        assert len(result['keywords']) == 8  # Limited to 8
        assert mock_client.extract_keywords.call_count == 2
    
    def test_enhance_keywords(self):
        """Test keyword enhancement."""
        keywords = ['machine learning', 'AI', 'technology']
        enhanced = self.node._enhance_keywords(keywords, 'Test Video', 'summary')
        
        assert len(enhanced) == 3
        assert enhanced[0]['keyword'] == 'machine learning'
        assert enhanced[0]['type'] == 'phrase'
        assert enhanced[0]['relevance_score'] == 10
        assert enhanced[1]['type'] == 'single_word'
        assert enhanced[1]['relevance_score'] == 9
    
    def test_calculate_keyword_stats(self):
        """Test keyword statistics calculation."""
        keywords = [
            {'keyword': 'machine learning', 'type': 'phrase', 'relevance_score': 10},
            {'keyword': 'AI', 'type': 'single_word', 'relevance_score': 9},
            {'keyword': 'technology', 'type': 'single_word', 'relevance_score': 8}
        ]
        source_text = 'This text discusses machine learning and AI technology applications.'
        
        stats = self.node._calculate_keyword_stats(keywords, source_text)
        
        assert stats['total_count'] == 3
        assert stats['phrase_count'] == 1
        assert stats['single_word_count'] == 2
        assert stats['coverage_percent'] == 100.0  # All keywords appear in text


class TestNodeIntegration:
    """Integration tests for node interactions."""
    
    def setup_method(self):
        """Set up test fixtures for integration testing."""
        self.store = Store()
        
    @patch('nodes.YouTubeTranscriptFetcher')
    @patch('nodes.YouTubeURLValidator')
    @patch('nodes.create_llm_client')
    def test_full_workflow_simulation(self, mock_create_client, mock_validator, mock_fetcher):
        """Test simulated full workflow through all nodes."""
        # Set up initial store
        self.store['youtube_url'] = 'https://www.youtube.com/watch?v=test'
        
        # Mock dependencies
        mock_validator.return_value.validate_and_extract.return_value = (True, 'test_video')
        
        mock_fetcher_instance = mock_fetcher.return_value
        mock_fetcher_instance.check_video_support.return_value = {'is_supported': True, 'issues': []}
        mock_fetcher_instance.check_transcript_availability.return_value = {
            'has_transcripts': True, 'available_transcripts': [{'language': 'en'}]
        }
        mock_fetcher_instance.fetch_transcript.return_value = {
            'transcript': 'This is a comprehensive test transcript about technology and innovation.',
            'raw_transcript': [{'start': 10, 'text': 'Technology segment', 'duration': 5}],
            'language': 'en',
            'duration_seconds': 120,
            'word_count': 50,
            'video_metadata': {'title': 'Tech Video', 'duration_seconds': 120}
        }
        
        mock_llm_client = Mock()
        mock_llm_client.summarize_text.return_value = {
            'summary': 'This video discusses technology and innovation concepts.',
            'word_count': 10,
            'compression_ratio': 5,
            'generation_metadata': {}
        }
        mock_llm_client.generate_timestamps.return_value = {
            'timestamps': [{
                'timestamp_seconds': 10.0,
                'timestamp_formatted': '0:10',
                'description': 'Key tech moment',
                'importance_rating': 8,
                'youtube_url': 'https://youtube.com/watch?v=test_video&t=10s'
            }],
            'generation_metadata': {}
        }
        mock_llm_client.extract_keywords.return_value = {
            'keywords': ['technology', 'innovation', 'concepts', 'video', 'discussion'],
            'generation_metadata': {}
        }
        mock_create_client.return_value = mock_llm_client
        
        # Test workflow through all nodes
        
        # 1. YouTube Transcript Node
        transcript_node = YouTubeTranscriptNode()
        prep_result = transcript_node.prep(self.store)
        assert prep_result['prep_status'] == 'success'
        
        exec_result = transcript_node.exec(self.store, prep_result)
        assert exec_result['exec_status'] == 'success'
        
        post_result = transcript_node.post(self.store, prep_result, exec_result)
        assert post_result['post_status'] == 'success'
        assert 'transcript_data' in self.store
        
        # 2. Summarization Node
        summary_node = SummarizationNode()
        with patch.dict(os.environ, {'LLM_PROVIDER': 'openai'}):
            prep_result = summary_node.prep(self.store)
            assert prep_result['prep_status'] == 'success'
        
        exec_result = summary_node.exec(self.store, prep_result)
        assert exec_result['exec_status'] == 'success'
        
        post_result = summary_node.post(self.store, prep_result, exec_result)
        assert post_result['post_status'] == 'success'
        assert 'summary_data' in self.store
        
        # 3. Timestamp Node
        timestamp_node = TimestampNode()
        prep_result = timestamp_node.prep(self.store)
        assert prep_result['prep_status'] == 'success'
        
        exec_result = timestamp_node.exec(self.store, prep_result)
        assert exec_result['exec_status'] == 'success'
        
        post_result = timestamp_node.post(self.store, prep_result, exec_result)
        assert post_result['post_status'] == 'success'
        assert 'timestamp_data' in self.store
        
        # 4. Keyword Extraction Node
        keyword_node = KeywordExtractionNode()
        prep_result = keyword_node.prep(self.store)
        assert prep_result['prep_status'] == 'success'
        
        exec_result = keyword_node.exec(self.store, prep_result)
        assert exec_result['exec_status'] == 'success'
        
        post_result = keyword_node.post(self.store, prep_result, exec_result)
        assert post_result['post_status'] == 'success'
        assert 'keyword_data' in self.store
        
        # Verify final store state
        assert 'transcript_data' in self.store
        assert 'summary_data' in self.store
        assert 'timestamp_data' in self.store
        assert 'keyword_data' in self.store


# Pytest fixtures and configuration
@pytest.fixture
def sample_store():
    """Fixture providing a sample store with test data."""
    store = Store()
    store['youtube_url'] = 'https://www.youtube.com/watch?v=dQw4w9WgXcQ'
    store['transcript_data'] = {
        'transcript_text': 'Sample transcript content for testing purposes.',
        'video_id': 'dQw4w9WgXcQ',
        'raw_transcript': [
            {'start': 0, 'text': 'Sample', 'duration': 1},
            {'start': 1, 'text': 'transcript', 'duration': 1},
            {'start': 2, 'text': 'content', 'duration': 1}
        ]
    }
    store['video_metadata'] = {
        'title': 'Sample Video',
        'duration_seconds': 180
    }
    return store


@pytest.fixture
def mock_llm_client():
    """Fixture providing a mocked LLM client."""
    client = Mock()
    client.summarize_text.return_value = {
        'summary': 'Test summary',
        'word_count': 10,
        'compression_ratio': 5,
        'generation_metadata': {}
    }
    client.extract_keywords.return_value = {
        'keywords': ['test', 'sample', 'content'],
        'generation_metadata': {}
    }
    client.generate_timestamps.return_value = {
        'timestamps': [{
            'timestamp_seconds': 10.0,
            'description': 'Test moment',
            'importance_rating': 7
        }],
        'generation_metadata': {}
    }
    return client


if __name__ == '__main__':
    pytest.main([__file__])