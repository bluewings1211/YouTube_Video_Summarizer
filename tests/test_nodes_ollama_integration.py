"""
Integration tests for PocketFlow nodes with Ollama integration.

This module tests the updated nodes that now use SmartLLMClient:
- SummarizationNode
- TimestampNode
- KeywordExtractionNode
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from src.nodes import (
    SummarizationNode, TimestampNode, KeywordExtractionNode, Store
)
from src.utils.smart_llm_client import TaskRequirements


class TestSummarizationNodeOllamaIntegration:
    """Test SummarizationNode with Ollama integration."""
    
    @pytest.fixture
    def summarization_node(self):
        """Create SummarizationNode instance."""
        return SummarizationNode()
    
    @pytest.fixture
    def mock_store_with_transcript(self):
        """Create mock store with transcript data."""
        store = Store()
        store.update({
            'transcript_data': {
                'video_id': 'test_video_123',
                'transcript_text': 'This is a test transcript with enough content for summarization. ' * 20,
                'raw_transcript': [
                    {'start': 0.0, 'text': 'This is a test'},
                    {'start': 5.0, 'text': 'transcript with content'}
                ],
                'language': 'en',
                'word_count': 100
            },
            'video_metadata': {
                'title': 'Test Video',
                'duration_seconds': 300
            }
        })
        return store
    
    @pytest.fixture
    def mock_store_with_chinese_transcript(self):
        """Create mock store with Chinese transcript data."""
        store = Store()
        store.update({
            'transcript_data': {
                'video_id': 'test_chinese_video_123',
                'transcript_text': '这是一个测试转录内容，包含足够的中文文本用于测试总结功能。' * 10,
                'raw_transcript': [
                    {'start': 0.0, 'text': '这是一个测试'},
                    {'start': 5.0, 'text': '转录内容'}
                ],
                'language': 'zh-CN',
                'word_count': 50
            },
            'video_metadata': {
                'title': '测试视频',
                'duration_seconds': 300
            }
        })
        return store
    
    @patch('src.nodes.SmartLLMClient')
    def test_summarization_node_prep_success(self, mock_smart_client_class, summarization_node, mock_store_with_transcript):
        """Test successful preparation phase."""
        mock_smart_client = Mock()
        mock_smart_client_class.return_value = mock_smart_client
        
        result = summarization_node.prep(mock_store_with_transcript)
        
        assert result['prep_status'] == 'success'
        assert result['video_id'] == 'test_video_123'
        assert result['original_word_count'] > 50
        assert result['target_word_count'] == 500
        mock_smart_client_class.assert_called_once()
    
    @patch('src.nodes.SmartLLMClient')
    def test_summarization_node_prep_short_transcript(self, mock_smart_client_class, summarization_node):
        """Test prep phase with too short transcript."""
        store = Store()
        store.update({
            'transcript_data': {
                'transcript_text': 'Too short',  # Only 2 words
                'video_id': 'test'
            }
        })
        
        result = summarization_node.prep(store)
        
        assert result['prep_status'] == 'failed'
        assert 'too short' in result['error']['message'].lower()
    
    @patch('src.nodes.detect_task_requirements')
    @patch('src.nodes.SmartLLMClient')
    def test_summarization_node_exec_success(self, mock_smart_client_class, mock_detect_requirements, summarization_node, mock_store_with_transcript):
        """Test successful execution phase."""
        # Setup mocks
        mock_smart_client = Mock()
        mock_smart_client_class.return_value = mock_smart_client
        
        mock_requirements = TaskRequirements(task_type="summarization")
        mock_detect_requirements.return_value = mock_requirements
        
        mock_smart_client.generate_text_with_chinese_optimization.return_value = {
            'text': 'This is a generated summary of the test content.',
            'usage': {'prompt_tokens': 100, 'completion_tokens': 50, 'total_tokens': 150},
            'provider': 'ollama',
            'model': 'llama3.1:8b'
        }
        
        # Prepare node
        summarization_node.prep(mock_store_with_transcript)
        
        # Create prep result
        prep_result = {
            'prep_status': 'success',
            'transcript_text': mock_store_with_transcript['transcript_data']['transcript_text'],
            'original_word_count': 100,
            'target_word_count': 500,
            'video_id': 'test_video_123',
            'video_title': 'Test Video'
        }
        
        result = summarization_node.exec(mock_store_with_transcript, prep_result)
        
        assert result['exec_status'] == 'success'
        assert 'summary_text' in result
        assert result['summary_word_count'] > 0
        assert result['video_id'] == 'test_video_123'
        mock_smart_client.generate_text_with_chinese_optimization.assert_called_once()
    
    @patch('src.nodes.detect_task_requirements')
    @patch('src.nodes.SmartLLMClient')
    def test_summarization_node_chinese_content(self, mock_smart_client_class, mock_detect_requirements, summarization_node, mock_store_with_chinese_transcript):
        """Test summarization with Chinese content."""
        # Setup mocks
        mock_smart_client = Mock()
        mock_smart_client_class.return_value = mock_smart_client
        
        mock_requirements = TaskRequirements(task_type="summarization", language="zh-CN")
        mock_detect_requirements.return_value = mock_requirements
        
        mock_smart_client.generate_text_with_chinese_optimization.return_value = {
            'text': '这是一个生成的中文摘要内容。',
            'usage': {'prompt_tokens': 80, 'completion_tokens': 30, 'total_tokens': 110},
            'provider': 'ollama',
            'model': 'qwen2.5:7b'
        }
        
        # Prepare and execute
        summarization_node.prep(mock_store_with_chinese_transcript)
        
        prep_result = {
            'prep_status': 'success',
            'transcript_text': mock_store_with_chinese_transcript['transcript_data']['transcript_text'],
            'original_word_count': 50,
            'target_word_count': 500,
            'video_id': 'test_chinese_video_123',
            'video_title': '测试视频'
        }
        
        result = summarization_node.exec(mock_store_with_chinese_transcript, prep_result)
        
        assert result['exec_status'] == 'success'
        assert '中文' in result['summary_text']
        mock_detect_requirements.assert_called_once()
        call_args = mock_detect_requirements.call_args[1]
        assert call_args['task_type'] == 'summarization'


class TestTimestampNodeOllamaIntegration:
    """Test TimestampNode with Ollama integration."""
    
    @pytest.fixture
    def timestamp_node(self):
        """Create TimestampNode instance."""
        return TimestampNode()
    
    @pytest.fixture
    def mock_store_with_transcript_and_metadata(self):
        """Create mock store with transcript and video data."""
        store = Store()
        store.update({
            'transcript_data': {
                'video_id': 'test_video_456',
                'raw_transcript': [
                    {'start': 0.0, 'text': 'Welcome to this tutorial'},
                    {'start': 10.0, 'text': 'First we will cover basics'},
                    {'start': 30.0, 'text': 'Then advanced topics'},
                    {'start': 60.0, 'text': 'Finally some examples'},
                    {'start': 90.0, 'text': 'Thanks for watching'}
                ]
            },
            'video_metadata': {
                'title': 'Programming Tutorial',
                'duration_seconds': 120
            }
        })
        return store
    
    @patch('src.nodes.SmartLLMClient')
    def test_timestamp_node_prep_success(self, mock_smart_client_class, timestamp_node, mock_store_with_transcript_and_metadata):
        """Test successful timestamp preparation."""
        mock_smart_client = Mock()
        mock_smart_client_class.return_value = mock_smart_client
        
        result = timestamp_node.prep(mock_store_with_transcript_and_metadata)
        
        assert result['prep_status'] == 'success'
        assert result['video_id'] == 'test_video_456'
        assert result['timestamp_count'] == 5
        assert len(result['raw_transcript']) >= 3  # Filtered entries
    
    @patch('src.nodes.detect_task_requirements')
    @patch('src.nodes.SmartLLMClient')
    def test_timestamp_node_exec_success(self, mock_smart_client_class, mock_detect_requirements, timestamp_node, mock_store_with_transcript_and_metadata):
        """Test successful timestamp execution."""
        # Setup mocks
        mock_smart_client = Mock()
        mock_smart_client_class.return_value = mock_smart_client
        
        mock_requirements = TaskRequirements(task_type="timestamps")
        mock_detect_requirements.return_value = mock_requirements
        
        # Mock the custom timestamp generation method
        timestamp_node._generate_timestamps_with_smart_client = Mock(return_value={
            'timestamps': [
                {
                    'timestamp_seconds': 10.0,
                    'timestamp_formatted': '0:10',
                    'description': 'Introduction to basics',
                    'importance_rating': 8,
                    'youtube_url': 'https://www.youtube.com/watch?v=test_video_456&t=10s'
                },
                {
                    'timestamp_seconds': 60.0,
                    'timestamp_formatted': '1:00',
                    'description': 'Advanced examples',
                    'importance_rating': 9,
                    'youtube_url': 'https://www.youtube.com/watch?v=test_video_456&t=60s'
                }
            ],
            'count': 2,
            'requested_count': 5,
            'video_id': 'test_video_456',
            'generation_metadata': {'provider': 'ollama', 'model': 'llama3.1:8b'}
        })
        
        # Prepare
        timestamp_node.prep(mock_store_with_transcript_and_metadata)
        
        prep_result = {
            'prep_status': 'success',
            'video_id': 'test_video_456',
            'video_title': 'Programming Tutorial',
            'video_duration': 120,
            'raw_transcript': mock_store_with_transcript_and_metadata['transcript_data']['raw_transcript'],
            'timestamp_count': 5
        }
        
        result = timestamp_node.exec(mock_store_with_transcript_and_metadata, prep_result)
        
        assert result['exec_status'] == 'success'
        assert len(result['timestamps']) == 2
        assert result['timestamps'][0]['importance_rating'] == 8
        assert 'youtube.com' in result['timestamps'][0]['youtube_url']


class TestKeywordExtractionNodeOllamaIntegration:
    """Test KeywordExtractionNode with Ollama integration."""
    
    @pytest.fixture
    def keyword_node(self):
        """Create KeywordExtractionNode instance."""
        return KeywordExtractionNode()
    
    @pytest.fixture
    def mock_store_with_summary(self):
        """Create mock store with summary data."""
        store = Store()
        store.update({
            'transcript_data': {
                'video_id': 'test_video_789',
                'transcript_text': 'This is a long transcript about programming and software development. ' * 10,
            },
            'summary_data': {
                'summary_text': 'This video covers programming fundamentals and software development best practices.',
            },
            'video_metadata': {
                'title': 'Programming Fundamentals'
            }
        })
        return store
    
    @patch('src.nodes.SmartLLMClient')
    def test_keyword_node_prep_success(self, mock_smart_client_class, keyword_node, mock_store_with_summary):
        """Test successful keyword extraction preparation."""
        mock_smart_client = Mock()
        mock_smart_client_class.return_value = mock_smart_client
        
        result = keyword_node.prep(mock_store_with_summary)
        
        assert result['prep_status'] == 'success'
        assert result['video_id'] == 'test_video_789'
        assert result['text_source'] == 'summary'  # Should prefer summary over transcript
        assert result['has_summary'] is True
    
    @patch('src.nodes.detect_task_requirements')
    @patch('src.nodes.SmartLLMClient')
    def test_keyword_node_exec_success(self, mock_smart_client_class, mock_detect_requirements, keyword_node, mock_store_with_summary):
        """Test successful keyword extraction execution."""
        # Setup mocks
        mock_smart_client = Mock()
        mock_smart_client_class.return_value = mock_smart_client
        
        mock_requirements = TaskRequirements(task_type="keywords")
        mock_detect_requirements.return_value = mock_requirements
        
        # Mock the custom keyword generation method
        keyword_node._generate_keywords_with_smart_client = Mock(return_value={
            'keywords': ['programming', 'software development', 'fundamentals', 'best practices', 'coding', 'tutorial'],
            'count': 6,
            'requested_count': 6,
            'generation_metadata': {'provider': 'ollama', 'model': 'mistral:7b'}
        })
        
        # Prepare
        keyword_node.prep(mock_store_with_summary)
        
        prep_result = {
            'prep_status': 'success',
            'video_id': 'test_video_789',
            'video_title': 'Programming Fundamentals',
            'extraction_text': 'This video covers programming fundamentals and software development best practices.',
            'text_source': 'summary',
            'keyword_count': 6
        }
        
        result = keyword_node.exec(mock_store_with_summary, prep_result)
        
        assert result['exec_status'] == 'success'
        assert len(result['keywords']) == 6
        assert result['keywords'][0]['keyword'] == 'programming'
        assert result['text_source'] == 'summary'


class TestNodesErrorHandling:
    """Test error handling in nodes with Ollama integration."""
    
    @patch('src.nodes.SmartLLMClient')
    def test_summarization_node_smart_client_init_failure(self, mock_smart_client_class):
        """Test handling of SmartLLMClient initialization failure."""
        mock_smart_client_class.side_effect = Exception("Smart client init failed")
        
        node = SummarizationNode()
        store = Store()
        store.update({
            'transcript_data': {
                'transcript_text': 'Test transcript with enough content for testing. ' * 20,
                'video_id': 'test'
            },
            'video_metadata': {'title': 'Test'}
        })
        
        result = node.prep(store)
        
        assert result['prep_status'] == 'failed'
        assert 'Smart LLM client' in result['error']['message']
    
    @patch('src.nodes.SmartLLMClient')
    def test_node_exec_with_failed_prep(self, mock_smart_client_class):
        """Test exec phase when prep phase failed."""
        node = SummarizationNode()
        store = Store()
        
        failed_prep_result = {
            'prep_status': 'failed',
            'error': 'Prep failed'
        }
        
        result = node.exec(store, failed_prep_result)
        
        assert result['exec_status'] == 'failed'
        assert 'Prep phase failed' in result['error']
    
    @patch('src.nodes.detect_task_requirements')
    @patch('src.nodes.SmartLLMClient')
    def test_summarization_node_generation_failure(self, mock_smart_client_class, mock_detect_requirements):
        """Test handling of text generation failure."""
        # Setup mocks
        mock_smart_client = Mock()
        mock_smart_client_class.return_value = mock_smart_client
        
        mock_requirements = TaskRequirements(task_type="summarization")
        mock_detect_requirements.return_value = mock_requirements
        
        # Make generation fail
        mock_smart_client.generate_text_with_chinese_optimization.side_effect = Exception("Generation failed")
        
        node = SummarizationNode()
        store = Store()
        store.update({
            'transcript_data': {
                'transcript_text': 'Test transcript content. ' * 20,
                'video_id': 'test'
            },
            'video_metadata': {'title': 'Test'}
        })
        
        # Prepare successfully first
        node.prep(store)
        
        prep_result = {
            'prep_status': 'success',
            'transcript_text': 'Test transcript content. ' * 20,
            'original_word_count': 100,
            'target_word_count': 500,
            'video_id': 'test',
            'video_title': 'Test'
        }
        
        result = node.exec(store, prep_result)
        
        assert result['exec_status'] == 'failed'
        assert 'Generation failed' in str(result['error'])


class TestNodesIntegrationFlows:
    """Test complete integration flows with multiple nodes."""
    
    @patch('src.nodes.detect_task_requirements')
    @patch('src.nodes.SmartLLMClient')
    def test_complete_processing_flow(self, mock_smart_client_class, mock_detect_requirements):
        """Test complete flow: transcript -> summary -> keywords -> timestamps."""
        # Setup common mocks
        mock_smart_client = Mock()
        mock_smart_client_class.return_value = mock_smart_client
        
        mock_detect_requirements.side_effect = [
            TaskRequirements(task_type="summarization"),
            TaskRequirements(task_type="keywords"),
            TaskRequirements(task_type="timestamps")
        ]
        
        # Mock generation responses
        mock_smart_client.generate_text_with_chinese_optimization.side_effect = [
            {  # Summary
                'text': 'This is a comprehensive summary of the video content.',
                'usage': {'prompt_tokens': 100, 'completion_tokens': 50, 'total_tokens': 150},
                'provider': 'ollama',
                'model': 'llama3.1:8b'
            },
            {  # Keywords (will be mocked differently)
                'text': 'programming\nsoftware\ndevelopment\ntutorial\nbest practices\ncoding',
                'usage': {'prompt_tokens': 50, 'completion_tokens': 25, 'total_tokens': 75},
                'provider': 'ollama',
                'model': 'mistral:7b'
            },
            {  # Timestamps (will be mocked differently)
                'text': '[10.5]s (Rating: 8/10) - Introduction to programming\n[45.2]s (Rating: 9/10) - Core concepts explained',
                'usage': {'prompt_tokens': 80, 'completion_tokens': 40, 'total_tokens': 120},
                'provider': 'ollama',
                'model': 'llama3.1:8b'
            }
        ]
        
        # Create store with initial data
        store = Store()
        store.update({
            'transcript_data': {
                'video_id': 'integration_test_123',
                'transcript_text': 'This is a detailed programming tutorial covering software development concepts. ' * 20,
                'raw_transcript': [
                    {'start': 10.5, 'text': 'Introduction to programming concepts'},
                    {'start': 45.2, 'text': 'Core software development principles'},
                    {'start': 78.9, 'text': 'Advanced programming techniques'}
                ],
                'language': 'en',
                'word_count': 200
            },
            'video_metadata': {
                'title': 'Complete Programming Tutorial',
                'duration_seconds': 600
            }
        })
        
        # Step 1: Summarization
        summary_node = SummarizationNode()
        summary_prep = summary_node.prep(store)
        assert summary_prep['prep_status'] == 'success'
        
        summary_exec = summary_node.exec(store, summary_prep)
        assert summary_exec['exec_status'] == 'success'
        
        summary_post = summary_node.post(store, summary_prep, summary_exec)
        assert summary_post['post_status'] == 'success'
        
        # Store should now have summary data
        assert 'summary_data' in store
        
        # Step 2: Keyword Extraction
        keyword_node = KeywordExtractionNode()
        # Mock the keyword generation method since it's custom
        keyword_node._generate_keywords_with_smart_client = Mock(return_value={
            'keywords': ['programming', 'software', 'development', 'tutorial', 'best practices', 'coding'],
            'count': 6,
            'requested_count': 6,
            'generation_metadata': {'provider': 'ollama', 'model': 'mistral:7b'}
        })
        
        keyword_prep = keyword_node.prep(store)
        assert keyword_prep['prep_status'] == 'success'
        
        keyword_exec = keyword_node.exec(store, keyword_prep)
        assert keyword_exec['exec_status'] == 'success'
        
        keyword_post = keyword_node.post(store, keyword_prep, keyword_exec)
        assert keyword_post['post_status'] == 'success'
        
        # Store should now have keyword data
        assert 'keyword_data' in store
        
        # Step 3: Timestamp Generation
        timestamp_node = TimestampNode()
        # Mock the timestamp generation method
        timestamp_node._generate_timestamps_with_smart_client = Mock(return_value={
            'timestamps': [
                {
                    'timestamp_seconds': 10.5,
                    'timestamp_formatted': '0:10',
                    'description': 'Introduction to programming',
                    'importance_rating': 8,
                    'youtube_url': 'https://www.youtube.com/watch?v=integration_test_123&t=10s'
                },
                {
                    'timestamp_seconds': 45.2,
                    'timestamp_formatted': '0:45',
                    'description': 'Core concepts explained',
                    'importance_rating': 9,
                    'youtube_url': 'https://www.youtube.com/watch?v=integration_test_123&t=45s'
                }
            ],
            'count': 2,
            'requested_count': 5,
            'video_id': 'integration_test_123',
            'generation_metadata': {'provider': 'ollama', 'model': 'llama3.1:8b'}
        })
        
        timestamp_prep = timestamp_node.prep(store)
        assert timestamp_prep['prep_status'] == 'success'
        
        timestamp_exec = timestamp_node.exec(store, timestamp_prep)
        assert timestamp_exec['exec_status'] == 'success'
        
        timestamp_post = timestamp_node.post(store, timestamp_prep, timestamp_exec)
        assert timestamp_post['post_status'] == 'success'
        
        # Store should now have timestamp data
        assert 'timestamp_data' in store
        
        # Verify complete processing
        assert store['summary_data']['summary_text'] == 'This is a comprehensive summary of the video content.'
        assert len(store['keyword_data']['keywords']) == 6
        assert len(store['timestamp_data']['timestamps']) == 2


if __name__ == '__main__':
    pytest.main([__file__])