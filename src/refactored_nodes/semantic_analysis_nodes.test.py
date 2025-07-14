"""
Tests for Semantic Analysis PocketFlow Nodes.

This module contains tests for the semantic analysis workflow nodes,
including SemanticAnalysisNode, VectorSearchNode, and EnhancedTimestampNode.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any

# Import the nodes to test
from .semantic_analysis_nodes import (
    SemanticAnalysisNode,
    VectorSearchNode,
    EnhancedTimestampNode
)
from .validation_nodes import Store


class TestSemanticAnalysisNode:
    """Test suite for SemanticAnalysisNode."""

    @pytest.fixture
    def semantic_node(self):
        """Create a SemanticAnalysisNode instance."""
        return SemanticAnalysisNode()

    @pytest.fixture
    def mock_store(self):
        """Create a mock Store with transcript data."""
        store = Store()
        store.update({
            'transcript_data': {
                'raw_transcript': [
                    {
                        'start': 0.0,
                        'duration': 5.0,
                        'text': 'Hello and welcome to this video about machine learning.'
                    },
                    {
                        'start': 5.0,
                        'duration': 4.0,
                        'text': 'Today we will explore neural networks and deep learning.'
                    },
                    {
                        'start': 9.0,
                        'duration': 6.0,
                        'text': 'First, let us understand what artificial intelligence means.'
                    }
                ],
                'video_id': 'test_video_123'
            },
            'video_metadata': {
                'title': 'Machine Learning Tutorial',
                'duration_seconds': 300
            },
            'video_id': 'test_video_123'
        })
        return store

    @pytest.fixture
    def empty_store(self):
        """Create an empty Store for error testing."""
        return Store()

    def test_node_initialization(self, semantic_node):
        """Test that the node initializes correctly."""
        assert semantic_node.name == "SemanticAnalysisNode"
        assert semantic_node.max_retries == 3
        assert semantic_node.retry_delay == 2.0
        assert semantic_node.semantic_service is None

    @patch('src.refactored_nodes.semantic_analysis_nodes.create_semantic_analysis_service')
    def test_prep_success(self, mock_create_service, semantic_node, mock_store):
        """Test successful preparation phase."""
        mock_service = Mock()
        mock_create_service.return_value = mock_service
        
        result = semantic_node.prep(mock_store)
        
        assert result['status'] == 'success'
        assert result['video_id'] == 'test_video_123'
        assert result['video_title'] == 'Machine Learning Tutorial'
        assert result['segment_count'] == 3
        assert 'total_duration' in result

    def test_prep_missing_data(self, semantic_node, empty_store):
        """Test preparation with missing required data."""
        result = semantic_node.prep(empty_store)
        
        assert result['status'] == 'failed'
        assert 'Missing required data for semantic analysis' in str(result['error'])

    def test_prep_empty_transcript(self, semantic_node):
        """Test preparation with empty transcript."""
        store = Store()
        store.update({
            'transcript_data': {
                'raw_transcript': [],
                'video_id': 'empty_video'
            }
        })
        
        result = semantic_node.prep(store)
        
        assert result['status'] == 'failed'
        assert 'No raw transcript data available' in str(result['error'])

    @patch('src.refactored_nodes.semantic_analysis_nodes.create_semantic_analysis_service')
    def test_exec_success(self, mock_create_service, semantic_node, mock_store):
        """Test successful execution phase."""
        # Setup mock service
        mock_service = Mock()
        mock_create_service.return_value = mock_service
        
        mock_analysis_result = {
            'status': 'success',
            'semantic_clusters': [{'cluster_id': 'test_cluster'}],
            'semantic_keywords': ['machine', 'learning'],
            'semantic_metrics': {'quality': 0.8},
            'timestamps': [{'timestamp': '0:30', 'description': 'Key moment'}]
        }
        mock_service.analyze_transcript.return_value = mock_analysis_result
        
        # Prep first
        prep_result = semantic_node.prep(mock_store)
        
        # Execute
        exec_result = semantic_node.exec(mock_store, prep_result)
        
        assert exec_result['exec_status'] == 'success'
        assert exec_result['video_id'] == 'test_video_123'
        assert exec_result['cluster_count'] == 1
        assert exec_result['keyword_count'] == 2
        assert exec_result['timestamp_count'] == 1

    def test_exec_failed_prep(self, semantic_node, mock_store):
        """Test execution with failed prep result."""
        failed_prep = {'status': 'failed', 'error': 'Test error'}
        
        exec_result = semantic_node.exec(mock_store, failed_prep)
        
        assert exec_result['exec_status'] == 'failed'
        assert exec_result['error'] == 'Prep phase failed'

    @patch('src.refactored_nodes.semantic_analysis_nodes.create_semantic_analysis_service')
    def test_exec_semantic_analysis_failure(self, mock_create_service, semantic_node, mock_store):
        """Test execution when semantic analysis fails."""
        # Setup mock service that fails
        mock_service = Mock()
        mock_create_service.return_value = mock_service
        mock_service.analyze_transcript.return_value = {'status': 'error', 'error': 'Analysis failed'}
        
        prep_result = semantic_node.prep(mock_store)
        exec_result = semantic_node.exec(mock_store, prep_result)
        
        assert exec_result['exec_status'] == 'failed'

    @patch('src.refactored_nodes.semantic_analysis_nodes.create_semantic_analysis_service')
    def test_post_success(self, mock_create_service, semantic_node, mock_store):
        """Test successful post-processing phase."""
        # Setup
        mock_service = Mock()
        mock_create_service.return_value = mock_service
        mock_service.analyze_transcript.return_value = {
            'status': 'success',
            'semantic_clusters': [],
            'semantic_keywords': [],
            'semantic_metrics': {},
            'timestamps': []
        }
        
        prep_result = semantic_node.prep(mock_store)
        exec_result = semantic_node.exec(mock_store, prep_result)
        
        # Post-process
        post_result = semantic_node.post(mock_store, prep_result, exec_result)
        
        assert post_result['post_status'] == 'success'
        assert post_result['semantic_analysis_ready'] == True
        assert 'semantic_analysis_result' in mock_store
        assert 'processing_metadata' in mock_store

    def test_post_failed_exec(self, semantic_node, mock_store):
        """Test post-processing with failed execution."""
        prep_result = {'status': 'success'}
        exec_result = {'exec_status': 'failed', 'error': 'Test error'}
        
        post_result = semantic_node.post(mock_store, prep_result, exec_result)
        
        assert post_result['post_status'] == 'failed'
        assert post_result['error'] == 'Execution phase failed'

    def test_calculate_total_duration(self, semantic_node):
        """Test total duration calculation."""
        segments = [
            {'duration': 5.0},
            {'duration': 3.0},
            {'duration': 7.0}
        ]
        
        duration = semantic_node._calculate_total_duration(segments)
        assert duration == 15.0

    def test_calculate_total_duration_no_duration(self, semantic_node):
        """Test total duration calculation without duration field."""
        segments = [
            {'start': 0.0},
            {'start': 5.0},
            {'start': 10.0}
        ]
        
        duration = semantic_node._calculate_total_duration(segments)
        assert duration == 15.0  # 10.0 + 5 seconds buffer


class TestVectorSearchNode:
    """Test suite for VectorSearchNode."""

    @pytest.fixture
    def vector_node(self):
        """Create a VectorSearchNode instance."""
        return VectorSearchNode()

    @pytest.fixture
    def mock_store_with_semantic_data(self):
        """Create a Store with semantic analysis results."""
        store = Store()
        store.update({
            'semantic_analysis_result': {
                'status': 'success',
                'semantic_clusters': [{'cluster_id': 'test_cluster'}]
            },
            'video_id': 'test_video_123',
            'transcript_data': {
                'raw_transcript': [
                    {'start': 0.0, 'text': 'Test content'}
                ]
            }
        })
        return store

    def test_node_initialization(self, vector_node):
        """Test that the node initializes correctly."""
        assert vector_node.name == "VectorSearchNode"
        assert vector_node.vector_engine is None

    @patch('src.refactored_nodes.semantic_analysis_nodes.create_vector_search_engine')
    def test_prep_success(self, mock_create_engine, vector_node, mock_store_with_semantic_data):
        """Test successful preparation phase."""
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine
        
        result = vector_node.prep(mock_store_with_semantic_data)
        
        assert result['status'] == 'success'
        assert result['video_id'] == 'test_video_123'
        assert result['vector_engine_ready'] == True

    def test_prep_missing_semantic_data(self, vector_node):
        """Test preparation without semantic analysis results."""
        empty_store = Store()
        
        result = vector_node.prep(empty_store)
        
        assert result['status'] == 'failed'
        assert 'Vector search requires semantic analysis results' in str(result['error'])

    def test_prep_failed_semantic_analysis(self, vector_node):
        """Test preparation with failed semantic analysis."""
        store = Store()
        store['semantic_analysis_result'] = {'status': 'failed'}
        
        result = vector_node.prep(store)
        
        assert result['status'] == 'failed'
        assert 'Semantic analysis must be successful' in str(result['error'])

    @patch('src.refactored_nodes.semantic_analysis_nodes.create_vector_search_engine')
    @patch('src.services.semantic_analysis_service.TranscriptSegment')
    def test_exec_success(self, mock_segment, mock_create_engine, vector_node, mock_store_with_semantic_data):
        """Test successful execution phase."""
        # Setup mocks
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine
        mock_engine.create_index.return_value = True
        mock_engine.analyze_semantic_coherence.return_value = {'coherence_score': 0.7}
        mock_engine.find_optimal_timestamps.return_value = []
        mock_engine.get_index_info.return_value = {'exists': True}
        
        prep_result = vector_node.prep(mock_store_with_semantic_data)
        exec_result = vector_node.exec(mock_store_with_semantic_data, prep_result)
        
        assert exec_result['exec_status'] == 'success'
        assert exec_result['vector_index_created'] == True
        assert 'coherence_analysis' in exec_result

    @patch('src.refactored_nodes.semantic_analysis_nodes.create_vector_search_engine')
    def test_post_success(self, mock_create_engine, vector_node, mock_store_with_semantic_data):
        """Test successful post-processing phase."""
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine
        mock_engine.create_index.return_value = True
        mock_engine.analyze_semantic_coherence.return_value = {'coherence_score': 0.7}
        mock_engine.find_optimal_timestamps.return_value = []
        mock_engine.get_index_info.return_value = {'exists': True}
        
        prep_result = vector_node.prep(mock_store_with_semantic_data)
        exec_result = vector_node.exec(mock_store_with_semantic_data, prep_result)
        post_result = vector_node.post(mock_store_with_semantic_data, prep_result, exec_result)
        
        assert post_result['post_status'] == 'success'
        assert post_result['vector_search_ready'] == True
        assert 'vector_search_result' in mock_store_with_semantic_data


class TestEnhancedTimestampNode:
    """Test suite for EnhancedTimestampNode."""

    @pytest.fixture
    def timestamp_node(self):
        """Create an EnhancedTimestampNode instance."""
        return EnhancedTimestampNode()

    @pytest.fixture
    def mock_store_with_all_data(self):
        """Create a Store with complete semantic and vector data."""
        store = Store()
        store.update({
            'semantic_analysis_result': {
                'status': 'success',
                'clusters': [],
                'keywords': []
            },
            'vector_search_result': {
                'exec_status': 'success',
                'optimized_timestamps': []
            },
            'transcript_data': {
                'raw_transcript': [
                    {'start': 0.0, 'text': 'Test content'}
                ]
            },
            'semantic_timestamps': [
                {'timestamp_seconds': 10.0, 'description': 'Semantic timestamp'}
            ],
            'optimized_timestamps': [
                {'timestamp_seconds': 20.0, 'description': 'Vector timestamp'}
            ],
            'video_id': 'test_video_123'
        })
        return store

    def test_node_initialization(self, timestamp_node):
        """Test that the node initializes correctly."""
        assert timestamp_node.name == "EnhancedTimestampNode"

    def test_prep_success(self, timestamp_node, mock_store_with_all_data):
        """Test successful preparation phase."""
        result = timestamp_node.prep(mock_store_with_all_data)
        
        assert result['status'] == 'success'
        assert result['video_id'] == 'test_video_123'
        assert result['semantic_available'] == True
        assert result['vector_available'] == True
        assert result['transcript_available'] == True

    def test_prep_missing_video_id(self, timestamp_node):
        """Test preparation without video ID."""
        store = Store()
        store['transcript_data'] = {'raw_transcript': []}
        
        result = timestamp_node.prep(store)
        
        assert result['status'] == 'failed'
        assert 'Video ID is required' in str(result['error'])

    def test_determine_timestamp_methods(self, timestamp_node, mock_store_with_all_data):
        """Test timestamp method determination."""
        methods = timestamp_node._determine_timestamp_methods(mock_store_with_all_data)
        
        assert 'semantic' in methods
        assert 'vector' in methods
        assert 'traditional' in methods

    def test_determine_timestamp_methods_minimal(self, timestamp_node):
        """Test timestamp method determination with minimal data."""
        store = Store()
        store['transcript_data'] = {'raw_transcript': []}
        
        methods = timestamp_node._determine_timestamp_methods(store)
        
        assert 'traditional' in methods
        assert 'semantic' not in methods
        assert 'vector' not in methods

    def test_exec_success(self, timestamp_node, mock_store_with_all_data):
        """Test successful execution phase."""
        prep_result = timestamp_node.prep(mock_store_with_all_data)
        exec_result = timestamp_node.exec(mock_store_with_all_data, prep_result)
        
        assert exec_result['exec_status'] == 'success'
        assert exec_result['video_id'] == 'test_video_123'
        assert 'enhanced_timestamps' in exec_result
        assert 'method_results' in exec_result

    def test_post_success(self, timestamp_node, mock_store_with_all_data):
        """Test successful post-processing phase."""
        prep_result = timestamp_node.prep(mock_store_with_all_data)
        exec_result = timestamp_node.exec(mock_store_with_all_data, prep_result)
        post_result = timestamp_node.post(mock_store_with_all_data, prep_result, exec_result)
        
        assert post_result['post_status'] == 'success'
        assert post_result['enhanced_timestamps_ready'] == True
        assert 'timestamps' in mock_store_with_all_data
        assert 'enhanced_timestamp_metadata' in mock_store_with_all_data

    def test_deduplicate_timestamps(self, timestamp_node):
        """Test timestamp deduplication logic."""
        timestamps = [
            {'timestamp_seconds': 10.0, 'description': 'First'},
            {'timestamp_seconds': 12.0, 'description': 'Second'},  # Too close to first
            {'timestamp_seconds': 20.0, 'description': 'Third'},
            {'timestamp_seconds': 22.0, 'description': 'Fourth'}   # Too close to third
        ]
        
        deduplicated = timestamp_node._deduplicate_timestamps(timestamps)
        
        # Should remove duplicates within 5 seconds
        assert len(deduplicated) == 2
        assert deduplicated[0]['timestamp_seconds'] == 10.0
        assert deduplicated[1]['timestamp_seconds'] == 20.0

    def test_calculate_timestamp_quality(self, timestamp_node):
        """Test timestamp quality calculation."""
        timestamps = [
            {'timestamp_seconds': 10.0, 'description': 'Good description'},
            {'timestamp_seconds': 30.0, 'description': 'Another good description'}
        ]
        
        method_results = {
            'semantic': {'count': 1, 'quality': 'high'},
            'vector': {'count': 1, 'quality': 'high'}
        }
        
        quality = timestamp_node._calculate_timestamp_quality(timestamps, method_results)
        
        assert 'overall_quality' in quality
        assert quality['overall_quality'] > 0.0
        assert quality['timestamp_count'] == 2

    def test_assess_time_distribution(self, timestamp_node):
        """Test time distribution assessment."""
        # Good distribution
        good_timestamps = [
            {'timestamp_seconds': 0.0},
            {'timestamp_seconds': 30.0},
            {'timestamp_seconds': 60.0}
        ]
        
        score = timestamp_node._assess_time_distribution(good_timestamps)
        assert score > 0.0

    def test_assess_description_quality(self, timestamp_node):
        """Test description quality assessment."""
        timestamps = [
            {'description': 'This is a meaningful description'},
            {'description': 'Another good description'},
            {'description': 'Key moment'}  # Generic description
        ]
        
        score = timestamp_node._assess_description_quality(timestamps)
        assert score > 0.0


class TestSemanticAnalysisNodesIntegration:
    """Integration tests for semantic analysis nodes working together."""

    @pytest.fixture
    def semantic_node(self):
        return SemanticAnalysisNode()

    @pytest.fixture
    def vector_node(self):
        return VectorSearchNode()

    @pytest.fixture
    def timestamp_node(self):
        return EnhancedTimestampNode()

    @pytest.fixture
    def integration_store(self):
        """Create a Store for integration testing."""
        store = Store()
        store.update({
            'transcript_data': {
                'raw_transcript': [
                    {
                        'start': 0.0,
                        'duration': 10.0,
                        'text': 'Welcome to this comprehensive tutorial on machine learning and artificial intelligence.'
                    },
                    {
                        'start': 10.0,
                        'duration': 8.0,
                        'text': 'Today we will explore neural networks, deep learning, and various AI applications.'
                    },
                    {
                        'start': 18.0,
                        'duration': 12.0,
                        'text': 'Machine learning algorithms can be categorized into supervised, unsupervised, and reinforcement learning.'
                    }
                ],
                'video_id': 'integration_test_video'
            },
            'video_metadata': {
                'title': 'AI and Machine Learning Tutorial',
                'duration_seconds': 300
            },
            'video_id': 'integration_test_video'
        })
        return store

    @patch('src.refactored_nodes.semantic_analysis_nodes.create_semantic_analysis_service')
    @patch('src.refactored_nodes.semantic_analysis_nodes.create_vector_search_engine')
    def test_sequential_node_execution(self, mock_vector_engine, mock_semantic_service, 
                                     semantic_node, vector_node, timestamp_node, integration_store):
        """Test executing nodes in sequence as they would in the workflow."""
        # Setup mocks
        mock_service = Mock()
        mock_semantic_service.return_value = mock_service
        mock_service.analyze_transcript.return_value = {
            'status': 'success',
            'semantic_clusters': [{'cluster_id': 'test'}],
            'semantic_keywords': ['machine', 'learning'],
            'semantic_metrics': {'quality': 0.8},
            'timestamps': [{'timestamp_seconds': 15.0, 'description': 'Key moment'}]
        }
        
        mock_engine = Mock()
        mock_vector_engine.return_value = mock_engine
        mock_engine.create_index.return_value = True
        mock_engine.analyze_semantic_coherence.return_value = {'coherence_score': 0.7}
        mock_engine.find_optimal_timestamps.return_value = []
        mock_engine.get_index_info.return_value = {'exists': True}
        
        # Execute semantic analysis node
        semantic_prep = semantic_node.prep(integration_store)
        semantic_exec = semantic_node.exec(integration_store, semantic_prep)
        semantic_post = semantic_node.post(integration_store, semantic_prep, semantic_exec)
        
        assert semantic_post['post_status'] == 'success'
        assert 'semantic_analysis_result' in integration_store
        
        # Execute vector search node
        vector_prep = vector_node.prep(integration_store)
        vector_exec = vector_node.exec(integration_store, vector_prep)
        vector_post = vector_node.post(integration_store, vector_prep, vector_exec)
        
        assert vector_post['post_status'] == 'success'
        assert 'vector_search_result' in integration_store
        
        # Execute enhanced timestamp node
        timestamp_prep = timestamp_node.prep(integration_store)
        timestamp_exec = timestamp_node.exec(integration_store, timestamp_prep)
        timestamp_post = timestamp_node.post(integration_store, timestamp_prep, timestamp_exec)
        
        assert timestamp_post['post_status'] == 'success'
        assert 'timestamps' in integration_store

    def test_error_propagation(self, semantic_node, vector_node, timestamp_node):
        """Test how errors propagate through the node chain."""
        # Start with invalid store
        invalid_store = Store()
        
        # Semantic node should fail
        semantic_prep = semantic_node.prep(invalid_store)
        assert semantic_prep['status'] == 'failed'
        
        # Vector node should fail without semantic data
        vector_prep = vector_node.prep(invalid_store)
        assert vector_prep['status'] == 'failed'
        
        # Timestamp node should still work with minimal data
        invalid_store['transcript_data'] = {'raw_transcript': []}
        invalid_store['video_id'] = 'test'
        timestamp_prep = timestamp_node.prep(invalid_store)
        assert timestamp_prep['status'] == 'success'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])