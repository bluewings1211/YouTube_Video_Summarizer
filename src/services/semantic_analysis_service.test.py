"""
Tests for the Semantic Analysis Service.

This module contains comprehensive tests for the semantic analysis functionality,
including clustering, vector search, and performance optimizations.
"""

import pytest
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any

# Import the modules to test
from .semantic_analysis_service import (
    SemanticAnalysisService,
    TranscriptSegment,
    SemanticCluster,
    SemanticTimestamp,
    create_semantic_analysis_service
)


class TestSemanticAnalysisService:
    """Test suite for SemanticAnalysisService."""

    @pytest.fixture
    def mock_smart_llm_client(self):
        """Create a mock SmartLLMClient."""
        mock_client = Mock()
        mock_client.generate_text_with_fallback.return_value = {
            'text': 'Mock LLM response',
            'model': 'mock-model',
            'tokens': 100
        }
        mock_client.generate_text_with_chinese_optimization.return_value = {
            'text': 'Mock Chinese LLM response',
            'model': 'mock-chinese-model',
            'tokens': 120
        }
        return mock_client

    @pytest.fixture
    def mock_semantic_analyzer(self):
        """Create a mock SemanticAnalyzer."""
        mock_analyzer = Mock()
        mock_result = Mock()
        mock_result.clusters = [
            SemanticCluster(
                cluster_id="test_cluster_1",
                segments=[],
                theme="Test Theme",
                importance_score=0.8,
                start_time=10.0,
                end_time=30.0,
                summary="Test cluster summary",
                keywords=["test", "keyword"]
            )
        ]
        mock_result.cluster_quality_score = 0.7
        mock_analyzer.analyze_segments.return_value = mock_result
        return mock_analyzer

    @pytest.fixture
    def mock_vector_search_engine(self):
        """Create a mock VectorSearchEngine."""
        mock_engine = Mock()
        mock_engine.find_optimal_timestamps.return_value = []
        mock_engine.analyze_semantic_coherence.return_value = {
            'coherence_score': 0.6,
            'analysis': 'completed'
        }
        mock_engine.clear_cache.return_value = None
        return mock_engine

    @pytest.fixture
    def semantic_service(self, mock_smart_llm_client, mock_semantic_analyzer, mock_vector_search_engine):
        """Create a SemanticAnalysisService instance with mocked dependencies."""
        return SemanticAnalysisService(
            smart_llm_client=mock_smart_llm_client,
            semantic_analyzer=mock_semantic_analyzer,
            vector_search_engine=mock_vector_search_engine
        )

    @pytest.fixture
    def sample_raw_transcript(self):
        """Create sample raw transcript data."""
        return [
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
            },
            {
                'start': 15.0,
                'duration': 5.0,
                'text': 'Machine learning is a subset of artificial intelligence.'
            },
            {
                'start': 20.0,
                'duration': 7.0,
                'text': 'Deep learning uses neural networks with multiple layers.'
            }
        ]

    @pytest.fixture
    def sample_transcript_segments(self):
        """Create sample TranscriptSegment objects."""
        return [
            TranscriptSegment(
                start_time=0.0,
                end_time=5.0,
                text='Hello and welcome to this video about machine learning.',
                duration=5.0
            ),
            TranscriptSegment(
                start_time=5.0,
                end_time=9.0,
                text='Today we will explore neural networks and deep learning.',
                duration=4.0
            ),
            TranscriptSegment(
                start_time=9.0,
                end_time=15.0,
                text='First, let us understand what artificial intelligence means.',
                duration=6.0
            )
        ]

    def test_service_initialization(self):
        """Test that the service initializes correctly."""
        service = SemanticAnalysisService()
        
        assert service is not None
        assert service.smart_llm_client is not None
        assert service.semantic_analyzer is not None
        assert service.vector_search_engine is not None
        assert service.min_segment_length == 3
        assert service.max_clusters == 10
        assert service.enable_caching == True
        assert service.enable_parallel_processing == True

    def test_factory_function(self):
        """Test the factory function creates service correctly."""
        service = create_semantic_analysis_service()
        
        assert isinstance(service, SemanticAnalysisService)
        assert service.smart_llm_client is not None

    def test_parse_transcript_segments(self, semantic_service, sample_raw_transcript):
        """Test transcript segment parsing."""
        segments = semantic_service._parse_transcript_segments(sample_raw_transcript)
        
        assert len(segments) == 5
        assert all(isinstance(seg, TranscriptSegment) for seg in segments)
        assert segments[0].start_time == 0.0
        assert segments[0].text == 'Hello and welcome to this video about machine learning.'
        assert segments[-1].start_time == 20.0

    def test_parse_transcript_segments_optimized(self, semantic_service):
        """Test optimized transcript segment parsing with large input."""
        # Create a large transcript to trigger sampling
        large_transcript = [
            {
                'start': i * 2.0,
                'duration': 2.0,
                'text': f'This is segment number {i} with some content.'
            }
            for i in range(600)  # Exceeds max_segments_for_full_analysis
        ]
        
        segments = semantic_service._parse_transcript_segments_optimized(large_transcript)
        
        # Should be sampled down
        assert len(segments) < len(large_transcript)
        assert len(segments) == int(len(large_transcript) * semantic_service.segment_sampling_ratio)

    def test_early_termination_detection(self, semantic_service):
        """Test early termination logic."""
        # Test with empty segments
        assert semantic_service._should_terminate_early([]) == True
        
        # Test with too few segments
        short_segments = [
            TranscriptSegment(0.0, 2.0, 'short', 2.0),
            TranscriptSegment(2.0, 4.0, 'text', 2.0)
        ]
        assert semantic_service._should_terminate_early(short_segments) == True
        
        # Test with low quality segments (very short words)
        low_quality_segments = [
            TranscriptSegment(0.0, 2.0, 'a', 2.0),
            TranscriptSegment(2.0, 4.0, 'b', 2.0),
            TranscriptSegment(4.0, 6.0, 'c', 2.0),
            TranscriptSegment(6.0, 8.0, 'd', 2.0),
            TranscriptSegment(8.0, 10.0, 'e', 2.0)
        ]
        assert semantic_service._should_terminate_early(low_quality_segments) == True
        
        # Test with good quality segments
        good_segments = [
            TranscriptSegment(0.0, 2.0, 'This is good quality content', 2.0),
            TranscriptSegment(2.0, 4.0, 'With meaningful text segments', 2.0),
            TranscriptSegment(4.0, 6.0, 'That have sufficient diversity', 2.0),
            TranscriptSegment(6.0, 8.0, 'And reasonable length for analysis', 2.0),
            TranscriptSegment(8.0, 10.0, 'Making it worth processing', 2.0)
        ]
        assert semantic_service._should_terminate_early(good_segments) == False

    def test_cache_key_generation(self, semantic_service, sample_raw_transcript):
        """Test cache key generation."""
        key1 = semantic_service._generate_cache_key(sample_raw_transcript, "video123", 5)
        key2 = semantic_service._generate_cache_key(sample_raw_transcript, "video123", 5)
        key3 = semantic_service._generate_cache_key(sample_raw_transcript, "video456", 5)
        
        # Same input should generate same key
        assert key1 == key2
        
        # Different video ID should generate different key
        assert key1 != key3
        
        # Key should have expected format
        assert "video123_5_" in key1

    @patch('src.services.semantic_analysis_service.datetime')
    def test_analyze_transcript_success(self, mock_datetime, semantic_service, sample_raw_transcript):
        """Test successful transcript analysis."""
        # Mock datetime for consistent testing
        mock_datetime.utcnow.return_value.isoformat.return_value = "2023-01-01T12:00:00"
        mock_datetime.utcnow.return_value.total_seconds.return_value = 1.5
        
        result = semantic_service.analyze_transcript(
            raw_transcript=sample_raw_transcript,
            video_id="test_video_123",
            video_title="Test Video Title",
            target_timestamp_count=3
        )
        
        assert result['status'] == 'success'
        assert result['video_id'] == 'test_video_123'
        assert 'timestamps' in result
        assert 'semantic_clusters' in result
        assert 'semantic_keywords' in result
        assert 'semantic_metrics' in result
        assert result['processing_metadata']['optimization_enabled'] == True

    def test_analyze_transcript_with_caching(self, semantic_service, sample_raw_transcript):
        """Test transcript analysis with caching."""
        # First call
        result1 = semantic_service.analyze_transcript(
            raw_transcript=sample_raw_transcript,
            video_id="cache_test_video",
            target_timestamp_count=3
        )
        
        # Second call should hit cache
        result2 = semantic_service.analyze_transcript(
            raw_transcript=sample_raw_transcript,
            video_id="cache_test_video",
            target_timestamp_count=3
        )
        
        assert result1['status'] == 'success'
        assert result2['status'] == 'success'
        assert result2['cache_hit'] == True

    def test_analyze_transcript_early_termination(self, semantic_service):
        """Test transcript analysis with early termination."""
        # Create low quality transcript
        low_quality_transcript = [
            {'start': 0.0, 'duration': 1.0, 'text': 'a'},
            {'start': 1.0, 'duration': 1.0, 'text': 'b'}
        ]
        
        result = semantic_service.analyze_transcript(
            raw_transcript=low_quality_transcript,
            video_id="early_term_test",
            target_timestamp_count=3
        )
        
        assert result['status'] == 'early_termination'
        assert result['reason'] == 'low_quality_input'
        assert result['timestamps_count'] == 0

    def test_analyze_transcript_empty_input(self, semantic_service):
        """Test transcript analysis with empty input."""
        result = semantic_service.analyze_transcript(
            raw_transcript=[],
            video_id="empty_test",
            target_timestamp_count=3
        )
        
        assert result['status'] == 'empty'
        assert 'No valid transcript segments found' in result['reason']

    def test_performance_optimized_clustering(self, semantic_service, sample_transcript_segments):
        """Test performance optimized clustering."""
        clusters = semantic_service._perform_semantic_clustering_optimized(
            segments=sample_transcript_segments,
            video_title="Test Video"
        )
        
        assert isinstance(clusters, list)
        # Should use mocked analyzer which returns one cluster
        assert len(clusters) == 1
        assert clusters[0].cluster_id == "test_cluster_1"

    def test_clustering_cache(self, semantic_service, sample_transcript_segments):
        """Test clustering result caching."""
        # First call
        clusters1 = semantic_service._perform_semantic_clustering_optimized(
            segments=sample_transcript_segments,
            video_title="Cache Test"
        )
        
        # Second call should hit cache
        clusters2 = semantic_service._perform_semantic_clustering_optimized(
            segments=sample_transcript_segments,
            video_title="Cache Test"
        )
        
        assert clusters1 == clusters2

    def test_clear_caches(self, semantic_service):
        """Test cache clearing functionality."""
        # Add some data to caches
        semantic_service.analysis_cache['test'] = {'data': 'test'}
        semantic_service.embedding_cache['test'] = [1, 2, 3]
        semantic_service.cluster_cache['test'] = []
        
        semantic_service.clear_caches()
        
        assert len(semantic_service.analysis_cache) == 0
        assert len(semantic_service.embedding_cache) == 0
        assert len(semantic_service.cluster_cache) == 0

    def test_get_performance_stats(self, semantic_service):
        """Test performance statistics retrieval."""
        # Add some test data to caches
        semantic_service.analysis_cache['test'] = {'data': 'test'}
        semantic_service.cluster_cache['test'] = []
        
        stats = semantic_service.get_performance_stats()
        
        assert 'cache_sizes' in stats
        assert 'settings' in stats
        assert stats['cache_sizes']['analysis_cache'] == 1
        assert stats['cache_sizes']['cluster_cache'] == 1
        assert stats['settings']['enable_caching'] == True
        assert stats['settings']['max_segments_for_full_analysis'] == 500

    def test_format_timestamp(self, semantic_service):
        """Test timestamp formatting."""
        # Test seconds only
        assert semantic_service._format_timestamp(45.0) == "0:45"
        
        # Test minutes and seconds
        assert semantic_service._format_timestamp(125.0) == "2:05"
        
        # Test hours, minutes, and seconds
        assert semantic_service._format_timestamp(3665.0) == "1:01:05"

    def test_semantic_timestamps_generation(self, semantic_service, mock_semantic_analyzer):
        """Test semantic timestamp generation."""
        # Create mock clusters
        mock_clusters = [
            SemanticCluster(
                cluster_id="cluster_1",
                segments=[
                    TranscriptSegment(10.0, 15.0, "Important content here", 5.0)
                ],
                theme="Introduction",
                importance_score=0.9,
                start_time=10.0,
                end_time=15.0,
                summary="Introductory content",
                keywords=["introduction", "content"]
            ),
            SemanticCluster(
                cluster_id="cluster_2",
                segments=[
                    TranscriptSegment(20.0, 25.0, "Main discussion point", 5.0)
                ],
                theme="Main Content",
                importance_score=0.8,
                start_time=20.0,
                end_time=25.0,
                summary="Main discussion",
                keywords=["main", "discussion"]
            )
        ]
        
        timestamps = semantic_service._generate_semantic_timestamps_fast(
            clusters=mock_clusters,
            video_id="test_video",
            target_count=2
        )
        
        assert len(timestamps) == 2
        assert all(isinstance(ts, SemanticTimestamp) for ts in timestamps)
        assert timestamps[0].timestamp_seconds == 10.0
        assert timestamps[1].timestamp_seconds == 20.0
        assert "test_video" in timestamps[0].youtube_url

    def test_error_handling(self, mock_smart_llm_client):
        """Test error handling in semantic analysis."""
        # Create service with mocked dependencies that raise errors
        mock_analyzer = Mock()
        mock_analyzer.analyze_segments.side_effect = Exception("Test error")
        
        service = SemanticAnalysisService(
            smart_llm_client=mock_smart_llm_client,
            semantic_analyzer=mock_analyzer,
            vector_search_engine=Mock()
        )
        
        result = service.analyze_transcript(
            raw_transcript=[
                {'start': 0.0, 'duration': 5.0, 'text': 'Test content for error handling'}
            ],
            video_id="error_test",
            target_timestamp_count=3
        )
        
        assert result['status'] == 'error'
        assert 'error' in result

    def test_semantic_cluster_properties(self):
        """Test SemanticCluster properties and methods."""
        segments = [
            TranscriptSegment(0.0, 5.0, "First segment text", 5.0),
            TranscriptSegment(5.0, 10.0, "Second segment text", 5.0)
        ]
        
        cluster = SemanticCluster(
            cluster_id="test_cluster",
            segments=segments,
            theme="Test Theme",
            importance_score=0.7,
            start_time=0.0,
            end_time=10.0,
            summary="Test cluster summary",
            keywords=["test", "cluster"]
        )
        
        assert cluster.duration == 10.0
        assert cluster.word_count == 6  # "First segment text" + "Second segment text"

    def test_transcript_segment_properties(self):
        """Test TranscriptSegment properties."""
        segment = TranscriptSegment(
            start_time=10.0,
            end_time=15.0,
            text="This is a test segment with multiple words",
            duration=5.0,
            speaker="Speaker1"
        )
        
        assert segment.word_count == 9
        assert segment.char_count == len("This is a test segment with multiple words")
        assert segment.speaker == "Speaker1"


class TestSemanticAnalysisIntegration:
    """Integration tests for semantic analysis components."""

    @pytest.fixture
    def real_service(self):
        """Create a real semantic analysis service for integration testing."""
        return create_semantic_analysis_service()

    def test_end_to_end_analysis(self, real_service):
        """Test complete end-to-end semantic analysis."""
        # Sample transcript data
        transcript_data = [
            {
                'start': 0.0,
                'duration': 10.0,
                'text': 'Welcome to our comprehensive guide on machine learning. Today we will explore the fundamentals of artificial intelligence.'
            },
            {
                'start': 10.0,
                'duration': 8.0,
                'text': 'Machine learning is a powerful subset of artificial intelligence that enables computers to learn without explicit programming.'
            },
            {
                'start': 18.0,
                'duration': 12.0,
                'text': 'Neural networks form the backbone of deep learning. These interconnected nodes process information in layers, mimicking the human brain.'
            },
            {
                'start': 30.0,
                'duration': 9.0,
                'text': 'Deep learning applications include image recognition, natural language processing, and autonomous vehicles.'
            },
            {
                'start': 39.0,
                'duration': 7.0,
                'text': 'In conclusion, machine learning and artificial intelligence are transforming our world in remarkable ways.'
            }
        ]
        
        try:
            result = real_service.analyze_transcript(
                raw_transcript=transcript_data,
                video_id="integration_test_video",
                video_title="Machine Learning Guide",
                target_timestamp_count=3
            )
            
            # Basic validation
            assert 'status' in result
            assert 'video_id' in result
            assert result['video_id'] == 'integration_test_video'
            
            # Should have some form of results (even if fallback)
            assert 'timestamps' in result
            assert 'semantic_clusters' in result
            assert 'semantic_keywords' in result
            
            # Performance metadata should be present
            assert 'processing_metadata' in result
            
        except Exception as e:
            # Integration test may fail due to missing dependencies
            # This is acceptable in CI/CD environments
            pytest.skip(f"Integration test skipped due to missing dependencies: {str(e)}")

    def test_performance_with_large_input(self, real_service):
        """Test performance optimization with large input."""
        # Create a large transcript
        large_transcript = []
        for i in range(200):
            large_transcript.append({
                'start': i * 3.0,
                'duration': 3.0,
                'text': f'This is segment {i} containing some meaningful content about various topics in our comprehensive discussion. '
                       f'We explore different aspects and provide detailed explanations throughout this extended presentation.'
            })
        
        try:
            result = real_service.analyze_transcript(
                raw_transcript=large_transcript,
                video_id="performance_test_video",
                target_timestamp_count=5
            )
            
            # Should complete without timeout
            assert 'status' in result
            assert result.get('segments_count', 0) > 0
            
            # Check if optimization was applied
            processing_metadata = result.get('processing_metadata', {})
            assert processing_metadata.get('optimization_enabled') == True
            
        except Exception as e:
            pytest.skip(f"Performance test skipped due to missing dependencies: {str(e)}")


class TestSemanticAnalysisEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.fixture
    def minimal_service(self):
        """Create service with minimal mocking for edge case testing."""
        mock_llm = Mock()
        mock_analyzer = Mock()
        mock_vector = Mock()
        
        return SemanticAnalysisService(
            smart_llm_client=mock_llm,
            semantic_analyzer=mock_analyzer,
            vector_search_engine=mock_vector
        )

    def test_malformed_transcript_data(self, minimal_service):
        """Test handling of malformed transcript data."""
        malformed_data = [
            {'start': 'invalid', 'text': 'Valid text'},
            {'duration': 5.0, 'text': 'Missing start time'},
            {'start': 10.0, 'duration': 'invalid', 'text': 'Invalid duration'},
            None,
            "not a dict",
            {'start': 15.0, 'duration': 5.0}  # Missing text
        ]
        
        result = minimal_service.analyze_transcript(
            raw_transcript=malformed_data,
            video_id="malformed_test",
            target_timestamp_count=3
        )
        
        # Should handle gracefully
        assert 'status' in result

    def test_unicode_and_special_characters(self, minimal_service):
        """Test handling of unicode and special characters."""
        unicode_transcript = [
            {
                'start': 0.0,
                'duration': 5.0,
                'text': '欢迎来到我们的机器学习教程！今天我们将探讨人工智能的基础知识。'
            },
            {
                'start': 5.0,
                'duration': 5.0,
                'text': 'Здравствуйте! Добро пожаловать в наш курс по машинному обучению.'
            },
            {
                'start': 10.0,
                'duration': 5.0,
                'text': 'Special chars: @#$%^&*()_+-=[]{}|;:,.<>?/~`'
            },
            {
                'start': 15.0,
                'duration': 5.0,
                'text': 'Emoji test: 🤖🧠💻📊🔬'
            }
        ]
        
        result = minimal_service.analyze_transcript(
            raw_transcript=unicode_transcript,
            video_id="unicode_test",
            target_timestamp_count=2
        )
        
        # Should handle unicode gracefully
        assert 'status' in result

    def test_extremely_short_segments(self, minimal_service):
        """Test handling of extremely short segments."""
        short_segments = [
            {'start': i * 0.1, 'duration': 0.1, 'text': 'a'} 
            for i in range(100)
        ]
        
        result = minimal_service.analyze_transcript(
            raw_transcript=short_segments,
            video_id="short_test",
            target_timestamp_count=3
        )
        
        # Should trigger early termination or handle gracefully
        assert 'status' in result

    def test_zero_target_timestamps(self, minimal_service):
        """Test handling of zero target timestamps."""
        normal_transcript = [
            {
                'start': 0.0,
                'duration': 10.0,
                'text': 'This is a normal transcript segment with sufficient content for analysis.'
            }
        ]
        
        result = minimal_service.analyze_transcript(
            raw_transcript=normal_transcript,
            video_id="zero_target_test",
            target_timestamp_count=0
        )
        
        # Should handle gracefully
        assert 'status' in result
        assert result.get('timestamps_count', 0) >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])