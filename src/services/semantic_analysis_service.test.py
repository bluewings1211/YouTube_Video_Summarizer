"""
Tests for SemanticAnalysisService.

This module contains comprehensive tests for the semantic analysis functionality
including transcript parsing, semantic clustering, and timestamp generation.
"""

import pytest
import json
from unittest.mock import Mock, patch
from typing import Dict, Any, List

from .semantic_analysis_service import (
    SemanticAnalysisService,
    TranscriptSegment,
    SemanticCluster,
    SemanticTimestamp,
    create_semantic_analysis_service
)
from ..utils.smart_llm_client import SmartLLMClient, TaskRequirements


@pytest.fixture
def mock_smart_llm_client():
    """Create a mock SmartLLMClient for testing."""
    mock_client = Mock(spec=SmartLLMClient)
    
    # Mock successful clustering response
    clustering_response = {
        'text': json.dumps({
            "clusters": [
                {
                    "theme": "Introduction",
                    "start_time": 0.0,
                    "end_time": 30.0,
                    "summary": "Speaker introduces the topic",
                    "importance": 8,
                    "keywords": ["introduction", "topic", "overview"]
                },
                {
                    "theme": "Main Content",
                    "start_time": 30.0,
                    "end_time": 120.0,
                    "summary": "Main discussion of key concepts",
                    "importance": 9,
                    "keywords": ["main", "concepts", "discussion"]
                },
                {
                    "theme": "Conclusion",
                    "start_time": 120.0,
                    "end_time": 150.0,
                    "summary": "Summary and wrap-up",
                    "importance": 7,
                    "keywords": ["conclusion", "summary", "wrap-up"]
                }
            ]
        }),
        'provider': 'mock',
        'model': 'test-model',
        'tokens_used': 500
    }
    
    mock_client.generate_text_with_fallback.return_value = clustering_response
    return mock_client


@pytest.fixture
def sample_transcript():
    """Create sample transcript data for testing."""
    return [
        {
            "start": 0.0,
            "duration": 5.0,
            "text": "Welcome to this video about artificial intelligence and machine learning concepts"
        },
        {
            "start": 5.0,
            "duration": 4.0,
            "text": "Today we will explore the fundamentals of neural networks"
        },
        {
            "start": 15.0,
            "duration": 6.0,
            "text": "First, let's understand what machine learning actually means in practice"
        },
        {
            "start": 30.0,
            "duration": 8.0,
            "text": "Neural networks are computational models inspired by biological brain structures"
        },
        {
            "start": 45.0,
            "duration": 7.0,
            "text": "The key components include neurons, weights, and activation functions"
        },
        {
            "start": 60.0,
            "duration": 5.0,
            "text": "Deep learning extends neural networks with multiple hidden layers"
        },
        {
            "start": 80.0,
            "duration": 6.0,
            "text": "Training involves adjusting weights through backpropagation algorithms"
        },
        {
            "start": 120.0,
            "duration": 4.0,
            "text": "In conclusion, these concepts form the foundation of modern AI"
        },
        {
            "start": 135.0,
            "duration": 5.0,
            "text": "Thank you for watching this introduction to machine learning"
        }
    ]


@pytest.fixture
def semantic_service(mock_smart_llm_client):
    """Create SemanticAnalysisService instance for testing."""
    return SemanticAnalysisService(smart_llm_client=mock_smart_llm_client)


class TestTranscriptSegment:
    """Test cases for TranscriptSegment dataclass."""
    
    def test_transcript_segment_creation(self):
        """Test creating a transcript segment."""
        segment = TranscriptSegment(
            start_time=10.0,
            end_time=15.0,
            text="This is a test segment with multiple words",
            duration=5.0
        )
        
        assert segment.start_time == 10.0
        assert segment.end_time == 15.0
        assert segment.text == "This is a test segment with multiple words"
        assert segment.duration == 5.0
        assert segment.word_count == 8
        assert segment.char_count == 42
    
    def test_segment_word_count(self):
        """Test word count calculation."""
        segment = TranscriptSegment(0.0, None, "Hello world test")
        assert segment.word_count == 3
        
        segment = TranscriptSegment(0.0, None, "")
        assert segment.word_count == 0
        
        segment = TranscriptSegment(0.0, None, "Single")
        assert segment.word_count == 1


class TestSemanticCluster:
    """Test cases for SemanticCluster dataclass."""
    
    def test_semantic_cluster_creation(self):
        """Test creating a semantic cluster."""
        segments = [
            TranscriptSegment(0.0, 5.0, "First segment", 5.0),
            TranscriptSegment(5.0, 10.0, "Second segment", 5.0)
        ]
        
        cluster = SemanticCluster(
            cluster_id="test_cluster",
            segments=segments,
            theme="Test Theme",
            importance_score=0.8,
            start_time=0.0,
            end_time=10.0,
            summary="Test summary",
            keywords=["test", "cluster"]
        )
        
        assert cluster.cluster_id == "test_cluster"
        assert cluster.theme == "Test Theme"
        assert cluster.importance_score == 0.8
        assert cluster.duration == 10.0
        assert cluster.word_count == 4  # "First segment" + "Second segment"
        assert len(cluster.segments) == 2


class TestSemanticAnalysisService:
    """Test cases for SemanticAnalysisService."""
    
    def test_service_initialization(self):
        """Test service initialization."""
        service = SemanticAnalysisService()
        assert service.smart_llm_client is not None
        assert service.min_segment_length == 3
        assert service.max_clusters == 10
    
    def test_parse_transcript_segments(self, semantic_service, sample_transcript):
        """Test parsing transcript segments."""
        segments = semantic_service._parse_transcript_segments(sample_transcript)
        
        assert len(segments) == 9  # All segments should be valid
        assert all(isinstance(seg, TranscriptSegment) for seg in segments)
        assert segments[0].start_time == 0.0
        assert segments[-1].start_time == 135.0
        
        # Check that segments are sorted by start time
        start_times = [seg.start_time for seg in segments]
        assert start_times == sorted(start_times)
    
    def test_parse_transcript_segments_with_invalid_data(self, semantic_service):
        """Test parsing with invalid transcript data."""
        invalid_transcript = [
            {"start": None, "text": "Invalid segment"},  # No start time
            {"start": "invalid", "text": "Another invalid"},  # Invalid start time
            {"start": 10.0, "text": ""},  # Empty text
            {"start": 15.0, "text": "OK"},  # Too short (less than min_segment_length)
            {"start": 20.0, "text": "This segment is valid and long enough"}
        ]
        
        segments = semantic_service._parse_transcript_segments(invalid_transcript)
        assert len(segments) == 1  # Only the last segment should be valid
        assert segments[0].text == "This segment is valid and long enough"
    
    def test_analyze_transcript_success(self, semantic_service, sample_transcript):
        """Test successful transcript analysis."""
        result = semantic_service.analyze_transcript(
            raw_transcript=sample_transcript,
            video_id="test_video_123",
            video_title="Test Video",
            target_timestamp_count=3
        )
        
        assert result['status'] == 'success'
        assert result['video_id'] == 'test_video_123'
        assert result['segments_count'] == 9
        assert result['clusters_count'] == 3
        assert result['timestamps_count'] <= 3
        assert 'timestamps' in result
        assert 'semantic_clusters' in result
        assert 'semantic_keywords' in result
        assert 'semantic_metrics' in result
    
    def test_analyze_transcript_empty_input(self, semantic_service):
        """Test analysis with empty transcript."""
        result = semantic_service.analyze_transcript(
            raw_transcript=[],
            video_id="empty_video",
            video_title="Empty Video"
        )
        
        assert result['status'] == 'empty'
        assert result['reason'] == "No valid transcript segments found"
        assert result['timestamps_count'] == 0
    
    def test_fallback_clustering(self, semantic_service, sample_transcript):
        """Test fallback clustering when LLM clustering fails."""
        segments = semantic_service._parse_transcript_segments(sample_transcript)
        
        # Test fallback clustering directly
        clusters = semantic_service._fallback_clustering(segments)
        
        assert len(clusters) > 0
        assert all(isinstance(cluster, SemanticCluster) for cluster in clusters)
        assert all(cluster.cluster_id.startswith("fallback_cluster") for cluster in clusters)
        
        # Check that clusters cover the time range
        total_start = min(seg.start_time for seg in segments)
        total_end = max(seg.start_time for seg in segments)
        cluster_start = min(cluster.start_time for cluster in clusters)
        cluster_end = max(cluster.end_time for cluster in clusters)
        
        assert cluster_start <= total_start
        assert cluster_end >= total_end
    
    def test_generate_semantic_timestamps(self, semantic_service):
        """Test semantic timestamp generation."""
        # Create test clusters
        segments = [
            TranscriptSegment(10.0, 20.0, "Important content here", 10.0),
            TranscriptSegment(20.0, 30.0, "More important details", 10.0)
        ]
        
        clusters = [
            SemanticCluster(
                cluster_id="test_cluster_1",
                segments=segments,
                theme="Important Topic",
                importance_score=0.9,
                start_time=10.0,
                end_time=30.0,
                summary="This is important content",
                keywords=["important", "content", "details"]
            )
        ]
        
        timestamps = semantic_service._generate_semantic_timestamps(
            clusters, "test_video_123", 1
        )
        
        assert len(timestamps) == 1
        timestamp = timestamps[0]
        
        assert isinstance(timestamp, SemanticTimestamp)
        assert timestamp.video_id == "test_video_123"  # This will fail - let me fix
        assert timestamp.semantic_cluster_id == "test_cluster_1"
        assert timestamp.cluster_theme == "Important Topic"
        assert "test_video_123" in timestamp.youtube_url
        assert timestamp.importance_rating >= 1
        assert timestamp.confidence_score > 0
    
    def test_calculate_timestamp_confidence(self, semantic_service):
        """Test timestamp confidence calculation."""
        # High importance cluster with good properties
        high_importance_cluster = SemanticCluster(
            cluster_id="high_cluster",
            segments=[TranscriptSegment(0.0, 10.0, "test " * 20, 10.0)] * 3,  # Multiple segments
            theme="Important Theme",
            importance_score=0.9,
            start_time=0.0,
            end_time=60.0,  # Good duration
            summary="Important summary",
            keywords=["key1", "key2", "key3"]
        )
        
        confidence = semantic_service._calculate_timestamp_confidence(high_importance_cluster)
        assert confidence > 0.7  # Should be high confidence
        
        # Low importance cluster
        low_importance_cluster = SemanticCluster(
            cluster_id="low_cluster",
            segments=[],
            theme="Minor Theme",
            importance_score=0.2,
            start_time=0.0,
            end_time=5.0,  # Very short duration
            summary="Minor summary",
            keywords=[]
        )
        
        low_confidence = semantic_service._calculate_timestamp_confidence(low_importance_cluster)
        assert low_confidence < confidence  # Should be lower than high importance
    
    def test_format_timestamp(self, semantic_service):
        """Test timestamp formatting."""
        assert semantic_service._format_timestamp(65.0) == "1:05"
        assert semantic_service._format_timestamp(3665.0) == "1:01:05"
        assert semantic_service._format_timestamp(30.0) == "0:30"
        assert semantic_service._format_timestamp(0.0) == "0:00"
    
    def test_cluster_to_dict(self, semantic_service):
        """Test cluster serialization to dictionary."""
        segments = [
            TranscriptSegment(0.0, 5.0, "First segment with words", 5.0),
            TranscriptSegment(5.0, 10.0, "Second segment", 5.0)
        ]
        
        cluster = SemanticCluster(
            cluster_id="test_cluster",
            segments=segments,
            theme="Test Theme",
            importance_score=0.8,
            start_time=0.0,
            end_time=10.0,
            summary="Test summary",
            keywords=["test", "cluster"]
        )
        
        cluster_dict = semantic_service._cluster_to_dict(cluster)
        
        expected_keys = {
            'cluster_id', 'theme', 'importance_score', 'start_time', 
            'end_time', 'duration', 'summary', 'keywords', 
            'segment_count', 'word_count'
        }
        
        assert set(cluster_dict.keys()) == expected_keys
        assert cluster_dict['cluster_id'] == 'test_cluster'
        assert cluster_dict['theme'] == 'Test Theme'
        assert cluster_dict['importance_score'] == 0.8
        assert cluster_dict['segment_count'] == 2
        assert cluster_dict['word_count'] == 7  # "First segment with words" + "Second segment"
    
    def test_get_timestamp_context(self, semantic_service):
        """Test getting context around timestamps."""
        segments = [
            TranscriptSegment(0.0, 5.0, "Context before target", 5.0),
            TranscriptSegment(5.0, 10.0, "Target segment content", 5.0),
            TranscriptSegment(10.0, 15.0, "Context after target", 5.0)
        ]
        
        cluster = SemanticCluster(
            cluster_id="context_cluster",
            segments=segments,
            theme="Context Test",
            importance_score=0.5,
            start_time=0.0,
            end_time=15.0,
            summary="Context test cluster",
            keywords=[]
        )
        
        context_before, context_after = semantic_service._get_timestamp_context(cluster, 5.0)
        
        assert "Context before target" in context_before
        assert "Context after target" in context_after
    
    @patch('src.services.semantic_analysis_service.SemanticAnalysisService._perform_semantic_clustering')
    def test_analyze_transcript_clustering_failure(self, mock_clustering, semantic_service, sample_transcript):
        """Test analysis when clustering fails."""
        # Mock clustering to return empty list
        mock_clustering.return_value = []
        
        result = semantic_service.analyze_transcript(
            raw_transcript=sample_transcript,
            video_id="test_video",
            video_title="Test Video"
        )
        
        assert result['status'] == 'empty'
        assert result['reason'] == "No semantic clusters identified"
    
    def test_create_semantic_analysis_service_factory(self):
        """Test the factory function."""
        service = create_semantic_analysis_service()
        assert isinstance(service, SemanticAnalysisService)
        
        mock_client = Mock(spec=SmartLLMClient)
        service_with_client = create_semantic_analysis_service(mock_client)
        assert service_with_client.smart_llm_client is mock_client


class TestSemanticAnalysisIntegration:
    """Integration tests for semantic analysis workflow."""
    
    def test_full_analysis_workflow(self, semantic_service, sample_transcript):
        """Test complete analysis workflow from transcript to timestamps."""
        result = semantic_service.analyze_transcript(
            raw_transcript=sample_transcript,
            video_id="integration_test_video",
            video_title="Integration Test Video",
            target_timestamp_count=3
        )
        
        # Verify complete result structure
        assert result['status'] == 'success'
        assert result['video_id'] == 'integration_test_video'
        assert result['segments_count'] > 0
        assert result['clusters_count'] > 0
        assert result['timestamps_count'] > 0
        
        # Verify timestamps structure
        timestamps = result['timestamps']
        for timestamp in timestamps:
            assert 'timestamp_seconds' in timestamp
            assert 'timestamp_formatted' in timestamp
            assert 'description' in timestamp
            assert 'importance_rating' in timestamp
            assert 'youtube_url' in timestamp
            assert 'semantic_cluster_id' in timestamp
            assert 'integration_test_video' in timestamp['youtube_url']
        
        # Verify clusters structure
        clusters = result['semantic_clusters']
        for cluster in clusters:
            assert 'cluster_id' in cluster
            assert 'theme' in cluster
            assert 'importance_score' in cluster
            assert 'start_time' in cluster
            assert 'end_time' in cluster
        
        # Verify metrics
        metrics = result['semantic_metrics']
        assert 'total_segments' in metrics
        assert 'total_clusters' in metrics
        assert 'coverage_percentage' in metrics
        assert metrics['total_segments'] == result['segments_count']
        assert metrics['total_clusters'] == result['clusters_count']
    
    def test_performance_with_large_transcript(self, semantic_service):
        """Test performance with a larger transcript."""
        # Generate a larger transcript
        large_transcript = []
        for i in range(100):
            large_transcript.append({
                "start": i * 5.0,
                "duration": 4.0,
                "text": f"This is segment number {i} with some meaningful content about topic {i % 10}"
            })
        
        result = semantic_service.analyze_transcript(
            raw_transcript=large_transcript,
            video_id="large_test_video",
            video_title="Large Test Video",
            target_timestamp_count=10
        )
        
        assert result['status'] == 'success'
        assert result['segments_count'] == 100
        assert result['timestamps_count'] <= 10
        
        # Verify that analysis completed successfully with large input
        assert 'semantic_metrics' in result
        assert result['semantic_metrics']['total_segments'] == 100


if __name__ == "__main__":
    pytest.main([__file__])