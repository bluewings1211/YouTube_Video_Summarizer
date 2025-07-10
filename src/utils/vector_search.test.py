"""
Tests for Vector Search utility.

This module contains comprehensive tests for embedding-based vector search
and semantic similarity functionality.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import List
from datetime import datetime

from .vector_search import (
    VectorSearchEngine,
    VectorSearchResult,
    EmbeddingIndex,
    create_vector_search_engine
)
from ..services.semantic_analysis_service import TranscriptSegment, SemanticCluster, SemanticTimestamp


@pytest.fixture
def sample_segments():
    """Create sample transcript segments for testing."""
    return [
        TranscriptSegment(0.0, 10.0, "Introduction to machine learning concepts and fundamentals", 10.0),
        TranscriptSegment(10.0, 20.0, "Neural networks are computational models inspired by biological brains", 10.0),
        TranscriptSegment(20.0, 30.0, "Deep learning extends neural networks with multiple layers", 10.0),
        TranscriptSegment(30.0, 40.0, "Training algorithms like backpropagation adjust network weights", 10.0),
        TranscriptSegment(40.0, 50.0, "Applications include image recognition and natural language processing", 10.0),
        TranscriptSegment(50.0, 60.0, "Conclusion: machine learning transforms modern technology", 10.0)
    ]


@pytest.fixture
def sample_clusters(sample_segments):
    """Create sample semantic clusters for testing."""
    return [
        SemanticCluster(
            cluster_id="cluster_1",
            segments=sample_segments[:2],
            theme="Introduction",
            importance_score=0.8,
            start_time=0.0,
            end_time=20.0,
            summary="Introduction to machine learning",
            keywords=["machine", "learning", "introduction"]
        ),
        SemanticCluster(
            cluster_id="cluster_2",
            segments=sample_segments[2:4],
            theme="Deep Learning",
            importance_score=0.9,
            start_time=20.0,
            end_time=40.0,
            summary="Deep learning and neural networks",
            keywords=["deep", "learning", "neural", "networks"]
        ),
        SemanticCluster(
            cluster_id="cluster_3",
            segments=sample_segments[4:],
            theme="Applications",
            importance_score=0.7,
            start_time=40.0,
            end_time=60.0,
            summary="Applications and conclusion",
            keywords=["applications", "image", "recognition"]
        )
    ]


@pytest.fixture
def mock_sentence_transformer():
    """Create a mock sentence transformer model."""
    mock_model = Mock()
    # Create realistic embedding-like vectors
    mock_model.encode.return_value = [0.1, 0.2, 0.3, 0.4, 0.5] * 76  # 384-dim vector
    return mock_model


class TestVectorSearchEngine:
    """Test cases for VectorSearchEngine."""
    
    def test_engine_initialization_without_model(self):
        """Test engine initialization when sentence transformers not available."""
        with patch('src.utils.vector_search.HAS_EMBEDDING_SUPPORT', False):
            engine = VectorSearchEngine()
            assert engine.model is None
            assert engine.model_name == "all-MiniLM-L6-v2"
            assert not engine.create_index([])
    
    def test_engine_initialization_with_model(self, mock_sentence_transformer):
        """Test engine initialization with available model."""
        with patch('src.utils.vector_search.HAS_EMBEDDING_SUPPORT', True):
            with patch('src.utils.vector_search.SentenceTransformer', return_value=mock_sentence_transformer):
                engine = VectorSearchEngine()
                assert engine.model is not None
                assert engine.model_name == "all-MiniLM-L6-v2"
    
    def test_create_index_success(self, sample_segments, mock_sentence_transformer):
        """Test successful index creation."""
        with patch('src.utils.vector_search.HAS_EMBEDDING_SUPPORT', True):
            with patch('src.utils.vector_search.SentenceTransformer', return_value=mock_sentence_transformer):
                engine = VectorSearchEngine()
                
                success = engine.create_index(sample_segments, "test_index")
                
                assert success is True
                assert "test_index" in engine.indexes
                
                index = engine.indexes["test_index"]
                assert len(index.segments) == len(sample_segments)
                assert len(index.embeddings) == len(sample_segments)
                assert index.model_name == "all-MiniLM-L6-v2"
    
    def test_create_index_without_model(self, sample_segments):
        """Test index creation failure without model."""
        engine = VectorSearchEngine()
        engine.model = None
        
        success = engine.create_index(sample_segments, "test_index")
        
        assert success is False
        assert "test_index" not in engine.indexes
    
    def test_search_similar_segments_success(self, sample_segments, mock_sentence_transformer):
        """Test successful similar segment search."""
        with patch('src.utils.vector_search.HAS_EMBEDDING_SUPPORT', True):
            with patch('src.utils.vector_search.SentenceTransformer', return_value=mock_sentence_transformer):
                with patch('src.utils.vector_search.cosine_similarity') as mock_cosine:
                    # Mock similarity scores
                    mock_cosine.return_value = [[0.9], [0.7], [0.8], [0.6], [0.5], [0.4]]
                    
                    engine = VectorSearchEngine()
                    engine.create_index(sample_segments, "test_index")
                    
                    result = engine.search_similar_segments(
                        query="machine learning concepts",
                        index_name="test_index",
                        top_k=3
                    )
                    
                    assert result.search_type == "similar_segments"
                    assert len(result.results) <= 3
                    assert result.processing_time > 0
                    assert len(result.similarity_scores) == len(result.results)
                    
                    # Results should be sorted by similarity (descending)
                    if len(result.similarity_scores) > 1:
                        for i in range(len(result.similarity_scores) - 1):
                            assert result.similarity_scores[i] >= result.similarity_scores[i + 1]
    
    def test_search_with_transcript_segment_query(self, sample_segments, mock_sentence_transformer):
        """Test search using a TranscriptSegment as query."""
        with patch('src.utils.vector_search.HAS_EMBEDDING_SUPPORT', True):
            with patch('src.utils.vector_search.SentenceTransformer', return_value=mock_sentence_transformer):
                with patch('src.utils.vector_search.cosine_similarity') as mock_cosine:
                    mock_cosine.return_value = [[0.8], [0.6], [0.7], [0.5], [0.4], [0.3]]
                    
                    engine = VectorSearchEngine()
                    engine.create_index(sample_segments, "test_index")
                    
                    query_segment = TranscriptSegment(0.0, 5.0, "neural network concepts", 5.0)
                    result = engine.search_similar_segments(query_segment, "test_index")
                    
                    assert result.query == "neural network concepts"
                    assert result.search_type == "similar_segments"
    
    def test_search_without_index(self, mock_sentence_transformer):
        """Test search with non-existent index."""
        with patch('src.utils.vector_search.HAS_EMBEDDING_SUPPORT', True):
            with patch('src.utils.vector_search.SentenceTransformer', return_value=mock_sentence_transformer):
                engine = VectorSearchEngine()
                
                result = engine.search_similar_segments("test query", "nonexistent_index")
                
                assert result.search_type == "empty"
                assert len(result.results) == 0
                assert "model_or_index_not_available" in result.metadata.get("error", "")
    
    def test_search_with_similarity_threshold(self, sample_segments, mock_sentence_transformer):
        """Test search with minimum similarity threshold."""
        with patch('src.utils.vector_search.HAS_EMBEDDING_SUPPORT', True):
            with patch('src.utils.vector_search.SentenceTransformer', return_value=mock_sentence_transformer):
                with patch('src.utils.vector_search.cosine_similarity') as mock_cosine:
                    # Set varying similarity scores
                    mock_cosine.return_value = [[0.9], [0.4], [0.8], [0.3], [0.7], [0.2]]
                    
                    engine = VectorSearchEngine()
                    engine.create_index(sample_segments, "test_index")
                    
                    result = engine.search_similar_segments(
                        query="test query",
                        index_name="test_index",
                        min_similarity=0.6  # Only scores >= 0.6
                    )
                    
                    # Should only return results with similarity >= 0.6
                    for score in result.similarity_scores:
                        assert score >= 0.6
    
    def test_find_optimal_timestamps(self, sample_clusters, mock_sentence_transformer):
        """Test optimal timestamp selection."""
        with patch('src.utils.vector_search.HAS_EMBEDDING_SUPPORT', True):
            with patch('src.utils.vector_search.SentenceTransformer', return_value=mock_sentence_transformer):
                with patch('src.utils.vector_search.HAS_NUMPY', True):
                    engine = VectorSearchEngine()
                    
                    timestamps = engine.find_optimal_timestamps(
                        semantic_clusters=sample_clusters,
                        target_count=3,
                        diversity_weight=0.3
                    )
                    
                    assert len(timestamps) <= 3
                    assert all(isinstance(ts, SemanticTimestamp) for ts in timestamps)
                    
                    if len(timestamps) > 1:
                        # Timestamps should be sorted by time
                        start_times = [ts.timestamp_seconds for ts in timestamps]
                        assert start_times == sorted(start_times)
    
    def test_find_optimal_timestamps_without_model(self, sample_clusters):
        """Test optimal timestamp selection without model."""
        engine = VectorSearchEngine()
        engine.model = None
        
        timestamps = engine.find_optimal_timestamps(sample_clusters, target_count=3)
        
        assert len(timestamps) == 0
    
    def test_analyze_semantic_coherence(self, sample_segments, mock_sentence_transformer):
        """Test semantic coherence analysis."""
        with patch('src.utils.vector_search.HAS_EMBEDDING_SUPPORT', True):
            with patch('src.utils.vector_search.SentenceTransformer', return_value=mock_sentence_transformer):
                with patch('src.utils.vector_search.cosine_similarity') as mock_cosine:
                    # Mock pairwise similarities
                    mock_cosine.side_effect = lambda x, y: [[0.8]]  # High coherence
                    
                    engine = VectorSearchEngine()
                    
                    result = engine.analyze_semantic_coherence(sample_segments)
                    
                    assert "coherence_score" in result
                    assert "average_similarity" in result
                    assert "sequential_coherence" in result
                    assert result["segment_count"] == len(sample_segments)
                    assert result["analysis"] == "completed"
                    assert 0.0 <= result["coherence_score"] <= 1.0
    
    def test_analyze_coherence_single_segment(self, mock_sentence_transformer):
        """Test coherence analysis with single segment."""
        with patch('src.utils.vector_search.HAS_EMBEDDING_SUPPORT', True):
            with patch('src.utils.vector_search.SentenceTransformer', return_value=mock_sentence_transformer):
                engine = VectorSearchEngine()
                
                single_segment = [TranscriptSegment(0.0, 10.0, "Single segment", 10.0)]
                result = engine.analyze_semantic_coherence(single_segment)
                
                assert result["coherence_score"] == 1.0
                assert result["analysis"] == "single_segment"
    
    def test_analyze_coherence_without_model(self, sample_segments):
        """Test coherence analysis without model."""
        engine = VectorSearchEngine()
        engine.model = None
        
        result = engine.analyze_semantic_coherence(sample_segments)
        
        assert result["coherence_score"] == 0.0
        assert result["analysis"] == "no_data"
    
    def test_embedding_caching(self, mock_sentence_transformer):
        """Test embedding caching functionality."""
        with patch('src.utils.vector_search.HAS_EMBEDDING_SUPPORT', True):
            with patch('src.utils.vector_search.SentenceTransformer', return_value=mock_sentence_transformer):
                engine = VectorSearchEngine()
                engine.cache_embeddings = True
                
                text = "test text for caching"
                
                # First call should generate embedding
                embeddings1 = engine._generate_embeddings([text])
                assert mock_sentence_transformer.encode.call_count == 1
                
                # Second call should use cache
                embeddings2 = engine._generate_embeddings([text])
                assert mock_sentence_transformer.encode.call_count == 1  # No additional calls
                
                assert len(embeddings1) == len(embeddings2)
    
    def test_similarity_calculation_fallback(self):
        """Test similarity calculation fallback without sklearn."""
        engine = VectorSearchEngine()
        
        emb1 = [1.0, 0.0, 0.0]
        emb2 = [0.0, 1.0, 0.0]
        emb3 = [1.0, 0.0, 0.0]  # Same as emb1
        
        # Different embeddings should have low similarity
        sim1 = engine._calculate_similarity(emb1, emb2)
        assert 0.0 <= sim1 <= 1.0
        
        # Same embeddings should have high similarity
        sim2 = engine._calculate_similarity(emb1, emb3)
        assert sim2 > sim1
    
    def test_get_index_info(self, sample_segments, mock_sentence_transformer):
        """Test getting index information."""
        with patch('src.utils.vector_search.HAS_EMBEDDING_SUPPORT', True):
            with patch('src.utils.vector_search.SentenceTransformer', return_value=mock_sentence_transformer):
                engine = VectorSearchEngine()
                
                # Test non-existent index
                info = engine.get_index_info("nonexistent")
                assert info["exists"] is False
                
                # Create index and test
                engine.create_index(sample_segments, "test_index")
                info = engine.get_index_info("test_index")
                
                assert info["exists"] is True
                assert info["segment_count"] == len(sample_segments)
                assert info["model_name"] == "all-MiniLM-L6-v2"
                assert "created_at" in info
                assert "metadata" in info
    
    def test_cache_management(self, mock_sentence_transformer):
        """Test embedding cache management."""
        with patch('src.utils.vector_search.HAS_EMBEDDING_SUPPORT', True):
            with patch('src.utils.vector_search.SentenceTransformer', return_value=mock_sentence_transformer):
                engine = VectorSearchEngine()
                
                # Add some embeddings to cache
                engine._generate_embeddings(["text1", "text2", "text3"])
                
                initial_size = engine.get_cache_size()
                assert initial_size > 0
                
                # Clear cache
                engine.clear_cache()
                assert engine.get_cache_size() == 0
    
    def test_format_timestamp(self):
        """Test timestamp formatting."""
        engine = VectorSearchEngine()
        
        assert engine._format_timestamp(65.0) == "1:05"
        assert engine._format_timestamp(3665.0) == "1:01:05"
        assert engine._format_timestamp(30.0) == "0:30"
        assert engine._format_timestamp(0.0) == "0:00"


class TestVectorSearchResult:
    """Test cases for VectorSearchResult dataclass."""
    
    def test_search_result_creation(self):
        """Test creating a vector search result."""
        result = VectorSearchResult(
            query="test query",
            results=[{"similarity": 0.8, "text": "result"}],
            search_type="similar_segments",
            processing_time=1.5,
            similarity_scores=[0.8],
            metadata={"test": "data"}
        )
        
        assert result.query == "test query"
        assert len(result.results) == 1
        assert result.search_type == "similar_segments"
        assert result.processing_time == 1.5
        assert result.similarity_scores == [0.8]
        assert result.metadata["test"] == "data"


class TestEmbeddingIndex:
    """Test cases for EmbeddingIndex dataclass."""
    
    def test_embedding_index_creation(self, sample_segments):
        """Test creating an embedding index."""
        mock_embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        creation_time = datetime.utcnow()
        
        index = EmbeddingIndex(
            embeddings=mock_embeddings,
            segments=sample_segments[:2],
            metadata={"test": "metadata"},
            model_name="test-model",
            created_at=creation_time
        )
        
        assert len(index.embeddings) == 2
        assert len(index.segments) == 2
        assert index.model_name == "test-model"
        assert index.created_at == creation_time
        assert index.metadata["test"] == "metadata"


class TestVectorSearchIntegration:
    """Integration tests for vector search functionality."""
    
    def test_full_search_workflow(self, sample_segments, mock_sentence_transformer):
        """Test complete search workflow from index creation to results."""
        with patch('src.utils.vector_search.HAS_EMBEDDING_SUPPORT', True):
            with patch('src.utils.vector_search.SentenceTransformer', return_value=mock_sentence_transformer):
                with patch('src.utils.vector_search.cosine_similarity') as mock_cosine:
                    mock_cosine.return_value = [[0.9], [0.7], [0.8], [0.6], [0.5], [0.4]]
                    
                    engine = VectorSearchEngine()
                    
                    # Create index
                    success = engine.create_index(sample_segments, "workflow_test")
                    assert success is True
                    
                    # Perform search
                    result = engine.search_similar_segments(
                        query="machine learning",
                        index_name="workflow_test",
                        top_k=3,
                        min_similarity=0.6
                    )
                    
                    # Verify results
                    assert result.search_type == "similar_segments"
                    assert len(result.results) <= 3
                    assert all(score >= 0.6 for score in result.similarity_scores)
                    
                    # Verify result structure
                    for res in result.results:
                        assert "segment" in res
                        assert "similarity" in res
                        assert "start_time" in res
                        assert "text" in res
                        assert "rank" in res
    
    def test_timestamp_optimization_workflow(self, sample_clusters, mock_sentence_transformer):
        """Test timestamp optimization workflow."""
        with patch('src.utils.vector_search.HAS_EMBEDDING_SUPPORT', True):
            with patch('src.utils.vector_search.SentenceTransformer', return_value=mock_sentence_transformer):
                with patch('src.utils.vector_search.HAS_NUMPY', True):
                    engine = VectorSearchEngine()
                    
                    timestamps = engine.find_optimal_timestamps(
                        semantic_clusters=sample_clusters,
                        target_count=2,
                        diversity_weight=0.5
                    )
                    
                    assert len(timestamps) <= 2
                    
                    # Verify timestamp structure
                    for ts in timestamps:
                        assert hasattr(ts, 'timestamp_seconds')
                        assert hasattr(ts, 'description')
                        assert hasattr(ts, 'importance_rating')
                        assert hasattr(ts, 'confidence_score')
                        assert 1 <= ts.importance_rating <= 10
                        assert 0.0 <= ts.confidence_score <= 1.0


class TestFactoryFunction:
    """Test cases for factory function."""
    
    def test_create_vector_search_engine(self):
        """Test vector search engine factory function."""
        engine = create_vector_search_engine()
        
        assert isinstance(engine, VectorSearchEngine)
        assert engine.model_name == "all-MiniLM-L6-v2"
    
    def test_create_vector_search_engine_custom_model(self):
        """Test factory function with custom model."""
        engine = create_vector_search_engine("custom-model")
        
        assert isinstance(engine, VectorSearchEngine)
        assert engine.model_name == "custom-model"


if __name__ == "__main__":
    pytest.main([__file__])