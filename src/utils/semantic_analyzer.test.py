"""
Tests for Semantic Analyzer utility.

This module contains comprehensive tests for the semantic grouping algorithms
and the main semantic analyzer functionality.
"""

import pytest
from unittest.mock import Mock, patch
from typing import List

from .semantic_analyzer import (
    SemanticAnalyzer,
    TfidfClusteringAlgorithm,
    ContentStructureAlgorithm,
    SemanticFeatures,
    ClusteringResult,
    create_semantic_analyzer,
    create_tfidf_analyzer,
    create_structure_analyzer
)
from ..services.semantic_analysis_service import TranscriptSegment, SemanticCluster


@pytest.fixture
def sample_segments():
    """Create sample transcript segments for testing."""
    return [
        TranscriptSegment(0.0, 10.0, "Welcome to this video about machine learning and artificial intelligence", 10.0),
        TranscriptSegment(10.0, 20.0, "Today we will explore the fundamentals of neural networks", 10.0),
        TranscriptSegment(20.0, 30.0, "First, let's understand what machine learning actually means", 10.0),
        TranscriptSegment(30.0, 40.0, "Neural networks are computational models inspired by biological brains", 10.0),
        TranscriptSegment(40.0, 50.0, "The key components include neurons, weights, and activation functions", 10.0),
        TranscriptSegment(50.0, 60.0, "Deep learning extends neural networks with multiple hidden layers", 10.0),
        TranscriptSegment(60.0, 70.0, "Training involves adjusting weights through backpropagation algorithms", 10.0),
        TranscriptSegment(70.0, 80.0, "Next, let's move on to practical applications of these concepts", 10.0),
        TranscriptSegment(80.0, 90.0, "Machine learning is used in image recognition and natural language processing", 10.0),
        TranscriptSegment(90.0, 100.0, "In conclusion, these concepts form the foundation of modern AI systems", 10.0),
        TranscriptSegment(100.0, 110.0, "Thank you for watching this introduction to machine learning", 10.0)
    ]


@pytest.fixture
def short_segments():
    """Create short segments for testing edge cases."""
    return [
        TranscriptSegment(0.0, 5.0, "Hello everyone", 5.0),
        TranscriptSegment(5.0, 10.0, "Short content here", 5.0),
        TranscriptSegment(10.0, 15.0, "Very brief segment", 5.0)
    ]


class TestSemanticFeatures:
    """Test cases for SemanticFeatures dataclass."""
    
    def test_semantic_features_creation(self):
        """Test creating semantic features."""
        features = SemanticFeatures(
            keywords=["test", "keyword"],
            topics=["Topic1", "Topic2"],
            sentiment_score=0.7,
            complexity_score=0.5,
            importance_indicators=["important", "key"],
            content_type="main_content",
            speaker_indicators=[]
        )
        
        assert features.keywords == ["test", "keyword"]
        assert features.topics == ["Topic1", "Topic2"]
        assert features.sentiment_score == 0.7
        assert features.content_type == "main_content"


class TestClusteringResult:
    """Test cases for ClusteringResult dataclass."""
    
    def test_clustering_result_creation(self):
        """Test creating clustering result."""
        clusters = [Mock(spec=SemanticCluster)]
        result = ClusteringResult(
            clusters=clusters,
            cluster_assignments=[0, 0, 1],
            cluster_quality_score=0.8,
            algorithm_used="TestAlgorithm",
            processing_time=1.5,
            metadata={"test": "data"}
        )
        
        assert len(result.clusters) == 1
        assert result.cluster_assignments == [0, 0, 1]
        assert result.cluster_quality_score == 0.8
        assert result.algorithm_used == "TestAlgorithm"
        assert result.processing_time == 1.5
        assert result.metadata["test"] == "data"


class TestTfidfClusteringAlgorithm:
    """Test cases for TF-IDF clustering algorithm."""
    
    def test_algorithm_initialization(self):
        """Test TF-IDF algorithm initialization."""
        algorithm = TfidfClusteringAlgorithm()
        assert algorithm.name == "TfidfClustering"
        assert algorithm.max_features == 1000
        assert algorithm.min_df == 2
        assert len(algorithm.stop_words) > 0
    
    def test_extract_features(self, sample_segments):
        """Test feature extraction."""
        algorithm = TfidfClusteringAlgorithm()
        segment = sample_segments[0]
        
        features = algorithm.extract_features(segment)
        
        assert isinstance(features, SemanticFeatures)
        assert len(features.keywords) > 0
        assert features.content_type in ["introduction", "main_content", "conclusion", "transition"]
        assert 0.0 <= features.complexity_score <= 1.0
        assert isinstance(features.importance_indicators, list)
    
    def test_detect_content_type(self, sample_segments):
        """Test content type detection."""
        algorithm = TfidfClusteringAlgorithm()
        
        # Test introduction detection
        intro_text = "welcome to this video about machine learning"
        assert algorithm._detect_content_type(intro_text) == "introduction"
        
        # Test conclusion detection
        conclusion_text = "in conclusion, these concepts are important"
        assert algorithm._detect_content_type(conclusion_text) == "conclusion"
        
        # Test transition detection
        transition_text = "now let's move on to the next topic"
        assert algorithm._detect_content_type(transition_text) == "transition"
        
        # Test main content detection
        main_text = "neural networks are computational models"
        assert algorithm._detect_content_type(main_text) == "main_content"
    
    def test_extract_importance_indicators(self):
        """Test importance indicator extraction."""
        algorithm = TfidfClusteringAlgorithm()
        
        text_with_indicators = "This is important and key information to remember"
        indicators = algorithm._extract_importance_indicators(text_with_indicators)
        
        assert "important" in indicators
        assert "key" in indicators
        assert "remember" in indicators
    
    def test_calculate_complexity_score(self):
        """Test complexity score calculation."""
        algorithm = TfidfClusteringAlgorithm()
        
        # Simple text
        simple_text = "This is simple text"
        simple_score = algorithm._calculate_complexity_score(simple_text)
        
        # Complex text
        complex_text = "This extraordinarily sophisticated methodology demonstrates unprecedented capabilities"
        complex_score = algorithm._calculate_complexity_score(complex_text)
        
        assert 0.0 <= simple_score <= 1.0
        assert 0.0 <= complex_score <= 1.0
        assert complex_score > simple_score
    
    @pytest.mark.skipif(not hasattr(pytest, "importorskip"), reason="sklearn may not be available")
    def test_cluster_segments_with_sklearn(self, sample_segments):
        """Test clustering with sklearn available."""
        algorithm = TfidfClusteringAlgorithm()
        
        with patch('src.utils.semantic_analyzer.HAS_SKLEARN', True):
            with patch('sklearn.feature_extraction.text.TfidfVectorizer') as mock_vectorizer:
                with patch('sklearn.cluster.KMeans') as mock_kmeans:
                    # Mock the vectorizer
                    mock_tfidf_instance = Mock()
                    mock_tfidf_instance.fit_transform.return_value = Mock()
                    mock_tfidf_instance.get_feature_names_out.return_value = ["word1", "word2"]
                    mock_vectorizer.return_value = mock_tfidf_instance
                    
                    # Mock the kmeans
                    mock_kmeans_instance = Mock()
                    mock_kmeans_instance.fit_predict.return_value = [0, 0, 1, 1, 2, 2, 2, 1, 1, 0, 0]
                    mock_kmeans.return_value = mock_kmeans_instance
                    
                    result = algorithm.cluster_segments(sample_segments, target_clusters=3)
                    
                    assert result.algorithm_used == "TfidfClustering"
                    assert len(result.clusters) > 0
                    assert len(result.cluster_assignments) == len(sample_segments)
    
    def test_cluster_segments_fallback(self, sample_segments):
        """Test clustering fallback when sklearn is not available."""
        algorithm = TfidfClusteringAlgorithm()
        
        with patch('src.utils.semantic_analyzer.HAS_SKLEARN', False):
            result = algorithm.cluster_segments(sample_segments, target_clusters=3)
            
            assert result.algorithm_used == "FallbackClustering"
            assert len(result.clusters) > 0
            assert len(result.cluster_assignments) == len(sample_segments)
    
    def test_fallback_clustering_directly(self, sample_segments):
        """Test fallback clustering method directly."""
        algorithm = TfidfClusteringAlgorithm()
        
        result = algorithm._fallback_clustering(sample_segments, target_clusters=3)
        
        assert result.algorithm_used == "FallbackClustering"
        assert len(result.clusters) > 0
        assert all(cluster.cluster_id.startswith("fallback_cluster") for cluster in result.clusters)
    
    def test_cluster_segments_empty_input(self):
        """Test clustering with empty input."""
        algorithm = TfidfClusteringAlgorithm()
        
        result = algorithm.cluster_segments([], target_clusters=3)
        
        assert len(result.clusters) == 0
        assert len(result.cluster_assignments) == 0
    
    def test_extract_cluster_theme(self):
        """Test cluster theme extraction."""
        algorithm = TfidfClusteringAlgorithm()
        
        text = "machine learning artificial intelligence neural networks deep learning"
        theme = algorithm._extract_cluster_theme(text)
        
        assert isinstance(theme, str)
        assert len(theme) > 0
        assert theme != "Content Section"  # Should extract meaningful theme
    
    def test_calculate_cluster_importance(self, sample_segments):
        """Test cluster importance calculation."""
        algorithm = TfidfClusteringAlgorithm()
        
        # High importance cluster (longer, with importance keywords)
        high_importance_segments = sample_segments[:3]  # Introduction segments
        high_score = algorithm._calculate_cluster_importance(high_importance_segments)
        
        # Low importance cluster (shorter)
        low_importance_segments = sample_segments[-1:]  # Just thank you
        low_score = algorithm._calculate_cluster_importance(low_importance_segments)
        
        assert 0.0 <= high_score <= 1.0
        assert 0.0 <= low_score <= 1.0


class TestContentStructureAlgorithm:
    """Test cases for content structure algorithm."""
    
    def test_algorithm_initialization(self):
        """Test structure algorithm initialization."""
        algorithm = ContentStructureAlgorithm()
        assert algorithm.name == "ContentStructure"
        assert len(algorithm.introduction_markers) > 0
        assert len(algorithm.conclusion_markers) > 0
        assert len(algorithm.transition_markers) > 0
    
    def test_classify_content_type(self):
        """Test content type classification."""
        algorithm = ContentStructureAlgorithm()
        
        # Test each content type
        assert algorithm._classify_content_type("welcome to this presentation") == "introduction"
        assert algorithm._classify_content_type("in conclusion we can see") == "conclusion"
        assert algorithm._classify_content_type("now let's move on to") == "transition"
        assert algorithm._classify_content_type("the main concept here is") == "main_content"
    
    def test_extract_features(self, sample_segments):
        """Test feature extraction for structure algorithm."""
        algorithm = ContentStructureAlgorithm()
        
        intro_segment = sample_segments[0]  # Contains "Welcome"
        features = algorithm.extract_features(intro_segment)
        
        assert features.content_type == "introduction"
        assert isinstance(features.keywords, list)
        assert 0.0 <= features.complexity_score <= 1.0
    
    def test_cluster_segments(self, sample_segments):
        """Test structural clustering."""
        algorithm = ContentStructureAlgorithm()
        
        result = algorithm.cluster_segments(sample_segments, target_clusters=5)
        
        assert result.algorithm_used == "ContentStructure"
        assert len(result.clusters) > 0
        assert len(result.cluster_assignments) == len(sample_segments)
        
        # Check that different content types are properly clustered
        cluster_themes = [cluster.theme for cluster in result.clusters]
        assert any("Introduction" in theme for theme in cluster_themes)
        assert any("Conclusion" in theme for theme in cluster_themes)
    
    def test_calculate_structural_importance(self):
        """Test structural importance calculation."""
        algorithm = ContentStructureAlgorithm()
        
        intro_score = algorithm._calculate_structural_importance("introduction", ["important"])
        conclusion_score = algorithm._calculate_structural_importance("conclusion", [])
        main_score = algorithm._calculate_structural_importance("main_content", ["key", "crucial"])
        transition_score = algorithm._calculate_structural_importance("transition", [])
        
        assert intro_score > transition_score
        assert conclusion_score > transition_score
        assert main_score > transition_score  # Due to importance indicators
    
    def test_create_structure_based_clusters(self, sample_segments):
        """Test structure-based cluster creation."""
        algorithm = ContentStructureAlgorithm()
        
        # Extract features first
        features = [algorithm.extract_features(segment) for segment in sample_segments]
        
        clusters = algorithm._create_structure_based_clusters(sample_segments, features)
        
        assert len(clusters) > 0
        assert all(isinstance(cluster, SemanticCluster) for cluster in clusters)
        
        # Verify time ordering
        start_times = [cluster.start_time for cluster in clusters]
        assert start_times == sorted(start_times)
    
    def test_extract_structural_keywords(self):
        """Test structural keyword extraction."""
        algorithm = ContentStructureAlgorithm()
        
        text = "Machine Learning and Artificial Intelligence concepts are important topics"
        keywords = algorithm._extract_structural_keywords(text.lower())
        
        assert isinstance(keywords, list)
        assert len(keywords) > 0
    
    def test_calculate_structure_quality(self):
        """Test structure quality calculation."""
        algorithm = ContentStructureAlgorithm()
        
        # Create mock clusters with different themes
        mock_clusters = [
            Mock(theme="Introduction", importance_score=0.8),
            Mock(theme="Main Content 1", importance_score=0.7),
            Mock(theme="Conclusion", importance_score=0.6)
        ]
        
        quality = algorithm._calculate_structure_quality(mock_clusters)
        
        assert 0.0 <= quality <= 1.0


class TestSemanticAnalyzer:
    """Test cases for the main semantic analyzer."""
    
    def test_analyzer_initialization(self):
        """Test semantic analyzer initialization."""
        analyzer = SemanticAnalyzer()
        
        assert 'tfidf' in analyzer.algorithms
        assert 'structure' in analyzer.algorithms
        assert analyzer.default_algorithm == 'tfidf'
        assert analyzer.fallback_algorithm == 'structure'
    
    def test_analyze_segments_with_specific_algorithm(self, sample_segments):
        """Test analysis with specific algorithm."""
        analyzer = SemanticAnalyzer()
        
        # Test with structure algorithm
        result = analyzer.analyze_segments(sample_segments, algorithm='structure', target_clusters=3)
        
        assert result.algorithm_used == "ContentStructure"
        assert len(result.clusters) > 0
    
    def test_analyze_segments_auto_selection(self, sample_segments):
        """Test automatic algorithm selection."""
        analyzer = SemanticAnalyzer()
        
        result = analyzer.analyze_segments(sample_segments, algorithm='auto', target_clusters=3)
        
        assert result.algorithm_used in ["TfidfClustering", "ContentStructure", "FallbackClustering"]
        assert len(result.clusters) > 0
    
    def test_analyze_segments_empty_input(self):
        """Test analysis with empty input."""
        analyzer = SemanticAnalyzer()
        
        result = analyzer.analyze_segments([], algorithm='tfidf')
        
        assert result.algorithm_used == "empty"
        assert len(result.clusters) == 0
    
    def test_select_optimal_algorithm(self, sample_segments, short_segments):
        """Test optimal algorithm selection logic."""
        analyzer = SemanticAnalyzer()
        
        # Test with long content (should prefer tfidf if available)
        long_algorithm = analyzer._select_optimal_algorithm(sample_segments)
        assert long_algorithm in ['tfidf', 'structure']
        
        # Test with short content (should prefer structure)
        short_algorithm = analyzer._select_optimal_algorithm(short_segments)
        assert short_algorithm == 'structure'
    
    def test_count_structure_indicators(self):
        """Test structure indicator counting."""
        analyzer = SemanticAnalyzer()
        
        text_with_structure = "Introduction to the topic. First, we will discuss key concepts. Finally, conclusion."
        count = analyzer._count_structure_indicators(text_with_structure)
        
        assert count > 0
    
    def test_get_available_algorithms(self):
        """Test getting available algorithms."""
        analyzer = SemanticAnalyzer()
        
        algorithms = analyzer.get_available_algorithms()
        
        assert 'tfidf' in algorithms
        assert 'structure' in algorithms
        assert isinstance(algorithms, list)
    
    def test_get_algorithm_info(self):
        """Test getting algorithm information."""
        analyzer = SemanticAnalyzer()
        
        tfidf_info = analyzer.get_algorithm_info('tfidf')
        structure_info = analyzer.get_algorithm_info('structure')
        unknown_info = analyzer.get_algorithm_info('unknown')
        
        assert tfidf_info['name'] == 'TfidfClustering'
        assert structure_info['name'] == 'ContentStructure'
        assert unknown_info == {}
        
        assert tfidf_info['available'] is True
        assert structure_info['available'] is True
    
    def test_fallback_mechanism(self, sample_segments):
        """Test fallback mechanism when primary algorithm fails."""
        analyzer = SemanticAnalyzer()
        
        # Mock the primary algorithm to fail
        with patch.object(analyzer.algorithms['tfidf'], 'cluster_segments', side_effect=Exception("Test failure")):
            result = analyzer.analyze_segments(sample_segments, algorithm='tfidf')
            
            # Should fallback to structure algorithm
            assert "fallback" in result.algorithm_used or result.algorithm_used == "ContentStructure"
    
    def test_analyze_segments_with_invalid_algorithm(self, sample_segments):
        """Test analysis with invalid algorithm name."""
        analyzer = SemanticAnalyzer()
        
        result = analyzer.analyze_segments(sample_segments, algorithm='invalid_algorithm')
        
        # Should use default algorithm
        assert result.algorithm_used in ["TfidfClustering", "ContentStructure", "FallbackClustering"]


class TestFactoryFunctions:
    """Test cases for factory functions."""
    
    def test_create_semantic_analyzer(self):
        """Test semantic analyzer factory function."""
        analyzer = create_semantic_analyzer()
        
        assert isinstance(analyzer, SemanticAnalyzer)
        assert 'tfidf' in analyzer.algorithms
        assert 'structure' in analyzer.algorithms
    
    def test_create_tfidf_analyzer(self):
        """Test TF-IDF analyzer factory function."""
        analyzer = create_tfidf_analyzer()
        
        assert isinstance(analyzer, TfidfClusteringAlgorithm)
        assert analyzer.name == "TfidfClustering"
    
    def test_create_structure_analyzer(self):
        """Test structure analyzer factory function."""
        analyzer = create_structure_analyzer()
        
        assert isinstance(analyzer, ContentStructureAlgorithm)
        assert analyzer.name == "ContentStructure"


class TestIntegrationScenarios:
    """Integration tests for realistic scenarios."""
    
    def test_educational_content_analysis(self):
        """Test analysis of educational content."""
        educational_segments = [
            TranscriptSegment(0.0, 10.0, "Welcome to today's lesson on photosynthesis", 10.0),
            TranscriptSegment(10.0, 20.0, "First, let's define what photosynthesis means", 10.0),
            TranscriptSegment(20.0, 40.0, "Photosynthesis is the process where plants convert light into energy", 20.0),
            TranscriptSegment(40.0, 60.0, "The key components are chlorophyll, carbon dioxide, and sunlight", 20.0),
            TranscriptSegment(60.0, 80.0, "Now let's examine the detailed chemical reactions involved", 20.0),
            TranscriptSegment(80.0, 100.0, "The light reactions occur in the thylakoid membranes", 20.0),
            TranscriptSegment(100.0, 120.0, "In conclusion, photosynthesis is essential for life on Earth", 20.0)
        ]
        
        analyzer = SemanticAnalyzer()
        result = analyzer.analyze_segments(educational_segments, algorithm='structure')
        
        assert len(result.clusters) > 0
        assert result.cluster_quality_score > 0.0
        
        # Should identify introduction and conclusion
        themes = [cluster.theme for cluster in result.clusters]
        assert any("Introduction" in theme for theme in themes)
        assert any("Conclusion" in theme for theme in themes)
    
    def test_interview_content_analysis(self):
        """Test analysis of interview content."""
        interview_segments = [
            TranscriptSegment(0.0, 15.0, "Thank you for joining us today for this interview", 15.0),
            TranscriptSegment(15.0, 30.0, "Can you tell us about your background in technology?", 15.0),
            TranscriptSegment(30.0, 60.0, "I have been working in artificial intelligence for over ten years", 30.0),
            TranscriptSegment(60.0, 90.0, "What are the main challenges in AI development today?", 30.0),
            TranscriptSegment(90.0, 150.0, "The key challenges include data quality, ethical considerations, and computational resources", 60.0),
            TranscriptSegment(150.0, 180.0, "How do you see AI evolving in the next five years?", 30.0),
            TranscriptSegment(180.0, 240.0, "I believe we will see significant advances in natural language processing", 60.0),
            TranscriptSegment(240.0, 260.0, "Thank you for your insights and time", 20.0)
        ]
        
        analyzer = SemanticAnalyzer()
        result = analyzer.analyze_segments(interview_segments, algorithm='auto')
        
        assert len(result.clusters) > 0
        assert result.processing_time > 0.0
        
        # Verify clusters contain reasonable content
        for cluster in result.clusters:
            assert cluster.start_time >= 0.0
            assert cluster.end_time > cluster.start_time
            assert len(cluster.segments) > 0
    
    def test_performance_with_many_segments(self):
        """Test performance with a large number of segments."""
        # Generate many segments
        many_segments = []
        for i in range(50):
            segment = TranscriptSegment(
                start_time=i * 5.0,
                end_time=(i + 1) * 5.0,
                text=f"This is segment number {i} discussing topic {i % 10} with relevant content",
                duration=5.0
            )
            many_segments.append(segment)
        
        analyzer = SemanticAnalyzer()
        result = analyzer.analyze_segments(many_segments, target_clusters=8)
        
        assert len(result.clusters) > 0
        assert len(result.cluster_assignments) == 50
        assert result.processing_time < 10.0  # Should complete within reasonable time
        
        # Verify all segments are assigned
        assert len(result.cluster_assignments) == len(many_segments)
        assert all(0 <= assignment < len(result.clusters) for assignment in result.cluster_assignments)


if __name__ == "__main__":
    pytest.main([__file__])