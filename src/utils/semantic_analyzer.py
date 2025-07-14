"""
Semantic Analyzer Utility for YouTube Video Summarizer.

This module implements advanced semantic grouping algorithms for transcript analysis,
providing sophisticated methods for clustering content by meaning and importance.
"""

import logging
import re
import math
import os
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime
from collections import Counter, defaultdict
from dataclasses import dataclass

# Fix tokenizers parallelism warning
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    # We'll implement fallback methods without sklearn

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    # Embedding features will use fallback methods

from .semantic_types import TranscriptSegment, SemanticCluster

logger = logging.getLogger(__name__)


@dataclass
class SemanticFeatures:
    """Features extracted from text for semantic analysis."""
    keywords: List[str]
    topics: List[str]
    sentiment_score: float
    complexity_score: float
    importance_indicators: List[str]
    content_type: str  # "introduction", "main_content", "conclusion", "transition"
    speaker_indicators: List[str]


@dataclass
class ClusteringResult:
    """Result of semantic clustering operation."""
    clusters: List[SemanticCluster]
    cluster_assignments: List[int]
    cluster_quality_score: float
    algorithm_used: str
    processing_time: float
    metadata: Dict[str, Any]


class SemanticGroupingAlgorithm:
    """Base class for semantic grouping algorithms."""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"{__name__}.{name}")
    
    def cluster_segments(
        self, 
        segments: List[TranscriptSegment], 
        target_clusters: int = 5,
        **kwargs
    ) -> ClusteringResult:
        """
        Cluster transcript segments into semantic groups.
        
        Args:
            segments: List of transcript segments
            target_clusters: Desired number of clusters
            **kwargs: Algorithm-specific parameters
            
        Returns:
            ClusteringResult with clustering information
        """
        raise NotImplementedError("Subclasses must implement cluster_segments")
    
    def extract_features(self, segment: TranscriptSegment) -> SemanticFeatures:
        """Extract semantic features from a transcript segment."""
        raise NotImplementedError("Subclasses must implement extract_features")


class TfidfClusteringAlgorithm(SemanticGroupingAlgorithm):
    """
    TF-IDF based clustering algorithm.
    
    Uses TF-IDF vectorization followed by K-means clustering to group
    semantically similar content segments.
    """
    
    def __init__(self):
        super().__init__("TfidfClustering")
        self.vectorizer = None
        self.kmeans = None
        
        # Configuration
        self.max_features = 1000
        self.stop_words = self._get_stop_words()
        self.min_df = 2
        self.max_df = 0.8
    
    def cluster_segments(
        self, 
        segments: List[TranscriptSegment], 
        target_clusters: int = 5,
        **kwargs
    ) -> ClusteringResult:
        """Cluster segments using TF-IDF and K-means."""
        start_time = datetime.utcnow()
        
        if not HAS_SKLEARN:
            self.logger.warning("scikit-learn not available, using fallback clustering")
            return self._fallback_clustering(segments, target_clusters)
        
        try:
            # Extract text from segments
            texts = [segment.text for segment in segments]
            
            if len(texts) < target_clusters:
                target_clusters = max(1, len(texts) // 2)
            
            # Create TF-IDF vectorizer
            self.vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                stop_words=self.stop_words,
                min_df=self.min_df,
                max_df=self.max_df,
                ngram_range=(1, 2)
            )
            
            # Fit and transform texts
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            
            # Perform K-means clustering
            self.kmeans = KMeans(
                n_clusters=target_clusters,
                random_state=42,
                n_init=10
            )
            cluster_labels = self.kmeans.fit_predict(tfidf_matrix)
            
            # Create semantic clusters
            clusters = self._create_clusters_from_labels(segments, cluster_labels)
            
            # Calculate quality score
            quality_score = self._calculate_cluster_quality(tfidf_matrix, cluster_labels)
            
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            return ClusteringResult(
                clusters=clusters,
                cluster_assignments=cluster_labels.tolist(),
                cluster_quality_score=quality_score,
                algorithm_used="TfidfClustering",
                processing_time=processing_time,
                metadata={
                    "vectorizer_features": self.vectorizer.get_feature_names_out().tolist()[:20],
                    "silhouette_score": quality_score,
                    "target_clusters": target_clusters,
                    "actual_clusters": len(clusters)
                }
            )
            
        except Exception as e:
            self.logger.error(f"TF-IDF clustering failed: {str(e)}")
            return self._fallback_clustering(segments, target_clusters)
    
    def extract_features(self, segment: TranscriptSegment) -> SemanticFeatures:
        """Extract TF-IDF based features from segment."""
        text = segment.text.lower()
        
        # Extract keywords using TF-IDF if available
        keywords = []
        if self.vectorizer and hasattr(self.vectorizer, 'vocabulary_'):
            # Use TF-IDF vocabulary to identify important terms
            words = text.split()
            vocab = self.vectorizer.vocabulary_
            keywords = [word for word in words if word in vocab][:10]
        else:
            # Fallback: simple frequency-based keywords
            words = re.findall(r'\b\w+\b', text)
            word_freq = Counter(words)
            keywords = [word for word, _ in word_freq.most_common(10)]
        
        # Detect content type
        content_type = self._detect_content_type(text)
        
        # Calculate importance indicators
        importance_indicators = self._extract_importance_indicators(text)
        
        # Calculate complexity score
        complexity_score = self._calculate_complexity_score(text)
        
        return SemanticFeatures(
            keywords=keywords,
            topics=self._extract_topics(text),
            sentiment_score=0.5,  # Neutral default
            complexity_score=complexity_score,
            importance_indicators=importance_indicators,
            content_type=content_type,
            speaker_indicators=[]
        )
    
    def _get_stop_words(self) -> List[str]:
        """Get list of stop words for filtering."""
        # Basic English stop words
        return [
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'were', 'will', 'with', 'you', 'your', 'this', 'these',
            'they', 'them', 'their', 'have', 'had', 'do', 'does', 'did', 'can',
            'could', 'should', 'would', 'but', 'not', 'what', 'when', 'where',
            'who', 'why', 'how', 'so', 'if', 'than', 'then', 'now', 'just',
            'like', 'well', 'know', 'think', 'see', 'get', 'go', 'come', 'want'
        ]
    
    def _create_clusters_from_labels(
        self, 
        segments: List[TranscriptSegment], 
        labels: List[int]
    ) -> List[SemanticCluster]:
        """Create SemanticCluster objects from clustering labels."""
        cluster_dict = defaultdict(list)
        
        for segment, label in zip(segments, labels):
            cluster_dict[label].append(segment)
        
        clusters = []
        for cluster_id, cluster_segments in cluster_dict.items():
            if not cluster_segments:
                continue
            
            # Calculate cluster properties
            start_time = min(seg.start_time for seg in cluster_segments)
            end_time = max(seg.end_time or seg.start_time + 5 for seg in cluster_segments)
            
            # Extract cluster theme
            all_text = " ".join(seg.text for seg in cluster_segments)
            theme = self._extract_cluster_theme(all_text)
            
            # Generate summary
            summary = self._generate_cluster_summary(cluster_segments)
            
            # Extract keywords
            keywords = self._extract_cluster_keywords(all_text)
            
            # Calculate importance score
            importance_score = self._calculate_cluster_importance(cluster_segments)
            
            cluster = SemanticCluster(
                cluster_id=f"tfidf_cluster_{cluster_id}",
                segments=cluster_segments,
                theme=theme,
                importance_score=importance_score,
                start_time=start_time,
                end_time=end_time,
                summary=summary,
                keywords=keywords
            )
            
            clusters.append(cluster)
        
        # Sort clusters by start time
        clusters.sort(key=lambda x: x.start_time)
        
        return clusters
    
    def _calculate_cluster_quality(self, tfidf_matrix, labels) -> float:
        """Calculate cluster quality using silhouette score."""
        if not HAS_SKLEARN:
            return 0.5
        
        try:
            from sklearn.metrics import silhouette_score
            
            if len(set(labels)) < 2:
                return 0.5
            
            score = silhouette_score(tfidf_matrix, labels)
            return max(0.0, min(1.0, (score + 1) / 2))  # Normalize to 0-1
            
        except Exception:
            return 0.5
    
    def _fallback_clustering(
        self, 
        segments: List[TranscriptSegment], 
        target_clusters: int
    ) -> ClusteringResult:
        """Fallback clustering when TF-IDF is not available."""
        start_time = datetime.utcnow()
        
        # Simple time-based clustering
        if not segments:
            return ClusteringResult([], [], 0.0, "FallbackClustering", 0.0, {})
        
        total_duration = segments[-1].start_time - segments[0].start_time
        cluster_duration = total_duration / target_clusters
        
        clusters = []
        cluster_assignments = []
        cluster_id = 0
        
        current_start = segments[0].start_time
        
        while current_start < segments[-1].start_time and cluster_id < target_clusters:
            cluster_end = current_start + cluster_duration
            
            # Find segments in this time range
            cluster_segments = []
            for i, segment in enumerate(segments):
                if current_start <= segment.start_time < cluster_end:
                    cluster_segments.append(segment)
                    cluster_assignments.append(cluster_id)
            
            if cluster_segments:
                cluster = SemanticCluster(
                    cluster_id=f"fallback_cluster_{cluster_id}",
                    segments=cluster_segments,
                    theme=f"Segment {cluster_id + 1}",
                    importance_score=0.5,
                    start_time=current_start,
                    end_time=min(cluster_end, segments[-1].start_time),
                    summary=f"Content segment {cluster_id + 1}",
                    keywords=[]
                )
                clusters.append(cluster)
            
            current_start = cluster_end
            cluster_id += 1
        
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        return ClusteringResult(
            clusters=clusters,
            cluster_assignments=cluster_assignments,
            cluster_quality_score=0.5,
            algorithm_used="FallbackClustering",
            processing_time=processing_time,
            metadata={"method": "time_based_fallback"}
        )
    
    def _detect_content_type(self, text: str) -> str:
        """Detect the type of content in the text."""
        text_lower = text.lower()
        
        # Introduction indicators
        intro_keywords = ['welcome', 'introduction', 'hello', 'today', 'going to', 'start']
        if any(keyword in text_lower for keyword in intro_keywords):
            return "introduction"
        
        # Conclusion indicators
        conclusion_keywords = ['conclusion', 'summary', 'finally', 'to conclude', 'thank you', 'that\'s all']
        if any(keyword in text_lower for keyword in conclusion_keywords):
            return "conclusion"
        
        # Transition indicators
        transition_keywords = ['next', 'now', 'moving on', 'let\'s', 'another', 'also']
        if any(keyword in text_lower for keyword in transition_keywords):
            return "transition"
        
        return "main_content"
    
    def _extract_importance_indicators(self, text: str) -> List[str]:
        """Extract indicators of content importance."""
        indicators = []
        text_lower = text.lower()
        
        # Emphasis indicators
        emphasis_patterns = [
            r'\bimportant\b', r'\bkey\b', r'\bcrucial\b', r'\bessential\b',
            r'\bmain\b', r'\bprimary\b', r'\bcentral\b', r'\bfundamental\b',
            r'\bremember\b', r'\bnote\b', r'\bnotice\b', r'\bobserve\b'
        ]
        
        for pattern in emphasis_patterns:
            if re.search(pattern, text_lower):
                indicators.append(pattern.replace(r'\b', '').replace('\\', ''))
        
        return indicators
    
    def _calculate_complexity_score(self, text: str) -> float:
        """Calculate text complexity score."""
        words = text.split()
        if not words:
            return 0.0
        
        # Factors contributing to complexity
        avg_word_length = sum(len(word) for word in words) / len(words)
        unique_words_ratio = len(set(words)) / len(words)
        sentence_count = text.count('.') + text.count('!') + text.count('?')
        avg_sentence_length = len(words) / max(sentence_count, 1)
        
        # Normalize and combine factors
        complexity = (
            min(avg_word_length / 10, 1.0) * 0.3 +
            unique_words_ratio * 0.3 +
            min(avg_sentence_length / 20, 1.0) * 0.4
        )
        
        return min(1.0, complexity)
    
    def _extract_topics(self, text: str) -> List[str]:
        """Extract topics from text."""
        # Simple noun phrase extraction
        words = re.findall(r'\b[A-Z][a-z]+\b', text)  # Capitalized words
        nouns = re.findall(r'\b\w+ing\b|\b\w+tion\b|\b\w+ment\b', text.lower())  # Common noun patterns
        
        topics = list(set(words + nouns))[:5]
        return topics
    
    def _extract_cluster_theme(self, text: str) -> str:
        """Extract main theme from cluster text."""
        # Find most frequent meaningful words
        words = re.findall(r'\b\w+\b', text.lower())
        filtered_words = [w for w in words if len(w) > 3 and w not in self.stop_words]
        
        if not filtered_words:
            return "Content Section"
        
        word_freq = Counter(filtered_words)
        top_words = [word for word, _ in word_freq.most_common(3)]
        
        return " ".join(top_words).title()
    
    def _generate_cluster_summary(self, segments: List[TranscriptSegment]) -> str:
        """Generate summary for a cluster of segments."""
        if not segments:
            return "Empty cluster"
        
        # Use the longest segment as representative
        longest_segment = max(segments, key=lambda x: len(x.text))
        
        # Truncate to reasonable length
        summary = longest_segment.text
        if len(summary) > 100:
            summary = summary[:97] + "..."
        
        return summary
    
    def _extract_cluster_keywords(self, text: str) -> List[str]:
        """Extract keywords for a cluster."""
        words = re.findall(r'\b\w+\b', text.lower())
        filtered_words = [w for w in words if len(w) > 3 and w not in self.stop_words]
        
        word_freq = Counter(filtered_words)
        keywords = [word for word, _ in word_freq.most_common(5)]
        
        return keywords
    
    def _calculate_cluster_importance(self, segments: List[TranscriptSegment]) -> float:
        """Calculate importance score for a cluster."""
        if not segments:
            return 0.0
        
        # Factors for importance
        total_words = sum(len(segment.text.split()) for segment in segments)
        avg_words_per_segment = total_words / len(segments)
        cluster_duration = sum(segment.duration or 5 for segment in segments)
        
        # Check for importance indicators
        all_text = " ".join(segment.text for segment in segments).lower()
        importance_keywords = ['important', 'key', 'main', 'crucial', 'essential', 'remember']
        importance_count = sum(all_text.count(keyword) for keyword in importance_keywords)
        
        # Calculate base importance
        word_factor = min(avg_words_per_segment / 20, 1.0)  # Normalize to 20 words
        duration_factor = min(cluster_duration / 60, 1.0)   # Normalize to 60 seconds
        importance_factor = min(importance_count / 3, 1.0)  # Normalize to 3 keywords
        
        importance_score = (word_factor * 0.4 + duration_factor * 0.3 + importance_factor * 0.3)
        
        return min(1.0, max(0.1, importance_score))


class ContentStructureAlgorithm(SemanticGroupingAlgorithm):
    """
    Content structure-based clustering algorithm.
    
    Groups content based on structural patterns and discourse markers
    rather than pure semantic similarity.
    """
    
    def __init__(self):
        super().__init__("ContentStructure")
        
        # Discourse markers for different content types
        self.introduction_markers = [
            'welcome', 'hello', 'today', 'introduction', 'overview', 'agenda',
            'going to discuss', 'start with', 'begin'
        ]
        
        self.transition_markers = [
            'next', 'now', 'moving on', 'let\'s', 'another', 'also', 'furthermore',
            'in addition', 'however', 'but', 'on the other hand'
        ]
        
        self.conclusion_markers = [
            'conclusion', 'summary', 'finally', 'to conclude', 'in summary',
            'wrap up', 'that\'s all', 'thank you', 'questions'
        ]
        
        self.emphasis_markers = [
            'important', 'key', 'remember', 'note', 'crucial', 'essential',
            'main point', 'focus on', 'pay attention'
        ]
    
    def cluster_segments(
        self, 
        segments: List[TranscriptSegment], 
        target_clusters: int = 5,
        **kwargs
    ) -> ClusteringResult:
        """Cluster segments based on content structure."""
        start_time = datetime.utcnow()
        
        # Analyze content structure
        segment_features = [self.extract_features(segment) for segment in segments]
        
        # Create clusters based on content types and transitions
        clusters = self._create_structure_based_clusters(segments, segment_features)
        
        # Create cluster assignments
        cluster_assignments = self._assign_segments_to_clusters(segments, clusters)
        
        # Calculate quality score
        quality_score = self._calculate_structure_quality(clusters)
        
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        return ClusteringResult(
            clusters=clusters,
            cluster_assignments=cluster_assignments,
            cluster_quality_score=quality_score,
            algorithm_used="ContentStructure",
            processing_time=processing_time,
            metadata={
                "structure_types": [cluster.theme for cluster in clusters],
                "transition_points": len([f for f in segment_features if f.content_type == "transition"]),
                "emphasis_points": sum(len(f.importance_indicators) for f in segment_features)
            }
        )
    
    def extract_features(self, segment: TranscriptSegment) -> SemanticFeatures:
        """Extract content structure features."""
        text = segment.text.lower()
        
        # Detect content type
        content_type = self._classify_content_type(text)
        
        # Extract importance indicators
        importance_indicators = []
        for marker in self.emphasis_markers:
            if marker in text:
                importance_indicators.append(marker)
        
        # Extract keywords (nouns and important terms)
        keywords = self._extract_structural_keywords(text)
        
        # Calculate importance based on structural position
        importance_score = self._calculate_structural_importance(content_type, importance_indicators)
        
        return SemanticFeatures(
            keywords=keywords,
            topics=[],
            sentiment_score=0.5,
            complexity_score=self._calculate_structural_complexity(text),
            importance_indicators=importance_indicators,
            content_type=content_type,
            speaker_indicators=[]
        )
    
    def _classify_content_type(self, text: str) -> str:
        """Classify content type based on discourse markers."""
        # Check for introduction markers
        if any(marker in text for marker in self.introduction_markers):
            return "introduction"
        
        # Check for conclusion markers
        if any(marker in text for marker in self.conclusion_markers):
            return "conclusion"
        
        # Check for transition markers
        if any(marker in text for marker in self.transition_markers):
            return "transition"
        
        return "main_content"
    
    def _extract_structural_keywords(self, text: str) -> List[str]:
        """Extract keywords based on structural importance."""
        # Find capitalized words (often important nouns)
        capitalized = re.findall(r'\b[A-Z][a-z]+\b', text)
        
        # Find repeated words (often important concepts)
        words = re.findall(r'\b\w+\b', text.lower())
        word_freq = Counter(words)
        repeated = [word for word, count in word_freq.items() if count > 1 and len(word) > 3]
        
        # Combine and deduplicate
        keywords = list(set(capitalized + repeated))[:10]
        return keywords
    
    def _calculate_structural_importance(self, content_type: str, indicators: List[str]) -> float:
        """Calculate importance based on structural position."""
        base_scores = {
            "introduction": 0.8,
            "conclusion": 0.7,
            "transition": 0.4,
            "main_content": 0.6
        }
        
        base_score = base_scores.get(content_type, 0.5)
        indicator_bonus = min(len(indicators) * 0.1, 0.3)
        
        return min(1.0, base_score + indicator_bonus)
    
    def _calculate_structural_complexity(self, text: str) -> float:
        """Calculate complexity based on structural elements."""
        # Count complex structures
        questions = text.count('?')
        exclamations = text.count('!')
        subordinate_clauses = text.count(' that ') + text.count(' which ') + text.count(' where ')
        
        complexity_factors = questions + exclamations + subordinate_clauses
        word_count = len(text.split())
        
        if word_count == 0:
            return 0.0
        
        complexity_ratio = complexity_factors / word_count
        return min(1.0, complexity_ratio * 10)  # Scale appropriately
    
    def _create_structure_based_clusters(
        self, 
        segments: List[TranscriptSegment], 
        features: List[SemanticFeatures]
    ) -> List[SemanticCluster]:
        """Create clusters based on structural analysis."""
        clusters = []
        current_cluster_segments = []
        current_cluster_type = None
        cluster_id = 0
        
        for segment, feature in zip(segments, features):
            # Start new cluster on content type change or transition
            if (current_cluster_type is None or 
                feature.content_type != current_cluster_type or
                feature.content_type == "transition"):
                
                # Save previous cluster if it exists
                if current_cluster_segments:
                    cluster = self._create_cluster_from_segments(
                        current_cluster_segments, 
                        current_cluster_type,
                        cluster_id
                    )
                    clusters.append(cluster)
                    cluster_id += 1
                
                # Start new cluster
                current_cluster_segments = [segment]
                current_cluster_type = feature.content_type
            else:
                current_cluster_segments.append(segment)
        
        # Add final cluster
        if current_cluster_segments:
            cluster = self._create_cluster_from_segments(
                current_cluster_segments, 
                current_cluster_type,
                cluster_id
            )
            clusters.append(cluster)
        
        return clusters
    
    def _create_cluster_from_segments(
        self, 
        segments: List[TranscriptSegment], 
        content_type: str,
        cluster_id: int
    ) -> SemanticCluster:
        """Create a semantic cluster from segments."""
        if not segments:
            raise ValueError("Cannot create cluster from empty segments")
        
        start_time = min(seg.start_time for seg in segments)
        end_time = max(seg.end_time or seg.start_time + 5 for seg in segments)
        
        # Generate theme based on content type
        theme_map = {
            "introduction": "Introduction",
            "conclusion": "Conclusion",
            "transition": "Transition",
            "main_content": f"Main Content {cluster_id + 1}"
        }
        theme = theme_map.get(content_type, f"Section {cluster_id + 1}")
        
        # Generate summary
        summary = self._generate_structure_summary(segments, content_type)
        
        # Extract keywords
        all_text = " ".join(seg.text for seg in segments)
        keywords = self._extract_structural_keywords(all_text.lower())
        
        # Calculate importance
        importance_score = self._calculate_cluster_structural_importance(segments, content_type)
        
        return SemanticCluster(
            cluster_id=f"structure_cluster_{cluster_id}",
            segments=segments,
            theme=theme,
            importance_score=importance_score,
            start_time=start_time,
            end_time=end_time,
            summary=summary,
            keywords=keywords
        )
    
    def _generate_structure_summary(self, segments: List[TranscriptSegment], content_type: str) -> str:
        """Generate summary based on structural role."""
        if not segments:
            return "Empty section"
        
        content_descriptions = {
            "introduction": "Opening remarks and topic introduction",
            "conclusion": "Summary and closing statements",
            "transition": "Transition between topics",
            "main_content": "Main discussion and key points"
        }
        
        base_description = content_descriptions.get(content_type, "Content section")
        
        # Add specific details from longest segment
        longest_segment = max(segments, key=lambda x: len(x.text))
        if len(longest_segment.text) > 50:
            specific_content = longest_segment.text[:50] + "..."
            return f"{base_description}: {specific_content}"
        
        return base_description
    
    def _calculate_cluster_structural_importance(
        self, 
        segments: List[TranscriptSegment], 
        content_type: str
    ) -> float:
        """Calculate importance based on structural role."""
        base_importance = {
            "introduction": 0.8,
            "conclusion": 0.7,
            "main_content": 0.6,
            "transition": 0.4
        }
        
        # Factor in cluster size and duration
        segment_count_factor = min(len(segments) / 5, 0.2)  # Bonus for substantial clusters
        total_duration = sum(seg.duration or 5 for seg in segments)
        duration_factor = min(total_duration / 60, 0.2)     # Bonus for longer clusters
        
        importance = base_importance.get(content_type, 0.5)
        importance += segment_count_factor + duration_factor
        
        return min(1.0, importance)
    
    def _assign_segments_to_clusters(
        self, 
        segments: List[TranscriptSegment], 
        clusters: List[SemanticCluster]
    ) -> List[int]:
        """Create cluster assignment list."""
        assignments = []
        
        for segment in segments:
            # Find which cluster this segment belongs to
            for i, cluster in enumerate(clusters):
                if segment in cluster.segments:
                    assignments.append(i)
                    break
            else:
                # Segment not found in any cluster (shouldn't happen)
                assignments.append(0)
        
        return assignments
    
    def _calculate_structure_quality(self, clusters: List[SemanticCluster]) -> float:
        """Calculate quality score for structural clustering."""
        if not clusters:
            return 0.0
        
        # Quality factors
        type_diversity = len(set(cluster.theme.split()[0] for cluster in clusters))
        avg_importance = sum(cluster.importance_score for cluster in clusters) / len(clusters)
        size_balance = 1.0 - abs(len(clusters) - 5) / 10  # Prefer ~5 clusters
        
        quality = (type_diversity / 4 * 0.3 + avg_importance * 0.4 + size_balance * 0.3)
        return min(1.0, max(0.0, quality))


class EmbeddingClusteringAlgorithm(SemanticGroupingAlgorithm):
    """
    Embedding-based clustering algorithm using sentence transformers.
    
    Uses pre-trained sentence embeddings for semantic similarity calculation
    and clustering, providing state-of-the-art semantic understanding.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        super().__init__("EmbeddingClustering")
        self.model_name = model_name
        self.model = None
        self.embeddings_cache = {}
        
        # Configuration
        self.similarity_threshold = 0.7
        self.min_cluster_size = 2
        self.max_cluster_size = 10
        
        # Initialize model if available
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the sentence transformer model."""
        if not HAS_SENTENCE_TRANSFORMERS:
            self.logger.warning("sentence-transformers not available, embedding clustering disabled")
            return
        
        try:
            self.model = SentenceTransformer(self.model_name)
            self.logger.info(f"Initialized embedding model: {self.model_name}")
        except Exception as e:
            self.logger.error(f"Failed to initialize embedding model: {str(e)}")
            self.model = None
    
    def cluster_segments(
        self, 
        segments: List[TranscriptSegment], 
        target_clusters: int = 5,
        **kwargs
    ) -> ClusteringResult:
        """Cluster segments using embedding-based similarity."""
        start_time = datetime.utcnow()
        
        if not self.model:
            self.logger.warning("Embedding model not available, using fallback")
            return self._fallback_clustering(segments, target_clusters)
        
        try:
            # Extract text from segments
            texts = [segment.text for segment in segments]
            
            # Generate embeddings
            embeddings = self._get_embeddings(texts)
            
            # Perform clustering based on embedding similarity
            cluster_labels = self._cluster_embeddings(embeddings, target_clusters)
            
            # Create semantic clusters
            clusters = self._create_clusters_from_embeddings(segments, cluster_labels, embeddings)
            
            # Calculate quality score using embedding similarity
            quality_score = self._calculate_embedding_quality(embeddings, cluster_labels)
            
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            return ClusteringResult(
                clusters=clusters,
                cluster_assignments=cluster_labels,
                cluster_quality_score=quality_score,
                algorithm_used="EmbeddingClustering",
                processing_time=processing_time,
                metadata={
                    "model_name": self.model_name,
                    "embedding_dim": embeddings.shape[1] if HAS_SKLEARN and hasattr(embeddings, 'shape') else 0,
                    "similarity_threshold": self.similarity_threshold,
                    "target_clusters": target_clusters,
                    "actual_clusters": len(clusters)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Embedding clustering failed: {str(e)}")
            return self._fallback_clustering(segments, target_clusters)
    
    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for texts with caching."""
        embeddings = []
        
        for text in texts:
            # Check cache first
            if text in self.embeddings_cache:
                embeddings.append(self.embeddings_cache[text])
            else:
                # Generate new embedding
                embedding = self.model.encode(text, convert_to_numpy=True)
                self.embeddings_cache[text] = embedding
                embeddings.append(embedding)
        
        if HAS_SKLEARN:
            return np.array(embeddings)
        else:
            # Fallback without numpy
            return embeddings
    
    def _cluster_embeddings(self, embeddings, target_clusters: int) -> List[int]:
        """Cluster embeddings using similarity-based approach."""
        if not HAS_SKLEARN:
            return self._simple_similarity_clustering(embeddings, target_clusters)
        
        try:
            # Use K-means clustering on embeddings
            kmeans = KMeans(n_clusters=target_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings)
            return cluster_labels.tolist()
            
        except Exception as e:
            self.logger.warning(f"K-means clustering failed: {str(e)}, using similarity clustering")
            return self._simple_similarity_clustering(embeddings, target_clusters)
    
    def _simple_similarity_clustering(self, embeddings: List, target_clusters: int) -> List[int]:
        """Simple similarity-based clustering without sklearn."""
        if not embeddings:
            return []
        
        clusters = []
        cluster_labels = [-1] * len(embeddings)
        current_cluster_id = 0
        
        for i, embedding in enumerate(embeddings):
            if cluster_labels[i] != -1:
                continue  # Already assigned
            
            # Start new cluster
            cluster_members = [i]
            cluster_labels[i] = current_cluster_id
            
            # Find similar embeddings
            for j in range(i + 1, len(embeddings)):
                if cluster_labels[j] != -1:
                    continue
                
                # Calculate similarity (simple dot product)
                similarity = self._calculate_similarity(embedding, embeddings[j])
                
                if similarity > self.similarity_threshold:
                    cluster_members.append(j)
                    cluster_labels[j] = current_cluster_id
            
            clusters.append(cluster_members)
            current_cluster_id += 1
            
            # Limit number of clusters
            if current_cluster_id >= target_clusters:
                break
        
        # Assign remaining items to closest cluster
        for i, label in enumerate(cluster_labels):
            if label == -1:
                cluster_labels[i] = 0  # Assign to first cluster
        
        return cluster_labels
    
    def _calculate_similarity(self, emb1, emb2) -> float:
        """Calculate similarity between two embeddings."""
        if HAS_SKLEARN:
            # Use cosine similarity
            return float(cosine_similarity([emb1], [emb2])[0][0])
        else:
            # Simple dot product similarity
            if hasattr(emb1, '__iter__') and hasattr(emb2, '__iter__'):
                dot_product = sum(a * b for a, b in zip(emb1, emb2))
                norm1 = sum(a * a for a in emb1) ** 0.5
                norm2 = sum(b * b for b in emb2) ** 0.5
                if norm1 > 0 and norm2 > 0:
                    return dot_product / (norm1 * norm2)
            return 0.5  # Default similarity
    
    def _create_clusters_from_embeddings(
        self, 
        segments: List[TranscriptSegment], 
        labels: List[int],
        embeddings
    ) -> List[SemanticCluster]:
        """Create semantic clusters from embedding clustering results."""
        cluster_dict = defaultdict(list)
        cluster_embeddings = defaultdict(list)
        
        for segment, label, embedding in zip(segments, labels, embeddings):
            cluster_dict[label].append(segment)
            cluster_embeddings[label].append(embedding)
        
        clusters = []
        for cluster_id, cluster_segments in cluster_dict.items():
            if not cluster_segments:
                continue
            
            # Calculate cluster properties
            start_time = min(seg.start_time for seg in cluster_segments)
            end_time = max(seg.end_time or seg.start_time + 5 for seg in cluster_segments)
            
            # Extract cluster theme using embeddings
            cluster_embs = cluster_embeddings[cluster_id]
            theme = self._extract_embedding_theme(cluster_segments, cluster_embs)
            
            # Generate summary
            summary = self._generate_embedding_summary(cluster_segments)
            
            # Extract keywords
            keywords = self._extract_embedding_keywords(cluster_segments)
            
            # Calculate importance score
            importance_score = self._calculate_embedding_importance(cluster_segments, cluster_embs)
            
            cluster = SemanticCluster(
                cluster_id=f"embedding_cluster_{cluster_id}",
                segments=cluster_segments,
                theme=theme,
                importance_score=importance_score,
                start_time=start_time,
                end_time=end_time,
                summary=summary,
                keywords=keywords
            )
            
            clusters.append(cluster)
        
        # Sort clusters by start time
        clusters.sort(key=lambda x: x.start_time)
        
        return clusters
    
    def _calculate_embedding_quality(self, embeddings, labels: List[int]) -> float:
        """Calculate clustering quality using embedding similarity."""
        if not HAS_SKLEARN or len(set(labels)) < 2:
            return 0.5
        
        try:
            from sklearn.metrics import silhouette_score
            score = silhouette_score(embeddings, labels)
            return max(0.0, min(1.0, (score + 1) / 2))  # Normalize to 0-1
        except Exception:
            return 0.5
    
    def _extract_embedding_theme(self, segments: List[TranscriptSegment], embeddings: List) -> str:
        """Extract theme using embedding analysis."""
        # Find the most central segment (closest to cluster centroid)
        if not segments or not embeddings:
            return "Content Section"
        
        if HAS_SKLEARN and len(embeddings) > 1:
            # Calculate centroid
            centroid = np.mean(embeddings, axis=0)
            
            # Find closest segment to centroid
            similarities = [self._calculate_similarity(emb, centroid) for emb in embeddings]
            best_idx = similarities.index(max(similarities))
            representative_text = segments[best_idx].text
        else:
            # Use longest segment as representative
            representative_text = max(segments, key=lambda x: len(x.text)).text
        
        # Extract key terms from representative text
        words = re.findall(r'\b[A-Za-z]+\b', representative_text.lower())
        filtered_words = [w for w in words if len(w) > 3][:3]
        
        if filtered_words:
            return " ".join(filtered_words).title()
        else:
            return "Content Section"
    
    def _generate_embedding_summary(self, segments: List[TranscriptSegment]) -> str:
        """Generate summary for embedding-based cluster."""
        if not segments:
            return "Empty cluster"
        
        # Use the segment with most words as summary base
        best_segment = max(segments, key=lambda x: len(x.text.split()))
        summary = best_segment.text
        
        if len(summary) > 120:
            summary = summary[:117] + "..."
        
        return summary
    
    def _extract_embedding_keywords(self, segments: List[TranscriptSegment]) -> List[str]:
        """Extract keywords from embedding cluster."""
        all_text = " ".join(seg.text for seg in segments).lower()
        words = re.findall(r'\b[a-z]+\b', all_text)
        
        # Filter meaningful words
        filtered_words = [w for w in words if len(w) > 3]
        word_freq = Counter(filtered_words)
        
        # Get top keywords
        keywords = [word for word, _ in word_freq.most_common(8)]
        return keywords
    
    def _calculate_embedding_importance(
        self, 
        segments: List[TranscriptSegment], 
        embeddings: List
    ) -> float:
        """Calculate importance score for embedding cluster."""
        if not segments:
            return 0.0
        
        # Base importance on cluster characteristics
        avg_length = sum(len(seg.text.split()) for seg in segments) / len(segments)
        cluster_duration = sum(seg.duration or 5 for seg in segments)
        
        # Embedding-specific factors
        embedding_diversity = self._calculate_embedding_diversity(embeddings)
        
        # Combine factors
        length_factor = min(avg_length / 25, 1.0)
        duration_factor = min(cluster_duration / 90, 1.0)
        diversity_factor = embedding_diversity
        
        importance = (length_factor * 0.4 + duration_factor * 0.3 + diversity_factor * 0.3)
        return min(1.0, max(0.1, importance))
    
    def _calculate_embedding_diversity(self, embeddings: List) -> float:
        """Calculate diversity within cluster embeddings."""
        if len(embeddings) < 2:
            return 0.5
        
        if not HAS_SKLEARN:
            return 0.5  # Default diversity
        
        try:
            # Calculate pairwise similarities
            similarities = []
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    sim = self._calculate_similarity(embeddings[i], embeddings[j])
                    similarities.append(sim)
            
            if similarities:
                avg_similarity = sum(similarities) / len(similarities)
                diversity = 1.0 - avg_similarity  # Higher diversity = lower average similarity
                return max(0.0, min(1.0, diversity))
        except Exception:
            pass
        
        return 0.5
    
    def _fallback_clustering(
        self, 
        segments: List[TranscriptSegment], 
        target_clusters: int
    ) -> ClusteringResult:
        """Fallback clustering when embeddings are not available."""
        # Use TF-IDF fallback
        tfidf_algorithm = TfidfClusteringAlgorithm()
        result = tfidf_algorithm.cluster_segments(segments, target_clusters)
        result.algorithm_used = "EmbeddingFallbackToTfidf"
        return result
    
    def extract_features(self, segment: TranscriptSegment) -> SemanticFeatures:
        """Extract features using embeddings."""
        if not self.model:
            # Fallback to simple feature extraction
            return self._extract_simple_features(segment)
        
        text = segment.text
        
        # Generate embedding
        embedding = self.model.encode(text, convert_to_numpy=True)
        
        # Extract features based on embedding analysis
        keywords = self._extract_embedding_keywords([segment])
        complexity_score = self._estimate_complexity_from_embedding(embedding)
        
        return SemanticFeatures(
            keywords=keywords,
            topics=[],
            sentiment_score=0.5,  # Could be enhanced with sentiment analysis
            complexity_score=complexity_score,
            importance_indicators=[],
            content_type="main_content",
            speaker_indicators=[]
        )
    
    def _extract_simple_features(self, segment: TranscriptSegment) -> SemanticFeatures:
        """Simple feature extraction fallback."""
        text = segment.text.lower()
        words = re.findall(r'\b\w+\b', text)
        keywords = [w for w in words if len(w) > 3][:5]
        
        return SemanticFeatures(
            keywords=keywords,
            topics=[],
            sentiment_score=0.5,
            complexity_score=len(words) / 20.0,
            importance_indicators=[],
            content_type="main_content",
            speaker_indicators=[]
        )
    
    def _estimate_complexity_from_embedding(self, embedding) -> float:
        """Estimate text complexity from embedding features."""
        if not HAS_SKLEARN or not hasattr(embedding, '__iter__'):
            return 0.5
        
        try:
            # Use embedding magnitude as complexity indicator
            magnitude = np.linalg.norm(embedding)
            # Normalize to 0-1 range (typical embeddings have magnitude around 1-10)
            complexity = min(1.0, magnitude / 10.0)
            return complexity
        except Exception:
            return 0.5


class SemanticAnalyzer:
    """
    Main semantic analyzer that combines multiple algorithms.
    
    Provides a unified interface for different semantic grouping approaches
    and can automatically select the best algorithm for the content.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Available algorithms
        self.algorithms = {
            'tfidf': TfidfClusteringAlgorithm(),
            'structure': ContentStructureAlgorithm(),
            'embedding': EmbeddingClusteringAlgorithm()
        }
        
        # Configuration
        self.default_algorithm = 'embedding' if HAS_SENTENCE_TRANSFORMERS else 'tfidf'
        self.fallback_algorithm = 'structure'
    
    def analyze_segments(
        self, 
        segments: List[TranscriptSegment],
        algorithm: str = None,
        target_clusters: int = 5,
        **kwargs
    ) -> ClusteringResult:
        """
        Analyze segments using specified or optimal algorithm.
        
        Args:
            segments: List of transcript segments
            algorithm: Algorithm to use ('tfidf', 'structure', or 'auto')
            target_clusters: Desired number of clusters
            **kwargs: Algorithm-specific parameters
            
        Returns:
            ClusteringResult with analysis results
        """
        if not segments:
            return ClusteringResult([], [], 0.0, "empty", 0.0, {})
        
        # Select algorithm
        if algorithm is None or algorithm == 'auto':
            selected_algorithm = self._select_optimal_algorithm(segments)
        else:
            selected_algorithm = algorithm
        
        if selected_algorithm not in self.algorithms:
            self.logger.warning(f"Unknown algorithm {selected_algorithm}, using default")
            selected_algorithm = self.default_algorithm
        
        # Perform analysis
        try:
            analyzer = self.algorithms[selected_algorithm]
            result = analyzer.cluster_segments(segments, target_clusters, **kwargs)
            
            self.logger.info(f"Semantic analysis completed using {selected_algorithm}: "
                           f"{len(result.clusters)} clusters, quality: {result.cluster_quality_score:.3f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Analysis failed with {selected_algorithm}: {str(e)}")
            
            # Try fallback algorithm
            if selected_algorithm != self.fallback_algorithm:
                try:
                    fallback_analyzer = self.algorithms[self.fallback_algorithm]
                    result = fallback_analyzer.cluster_segments(segments, target_clusters, **kwargs)
                    result.algorithm_used = f"{selected_algorithm}_fallback_to_{self.fallback_algorithm}"
                    return result
                except Exception as fallback_error:
                    self.logger.error(f"Fallback analysis also failed: {str(fallback_error)}")
            
            # Return empty result
            return ClusteringResult([], [], 0.0, "failed", 0.0, {"error": str(e)})
    
    def _select_optimal_algorithm(self, segments: List[TranscriptSegment]) -> str:
        """Select the most appropriate algorithm for the content."""
        # Analyze content characteristics
        total_text = " ".join(segment.text for segment in segments)
        word_count = len(total_text.split())
        
        # Check for structural indicators
        structure_indicators = self._count_structure_indicators(total_text)
        
        # Decision logic with embedding preference
        if HAS_SENTENCE_TRANSFORMERS and word_count > 50:
            # Use embeddings for substantial content when available
            return 'embedding'
        elif word_count < 100:
            # Short content: use structure-based
            return 'structure'
        elif structure_indicators > 3:
            # Content with clear structure: use structure-based
            return 'structure'
        elif HAS_SKLEARN and word_count > 200:
            # Long content with sklearn available: use TF-IDF
            return 'tfidf'
        else:
            # Default to structure-based
            return 'structure'
    
    def _count_structure_indicators(self, text: str) -> int:
        """Count indicators of structured content."""
        text_lower = text.lower()
        
        indicators = [
            'introduction', 'conclusion', 'summary', 'next', 'first', 'second',
            'finally', 'important', 'key', 'main', 'welcome', 'thank you'
        ]
        
        count = sum(text_lower.count(indicator) for indicator in indicators)
        return count
    
    def get_available_algorithms(self) -> List[str]:
        """Get list of available algorithms."""
        return list(self.algorithms.keys())
    
    def get_algorithm_info(self, algorithm: str) -> Dict[str, Any]:
        """Get information about a specific algorithm."""
        if algorithm not in self.algorithms:
            return {}
        
        analyzer = self.algorithms[algorithm]
        return {
            'name': analyzer.name,
            'type': type(analyzer).__name__,
            'description': analyzer.__class__.__doc__,
            'available': True
        }


# Factory functions for easy instantiation
def create_semantic_analyzer() -> SemanticAnalyzer:
    """Create a new semantic analyzer instance."""
    return SemanticAnalyzer()


def create_tfidf_analyzer() -> TfidfClusteringAlgorithm:
    """Create a TF-IDF clustering analyzer."""
    return TfidfClusteringAlgorithm()


def create_structure_analyzer() -> ContentStructureAlgorithm:
    """Create a content structure analyzer."""
    return ContentStructureAlgorithm()


def create_embedding_analyzer(model_name: str = "all-MiniLM-L6-v2") -> EmbeddingClusteringAlgorithm:
    """Create an embedding-based analyzer."""
    return EmbeddingClusteringAlgorithm(model_name)