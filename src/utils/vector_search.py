"""
Vector Search Utility for YouTube Video Summarizer.

This module implements embedding-based vector search capabilities for finding
semantically similar content segments and performing intelligent timestamp matching.
"""

import logging
import json
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
from dataclasses import dataclass

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_EMBEDDING_SUPPORT = True
except ImportError:
    HAS_EMBEDDING_SUPPORT = False

from ..services.semantic_analysis_service import TranscriptSegment, SemanticCluster, SemanticTimestamp

logger = logging.getLogger(__name__)


@dataclass
class VectorSearchResult:
    """Result from vector-based semantic search."""
    query: str
    results: List[Dict[str, Any]]
    search_type: str
    processing_time: float
    similarity_scores: List[float]
    metadata: Dict[str, Any]


@dataclass
class EmbeddingIndex:
    """Index structure for embedding-based search."""
    embeddings: List[Any]  # List of embeddings
    segments: List[TranscriptSegment]
    metadata: Dict[str, Any]
    model_name: str
    created_at: datetime


class VectorSearchEngine:
    """
    Vector search engine for semantic content discovery.
    
    Provides embedding-based search capabilities for finding semantically
    similar content within video transcripts and clusters.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the vector search engine.
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        self.logger = logging.getLogger(__name__)
        self.model_name = model_name
        self.model = None
        self.indexes = {}  # Store multiple indexes
        
        # Configuration
        self.similarity_threshold = 0.5
        self.max_results = 10
        self.cache_embeddings = True
        self.embedding_cache = {}
        
        # Initialize model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the sentence transformer model."""
        if not HAS_EMBEDDING_SUPPORT:
            self.logger.warning("Embedding support not available, vector search disabled")
            return
        
        try:
            self.model = SentenceTransformer(self.model_name)
            self.logger.info(f"Initialized vector search model: {self.model_name}")
        except Exception as e:
            self.logger.error(f"Failed to initialize vector search model: {str(e)}")
            self.model = None
    
    def create_index(
        self, 
        segments: List[TranscriptSegment], 
        index_name: str = "default"
    ) -> bool:
        """
        Create a vector index from transcript segments.
        
        Args:
            segments: List of transcript segments to index
            index_name: Name for the index
            
        Returns:
            True if index creation succeeded
        """
        if not self.model:
            self.logger.error("Vector search model not available")
            return False
        
        try:
            start_time = datetime.utcnow()
            
            # Extract texts from segments
            texts = [segment.text for segment in segments]
            
            # Generate embeddings
            embeddings = self._generate_embeddings(texts)
            
            if not embeddings:
                self.logger.error("Failed to generate embeddings")
                return False
            
            # Create index
            index = EmbeddingIndex(
                embeddings=embeddings,
                segments=segments,
                metadata={
                    "segment_count": len(segments),
                    "total_duration": sum(seg.duration or 0 for seg in segments),
                    "avg_segment_length": sum(len(seg.text) for seg in segments) / len(segments) if segments else 0
                },
                model_name=self.model_name,
                created_at=start_time
            )
            
            self.indexes[index_name] = index
            
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            self.logger.info(f"Created vector index '{index_name}' with {len(segments)} segments in {processing_time:.2f}s")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create vector index: {str(e)}")
            return False
    
    def search_similar_segments(
        self, 
        query: Union[str, TranscriptSegment], 
        index_name: str = "default",
        top_k: int = 5,
        min_similarity: float = None
    ) -> VectorSearchResult:
        """
        Search for segments similar to the query.
        
        Args:
            query: Text query or transcript segment to search for
            index_name: Name of the index to search
            top_k: Number of top results to return
            min_similarity: Minimum similarity threshold
            
        Returns:
            VectorSearchResult with search results
        """
        start_time = datetime.utcnow()
        
        if not self.model or index_name not in self.indexes:
            return self._create_empty_search_result(query, "model_or_index_not_available")
        
        try:
            # Extract query text
            if isinstance(query, TranscriptSegment):
                query_text = query.text
            else:
                query_text = str(query)
            
            # Generate query embedding
            query_embedding = self._generate_embeddings([query_text])[0]
            
            # Get index
            index = self.indexes[index_name]
            
            # Calculate similarities
            similarities = self._calculate_similarities(query_embedding, index.embeddings)
            
            # Filter by minimum similarity
            min_sim = min_similarity or self.similarity_threshold
            filtered_results = [
                (i, sim) for i, sim in enumerate(similarities) 
                if sim >= min_sim
            ]
            
            # Sort by similarity (descending)
            filtered_results.sort(key=lambda x: x[1], reverse=True)
            
            # Take top k results
            top_results = filtered_results[:top_k]
            
            # Build result objects
            results = []
            similarity_scores = []
            
            for idx, similarity in top_results:
                segment = index.segments[idx]
                result = {
                    "segment": segment,
                    "similarity": float(similarity),
                    "start_time": segment.start_time,
                    "end_time": segment.end_time,
                    "text": segment.text,
                    "duration": segment.duration,
                    "rank": len(results) + 1
                }
                results.append(result)
                similarity_scores.append(float(similarity))
            
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            return VectorSearchResult(
                query=query_text,
                results=results,
                search_type="similar_segments",
                processing_time=processing_time,
                similarity_scores=similarity_scores,
                metadata={
                    "index_name": index_name,
                    "total_candidates": len(index.segments),
                    "filtered_candidates": len(filtered_results),
                    "min_similarity": min_sim,
                    "model_name": self.model_name
                }
            )
            
        except Exception as e:
            self.logger.error(f"Vector search failed: {str(e)}")
            return self._create_empty_search_result(query, f"search_failed: {str(e)}")
    
    def find_optimal_timestamps(
        self, 
        semantic_clusters: List[SemanticCluster],
        target_count: int = 5,
        diversity_weight: float = 0.3
    ) -> List[SemanticTimestamp]:
        """
        Find optimal timestamps using vector-based analysis.
        
        Args:
            semantic_clusters: List of semantic clusters
            target_count: Number of timestamps to generate
            diversity_weight: Weight for diversity in timestamp selection
            
        Returns:
            List of optimized semantic timestamps
        """
        if not self.model or not semantic_clusters:
            return []
        
        try:
            # Create temporary index from clusters
            all_segments = []
            cluster_map = {}
            
            for cluster in semantic_clusters:
                for segment in cluster.segments:
                    all_segments.append(segment)
                    cluster_map[len(all_segments) - 1] = cluster
            
            if not all_segments:
                return []
            
            # Generate embeddings for all segments
            texts = [seg.text for seg in all_segments]
            embeddings = self._generate_embeddings(texts)
            
            if not embeddings:
                return []
            
            # Select diverse, high-importance timestamps
            selected_indices = self._select_diverse_timestamps(
                embeddings, 
                all_segments, 
                cluster_map, 
                target_count, 
                diversity_weight
            )
            
            # Convert to semantic timestamps
            timestamps = []
            for idx in selected_indices:
                segment = all_segments[idx]
                cluster = cluster_map[idx]
                
                timestamp = self._create_vector_optimized_timestamp(
                    segment, 
                    cluster, 
                    embeddings[idx],
                    len(timestamps) + 1
                )
                timestamps.append(timestamp)
            
            # Sort by start time
            timestamps.sort(key=lambda x: x.timestamp_seconds)
            
            return timestamps
            
        except Exception as e:
            self.logger.error(f"Optimal timestamp selection failed: {str(e)}")
            return []
    
    def analyze_semantic_coherence(
        self, 
        segments: List[TranscriptSegment]
    ) -> Dict[str, Any]:
        """
        Analyze semantic coherence using vector similarities.
        
        Args:
            segments: List of transcript segments to analyze
            
        Returns:
            Dict with coherence analysis results
        """
        if not self.model or not segments:
            return {"coherence_score": 0.0, "analysis": "no_data"}
        
        try:
            # Generate embeddings
            texts = [seg.text for seg in segments]
            embeddings = self._generate_embeddings(texts)
            
            if len(embeddings) < 2:
                return {"coherence_score": 1.0, "analysis": "single_segment"}
            
            # Calculate pairwise similarities
            similarities = []
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    sim = self._calculate_similarity(embeddings[i], embeddings[j])
                    similarities.append(sim)
            
            # Calculate coherence metrics
            avg_similarity = sum(similarities) / len(similarities)
            max_similarity = max(similarities)
            min_similarity = min(similarities)
            similarity_variance = np.var(similarities) if HAS_NUMPY else 0.0
            
            # Calculate sequential coherence (adjacent segments)
            sequential_similarities = []
            for i in range(len(embeddings) - 1):
                sim = self._calculate_similarity(embeddings[i], embeddings[i + 1])
                sequential_similarities.append(sim)
            
            sequential_coherence = sum(sequential_similarities) / len(sequential_similarities) if sequential_similarities else 0.0
            
            # Overall coherence score
            coherence_score = (avg_similarity * 0.4 + sequential_coherence * 0.6)
            
            return {
                "coherence_score": float(coherence_score),
                "average_similarity": float(avg_similarity),
                "sequential_coherence": float(sequential_coherence),
                "max_similarity": float(max_similarity),
                "min_similarity": float(min_similarity),
                "similarity_variance": float(similarity_variance),
                "segment_count": len(segments),
                "analysis": "completed"
            }
            
        except Exception as e:
            self.logger.error(f"Coherence analysis failed: {str(e)}")
            return {"coherence_score": 0.0, "analysis": f"failed: {str(e)}"}
    
    def _generate_embeddings(self, texts: List[str]) -> List[Any]:
        """Generate embeddings for texts with caching."""
        if not self.model:
            return []
        
        embeddings = []
        
        for text in texts:
            # Check cache
            if self.cache_embeddings and text in self.embedding_cache:
                embeddings.append(self.embedding_cache[text])
            else:
                # Generate new embedding
                try:
                    embedding = self.model.encode(text, convert_to_numpy=True)
                    if self.cache_embeddings:
                        self.embedding_cache[text] = embedding
                    embeddings.append(embedding)
                except Exception as e:
                    self.logger.warning(f"Failed to generate embedding for text: {str(e)}")
                    continue
        
        return embeddings
    
    def _calculate_similarities(self, query_embedding, embeddings: List) -> List[float]:
        """Calculate similarities between query and all embeddings."""
        similarities = []
        
        for embedding in embeddings:
            similarity = self._calculate_similarity(query_embedding, embedding)
            similarities.append(similarity)
        
        return similarities
    
    def _calculate_similarity(self, emb1, emb2) -> float:
        """Calculate cosine similarity between two embeddings."""
        if HAS_EMBEDDING_SUPPORT and HAS_NUMPY:
            try:
                # Use sklearn cosine similarity
                return float(cosine_similarity([emb1], [emb2])[0][0])
            except Exception:
                pass
        
        # Fallback to manual calculation
        if hasattr(emb1, '__iter__') and hasattr(emb2, '__iter__'):
            try:
                dot_product = sum(a * b for a, b in zip(emb1, emb2))
                norm1 = sum(a * a for a in emb1) ** 0.5
                norm2 = sum(b * b for b in emb2) ** 0.5
                if norm1 > 0 and norm2 > 0:
                    return dot_product / (norm1 * norm2)
            except Exception:
                pass
        
        return 0.0
    
    def _select_diverse_timestamps(
        self, 
        embeddings: List, 
        segments: List[TranscriptSegment], 
        cluster_map: Dict[int, SemanticCluster],
        target_count: int,
        diversity_weight: float
    ) -> List[int]:
        """Select diverse timestamps using embedding-based analysis."""
        if len(segments) <= target_count:
            return list(range(len(segments)))
        
        # Calculate importance scores for each segment
        importance_scores = []
        for i, segment in enumerate(segments):
            cluster = cluster_map.get(i)
            if cluster:
                importance = cluster.importance_score
            else:
                importance = 0.5  # Default
            
            # Factor in segment length and duration
            length_factor = min(len(segment.text.split()) / 20, 1.0)
            duration_factor = min((segment.duration or 5) / 30, 1.0)
            
            combined_importance = importance * 0.6 + length_factor * 0.2 + duration_factor * 0.2
            importance_scores.append(combined_importance)
        
        # Select timestamps balancing importance and diversity
        selected_indices = []
        remaining_indices = list(range(len(segments)))
        
        # Always select the most important first
        best_idx = max(remaining_indices, key=lambda i: importance_scores[i])
        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)
        
        # Select remaining timestamps balancing importance and diversity
        while len(selected_indices) < target_count and remaining_indices:
            best_score = -1
            best_idx = None
            
            for idx in remaining_indices:
                # Importance component
                importance_component = importance_scores[idx]
                
                # Diversity component (distance from selected)
                if HAS_NUMPY and embeddings:
                    min_similarity = min(
                        self._calculate_similarity(embeddings[idx], embeddings[selected_idx])
                        for selected_idx in selected_indices
                    )
                    diversity_component = 1.0 - min_similarity
                else:
                    diversity_component = 0.5  # Default diversity
                
                # Combined score
                combined_score = (
                    importance_component * (1 - diversity_weight) + 
                    diversity_component * diversity_weight
                )
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_idx = idx
            
            if best_idx is not None:
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)
            else:
                break
        
        return selected_indices
    
    def _create_vector_optimized_timestamp(
        self, 
        segment: TranscriptSegment, 
        cluster: SemanticCluster,
        embedding: Any,
        position: int
    ) -> SemanticTimestamp:
        """Create an optimized semantic timestamp using vector analysis."""
        # Calculate confidence based on embedding properties
        confidence = self._calculate_embedding_confidence(embedding, cluster)
        
        # Generate description
        description = cluster.summary or f"Key moment: {cluster.theme}"
        if len(description) > 100:
            description = description[:97] + "..."
        
        # Format timestamp
        timestamp_formatted = self._format_timestamp(segment.start_time)
        
        # Generate YouTube URL
        video_id = getattr(segment, 'video_id', 'unknown')
        youtube_url = f"https://www.youtube.com/watch?v={video_id}&t={int(segment.start_time)}s"
        
        # Get context information
        context_before, context_after = self._get_segment_context(segment, cluster)
        
        return SemanticTimestamp(
            timestamp_seconds=segment.start_time,
            timestamp_formatted=timestamp_formatted,
            description=description,
            importance_rating=max(1, min(10, int(cluster.importance_score * 10))),
            youtube_url=youtube_url,
            video_id=video_id,
            semantic_cluster_id=cluster.cluster_id,
            cluster_theme=cluster.theme,
            context_before=context_before,
            context_after=context_after,
            semantic_keywords=cluster.keywords,
            confidence_score=confidence
        )
    
    def _calculate_embedding_confidence(self, embedding: Any, cluster: SemanticCluster) -> float:
        """Calculate confidence score based on embedding characteristics."""
        base_confidence = cluster.importance_score
        
        # Factor in cluster size
        size_factor = min(len(cluster.segments) / 5, 0.2)
        
        # Factor in embedding magnitude (complexity indicator)
        if HAS_NUMPY and hasattr(embedding, '__iter__'):
            try:
                magnitude = np.linalg.norm(embedding)
                magnitude_factor = min(magnitude / 10, 0.1)
            except Exception:
                magnitude_factor = 0.0
        else:
            magnitude_factor = 0.0
        
        confidence = base_confidence + size_factor + magnitude_factor
        return min(1.0, max(0.1, confidence))
    
    def _get_segment_context(
        self, 
        segment: TranscriptSegment, 
        cluster: SemanticCluster
    ) -> Tuple[str, str]:
        """Get context before and after a segment."""
        context_before = ""
        context_after = ""
        
        if cluster.segments:
            sorted_segments = sorted(cluster.segments, key=lambda x: x.start_time)
            
            try:
                segment_index = sorted_segments.index(segment)
                
                # Context before
                if segment_index > 0:
                    before_text = sorted_segments[segment_index - 1].text
                    context_before = (before_text[:80] + "...") if len(before_text) > 80 else before_text
                
                # Context after
                if segment_index < len(sorted_segments) - 1:
                    after_text = sorted_segments[segment_index + 1].text
                    context_after = ("..." + after_text[:80]) if len(after_text) > 80 else ("..." + after_text)
                    
            except ValueError:
                # Segment not found in cluster, use general context
                if cluster.segments:
                    context_before = cluster.segments[0].text[:80] + "..." if len(cluster.segments[0].text) > 80 else cluster.segments[0].text
        
        return context_before, context_after
    
    def _format_timestamp(self, seconds: float) -> str:
        """Format timestamp in MM:SS or HH:MM:SS format."""
        total_seconds = int(seconds)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        
        if hours > 0:
            return f"{hours}:{minutes:02d}:{seconds:02d}"
        else:
            return f"{minutes}:{seconds:02d}"
    
    def _create_empty_search_result(self, query: Union[str, TranscriptSegment], reason: str) -> VectorSearchResult:
        """Create empty search result with error information."""
        query_text = query.text if isinstance(query, TranscriptSegment) else str(query)
        
        return VectorSearchResult(
            query=query_text,
            results=[],
            search_type="empty",
            processing_time=0.0,
            similarity_scores=[],
            metadata={"error": reason}
        )
    
    def get_index_info(self, index_name: str = "default") -> Dict[str, Any]:
        """Get information about a vector index."""
        if index_name not in self.indexes:
            return {"exists": False}
        
        index = self.indexes[index_name]
        return {
            "exists": True,
            "segment_count": len(index.segments),
            "embedding_count": len(index.embeddings),
            "model_name": index.model_name,
            "created_at": index.created_at.isoformat(),
            "metadata": index.metadata
        }
    
    def clear_cache(self):
        """Clear embedding cache."""
        self.embedding_cache.clear()
        self.logger.info("Embedding cache cleared")
    
    def get_cache_size(self) -> int:
        """Get current cache size."""
        return len(self.embedding_cache)


# Factory function for easy instantiation
def create_vector_search_engine(model_name: str = "all-MiniLM-L6-v2") -> VectorSearchEngine:
    """
    Factory function to create a vector search engine.
    
    Args:
        model_name: Name of the sentence transformer model
        
    Returns:
        VectorSearchEngine instance
    """
    return VectorSearchEngine(model_name)