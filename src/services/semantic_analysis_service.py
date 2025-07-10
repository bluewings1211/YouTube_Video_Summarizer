"""
Semantic Analysis Service for YouTube Video Summarizer.

This service implements advanced semantic analysis for timestamped segments,
providing improved accuracy in timestamp generation through semantic grouping
and embedding-based vector search.
"""

import logging
import re
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    # We'll implement fallback methods without numpy

from ..utils.smart_llm_client import SmartLLMClient, TaskRequirements, detect_task_requirements
from ..utils.semantic_analyzer import SemanticAnalyzer, create_semantic_analyzer
from ..utils.vector_search import VectorSearchEngine, create_vector_search_engine

logger = logging.getLogger(__name__)


@dataclass
class TranscriptSegment:
    """Represents a single transcript segment with metadata."""
    start_time: float
    end_time: Optional[float]
    text: str
    duration: Optional[float] = None
    speaker: Optional[str] = None
    
    @property
    def word_count(self) -> int:
        """Get word count of the segment."""
        return len(self.text.split())
    
    @property
    def char_count(self) -> int:
        """Get character count of the segment."""
        return len(self.text)


@dataclass
class SemanticCluster:
    """Represents a semantically related group of transcript segments."""
    cluster_id: str
    segments: List[TranscriptSegment]
    theme: str
    importance_score: float
    start_time: float
    end_time: float
    summary: str
    keywords: List[str]
    
    @property
    def duration(self) -> float:
        """Get total duration of the cluster."""
        return self.end_time - self.start_time
    
    @property
    def word_count(self) -> int:
        """Get total word count of the cluster."""
        return sum(segment.word_count for segment in self.segments)


@dataclass
class SemanticTimestamp:
    """Enhanced timestamp with semantic information."""
    timestamp_seconds: float
    timestamp_formatted: str
    description: str
    importance_rating: int
    youtube_url: str
    video_id: str
    semantic_cluster_id: str
    cluster_theme: str
    context_before: str
    context_after: str
    semantic_keywords: List[str]
    confidence_score: float


class SemanticAnalysisService:
    """
    Service for performing semantic analysis on video transcripts.
    
    This service provides advanced timestamp generation with semantic understanding,
    improving accuracy over simple time-based segmentation.
    """
    
    def __init__(
        self, 
        smart_llm_client: Optional[SmartLLMClient] = None, 
        semantic_analyzer: Optional[SemanticAnalyzer] = None,
        vector_search_engine: Optional[VectorSearchEngine] = None
    ):
        """
        Initialize the semantic analysis service.
        
        Args:
            smart_llm_client: Optional pre-initialized LLM client
            semantic_analyzer: Optional pre-initialized semantic analyzer
            vector_search_engine: Optional pre-initialized vector search engine
        """
        self.logger = logging.getLogger(__name__)
        self.smart_llm_client = smart_llm_client or SmartLLMClient()
        self.semantic_analyzer = semantic_analyzer or create_semantic_analyzer()
        self.vector_search_engine = vector_search_engine or create_vector_search_engine()
        
        # Configuration
        self.min_segment_length = 3  # Minimum words per segment
        self.max_clusters = 10
        self.min_cluster_duration = 10.0  # Minimum seconds for a cluster
        self.importance_threshold = 0.5
        
        # Semantic analysis settings
        self.semantic_similarity_threshold = 0.6
        self.keyword_extraction_count = 5
        
        # Vector search settings
        self.use_vector_optimization = True
        self.vector_diversity_weight = 0.3
        
        # Performance optimization settings
        self.enable_caching = True
        self.enable_parallel_processing = True
        self.max_segments_for_full_analysis = 500  # Switch to sampling for larger transcripts
        self.segment_sampling_ratio = 0.6  # Use 60% of segments when sampling
        self.early_termination_threshold = 0.2  # Quality threshold for early termination
        self.memory_optimization = True
        
        # Caching
        self.analysis_cache = {}
        self.embedding_cache = {}
        self.cluster_cache = {}
        
    def analyze_transcript(
        self, 
        raw_transcript: List[Dict[str, Any]], 
        video_id: str,
        video_title: str = "",
        target_timestamp_count: int = 5
    ) -> Dict[str, Any]:
        """
        Perform comprehensive semantic analysis on transcript data with performance optimizations.
        
        Args:
            raw_transcript: Raw transcript data from YouTube
            video_id: YouTube video ID
            video_title: Video title for context
            target_timestamp_count: Desired number of timestamps
            
        Returns:
            Dict containing semantic analysis results
        """
        start_time = datetime.utcnow()
        self.logger.info(f"Starting optimized semantic analysis for video {video_id}")
        
        try:
            # Check cache first
            cache_key = self._generate_cache_key(raw_transcript, video_id, target_timestamp_count)
            if self.enable_caching and cache_key in self.analysis_cache:
                self.logger.info(f"Returning cached analysis for video {video_id}")
                cached_result = self.analysis_cache[cache_key].copy()
                cached_result['cache_hit'] = True
                return cached_result
            
            # Step 1: Parse and structure transcript segments with optimization
            segments = self._parse_transcript_segments_optimized(raw_transcript)
            if not segments:
                return self._create_empty_result("No valid transcript segments found")
            
            # Early quality assessment - terminate early for low-quality inputs
            if self._should_terminate_early(segments):
                self.logger.info(f"Early termination for video {video_id} - low quality input")
                return self._create_early_termination_result(segments, video_id)
            
            # Step 2: Perform optimized semantic clustering
            clusters = self._perform_semantic_clustering_optimized(segments, video_title)
            if not clusters:
                return self._create_empty_result("No semantic clusters identified")
            
            # Step 3: Extract semantic keywords efficiently
            all_keywords = self._extract_semantic_keywords_optimized(clusters)
            
            # Step 4: Generate enhanced timestamps with performance optimization
            if self.use_vector_optimization and self.vector_search_engine:
                timestamps = self._generate_vector_optimized_timestamps_fast(
                    clusters, 
                    video_id, 
                    target_timestamp_count
                )
            else:
                timestamps = self._generate_semantic_timestamps_fast(
                    clusters, 
                    video_id, 
                    target_timestamp_count
                )
            
            # Step 5: Calculate semantic metrics efficiently
            metrics = self._calculate_semantic_metrics_optimized(segments, clusters, timestamps)
            
            # Step 6: Add vector search analysis if available (with optimization)
            if self.vector_search_engine and len(segments) <= self.max_segments_for_full_analysis:
                try:
                    coherence_analysis = self.vector_search_engine.analyze_semantic_coherence(segments)
                    metrics['vector_analysis'] = coherence_analysis
                except Exception as e:
                    self.logger.warning(f"Vector coherence analysis failed: {str(e)}")
                    metrics['vector_analysis'] = {"analysis": "failed"}
            else:
                metrics['vector_analysis'] = {"analysis": "skipped_for_performance"}
            
            # Calculate processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            result = {
                'status': 'success',
                'video_id': video_id,
                'analysis_timestamp': datetime.utcnow().isoformat(),
                'segments_count': len(segments),
                'clusters_count': len(clusters),
                'timestamps_count': len(timestamps),
                'timestamps': timestamps,
                'semantic_clusters': [self._cluster_to_dict(cluster) for cluster in clusters],
                'semantic_keywords': all_keywords,
                'semantic_metrics': metrics,
                'processing_metadata': {
                    'llm_provider': 'smart_client',
                    'semantic_method': 'optimized_clustering_and_embedding',
                    'confidence_avg': np.mean([ts.confidence_score for ts in timestamps]) if timestamps and HAS_NUMPY else 0.5,
                    'processing_time_seconds': processing_time,
                    'optimization_enabled': True,
                    'segments_processed': len(segments),
                    'cache_hit': False
                }
            }
            
            # Cache the result for future use
            if self.enable_caching:
                self.analysis_cache[cache_key] = result.copy()
                # Limit cache size to prevent memory issues
                if len(self.analysis_cache) > 100:
                    # Remove oldest entries
                    oldest_key = next(iter(self.analysis_cache))
                    del self.analysis_cache[oldest_key]
            
            self.logger.info(f"Optimized semantic analysis completed in {processing_time:.2f}s: {len(timestamps)} timestamps from {len(clusters)} clusters")
            return result
            
        except Exception as e:
            self.logger.error(f"Semantic analysis failed for video {video_id}: {str(e)}")
            return self._create_error_result(str(e))
    
    def _parse_transcript_segments(self, raw_transcript: List[Dict[str, Any]]) -> List[TranscriptSegment]:
        """Parse raw transcript into structured segments."""
        segments = []
        
        for entry in raw_transcript:
            if not isinstance(entry, dict):
                continue
            
            # Extract timing information
            start_time = entry.get('start')
            if start_time is None:
                continue
            
            try:
                start_time = float(start_time)
            except (ValueError, TypeError):
                continue
            
            # Extract text content
            text = entry.get('text', '').strip()
            if len(text.split()) < self.min_segment_length:
                continue
            
            # Calculate end time and duration
            duration = entry.get('duration', 0)
            try:
                duration = float(duration) if duration else 0
            except (ValueError, TypeError):
                duration = 0
            
            end_time = start_time + duration if duration > 0 else None
            
            # Create segment
            segment = TranscriptSegment(
                start_time=start_time,
                end_time=end_time,
                text=text,
                duration=duration,
                speaker=entry.get('speaker')  # In case we have speaker information
            )
            
            segments.append(segment)
        
        # Sort by start time
        segments.sort(key=lambda x: x.start_time)
        
        self.logger.debug(f"Parsed {len(segments)} valid transcript segments")
        return segments
    
    def _perform_semantic_clustering(
        self, 
        segments: List[TranscriptSegment], 
        video_title: str = ""
    ) -> List[SemanticCluster]:
        """
        Group transcript segments into semantic clusters.
        
        This method uses advanced semantic analysis algorithms to identify thematically related segments.
        """
        if not segments:
            return []
        
        try:
            # First try using the advanced semantic analyzer
            target_clusters = min(self.max_clusters, max(3, len(segments) // 3))
            
            # Use semantic analyzer for clustering
            clustering_result = self.semantic_analyzer.analyze_segments(
                segments=segments,
                algorithm='auto',  # Let the analyzer choose the best algorithm
                target_clusters=target_clusters
            )
            
            if clustering_result.clusters and clustering_result.cluster_quality_score > 0.3:
                self.logger.info(f"Advanced semantic clustering succeeded with quality {clustering_result.cluster_quality_score:.3f}")
                return clustering_result.clusters
            else:
                self.logger.warning("Advanced semantic clustering failed or poor quality, trying LLM-based method")
                # Fallback to LLM-based clustering
                return self._llm_based_clustering(segments, video_title)
            
        except Exception as e:
            self.logger.warning(f"Advanced semantic clustering failed: {str(e)}. Using LLM-based fallback.")
            return self._llm_based_clustering(segments, video_title)
    
    def _llm_based_clustering(
        self, 
        segments: List[TranscriptSegment], 
        video_title: str = ""
    ) -> List[SemanticCluster]:
        """
        LLM-based clustering fallback method.
        
        Uses the LLM to analyze and group segments when advanced algorithms fail.
        """
        try:
            # Create text for semantic analysis
            segment_texts = []
            for i, segment in enumerate(segments):
                time_marker = f"[{segment.start_time:.1f}s]"
                segment_texts.append(f"{time_marker} {segment.text}")
            
            # Prepare clustering prompt
            clustering_prompt = self._create_clustering_prompt(segment_texts, video_title)
            
            # Use LLM for semantic clustering
            task_requirements = TaskRequirements(
                complexity_level="medium",
                quality_level="high",
                task_type="analysis",
                max_tokens=1500,
                requires_reasoning=True
            )
            
            result = self.smart_llm_client.generate_text_with_fallback(
                prompt=clustering_prompt,
                task_requirements=task_requirements,
                max_tokens=1500,
                temperature=0.3
            )
            
            # Parse clustering results
            clusters = self._parse_clustering_results(result['text'], segments)
            
            # Filter and enhance clusters
            enhanced_clusters = self._enhance_clusters(clusters, segments)
            
            return enhanced_clusters
            
        except Exception as e:
            self.logger.warning(f"LLM-based clustering failed: {str(e)}. Using time-based fallback.")
            return self._fallback_clustering(segments)
    
    def _create_clustering_prompt(self, segment_texts: List[str], video_title: str) -> str:
        """Create a prompt for LLM-based semantic clustering."""
        segments_text = "\n".join(segment_texts)
        
        prompt = f"""You are an expert at analyzing video content and identifying thematic segments. 

Video Title: {video_title}

Analyze the following transcript segments and group them into semantic clusters based on topics, themes, or content shifts:

{segments_text}

Instructions:
1. Identify 3-7 major semantic clusters/themes in this content
2. Each cluster should represent a distinct topic or content section
3. For each cluster, provide:
   - Theme/topic name (2-4 words)
   - Time range (start-end in seconds)
   - Brief summary (1-2 sentences)
   - Importance score (1-10)
   - Key keywords (3-5 words)

Format your response as JSON:
{{
  "clusters": [
    {{
      "theme": "Introduction and Overview",
      "start_time": 0.0,
      "end_time": 45.5,
      "summary": "Speaker introduces the main topic and provides overview",
      "importance": 8,
      "keywords": ["introduction", "overview", "topic", "main", "speaker"]
    }}
  ]
}}

Focus on identifying natural content boundaries and meaningful topic shifts."""
        
        return prompt
    
    def _parse_clustering_results(
        self, 
        llm_response: str, 
        segments: List[TranscriptSegment]
    ) -> List[SemanticCluster]:
        """Parse LLM clustering results into SemanticCluster objects."""
        clusters = []
        
        try:
            # Try to extract JSON from the response
            json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
            if not json_match:
                raise ValueError("No JSON found in LLM response")
            
            data = json.loads(json_match.group())
            cluster_data = data.get('clusters', [])
            
            for i, cluster_info in enumerate(cluster_data):
                cluster_id = f"cluster_{i+1}"
                theme = cluster_info.get('theme', f'Theme {i+1}')
                start_time = float(cluster_info.get('start_time', 0))
                end_time = float(cluster_info.get('end_time', start_time + 30))
                summary = cluster_info.get('summary', 'No summary available')
                importance = int(cluster_info.get('importance', 5))
                keywords = cluster_info.get('keywords', [])
                
                # Find segments that belong to this cluster
                cluster_segments = [
                    segment for segment in segments
                    if start_time <= segment.start_time <= end_time
                ]
                
                if cluster_segments:
                    cluster = SemanticCluster(
                        cluster_id=cluster_id,
                        segments=cluster_segments,
                        theme=theme,
                        importance_score=importance / 10.0,  # Normalize to 0-1
                        start_time=start_time,
                        end_time=end_time,
                        summary=summary,
                        keywords=keywords if isinstance(keywords, list) else []
                    )
                    clusters.append(cluster)
            
        except Exception as e:
            self.logger.warning(f"Failed to parse clustering results: {str(e)}")
        
        return clusters
    
    def _fallback_clustering(self, segments: List[TranscriptSegment]) -> List[SemanticCluster]:
        """Fallback clustering method using simple time-based segmentation."""
        if not segments:
            return []
        
        clusters = []
        total_duration = segments[-1].start_time - segments[0].start_time
        cluster_duration = max(total_duration / 5, self.min_cluster_duration)
        
        current_start = segments[0].start_time
        cluster_num = 1
        
        while current_start < segments[-1].start_time:
            cluster_end = current_start + cluster_duration
            
            # Find segments in this time range
            cluster_segments = [
                segment for segment in segments
                if current_start <= segment.start_time < cluster_end
            ]
            
            if cluster_segments:
                cluster = SemanticCluster(
                    cluster_id=f"fallback_cluster_{cluster_num}",
                    segments=cluster_segments,
                    theme=f"Section {cluster_num}",
                    importance_score=0.5,  # Default importance
                    start_time=current_start,
                    end_time=min(cluster_end, segments[-1].start_time),
                    summary=f"Content section {cluster_num}",
                    keywords=[]
                )
                clusters.append(cluster)
            
            current_start = cluster_end
            cluster_num += 1
        
        return clusters
    
    def _enhance_clusters(
        self, 
        clusters: List[SemanticCluster], 
        all_segments: List[TranscriptSegment]
    ) -> List[SemanticCluster]:
        """Enhance clusters with additional semantic information."""
        enhanced_clusters = []
        
        for cluster in clusters:
            # Ensure cluster has segments
            if not cluster.segments:
                continue
            
            # Update timing based on actual segments
            actual_start = min(segment.start_time for segment in cluster.segments)
            actual_end = max(segment.end_time or segment.start_time + 5 for segment in cluster.segments)
            
            # Create enhanced cluster
            enhanced_cluster = SemanticCluster(
                cluster_id=cluster.cluster_id,
                segments=cluster.segments,
                theme=cluster.theme,
                importance_score=cluster.importance_score,
                start_time=actual_start,
                end_time=actual_end,
                summary=cluster.summary,
                keywords=cluster.keywords
            )
            
            enhanced_clusters.append(enhanced_cluster)
        
        # Sort by start time
        enhanced_clusters.sort(key=lambda x: x.start_time)
        
        return enhanced_clusters
    
    def _extract_semantic_keywords(self, clusters: List[SemanticCluster]) -> List[str]:
        """Extract semantic keywords from all clusters."""
        all_keywords = []
        
        for cluster in clusters:
            all_keywords.extend(cluster.keywords)
        
        # Remove duplicates and return top keywords
        unique_keywords = list(dict.fromkeys(all_keywords))  # Preserves order
        return unique_keywords[:self.keyword_extraction_count * 2]  # Return more keywords
    
    def _generate_semantic_timestamps(
        self, 
        clusters: List[SemanticCluster], 
        video_id: str, 
        target_count: int
    ) -> List[SemanticTimestamp]:
        """Generate enhanced timestamps based on semantic clusters."""
        timestamps = []
        
        if not clusters:
            return timestamps
        
        # Sort clusters by importance
        sorted_clusters = sorted(clusters, key=lambda x: x.importance_score, reverse=True)
        
        # Select top clusters for timestamp generation
        selected_clusters = sorted_clusters[:min(target_count, len(clusters))]
        
        for i, cluster in enumerate(selected_clusters):
            # Choose representative timestamp within cluster
            # Use the beginning of the most important segment or cluster start
            if cluster.segments:
                # Find the segment with most words (likely most informative)
                best_segment = max(cluster.segments, key=lambda x: x.word_count)
                timestamp_seconds = best_segment.start_time
            else:
                timestamp_seconds = cluster.start_time
            
            # Generate YouTube URL
            youtube_url = f"https://www.youtube.com/watch?v={video_id}&t={int(timestamp_seconds)}s"
            
            # Calculate confidence based on cluster properties
            confidence = self._calculate_timestamp_confidence(cluster)
            
            # Create context information
            context_before, context_after = self._get_timestamp_context(cluster, timestamp_seconds)
            
            # Convert importance score to 1-10 scale
            importance_rating = max(1, min(10, int(cluster.importance_score * 10)))
            
            timestamp = SemanticTimestamp(
                timestamp_seconds=timestamp_seconds,
                timestamp_formatted=self._format_timestamp(timestamp_seconds),
                description=cluster.summary or f"Key moment: {cluster.theme}",
                importance_rating=importance_rating,
                youtube_url=youtube_url,
                video_id=video_id,
                semantic_cluster_id=cluster.cluster_id,
                cluster_theme=cluster.theme,
                context_before=context_before,
                context_after=context_after,
                semantic_keywords=cluster.keywords,
                confidence_score=confidence
            )
            
            timestamps.append(timestamp)
        
        # Sort by timestamp
        timestamps.sort(key=lambda x: x.timestamp_seconds)
        
        return timestamps
    
    def _generate_vector_optimized_timestamps(
        self, 
        clusters: List[SemanticCluster], 
        video_id: str, 
        target_count: int
    ) -> List[SemanticTimestamp]:
        """Generate timestamps using vector search optimization."""
        try:
            # Use vector search engine for optimal timestamp selection
            optimized_timestamps = self.vector_search_engine.find_optimal_timestamps(
                semantic_clusters=clusters,
                target_count=target_count,
                diversity_weight=self.vector_diversity_weight
            )
            
            if optimized_timestamps:
                self.logger.info(f"Vector optimization generated {len(optimized_timestamps)} timestamps")
                return optimized_timestamps
            else:
                self.logger.warning("Vector optimization failed, falling back to semantic method")
                return self._generate_semantic_timestamps(clusters, video_id, target_count)
                
        except Exception as e:
            self.logger.warning(f"Vector optimization failed: {str(e)}, using fallback")
            return self._generate_semantic_timestamps(clusters, video_id, target_count)
    
    def _calculate_timestamp_confidence(self, cluster: SemanticCluster) -> float:
        """Calculate confidence score for a timestamp based on cluster properties."""
        confidence = 0.5  # Base confidence
        
        # Factor in importance score
        confidence += cluster.importance_score * 0.3
        
        # Factor in cluster size (more segments = higher confidence)
        if cluster.segments:
            segment_factor = min(len(cluster.segments) / 5, 0.2)  # Max 0.2 bonus
            confidence += segment_factor
        
        # Factor in cluster duration (appropriate length clusters get bonus)
        duration_factor = 0
        if 30 <= cluster.duration <= 120:  # Sweet spot duration
            duration_factor = 0.1
        elif cluster.duration > 300:  # Very long clusters get penalty
            duration_factor = -0.1
        confidence += duration_factor
        
        # Factor in keyword availability
        if cluster.keywords:
            confidence += min(len(cluster.keywords) / 10, 0.1)
        
        return max(0.1, min(1.0, confidence))
    
    def _get_timestamp_context(self, cluster: SemanticCluster, timestamp: float) -> Tuple[str, str]:
        """Get context before and after a timestamp."""
        context_before = ""
        context_after = ""
        
        if cluster.segments:
            # Find the segment closest to the timestamp
            closest_segment = min(
                cluster.segments, 
                key=lambda x: abs(x.start_time - timestamp)
            )
            
            # Get text from segments around the timestamp
            segments_sorted = sorted(cluster.segments, key=lambda x: x.start_time)
            closest_index = segments_sorted.index(closest_segment)
            
            # Context before
            if closest_index > 0:
                context_before = segments_sorted[closest_index - 1].text[:100] + "..."
            
            # Context after
            if closest_index < len(segments_sorted) - 1:
                context_after = "..." + segments_sorted[closest_index + 1].text[:100]
        
        return context_before, context_after
    
    def _calculate_semantic_metrics(
        self, 
        segments: List[TranscriptSegment], 
        clusters: List[SemanticCluster], 
        timestamps: List[SemanticTimestamp]
    ) -> Dict[str, Any]:
        """Calculate metrics about the semantic analysis."""
        if not segments:
            return {}
        
        total_duration = segments[-1].start_time - segments[0].start_time if segments else 0
        clustered_duration = sum(cluster.duration for cluster in clusters)
        
        metrics = {
            'total_segments': len(segments),
            'total_clusters': len(clusters),
            'total_timestamps': len(timestamps),
            'coverage_percentage': (clustered_duration / total_duration * 100) if total_duration > 0 else 0,
            'avg_cluster_importance': np.mean([c.importance_score for c in clusters]) if clusters and HAS_NUMPY else 0.5,
            'avg_timestamp_confidence': np.mean([t.confidence_score for t in timestamps]) if timestamps and HAS_NUMPY else 0.5,
            'semantic_density': len(clusters) / (total_duration / 60) if total_duration > 0 else 0,  # clusters per minute
            'avg_cluster_duration': np.mean([c.duration for c in clusters]) if clusters and HAS_NUMPY else 0,
            'total_keywords': sum(len(c.keywords) for c in clusters),
            'processing_time': datetime.utcnow().isoformat()
        }
        
        return metrics
    
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
    
    def _cluster_to_dict(self, cluster: SemanticCluster) -> Dict[str, Any]:
        """Convert SemanticCluster to dictionary representation."""
        return {
            'cluster_id': cluster.cluster_id,
            'theme': cluster.theme,
            'importance_score': cluster.importance_score,
            'start_time': cluster.start_time,
            'end_time': cluster.end_time,
            'duration': cluster.duration,
            'summary': cluster.summary,
            'keywords': cluster.keywords,
            'segment_count': len(cluster.segments),
            'word_count': cluster.word_count
        }
    
    def _create_empty_result(self, reason: str) -> Dict[str, Any]:
        """Create an empty result with error information."""
        return {
            'status': 'empty',
            'reason': reason,
            'video_id': '',
            'analysis_timestamp': datetime.utcnow().isoformat(),
            'segments_count': 0,
            'clusters_count': 0,
            'timestamps_count': 0,
            'timestamps': [],
            'semantic_clusters': [],
            'semantic_keywords': [],
            'semantic_metrics': {},
            'processing_metadata': {}
        }
    
    # Performance optimization methods
    def _generate_cache_key(self, raw_transcript: List[Dict[str, Any]], video_id: str, target_count: int) -> str:
        """Generate a cache key for the analysis."""
        import hashlib
        # Create a hash of the transcript content for caching
        transcript_text = str(sorted([entry.get('text', '') for entry in raw_transcript if entry.get('text')]))
        content_hash = hashlib.md5(transcript_text.encode()).hexdigest()[:16]
        return f"{video_id}_{target_count}_{content_hash}"
    
    def _parse_transcript_segments_optimized(self, raw_transcript: List[Dict[str, Any]]) -> List[TranscriptSegment]:
        """Optimized version of transcript segment parsing."""
        if len(raw_transcript) > self.max_segments_for_full_analysis:
            # Sample segments for large transcripts
            import random
            sample_size = int(len(raw_transcript) * self.segment_sampling_ratio)
            sampled_transcript = random.sample(raw_transcript, sample_size)
            self.logger.info(f"Sampling {sample_size} segments from {len(raw_transcript)} for performance")
            return self._parse_transcript_segments(sampled_transcript)
        else:
            return self._parse_transcript_segments(raw_transcript)
    
    def _should_terminate_early(self, segments: List[TranscriptSegment]) -> bool:
        """Determine if we should terminate early due to low quality input."""
        if not segments:
            return True
        
        # Check for minimum viable content
        if len(segments) < 5:
            return True
        
        # Check average segment quality
        total_words = sum(len(segment.text.split()) for segment in segments)
        avg_words_per_segment = total_words / len(segments)
        
        if avg_words_per_segment < 2:  # Very short segments
            return True
        
        # Check for content diversity
        unique_texts = set(segment.text.lower() for segment in segments)
        diversity_ratio = len(unique_texts) / len(segments)
        
        if diversity_ratio < self.early_termination_threshold:
            return True
        
        return False
    
    def _create_early_termination_result(self, segments: List[TranscriptSegment], video_id: str) -> Dict[str, Any]:
        """Create a minimal result for early termination."""
        return {
            'status': 'early_termination',
            'reason': 'low_quality_input',
            'video_id': video_id,
            'analysis_timestamp': datetime.utcnow().isoformat(),
            'segments_count': len(segments),
            'clusters_count': 0,
            'timestamps_count': 0,
            'timestamps': [],
            'semantic_clusters': [],
            'semantic_keywords': [],
            'semantic_metrics': {'early_termination': True},
            'processing_metadata': {
                'optimization_enabled': True,
                'early_termination': True,
                'reason': 'insufficient_quality'
            }
        }
    
    def _perform_semantic_clustering_optimized(
        self, 
        segments: List[TranscriptSegment], 
        video_title: str = ""
    ) -> List[SemanticCluster]:
        """Optimized semantic clustering with caching and smart algorithm selection."""
        cache_key = f"clustering_{len(segments)}_{hash(video_title)}"
        
        if self.enable_caching and cache_key in self.cluster_cache:
            self.logger.debug("Using cached clustering result")
            return self.cluster_cache[cache_key]
        
        # Smart algorithm selection based on content size
        if len(segments) < 20:
            algorithm = 'structure'
        elif len(segments) < 100:
            algorithm = 'tfidf'
        else:
            algorithm = 'embedding' if HAS_SKLEARN else 'tfidf'
        
        try:
            target_clusters = min(self.max_clusters, max(3, len(segments) // 5))
            
            clustering_result = self.semantic_analyzer.analyze_segments(
                segments=segments,
                algorithm=algorithm,
                target_clusters=target_clusters
            )
            
            clusters = clustering_result.clusters if clustering_result.clusters else []
            
            # Cache the result
            if self.enable_caching and clusters:
                self.cluster_cache[cache_key] = clusters
                if len(self.cluster_cache) > 50:
                    oldest_key = next(iter(self.cluster_cache))
                    del self.cluster_cache[oldest_key]
            
            return clusters
            
        except Exception as e:
            self.logger.warning(f"Optimized clustering failed: {str(e)}, using fallback")
            return self._fallback_clustering(segments)
    
    def clear_caches(self) -> None:
        """Clear all caches to free memory."""
        self.analysis_cache.clear()
        self.embedding_cache.clear() 
        self.cluster_cache.clear()
        if self.vector_search_engine:
            self.vector_search_engine.clear_cache()
        self.logger.info("All caches cleared")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return {
            'cache_sizes': {
                'analysis_cache': len(self.analysis_cache),
                'embedding_cache': len(self.embedding_cache),
                'cluster_cache': len(self.cluster_cache)
            },
            'settings': {
                'enable_caching': self.enable_caching,
                'enable_parallel_processing': self.enable_parallel_processing,
                'max_segments_for_full_analysis': self.max_segments_for_full_analysis,
                'segment_sampling_ratio': self.segment_sampling_ratio,
                'early_termination_threshold': self.early_termination_threshold
            }
        }

    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create an error result."""
        return {
            'status': 'error',
            'error': error_message,
            'video_id': '',
            'analysis_timestamp': datetime.utcnow().isoformat(),
            'segments_count': 0,
            'clusters_count': 0,
            'timestamps_count': 0,
            'timestamps': [],
            'semantic_clusters': [],
            'semantic_keywords': [],
            'semantic_metrics': {},
            'processing_metadata': {}
        }


# Factory function for easy instantiation
def create_semantic_analysis_service(
    smart_llm_client: Optional[SmartLLMClient] = None,
    semantic_analyzer: Optional[SemanticAnalyzer] = None,
    vector_search_engine: Optional[VectorSearchEngine] = None
) -> SemanticAnalysisService:
    """
    Factory function to create a semantic analysis service.
    
    Args:
        smart_llm_client: Optional pre-initialized LLM client
        semantic_analyzer: Optional pre-initialized semantic analyzer
        vector_search_engine: Optional pre-initialized vector search engine
        
    Returns:
        SemanticAnalysisService instance
    """
    return SemanticAnalysisService(smart_llm_client, semantic_analyzer, vector_search_engine)


