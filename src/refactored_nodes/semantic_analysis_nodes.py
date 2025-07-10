"""
Semantic Analysis PocketFlow Nodes.

This module contains specialized nodes for semantic analysis operations within
the PocketFlow workflow system, including clustering, vector search, and
enhanced timestamp generation with semantic understanding.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from .validation_nodes import BaseProcessingNode, Store
from ..services.semantic_analysis_service import create_semantic_analysis_service
from ..utils.semantic_analyzer import create_semantic_analyzer
from ..utils.vector_search import create_vector_search_engine

logger = logging.getLogger(__name__)


class SemanticAnalysisNode(BaseProcessingNode):
    """
    Core semantic analysis node for transcript processing.
    
    Performs comprehensive semantic analysis including clustering,
    keyword extraction, and semantic understanding of video content.
    """
    
    def __init__(self, max_retries: int = 3, retry_delay: float = 2.0):
        super().__init__("SemanticAnalysisNode", max_retries, retry_delay)
        self.semantic_service = None
        
    def prep(self, store: Store) -> Dict[str, Any]:
        """
        Prepare for semantic analysis.
        
        Args:
            store: Data store containing transcript and video data
            
        Returns:
            Dict containing prep results and configuration
        """
        self.logger.info("Starting semantic analysis preparation")
        
        try:
            # Validate required input
            required_keys = ['transcript_data']
            is_valid, missing_keys = self._validate_store_data(store, required_keys)
            if not is_valid:
                raise ValueError(f"Missing required data for semantic analysis: {missing_keys}")
            
            transcript_data = store['transcript_data']
            raw_transcript = transcript_data.get('raw_transcript', [])
            
            if not raw_transcript:
                raise ValueError("No raw transcript data available for semantic analysis")
            
            # Get video metadata
            video_metadata = store.get('video_metadata', {})
            video_id = store.get('video_id', transcript_data.get('video_id', ''))
            video_title = video_metadata.get('title', 'Unknown Video')
            
            if not video_id:
                raise ValueError("Video ID is required for semantic analysis")
            
            # Initialize semantic analysis service
            try:
                self.semantic_service = create_semantic_analysis_service()
            except Exception as e:
                raise ValueError(f"Failed to initialize semantic analysis service: {str(e)}")
            
            # Validate transcript format
            valid_segments = []
            for segment in raw_transcript:
                if isinstance(segment, dict) and 'text' in segment and 'start' in segment:
                    valid_segments.append(segment)
            
            if not valid_segments:
                raise ValueError("No valid transcript segments found for semantic analysis")
            
            prep_result = {
                'status': 'success',
                'video_id': video_id,
                'video_title': video_title,
                'raw_transcript': valid_segments,
                'segment_count': len(valid_segments),
                'total_duration': self._calculate_total_duration(valid_segments),
                'prep_timestamp': datetime.utcnow().isoformat()
            }
            
            self.logger.info(f"Semantic analysis prep successful: {len(valid_segments)} segments")
            return prep_result
            
        except Exception as e:
            error_info = self._handle_error(e, "Semantic analysis preparation failed")
            return {
                'status': 'failed',
                'error': error_info.__dict__,
                'prep_timestamp': datetime.utcnow().isoformat()
            }
    
    def exec(self, store: Store, prep_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute semantic analysis.
        
        Args:
            store: Data store
            prep_result: Results from prep phase
            
        Returns:
            Dict containing execution results
        """
        if prep_result.get('status') != 'success':
            return {
                'exec_status': 'failed',
                'error': 'Prep phase failed',
                'exec_timestamp': datetime.utcnow().isoformat()
            }
        
        video_id = prep_result['video_id']
        video_title = prep_result['video_title']
        raw_transcript = prep_result['raw_transcript']
        last_error = None
        
        for retry_count in range(self.max_retries + 1):
            try:
                self._retry_with_delay(retry_count)
                
                # Perform semantic analysis
                analysis_result = self.semantic_service.analyze_transcript(
                    raw_transcript=raw_transcript,
                    video_id=video_id,
                    video_title=video_title,
                    target_timestamp_count=5  # Default timestamp count
                )
                
                if analysis_result['status'] != 'success':
                    raise ValueError(f"Semantic analysis failed: {analysis_result.get('reason', 'Unknown error')}")
                
                # Extract results
                semantic_clusters = analysis_result.get('semantic_clusters', [])
                semantic_keywords = analysis_result.get('semantic_keywords', [])
                semantic_metrics = analysis_result.get('semantic_metrics', {})
                timestamps = analysis_result.get('timestamps', [])
                
                exec_result = {
                    'exec_status': 'success',
                    'video_id': video_id,
                    'video_title': video_title,
                    'semantic_analysis': analysis_result,
                    'clusters': semantic_clusters,
                    'semantic_keywords': semantic_keywords,
                    'semantic_metrics': semantic_metrics,
                    'semantic_timestamps': timestamps,
                    'cluster_count': len(semantic_clusters),
                    'keyword_count': len(semantic_keywords),
                    'timestamp_count': len(timestamps),
                    'exec_timestamp': datetime.utcnow().isoformat(),
                    'retry_count': retry_count
                }
                
                self.logger.info(f"Semantic analysis successful: {len(semantic_clusters)} clusters, {len(timestamps)} timestamps")
                return exec_result
                
            except Exception as e:
                last_error = self._handle_error(e, f"Semantic analysis execution failed", retry_count)
                if retry_count >= self.max_retries:
                    break
        
        return {
            'exec_status': 'failed',
            'error': last_error.__dict__ if last_error else 'Unknown error',
            'exec_timestamp': datetime.utcnow().isoformat(),
            'retry_count': self.max_retries
        }
    
    def post(self, store: Store, prep_result: Dict[str, Any], exec_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Post-process semantic analysis results.
        
        Args:
            store: Data store
            prep_result: Results from prep phase
            exec_result: Results from exec phase
            
        Returns:
            Dict containing final processing results
        """
        try:
            if exec_result.get('exec_status') != 'success':
                return {
                    'post_status': 'failed',
                    'error': 'Execution phase failed',
                    'post_timestamp': datetime.utcnow().isoformat()
                }
            
            # Update store with semantic analysis results
            store['semantic_analysis_result'] = exec_result['semantic_analysis']
            store['semantic_clusters'] = exec_result['clusters']
            store['semantic_keywords'] = exec_result['semantic_keywords']
            store['semantic_metrics'] = exec_result['semantic_metrics']
            store['semantic_timestamps'] = exec_result['semantic_timestamps']
            
            # Add to processing metadata
            processing_metadata = store.get('processing_metadata', {})
            processing_metadata['semantic_analysis'] = {
                'completed': True,
                'cluster_count': exec_result['cluster_count'],
                'keyword_count': exec_result['keyword_count'],
                'timestamp_count': exec_result['timestamp_count'],
                'processing_time': exec_result.get('processing_time', 0),
                'timestamp': exec_result['exec_timestamp']
            }
            store['processing_metadata'] = processing_metadata
            
            post_result = {
                'post_status': 'success',
                'semantic_analysis_ready': True,
                'cluster_count': exec_result['cluster_count'],
                'keyword_count': exec_result['keyword_count'],
                'timestamp_count': exec_result['timestamp_count'],
                'video_id': exec_result['video_id'],
                'post_timestamp': datetime.utcnow().isoformat()
            }
            
            self.logger.info(f"Semantic analysis post-processing successful for video {exec_result['video_id']}")
            return post_result
            
        except Exception as e:
            error_info = self._handle_error(e, "Semantic analysis post-processing failed")
            return {
                'post_status': 'failed',
                'error': error_info.__dict__,
                'post_timestamp': datetime.utcnow().isoformat()
            }
    
    def _calculate_total_duration(self, transcript_segments: List[Dict[str, Any]]) -> float:
        """Calculate total duration of transcript segments."""
        if not transcript_segments:
            return 0.0
        
        total_duration = 0.0
        for segment in transcript_segments:
            duration = segment.get('duration', 0)
            if duration:
                total_duration += float(duration)
        
        # If no duration info, estimate from start times
        if total_duration == 0 and transcript_segments:
            last_segment = transcript_segments[-1]
            total_duration = last_segment.get('start', 0) + 5  # Add 5 seconds buffer
        
        return total_duration


class VectorSearchNode(BaseProcessingNode):
    """
    Vector search node for embedding-based semantic operations.
    
    Provides vector search capabilities for finding semantically similar
    content and performing advanced timestamp optimization.
    """
    
    def __init__(self, max_retries: int = 3, retry_delay: float = 2.0):
        super().__init__("VectorSearchNode", max_retries, retry_delay)
        self.vector_engine = None
        
    def prep(self, store: Store) -> Dict[str, Any]:
        """
        Prepare for vector search operations.
        
        Args:
            store: Data store containing semantic analysis results
            
        Returns:
            Dict containing prep results
        """
        self.logger.info("Starting vector search preparation")
        
        try:
            # Check for semantic analysis results
            if 'semantic_analysis_result' not in store:
                raise ValueError("Vector search requires semantic analysis results")
            
            semantic_result = store['semantic_analysis_result']
            if semantic_result.get('status') != 'success':
                raise ValueError("Semantic analysis must be successful for vector search")
            
            # Initialize vector search engine
            try:
                self.vector_engine = create_vector_search_engine()
            except Exception as e:
                raise ValueError(f"Failed to initialize vector search engine: {str(e)}")
            
            # Get required data
            video_id = store.get('video_id', '')
            transcript_data = store.get('transcript_data', {})
            raw_transcript = transcript_data.get('raw_transcript', [])
            
            if not video_id or not raw_transcript:
                raise ValueError("Video ID and transcript data required for vector search")
            
            prep_result = {
                'status': 'success',
                'video_id': video_id,
                'segment_count': len(raw_transcript),
                'vector_engine_ready': True,
                'prep_timestamp': datetime.utcnow().isoformat()
            }
            
            self.logger.info(f"Vector search prep successful for video {video_id}")
            return prep_result
            
        except Exception as e:
            error_info = self._handle_error(e, "Vector search preparation failed")
            return {
                'status': 'failed',
                'error': error_info.__dict__,
                'prep_timestamp': datetime.utcnow().isoformat()
            }
    
    def exec(self, store: Store, prep_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute vector search operations.
        
        Args:
            store: Data store
            prep_result: Results from prep phase
            
        Returns:
            Dict containing execution results
        """
        if prep_result.get('status') != 'success':
            return {
                'exec_status': 'failed',
                'error': 'Prep phase failed',
                'exec_timestamp': datetime.utcnow().isoformat()
            }
        
        video_id = prep_result['video_id']
        last_error = None
        
        for retry_count in range(self.max_retries + 1):
            try:
                self._retry_with_delay(retry_count)
                
                # Get transcript segments for indexing
                transcript_data = store['transcript_data']
                raw_transcript = transcript_data['raw_transcript']
                
                # Convert to TranscriptSegment objects (simplified)
                from ..services.semantic_analysis_service import TranscriptSegment
                segments = []
                for seg_data in raw_transcript:
                    segment = TranscriptSegment(
                        start_time=float(seg_data.get('start', 0)),
                        end_time=seg_data.get('end'),
                        text=seg_data.get('text', ''),
                        duration=seg_data.get('duration')
                    )
                    segments.append(segment)
                
                # Create vector index
                index_created = self.vector_engine.create_index(segments, f"video_{video_id}")
                
                if not index_created:
                    raise ValueError("Failed to create vector index")
                
                # Perform coherence analysis
                coherence_analysis = self.vector_engine.analyze_semantic_coherence(segments)
                
                # Get optimized timestamps if semantic clusters are available
                optimized_timestamps = []
                semantic_clusters = store.get('semantic_clusters', [])
                if semantic_clusters:
                    try:
                        # Convert cluster data to objects (this would need proper implementation)
                        optimized_timestamps = self.vector_engine.find_optimal_timestamps(
                            semantic_clusters=semantic_clusters,
                            target_count=5,
                            diversity_weight=0.3
                        )
                    except Exception as e:
                        self.logger.warning(f"Timestamp optimization failed: {str(e)}")
                
                exec_result = {
                    'exec_status': 'success',
                    'video_id': video_id,
                    'vector_index_created': True,
                    'coherence_analysis': coherence_analysis,
                    'optimized_timestamps': optimized_timestamps,
                    'index_info': self.vector_engine.get_index_info(f"video_{video_id}"),
                    'exec_timestamp': datetime.utcnow().isoformat(),
                    'retry_count': retry_count
                }
                
                self.logger.info(f"Vector search execution successful for video {video_id}")
                return exec_result
                
            except Exception as e:
                last_error = self._handle_error(e, f"Vector search execution failed", retry_count)
                if retry_count >= self.max_retries:
                    break
        
        return {
            'exec_status': 'failed',
            'error': last_error.__dict__ if last_error else 'Unknown error',
            'exec_timestamp': datetime.utcnow().isoformat(),
            'retry_count': self.max_retries
        }
    
    def post(self, store: Store, prep_result: Dict[str, Any], exec_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Post-process vector search results.
        
        Args:
            store: Data store
            prep_result: Results from prep phase
            exec_result: Results from exec phase
            
        Returns:
            Dict containing final processing results
        """
        try:
            if exec_result.get('exec_status') != 'success':
                return {
                    'post_status': 'failed',
                    'error': 'Execution phase failed',
                    'post_timestamp': datetime.utcnow().isoformat()
                }
            
            # Update store with vector search results
            store['vector_search_result'] = exec_result
            store['coherence_analysis'] = exec_result['coherence_analysis']
            store['vector_index_info'] = exec_result['index_info']
            
            # Update optimized timestamps if available
            if exec_result.get('optimized_timestamps'):
                store['optimized_timestamps'] = exec_result['optimized_timestamps']
            
            # Update processing metadata
            processing_metadata = store.get('processing_metadata', {})
            processing_metadata['vector_search'] = {
                'completed': True,
                'index_created': exec_result['vector_index_created'],
                'coherence_score': exec_result['coherence_analysis'].get('coherence_score', 0),
                'optimized_timestamp_count': len(exec_result.get('optimized_timestamps', [])),
                'timestamp': exec_result['exec_timestamp']
            }
            store['processing_metadata'] = processing_metadata
            
            post_result = {
                'post_status': 'success',
                'vector_search_ready': True,
                'index_created': exec_result['vector_index_created'],
                'coherence_score': exec_result['coherence_analysis'].get('coherence_score', 0),
                'video_id': exec_result['video_id'],
                'post_timestamp': datetime.utcnow().isoformat()
            }
            
            self.logger.info(f"Vector search post-processing successful for video {exec_result['video_id']}")
            return post_result
            
        except Exception as e:
            error_info = self._handle_error(e, "Vector search post-processing failed")
            return {
                'post_status': 'failed',
                'error': error_info.__dict__,
                'post_timestamp': datetime.utcnow().isoformat()
            }


class EnhancedTimestampNode(BaseProcessingNode):
    """
    Enhanced timestamp generation node with semantic analysis integration.
    
    Combines multiple timestamp generation methods including semantic analysis,
    vector optimization, and traditional LLM-based approaches for maximum accuracy.
    """
    
    def __init__(self, max_retries: int = 3, retry_delay: float = 2.0):
        super().__init__("EnhancedTimestampNode", max_retries, retry_delay)
        
    def prep(self, store: Store) -> Dict[str, Any]:
        """
        Prepare for enhanced timestamp generation.
        
        Args:
            store: Data store containing semantic analysis and vector search results
            
        Returns:
            Dict containing prep results
        """
        self.logger.info("Starting enhanced timestamp preparation")
        
        try:
            # Validate required input
            required_keys = ['semantic_analysis_result']
            is_valid, missing_keys = self._validate_store_data(store, required_keys)
            if not is_valid:
                self.logger.warning(f"Missing semantic analysis data: {missing_keys}, will use fallback methods")
            
            video_id = store.get('video_id', '')
            if not video_id:
                raise ValueError("Video ID is required for timestamp generation")
            
            # Get available data sources
            semantic_available = 'semantic_analysis_result' in store
            vector_available = 'vector_search_result' in store
            transcript_available = 'transcript_data' in store
            
            if not transcript_available:
                raise ValueError("Transcript data is required for timestamp generation")
            
            prep_result = {
                'status': 'success',
                'video_id': video_id,
                'semantic_available': semantic_available,
                'vector_available': vector_available,
                'transcript_available': transcript_available,
                'timestamp_methods': self._determine_timestamp_methods(store),
                'prep_timestamp': datetime.utcnow().isoformat()
            }
            
            self.logger.info(f"Enhanced timestamp prep successful, methods: {prep_result['timestamp_methods']}")
            return prep_result
            
        except Exception as e:
            error_info = self._handle_error(e, "Enhanced timestamp preparation failed")
            return {
                'status': 'failed',
                'error': error_info.__dict__,
                'prep_timestamp': datetime.utcnow().isoformat()
            }
    
    def exec(self, store: Store, prep_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute enhanced timestamp generation.
        
        Args:
            store: Data store
            prep_result: Results from prep phase
            
        Returns:
            Dict containing execution results
        """
        if prep_result.get('status') != 'success':
            return {
                'exec_status': 'failed',
                'error': 'Prep phase failed',
                'exec_timestamp': datetime.utcnow().isoformat()
            }
        
        video_id = prep_result['video_id']
        timestamp_methods = prep_result['timestamp_methods']
        last_error = None
        
        for retry_count in range(self.max_retries + 1):
            try:
                self._retry_with_delay(retry_count)
                
                # Collect timestamps from all available methods
                all_timestamps = []
                method_results = {}
                
                # Method 1: Semantic analysis timestamps (highest priority)
                if 'semantic' in timestamp_methods:
                    semantic_timestamps = store.get('semantic_timestamps', [])
                    if semantic_timestamps:
                        all_timestamps.extend(semantic_timestamps)
                        method_results['semantic'] = {
                            'count': len(semantic_timestamps),
                            'quality': 'high'
                        }
                        self.logger.info(f"Added {len(semantic_timestamps)} semantic timestamps")
                
                # Method 2: Vector-optimized timestamps
                if 'vector' in timestamp_methods:
                    optimized_timestamps = store.get('optimized_timestamps', [])
                    if optimized_timestamps:
                        all_timestamps.extend(optimized_timestamps)
                        method_results['vector'] = {
                            'count': len(optimized_timestamps),
                            'quality': 'high'
                        }
                        self.logger.info(f"Added {len(optimized_timestamps)} vector-optimized timestamps")
                
                # Method 3: Traditional LLM-based timestamps (fallback)
                if 'traditional' in timestamp_methods or not all_timestamps:
                    try:
                        # Use the existing TimestampNode logic as fallback
                        from .summary_nodes import TimestampNode
                        timestamp_node = TimestampNode()
                        
                        # Initialize the node with prep
                        timestamp_prep = timestamp_node.prep(store)
                        if timestamp_prep.get('status') == 'success':
                            timestamp_exec = timestamp_node.exec(store, timestamp_prep)
                            if timestamp_exec.get('exec_status') == 'success':
                                traditional_timestamps = timestamp_exec.get('timestamps', [])
                                all_timestamps.extend(traditional_timestamps)
                                method_results['traditional'] = {
                                    'count': len(traditional_timestamps),
                                    'quality': 'medium'
                                }
                                self.logger.info(f"Added {len(traditional_timestamps)} traditional timestamps")
                    except Exception as e:
                        self.logger.warning(f"Traditional timestamp generation failed: {str(e)}")
                
                # Deduplicate and sort timestamps
                final_timestamps = self._deduplicate_timestamps(all_timestamps)
                
                if not final_timestamps:
                    raise ValueError("No timestamps generated by any method")
                
                # Calculate quality metrics
                quality_metrics = self._calculate_timestamp_quality(final_timestamps, method_results)
                
                exec_result = {
                    'exec_status': 'success',
                    'video_id': video_id,
                    'enhanced_timestamps': final_timestamps,
                    'timestamp_count': len(final_timestamps),
                    'method_results': method_results,
                    'quality_metrics': quality_metrics,
                    'methods_used': list(method_results.keys()),
                    'exec_timestamp': datetime.utcnow().isoformat(),
                    'retry_count': retry_count
                }
                
                self.logger.info(f"Enhanced timestamp generation successful: {len(final_timestamps)} timestamps using {list(method_results.keys())}")
                return exec_result
                
            except Exception as e:
                last_error = self._handle_error(e, f"Enhanced timestamp execution failed", retry_count)
                if retry_count >= self.max_retries:
                    break
        
        return {
            'exec_status': 'failed',
            'error': last_error.__dict__ if last_error else 'Unknown error',
            'exec_timestamp': datetime.utcnow().isoformat(),
            'retry_count': self.max_retries
        }
    
    def post(self, store: Store, prep_result: Dict[str, Any], exec_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Post-process enhanced timestamp results.
        
        Args:
            store: Data store
            prep_result: Results from prep phase
            exec_result: Results from exec phase
            
        Returns:
            Dict containing final processing results
        """
        try:
            if exec_result.get('exec_status') != 'success':
                return {
                    'post_status': 'failed',
                    'error': 'Execution phase failed',
                    'post_timestamp': datetime.utcnow().isoformat()
                }
            
            # Update store with enhanced timestamps (replaces any existing timestamps)
            store['timestamps'] = exec_result['enhanced_timestamps']
            store['enhanced_timestamp_metadata'] = {
                'methods_used': exec_result['methods_used'],
                'method_results': exec_result['method_results'],
                'quality_metrics': exec_result['quality_metrics'],
                'generation_timestamp': exec_result['exec_timestamp']
            }
            
            # Update processing metadata
            processing_metadata = store.get('processing_metadata', {})
            processing_metadata['enhanced_timestamps'] = {
                'completed': True,
                'timestamp_count': exec_result['timestamp_count'],
                'methods_used': exec_result['methods_used'],
                'quality_score': exec_result['quality_metrics'].get('overall_quality', 0),
                'timestamp': exec_result['exec_timestamp']
            }
            store['processing_metadata'] = processing_metadata
            
            post_result = {
                'post_status': 'success',
                'enhanced_timestamps_ready': True,
                'timestamp_count': exec_result['timestamp_count'],
                'methods_used': exec_result['methods_used'],
                'quality_score': exec_result['quality_metrics'].get('overall_quality', 0),
                'video_id': exec_result['video_id'],
                'post_timestamp': datetime.utcnow().isoformat()
            }
            
            self.logger.info(f"Enhanced timestamp post-processing successful for video {exec_result['video_id']}")
            return post_result
            
        except Exception as e:
            error_info = self._handle_error(e, "Enhanced timestamp post-processing failed")
            return {
                'post_status': 'failed',
                'error': error_info.__dict__,
                'post_timestamp': datetime.utcnow().isoformat()
            }
    
    def _determine_timestamp_methods(self, store: Store) -> List[str]:
        """Determine which timestamp generation methods are available."""
        methods = []
        
        if 'semantic_analysis_result' in store:
            methods.append('semantic')
        
        if 'vector_search_result' in store:
            methods.append('vector')
        
        # Traditional method is always available as fallback
        methods.append('traditional')
        
        return methods
    
    def _deduplicate_timestamps(self, timestamps: List[Any]) -> List[Dict[str, Any]]:
        """Deduplicate timestamps and ensure consistent format."""
        if not timestamps:
            return []
        
        # Convert all timestamps to consistent format
        normalized_timestamps = []
        seen_times = set()
        
        for ts in timestamps:
            # Handle different timestamp formats
            if hasattr(ts, 'timestamp_seconds'):
                # SemanticTimestamp object
                timestamp_seconds = ts.timestamp_seconds
                timestamp_dict = {
                    'timestamp_seconds': timestamp_seconds,
                    'timestamp_formatted': ts.timestamp_formatted,
                    'description': ts.description,
                    'importance_rating': ts.importance_rating,
                    'youtube_url': ts.youtube_url
                }
            elif isinstance(ts, dict):
                # Dictionary format
                timestamp_seconds = ts.get('timestamp_seconds', 0)
                timestamp_dict = ts.copy()
            else:
                continue
            
            # Skip duplicates (within 5 seconds)
            duplicate = False
            for seen_time in seen_times:
                if abs(timestamp_seconds - seen_time) < 5:
                    duplicate = True
                    break
            
            if not duplicate:
                seen_times.add(timestamp_seconds)
                normalized_timestamps.append(timestamp_dict)
        
        # Sort by timestamp
        normalized_timestamps.sort(key=lambda x: x.get('timestamp_seconds', 0))
        
        return normalized_timestamps
    
    def _calculate_timestamp_quality(self, timestamps: List[Dict[str, Any]], method_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate quality metrics for generated timestamps."""
        if not timestamps:
            return {'overall_quality': 0.0}
        
        # Quality factors
        method_quality_weights = {
            'semantic': 1.0,    # Highest quality
            'vector': 0.9,      # High quality
            'traditional': 0.7  # Good quality
        }
        
        # Calculate weighted quality score
        total_weight = 0
        total_timestamps = 0
        
        for method, results in method_results.items():
            weight = method_quality_weights.get(method, 0.5)
            count = results.get('count', 0)
            total_weight += weight * count
            total_timestamps += count
        
        overall_quality = total_weight / total_timestamps if total_timestamps > 0 else 0.0
        
        # Additional quality factors
        time_distribution_quality = self._assess_time_distribution(timestamps)
        description_quality = self._assess_description_quality(timestamps)
        
        # Combined quality score
        combined_quality = (
            overall_quality * 0.5 +
            time_distribution_quality * 0.3 +
            description_quality * 0.2
        )
        
        return {
            'overall_quality': min(1.0, combined_quality),
            'method_quality': overall_quality,
            'time_distribution_quality': time_distribution_quality,
            'description_quality': description_quality,
            'timestamp_count': len(timestamps),
            'methods_used': list(method_results.keys())
        }
    
    def _assess_time_distribution(self, timestamps: List[Dict[str, Any]]) -> float:
        """Assess quality of timestamp time distribution."""
        if len(timestamps) < 2:
            return 0.5
        
        # Check for good distribution across video duration
        times = [ts.get('timestamp_seconds', 0) for ts in timestamps]
        times.sort()
        
        # Calculate intervals
        intervals = [times[i+1] - times[i] for i in range(len(times)-1)]
        
        if not intervals:
            return 0.5
        
        # Good distribution has reasonably even intervals
        avg_interval = sum(intervals) / len(intervals)
        interval_variance = sum((interval - avg_interval) ** 2 for interval in intervals) / len(intervals)
        
        # Lower variance = better distribution
        distribution_score = max(0.0, 1.0 - (interval_variance / (avg_interval ** 2)))
        
        return distribution_score
    
    def _assess_description_quality(self, timestamps: List[Dict[str, Any]]) -> float:
        """Assess quality of timestamp descriptions."""
        if not timestamps:
            return 0.0
        
        quality_score = 0.0
        
        for ts in timestamps:
            description = ts.get('description', '')
            
            # Check description length (optimal 10-50 characters)
            if 10 <= len(description) <= 50:
                quality_score += 0.3
            elif 5 <= len(description) <= 80:
                quality_score += 0.2
            else:
                quality_score += 0.1
            
            # Check for meaningful content (not generic)
            if description and description not in ['Key moment', 'Important section', 'Content section']:
                quality_score += 0.2
            
        return min(1.0, quality_score / len(timestamps))