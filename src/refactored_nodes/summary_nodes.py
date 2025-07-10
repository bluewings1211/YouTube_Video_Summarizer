"""
Summary and timestamp generation nodes.

This module contains nodes responsible for generating timestamped segments
and analyzing video content for important moments.
"""

import logging
import re
from typing import Dict, Any, List, Optional
from datetime import datetime

from .validation_nodes import BaseProcessingNode, Store
from ..utils.smart_llm_client import SmartLLMClient, TaskRequirements, detect_task_requirements
from ..services.video_service import VideoService

logger = logging.getLogger(__name__)


class TimestampNode(BaseProcessingNode):
    """
    Node for generating timestamped URLs with descriptions and importance ratings.
    
    This node analyzes video transcripts to identify key moments and creates
    timestamped YouTube URLs with AI-generated descriptions and importance ratings.
    """
    
    def __init__(self, max_retries: int = 3, retry_delay: float = 2.0, video_service: Optional[VideoService] = None):
        super().__init__("TimestampNode", max_retries, retry_delay)
        self.smart_llm_client = None
        self.default_timestamp_count = 5
        self.video_service = video_service
    
    def prep(self, store: Store) -> Dict[str, Any]:
        """
        Prepare for timestamp generation by validating input data.
        
        Args:
            store: Data store containing transcript and video data
            
        Returns:
            Dict containing prep results and configuration
        """
        self.logger.info("Starting timestamp preparation")
        
        try:
            # Validate required input
            required_keys = ['transcript_data', 'video_metadata']
            is_valid, missing_keys = self._validate_store_data(store, required_keys)
            if not is_valid:
                raise ValueError(f"Missing required data: {missing_keys}")
            
            transcript_data = store['transcript_data']
            video_metadata = store['video_metadata']
            
            # Validate transcript data
            raw_transcript = transcript_data.get('raw_transcript', [])
            video_id = store.get('video_id', transcript_data.get('video_id', ''))
            
            if not raw_transcript:
                raise ValueError("No raw transcript data available for timestamp generation")
            
            if not video_id:
                raise ValueError("Video ID is required for timestamp generation")
            
            # Validate video metadata
            video_title = video_metadata.get('title', 'Unknown')
            video_duration = video_metadata.get('duration_seconds', 0)
            
            if video_duration is None or video_duration <= 0:
                self.logger.warning("Video duration not available, proceeding anyway")
            
            # Initialize Smart LLM client
            try:
                self.smart_llm_client = SmartLLMClient()
            except Exception as e:
                raise ValueError(f"Failed to initialize Smart LLM client: {str(e)}")
            
            # Filter and validate transcript entries
            valid_transcript = self._filter_transcript_entries(raw_transcript)
            
            if len(valid_transcript) == 0:
                self.logger.warning("No valid transcript entries found, using empty fallback")
                # Use empty fallback for completely empty transcript
                timestamp_count = 0
            elif len(valid_transcript) < 3:
                self.logger.warning(f"Limited transcript data: only {len(valid_transcript)} entries available for timestamp generation")
                # Proceed with available data, but adjust timestamp count
                timestamp_count = min(self.default_timestamp_count, len(valid_transcript))
            else:
                timestamp_count = self.default_timestamp_count
            
            prep_result = {
                'video_id': video_id,
                'video_title': video_title,
                'video_duration': video_duration,
                'raw_transcript': valid_transcript,
                'transcript_count': len(valid_transcript),
                'timestamp_count': timestamp_count,
                'prep_timestamp': datetime.utcnow().isoformat(),
                'prep_status': 'success'
            }
            
            self.logger.info(f"Timestamp prep successful for video {video_id}")
            return prep_result
            
        except Exception as e:
            error_info = self._handle_error(e, "Timestamp preparation failed")
            return {
                'prep_status': 'failed',
                'error': error_info.__dict__,
                'prep_timestamp': datetime.utcnow().isoformat()
            }
    
    def exec(self, store: Store, prep_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute timestamp generation with AI analysis.
        
        Args:
            store: Data store
            prep_result: Results from prep phase
            
        Returns:
            Dict containing generated timestamps and metadata
        """
        self.logger.info("Starting timestamp execution")
        
        if prep_result.get('prep_status') != 'success':
            return {
                'exec_status': 'failed',
                'error': 'Prep phase failed',
                'exec_timestamp': datetime.utcnow().isoformat()
            }
        
        video_id = prep_result['video_id']
        video_title = prep_result['video_title']
        raw_transcript = prep_result['raw_transcript']
        timestamp_count = prep_result['timestamp_count']
        last_error = None
        
        for retry_count in range(self.max_retries + 1):
            try:
                self._retry_with_delay(retry_count)
                
                # Get language detection results for language-specific processing
                language_detection_result = store.get('language_detection_result', {})
                detected_language = language_detection_result.get('detected_language', 'en')
                is_chinese = language_detection_result.get('is_chinese', False)
                
                # Create task requirements for smart model selection
                transcript_text = self._format_transcript_for_analysis(raw_transcript)
                task_requirements = detect_task_requirements(
                    text=transcript_text,
                    task_type="timestamps",
                    quality_level="medium"
                )
                
                # Add language information to task requirements
                if detected_language:
                    task_requirements.language = detected_language
                    task_requirements.is_chinese = is_chinese
                
                # Generate timestamps using smart client with language awareness
                timestamps_result = self._generate_timestamps_with_smart_client(
                    raw_transcript,
                    video_id,
                    timestamp_count,
                    task_requirements,
                    is_chinese=is_chinese
                )
                
                # Validate generated timestamps
                timestamps = timestamps_result['timestamps']
                
                if not timestamps:
                    raise ValueError("No timestamps generated")
                
                # Enhance timestamps with additional metadata
                enhanced_timestamps = self._enhance_timestamps(
                    timestamps, 
                    video_title,
                    prep_result['video_duration']
                )
                
                # Calculate timestamp statistics
                timestamp_stats = self._calculate_timestamp_stats(
                    enhanced_timestamps,
                    raw_transcript
                )
                
                exec_result = {
                    'exec_status': 'success',
                    'timestamps': enhanced_timestamps,
                    'timestamp_count': len(enhanced_timestamps),
                    'requested_count': timestamp_count,
                    'video_id': video_id,
                    'video_title': video_title,
                    'timestamp_stats': timestamp_stats,
                    'llm_metadata': timestamps_result['generation_metadata'],
                    'exec_timestamp': datetime.utcnow().isoformat(),
                    'retry_count': retry_count
                }
                
                self.logger.info(f"Timestamp generation successful: {len(enhanced_timestamps)} timestamps")
                return exec_result
                
            except Exception as e:
                last_error = self._handle_error(e, f"Timestamp execution failed", retry_count)
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
        Post-process timestamps and update store.
        
        Args:
            store: Data store
            prep_result: Results from prep phase
            exec_result: Results from exec phase
            
        Returns:
            Dict containing final processing results
        """
        self.logger.info("Starting timestamp post-processing")
        
        try:
            if exec_result.get('exec_status') != 'success':
                return {
                    'post_status': 'failed',
                    'error': 'Execution phase failed',
                    'post_timestamp': datetime.utcnow().isoformat()
                }
            
            # Extract timestamp information
            timestamps = exec_result['timestamps']
            video_id = exec_result['video_id']
            video_title = exec_result['video_title']
            
            # Sort timestamps by importance rating (descending)
            sorted_timestamps = sorted(
                timestamps, 
                key=lambda x: x.get('importance_rating', 0), 
                reverse=True
            )
            
            # Create different timestamp groupings
            high_importance = [t for t in timestamps if t.get('importance_rating', 0) >= 8]
            medium_importance = [t for t in timestamps if 5 <= t.get('importance_rating', 0) < 8]
            low_importance = [t for t in timestamps if t.get('importance_rating', 0) < 5]
            
            # Prepare store data
            store_data = {
                'timestamps': timestamps,  # Store at top level for API compatibility
                'timestamp_data': {
                    'timestamps': timestamps,
                    'sorted_by_importance': sorted_timestamps,
                    'high_importance': high_importance,
                    'medium_importance': medium_importance,
                    'low_importance': low_importance,
                    'count': len(timestamps),
                    'stats': exec_result['timestamp_stats'],
                    'generated_at': exec_result['exec_timestamp']
                },
                'timestamp_metadata': {
                    'video_id': video_id,
                    'video_title': video_title,
                    'requested_count': exec_result['requested_count'],
                    'actual_count': len(timestamps),
                    'avg_importance': sum(t.get('importance_rating', 0) for t in timestamps) / len(timestamps),
                    'processing_duration': self._calculate_duration(
                        prep_result.get('prep_timestamp', ''),
                        exec_result.get('exec_timestamp', '')
                    ),
                    'retry_count': exec_result.get('retry_count', 0),
                    'llm_provider': exec_result.get('llm_metadata', {}).get('provider', 'unknown'),
                    'llm_model': exec_result.get('llm_metadata', {}).get('model', 'unknown')
                }
            }
            
            # Update store
            self._safe_store_update(store, store_data)
            
            post_result = {
                'post_status': 'success',
                'timestamps_ready': True,
                'timestamp_count': len(timestamps),
                'high_importance_count': len(high_importance),
                'avg_importance_rating': round(sum(t.get('importance_rating', 0) for t in timestamps) / len(timestamps), 2),
                'video_id': video_id,
                'video_title': video_title,
                'post_timestamp': datetime.utcnow().isoformat(),
                'database_saved': False
            }
            
            # Save timestamped segments to database if video service is available
            if self.video_service:
                try:
                    # Save timestamps directly (now synchronous)
                    self.video_service.save_timestamped_segments(
                        video_id=video_id,
                        segments_data={
                            'timestamps': timestamps,
                            'sorted_by_importance': sorted_timestamps,
                            'high_importance': high_importance,
                            'medium_importance': medium_importance,
                            'low_importance': low_importance,
                            'count': len(timestamps),
                            'stats': exec_result['timestamp_stats'],
                            'generated_at': exec_result['exec_timestamp']
                        }
                    )
                    
                    post_result['database_saved'] = True
                    self.logger.info(f"Timestamped segments saved to database for video: {video_id}")
                    
                except Exception as db_error:
                    self.logger.warning(f"Failed to save timestamped segments to database: {str(db_error)}")
                    post_result['database_saved'] = False
                    post_result['database_error'] = str(db_error)
            
            self.logger.info(f"Timestamp post-processing successful for video {video_id}")
            return post_result
            
        except Exception as e:
            error_info = self._handle_error(e, "Timestamp post-processing failed")
            return {
                'post_status': 'failed',
                'error': error_info.__dict__,
                'post_timestamp': datetime.utcnow().isoformat()
            }
    
    def _filter_transcript_entries(self, raw_transcript: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter and validate transcript entries for timestamp generation."""
        valid_entries = []
        
        for entry in raw_transcript:
            if not isinstance(entry, dict):
                continue
            
            # Validate required fields
            start_time = entry.get('start')
            text = entry.get('text', '').strip()
            
            if start_time is None or not text:
                continue
            
            # Convert start time to float if needed
            try:
                start_time = float(start_time)
            except (ValueError, TypeError):
                continue
            
            # Skip very short text segments
            if len(text.split()) < 3:
                continue
            
            valid_entries.append({
                'start': start_time,
                'text': text,
                'duration': entry.get('duration', 0)
            })
        
        return valid_entries
    
    def _enhance_timestamps(
        self, 
        timestamps: List[Dict[str, Any]], 
        video_title: str, 
        video_duration: int
    ) -> List[Dict[str, Any]]:
        """Enhance timestamps with additional metadata."""
        enhanced = []
        
        for i, timestamp in enumerate(timestamps):
            enhanced_timestamp = timestamp.copy()
            
            # Add position information
            enhanced_timestamp['position'] = i + 1
            enhanced_timestamp['video_title'] = video_title
            
            # Add relative position in video
            if video_duration > 0:
                relative_position = (timestamp['timestamp_seconds'] / video_duration) * 100
                enhanced_timestamp['relative_position_percent'] = round(relative_position, 2)
            
            # Add context tags based on importance
            importance = timestamp.get('importance_rating', 0)
            if importance >= 8:
                enhanced_timestamp['context_tag'] = 'critical'
            elif importance >= 6:
                enhanced_timestamp['context_tag'] = 'important'
            elif importance >= 4:
                enhanced_timestamp['context_tag'] = 'notable'
            else:
                enhanced_timestamp['context_tag'] = 'minor'
            
            # Add shareable title
            enhanced_timestamp['shareable_title'] = f"{video_title} - {timestamp['description']}"
            
            enhanced.append(enhanced_timestamp)
        
        return enhanced
    
    def _format_transcript_for_analysis(self, raw_transcript: List[Dict[str, Any]]) -> str:
        """Format transcript data for text analysis."""
        transcript_text = ""
        for i, entry in enumerate(raw_transcript[:50]):  # Use first 50 entries
            start_time = entry.get('start', 0)
            text = entry.get('text', '')
            transcript_text += f"[{start_time:.1f}s] {text}\n"
        return transcript_text
    
    def _generate_timestamps_with_smart_client(
        self, 
        raw_transcript: List[Dict[str, Any]], 
        video_id: str, 
        count: int,
        task_requirements: TaskRequirements,
        is_chinese: bool = False
    ) -> Dict[str, Any]:
        """Generate timestamps using the smart LLM client with semantic analysis."""
        
        # First, try to use semantic analysis for better timestamp accuracy
        try:
            semantic_timestamps = self._generate_semantic_timestamps(
                raw_transcript, video_id, count
            )
            
            if semantic_timestamps and len(semantic_timestamps) >= count // 2:
                self.logger.info(f"Semantic analysis generated {len(semantic_timestamps)} high-quality timestamps")
                return {
                    'timestamps': semantic_timestamps,
                    'count': len(semantic_timestamps),
                    'requested_count': count,
                    'video_id': video_id,
                    'generation_metadata': {
                        'method': 'semantic_analysis',
                        'text': 'Generated using advanced semantic analysis',
                        'model': 'SemanticAnalysisService'
                    }
                }
        except Exception as e:
            self.logger.warning(f"Semantic analysis failed, falling back to LLM: {str(e)}")
        
        # Fallback to original LLM-based method
        # Format transcript for processing
        transcript_text = self._format_transcript_for_analysis(raw_transcript)
        
        # Create timestamp prompt
        system_prompt = f"""You are an expert at identifying the most important moments in video content. Analyze the provided transcript and identify the {count} most significant timestamps.

Guidelines:
- Focus on key insights, important announcements, or turning points
- Include a brief description of what happens at each timestamp
- Rate the importance on a scale of 1-10
- Provide timestamps in the format: [timestamp]s
- Choose moments that viewers would want to jump to directly"""

        user_prompt = f"""Analyze this transcript and identify the {count} most important timestamps:

{transcript_text}

IMPORTANT: You must respond with EXACTLY {count} timestamps. Each timestamp must be on a separate line.

For each timestamp, provide:
1. The timestamp in seconds
2. A brief description (10-20 words)
3. Importance rating (1-10)

Format each entry EXACTLY as:
[XXX.X]s (Rating: X/10) - Description

Example format:
[10.5]s (Rating: 8/10) - Introduction to main topic
[45.2]s (Rating: 9/10) - Key insight about the subject
[120.0]s (Rating: 7/10) - Summary of important points

Do not include any other text or explanations."""

        # Generate with smart client using language-appropriate method
        if is_chinese:
            # Use Chinese-optimized processing
            result = self.smart_llm_client.generate_text_with_chinese_optimization(
                text=user_prompt,
                task_requirements=task_requirements,
                max_tokens=600,
                temperature=0.4
            )
        else:
            # Use standard processing for English/other languages
            result = self.smart_llm_client.generate_text_with_fallback(
                prompt=user_prompt,
                task_requirements=task_requirements,
                max_tokens=600,
                temperature=0.4
            )
        
        # Parse the timestamps from the response
        timestamps_text = result['text']
        timestamps = self._parse_timestamps(timestamps_text, video_id)
        
        # If no timestamps were parsed, create fallback timestamps
        if not timestamps:
            self.logger.warning("No timestamps parsed from LLM response, generating fallback timestamps")
            timestamps = self._generate_fallback_timestamps(raw_transcript, video_id, count)
        
        return {
            'timestamps': timestamps,
            'count': len(timestamps),
            'requested_count': count,
            'video_id': video_id,
            'generation_metadata': result
        }
    
    def _generate_semantic_timestamps(
        self, 
        raw_transcript: List[Dict[str, Any]], 
        video_id: str, 
        count: int
    ) -> List[Dict[str, Any]]:
        """Generate timestamps using semantic analysis for improved accuracy."""
        try:
            # Import semantic analysis service
            from ..services.semantic_analysis_service import create_semantic_analysis_service
            
            # Create semantic analysis service
            semantic_service = create_semantic_analysis_service()
            
            # Analyze transcript
            analysis_result = semantic_service.analyze_transcript(
                raw_transcript=raw_transcript,
                video_id=video_id,
                video_title="",  # We don't have title here, but service can work without it
                target_timestamp_count=count
            )
            
            if analysis_result['status'] != 'success':
                self.logger.warning(f"Semantic analysis failed: {analysis_result.get('reason', 'Unknown error')}")
                return []
            
            semantic_timestamps = analysis_result.get('timestamps', [])
            
            if not semantic_timestamps:
                self.logger.warning("Semantic analysis returned no timestamps")
                return []
            
            # Convert semantic timestamps to the expected format
            converted_timestamps = []
            for semantic_ts in semantic_timestamps:
                converted_timestamps.append({
                    'timestamp_seconds': semantic_ts.timestamp_seconds,
                    'timestamp_formatted': semantic_ts.timestamp_formatted,
                    'description': semantic_ts.description,
                    'importance_rating': semantic_ts.importance_rating,
                    'youtube_url': semantic_ts.youtube_url,
                    'semantic_metadata': {
                        'cluster_id': semantic_ts.semantic_cluster_id,
                        'cluster_theme': semantic_ts.cluster_theme,
                        'keywords': semantic_ts.semantic_keywords,
                        'confidence_score': semantic_ts.confidence_score,
                        'context_before': semantic_ts.context_before,
                        'context_after': semantic_ts.context_after
                    }
                })
            
            self.logger.info(f"Successfully generated {len(converted_timestamps)} semantic timestamps")
            return converted_timestamps
            
        except ImportError as e:
            self.logger.warning(f"Semantic analysis service not available: {str(e)}")
            return []
        except Exception as e:
            self.logger.error(f"Semantic timestamp generation failed: {str(e)}")
            return []
    
    def _calculate_timestamp_stats(
        self, 
        timestamps: List[Dict[str, Any]], 
        raw_transcript: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate statistics for generated timestamps."""
        if not timestamps:
            return {}
        
        # Calculate importance statistics
        importance_ratings = [t.get('importance_rating', 0) for t in timestamps]
        
        # Calculate time distribution
        timestamp_times = [t['timestamp_seconds'] for t in timestamps]
        total_video_time = max(entry.get('start', 0) for entry in raw_transcript) if raw_transcript else 0
        
        # Calculate coverage
        coverage_percent = 0
        if total_video_time > 0:
            time_coverage = max(timestamp_times) - min(timestamp_times)
            coverage_percent = (time_coverage / total_video_time) * 100
        
        return {
            'count': len(timestamps),
            'avg_importance': round(sum(importance_ratings) / len(importance_ratings), 2),
            'max_importance': max(importance_ratings),
            'min_importance': min(importance_ratings),
            'high_importance_count': len([r for r in importance_ratings if r >= 8]),
            'medium_importance_count': len([r for r in importance_ratings if 5 <= r < 8]),
            'low_importance_count': len([r for r in importance_ratings if r < 5]),
            'time_coverage_percent': round(coverage_percent, 2),
            'earliest_timestamp': min(timestamp_times),
            'latest_timestamp': max(timestamp_times),
            'avg_timestamp_length': round(sum(len(t['description'].split()) for t in timestamps) / len(timestamps), 2)
        }
    
    def _parse_timestamps(self, timestamps_text: str, video_id: str) -> List[Dict[str, Any]]:
        """Parse generated timestamps into structured format."""
        timestamps = []
        
        # Log the raw response for debugging
        self.logger.debug(f"Raw timestamps response: {timestamps_text}")
        
        for line in timestamps_text.split('\n'):
            if not line.strip():
                continue
            
            # Try multiple patterns to be more robust
            patterns = [
                # Original pattern: [10.5]s (Rating: 8/10) - Description
                r'\[(\d+(?:\.\d+)?)\]s\s*\(Rating:\s*(\d+)/10\)\s*-\s*(.+)',
                # Alternative pattern: 10.5s (Rating: 8/10) - Description
                r'(\d+(?:\.\d+)?)s\s*\(Rating:\s*(\d+)/10\)\s*-\s*(.+)',
                # Simpler pattern: [10.5]s - Description
                r'\[(\d+(?:\.\d+)?)\]s\s*-\s*(.+)',
                # Even simpler: 10.5s - Description
                r'(\d+(?:\.\d+)?)s\s*-\s*(.+)',
                # Just timestamp and description: 10.5 - Description
                r'(\d+(?:\.\d+)?)\s*-\s*(.+)',
                # Bullet point format: • 10.5s - Description
                r'[•\-\*]\s*(\d+(?:\.\d+)?)s?\s*-\s*(.+)',
                # Number format: 1. 10.5s - Description
                r'\d+\.\s*(\d+(?:\.\d+)?)s?\s*-\s*(.+)',
            ]
            
            matched = False
            for pattern in patterns:
                match = re.match(pattern, line.strip())
                if match:
                    timestamp_seconds = float(match.group(1))
                    groups = match.groups()
                    
                    # Handle different patterns with different group counts
                    if len(groups) >= 3 and groups[1] and groups[1].isdigit():
                        rating = int(groups[1])
                        description = groups[2].strip()
                    elif len(groups) >= 2:
                        rating = 5  # Default rating
                        description = groups[1].strip()
                    else:
                        rating = 5
                        description = groups[0].strip() if groups else "No description"
                    
                    # Generate YouTube URL with timestamp
                    youtube_url = f"https://www.youtube.com/watch?v={video_id}&t={int(timestamp_seconds)}s"
                    
                    timestamps.append({
                        'timestamp_seconds': timestamp_seconds,
                        'timestamp_formatted': self._format_timestamp(timestamp_seconds),
                        'description': description,
                        'importance_rating': rating,
                        'youtube_url': youtube_url
                    })
                    matched = True
                    break
            
            if not matched:
                self.logger.warning(f"Could not parse timestamp line: {line.strip()}")
        
        # Sort by timestamp
        timestamps.sort(key=lambda x: x['timestamp_seconds'])
        
        # Log results for debugging
        self.logger.debug(f"Parsed {len(timestamps)} timestamps from response")
        
        return timestamps
    
    def _generate_fallback_timestamps(self, raw_transcript: List[Dict[str, Any]], video_id: str, count: int) -> List[Dict[str, Any]]:
        """Generate fallback timestamps when LLM parsing fails."""
        timestamps = []
        
        if not raw_transcript:
            return timestamps
        
        # Calculate video duration
        total_duration = sum(segment.get('duration', 0) for segment in raw_transcript)
        if total_duration == 0:
            # Fallback: use last segment's end time
            last_segment = raw_transcript[-1]
            total_duration = last_segment.get('start', 0) + last_segment.get('duration', 30)
        
        # Generate evenly spaced timestamps
        for i in range(count):
            timestamp_seconds = (i + 1) * (total_duration / (count + 1))
            
            # Find the closest transcript segment
            closest_segment = min(raw_transcript, key=lambda x: abs(x.get('start', 0) - timestamp_seconds))
            
            # Generate YouTube URL with timestamp
            youtube_url = f"https://www.youtube.com/watch?v={video_id}&t={int(timestamp_seconds)}s"
            
            timestamps.append({
                'timestamp_seconds': timestamp_seconds,
                'timestamp_formatted': self._format_timestamp(timestamp_seconds),
                'description': f"Key moment: {closest_segment.get('text', 'Content at this timestamp')[:50]}...",
                'importance_rating': 5,  # Default rating
                'youtube_url': youtube_url
            })
        
        return timestamps
    
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
    
    def _calculate_duration(self, start_time: str, end_time: str) -> float:
        """Calculate duration between two timestamps."""
        try:
            start = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
            end = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
            return (end - start).total_seconds()
        except Exception:
            return 0.0