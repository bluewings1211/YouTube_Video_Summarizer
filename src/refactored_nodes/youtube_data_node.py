"""
Single YouTube data acquisition node.

This node handles all YouTube API interactions in one place to eliminate
duplicate API calls and optimize rate limit usage.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from .validation_nodes import BaseProcessingNode, Store
from ..utils.youtube_fetcher import YouTubeUnifiedFetcher
from ..utils.youtube_core import YouTubeTranscriptError
from ..utils.validators import YouTubeURLValidator
from ..services.video_service import VideoService

logger = logging.getLogger(__name__)


class YouTubeDataNode(BaseProcessingNode):
    """
    Unified YouTube data acquisition node.
    
    This node handles all YouTube API interactions in a single operation:
    - Video metadata extraction
    - Available transcripts discovery
    - Transcript content fetching
    - Video validation and support checking
    """
    
    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0, 
                 enable_enhanced_acquisition: bool = True, 
                 enable_detailed_logging: bool = True,
                 video_service: Optional[VideoService] = None):
        super().__init__("YouTubeDataNode", max_retries, retry_delay)
        self.unified_fetcher = YouTubeUnifiedFetcher(
            enable_detailed_logging=enable_detailed_logging
        )
        self.enable_enhanced_acquisition = enable_enhanced_acquisition
        self.video_service = video_service
    
    def prep(self, store: Store) -> Dict[str, Any]:
        """
        Prepare for YouTube data acquisition by validating input.
        
        Args:
            store: Data store containing 'youtube_url'
            
        Returns:
            Dict containing prep results and validation status
        """
        self.logger.info("Starting YouTube data acquisition preparation")
        
        try:
            # Validate required inputs
            self._validate_inputs(['youtube_url'], store)
            
            youtube_url = store['youtube_url']
            self.logger.debug(f"Processing URL: {youtube_url}")
            
            # Validate YouTube URL format
            def validate_url():
                url_validator = YouTubeURLValidator()
                is_valid_url, video_id = url_validator.validate_and_extract(youtube_url)
                if not is_valid_url or not video_id:
                    raise ValueError(f"Invalid YouTube URL: {youtube_url}")
                return video_id
            
            video_id = self._execute_with_retry(validate_url)
            
            prep_result = {
                'video_id': video_id,
                'youtube_url': youtube_url,
                'validation_status': 'success',
                'prep_timestamp': datetime.utcnow().isoformat(),
                'enhanced_acquisition': self.enable_enhanced_acquisition
            }
            
            self.logger.info(f"YouTube data acquisition preparation successful for video: {video_id}")
            return prep_result
            
        except Exception as e:
            error_info = self._handle_error(e, "YouTube data acquisition preparation failed")
            prep_result = {
                'validation_status': 'failed',
                'error': {
                    'type': error_info.error_type,
                    'message': error_info.message,
                    'timestamp': error_info.timestamp,
                    'is_recoverable': error_info.is_recoverable
                },
                'prep_timestamp': datetime.utcnow().isoformat()
            }
            
            if not error_info.is_recoverable:
                self.logger.error(f"Non-recoverable error in YouTube data acquisition preparation: {str(e)}")
            
            return prep_result
    
    def exec(self, store: Store, prep_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute unified YouTube data acquisition.
        
        Fetches all required data in a single operation:
        1. Video metadata
        2. Available transcripts
        3. Selected transcript content
        
        Args:
            store: Data store
            prep_result: Results from preparation phase
            
        Returns:
            Dict containing all YouTube data
        """
        self.logger.info("Starting unified YouTube data acquisition")
        
        try:
            # Check if preparation was successful
            if prep_result.get('validation_status') != 'success':
                raise ValueError(f"Cannot execute YouTube data acquisition: preparation failed")
            
            video_id = prep_result['video_id']
            self.logger.debug(f"Acquiring all YouTube data for video: {video_id}")
            
            # UNIFIED DATA ACQUISITION: Fetch all data in single optimized operation
            # This reduces API calls from 3 to 2 (metadata + transcript)
            def fetch_all_youtube_data():
                self.logger.debug(f"Fetching all YouTube data for {video_id} using unified approach")
                return self.unified_fetcher.fetch_all_data(
                    video_id=video_id,
                    languages=None,  # Use default language detection
                    max_duration_seconds=1800,  # 30 minutes max
                    check_unsupported=True
                )
            
            # Single retry wrapper for the entire unified operation
            youtube_data = self._execute_with_retry(fetch_all_youtube_data)
            
            # Process and enhance data
            processed_data = self._process_youtube_data(youtube_data)
            
            # Generate statistics
            stats = self._generate_youtube_stats(processed_data)
            
            exec_result = {
                'youtube_data': processed_data,
                'youtube_stats': stats,
                'processing_metadata': {
                    'enhanced_acquisition_used': self.enable_enhanced_acquisition,
                    'selected_language': youtube_data['selected_transcript'].get('language_code'),
                    'available_languages': [t.get('language_code') for t in youtube_data['available_transcripts']],
                    'processing_timestamp': datetime.utcnow().isoformat(),
                    'api_calls_made': youtube_data.get('api_calls_made', 2),  # Unified approach: 2 calls
                    'optimization_applied': youtube_data.get('optimization_applied', True)
                },
                'execution_status': 'success'
            }
            
            self.logger.info(f"YouTube data acquisition successful for video: {video_id}")
            return exec_result
            
        except Exception as e:
            error_info = self._handle_error(e, "YouTube data acquisition failed")
            exec_result = {
                'execution_status': 'failed',
                'error': {
                    'type': error_info.error_type,
                    'message': error_info.message,
                    'timestamp': error_info.timestamp,
                    'is_recoverable': error_info.is_recoverable
                },
                'processing_timestamp': datetime.utcnow().isoformat()
            }
            
            if not error_info.is_recoverable:
                self.logger.error(f"Non-recoverable error in YouTube data acquisition: {str(e)}")
            
            return exec_result
    
    def post(self, store: Store, prep_result: Dict[str, Any], exec_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Post-process YouTube data and update store.
        
        Args:
            store: Data store
            prep_result: Results from preparation phase
            exec_result: Results from execution phase
            
        Returns:
            Dict containing final processing results
        """
        self.logger.info("Starting YouTube data post-processing")
        
        try:
            # Handle both success and partial success cases
            if exec_result.get('execution_status') == 'success':
                # Full success - use all acquired data
                youtube_data = exec_result['youtube_data']
                
                # Update store with all YouTube data
                store_updates = {
                    'video_metadata': youtube_data.get('video_metadata', {}),
                    'transcript_data': youtube_data.get('transcript_data', {}),
                    'available_transcripts': youtube_data.get('available_transcripts', []),
                    'selected_transcript': youtube_data.get('selected_transcript', {}),
                    'video_id': youtube_data.get('video_metadata', {}).get('video_id', prep_result.get('video_id')),
                    'video_title': youtube_data.get('video_metadata', {}).get('title', ''),
                    'video_duration': youtube_data.get('video_metadata', {}).get('duration_seconds', 0),
                    'youtube_stats': exec_result.get('youtube_stats', {}),
                    'processing_metadata': exec_result.get('processing_metadata', {})
                }
                
                post_result = {
                    'processing_status': 'success',
                    'video_id': store_updates['video_id'],
                    'store_updated': True,
                    'post_timestamp': datetime.utcnow().isoformat(),
                    'database_saved': False,
                    'data_completeness': 'full'
                }
                
            elif exec_result.get('execution_status') == 'failed':
                # Execution failed - try to save any partial data we might have
                self.logger.warning("Execution failed, attempting to save any available partial data")
                
                # Try to extract any partial metadata from prep phase
                store_updates = {
                    'video_id': prep_result.get('video_id', ''),
                    'youtube_url': prep_result.get('youtube_url', ''),
                    'processing_error': exec_result.get('error', {})
                }
                
                post_result = {
                    'processing_status': 'failed',
                    'video_id': prep_result.get('video_id', ''),
                    'store_updated': True,
                    'post_timestamp': datetime.utcnow().isoformat(),
                    'database_saved': False,
                    'data_completeness': 'none',
                    'error': exec_result.get('error', {})
                }
            else:
                raise ValueError("Unknown execution status")
            
            # Apply store updates
            self._safe_store_update(store, store_updates)
            
            # Calculate processing metrics
            processing_time = self._measure_execution_time(
                prep_result.get('prep_timestamp', ''),
                exec_result.get('processing_timestamp', '')
            )
            post_result['processing_time_seconds'] = processing_time
            
            # Create video record in database only if execution was successful
            if exec_result.get('execution_status') == 'success':
                self._create_video_record(exec_result, processing_time)
            else:
                self.logger.info(f"Skipping video record creation due to execution status: {exec_result.get('execution_status')}")
            
            self.logger.info(f"YouTube data post-processing completed: {post_result['processing_status']}")
            return post_result
            
        except Exception as e:
            error_info = self._handle_error(e, "YouTube data post-processing failed")
            post_result = {
                'processing_status': 'failed',
                'error': {
                    'type': error_info.error_type,
                    'message': error_info.message,
                    'timestamp': error_info.timestamp,
                    'is_recoverable': error_info.is_recoverable
                },
                'store_updated': False,
                'post_timestamp': datetime.utcnow().isoformat()
            }
            
            if not error_info.is_recoverable:
                self.logger.error(f"Non-recoverable error in YouTube data post-processing: {str(e)}")
            
            return post_result
    
    # _select_best_transcript method removed - now handled by YouTubeUnifiedFetcher internally
    
    def _process_youtube_data(self, youtube_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process and enhance the acquired YouTube data."""
        # Extract transcript content for easier access
        transcript_data = youtube_data.get('transcript_data', {})
        
        # Create processed structure with backward compatibility
        transcript_text = transcript_data.get('transcript', '')
        
        processed = {
            'video_metadata': youtube_data.get('video_metadata', {}),
            'transcript_data': {
                'full_text': transcript_text,
                'transcript_text': transcript_text,  # For backward compatibility with other nodes
                'transcript': transcript_text,       # For internal consistency 
                'segments': transcript_data.get('raw_transcript', []),
                'raw_transcript': transcript_data.get('raw_transcript', []),  # For backward compatibility
                'language': transcript_data.get('language', 'unknown'),
                'word_count': transcript_data.get('word_count', 0),
                'duration': transcript_data.get('duration_seconds', 0),
                'duration_seconds': transcript_data.get('duration_seconds', 0)  # For backward compatibility
            },
            'available_transcripts': youtube_data.get('available_transcripts', []),
            'selected_transcript': youtube_data.get('selected_transcript', {})
        }
        
        return processed
    
    def _generate_youtube_stats(self, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate statistics about the acquired YouTube data."""
        video_metadata = processed_data.get('video_metadata', {})
        transcript_data = processed_data.get('transcript_data', {})
        
        stats = {
            'video_info': {
                'duration_seconds': video_metadata.get('duration_seconds', 0),
                'view_count': video_metadata.get('view_count', 0),
                'channel': video_metadata.get('channel_name', 'Unknown')
            },
            'transcript_info': {
                'language': transcript_data.get('language', 'unknown'),
                'word_count': transcript_data.get('word_count', 0),
                'segment_count': len(transcript_data.get('segments', [])),
                'is_auto_generated': processed_data.get('selected_transcript', {}).get('is_generated', True)
            },
            'acquisition_info': {
                'available_languages': len(processed_data.get('available_transcripts', [])),
                'acquisition_timestamp': datetime.utcnow().isoformat()
            }
        }
        
        return stats
    
    def _create_video_record(self, exec_result: Dict[str, Any], processing_time: float) -> None:
        """
        Create video record in database using the video service.
        
        Args:
            exec_result: Result from exec containing video metadata and transcript
            processing_time: Processing time in seconds
        """
        self.logger.info(f"Creating video record - video_service available: {self.video_service is not None}")
        if not self.video_service:
            self.logger.warning("No video service available, skipping video record creation")
            return
        
        try:
            # Extract data from exec_result - correct structure
            youtube_data = exec_result.get('youtube_data', {})
            video_metadata = youtube_data.get('video_metadata', {})
            transcript_data = youtube_data.get('transcript_data', {})
            
            if not video_metadata.get('video_id'):
                self.logger.warning(f"No video_id found in video_metadata: {list(video_metadata.keys())}")
                return
            
            # Debug: Log video_metadata keys and duration value
            duration_value = video_metadata.get('duration', video_metadata.get('duration_seconds', 0))
            self.logger.info(f"Video metadata keys: {list(video_metadata.keys())}")
            self.logger.info(f"Duration value: {duration_value} (type: {type(duration_value)})")
            
            # Ensure duration is positive
            if duration_value <= 0:
                self.logger.warning(f"Duration was {duration_value}, setting to 1 second minimum")
                duration_value = 1  # Set minimum duration to 1 second
            
            # Create video record using VideoService
            video_record = self.video_service.create_video_record(
                video_id=video_metadata['video_id'],
                title=video_metadata.get('title', 'Unknown Title'),
                duration=duration_value,
                url=f"https://www.youtube.com/watch?v={video_metadata['video_id']}",
                channel_name=video_metadata.get('channel_name'),
                description=video_metadata.get('description'),
                published_date=video_metadata.get('published_date'),
                view_count=video_metadata.get('view_count'),
                transcript_content=transcript_data.get('transcript_text', ''),
                transcript_language=transcript_data.get('language', 'unknown')
            )
            
            if video_record:
                self.logger.info(f"Successfully created video record for {video_metadata['video_id']}")
            else:
                self.logger.warning(f"Failed to create video record for {video_metadata['video_id']}")
                
        except Exception as e:
            self.logger.error(f"Error creating video record: {str(e)}")
            # Don't raise the exception to avoid breaking the workflow
            # The workflow can continue without database storage