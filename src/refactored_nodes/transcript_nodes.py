"""
YouTube transcript extraction and processing nodes.

This module contains nodes responsible for fetching and processing YouTube
video transcripts, including metadata extraction and validation.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from .validation_nodes import BaseProcessingNode, Store
from ..utils.youtube_api import YouTubeTranscriptFetcher, YouTubeTranscriptError
from ..utils.validators import YouTubeURLValidator
from ..services.video_service import VideoService

logger = logging.getLogger(__name__)


class YouTubeTranscriptNode(BaseProcessingNode):
    """
    Node for fetching YouTube video transcripts with metadata.
    
    This node handles the extraction of video transcripts from YouTube URLs,
    including video validation, metadata extraction, and error handling.
    """
    
    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0, 
                 enable_enhanced_acquisition: bool = True, enable_detailed_logging: bool = True,
                 video_service: Optional[VideoService] = None):
        super().__init__("YouTubeTranscriptNode", max_retries, retry_delay)
        self.transcript_fetcher = YouTubeTranscriptFetcher(
            enable_detailed_logging=enable_detailed_logging
        )
        self.enable_enhanced_acquisition = enable_enhanced_acquisition
        self.video_service = video_service
    
    def prep(self, store: Store) -> Dict[str, Any]:
        """
        Prepare for transcript extraction by validating input and checking video support.
        
        Args:
            store: Data store containing 'youtube_url'
            
        Returns:
            Dict containing prep results and validation status
        """
        self.logger.info("Starting transcript preparation")
        
        try:
            # Validate required inputs using enhanced validation
            self._validate_inputs(['youtube_url'], store)
            
            youtube_url = store['youtube_url']
            self.logger.debug(f"Processing URL: {youtube_url}")
            
            # Validate YouTube URL format with enhanced validation
            def validate_url():
                url_validator = YouTubeURLValidator()
                is_valid_url, video_id = url_validator.validate_and_extract(youtube_url)
                if not is_valid_url or not video_id:
                    raise ValueError(f"Invalid YouTube URL: {youtube_url}")
                return video_id
            
            video_id = self._execute_with_retry(validate_url)
            
            # Collect video metadata with enhanced metadata extraction
            def fetch_metadata():
                metadata = self.transcript_fetcher.get_video_metadata(video_id)
                if not metadata:
                    raise ValueError(f"Could not fetch metadata for video: {video_id}")
                return metadata
            
            video_metadata = self._execute_with_retry(fetch_metadata)
            
            # Enhanced video support validation
            def validate_video():
                # Check if video is available for transcript extraction
                if not self.transcript_fetcher.is_video_supported(video_id):
                    raise ValueError(f"Video not supported for transcript extraction: {video_id}")
                
                # Check transcript availability
                available_transcripts = self.transcript_fetcher.get_available_transcripts(video_id)
                if not available_transcripts:
                    raise ValueError(f"No transcripts available for video: {video_id}")
                
                return available_transcripts
            
            available_transcripts = self._execute_with_retry(validate_video)
            
            # Determine best transcript language
            best_transcript = self._select_best_transcript(available_transcripts)
            
            prep_result = {
                'video_id': video_id,
                'video_metadata': video_metadata,
                'available_transcripts': available_transcripts,
                'selected_transcript': best_transcript,
                'validation_status': 'success',
                'prep_timestamp': datetime.utcnow().isoformat(),
                'enhanced_acquisition': self.enable_enhanced_acquisition
            }
            
            self.logger.info(f"Transcript preparation successful for video: {video_id}")
            return prep_result
            
        except Exception as e:
            error_info = self._handle_error(e, "Transcript preparation failed")
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
                self.logger.error(f"Non-recoverable error in transcript preparation: {str(e)}")
            
            return prep_result
    
    def exec(self, store: Store, prep_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute transcript fetching with enhanced acquisition capabilities.
        
        Args:
            store: Data store
            prep_result: Results from preparation phase
            
        Returns:
            Dict containing transcript data and processing results
        """
        self.logger.info("Starting transcript execution")
        
        try:
            # Check if preparation was successful
            if prep_result.get('validation_status') != 'success':
                raise ValueError(f"Cannot execute transcript fetch: preparation failed")
            
            video_id = prep_result['video_id']
            selected_transcript = prep_result['selected_transcript']
            
            self.logger.debug(f"Fetching transcript for video: {video_id}")
            
            # Fetch transcript with enhanced acquisition
            def fetch_transcript():
                transcript_data = self.transcript_fetcher.get_transcript(
                    video_id, 
                    language_code=selected_transcript.get('language_code'),
                    enable_enhanced_acquisition=self.enable_enhanced_acquisition
                )
                if not transcript_data:
                    raise ValueError(f"Failed to fetch transcript for video: {video_id}")
                return transcript_data
            
            transcript_result = self._execute_with_retry(fetch_transcript)
            
            # Process and enhance transcript data
            processed_transcript = self._process_transcript_data(
                transcript_result, 
                prep_result['video_metadata']
            )
            
            # Generate transcript statistics
            transcript_stats = self._generate_transcript_stats(processed_transcript)
            
            exec_result = {
                'transcript_data': processed_transcript,
                'transcript_stats': transcript_stats,
                'video_metadata': prep_result['video_metadata'],
                'processing_metadata': {
                    'enhanced_acquisition_used': self.enable_enhanced_acquisition,
                    'selected_language': selected_transcript.get('language_code'),
                    'available_languages': [t.get('language_code') for t in prep_result['available_transcripts']],
                    'processing_timestamp': datetime.utcnow().isoformat()
                },
                'execution_status': 'success'
            }
            
            self.logger.info(f"Transcript execution successful for video: {video_id}")
            return exec_result
            
        except Exception as e:
            error_info = self._handle_error(e, "Transcript execution failed")
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
                self.logger.error(f"Non-recoverable error in transcript execution: {str(e)}")
            
            return exec_result
    
    async def post(self, store: Store, prep_result: Dict[str, Any], exec_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Post-process transcript data and update store.
        
        Args:
            store: Data store
            prep_result: Results from preparation phase
            exec_result: Results from execution phase
            
        Returns:
            Dict containing final processing results
        """
        self.logger.info("Starting transcript post-processing")
        
        try:
            # Check if execution was successful
            if exec_result.get('execution_status') != 'success':
                raise ValueError("Cannot post-process transcript: execution failed")
            
            transcript_data = exec_result['transcript_data']
            video_metadata = exec_result['video_metadata']
            
            # Update store with transcript data
            store_updates = {
                'transcript_data': transcript_data,
                'video_metadata': video_metadata,
                'transcript_stats': exec_result['transcript_stats'],
                'processing_metadata': exec_result['processing_metadata']
            }
            
            self._safe_store_update(store, store_updates)
            
            # Calculate final processing metrics
            processing_time = self._measure_execution_time(
                prep_result.get('prep_timestamp', ''),
                exec_result.get('processing_timestamp', '')
            )
            
            post_result = {
                'processing_status': 'success',
                'video_id': prep_result.get('video_id'),
                'transcript_word_count': transcript_data.get('word_count', 0),
                'transcript_duration': transcript_data.get('duration', 0),
                'processing_time_seconds': processing_time,
                'store_updated': True,
                'post_timestamp': datetime.utcnow().isoformat(),
                'database_saved': False
            }
            
            # Save transcript to database if video service is available
            if self.video_service:
                try:
                    video_id = prep_result.get('video_id')
                    transcript_content = transcript_data.get('full_text', '')
                    language = transcript_data.get('language', 'en')
                    
                    # Create video record if it doesn't exist
                    await self.video_service.create_video_record(
                        video_id=video_id,
                        title=video_metadata.get('title', 'Unknown Title'),
                        duration=video_metadata.get('duration', 0),
                        url=f"https://www.youtube.com/watch?v={video_id}"
                    )
                    
                    # Save transcript
                    await self.video_service.save_transcript(
                        video_id=video_id,
                        content=transcript_content,
                        language=language
                    )
                    
                    post_result['database_saved'] = True
                    self.logger.info(f"Transcript saved to database for video: {video_id}")
                    
                except Exception as db_error:
                    self.logger.error(f"Failed to save transcript to database: {str(db_error)}")
                    post_result['database_error'] = str(db_error)
            
            self.logger.info(f"Transcript post-processing successful")
            return post_result
            
        except Exception as e:
            error_info = self._handle_error(e, "Transcript post-processing failed")
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
                self.logger.error(f"Non-recoverable error in transcript post-processing: {str(e)}")
            
            return post_result
    
    def _select_best_transcript(self, available_transcripts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Select the best transcript from available options."""
        if not available_transcripts:
            raise ValueError("No transcripts available for selection")
        
        # Priority order: English, manually created, auto-generated
        priority_languages = ['en', 'en-US', 'en-GB']
        
        # First, try to find manually created English transcripts
        for lang in priority_languages:
            for transcript in available_transcripts:
                if (transcript.get('language_code') == lang and 
                    not transcript.get('is_generated', True)):
                    return transcript
        
        # Next, try auto-generated English transcripts
        for lang in priority_languages:
            for transcript in available_transcripts:
                if transcript.get('language_code') == lang:
                    return transcript
        
        # Finally, return the first available transcript
        return available_transcripts[0]
    
    def _process_transcript_data(self, transcript_result: Dict[str, Any], video_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Process and enhance transcript data."""
        raw_transcript = transcript_result.get('transcript', [])
        
        # Extract text content
        transcript_text = ' '.join([entry.get('text', '') for entry in raw_transcript])
        
        # Calculate basic statistics
        word_count = len(transcript_text.split())
        duration = raw_transcript[-1].get('start', 0) + raw_transcript[-1].get('duration', 0) if raw_transcript else 0
        
        processed_data = {
            'video_id': video_metadata.get('video_id'),
            'transcript_text': transcript_text,
            'raw_transcript': raw_transcript,
            'word_count': word_count,
            'duration': duration,
            'language': transcript_result.get('language_code', 'unknown'),
            'is_generated': transcript_result.get('is_generated', True),
            'video_title': video_metadata.get('title', ''),
            'video_duration': video_metadata.get('duration', 0),
            'processing_timestamp': datetime.utcnow().isoformat()
        }
        
        return processed_data
    
    def _generate_transcript_stats(self, transcript_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate statistics for the transcript."""
        transcript_text = transcript_data.get('transcript_text', '')
        raw_transcript = transcript_data.get('raw_transcript', [])
        
        if not transcript_text:
            return {}
        
        words = transcript_text.split()
        sentences = [s.strip() for s in transcript_text.split('.') if s.strip()]
        
        return {
            'word_count': len(words),
            'sentence_count': len(sentences),
            'character_count': len(transcript_text),
            'segment_count': len(raw_transcript),
            'avg_words_per_sentence': len(words) / max(len(sentences), 1),
            'avg_segment_duration': sum(seg.get('duration', 0) for seg in raw_transcript) / max(len(raw_transcript), 1),
            'language': transcript_data.get('language', 'unknown'),
            'is_auto_generated': transcript_data.get('is_generated', True)
        }