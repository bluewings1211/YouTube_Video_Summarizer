"""
PocketFlow Node implementations for YouTube video summarization.

This module implements all the processing nodes required for the YouTube
video summarization workflow using PocketFlow's prep/exec/post pattern.
"""

import logging
import os
import re
import time
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

try:
    from pocketflow import Node, NodeState, Store
except ImportError:
    # Fallback base classes for development
    class Node(ABC):
        def __init__(self, name: str):
            self.name = name
            self.logger = logging.getLogger(f"{__name__}.{name}")
        
        @abstractmethod
        def prep(self, store: "Store") -> Dict[str, Any]:
            pass
        
        @abstractmethod
        def exec(self, store: "Store", prep_result: Dict[str, Any]) -> Dict[str, Any]:
            pass
        
        @abstractmethod
        def post(self, store: "Store", prep_result: Dict[str, Any], exec_result: Dict[str, Any]) -> Dict[str, Any]:
            pass
    
    class Store(dict):
        def __init__(self):
            super().__init__()
    
    class NodeState:
        PENDING = "pending"
        PREP = "prep"
        EXEC = "exec"
        POST = "post"
        SUCCESS = "success"
        FAILED = "failed"

from src.utils.youtube_api import YouTubeTranscriptFetcher, YouTubeTranscriptError
from src.utils.validators import YouTubeURLValidator
from src.utils.call_llm import create_llm_client, LLMError

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class NodeError:
    """Structured error information for nodes."""
    node_name: str
    error_type: str
    message: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    retry_count: int = 0
    is_recoverable: bool = True


class BaseProcessingNode(Node):
    """Base class for all processing nodes with common functionality."""
    
    def __init__(self, name: str, max_retries: int = 3, retry_delay: float = 1.0):
        super().__init__(name)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.logger = logging.getLogger(f"{__name__}.{name}")
    
    def _handle_error(self, error: Exception, context: str, retry_count: int = 0) -> NodeError:
        """Handle and log errors with structured information."""
        # Determine if error is recoverable based on error type
        is_recoverable = self._is_error_recoverable(error) and retry_count < self.max_retries
        
        error_info = NodeError(
            node_name=self.name,
            error_type=type(error).__name__,
            message=str(error),
            retry_count=retry_count,
            is_recoverable=is_recoverable
        )
        
        log_msg = f"{context}: {error_info.error_type} - {error_info.message}"
        if retry_count > 0:
            log_msg += f" (retry {retry_count}/{self.max_retries})"
        
        # Include error details for debugging
        if hasattr(error, 'error_code'):
            log_msg += f" [Code: {error.error_code}]"
        
        if is_recoverable:
            self.logger.warning(log_msg)
        else:
            self.logger.error(log_msg, exc_info=True)
        
        return error_info
    
    def _is_error_recoverable(self, error: Exception) -> bool:
        """Determine if an error is recoverable and worth retrying."""
        # Import here to avoid circular imports
        from .utils.youtube_api import (
            PrivateVideoError, LiveVideoError, NoTranscriptAvailableError, 
            VideoTooLongError, NetworkTimeoutError, RateLimitError
        )
        from .utils.call_llm import (
            LLMRateLimitError, LLMTimeoutError, LLMServerError, 
            LLMAuthenticationError, LLMQuotaError
        )
        
        # Non-recoverable errors
        non_recoverable_errors = (
            PrivateVideoError,
            LiveVideoError, 
            NoTranscriptAvailableError,
            VideoTooLongError,
            LLMAuthenticationError,
            LLMQuotaError,
            ValueError,  # Usually validation errors
            TypeError,   # Usually programming errors
        )
        
        if isinstance(error, non_recoverable_errors):
            return False
        
        # Recoverable errors (worth retrying)
        recoverable_errors = (
            NetworkTimeoutError,
            RateLimitError,
            LLMRateLimitError,
            LLMTimeoutError,
            LLMServerError,
            ConnectionError,
            TimeoutError,
        )
        
        if isinstance(error, recoverable_errors):
            return True
        
        # For generic exceptions, check the error message
        error_message = str(error).lower()
        if any(keyword in error_message for keyword in ['timeout', 'connection', 'network', 'rate limit']):
            return True
        
        # Default to non-recoverable for unknown errors
        return False
    
    def _retry_with_delay(self, retry_count: int) -> None:
        """Sleep with exponential backoff for retries."""
        if retry_count > 0:
            delay = self.retry_delay * (2 ** (retry_count - 1))
            self.logger.info(f"Retrying in {delay:.2f} seconds...")
            time.sleep(delay)
    
    def _validate_store_data(self, store: Store, required_keys: List[str]) -> Tuple[bool, List[str]]:
        """Validate that required data is present in the store."""
        missing_keys = []
        for key in required_keys:
            if key not in store:
                missing_keys.append(key)
        
        return len(missing_keys) == 0, missing_keys
    
    def _safe_store_update(self, store: Store, data: Dict[str, Any]) -> None:
        """Safely update store with new data."""
        try:
            store.update(data)
            self.logger.debug(f"Store updated with keys: {list(data.keys())}")
        except Exception as e:
            self.logger.error(f"Failed to update store: {str(e)}")
            raise
    
    def _execute_with_retry(self, operation, *args, **kwargs):
        """Execute an operation with retry logic and proper error handling."""
        last_error = None
        
        for attempt in range(self.max_retries + 1):
            try:
                self.logger.debug(f"Executing operation (attempt {attempt + 1}/{self.max_retries + 1})")
                return operation(*args, **kwargs)
                
            except Exception as e:
                last_error = e
                error_info = self._handle_error(e, f"Operation failed", attempt)
                
                # If not recoverable or max retries reached, raise the error
                if not error_info.is_recoverable or attempt >= self.max_retries:
                    self.logger.error(f"Operation failed permanently after {attempt + 1} attempts")
                    raise e
                
                # Wait before retrying
                self._retry_with_delay(attempt + 1)
        
        # This should never be reached, but just in case
        if last_error:
            raise last_error
        else:
            raise RuntimeError("Operation failed without specific error")
    
    def _validate_inputs(self, required_keys: List[str], store: Store) -> None:
        """Validate required inputs are present in store."""
        is_valid, missing_keys = self._validate_store_data(store, required_keys)
        if not is_valid:
            raise ValueError(f"Missing required inputs: {missing_keys}")
    
    def _handle_node_exception(self, phase: str, error: Exception) -> Dict[str, Any]:
        """Handle exceptions that occur during node execution phases."""
        error_info = self._handle_error(error, f"{phase} phase failed", 0)
        
        return {
            'status': 'failed',
            'error': {
                'type': error_info.error_type,
                'message': error_info.message,
                'node': self.name,
                'phase': phase,
                'timestamp': error_info.timestamp,
                'is_recoverable': error_info.is_recoverable
            }
        }


class YouTubeTranscriptNode(BaseProcessingNode):
    """
    Node for fetching YouTube video transcripts with metadata.
    
    This node handles the extraction of video transcripts from YouTube URLs,
    including video validation, metadata extraction, and error handling.
    """
    
    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0):
        super().__init__("YouTubeTranscriptNode", max_retries, retry_delay)
        self.transcript_fetcher = YouTubeTranscriptFetcher()
    
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
            
            # Execute URL validation with retry logic
            video_id = self._execute_with_retry(validate_url)
            
            # Check video support status with retry logic
            def check_support():
                return self.transcript_fetcher.check_video_support(video_id)
            
            support_status = self._execute_with_retry(check_support)
            
            # Check transcript availability with retry logic
            def check_transcripts():
                return self.transcript_fetcher.check_transcript_availability(video_id)
            
            transcript_availability = self._execute_with_retry(check_transcripts)
            
            prep_result = {
                'video_id': video_id,
                'youtube_url': youtube_url,
                'is_supported': support_status['is_supported'],
                'support_issues': support_status.get('issues', []),
                'has_transcripts': transcript_availability['has_transcripts'],
                'available_transcripts': transcript_availability['available_transcripts'],
                'prep_timestamp': datetime.utcnow().isoformat(),
                'prep_status': 'success'
            }
            
            if not support_status['is_supported']:
                issues = [issue['message'] for issue in support_status['issues']]
                raise ValueError(f"Video not supported: {'; '.join(issues)}")
            
            if not transcript_availability['has_transcripts']:
                raise ValueError("No transcripts available for this video")
            
            self.logger.info(f"Prep successful for video {video_id}")
            return prep_result
            
        except Exception as e:
            return self._handle_node_exception("prep", e)
    
    def exec(self, store: Store, prep_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute transcript fetching with retry logic.
        
        Args:
            store: Data store
            prep_result: Results from prep phase
            
        Returns:
            Dict containing transcript data and metadata
        """
        self.logger.info("Starting transcript execution")
        
        if prep_result.get('prep_status') != 'success':
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
                
                # Fetch transcript with metadata
                transcript_data = self.transcript_fetcher.fetch_transcript(
                    video_id=video_id,
                    languages=['en'],
                    include_metadata=True,
                    check_unsupported=False,
                    max_duration_seconds=1800
                )
                
                exec_result = {
                    'exec_status': 'success',
                    'video_id': video_id,
                    'transcript_text': transcript_data['transcript'],
                    'raw_transcript': transcript_data['raw_transcript'],
                    'transcript_language': transcript_data['language'],
                    'transcript_duration': transcript_data['duration_seconds'],
                    'transcript_word_count': transcript_data['word_count'],
                    'video_metadata': transcript_data.get('video_metadata', {}),
                    'exec_timestamp': datetime.utcnow().isoformat(),
                    'retry_count': retry_count
                }
                
                self.logger.info(f"Transcript execution successful for video {video_id}")
                return exec_result
                
            except Exception as e:
                last_error = self._handle_error(e, f"Transcript execution failed", retry_count)
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
        Post-process transcript data and update store.
        
        Args:
            store: Data store
            prep_result: Results from prep phase
            exec_result: Results from exec phase
            
        Returns:
            Dict containing final processing results
        """
        self.logger.info("Starting transcript post-processing")
        
        try:
            if exec_result.get('exec_status') != 'success':
                return {
                    'post_status': 'failed',
                    'error': 'Execution phase failed',
                    'post_timestamp': datetime.utcnow().isoformat()
                }
            
            # Extract key information
            video_id = exec_result['video_id']
            transcript_text = exec_result['transcript_text']
            video_metadata = exec_result['video_metadata']
            
            # Calculate additional metrics
            transcript_stats = self._calculate_transcript_stats(transcript_text)
            
            # Prepare store data
            store_data = {
                'transcript_data': {
                    'video_id': video_id,
                    'transcript_text': transcript_text,
                    'raw_transcript': exec_result['raw_transcript'],
                    'language': exec_result['transcript_language'],
                    'duration_seconds': exec_result['transcript_duration'],
                    'word_count': exec_result['transcript_word_count'],
                    'stats': transcript_stats
                },
                'video_metadata': video_metadata,
                'processing_metadata': {
                    'transcript_fetched_at': exec_result['exec_timestamp'],
                    'prep_duration': self._calculate_duration(
                        prep_result.get('prep_timestamp', ''),
                        exec_result.get('exec_timestamp', '')
                    ),
                    'retry_count': exec_result.get('retry_count', 0)
                }
            }
            
            # Update store
            self._safe_store_update(store, store_data)
            
            post_result = {
                'post_status': 'success',
                'video_id': video_id,
                'transcript_ready': True,
                'transcript_length': len(transcript_text),
                'video_title': video_metadata.get('title', 'Unknown'),
                'video_duration': video_metadata.get('duration_seconds', 0),
                'post_timestamp': datetime.utcnow().isoformat()
            }
            
            self.logger.info(f"Transcript post-processing successful for video {video_id}")
            return post_result
            
        except Exception as e:
            error_info = self._handle_error(e, "Transcript post-processing failed")
            return {
                'post_status': 'failed',
                'error': error_info.__dict__,
                'post_timestamp': datetime.utcnow().isoformat()
            }
    
    def _calculate_transcript_stats(self, transcript_text: str) -> Dict[str, Any]:
        """Calculate additional statistics for the transcript."""
        if not transcript_text:
            return {'char_count': 0, 'word_count': 0, 'sentence_count': 0}
        
        words = transcript_text.split()
        sentences = re.split(r'[.!?]+', transcript_text)
        
        return {
            'char_count': len(transcript_text),
            'word_count': len(words),
            'sentence_count': len([s for s in sentences if s.strip()]),
            'avg_words_per_sentence': len(words) / max(len(sentences), 1)
        }
    
    def _calculate_duration(self, start_time: str, end_time: str) -> float:
        """Calculate duration between two ISO timestamps."""
        try:
            if not start_time or not end_time:
                return 0.0
            
            start = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
            end = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
            
            return (end - start).total_seconds()
        except Exception:
            return 0.0


class SummarizationNode(BaseProcessingNode):
    """
    Node for generating AI-powered summaries of video transcripts.
    
    This node takes transcript data and generates a professional summary
    of approximately 500 words using configured LLM providers.
    """
    
    def __init__(self, max_retries: int = 3, retry_delay: float = 2.0):
        super().__init__("SummarizationNode", max_retries, retry_delay)
        self.llm_client = None
        self.target_word_count = 500
    
    def prep(self, store: Store) -> Dict[str, Any]:
        """
        Prepare for text summarization by validating input and setting up LLM client.
        
        Args:
            store: Data store containing transcript data
            
        Returns:
            Dict containing prep results and LLM configuration
        """
        self.logger.info("Starting summarization preparation")
        
        try:
            # Validate required input
            is_valid, missing_keys = self._validate_store_data(store, ['transcript_data'])
            if not is_valid:
                raise ValueError(f"Missing required data: {missing_keys}")
            
            transcript_data = store['transcript_data']
            transcript_text = transcript_data.get('transcript_text', '')
            
            if not transcript_text or not transcript_text.strip():
                raise ValueError("No transcript text available for summarization")
            
            # Check transcript length
            word_count = len(transcript_text.split())
            if word_count < 50:
                raise ValueError(f"Transcript too short for summarization ({word_count} words)")
            
            # Initialize LLM client
            try:
                # Try OpenAI first, fall back to environment detection
                provider = os.getenv('LLM_PROVIDER', 'openai')
                model = os.getenv('LLM_MODEL', 'gpt-3.5-turbo')
                
                self.llm_client = create_llm_client(
                    provider=provider,
                    model=model,
                    max_tokens=800,  # Allow extra tokens for 500-word summary
                    temperature=0.3  # Lower temperature for consistent summaries
                )
                
            except Exception as e:
                raise ValueError(f"Failed to initialize LLM client: {str(e)}")
            
            prep_result = {
                'transcript_text': transcript_text,
                'original_word_count': word_count,
                'target_word_count': self.target_word_count,
                'video_id': transcript_data.get('video_id', 'unknown'),
                'video_title': store.get('video_metadata', {}).get('title', 'Unknown'),
                'llm_provider': provider,
                'llm_model': model,
                'prep_timestamp': datetime.utcnow().isoformat(),
                'prep_status': 'success'
            }
            
            self.logger.info(f"Summarization prep successful for {word_count} words")
            return prep_result
            
        except Exception as e:
            error_info = self._handle_error(e, "Summarization preparation failed")
            return {
                'prep_status': 'failed',
                'error': error_info.__dict__,
                'prep_timestamp': datetime.utcnow().isoformat()
            }
    
    def exec(self, store: Store, prep_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute text summarization with retry logic.
        
        Args:
            store: Data store
            prep_result: Results from prep phase
            
        Returns:
            Dict containing generated summary and metadata
        """
        self.logger.info("Starting summarization execution")
        
        if prep_result.get('prep_status') != 'success':
            return {
                'exec_status': 'failed',
                'error': 'Prep phase failed',
                'exec_timestamp': datetime.utcnow().isoformat()
            }
        
        transcript_text = prep_result['transcript_text']
        video_title = prep_result['video_title']
        target_word_count = prep_result['target_word_count']
        last_error = None
        
        for retry_count in range(self.max_retries + 1):
            try:
                self._retry_with_delay(retry_count)
                
                # Generate summary with context
                summary_result = self.llm_client.summarize_text(
                    text=transcript_text,
                    target_length=target_word_count,
                    style="professional"
                )
                
                # Validate summary quality
                summary_text = summary_result['summary']
                summary_word_count = summary_result['word_count']
                
                if not summary_text or summary_word_count < 50:
                    raise ValueError(f"Generated summary too short ({summary_word_count} words)")
                
                # Calculate summary statistics
                summary_stats = self._calculate_summary_stats(
                    summary_text, 
                    transcript_text,
                    target_word_count
                )
                
                exec_result = {
                    'exec_status': 'success',
                    'summary_text': summary_text,
                    'summary_word_count': summary_word_count,
                    'target_word_count': target_word_count,
                    'original_word_count': prep_result['original_word_count'],
                    'compression_ratio': summary_result['compression_ratio'],
                    'summary_stats': summary_stats,
                    'video_id': prep_result['video_id'],
                    'video_title': video_title,
                    'llm_metadata': summary_result['generation_metadata'],
                    'exec_timestamp': datetime.utcnow().isoformat(),
                    'retry_count': retry_count
                }
                
                self.logger.info(f"Summarization successful: {summary_word_count} words")
                return exec_result
                
            except Exception as e:
                last_error = self._handle_error(e, f"Summarization execution failed", retry_count)
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
        Post-process summary and update store.
        
        Args:
            store: Data store
            prep_result: Results from prep phase
            exec_result: Results from exec phase
            
        Returns:
            Dict containing final processing results
        """
        self.logger.info("Starting summarization post-processing")
        
        try:
            if exec_result.get('exec_status') != 'success':
                return {
                    'post_status': 'failed',
                    'error': 'Execution phase failed',
                    'post_timestamp': datetime.utcnow().isoformat()
                }
            
            # Extract summary information
            summary_text = exec_result['summary_text']
            summary_word_count = exec_result['summary_word_count']
            video_id = exec_result['video_id']
            video_title = exec_result['video_title']
            
            # Prepare store data
            store_data = {
                'summary_data': {
                    'summary_text': summary_text,
                    'word_count': summary_word_count,
                    'target_word_count': exec_result['target_word_count'],
                    'compression_ratio': exec_result['compression_ratio'],
                    'stats': exec_result['summary_stats'],
                    'generated_at': exec_result['exec_timestamp']
                },
                'summary_metadata': {
                    'video_id': video_id,
                    'video_title': video_title,
                    'original_word_count': exec_result['original_word_count'],
                    'processing_duration': self._calculate_duration(
                        prep_result.get('prep_timestamp', ''),
                        exec_result.get('exec_timestamp', '')
                    ),
                    'retry_count': exec_result.get('retry_count', 0),
                    'llm_provider': prep_result.get('llm_provider', 'unknown'),
                    'llm_model': prep_result.get('llm_model', 'unknown')
                }
            }
            
            # Update store
            self._safe_store_update(store, store_data)
            
            post_result = {
                'post_status': 'success',
                'summary_ready': True,
                'summary_length': summary_word_count,
                'video_id': video_id,
                'video_title': video_title,
                'compression_ratio': exec_result['compression_ratio'],
                'post_timestamp': datetime.utcnow().isoformat()
            }
            
            self.logger.info(f"Summary post-processing successful for video {video_id}")
            return post_result
            
        except Exception as e:
            error_info = self._handle_error(e, "Summary post-processing failed")
            return {
                'post_status': 'failed',
                'error': error_info.__dict__,
                'post_timestamp': datetime.utcnow().isoformat()
            }
    
    def _calculate_summary_stats(
        self, 
        summary_text: str, 
        original_text: str, 
        target_word_count: int
    ) -> Dict[str, Any]:
        """Calculate statistics for the generated summary."""
        if not summary_text:
            return {}
        
        summary_words = summary_text.split()
        original_words = original_text.split()
        
        # Calculate readability metrics
        sentences = re.split(r'[.!?]+', summary_text)
        sentences = [s for s in sentences if s.strip()]
        
        avg_sentence_length = len(summary_words) / max(len(sentences), 1)
        
        # Calculate target accuracy
        target_accuracy = min(100, (target_word_count / max(len(summary_words), 1)) * 100)
        
        return {
            'char_count': len(summary_text),
            'word_count': len(summary_words),
            'sentence_count': len(sentences),
            'avg_sentence_length': round(avg_sentence_length, 2),
            'target_accuracy': round(target_accuracy, 2),
            'compression_ratio': len(original_words) / max(len(summary_words), 1),
            'estimated_reading_time_minutes': round(len(summary_words) / 200, 1)  # ~200 WPM
        }


class TimestampNode(BaseProcessingNode):
    """
    Node for generating timestamped URLs with descriptions and importance ratings.
    
    This node analyzes video transcripts to identify key moments and creates
    timestamped YouTube URLs with AI-generated descriptions and importance ratings.
    """
    
    def __init__(self, max_retries: int = 3, retry_delay: float = 2.0):
        super().__init__("TimestampNode", max_retries, retry_delay)
        self.llm_client = None
        self.default_timestamp_count = 5
    
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
            video_id = transcript_data.get('video_id', '')
            
            if not raw_transcript:
                raise ValueError("No raw transcript data available for timestamp generation")
            
            if not video_id:
                raise ValueError("Video ID is required for timestamp generation")
            
            # Validate video metadata
            video_title = video_metadata.get('title', 'Unknown')
            video_duration = video_metadata.get('duration_seconds', 0)
            
            if video_duration <= 0:
                self.logger.warning("Video duration not available, proceeding anyway")
            
            # Initialize LLM client
            try:
                provider = os.getenv('LLM_PROVIDER', 'openai')
                model = os.getenv('LLM_MODEL', 'gpt-3.5-turbo')
                
                self.llm_client = create_llm_client(
                    provider=provider,
                    model=model,
                    max_tokens=600,  # Sufficient for timestamp descriptions
                    temperature=0.4  # Balanced creativity for descriptions
                )
                
            except Exception as e:
                raise ValueError(f"Failed to initialize LLM client: {str(e)}")
            
            # Filter and validate transcript entries
            valid_transcript = self._filter_transcript_entries(raw_transcript)
            
            if len(valid_transcript) < 3:
                raise ValueError("Insufficient transcript data for meaningful timestamp generation")
            
            prep_result = {
                'video_id': video_id,
                'video_title': video_title,
                'video_duration': video_duration,
                'raw_transcript': valid_transcript,
                'transcript_count': len(valid_transcript),
                'timestamp_count': self.default_timestamp_count,
                'llm_provider': provider,
                'llm_model': model,
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
                
                # Generate timestamps using LLM
                timestamps_result = self.llm_client.generate_timestamps(
                    transcript_data=raw_transcript,
                    video_id=video_id,
                    count=timestamp_count
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
                    'llm_provider': prep_result.get('llm_provider', 'unknown'),
                    'llm_model': prep_result.get('llm_model', 'unknown')
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
                'post_timestamp': datetime.utcnow().isoformat()
            }
            
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


class KeywordExtractionNode(BaseProcessingNode):
    """
    Node for extracting relevant keywords from video content.
    
    This node analyzes video transcripts and summaries to extract 5-8 relevant
    keywords that best represent the video content for categorization and search.
    """
    
    def __init__(self, max_retries: int = 3, retry_delay: float = 1.5):
        super().__init__("KeywordExtractionNode", max_retries, retry_delay)
        self.llm_client = None
        self.default_keyword_count = 6  # Middle of 5-8 range
    
    def prep(self, store: Store) -> Dict[str, Any]:
        """
        Prepare for keyword extraction by validating input data.
        
        Args:
            store: Data store containing transcript and summary data
            
        Returns:
            Dict containing prep results and configuration
        """
        self.logger.info("Starting keyword extraction preparation")
        
        try:
            # Validate required input - we need at least transcript data
            required_keys = ['transcript_data']
            is_valid, missing_keys = self._validate_store_data(store, required_keys)
            if not is_valid:
                raise ValueError(f"Missing required data: {missing_keys}")
            
            transcript_data = store['transcript_data']
            transcript_text = transcript_data.get('transcript_text', '')
            
            if not transcript_text or not transcript_text.strip():
                raise ValueError("No transcript text available for keyword extraction")
            
            # Get summary data if available (preferred for keyword extraction)
            summary_data = store.get('summary_data', {})
            summary_text = summary_data.get('summary_text', '')
            
            # Use summary if available, otherwise use transcript
            extraction_text = summary_text if summary_text else transcript_text
            
            # Validate text length
            word_count = len(extraction_text.split())
            if word_count < 20:
                raise ValueError(f"Text too short for keyword extraction ({word_count} words)")
            
            # Get video metadata
            video_metadata = store.get('video_metadata', {})
            video_id = transcript_data.get('video_id', 'unknown')
            video_title = video_metadata.get('title', 'Unknown')
            
            # Initialize LLM client
            try:
                provider = os.getenv('LLM_PROVIDER', 'openai')
                model = os.getenv('LLM_MODEL', 'gpt-3.5-turbo')
                
                self.llm_client = create_llm_client(
                    provider=provider,
                    model=model,
                    max_tokens=300,  # Sufficient for keyword extraction
                    temperature=0.2  # Low temperature for consistent results
                )
                
            except Exception as e:
                raise ValueError(f"Failed to initialize LLM client: {str(e)}")
            
            prep_result = {
                'video_id': video_id,
                'video_title': video_title,
                'extraction_text': extraction_text,
                'text_source': 'summary' if summary_text else 'transcript',
                'text_word_count': word_count,
                'keyword_count': self.default_keyword_count,
                'has_summary': bool(summary_text),
                'llm_provider': provider,
                'llm_model': model,
                'prep_timestamp': datetime.utcnow().isoformat(),
                'prep_status': 'success'
            }
            
            self.logger.info(f"Keyword extraction prep successful for video {video_id}")
            return prep_result
            
        except Exception as e:
            error_info = self._handle_error(e, "Keyword extraction preparation failed")
            return {
                'prep_status': 'failed',
                'error': error_info.__dict__,
                'prep_timestamp': datetime.utcnow().isoformat()
            }
    
    def exec(self, store: Store, prep_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute keyword extraction with AI analysis.
        
        Args:
            store: Data store
            prep_result: Results from prep phase
            
        Returns:
            Dict containing extracted keywords and metadata
        """
        self.logger.info("Starting keyword extraction execution")
        
        if prep_result.get('prep_status') != 'success':
            return {
                'exec_status': 'failed',
                'error': 'Prep phase failed',
                'exec_timestamp': datetime.utcnow().isoformat()
            }
        
        extraction_text = prep_result['extraction_text']
        video_id = prep_result['video_id']
        video_title = prep_result['video_title']
        keyword_count = prep_result['keyword_count']
        last_error = None
        
        for retry_count in range(self.max_retries + 1):
            try:
                self._retry_with_delay(retry_count)
                
                # Extract keywords using LLM
                keywords_result = self.llm_client.extract_keywords(
                    text=extraction_text,
                    count=keyword_count,
                    include_phrases=True
                )
                
                # Validate extracted keywords
                keywords = keywords_result['keywords']
                
                if not keywords:
                    raise ValueError("No keywords extracted")
                
                # Ensure we have the right number of keywords (5-8)
                if len(keywords) < 5:
                    # Try to extract more if we have too few
                    additional_result = self.llm_client.extract_keywords(
                        text=extraction_text,
                        count=8,
                        include_phrases=True
                    )
                    keywords = additional_result['keywords'][:8]
                
                # Limit to maximum of 8 keywords
                keywords = keywords[:8]
                
                # Enhance keywords with metadata
                enhanced_keywords = self._enhance_keywords(
                    keywords, 
                    video_title, 
                    prep_result['text_source']
                )
                
                # Calculate keyword statistics
                keyword_stats = self._calculate_keyword_stats(
                    enhanced_keywords,
                    extraction_text
                )
                
                exec_result = {
                    'exec_status': 'success',
                    'keywords': enhanced_keywords,
                    'keyword_count': len(enhanced_keywords),
                    'requested_count': keyword_count,
                    'video_id': video_id,
                    'video_title': video_title,
                    'text_source': prep_result['text_source'],
                    'keyword_stats': keyword_stats,
                    'llm_metadata': keywords_result['generation_metadata'],
                    'exec_timestamp': datetime.utcnow().isoformat(),
                    'retry_count': retry_count
                }
                
                self.logger.info(f"Keyword extraction successful: {len(enhanced_keywords)} keywords")
                return exec_result
                
            except Exception as e:
                last_error = self._handle_error(e, f"Keyword extraction execution failed", retry_count)
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
        Post-process keywords and update store.
        
        Args:
            store: Data store
            prep_result: Results from prep phase
            exec_result: Results from exec phase
            
        Returns:
            Dict containing final processing results
        """
        self.logger.info("Starting keyword extraction post-processing")
        
        try:
            if exec_result.get('exec_status') != 'success':
                return {
                    'post_status': 'failed',
                    'error': 'Execution phase failed',
                    'post_timestamp': datetime.utcnow().isoformat()
                }
            
            # Extract keyword information
            keywords = exec_result['keywords']
            video_id = exec_result['video_id']
            video_title = exec_result['video_title']
            
            # Categorize keywords
            single_words = [kw for kw in keywords if len(kw['keyword'].split()) == 1]
            phrases = [kw for kw in keywords if len(kw['keyword'].split()) > 1]
            
            # Create searchable keyword string
            keyword_string = ', '.join([kw['keyword'] for kw in keywords])
            
            # Prepare store data
            store_data = {
                'keyword_data': {
                    'keywords': keywords,
                    'single_words': single_words,
                    'phrases': phrases,
                    'keyword_string': keyword_string,
                    'count': len(keywords),
                    'stats': exec_result['keyword_stats'],
                    'text_source': exec_result['text_source'],
                    'generated_at': exec_result['exec_timestamp']
                },
                'keyword_metadata': {
                    'video_id': video_id,
                    'video_title': video_title,
                    'requested_count': exec_result['requested_count'],
                    'actual_count': len(keywords),
                    'single_word_count': len(single_words),
                    'phrase_count': len(phrases),
                    'processing_duration': self._calculate_duration(
                        prep_result.get('prep_timestamp', ''),
                        exec_result.get('exec_timestamp', '')
                    ),
                    'retry_count': exec_result.get('retry_count', 0),
                    'llm_provider': prep_result.get('llm_provider', 'unknown'),
                    'llm_model': prep_result.get('llm_model', 'unknown')
                }
            }
            
            # Update store
            self._safe_store_update(store, store_data)
            
            post_result = {
                'post_status': 'success',
                'keywords_ready': True,
                'keyword_count': len(keywords),
                'single_word_count': len(single_words),
                'phrase_count': len(phrases),
                'keyword_string': keyword_string,
                'video_id': video_id,
                'video_title': video_title,
                'post_timestamp': datetime.utcnow().isoformat()
            }
            
            self.logger.info(f"Keyword extraction post-processing successful for video {video_id}")
            return post_result
            
        except Exception as e:
            error_info = self._handle_error(e, "Keyword extraction post-processing failed")
            return {
                'post_status': 'failed',
                'error': error_info.__dict__,
                'post_timestamp': datetime.utcnow().isoformat()
            }
    
    def _enhance_keywords(
        self, 
        keywords: List[str], 
        video_title: str, 
        text_source: str
    ) -> List[Dict[str, Any]]:
        """Enhance keywords with additional metadata."""
        enhanced = []
        
        for i, keyword in enumerate(keywords):
            keyword_clean = keyword.strip()
            word_count = len(keyword_clean.split())
            
            # Determine keyword type
            if word_count == 1:
                keyword_type = 'single_word'
            elif word_count == 2:
                keyword_type = 'phrase'
            else:
                keyword_type = 'long_phrase'
            
            # Calculate relevance score based on position (earlier = more relevant)
            relevance_score = max(1, 10 - i)  # Score from 10 (first) to 1 (last)
            
            enhanced_keyword = {
                'keyword': keyword_clean,
                'position': i + 1,
                'type': keyword_type,
                'word_count': word_count,
                'relevance_score': relevance_score,
                'text_source': text_source,
                'video_title': video_title,
                'length': len(keyword_clean),
                'is_phrase': word_count > 1
            }
            
            enhanced.append(enhanced_keyword)
        
        return enhanced
    
    def _calculate_keyword_stats(
        self, 
        keywords: List[Dict[str, Any]], 
        source_text: str
    ) -> Dict[str, Any]:
        """Calculate statistics for extracted keywords."""
        if not keywords:
            return {}
        
        # Basic counts
        total_keywords = len(keywords)
        single_words = sum(1 for kw in keywords if kw['type'] == 'single_word')
        phrases = sum(1 for kw in keywords if kw['type'] == 'phrase')
        long_phrases = sum(1 for kw in keywords if kw['type'] == 'long_phrase')
        
        # Length statistics
        keyword_lengths = [len(kw['keyword']) for kw in keywords]
        avg_length = sum(keyword_lengths) / total_keywords
        
        # Relevance statistics
        relevance_scores = [kw['relevance_score'] for kw in keywords]
        avg_relevance = sum(relevance_scores) / total_keywords
        
        # Coverage analysis (how many keywords appear in source text)
        coverage_count = 0
        source_lower = source_text.lower()
        for kw in keywords:
            if kw['keyword'].lower() in source_lower:
                coverage_count += 1
        
        coverage_percent = (coverage_count / total_keywords) * 100
        
        return {
            'total_count': total_keywords,
            'single_word_count': single_words,
            'phrase_count': phrases,
            'long_phrase_count': long_phrases,
            'avg_length': round(avg_length, 2),
            'min_length': min(keyword_lengths),
            'max_length': max(keyword_lengths),
            'avg_relevance_score': round(avg_relevance, 2),
            'coverage_percent': round(coverage_percent, 2),
            'covered_keywords': coverage_count,
            'phrase_ratio': round((phrases + long_phrases) / total_keywords * 100, 2)
        }
