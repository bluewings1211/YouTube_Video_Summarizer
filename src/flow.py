"""
PocketFlow workflow orchestration for YouTube video summarization.

This module implements the main workflow that orchestrates all processing nodes
to transform a YouTube URL into a comprehensive video summary with timestamps
and keywords.
"""

import logging
import time
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

try:
    from pocketflow import Flow, Node, Store, FlowState
except ImportError:
    # Fallback base classes for development
    class Flow(ABC):
        def __init__(self, name: str):
            self.name = name
            self.logger = logging.getLogger(f"{__name__}.{name}")
            self.nodes: List[Node] = []
            self.store = Store()
        
        @abstractmethod
        def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
            pass
    
    class Store(dict):
        def __init__(self):
            super().__init__()
    
    class FlowState:
        PENDING = "pending"
        RUNNING = "running"
        SUCCESS = "success"
        FAILED = "failed"
        PARTIAL = "partial"

from .nodes import (
    YouTubeTranscriptNode,
    SummarizationNode,
    TimestampNode,
    KeywordExtractionNode,
    NodeError
)

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class WorkflowError:
    """Structured error information for workflow execution."""
    flow_name: str
    error_type: str
    message: str
    failed_node: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    is_recoverable: bool = True
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowMetrics:
    """Metrics tracking for workflow execution."""
    start_time: float
    end_time: Optional[float] = None
    total_duration: float = 0.0
    node_durations: Dict[str, float] = field(default_factory=dict)
    node_retry_counts: Dict[str, int] = field(default_factory=dict)
    success_rate: float = 0.0
    memory_usage: Dict[str, Any] = field(default_factory=dict)


class YouTubeSummarizerFlow(Flow):
    """
    Main workflow for YouTube video summarization.
    
    This flow orchestrates all processing nodes to transform a YouTube URL
    into a comprehensive summary with timestamps and keywords.
    """
    
    def __init__(self, 
                 enable_monitoring: bool = True,
                 enable_fallbacks: bool = True,
                 max_retries: int = 2,
                 timeout_seconds: int = 300):
        """
        Initialize the YouTube summarizer workflow.
        
        Args:
            enable_monitoring: Whether to enable performance monitoring
            enable_fallbacks: Whether to enable fallback mechanisms
            max_retries: Maximum number of workflow retries
            timeout_seconds: Total workflow timeout in seconds
        """
        super().__init__("YouTubeSummarizerFlow")
        
        self.enable_monitoring = enable_monitoring
        self.enable_fallbacks = enable_fallbacks
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds
        
        # Initialize metrics
        self.metrics = None
        self.workflow_errors: List[WorkflowError] = []
        
        # Initialize nodes
        self._initialize_nodes()
        
        # Setup monitoring
        if self.enable_monitoring:
            self._setup_monitoring()
        
        logger.info(f"YouTubeSummarizerFlow initialized with {len(self.nodes)} nodes")
    
    def _initialize_nodes(self) -> None:
        """Initialize all processing nodes in the correct order."""
        try:
            # Node 1: Fetch YouTube transcript
            self.transcript_node = YouTubeTranscriptNode(
                max_retries=3,
                retry_delay=1.0
            )
            
            # Node 2: Generate summary
            self.summary_node = SummarizationNode(
                max_retries=3,
                retry_delay=2.0
            )
            
            # Node 3: Create timestamps
            self.timestamp_node = TimestampNode(
                max_retries=3,
                retry_delay=2.0
            )
            
            # Node 4: Extract keywords
            self.keyword_node = KeywordExtractionNode(
                max_retries=3,
                retry_delay=1.5
            )
            
            # Store nodes in execution order
            self.nodes = [
                self.transcript_node,
                self.summary_node,
                self.timestamp_node,
                self.keyword_node
            ]
            
            logger.info("All workflow nodes initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize workflow nodes: {str(e)}")
            raise
    
    def _setup_monitoring(self) -> None:
        """Setup performance monitoring for the workflow."""
        try:
            # Initialize metrics tracking
            self.metrics = WorkflowMetrics(start_time=time.time())
            
            # Setup memory monitoring if available
            try:
                import psutil
                import os
                self.process = psutil.Process(os.getpid())
                self.memory_monitoring_available = True
            except ImportError:
                self.memory_monitoring_available = False
                logger.warning("psutil not available, memory monitoring disabled")
            
            logger.debug("Performance monitoring setup complete")
            
        except Exception as e:
            logger.warning(f"Failed to setup monitoring: {str(e)}")
            self.enable_monitoring = False
    
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the complete workflow for YouTube video summarization.
        
        Args:
            input_data: Dictionary containing 'youtube_url' and optional configuration
            
        Returns:
            Dictionary containing all processing results or error information
        """
        logger.info(f"Starting YouTube summarizer workflow")
        
        # Initialize workflow state
        if self.enable_monitoring:
            self.metrics = WorkflowMetrics(start_time=time.time())
        
        workflow_start_time = time.time()
        
        try:
            # Validate input
            self._validate_input(input_data)
            
            # Initialize store with input data
            self.store = Store()
            self.store.update(input_data)
            
            # Execute workflow with timeout
            result = self._execute_workflow_with_timeout()
            
            # Calculate final metrics
            if self.enable_monitoring:
                self._finalize_metrics(success=True)
            
            # Prepare final result
            final_result = self._prepare_final_result(result)
            
            logger.info(f"Workflow completed successfully in {time.time() - workflow_start_time:.2f}s")
            return final_result
            
        except Exception as e:
            error_info = self._handle_workflow_error(e, "Workflow execution failed")
            
            if self.enable_monitoring:
                self._finalize_metrics(success=False)
            
            # Try fallback if enabled
            if self.enable_fallbacks and error_info.is_recoverable:
                return self._execute_fallback_workflow(input_data, error_info)
            
            # Return error result
            return self._prepare_error_result(error_info)
    
    def _validate_input(self, input_data: Dict[str, Any]) -> None:
        """Validate input data for the workflow."""
        if not isinstance(input_data, dict):
            raise ValueError("Input data must be a dictionary")
        
        if 'youtube_url' not in input_data:
            raise ValueError("Missing required 'youtube_url' in input data")
        
        youtube_url = input_data['youtube_url']
        if not isinstance(youtube_url, str) or not youtube_url.strip():
            raise ValueError("youtube_url must be a non-empty string")
        
        logger.debug("Input validation successful")
    
    def _execute_workflow_with_timeout(self) -> Dict[str, Any]:
        """Execute the workflow with timeout protection."""
        start_time = time.time()
        
        for retry_count in range(self.max_retries + 1):
            try:
                # Check timeout
                if time.time() - start_time > self.timeout_seconds:
                    raise TimeoutError(f"Workflow exceeded timeout of {self.timeout_seconds}s")
                
                # Execute all nodes in sequence
                return self._execute_nodes_sequence()
                
            except Exception as e:
                if retry_count >= self.max_retries:
                    raise
                
                logger.warning(f"Workflow retry {retry_count + 1}/{self.max_retries} due to: {str(e)}")
                time.sleep(2 ** retry_count)  # Exponential backoff
        
        raise RuntimeError("Workflow failed after all retries")
    
    def _execute_nodes_sequence(self) -> Dict[str, Any]:
        """Execute all nodes in the correct sequence."""
        results = {}
        
        for node in self.nodes:
            node_start_time = time.time()
            
            try:
                logger.info(f"Executing node: {node.name}")
                
                # Execute node phases
                prep_result = node.prep(self.store)
                exec_result = node.exec(self.store, prep_result)
                post_result = node.post(self.store, prep_result, exec_result)
                
                # Store node results
                results[node.name] = {
                    'prep': prep_result,
                    'exec': exec_result,
                    'post': post_result
                }
                
                # Track metrics
                if self.enable_monitoring:
                    node_duration = time.time() - node_start_time
                    self.metrics.node_durations[node.name] = node_duration
                    
                    # Track retry count
                    retry_count = exec_result.get('retry_count', 0)
                    self.metrics.node_retry_counts[node.name] = retry_count
                
                # Check for node failures
                if post_result.get('post_status') != 'success':
                    error_msg = post_result.get('error', 'Unknown node error')
                    raise RuntimeError(f"Node {node.name} failed: {error_msg}")
                
                logger.info(f"Node {node.name} completed successfully")
                
            except Exception as e:
                error_msg = f"Node {node.name} execution failed: {str(e)}"
                logger.error(error_msg)
                
                # Store partial results
                results[node.name] = {
                    'error': str(e),
                    'status': 'failed',
                    'timestamp': datetime.utcnow().isoformat()
                }
                
                raise RuntimeError(error_msg)
        
        return results
    
    def _handle_workflow_error(self, error: Exception, context: str) -> WorkflowError:
        """Handle and log workflow errors."""
        error_info = WorkflowError(
            flow_name=self.name,
            error_type=type(error).__name__,
            message=str(error),
            is_recoverable=isinstance(error, (ConnectionError, TimeoutError)),
            context={'workflow_context': context}
        )
        
        self.workflow_errors.append(error_info)
        
        log_msg = f"{context}: {error_info.error_type} - {error_info.message}"
        
        if error_info.is_recoverable:
            logger.warning(log_msg)
        else:
            logger.error(log_msg)
        
        return error_info
    
    def _execute_fallback_workflow(self, input_data: Dict[str, Any], error_info: WorkflowError) -> Dict[str, Any]:
        """Execute fallback workflow with reduced functionality."""
        logger.info("Executing fallback workflow")
        
        try:
            # Simplified workflow - just try to get transcript and summary
            fallback_store = Store()
            fallback_store.update(input_data)
            
            # Try transcript node only
            transcript_result = self.transcript_node.prep(fallback_store)
            if transcript_result.get('prep_status') == 'success':
                exec_result = self.transcript_node.exec(fallback_store, transcript_result)
                if exec_result.get('exec_status') == 'success':
                    post_result = self.transcript_node.post(fallback_store, transcript_result, exec_result)
                    
                    if post_result.get('post_status') == 'success':
                        return {
                            'status': 'partial_success',
                            'data': {
                                'transcript': fallback_store.get('transcript_data', {}),
                                'video_metadata': fallback_store.get('video_metadata', {})
                            },
                            'fallback_used': True,
                            'original_error': error_info.__dict__,
                            'timestamp': datetime.utcnow().isoformat()
                        }
            
            # If we get here, even fallback failed
            raise RuntimeError("Fallback workflow also failed")
            
        except Exception as e:
            logger.error(f"Fallback workflow failed: {str(e)}")
            return self._prepare_error_result(error_info)
    
    def _prepare_final_result(self, node_results: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare the final workflow result."""
        try:
            # Extract data from store
            transcript_data = self.store.get('transcript_data', {})
            summary_data = self.store.get('summary_data', {})
            timestamp_data = self.store.get('timestamp_data', {})
            keyword_data = self.store.get('keyword_data', {})
            video_metadata = self.store.get('video_metadata', {})
            
            # Prepare final response
            final_result = {
                'status': 'success',
                'data': {
                    'video_id': transcript_data.get('video_id', ''),
                    'title': video_metadata.get('title', ''),
                    'duration': video_metadata.get('duration_seconds', 0),
                    'summary': summary_data.get('summary_text', ''),
                    'timestamps': timestamp_data.get('timestamps', []),
                    'keywords': keyword_data.get('keywords', [])
                },
                'metadata': {
                    'processing_time': time.time() - self.metrics.start_time if self.metrics else 0,
                    'node_count': len(self.nodes),
                    'timestamp_count': len(timestamp_data.get('timestamps', [])),
                    'keyword_count': len(keyword_data.get('keywords', [])),
                    'summary_word_count': summary_data.get('word_count', 0),
                    'workflow_version': '1.0'
                },
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Add monitoring data if available
            if self.enable_monitoring and self.metrics:
                final_result['performance'] = {
                    'total_duration': self.metrics.total_duration,
                    'node_durations': self.metrics.node_durations,
                    'retry_counts': self.metrics.node_retry_counts,
                    'success_rate': self.metrics.success_rate
                }
            
            return final_result
            
        except Exception as e:
            logger.error(f"Failed to prepare final result: {str(e)}")
            return self._prepare_error_result(
                WorkflowError(
                    flow_name=self.name,
                    error_type=type(e).__name__,
                    message=f"Result preparation failed: {str(e)}",
                    is_recoverable=False
                )
            )
    
    def _prepare_error_result(self, error_info: WorkflowError) -> Dict[str, Any]:
        """Prepare error result for workflow failure."""
        return {
            'status': 'failed',
            'error': {
                'type': error_info.error_type,
                'message': error_info.message,
                'flow_name': error_info.flow_name,
                'failed_node': error_info.failed_node,
                'is_recoverable': error_info.is_recoverable,
                'timestamp': error_info.timestamp
            },
            'metadata': {
                'workflow_version': '1.0',
                'error_count': len(self.workflow_errors),
                'fallback_attempted': self.enable_fallbacks
            },
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def _finalize_metrics(self, success: bool) -> None:
        """Finalize performance metrics."""
        if not self.metrics:
            return
        
        self.metrics.end_time = time.time()
        self.metrics.total_duration = self.metrics.end_time - self.metrics.start_time
        self.metrics.success_rate = 100.0 if success else 0.0
        
        # Calculate memory usage if available
        if self.memory_monitoring_available:
            try:
                memory_info = self.process.memory_info()
                self.metrics.memory_usage = {
                    'rss': memory_info.rss,
                    'vms': memory_info.vms,
                    'peak_rss': getattr(memory_info, 'peak_wset', memory_info.rss)
                }
            except Exception as e:
                logger.warning(f"Failed to collect memory metrics: {str(e)}")
        
        logger.info(f"Workflow metrics: duration={self.metrics.total_duration:.2f}s, "
                   f"success={success}, retries={sum(self.metrics.node_retry_counts.values())}")
    
    def get_workflow_status(self) -> Dict[str, Any]:
        """Get current workflow status and metrics."""
        if not self.metrics:
            return {'status': 'not_started'}
        
        current_time = time.time()
        elapsed_time = current_time - self.metrics.start_time
        
        return {
            'status': 'running' if not self.metrics.end_time else 'completed',
            'elapsed_time': elapsed_time,
            'completed_nodes': len(self.metrics.node_durations),
            'total_nodes': len(self.nodes),
            'progress_percent': (len(self.metrics.node_durations) / len(self.nodes)) * 100,
            'current_node': self.nodes[len(self.metrics.node_durations)].name if len(self.metrics.node_durations) < len(self.nodes) else None,
            'error_count': len(self.workflow_errors),
            'retry_count': sum(self.metrics.node_retry_counts.values()),
            'memory_usage': self.metrics.memory_usage
        }
    
    def reset_workflow(self) -> None:
        """Reset workflow state for reuse."""
        self.store = Store()
        self.workflow_errors = []
        self.metrics = None
        
        logger.info("Workflow state reset")


def create_youtube_summarizer_flow(config: Optional[Dict[str, Any]] = None) -> YouTubeSummarizerFlow:
    """
    Factory function to create and configure a YouTube summarizer flow.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Configured YouTubeSummarizerFlow instance
    """
    # Default configuration
    default_config = {
        'enable_monitoring': True,
        'enable_fallbacks': True,
        'max_retries': 2,
        'timeout_seconds': 300
    }
    
    # Merge with provided config
    if config:
        default_config.update(config)
    
    # Create and return flow
    flow = YouTubeSummarizerFlow(**default_config)
    
    logger.info(f"Created YouTube summarizer flow with config: {default_config}")
    return flow


# Convenience function for direct usage
def process_youtube_video(youtube_url: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Process a YouTube video through the complete summarization workflow.
    
    Args:
        youtube_url: YouTube video URL to process
        config: Optional workflow configuration
        
    Returns:
        Complete processing results or error information
    """
    flow = create_youtube_summarizer_flow(config)
    
    input_data = {
        'youtube_url': youtube_url,
        'processing_start_time': datetime.utcnow().isoformat()
    }
    
    return flow.run(input_data)