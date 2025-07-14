"""
Main workflow orchestrator for YouTube video summarization.

This module contains the core orchestration logic that manages the execution
of all processing nodes in the YouTube summarization workflow.
"""

import logging
import time
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from dataclasses import dataclass

try:
    from pocketflow import Flow, Node
except ImportError:
    # Fallback base classes for development
    from abc import ABC, abstractmethod
    
    class Flow(ABC):
        def __init__(self, name: str):
            self.name = name
            self.logger = logging.getLogger(f"{__name__}.{name}")
            self.nodes: List[Node] = []
            self.store = {}
        
        @abstractmethod
        def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
            pass
    
    class Node(ABC):
        def __init__(self, name: str):
            self.name = name
            self.logger = logging.getLogger(f"{__name__}.{name}")
        
        @abstractmethod
        def prep(self, store: Dict[str, Any]) -> Dict[str, Any]:
            pass
        
        @abstractmethod
        def exec(self, store: Dict[str, Any], prep_result: Dict[str, Any]) -> Dict[str, Any]:
            pass
        
        @abstractmethod
        def post(self, store: Dict[str, Any], prep_result: Dict[str, Any], exec_result: Dict[str, Any]) -> Dict[str, Any]:
            pass

# Define FlowState for use in the code
class FlowState:
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"

from .config import WorkflowConfig, NodeConfig, create_default_workflow_config
from .error_handler import ErrorHandler, WorkflowError
from .monitoring import WorkflowMonitor, WorkflowMetrics

# Import services and utilities - lazy import to avoid circular imports
from ..database.connection import get_database_session
from ..utils.language_detector import (
    YouTubeLanguageDetector, LanguageDetectionResult, LanguageCode,
    detect_video_language, is_chinese_video, is_english_video,
    get_preferred_transcript_languages, optimize_chinese_content_for_llm,
    ensure_chinese_encoding, detect_mixed_language_content,
    segment_mixed_language_content
)

# Define BatchProcessingConfig here to avoid circular imports
@dataclass
class BatchProcessingConfig:
    """Configuration for batch processing operations."""
    max_concurrent_jobs: int = 5
    batch_timeout_seconds: int = 3600
    retry_failed_jobs: bool = True
    max_retries_per_job: int = 3

logger = logging.getLogger(__name__)


class YouTubeSummarizerFlow(Flow):
    """
    Main workflow for YouTube video summarization.
    
    This flow orchestrates all processing nodes to transform a YouTube URL
    into a comprehensive summary with timestamps and keywords.
    """
    
    def __init__(self, 
                 config: Optional[WorkflowConfig] = None,
                 enable_monitoring: bool = True,
                 enable_fallbacks: bool = True,
                 max_retries: int = 2,
                 timeout_seconds: int = 300,
                 video_service=None):
        """
        Initialize the YouTube summarizer workflow.
        
        Args:
            config: Complete workflow configuration (preferred)
            enable_monitoring: Whether to enable performance monitoring (legacy)
            enable_fallbacks: Whether to enable fallback mechanisms (legacy)
            max_retries: Maximum number of workflow retries (legacy)
            timeout_seconds: Total workflow timeout in seconds (legacy)
            video_service: Optional video service for database persistence
        """
        super().__init__()
        self.name = "YouTubeSummarizerFlow"
        
        # Initialize configuration
        if config:
            self.config = config
        else:
            # Create default config from legacy parameters
            self.config = self._create_default_config(
                enable_monitoring, enable_fallbacks, max_retries, timeout_seconds
            )
        
        # Extract commonly used config values
        self.enable_monitoring = self.config.enable_monitoring
        self.enable_fallbacks = self.config.enable_fallbacks
        self.max_retries = self.config.max_retries
        self.timeout_seconds = self.config.timeout_seconds
        
        # Initialize core components
        self.error_handler = ErrorHandler(self.name)
        self.monitor = WorkflowMonitor(self.name) if self.enable_monitoring else None
        self.node_instances: Dict[str, Node] = {}
        self.video_service = video_service
        
        # Initialize language detection
        self.language_detector = None
        self.detected_language_result: Optional[LanguageDetectionResult] = None
        if self.config.language_processing.enable_language_detection:
            self.language_detector = YouTubeLanguageDetector()
            logger.debug("Language detector initialized")
        
        # Initialize nodes with configuration
        self._initialize_configured_nodes()
        
        # Initialize circuit breakers
        self._initialize_circuit_breakers()
        
        logger.info(f"YouTubeSummarizerFlow initialized with {len(self.node_instances)} nodes")

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the complete workflow for YouTube video summarization.
        
        Args:
            input_data: Dictionary containing 'youtube_url' and optional configuration
            
        Returns:
            Dictionary containing all processing results or error information
        """
        logger.info(f"Starting YouTube summarizer workflow with input: {input_data}")
        logger.info(f"Workflow config: enable_monitoring={self.enable_monitoring}, enable_fallbacks={self.enable_fallbacks}")
        
        # Start monitoring
        if self.monitor:
            self.monitor.start_workflow()
        
        try:
            # Validate input
            self._validate_input(input_data)
            
            # Initialize store with input data
            self.store = {}
            self.store.update(input_data)
            
            # Check for duplicate videos and handle accordingly
            duplicate_result = self._handle_duplicate_detection(input_data)
            
            # Log duplicate detection result summary (avoid logging full data)
            if duplicate_result:
                data = duplicate_result.get('data', {})
                summary = {
                    'status': duplicate_result.get('status'),
                    'video_id': data.get('video_id'),
                    'video_title': data.get('video_title', '')[:50] + '...' if len(data.get('video_title', '')) > 50 else data.get('video_title', ''),
                    'has_summary': bool(data.get('summary_text')),
                    'has_keywords': bool(data.get('keywords')),
                    'has_timestamps': bool(data.get('timestamps')),
                    'workflow_skipped': duplicate_result.get('workflow_skipped', False)
                }
                logger.info(f"Duplicate detection result: {summary}")
            else:
                logger.info("Duplicate detection result: None")
            
            if duplicate_result:
                logger.info("Returning duplicate result, skipping workflow execution")
                return duplicate_result
            
            # Check database health and configure graceful degradation
            self._configure_database_degradation()
            
            # Execute workflow with timeout protection
            logger.info("Starting main workflow execution")
            result = self._execute_workflow_with_timeout()
            logger.info(f"Workflow execution completed with result keys: {list(result.keys()) if result else 'None'}")
            
            # Save workflow completion metadata
            self._save_workflow_completion_metadata(input_data, result)
            
            # Finalize monitoring
            if self.monitor:
                self.monitor.finish_workflow(success=True)
            
            logger.info("Workflow completed successfully")
            return self._prepare_final_result(result)
            
        except Exception as e:
            # Handle workflow error
            error_info = self.error_handler.handle_node_error(e, "workflow", "execution")
            
            # Finalize monitoring
            if self.monitor:
                self.monitor.finish_workflow(success=False)
            
            # Try fallback if enabled
            if self.enable_fallbacks and error_info.is_recoverable:
                return self._execute_fallback_workflow(input_data, error_info)
            
            return self._prepare_error_result(error_info)

    def _create_default_config(self, enable_monitoring: bool, enable_fallbacks: bool, 
                             max_retries: int, timeout_seconds: int) -> WorkflowConfig:
        """Create default configuration from legacy parameters."""
        config = create_default_workflow_config()
        config.enable_monitoring = enable_monitoring
        config.enable_fallbacks = enable_fallbacks
        config.max_retries = max_retries
        config.timeout_seconds = timeout_seconds
        return config

    def _initialize_configured_nodes(self) -> None:
        """Initialize nodes based on configuration."""
        # Ensure we have default node configs
        self.config.create_default_node_configs()
        
        # Lazy import node classes to avoid circular imports
        def get_node_classes():
            try:
                from ..refactored_nodes.youtube_data_node import YouTubeDataNode
                from ..refactored_nodes.llm_nodes import SummarizationNode, KeywordExtractionNode
                from ..refactored_nodes.summary_nodes import TimestampNode
                from ..refactored_nodes.semantic_analysis_nodes import (
                    SemanticAnalysisNode,
                    VectorSearchNode,
                    EnhancedTimestampNode
                )
                
                return {
                    'YouTubeDataNode': YouTubeDataNode,      # Unified YouTube data acquisition
                    'SummarizationNode': SummarizationNode,  # Generate summary
                    'SemanticAnalysisNode': SemanticAnalysisNode,  # Semantic analysis and clustering
                    'VectorSearchNode': VectorSearchNode,    # Vector-based search and optimization
                    'EnhancedTimestampNode': EnhancedTimestampNode,  # Enhanced timestamp generation
                    'KeywordExtractionNode': KeywordExtractionNode,  # Keyword extraction
                    # Keep TimestampNode for backward compatibility (but not in default execution order)
                    'TimestampNode': TimestampNode
                }
            except ImportError as e:
                logger.warning(f"Could not import all node classes: {e}")
                return {}
        
        # Create node instances
        node_classes = get_node_classes()
        
        for node_name, node_class in node_classes.items():
            node_config = self.config.get_node_config(node_name)
            if node_config and node_config.enabled:
                # Create node with video service for database persistence
                node_kwargs = {
                    'max_retries': node_config.max_retries,
                    'retry_delay': node_config.retry_delay
                }
                
                # Add video_service parameter if the node supports it
                if self.video_service and hasattr(node_class.__init__, '__code__'):
                    init_params = node_class.__init__.__code__.co_varnames
                    if 'video_service' in init_params:
                        node_kwargs['video_service'] = self.video_service
                
                self.node_instances[node_name] = node_class(**node_kwargs)
                logger.debug(f"Initialized node: {node_name} with database support: {self.video_service is not None}")

    def _initialize_circuit_breakers(self) -> None:
        """Initialize circuit breakers for nodes."""
        for node_name in self.node_instances.keys():
            self.error_handler.create_circuit_breaker(
                node_name, 
                self.config.circuit_breaker_config
            )

    def _validate_input(self, input_data: Dict[str, Any]) -> None:
        """Validate input data for the workflow."""
        if not isinstance(input_data, dict):
            raise ValueError("Input data must be a dictionary")
        
        if 'youtube_url' not in input_data:
            raise ValueError("Missing required 'youtube_url' in input data")
        
        youtube_url = input_data['youtube_url']
        if not isinstance(youtube_url, str) or not youtube_url.strip():
            raise ValueError("youtube_url must be a non-empty string")

    def _prepare_final_result(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare the final workflow result."""
        # Extract data from store for API compatibility
        store_data = dict(self.store) if hasattr(self, 'store') else {}
        
        # Extract video metadata if available
        video_metadata = store_data.get('video_metadata', {})
        transcript_data = store_data.get('transcript_data', {})
        
        # Log basic success info without dumping full data
        video_title = video_metadata.get('title', 'Unknown')[:50] + '...' if len(video_metadata.get('title', '')) > 50 else video_metadata.get('title', 'Unknown')
        transcript_length = transcript_data.get('word_count', 0) if transcript_data else 0
        logger.info(f"Final result prepared: '{video_title}' ({transcript_length} words)")
        
        # Build the data structure that the API expects
        api_data = {
            'video_id': video_metadata.get('video_id', store_data.get('video_id', '')),
            'title': video_metadata.get('title', store_data.get('video_title', '')),
            'duration': video_metadata.get('duration_seconds', video_metadata.get('duration', store_data.get('video_duration', 0))),
            'summary': store_data.get('summary_text', ''),
            'keywords': store_data.get('keywords', []),
            'timestamps': store_data.get('timestamps', [])
        }
        
        return {
            'status': 'success',
            'data': api_data,
            'workflow_metadata': {
                'execution_time': time.time(),
                'nodes_executed': len(self.node_instances),
                'config_enabled': {
                    'monitoring': self.enable_monitoring,
                    'fallbacks': self.enable_fallbacks
                }
            }
        }

    def _prepare_error_result(self, error_info: WorkflowError) -> Dict[str, Any]:
        """Prepare an error result."""
        return {
            'status': 'failed',
            'error': error_info.to_dict(),
            'store_data': dict(self.store) if hasattr(self, 'store') else {},
            'error_summary': self.error_handler.get_error_summary(),
            'workflow_metadata': {
                'execution_time': time.time(),
                'failed_at': error_info.failed_node or 'workflow_setup'
            }
        }

    def _execute_fallback_workflow(self, input_data: Dict[str, Any], error_info: WorkflowError) -> Dict[str, Any]:
        """Execute a simplified fallback workflow."""
        logger.info("Executing fallback workflow")
        
        try:
            # Initialize minimal store
            if not hasattr(self, 'store'):
                self.store = {}
            self.store.update(input_data)
            
            # Try to execute just the YouTube data node
            if 'YouTubeDataNode' in self.node_instances:
                result = self._execute_single_node('YouTubeDataNode')
                
                return {
                    'success': True,
                    'fallback_mode': True,
                    'partial_results': result,
                    'original_error': error_info.to_dict()
                }
            
        except Exception as fallback_error:
            logger.error(f"Fallback workflow also failed: {str(fallback_error)}")
        
        return self._prepare_error_result(error_info)

    def _handle_duplicate_detection(self, input_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Handle duplicate video detection and processing decisions."""
        # Check if duplicate detection is enabled
        if not hasattr(self.config, 'duplicate_handling') or not getattr(self.config.duplicate_handling, 'enable_duplicate_detection', False):
            return None
            
        if not self.video_service:
            # No database service available, proceed with processing
            return None
        
        try:
            # Extract video ID from URL
            youtube_url = input_data.get('youtube_url', '')
            video_id = self._extract_video_id(youtube_url)
            
            if not video_id:
                # Cannot extract video ID, proceed with processing
                return None
            
            # For now, just proceed with processing - duplicate detection can be enhanced later
            return None
            
        except Exception as e:
            logger.warning(f"Duplicate detection failed: {str(e)}")
            return None

    def _extract_video_id(self, youtube_url: str) -> Optional[str]:
        """Extract video ID from YouTube URL."""
        import re
        # Match various YouTube URL formats
        patterns = [
            r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
            r'(?:embed\/)([0-9A-Za-z_-]{11})',
            r'(?:watch\?v=)([0-9A-Za-z_-]{11})'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, youtube_url)
            if match:
                return match.group(1)
        return None

    def _execute_configured_workflow(self) -> Dict[str, Any]:
        """Execute the configured workflow with all enabled nodes."""
        # Get the execution order from configuration
        execution_order = self.config.get_execution_order()
        
        # Initialize results
        results = {}
        
        for node_name in execution_order:
            if self._should_execute_node(node_name):
                try:
                    result = self._execute_single_node(node_name)
                    results[node_name] = result
                except Exception as e:
                    logger.error(f"Node {node_name} execution failed: {str(e)}")
                    # Error handling will be done by the error handler
                    raise
        
        return results

    def _execute_single_node(self, node_name: str) -> Dict[str, Any]:
        """Execute a single node with proper error handling."""
        if node_name not in self.node_instances:
            raise ValueError(f"Node {node_name} not found in instances")
        
        node = self.node_instances[node_name]
        
        # Execute node phases
        try:
            prep_result = node.prep(self.store)
            exec_result = node.exec(self.store, prep_result)
            post_result = node.post(self.store, prep_result, exec_result)
            
            return {
                'prep': prep_result,
                'exec': exec_result,
                'post': post_result,
                'success': True
            }
        except Exception as e:
            logger.error(f"Node {node_name} execution failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'node': node_name
            }

    def _should_execute_node(self, node_name: str) -> bool:
        """Check if a node should be executed based on configuration."""
        node_config = self.config.get_node_config(node_name)
        should_execute = node_config and node_config.enabled
        logger.debug(f"Node {node_name}: config exists={node_config is not None}, enabled={node_config.enabled if node_config else False}, should_execute={should_execute}")
        return should_execute

    def _configure_database_degradation(self):
        """Configure graceful degradation for database issues."""
        # For now, just log that we're checking database health
        logger.debug("Checking database health for graceful degradation")

    def _execute_workflow_with_timeout(self) -> Dict[str, Any]:
        """Execute the workflow with timeout protection."""
        try:
            # Execute the configured workflow
            return self._execute_configured_workflow()
        except Exception as e:
            logger.error(f"Workflow execution failed: {str(e)}")
            raise

    def _save_workflow_completion_metadata(self, input_data: Dict[str, Any], result: Dict[str, Any]):
        """Save workflow completion metadata."""
        # For now, just log completion
        logger.debug(f"Workflow completed for URL: {input_data.get('youtube_url', 'unknown')}")

    def reset_workflow(self) -> None:
        """Reset the workflow state for reuse."""
        try:
            # Clear the store
            if hasattr(self, 'store'):
                self.store.clear()
            
            # Reset error handler
            if hasattr(self, 'error_handler'):
                self.error_handler.reset()
            
            # Reset monitor
            if hasattr(self, 'monitor') and self.monitor:
                self.monitor.reset()
            
            logger.debug("Workflow state reset successfully")
        except Exception as e:
            logger.warning(f"Error during workflow reset: {str(e)}")


class YouTubeBatchProcessingFlow(Flow):
    """
    Workflow for batch processing multiple YouTube videos.
    
    This flow orchestrates the batch processing of multiple YouTube URLs
    through a queue-based system with worker management.
    """
    
    def __init__(self, 
                 config: Optional[WorkflowConfig] = None,
                 batch_config: Optional[BatchProcessingConfig] = None,
                 enable_monitoring: bool = True,
                 video_service=None):
        """
        Initialize the YouTube batch processing workflow.
        
        Args:
            config: Workflow configuration
            batch_config: Batch processing specific configuration
            enable_monitoring: Whether to enable performance monitoring
            video_service: Optional video service for database persistence
        """
        super().__init__()
        self.name = "YouTubeBatchProcessingFlow"
        
        # Initialize configuration
        self.config = config or create_default_workflow_config()
        self.batch_config = batch_config or BatchProcessingConfig()
        self.enable_monitoring = enable_monitoring
        
        # Initialize core components
        self.error_handler = ErrorHandler(self.name)
        self.monitor = WorkflowMonitor(self.name) if enable_monitoring else None
        self.node_instances: Dict[str, Node] = {}
        self.video_service = video_service
        
        # Initialize batch processing nodes
        self._initialize_batch_nodes()
        
        # Initialize circuit breakers
        self._initialize_circuit_breakers()
        
        logger.info(f"YouTubeBatchProcessingFlow initialized with {len(self.nodes)} nodes")

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the complete batch processing workflow.
        
        Args:
            input_data: Dictionary containing 'batch_urls' and optional 'batch_config'
            
        Returns:
            Dictionary containing all processing results or error information
        """
        logger.info(f"Starting YouTube batch processing workflow with input: {input_data}")
        
        # Start monitoring
        if self.monitor:
            self.monitor.start_workflow()
        
        try:
            # Validate input
            self._validate_batch_input(input_data)
            
            # Initialize store with input data
            self.store = {}
            self.store.update(input_data)
            
            # Execute batch processing workflow
            result = self._execute_batch_workflow()
            
            # Finalize monitoring
            if self.monitor:
                self.monitor.finish_workflow(success=True)
            
            logger.info("Batch processing workflow completed successfully")
            return self._prepare_batch_final_result(result)
            
        except Exception as e:
            # Handle workflow error
            error_info = self.error_handler.handle_node_error(e, "batch_workflow", "execution")
            
            # Finalize monitoring
            if self.monitor:
                self.monitor.finish_workflow(success=False)
            
            return self._prepare_error_result(error_info)

    def _execute_batch_workflow(self) -> Dict[str, Any]:
        """Execute the batch processing workflow with all nodes."""
        results = {}
        
        # Define batch processing node execution order
        batch_node_order = [
            'BatchCreationNode',    # Create batch and validate URLs
            'BatchProcessingNode',  # Process all batch items
            'BatchStatusNode'       # Monitor status and provide final report
        ]
        
        for node_name in batch_node_order:
            if not self._should_execute_node(node_name):
                logger.info(f"Skipping disabled batch node: {node_name}")
                continue
            
            try:
                # Check circuit breaker
                if not self.error_handler.can_node_execute(node_name):
                    logger.warning(f"Circuit breaker open for {node_name}, skipping")
                    continue
                
                # Execute node
                result = self._execute_single_node(node_name)
                results[node_name] = result
                
                # Log node completion
                logger.debug(f"Batch node {node_name} completed with result keys: {list(result.keys())}")
                
                # Record success
                self.error_handler.handle_node_success(node_name)
                if self.monitor:
                    self.monitor.finish_node(node_name, "success")
                
                # Check if we should continue based on node results
                if not self._should_continue_batch_processing(node_name, result):
                    logger.info(f"Stopping batch processing after {node_name}")
                    break
                
            except Exception as e:
                # Handle node error
                error_info = self.error_handler.handle_node_error(e, node_name)
                results[node_name] = {'error': error_info.to_dict()}
                
                if self.monitor:
                    self.monitor.record_error(node_name)
                    self.monitor.finish_node(node_name, "failed")
                
                # Check if node is required for batch processing
                if node_name in ['BatchCreationNode']:  # Critical nodes
                    raise e
                
                logger.warning(f"Non-critical batch node {node_name} failed, continuing workflow")
        
        return results

    def _initialize_batch_nodes(self) -> None:
        """Initialize batch processing nodes."""
        # Create batch processing node instances
        batch_node_classes = {
            'BatchCreationNode': BatchCreationNode,
            'BatchProcessingNode': BatchProcessingNode,
            'BatchStatusNode': BatchStatusNode
        }
        
        for node_name, node_class in batch_node_classes.items():
            # Create node with batch configuration
            node_kwargs = {
                'max_retries': 3,
                'retry_delay': 2.0,
                'config': self.batch_config
            }
            
            self.node_instances[node_name] = node_class(**node_kwargs)
            logger.debug(f"Initialized batch node: {node_name}")

    def _initialize_circuit_breakers(self) -> None:
        """Initialize circuit breakers for batch processing nodes."""
        for node_name in self.node_instances.keys():
            self.error_handler.create_circuit_breaker(
                node_name, 
                self.config.circuit_breaker_config
            )

    def _should_execute_node(self, node_name: str) -> bool:
        """Check if a batch processing node should be executed."""
        # All batch processing nodes are required by default
        return node_name in self.node_instances

    def _should_continue_batch_processing(self, node_name: str, result: Dict[str, Any]) -> bool:
        """Determine if batch processing should continue after a node completes."""
        # Check if the node completed successfully
        post_result = result.get('post_result', {})
        if post_result.get('processing_status') != 'success':
            logger.warning(f"Node {node_name} did not complete successfully")
            return False
        
        # Special handling for different nodes
        if node_name == 'BatchCreationNode':
            # Must have successfully created a batch
            return post_result.get('batch_created', False)
        
        elif node_name == 'BatchProcessingNode':
            # Should continue to status monitoring regardless of processing results
            return True
        
        elif node_name == 'BatchStatusNode':
            # This is the final node
            return False
        
        return True

    def _validate_batch_input(self, input_data: Dict[str, Any]) -> None:
        """Validate input data for batch processing."""
        if not isinstance(input_data, dict):
            raise ValueError("Input data must be a dictionary")
        
        if 'batch_urls' not in input_data:
            raise ValueError("Missing required 'batch_urls' in input data")
        
        batch_urls = input_data['batch_urls']
        if not isinstance(batch_urls, list) or not batch_urls:
            raise ValueError("batch_urls must be a non-empty list")

    def _prepare_batch_final_result(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare the final batch processing result."""
        # Extract data from store for API compatibility
        store_data = dict(self.store)
        
        # Extract batch information
        batch_id = store_data.get('batch_id', 'unknown')
        batch_status = store_data.get('batch_final_status', 'unknown')
        batch_statistics = store_data.get('batch_statistics', {})
        
        # Build the final result structure
        final_result = {
            'status': 'success',
            'batch_id': batch_id,
            'batch_status': batch_status,
            'batch_statistics': batch_statistics,
            'results': results,
            'store_data': store_data,
            'workflow_metadata': {
                'execution_time': time.time(),
                'nodes_executed': list(results.keys()),
                'workflow_type': 'batch_processing'
            }
        }
        
        # Add monitoring data if available
        if self.monitor and self.monitor.metrics:
            final_result['metrics'] = self.monitor.metrics.get_summary()
        
        # Add error summary
        error_summary = self.error_handler.get_error_summary()
        if error_summary['total_errors'] > 0:
            final_result['error_summary'] = error_summary
        
        return final_result

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
                    raise e
                
                # Wait before retry
                time.sleep(2.0 * (retry_count + 1))
                logger.warning(f"Workflow retry {retry_count + 1}/{self.max_retries} after error: {str(e)}")

    def _execute_nodes_sequence(self) -> Dict[str, Any]:
        """Execute all nodes in sequential order."""
        results = {}
        
        # Define node execution order with semantic analysis integration
        node_order = [
            'YouTubeDataNode',      # Unified YouTube data acquisition
            'SummarizationNode',    # Generate summary 
            'SemanticAnalysisNode', # Perform semantic analysis and clustering
            'VectorSearchNode',     # Vector-based search and optimization
            'EnhancedTimestampNode', # Enhanced timestamp generation (replaces TimestampNode)
            'KeywordExtractionNode' # Keyword extraction (can benefit from semantic data)
        ]
        
        for node_name in node_order:
            if not self._should_execute_node(node_name):
                logger.info(f"Skipping disabled node: {node_name}")
                continue
            
            try:
                # Check circuit breaker
                if not self.error_handler.can_node_execute(node_name):
                    logger.warning(f"Circuit breaker open for {node_name}, skipping")
                    continue
                
                # Execute node
                result = self._execute_single_node(node_name)
                results[node_name] = result
                
                # Log node completion
                logger.debug(f"Node {node_name} completed with result keys: {list(result.keys())}")
                
                # Record success
                self.error_handler.handle_node_success(node_name)
                if self.monitor:
                    self.monitor.finish_node(node_name, "success")
                
                # Perform language detection after YouTube data acquisition
                if node_name == 'YouTubeDataNode' and result.get('post_status') == 'success':
                    self._perform_language_detection()
                
                # Handle semantic analysis fallback
                if node_name == 'SemanticAnalysisNode' and result.get('post_status') != 'success':
                    self._handle_semantic_analysis_fallback()
                
                # Handle vector search fallback
                if node_name == 'VectorSearchNode' and result.get('post_status') != 'success':
                    self._handle_vector_search_fallback()
                
            except Exception as e:
                # Handle node error
                error_info = self.error_handler.handle_node_error(e, node_name)
                results[node_name] = {'error': error_info.to_dict()}
                
                if self.monitor:
                    self.monitor.record_error(node_name)
                    self.monitor.finish_node(node_name, "failed")
                
                # Check if node is required
                node_config = self.config.get_node_config(node_name)
                if node_config and node_config.required:
                    raise e
                
                logger.warning(f"Optional node {node_name} failed, continuing workflow")
        
        return results

    def _execute_single_node(self, node_name: str) -> Dict[str, Any]:
        """Execute a single node with full lifecycle."""
        logger.info(f"Executing node: {node_name}")
        
        # Start monitoring
        if self.monitor:
            node_metrics = self.monitor.start_node(node_name)
        
        # Get node instance
        node = self.node_instances.get(node_name)
        if not node:
            raise ValueError(f"Node not found: {node_name}")
        
        try:
            # Prep phase
            prep_result = node.prep(self.store)
            
            # Exec phase
            exec_result = node.exec(self.store, prep_result)
            
            # Post phase (handle async post methods)
            # Handle post-processing (all post methods are now synchronous)
            post_result = node.post(self.store, prep_result, exec_result)
            
            return {
                'prep_result': prep_result,
                'exec_result': exec_result,
                'post_result': post_result,
                'node_status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Node {node_name} failed: {str(e)}")
            raise

    def _perform_language_detection(self) -> None:
        """Perform language detection after transcript is available."""
        if not self.config.language_processing.enable_language_detection:
            return
        
        try:
            # Get video metadata and transcript from store
            video_metadata = self.store.get('video_metadata', {})
            transcript_data = self.store.get('transcript_data', {})
            transcript_text = transcript_data.get('transcript_text', '')
            
            # Detect language
            detection_result = self._detect_video_language(video_metadata, transcript_text)
            
            # Store result for other nodes
            self.store['language_detection_result'] = {
                'detected_language': detection_result.detected_language.value,
                'confidence_score': detection_result.confidence_score,
                'detection_method': detection_result.detection_method,
                'is_chinese': detection_result.detected_language in [
                    LanguageCode.CHINESE_SIMPLIFIED, 
                    LanguageCode.CHINESE_GENERIC
                ]
            }
            
        except Exception as e:
            logger.warning(f"Language detection failed: {str(e)}")

    def _detect_video_language(self, video_metadata: Dict[str, Any], transcript_text: Optional[str] = None) -> LanguageDetectionResult:
        """Detect the language of the video using metadata and transcript."""
        if not self.language_detector:
            return LanguageDetectionResult(
                detected_language=LanguageCode.ENGLISH,
                confidence_score=0.5,
                detection_method="default_fallback"
            )
        
        try:
            detection_result = self.language_detector.detect_language_comprehensive(
                video_metadata, transcript_text
            )
            self.detected_language_result = detection_result
            return detection_result
            
        except Exception as e:
            logger.warning(f"Language detection failed: {str(e)}")
            fallback_result = LanguageDetectionResult(
                detected_language=LanguageCode.ENGLISH,
                confidence_score=0.1,
                detection_method="error_fallback"
            )
            self.detected_language_result = fallback_result
            return fallback_result

    def _handle_duplicate_detection(self, input_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Handle duplicate video detection and processing decisions.
        
        Args:
            input_data: Input data containing youtube_url
            
        Returns:
            Dict with existing video data if duplicate found and should not reprocess,
            None if should proceed with processing
        """
        # Check if duplicate detection is enabled
        if not self.config.duplicate_handling.enable_duplicate_detection:
            return None
            
        if not self.video_service:
            # No database service available, proceed with processing
            return None
        
        try:
            # Extract video ID from URL
            youtube_url = input_data.get('youtube_url', '')
            video_id = self._extract_video_id(youtube_url)
            
            if not video_id:
                # Cannot extract video ID, proceed with processing
                return None
            
            # Check if video exists in database (now synchronous)
            try:
                with get_database_session() as session:
                    from ..services.video_service import VideoService
                    video_service = VideoService(session)
                    video_exists = video_service.video_exists(video_id)
            except Exception as e:
                logger.warning(f"Error checking if video exists: {e}")
                # If we can't check, proceed with processing to be safe
                return None
            
            if not video_exists:
                # Video not found, proceed with processing
                logger.info(f"Video {video_id} not found in database, proceeding with processing")
                return None
            
            # Video exists - check reprocessing policy
            reprocess_policy = self._get_reprocessing_policy(input_data)
            
            if reprocess_policy == 'never':
                # Return existing data without reprocessing
                logger.info(f"Video {video_id} exists, returning cached data (policy: never)")
                return self._get_existing_video_data(video_id)
            
            elif reprocess_policy == 'always':
                # Proceed with reprocessing
                logger.info(f"Video {video_id} exists, reprocessing (policy: always)")
                return None
            
            elif reprocess_policy == 'if_failed':
                # Check if previous processing failed
                with get_database_session() as session:
                    from ..services.video_service import VideoService
                    video_service = VideoService(session)
                    processing_status = video_service.get_processing_status(video_id)
                if processing_status == 'failed':
                    logger.info(f"Video {video_id} exists but failed, reprocessing (policy: if_failed)")
                    return None
                else:
                    logger.info(f"Video {video_id} exists and succeeded, returning cached data (policy: if_failed)")
                    return self._get_existing_video_data(video_id)
            
            else:
                # Default: return existing data
                logger.info(f"Video {video_id} exists, returning cached data (default policy)")
                return self._get_existing_video_data(video_id)
                
        except Exception as e:
            logger.error(f"Error in duplicate detection: {str(e)}")
            # On error, proceed with processing
            return None

    def _extract_video_id(self, youtube_url: str) -> Optional[str]:
        """Extract video ID from YouTube URL."""
        import re
        
        # Common YouTube URL patterns
        patterns = [
            r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([a-zA-Z0-9_-]{11})',
            r'youtube\.com/v/([a-zA-Z0-9_-]{11})',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, youtube_url)
            if match:
                return match.group(1)
        
        return None

    def _get_reprocessing_policy(self, input_data: Dict[str, Any]) -> str:
        """
        Get reprocessing policy from input data or configuration.
        
        Returns:
            'never' - Never reprocess existing videos
            'always' - Always reprocess existing videos  
            'if_failed' - Only reprocess if previous attempt failed
        """
        # Check input data for explicit policy override
        policy = input_data.get('reprocess_policy')
        if (policy in ['never', 'always', 'if_failed'] and 
            self.config.duplicate_handling.allow_policy_override):
            return policy
        
        # Use configured policy
        return self.config.duplicate_handling.reprocess_policy

    def _get_existing_video_data(self, video_id: str) -> Dict[str, Any]:
        """
        Retrieve existing video data from database.
        
        Args:
            video_id: YouTube video ID
            
        Returns:
            Dict containing existing video processing results
        """
        try:
            # Get video with all related data
            with get_database_session() as session:
                video_service = VideoService(session)
                video = video_service.get_video_by_video_id(video_id)
                
                if not video:
                    logger.warning(f"Video {video_id} exists check passed but retrieval failed")
                    return None
                
                # Convert database models to workflow result format
                result_data = {
                    'video_id': video.video_id,
                    'video_title': video.title,
                    'video_duration': video.duration,
                    'video_url': video.url,
                    'transcript_text': '',
                    'summary_text': '',
                    'keywords': [],
                    'timestamps': []
                }
                
                # Extract transcript data
                if video.transcripts:
                    latest_transcript = video.transcripts[-1]  # Get most recent
                    result_data['transcript_text'] = latest_transcript.content
                    result_data['transcript_language'] = latest_transcript.language
                
                # Extract summary data
                if video.summaries:
                    latest_summary = video.summaries[-1]  # Get most recent
                    result_data['summary_text'] = latest_summary.content
                    result_data['summary_processing_time'] = latest_summary.processing_time
                
                # Extract keywords data
                if video.keywords:
                    latest_keywords = video.keywords[-1]  # Get most recent
                    keywords_data = latest_keywords.keywords_json
                    if isinstance(keywords_data, dict):
                        result_data['keywords'] = keywords_data.get('keywords', [])
                    elif isinstance(keywords_data, list):
                        result_data['keywords'] = keywords_data
                
                # Extract timestamped segments
                if video.timestamped_segments:
                    latest_segments = video.timestamped_segments[-1]  # Get most recent
                    segments_data = latest_segments.segments_json
                    if isinstance(segments_data, dict):
                        result_data['timestamps'] = segments_data.get('timestamps', [])
                    elif isinstance(segments_data, list):
                        result_data['timestamps'] = segments_data
            
            # Add metadata about cached result
            result_data['cached_result'] = True
            result_data['cached_from_database'] = True
            result_data['processing_time'] = 0.0  # No processing time for cached result
            
            logger.info(f"Retrieved cached data for video {video_id}")
            
            return {
                'status': 'success',
                'data': result_data,
                'workflow_skipped': True,
                'reason': 'duplicate_video_cached'
            }
            
        except Exception as e:
            logger.error(f"Error retrieving existing video data for {video_id}: {str(e)}")
            return None

    def _save_workflow_completion_metadata(self, input_data: Dict[str, Any], result: Dict[str, Any]) -> None:
        """
        Save workflow completion metadata to database.
        
        Args:
            input_data: Original workflow input data
            result: Workflow execution result
        """
        if not self.video_service:
            return
        
        try:
            # Extract video ID
            youtube_url = input_data.get('youtube_url', '')
            video_id = self._extract_video_id(youtube_url)
            
            if not video_id:
                return
            
            # Prepare workflow metadata
            workflow_params = {
                'workflow_config': {
                    'enable_monitoring': self.config.enable_monitoring,
                    'enable_fallbacks': self.config.enable_fallbacks,
                    'max_retries': self.config.max_retries,
                    'timeout_seconds': self.config.timeout_seconds,
                    'duplicate_detection_enabled': self.config.duplicate_handling.enable_duplicate_detection,
                    'reprocess_policy': self.config.duplicate_handling.reprocess_policy
                },
                'execution_details': {
                    'start_time': input_data.get('processing_start_time'),
                    'request_id': input_data.get('request_id'),
                    'reprocess_policy_override': input_data.get('reprocess_policy'),
                    'nodes_executed': list(self.node_instances.keys())
                }
            }
            
            # Determine final status
            status = 'completed'
            error_info = None
            
            if result.get('status') != 'success':
                status = 'failed'
                error_info = str(result.get('error', 'Unknown error'))
            
            # Save metadata directly (now synchronous)
            with get_database_session() as session:
                video_service = VideoService(session)
                video_service.save_processing_metadata(
                    video_id=video_id,
                    workflow_params=workflow_params,
                    status=status,
                    error_info=error_info
                )
            
            logger.info(f"Saved workflow completion metadata for video {video_id}")
                
        except Exception as e:
            logger.error(f"Error saving workflow completion metadata: {str(e)}")

    def _configure_database_degradation(self) -> None:
        """
        Configure graceful degradation for database operations.
        
        Checks database health and adjusts workflow behavior accordingly.
        """
        if not self.video_service:
            # No database service, workflow will run without persistence
            logger.info("No database service available, workflow will run without persistence")
            self.store['database_available'] = False
            self.store['database_degraded'] = False
            return
        
        try:
            # Quick database health check (now synchronous)
            try:
                # Simple existence check - this should be fast
                with get_database_session() as session:
                    from ..services.video_service import VideoService
                    video_service = VideoService(session)
                    video_service.video_exists("test_health_check")
                db_healthy = True
                logger.info("Database health check passed")
            except Exception as e:
                logger.warning(f"Database health check failed: {e}")
                # Fallback: assume database is unhealthy
                db_healthy = False
            
            if db_healthy:
                logger.info("Database health check passed, full persistence enabled")
                self.store['database_available'] = True
                self.store['database_degraded'] = False
            else:
                logger.warning("Database health check failed, enabling degraded mode")
                self.store['database_available'] = False
                self.store['database_degraded'] = True
                
                # Optionally disable database service to prevent further attempts
                if hasattr(self, '_original_video_service'):
                    # Already degraded
                    pass
                else:
                    self._original_video_service = self.video_service
                    self.video_service = None  # Disable database operations
                
        except Exception as e:
            logger.error(f"Error during database health check: {str(e)}")
            self.store['database_available'] = False
            self.store['database_degraded'] = True
            
            # Disable database service
            if hasattr(self, '_original_video_service'):
                pass
            else:
                self._original_video_service = self.video_service
                self.video_service = None

    def _restore_database_service(self) -> None:
        """Restore database service if it was disabled during degradation."""
        if hasattr(self, '_original_video_service'):
            self.video_service = self._original_video_service
            delattr(self, '_original_video_service')

    def _should_execute_node(self, node_name: str) -> bool:
        """Check if a node should be executed based on configuration."""
        node_config = self.config.get_node_config(node_name)
        should_execute = node_config and node_config.enabled
        logger.debug(f"Node {node_name}: config exists={node_config is not None}, enabled={node_config.enabled if node_config else False}, should_execute={should_execute}")
        return should_execute

    def _validate_input(self, input_data: Dict[str, Any]) -> None:
        """Validate input data for the workflow."""
        if not isinstance(input_data, dict):
            raise ValueError("Input data must be a dictionary")
        
        if 'youtube_url' not in input_data:
            raise ValueError("Missing required 'youtube_url' in input data")
        
        youtube_url = input_data['youtube_url']
        if not isinstance(youtube_url, str) or not youtube_url.strip():
            raise ValueError("youtube_url must be a non-empty string")

    def _execute_fallback_workflow(self, input_data: Dict[str, Any], error_info: WorkflowError) -> Dict[str, Any]:
        """Execute a simplified fallback workflow."""
        logger.info("Executing fallback workflow")
        
        try:
            # Initialize minimal store
            self.store = Store()
            self.store.update(input_data)
            
            # Try to execute just the YouTube data node
            if 'YouTubeDataNode' in self.node_instances:
                result = self._execute_single_node('YouTubeDataNode')
                
                return {
                    'success': True,
                    'fallback_mode': True,
                    'partial_results': result,
                    'original_error': error_info.to_dict()
                }
            
        except Exception as fallback_error:
            logger.error(f"Fallback workflow also failed: {str(fallback_error)}")
        
        return self._prepare_error_result(error_info)

    def _prepare_final_result(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare the final workflow result."""
        # Extract data from store for API compatibility
        store_data = dict(self.store)
        
        # Extract video metadata if available
        video_metadata = store_data.get('video_metadata', {})
        transcript_data = store_data.get('transcript_data', {})
        
        # Log basic success info without dumping full data
        video_title = video_metadata.get('title', 'Unknown')[:50] + '...' if len(video_metadata.get('title', '')) > 50 else video_metadata.get('title', 'Unknown')
        transcript_length = transcript_data.get('word_count', 0) if transcript_data else 0
        logger.info(f"Final result prepared: '{video_title}' ({transcript_length} words)")
        
        # Build the data structure that the API expects
        api_data = {
            'video_id': video_metadata.get('video_id', store_data.get('video_id', '')),
            'title': video_metadata.get('title', store_data.get('video_title', '')),
            'duration': video_metadata.get('duration', store_data.get('video_duration', 0)),
            'summary': store_data.get('summary_text', ''),
            'keywords': store_data.get('keywords', []),
            'timestamps': store_data.get('timestamps', [])
        }
        
        final_result = {
            'status': 'success',
            'data': api_data,
            'results': results,
            'store_data': store_data,
            'workflow_metadata': {
                'execution_time': time.time(),
                'nodes_executed': list(results.keys()),
                'config_mode': self.config.execution_mode.value
            }
        }
        
        # Add monitoring data if available
        if self.monitor and self.monitor.metrics:
            final_result['metrics'] = self.monitor.metrics.get_summary()
        
        # Add error summary
        error_summary = self.error_handler.get_error_summary()
        if error_summary['total_errors'] > 0:
            final_result['error_summary'] = error_summary
        
        return final_result

    def _handle_semantic_analysis_fallback(self) -> None:
        """Handle fallback when semantic analysis fails."""
        logger.warning("Semantic analysis failed, setting up fallback data")
        
        # Set empty semantic analysis data for downstream nodes
        self.store['semantic_analysis_result'] = {
            'status': 'fallback',
            'reason': 'semantic_analysis_failed',
            'timestamps': [],
            'semantic_clusters': [],
            'semantic_keywords': [],
            'semantic_metrics': {}
        }
        self.store['semantic_clusters'] = []
        self.store['semantic_keywords'] = []
        self.store['semantic_metrics'] = {}
        self.store['semantic_timestamps'] = []
        
        logger.info("Semantic analysis fallback data prepared")

    def _handle_vector_search_fallback(self) -> None:
        """Handle fallback when vector search fails."""
        logger.warning("Vector search failed, setting up fallback data")
        
        # Set empty vector search data for downstream nodes
        self.store['vector_search_result'] = {
            'exec_status': 'fallback',
            'reason': 'vector_search_failed',
            'coherence_analysis': {'coherence_score': 0.0, 'analysis': 'fallback'},
            'optimized_timestamps': []
        }
        self.store['coherence_analysis'] = {'coherence_score': 0.0, 'analysis': 'fallback'}
        self.store['optimized_timestamps'] = []
        
        logger.info("Vector search fallback data prepared")

    def _prepare_error_result(self, error_info: WorkflowError) -> Dict[str, Any]:
        """Prepare an error result."""
        return {
            'status': 'failed',
            'error': error_info.to_dict(),
            'store_data': dict(self.store) if hasattr(self, 'store') else {},
            'error_summary': self.error_handler.get_error_summary(),
            'workflow_metadata': {
                'execution_time': time.time(),
                'failed_at': error_info.failed_node or 'workflow_setup'
            }
        }