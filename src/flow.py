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
from enum import Enum

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


class NodeExecutionMode(Enum):
    """Execution modes for workflow nodes."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"  # For future implementation
    CONDITIONAL = "conditional"  # For future implementation


@dataclass
class NodeConfig:
    """Configuration for individual nodes in the workflow."""
    name: str
    enabled: bool = True
    required: bool = True
    max_retries: int = 3
    retry_delay: float = 1.0
    timeout_seconds: int = 60
    dependencies: List[str] = field(default_factory=list)
    output_keys: List[str] = field(default_factory=list)
    
    
@dataclass
class DataFlowConfig:
    """Configuration for data flow between nodes."""
    input_mapping: Dict[str, str] = field(default_factory=dict)
    output_mapping: Dict[str, str] = field(default_factory=dict)
    data_validation: Dict[str, Any] = field(default_factory=dict)
    cleanup_keys: List[str] = field(default_factory=list)
    
    
@dataclass
class WorkflowConfig:
    """Complete workflow configuration."""
    execution_mode: NodeExecutionMode = NodeExecutionMode.SEQUENTIAL
    node_configs: Dict[str, NodeConfig] = field(default_factory=dict)
    data_flow_config: DataFlowConfig = field(default_factory=DataFlowConfig)
    enable_monitoring: bool = True
    enable_fallbacks: bool = True
    max_retries: int = 2
    timeout_seconds: int = 300
    store_cleanup: bool = True

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
                 config: Optional[WorkflowConfig] = None,
                 enable_monitoring: bool = True,
                 enable_fallbacks: bool = True,
                 max_retries: int = 2,
                 timeout_seconds: int = 300):
        """
        Initialize the YouTube summarizer workflow.
        
        Args:
            config: Complete workflow configuration (preferred)
            enable_monitoring: Whether to enable performance monitoring (legacy)
            enable_fallbacks: Whether to enable fallback mechanisms (legacy)
            max_retries: Maximum number of workflow retries (legacy)
            timeout_seconds: Total workflow timeout in seconds (legacy)
        """
        super().__init__("YouTubeSummarizerFlow")
        
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
        
        # Initialize metrics and tracking
        self.metrics = None
        self.workflow_errors: List[WorkflowError] = []
        self.node_instances: Dict[str, Node] = {}
        
        # Initialize nodes with configuration
        self._initialize_configured_nodes()
        
        # Setup monitoring
        if self.enable_monitoring:
            self._setup_monitoring()
        
        logger.info(f"YouTubeSummarizerFlow initialized with {len(self.nodes)} nodes (config: {self.config.execution_mode.value})")
    
    def _create_default_config(
        self, 
        enable_monitoring: bool, 
        enable_fallbacks: bool, 
        max_retries: int, 
        timeout_seconds: int
    ) -> WorkflowConfig:
        """Create default workflow configuration from legacy parameters."""
        
        # Default node configurations
        node_configs = {
            'YouTubeTranscriptNode': NodeConfig(
                name='YouTubeTranscriptNode',
                enabled=True,
                required=True,
                max_retries=3,
                retry_delay=1.0,
                timeout_seconds=60,
                dependencies=[],
                output_keys=['transcript_data', 'video_metadata']
            ),
            'SummarizationNode': NodeConfig(
                name='SummarizationNode',
                enabled=True,
                required=True,
                max_retries=3,
                retry_delay=2.0,
                timeout_seconds=120,
                dependencies=['YouTubeTranscriptNode'],
                output_keys=['summary_data', 'summary_metadata']
            ),
            'TimestampNode': NodeConfig(
                name='TimestampNode',
                enabled=True,
                required=False,  # Optional for graceful degradation
                max_retries=3,
                retry_delay=2.0,
                timeout_seconds=90,
                dependencies=['YouTubeTranscriptNode'],
                output_keys=['timestamp_data', 'timestamp_metadata']
            ),
            'KeywordExtractionNode': NodeConfig(
                name='KeywordExtractionNode',
                enabled=True,
                required=False,  # Optional for graceful degradation
                max_retries=3,
                retry_delay=1.5,
                timeout_seconds=60,
                dependencies=['YouTubeTranscriptNode'],
                output_keys=['keyword_data', 'keyword_metadata']
            )
        }
        
        # Default data flow configuration
        data_flow_config = DataFlowConfig(
            input_mapping={
                'youtube_url': 'youtube_url',
                'processing_start_time': 'processing_start_time'
            },
            output_mapping={
                'transcript_data': 'transcript_data',
                'video_metadata': 'video_metadata',
                'summary_data': 'summary_data',
                'timestamp_data': 'timestamp_data',
                'keyword_data': 'keyword_data'
            },
            data_validation={
                'youtube_url': {'type': str, 'required': True},
                'transcript_data': {'type': dict, 'required': True},
                'video_metadata': {'type': dict, 'required': True}
            },
            cleanup_keys=['_internal_processing_state', '_temp_data']
        )
        
        return WorkflowConfig(
            execution_mode=NodeExecutionMode.SEQUENTIAL,
            node_configs=node_configs,
            data_flow_config=data_flow_config,
            enable_monitoring=enable_monitoring,
            enable_fallbacks=enable_fallbacks,
            max_retries=max_retries,
            timeout_seconds=timeout_seconds,
            store_cleanup=True
        )
    
    def _initialize_configured_nodes(self) -> None:
        """Initialize nodes based on configuration."""
        try:
            enabled_nodes = []
            
            # Create node instances based on configuration
            for node_name, node_config in self.config.node_configs.items():
                if not node_config.enabled:
                    logger.info(f"Skipping disabled node: {node_name}")
                    continue
                
                node_instance = self._create_node_instance(node_name, node_config)
                if node_instance:
                    self.node_instances[node_name] = node_instance
                    enabled_nodes.append(node_instance)
                    logger.debug(f"Initialized node: {node_name}")
            
            # Sort nodes by dependencies for execution order
            self.nodes = self._sort_nodes_by_dependencies(enabled_nodes)
            
            logger.info(f"Initialized {len(self.nodes)} nodes in configured order")
            
        except Exception as e:
            logger.error(f"Failed to initialize configured nodes: {str(e)}")
            raise
    
    def _create_node_instance(self, node_name: str, node_config: NodeConfig) -> Optional[Node]:
        """Create a node instance based on configuration."""
        try:
            if node_name == 'YouTubeTranscriptNode':
                return YouTubeTranscriptNode(
                    max_retries=node_config.max_retries,
                    retry_delay=node_config.retry_delay
                )
            elif node_name == 'SummarizationNode':
                return SummarizationNode(
                    max_retries=node_config.max_retries,
                    retry_delay=node_config.retry_delay
                )
            elif node_name == 'TimestampNode':
                return TimestampNode(
                    max_retries=node_config.max_retries,
                    retry_delay=node_config.retry_delay
                )
            elif node_name == 'KeywordExtractionNode':
                return KeywordExtractionNode(
                    max_retries=node_config.max_retries,
                    retry_delay=node_config.retry_delay
                )
            else:
                logger.warning(f"Unknown node type: {node_name}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to create node {node_name}: {str(e)}")
            return None
    
    def _sort_nodes_by_dependencies(self, nodes: List[Node]) -> List[Node]:
        """Sort nodes based on their dependencies."""
        # For now, use the predefined order since all nodes are sequential
        # In the future, this could implement topological sorting
        
        node_order = [
            'YouTubeTranscriptNode',
            'SummarizationNode', 
            'TimestampNode',
            'KeywordExtractionNode'
        ]
        
        sorted_nodes = []
        node_map = {node.name: node for node in nodes}
        
        for node_name in node_order:
            if node_name in node_map:
                sorted_nodes.append(node_map[node_name])
        
        return sorted_nodes
    
    def _validate_data_flow(self, current_node_name: str) -> None:
        """Validate data flow requirements before node execution."""
        if current_node_name not in self.config.node_configs:
            return
        
        node_config = self.config.node_configs[current_node_name]
        
        # Check dependencies
        for dependency in node_config.dependencies:
            if dependency not in self.node_instances:
                raise ValueError(f"Node {current_node_name} depends on {dependency} but it's not available")
        
        # Validate required data in store based on data flow config
        data_validation = self.config.data_flow_config.data_validation
        
        for key, validation_rules in data_validation.items():
            if validation_rules.get('required', False):
                if key not in self.store:
                    # Check if this is a dependency issue
                    if any(dep in current_node_name for dep in node_config.dependencies):
                        raise ValueError(f"Required data '{key}' not found in store for node {current_node_name}")
                
                # Type validation
                expected_type = validation_rules.get('type')
                if expected_type and key in self.store:
                    actual_value = self.store[key]
                    if not isinstance(actual_value, expected_type):
                        raise ValueError(f"Data '{key}' has type {type(actual_value).__name__} but expected {expected_type.__name__}")
    
    def _cleanup_store_data(self) -> None:
        """Clean up temporary data from store based on configuration."""
        if not self.config.store_cleanup:
            return
        
        cleanup_keys = self.config.data_flow_config.cleanup_keys
        
        for key in cleanup_keys:
            if key in self.store:
                del self.store[key]
                logger.debug(f"Cleaned up store key: {key}")
    
    def _map_output_data(self, node_name: str, node_result: Dict[str, Any]) -> None:
        """Map node output to store based on data flow configuration."""
        if node_name not in self.config.node_configs:
            return
        
        node_config = self.config.node_configs[node_name]
        output_mapping = self.config.data_flow_config.output_mapping
        
        # Map configured output keys
        for output_key in node_config.output_keys:
            if output_key in output_mapping:
                mapped_key = output_mapping[output_key]
                if output_key in self.store:
                    # Create mapped copy if different
                    if mapped_key != output_key:
                        self.store[mapped_key] = self.store[output_key]
                        logger.debug(f"Mapped output {output_key} -> {mapped_key}")
        
        # Store node execution results for debugging
        if self.enable_monitoring:
            self.store[f'_node_results_{node_name}'] = {
                'execution_time': time.time(),
                'status': 'completed',
                'result_keys': list(node_result.keys())
            }
    
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
        """Execute all nodes in the configured sequence with data flow validation."""
        results = {}
        
        for node in self.nodes:
            node_start_time = time.time()
            node_config = self.config.node_configs.get(node.name)
            
            try:
                logger.info(f"Executing node: {node.name}")
                
                # Validate data flow requirements before execution
                self._validate_data_flow(node.name)
                
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
                
                # Handle node failure based on configuration
                if post_result.get('post_status') != 'success':
                    error_msg = post_result.get('error', 'Unknown node error')
                    
                    # Check if this is a required node
                    if node_config and node_config.required:
                        raise RuntimeError(f"Required node {node.name} failed: {error_msg}")
                    else:
                        # Optional node failure - log warning and continue
                        logger.warning(f"Optional node {node.name} failed: {error_msg}")
                        results[node.name]['status'] = 'failed_optional'
                        
                        # Track metrics even for failed optional nodes
                        if self.enable_monitoring:
                            node_duration = time.time() - node_start_time
                            self.metrics.node_durations[node.name] = node_duration
                            self.metrics.node_retry_counts[node.name] = exec_result.get('retry_count', 0)
                        
                        continue
                
                # Map output data based on configuration
                self._map_output_data(node.name, post_result)
                
                # Track metrics for successful execution
                if self.enable_monitoring:
                    node_duration = time.time() - node_start_time
                    self.metrics.node_durations[node.name] = node_duration
                    
                    # Track retry count
                    retry_count = exec_result.get('retry_count', 0)
                    self.metrics.node_retry_counts[node.name] = retry_count
                
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
                
                # Check if this is a required node
                if node_config and node_config.required:
                    raise RuntimeError(error_msg)
                else:
                    # Optional node failure - continue with warning
                    logger.warning(f"Optional node {node.name} failed, continuing: {error_msg}")
                    continue
        
        # Clean up temporary data after all nodes complete
        self._cleanup_store_data()
        
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


def create_youtube_summarizer_flow(config: Optional[Union[Dict[str, Any], WorkflowConfig]] = None) -> YouTubeSummarizerFlow:
    """
    Factory function to create and configure a YouTube summarizer flow.
    
    Args:
        config: Optional configuration (dict for legacy support or WorkflowConfig)
        
    Returns:
        Configured YouTubeSummarizerFlow instance
    """
    if isinstance(config, WorkflowConfig):
        # Use provided WorkflowConfig directly
        flow = YouTubeSummarizerFlow(config=config)
        logger.info(f"Created YouTube summarizer flow with WorkflowConfig: {config.execution_mode.value}")
        return flow
    
    # Legacy dict-based configuration
    default_config = {
        'enable_monitoring': True,
        'enable_fallbacks': True,
        'max_retries': 2,
        'timeout_seconds': 300
    }
    
    # Merge with provided config
    if config:
        default_config.update(config)
    
    # Create and return flow with legacy parameters
    flow = YouTubeSummarizerFlow(**default_config)
    
    logger.info(f"Created YouTube summarizer flow with legacy config: {default_config}")
    return flow


def create_custom_workflow_config(
    node_configs: Optional[Dict[str, Dict[str, Any]]] = None,
    data_flow_overrides: Optional[Dict[str, Any]] = None,
    execution_mode: NodeExecutionMode = NodeExecutionMode.SEQUENTIAL,
    **kwargs
) -> WorkflowConfig:
    """
    Create a custom workflow configuration with specific node and data flow settings.
    
    Args:
        node_configs: Dictionary of node configurations
        data_flow_overrides: Data flow configuration overrides
        execution_mode: Node execution mode
        **kwargs: Additional workflow configuration parameters
        
    Returns:
        Custom WorkflowConfig instance
    """
    # Start with default node configurations
    default_node_configs = {
        'YouTubeTranscriptNode': {
            'enabled': True,
            'required': True,
            'max_retries': 3,
            'retry_delay': 1.0,
            'timeout_seconds': 60
        },
        'SummarizationNode': {
            'enabled': True,
            'required': True,
            'max_retries': 3,
            'retry_delay': 2.0,
            'timeout_seconds': 120
        },
        'TimestampNode': {
            'enabled': True,
            'required': False,
            'max_retries': 3,
            'retry_delay': 2.0,
            'timeout_seconds': 90
        },
        'KeywordExtractionNode': {
            'enabled': True,
            'required': False,
            'max_retries': 3,
            'retry_delay': 1.5,
            'timeout_seconds': 60
        }
    }
    
    # Merge with provided node configs
    if node_configs:
        for node_name, node_config in node_configs.items():
            if node_name in default_node_configs:
                default_node_configs[node_name].update(node_config)
    
    # Convert to NodeConfig objects
    node_config_objects = {}
    for node_name, config_dict in default_node_configs.items():
        node_config_objects[node_name] = NodeConfig(
            name=node_name,
            enabled=config_dict.get('enabled', True),
            required=config_dict.get('required', True),
            max_retries=config_dict.get('max_retries', 3),
            retry_delay=config_dict.get('retry_delay', 1.0),
            timeout_seconds=config_dict.get('timeout_seconds', 60),
            dependencies=config_dict.get('dependencies', []),
            output_keys=config_dict.get('output_keys', [])
        )
    
    # Default data flow config
    data_flow_config = DataFlowConfig(
        input_mapping={'youtube_url': 'youtube_url'},
        output_mapping={
            'transcript_data': 'transcript_data',
            'summary_data': 'summary_data',
            'timestamp_data': 'timestamp_data',
            'keyword_data': 'keyword_data'
        },
        data_validation={
            'youtube_url': {'type': str, 'required': True},
            'transcript_data': {'type': dict, 'required': True}
        },
        cleanup_keys=['_internal_processing_state', '_temp_data']
    )
    
    # Apply data flow overrides
    if data_flow_overrides:
        if 'input_mapping' in data_flow_overrides:
            data_flow_config.input_mapping.update(data_flow_overrides['input_mapping'])
        if 'output_mapping' in data_flow_overrides:
            data_flow_config.output_mapping.update(data_flow_overrides['output_mapping'])
        if 'data_validation' in data_flow_overrides:
            data_flow_config.data_validation.update(data_flow_overrides['data_validation'])
        if 'cleanup_keys' in data_flow_overrides:
            data_flow_config.cleanup_keys.extend(data_flow_overrides['cleanup_keys'])
    
    # Create workflow config
    workflow_config = WorkflowConfig(
        execution_mode=execution_mode,
        node_configs=node_config_objects,
        data_flow_config=data_flow_config,
        enable_monitoring=kwargs.get('enable_monitoring', True),
        enable_fallbacks=kwargs.get('enable_fallbacks', True),
        max_retries=kwargs.get('max_retries', 2),
        timeout_seconds=kwargs.get('timeout_seconds', 300),
        store_cleanup=kwargs.get('store_cleanup', True)
    )
    
    logger.info(f"Created custom workflow config with {len(node_config_objects)} nodes")
    return workflow_config


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