"""
Main workflow orchestrator for YouTube video summarization.

This module contains the core orchestration logic that manages the execution
of all processing nodes in the YouTube summarization workflow.
"""

import logging
import time
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

try:
    from pocketflow import Flow, Node, Store, FlowState
except ImportError:
    # Fallback base classes for development
    from abc import ABC, abstractmethod
    
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

from .config import WorkflowConfig, NodeConfig, create_default_workflow_config
from .error_handler import ErrorHandler, WorkflowError
from .monitoring import WorkflowMonitor, WorkflowMetrics

# Import nodes with fallback for development
try:
    from ..nodes import (
        YouTubeTranscriptNode,
        SummarizationNode,
        TimestampNode,
        KeywordExtractionNode,
        NodeError
    )
    from ..utils.language_detector import (
        YouTubeLanguageDetector, LanguageDetectionResult, LanguageCode,
        detect_video_language, is_chinese_video, is_english_video,
        get_preferred_transcript_languages, optimize_chinese_content_for_llm,
        ensure_chinese_encoding, detect_mixed_language_content,
        segment_mixed_language_content
    )
except ImportError:
    # Create mock implementations for development
    class MockNode:
        def __init__(self, *args, **kwargs):
            self.name = kwargs.get('name', 'MockNode')
        def prep(self, store): return {'prep_status': 'success'}
        def exec(self, store, prep_result): return {'exec_status': 'success', 'retry_count': 0}
        def post(self, store, prep_result, exec_result): return {'post_status': 'success'}
    
    YouTubeTranscriptNode = MockNode
    SummarizationNode = MockNode
    TimestampNode = MockNode
    KeywordExtractionNode = MockNode
    
    class NodeError: pass
    class LanguageCode:
        ENGLISH = "en"
        CHINESE_SIMPLIFIED = "zh-CN"
        CHINESE_GENERIC = "zh"
    
    class LanguageDetectionResult:
        def __init__(self, detected_language, confidence_score, detection_method):
            self.detected_language = detected_language
            self.confidence_score = confidence_score
            self.detection_method = detection_method
    
    class YouTubeLanguageDetector:
        def detect_language_comprehensive(self, metadata, transcript=None):
            return LanguageDetectionResult(LanguageCode.ENGLISH, 0.9, "mock")
    
    def detect_video_language(metadata, transcript=None):
        return LanguageDetectionResult(LanguageCode.ENGLISH, 0.9, "mock")
    def is_chinese_video(metadata, transcript=None): return False
    def is_english_video(metadata, transcript=None): return True
    def get_preferred_transcript_languages(metadata, transcript=None): return ['en']
    def optimize_chinese_content_for_llm(text, task_type="summarization"): 
        return {'optimized_text': text, 'system_prompt': f'Process for {task_type}', 'user_prompt': text}
    def ensure_chinese_encoding(text): return text
    def detect_mixed_language_content(text, chunk_size=500): 
        return {'is_mixed': False, 'primary_language': 'en'}
    def segment_mixed_language_content(text, target_languages=None): 
        return {'is_segmented': False, 'segments': [{'language': 'en', 'content': text}]}

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
        
        # Initialize core components
        self.error_handler = ErrorHandler(self.name)
        self.monitor = WorkflowMonitor(self.name) if self.enable_monitoring else None
        self.node_instances: Dict[str, Node] = {}
        
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
        
        logger.info(f"YouTubeSummarizerFlow initialized with {len(self.nodes)} nodes")

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the complete workflow for YouTube video summarization.
        
        Args:
            input_data: Dictionary containing 'youtube_url' and optional configuration
            
        Returns:
            Dictionary containing all processing results or error information
        """
        logger.info("Starting YouTube summarizer workflow")
        
        # Start monitoring
        if self.monitor:
            self.monitor.start_workflow()
        
        try:
            # Validate input
            self._validate_input(input_data)
            
            # Initialize store with input data
            self.store = Store()
            self.store.update(input_data)
            
            # Execute workflow with timeout protection
            result = self._execute_workflow_with_timeout()
            
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
        
        # Define node execution order
        node_order = [
            'YouTubeTranscriptNode',
            'SummarizationNode', 
            'TimestampNode',
            'KeywordExtractionNode'
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
                
                # Record success
                self.error_handler.handle_node_success(node_name)
                if self.monitor:
                    self.monitor.finish_node(node_name, "success")
                
                # Perform language detection after transcript
                if node_name == 'YouTubeTranscriptNode' and result.get('post_status') == 'success':
                    self._perform_language_detection()
                
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
            
            # Post phase
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

    def _should_execute_node(self, node_name: str) -> bool:
        """Check if a node should be executed based on configuration."""
        node_config = self.config.get_node_config(node_name)
        return node_config and node_config.enabled

    def _initialize_configured_nodes(self) -> None:
        """Initialize nodes based on configuration."""
        # Ensure we have default node configs
        self.config.create_default_node_configs()
        
        # Create node instances
        node_classes = {
            'YouTubeTranscriptNode': YouTubeTranscriptNode,
            'SummarizationNode': SummarizationNode,
            'TimestampNode': TimestampNode,
            'KeywordExtractionNode': KeywordExtractionNode
        }
        
        for node_name, node_class in node_classes.items():
            node_config = self.config.get_node_config(node_name)
            if node_config and node_config.enabled:
                self.node_instances[node_name] = node_class(
                    max_retries=node_config.max_retries,
                    retry_delay=node_config.retry_delay
                )
                logger.debug(f"Initialized node: {node_name}")

    def _initialize_circuit_breakers(self) -> None:
        """Initialize circuit breakers for nodes."""
        for node_name in self.node_instances.keys():
            self.error_handler.create_circuit_breaker(
                node_name, 
                self.config.circuit_breaker_config
            )

    def _create_default_config(self, enable_monitoring: bool, enable_fallbacks: bool, 
                             max_retries: int, timeout_seconds: int) -> WorkflowConfig:
        """Create default configuration from legacy parameters."""
        config = create_default_workflow_config()
        config.enable_monitoring = enable_monitoring
        config.enable_fallbacks = enable_fallbacks
        config.max_retries = max_retries
        config.timeout_seconds = timeout_seconds
        return config

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
            
            # Try to execute just the transcript node
            if 'YouTubeTranscriptNode' in self.node_instances:
                result = self._execute_single_node('YouTubeTranscriptNode')
                
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
        final_result = {
            'success': True,
            'results': results,
            'store_data': dict(self.store),
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

    def _prepare_error_result(self, error_info: WorkflowError) -> Dict[str, Any]:
        """Prepare an error result."""
        return {
            'success': False,
            'error': error_info.to_dict(),
            'store_data': dict(self.store) if hasattr(self, 'store') else {},
            'error_summary': self.error_handler.get_error_summary(),
            'workflow_metadata': {
                'execution_time': time.time(),
                'failed_at': error_info.failed_node or 'workflow_setup'
            }
        }