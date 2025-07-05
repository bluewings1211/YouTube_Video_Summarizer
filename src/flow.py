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
    
    
try:
    from .nodes import (
        YouTubeTranscriptNode,
        SummarizationNode,
        TimestampNode,
        KeywordExtractionNode,
        NodeError
    )
    from .utils.language_detector import (
        YouTubeLanguageDetector, LanguageDetectionResult, LanguageCode,
        detect_video_language, is_chinese_video, is_english_video,
        get_preferred_transcript_languages, optimize_chinese_content_for_llm,
        ensure_chinese_encoding, detect_mixed_language_content,
        segment_mixed_language_content
    )
except ImportError:
    # For testing - try absolute import
    try:
        from nodes import (
            YouTubeTranscriptNode,
            SummarizationNode,
            TimestampNode,
            KeywordExtractionNode,
            NodeError
        )
        from utils.language_detector import (
            YouTubeLanguageDetector, LanguageDetectionResult, LanguageCode,
            detect_video_language, is_chinese_video, is_english_video,
            get_preferred_transcript_languages, optimize_chinese_content_for_llm,
            ensure_chinese_encoding, detect_mixed_language_content,
            segment_mixed_language_content
        )
    except ImportError:
        # Create mock nodes for testing
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
        
        class NodeError:
            pass
        
        # Mock language detection functions
        class LanguageCode:
            ENGLISH = "en"
            CHINESE_SIMPLIFIED = "zh-CN"
            CHINESE_GENERIC = "zh"
        
        def detect_video_language(metadata, transcript=None):
            return type('MockResult', (), {'detected_language': LanguageCode.ENGLISH, 'confidence_score': 0.9})()
        
        def is_chinese_video(metadata, transcript=None): return False
        def is_english_video(metadata, transcript=None): return True
        def get_preferred_transcript_languages(metadata, transcript=None): return ['en']
        def optimize_chinese_content_for_llm(text, task_type="summarization"): return {'optimized_text': text}
        def ensure_chinese_encoding(text): return text
        def detect_mixed_language_content(text, chunk_size=500): return {'is_mixed': False, 'primary_language': 'en'}
        def segment_mixed_language_content(text, target_languages=None): return {'is_segmented': False, 'segments': [{'language': 'en', 'content': text}]}

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
    retry_count: int = 0
    node_phase: Optional[str] = None  # prep, exec, post
    recovery_action: Optional[str] = None
    
    
@dataclass
class FallbackStrategy:
    """Configuration for fallback strategies."""
    enable_transcript_only: bool = True      # Fallback to transcript only
    enable_summary_fallback: bool = True     # Use simpler summary if AI fails
    enable_partial_results: bool = True      # Return partial results on failure
    enable_retry_with_degraded: bool = True  # Retry with reduced functionality
    max_fallback_attempts: int = 2           # Maximum fallback attempts
    fallback_timeout_factor: float = 0.5     # Reduce timeout for fallbacks
    
    
@dataclass 
class CircuitBreakerConfig:
    """Configuration for circuit breaker pattern."""
    failure_threshold: int = 3               # Failures before opening circuit
    recovery_timeout: int = 60               # Seconds before attempting recovery
    success_threshold: int = 2               # Successes needed to close circuit
    enabled: bool = True                     # Enable circuit breaker
    
    
class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Circuit open, blocking requests
    HALF_OPEN = "half_open"  # Testing if service recovered
    
    
@dataclass
class CircuitBreaker:
    """Circuit breaker implementation for node failures."""
    node_name: str
    config: CircuitBreakerConfig
    state: CircuitBreakerState = CircuitBreakerState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[float] = None
    consecutive_failures: int = 0
    
    def can_execute(self) -> bool:
        """Check if node can execute based on circuit breaker state."""
        if not self.config.enabled:
            return True
            
        current_time = time.time()
        
        if self.state == CircuitBreakerState.CLOSED:
            return True
        elif self.state == CircuitBreakerState.OPEN:
            if (self.last_failure_time and 
                current_time - self.last_failure_time > self.config.recovery_timeout):
                self.state = CircuitBreakerState.HALF_OPEN
                return True
            return False
        elif self.state == CircuitBreakerState.HALF_OPEN:
            return True
        
        return False
    
    def record_success(self) -> None:
        """Record successful execution."""
        self.success_count += 1
        self.consecutive_failures = 0
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
    
    def record_failure(self) -> None:
        """Record failed execution."""
        self.failure_count += 1
        self.consecutive_failures += 1
        self.last_failure_time = time.time()
        
        if (self.state == CircuitBreakerState.CLOSED and 
            self.consecutive_failures >= self.config.failure_threshold):
            self.state = CircuitBreakerState.OPEN
        elif self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.OPEN
            
            
class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"           # Minor issues, continue processing
    MEDIUM = "medium"     # Significant issues, try fallback
    HIGH = "high"         # Major issues, stop processing
    CRITICAL = "critical" # System-level issues, immediate failure


@dataclass
class LanguageProcessingConfig:
    """Configuration for language-specific processing."""
    enable_language_detection: bool = True
    enable_chinese_optimization: bool = True
    default_language: str = "en"  # Default language when detection fails
    preferred_languages: List[str] = field(default_factory=lambda: ["en", "zh-CN", "zh-TW"])
    chinese_prompt_optimization: bool = True
    language_confidence_threshold: float = 0.5
    mixed_language_handling: str = "primary"  # "primary", "segment", "dual"
    preserve_chinese_encoding: bool = True
    enable_transcript_language_preference: bool = True


@dataclass
class WorkflowConfig:
    """Complete workflow configuration."""
    execution_mode: NodeExecutionMode = NodeExecutionMode.SEQUENTIAL
    node_configs: Dict[str, NodeConfig] = field(default_factory=dict)
    data_flow_config: DataFlowConfig = field(default_factory=DataFlowConfig)
    fallback_strategy: FallbackStrategy = field(default_factory=FallbackStrategy)
    circuit_breaker_config: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)
    language_processing: LanguageProcessingConfig = field(default_factory=LanguageProcessingConfig)
    enable_monitoring: bool = True
    enable_fallbacks: bool = True
    max_retries: int = 2
    timeout_seconds: int = 300
    store_cleanup: bool = True


@dataclass
class NodeMetrics:
    """Detailed metrics for individual node execution."""
    node_name: str
    start_time: float
    end_time: Optional[float] = None
    duration: float = 0.0
    retry_count: int = 0
    status: str = "pending"  # pending, running, success, failed, skipped
    memory_before: Optional[int] = None
    memory_after: Optional[int] = None
    memory_peak: Optional[int] = None
    error_count: int = 0
    fallback_used: bool = False
    circuit_breaker_state: str = "closed"
    phase_durations: Dict[str, float] = field(default_factory=dict)  # prep, exec, post
    performance_score: Optional[float] = None
    

@dataclass 
class WorkflowMetrics:
    """Comprehensive metrics tracking for workflow execution."""
    start_time: float
    end_time: Optional[float] = None
    total_duration: float = 0.0
    
    # Node-level metrics
    node_metrics: Dict[str, NodeMetrics] = field(default_factory=dict)
    node_durations: Dict[str, float] = field(default_factory=dict)  # Legacy compatibility
    node_retry_counts: Dict[str, int] = field(default_factory=dict)  # Legacy compatibility
    
    # Overall workflow metrics
    success_rate: float = 0.0
    completion_rate: float = 0.0  # Percentage of nodes completed
    error_rate: float = 0.0
    fallback_usage_rate: float = 0.0
    
    # Performance metrics
    throughput: float = 0.0  # nodes per second
    efficiency_score: float = 0.0  # overall efficiency rating
    
    # Resource usage
    memory_usage: Dict[str, Any] = field(default_factory=dict)
    peak_memory: Optional[int] = None
    memory_efficiency: Optional[float] = None
    
    # Error and reliability metrics
    total_errors: int = 0
    circuit_breaker_trips: int = 0
    fallback_attempts: int = 0
    
    # Timing breakdown
    setup_duration: float = 0.0
    execution_duration: float = 0.0
    cleanup_duration: float = 0.0
    
    # Quality metrics
    data_quality_score: Optional[float] = None
    result_completeness: float = 0.0


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
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.fallback_attempts: Dict[str, int] = {}
        
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
        
        # Setup monitoring
        if self.enable_monitoring:
            self._setup_monitoring()
        
        logger.info(f"YouTubeSummarizerFlow initialized with {len(self.nodes)} nodes (config: {self.config.execution_mode.value})")
    
    def _detect_video_language(self, video_metadata: Dict[str, Any], transcript_text: Optional[str] = None) -> LanguageDetectionResult:
        """
        Detect the language of the video using metadata and transcript.
        
        Args:
            video_metadata: Video metadata from YouTube
            transcript_text: Optional transcript text for analysis
            
        Returns:
            LanguageDetectionResult with detection information
        """
        if not self.language_detector:
            # Return default English result if language detection is disabled
            return LanguageDetectionResult(
                detected_language=LanguageCode.ENGLISH,
                confidence_score=0.5,
                detection_method="default_fallback"
            )
        
        try:
            logger.info("Starting video language detection")
            
            # Perform comprehensive language detection
            detection_result = self.language_detector.detect_language_comprehensive(
                video_metadata, transcript_text
            )
            
            # Cache the result for use by other nodes
            self.detected_language_result = detection_result
            
            logger.info(f"Language detected: {detection_result.detected_language.value} "
                       f"(confidence: {detection_result.confidence_score:.3f}, "
                       f"method: {detection_result.detection_method})")
            
            return detection_result
            
        except Exception as e:
            logger.warning(f"Language detection failed: {str(e)}, using default")
            # Return default language on failure
            fallback_result = LanguageDetectionResult(
                detected_language=LanguageCode.ENGLISH if self.config.language_processing.default_language == "en" else LanguageCode.CHINESE_GENERIC,
                confidence_score=0.1,
                detection_method="error_fallback"
            )
            self.detected_language_result = fallback_result
            return fallback_result
    
    def _apply_language_specific_processing(self, text: str, task_type: str = "summarization") -> Dict[str, str]:
        """
        Apply language-specific processing optimizations.
        
        Args:
            text: Text content to process
            task_type: Type of task (summarization, keywords, timestamps)
            
        Returns:
            Dictionary with optimized prompts and content
        """
        if not self.detected_language_result:
            # No language detection available, return basic processing
            return {
                'system_prompt': f"Process the following content for {task_type}:",
                'user_prompt': text,
                'optimized_text': text
            }
        
        detected_lang = self.detected_language_result.detected_language
        
        # Check if this is Chinese content and optimization is enabled
        if (detected_lang in [LanguageCode.CHINESE_SIMPLIFIED, LanguageCode.CHINESE_TRADITIONAL, LanguageCode.CHINESE_GENERIC] and
            self.config.language_processing.enable_chinese_optimization):
            
            logger.debug(f"Applying Chinese optimization for {task_type}")
            
            # Ensure proper encoding for Chinese content
            if self.config.language_processing.preserve_chinese_encoding:
                text = ensure_chinese_encoding(text)
            
            # Apply Chinese-specific optimization
            return optimize_chinese_content_for_llm(text, task_type)
        
        else:
            # English or other language - use standard processing
            logger.debug(f"Applying standard processing for {detected_lang.value}")
            return {
                'system_prompt': f"Process the following content for {task_type}. Provide a clear and comprehensive response.",
                'user_prompt': text,
                'optimized_text': text
            }
    
    def _get_preferred_transcript_languages_for_video(self) -> List[str]:
        """
        Get preferred transcript languages based on detected video language.
        
        Returns:
            List of language codes in order of preference
        """
        if not self.detected_language_result:
            return self.config.language_processing.preferred_languages
        
        if self.config.language_processing.enable_transcript_language_preference:
            # Use language detection result to determine transcript preferences
            return get_preferred_transcript_languages(
                {'language': self.detected_language_result.detected_language.value}
            )
        else:
            return self.config.language_processing.preferred_languages
    
    def _handle_mixed_language_content(self, text: str) -> Dict[str, Any]:
        """
        Handle mixed-language content based on configuration.
        
        Args:
            text: Text content that may contain multiple languages
            
        Returns:
            Dictionary with processing strategy and segmented content if applicable
        """
        if not self.detected_language_result:
            return {'strategy': 'single', 'content': text}
        
        strategy = self.config.language_processing.mixed_language_handling
        
        # Check if we have mixed-language analysis available
        mixed_analysis = None
        language_result = self.store.get('language_detection_result', {})
        if language_result and 'mixed_language_analysis' in language_result:
            mixed_analysis = language_result['mixed_language_analysis']
        
        if strategy == "primary":
            # Use primary detected language for entire content
            primary_language = self.detected_language_result.detected_language.value
            
            # If we have mixed-language analysis, use its primary language
            if mixed_analysis and mixed_analysis.get('primary_language'):
                primary_language = mixed_analysis['primary_language']
            
            return {
                'strategy': 'primary',
                'content': text,
                'primary_language': primary_language,
                'is_mixed': mixed_analysis.get('is_mixed', False) if mixed_analysis else False
            }
        
        elif strategy == "segment":
            # Segment by language using the new functions
            logger.debug("Applying language segmentation")
            
            if mixed_analysis and mixed_analysis.get('is_mixed', False):
                # Perform actual segmentation
                segmentation_result = segment_mixed_language_content(text)
                
                return {
                    'strategy': 'segment',
                    'content': text,
                    'segmentation_result': segmentation_result,
                    'segments': segmentation_result.get('segments', []),
                    'is_mixed': True,
                    'primary_language': segmentation_result.get('primary_language', 'en')
                }
            else:
                # Not mixed content, treat as single language
                return {
                    'strategy': 'primary',
                    'content': text,
                    'primary_language': self.detected_language_result.detected_language.value,
                    'is_mixed': False
                }
        
        elif strategy == "dual":
            # Dual-language processing: process in both detected languages
            logger.debug("Applying dual-language processing")
            
            if mixed_analysis and mixed_analysis.get('is_mixed', False):
                # Get language distribution from mixed analysis
                language_distribution = mixed_analysis.get('language_distribution', {})
                
                # Find the top 2 languages
                sorted_languages = sorted(
                    language_distribution.items(), 
                    key=lambda x: x[1].get('percentage', 0), 
                    reverse=True
                )
                
                primary_lang = sorted_languages[0][0] if sorted_languages else 'en'
                secondary_lang = sorted_languages[1][0] if len(sorted_languages) > 1 else None
                
                return {
                    'strategy': 'dual',
                    'content': text,
                    'primary_language': primary_lang,
                    'secondary_language': secondary_lang,
                    'language_distribution': language_distribution,
                    'is_mixed': True
                }
            else:
                # Not mixed content, fallback to primary
                return {
                    'strategy': 'primary',
                    'content': text,
                    'primary_language': self.detected_language_result.detected_language.value,
                    'is_mixed': False
                }
        
        else:
            # Default to single language processing
            return {'strategy': 'single', 'content': text}

    def _create_node_metrics(self, node_name: str) -> NodeMetrics:
        """Create metrics tracking for a node."""
        current_time = time.time()
        
        # Get circuit breaker state
        cb_state = "closed"
        if node_name in self.circuit_breakers:
            cb_state = self.circuit_breakers[node_name].state.value
        
        # Get memory usage if available
        memory_before = None
        if self.memory_monitoring_available:
            try:
                memory_info = self.process.memory_info()
                memory_before = memory_info.rss
            except Exception:
                pass
        
        return NodeMetrics(
            node_name=node_name,
            start_time=current_time,
            memory_before=memory_before,
            circuit_breaker_state=cb_state
        )
    
    def _finalize_node_metrics(self, node_metrics: NodeMetrics, node_result: Dict[str, Any]) -> None:
        """Finalize metrics for a completed node."""
        current_time = time.time()
        node_metrics.end_time = current_time
        node_metrics.duration = current_time - node_metrics.start_time
        
        # Update status based on result
        status = node_result.get('status', 'unknown')
        node_metrics.status = status
        
        # Check if fallback was used
        node_metrics.fallback_used = node_result.get('fallback_used', False)
        
        # Get retry count
        exec_result = node_result.get('exec', {})
        node_metrics.retry_count = exec_result.get('retry_count', 0)
        
        # Calculate phase durations if available
        if 'prep' in node_result and 'exec' in node_result and 'post' in node_result:
            prep_time = node_result['prep'].get('prep_timestamp')
            exec_time = node_result['exec'].get('exec_timestamp') 
            post_time = node_result['post'].get('post_timestamp')
            
            if prep_time and exec_time and post_time:
                try:
                    prep_dt = datetime.fromisoformat(prep_time.replace('Z', '+00:00'))
                    exec_dt = datetime.fromisoformat(exec_time.replace('Z', '+00:00'))
                    post_dt = datetime.fromisoformat(post_time.replace('Z', '+00:00'))
                    
                    node_metrics.phase_durations = {
                        'prep': (exec_dt - prep_dt).total_seconds(),
                        'exec': (post_dt - exec_dt).total_seconds(),
                        'post': max(0, node_metrics.duration - (post_dt - prep_dt).total_seconds())
                    }
                except Exception:
                    pass
        
        # Get final memory usage
        if self.memory_monitoring_available:
            try:
                memory_info = self.process.memory_info()
                node_metrics.memory_after = memory_info.rss
                node_metrics.memory_peak = getattr(memory_info, 'peak_wset', memory_info.rss)
            except Exception:
                pass
        
        # Calculate performance score
        node_metrics.performance_score = self._calculate_node_performance_score(node_metrics)
        
        # Update circuit breaker state
        if node_metrics.node_name in self.circuit_breakers:
            node_metrics.circuit_breaker_state = self.circuit_breakers[node_metrics.node_name].state.value
    
    def _calculate_node_performance_score(self, node_metrics: NodeMetrics) -> float:
        """Calculate a performance score for a node (0-100)."""
        try:
            base_score = 100.0
            
            # Penalty for failures
            if node_metrics.status == 'failed':
                base_score = 0.0
            elif node_metrics.status == 'partial_success':
                base_score = 60.0
            
            # Penalty for retries
            retry_penalty = min(30, node_metrics.retry_count * 10)
            base_score -= retry_penalty
            
            # Penalty for fallback usage
            if node_metrics.fallback_used:
                base_score -= 20
            
            # Performance bonus/penalty based on duration
            expected_durations = {
                'YouTubeTranscriptNode': 30.0,
                'SummarizationNode': 60.0,
                'TimestampNode': 45.0,
                'KeywordExtractionNode': 30.0
            }
            
            expected = expected_durations.get(node_metrics.node_name, 45.0)
            if node_metrics.duration <= expected:
                # Bonus for fast execution
                speed_bonus = min(20, (expected - node_metrics.duration) / expected * 20)
                base_score += speed_bonus
            else:
                # Penalty for slow execution
                speed_penalty = min(30, (node_metrics.duration - expected) / expected * 30)
                base_score -= speed_penalty
            
            return max(0.0, min(100.0, base_score))
            
        except Exception:
            return 50.0  # Default score on calculation error
    
    def _calculate_workflow_metrics(self) -> None:
        """Calculate comprehensive workflow metrics."""
        if not self.metrics:
            return
        
        current_time = time.time()
        self.metrics.total_duration = current_time - self.metrics.start_time
        
        # Calculate node-level aggregates
        total_nodes = len(self.config.node_configs)
        completed_nodes = len([m for m in self.metrics.node_metrics.values() if m.status in ['success', 'partial_success']])
        failed_nodes = len([m for m in self.metrics.node_metrics.values() if m.status == 'failed'])
        fallback_nodes = len([m for m in self.metrics.node_metrics.values() if m.fallback_used])
        
        # Basic rates
        self.metrics.completion_rate = (completed_nodes / total_nodes) * 100 if total_nodes > 0 else 0
        self.metrics.success_rate = (len([m for m in self.metrics.node_metrics.values() if m.status == 'success']) / total_nodes) * 100 if total_nodes > 0 else 0
        self.metrics.error_rate = (failed_nodes / total_nodes) * 100 if total_nodes > 0 else 0
        self.metrics.fallback_usage_rate = (fallback_nodes / total_nodes) * 100 if total_nodes > 0 else 0
        
        # Performance metrics
        if self.metrics.execution_duration > 0:
            self.metrics.throughput = completed_nodes / self.metrics.execution_duration
        
        # Calculate efficiency score
        performance_scores = [m.performance_score for m in self.metrics.node_metrics.values() if m.performance_score is not None]
        if performance_scores:
            self.metrics.efficiency_score = sum(performance_scores) / len(performance_scores)
        
        # Memory efficiency
        if self.metrics.peak_memory and self.metrics.memory_usage.get('rss'):
            self.metrics.memory_efficiency = (self.metrics.memory_usage['rss'] / self.metrics.peak_memory) * 100
        
        # Error and reliability metrics
        self.metrics.total_errors = sum(m.error_count for m in self.metrics.node_metrics.values())
        self.metrics.fallback_attempts = sum(self.fallback_attempts.values())
        
        # Circuit breaker trips
        self.metrics.circuit_breaker_trips = sum(
            1 for cb in self.circuit_breakers.values() 
            if cb.state != CircuitBreakerState.CLOSED
        )
        
        # Result completeness
        expected_outputs = ['transcript_data', 'summary_data', 'timestamp_data', 'keyword_data']
        available_outputs = sum(1 for key in expected_outputs if key in self.store and self.store[key])
        self.metrics.result_completeness = (available_outputs / len(expected_outputs)) * 100
        
        # Data quality score (basic implementation)
        self.metrics.data_quality_score = self._calculate_data_quality_score()
    
    def _calculate_data_quality_score(self) -> float:
        """Calculate a data quality score based on available results."""
        try:
            score = 0.0
            max_score = 100.0
            
            # Transcript quality (25 points)
            transcript_data = self.store.get('transcript_data', {})
            if transcript_data:
                transcript_text = transcript_data.get('transcript_text', '')
                word_count = len(transcript_text.split()) if transcript_text else 0
                if word_count > 100:
                    score += 25.0
                elif word_count > 50:
                    score += 15.0
                elif word_count > 0:
                    score += 10.0
            
            # Summary quality (25 points)
            summary_data = self.store.get('summary_data', {})
            if summary_data:
                summary_text = summary_data.get('summary_text', '')
                word_count = len(summary_text.split()) if summary_text else 0
                fallback_used = any(m.fallback_used for m in self.metrics.node_metrics.values() if m.node_name == 'SummarizationNode')
                
                if word_count > 200 and not fallback_used:
                    score += 25.0
                elif word_count > 100:
                    score += 20.0
                elif word_count > 50:
                    score += 15.0
                elif word_count > 0:
                    score += 10.0
            
            # Timestamp quality (25 points)
            timestamp_data = self.store.get('timestamp_data', {})
            if timestamp_data:
                timestamps = timestamp_data.get('timestamps', [])
                if len(timestamps) >= 5:
                    score += 25.0
                elif len(timestamps) >= 3:
                    score += 20.0
                elif len(timestamps) >= 1:
                    score += 15.0
            
            # Keyword quality (25 points)
            keyword_data = self.store.get('keyword_data', {})
            if keyword_data:
                keywords = keyword_data.get('keywords', [])
                if len(keywords) >= 5:
                    score += 25.0
                elif len(keywords) >= 3:
                    score += 20.0
                elif len(keywords) >= 1:
                    score += 15.0
            
            return min(max_score, score)
            
        except Exception:
            return 50.0  # Default score on calculation error
    
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
            fallback_strategy=FallbackStrategy(),
            circuit_breaker_config=CircuitBreakerConfig(),
            language_processing=LanguageProcessingConfig(),
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
    
    def _create_node_instance(self, node_name: str, node_config: NodeConfig) -> Optional['Node']:
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
    
    def _sort_nodes_by_dependencies(self, nodes: List['Node']) -> List['Node']:
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
    
    def _initialize_circuit_breakers(self) -> None:
        """Initialize circuit breakers for all enabled nodes."""
        if not self.config.circuit_breaker_config.enabled:
            logger.info("Circuit breakers disabled in configuration")
            return
        
        for node_name in self.config.node_configs.keys():
            if self.config.node_configs[node_name].enabled:
                self.circuit_breakers[node_name] = CircuitBreaker(
                    node_name=node_name,
                    config=self.config.circuit_breaker_config
                )
                logger.debug(f"Initialized circuit breaker for {node_name}")
        
        logger.info(f"Initialized {len(self.circuit_breakers)} circuit breakers")
    
    def _classify_error_severity(self, error: Exception, node_name: str) -> ErrorSeverity:
        """Classify error severity for appropriate handling."""
        error_type = type(error).__name__
        error_message = str(error).lower()
        
        # Critical system-level errors
        if isinstance(error, (MemoryError, SystemError)):
            return ErrorSeverity.CRITICAL
        
        # High severity - likely requires stopping
        if isinstance(error, (ConnectionError, TimeoutError)):
            return ErrorSeverity.HIGH
        
        # Node-specific error classification
        if node_name == "YouTubeTranscriptNode":
            # Transcript errors are usually high severity since other nodes depend on it
            if "invalid" in error_message or "not found" in error_message:
                return ErrorSeverity.HIGH
            return ErrorSeverity.MEDIUM
        
        elif node_name in ["TimestampNode", "KeywordExtractionNode"]:
            # These are optional nodes, so failures are medium severity
            return ErrorSeverity.MEDIUM
        
        elif node_name == "SummarizationNode":
            # Summary failures are significant but recoverable
            if "api" in error_message or "rate limit" in error_message:
                return ErrorSeverity.HIGH
            return ErrorSeverity.MEDIUM
        
        # Default to medium severity
        return ErrorSeverity.MEDIUM
    
    def _should_attempt_fallback(self, error: WorkflowError, node_name: str) -> bool:
        """Determine if fallback should be attempted for an error."""
        if not self.enable_fallbacks:
            return False
        
        # Check fallback attempt limits
        attempts = self.fallback_attempts.get(node_name, 0)
        if attempts >= self.config.fallback_strategy.max_fallback_attempts:
            logger.warning(f"Max fallback attempts reached for {node_name}")
            return False
        
        # Check error severity
        severity = self._classify_error_severity(Exception(error.message), node_name)
        if severity == ErrorSeverity.CRITICAL:
            return False
        
        # Check circuit breaker state
        if node_name in self.circuit_breakers:
            if not self.circuit_breakers[node_name].can_execute():
                logger.info(f"Circuit breaker open for {node_name}, skipping fallback")
                return False
        
        # Check specific fallback strategy settings
        strategy = self.config.fallback_strategy
        
        if node_name == "YouTubeTranscriptNode" and strategy.enable_transcript_only:
            return True
        elif node_name == "SummarizationNode" and strategy.enable_summary_fallback:
            return True
        elif node_name in ["TimestampNode", "KeywordExtractionNode"] and strategy.enable_partial_results:
            return True
        
        return strategy.enable_retry_with_degraded
    
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
        
        # Initialize workflow state and monitoring
        setup_start_time = time.time()
        if self.enable_monitoring:
            self.metrics = WorkflowMetrics(start_time=setup_start_time)
        
        workflow_start_time = setup_start_time
        
        try:
            # Validate input
            self._validate_input(input_data)
            
            # Initialize store with input data
            self.store = Store()
            self.store.update(input_data)
            
            # Add language preferences if enabled
            if self.config.language_processing.enable_language_detection:
                # Store default language preferences for initial transcript acquisition
                default_preferences = self.config.language_processing.preferred_languages
                self.store['default_language_preferences'] = default_preferences
                logger.debug(f"Stored default language preferences: {default_preferences}")
            
            # Record setup duration
            if self.enable_monitoring and self.metrics:
                self.metrics.setup_duration = time.time() - setup_start_time
            
            # Execute workflow with timeout
            result = self._execute_workflow_with_timeout()
            
            # Calculate final metrics
            if self.enable_monitoring:
                self._calculate_workflow_metrics()
                self._finalize_metrics(success=True)
            
            # Prepare final result
            final_result = self._prepare_final_result(result)
            
            total_duration = time.time() - workflow_start_time
            logger.info(f"Workflow completed successfully in {total_duration:.2f}s")
            
            # Log performance summary
            if self.enable_monitoring and self.metrics:
                logger.info(f"Performance summary: efficiency={self.metrics.efficiency_score:.1f}%, "
                           f"completion={self.metrics.completion_rate:.1f}%, "
                           f"fallback_usage={self.metrics.fallback_usage_rate:.1f}%")
            
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
        """Execute all nodes in the configured sequence with enhanced monitoring."""
        results = {}
        execution_start_time = time.time()
        
        # Update metrics with execution start
        if self.enable_monitoring and self.metrics:
            self.metrics.execution_duration = 0.0  # Will be updated at end
        
        for node in self.nodes:
            node_config = self.config.node_configs.get(node.name)
            
            # Create metrics tracking for this node
            node_metrics = None
            if self.enable_monitoring:
                node_metrics = self._create_node_metrics(node.name)
                self.metrics.node_metrics[node.name] = node_metrics
                node_metrics.status = "running"
            
            try:
                logger.info(f"Executing node: {node.name}")
                
                # Check circuit breaker before execution
                if node.name in self.circuit_breakers:
                    if not self.circuit_breakers[node.name].can_execute():
                        logger.warning(f"Circuit breaker open for {node.name}, skipping execution")
                        
                        # Update metrics for skipped node
                        if node_metrics:
                            node_metrics.status = "skipped_circuit_breaker"
                            node_metrics.end_time = time.time()
                            node_metrics.duration = node_metrics.end_time - node_metrics.start_time
                        
                        results[node.name] = {
                            'status': 'skipped_circuit_breaker',
                            'timestamp': datetime.utcnow().isoformat(),
                            'circuit_breaker_state': self.circuit_breakers[node.name].state.value
                        }
                        
                        # Skip optional nodes, fail required nodes
                        if node_config and node_config.required:
                            raise RuntimeError(f"Required node {node.name} skipped due to circuit breaker")
                        continue
                
                # Validate data flow requirements before execution
                self._validate_data_flow(node.name)
                
                # Execute node with enhanced error handling and monitoring
                node_result = self._execute_single_node_with_fallback(node, node_config)
                results[node.name] = node_result
                
                # Update node metrics
                if node_metrics:
                    self._finalize_node_metrics(node_metrics, node_result)
                    
                    # Update legacy metrics for compatibility
                    self.metrics.node_durations[node.name] = node_metrics.duration
                    self.metrics.node_retry_counts[node.name] = node_metrics.retry_count
                
                # Record success in circuit breaker
                if node.name in self.circuit_breakers and node_result.get('status') == 'success':
                    self.circuit_breakers[node.name].record_success()
                
                # Handle node failure based on configuration
                if node_result.get('status') not in ['success', 'partial_success']:
                    # Record failure in circuit breaker
                    if node.name in self.circuit_breakers:
                        self.circuit_breakers[node.name].record_failure()
                    
                    # Update error count in metrics
                    if node_metrics:
                        node_metrics.error_count += 1
                    
                    # Check if this is a required node
                    if node_config and node_config.required:
                        error_msg = node_result.get('error', 'Unknown node error')
                        raise RuntimeError(f"Required node {node.name} failed: {error_msg}")
                    else:
                        # Optional node failure - log warning and continue
                        logger.warning(f"Optional node {node.name} failed, continuing")
                        continue
                
                # Perform language detection after transcript node completes
                if (node.name == "YouTubeTranscriptNode" and 
                    node_result.get('post', {}).get('post_status') == 'success' and
                    self.config.language_processing.enable_language_detection):
                    
                    video_metadata = self.store.get('video_metadata', {})
                    transcript_data = self.store.get('transcript_data', {})
                    transcript_text = transcript_data.get('transcript_text', '')
                    
                    # Perform language detection
                    language_result = self._detect_video_language(video_metadata, transcript_text)
                    
                    # Perform mixed-language content analysis
                    mixed_language_analysis = None
                    if transcript_text and self.config.language_processing.mixed_language_handling != "disabled":
                        mixed_language_analysis = detect_mixed_language_content(transcript_text)
                        logger.info(f"Mixed-language analysis: is_mixed={mixed_language_analysis['is_mixed']}, "
                                   f"primary={mixed_language_analysis['primary_language']}")
                    
                    # Store language detection result
                    self.store['language_detection_result'] = {
                        'detected_language': language_result.detected_language.value,
                        'confidence_score': language_result.confidence_score,
                        'detection_method': language_result.detection_method,
                        'metadata_language': language_result.metadata_language,
                        'title_language': language_result.title_language,
                        'description_language': language_result.description_language,
                        'transcript_language': language_result.transcript_language,
                        'alternative_languages': [(lang.value, score) for lang, score in language_result.alternative_languages],
                        'detection_timestamp': language_result.detection_timestamp,
                        'is_chinese': language_result.detected_language in [LanguageCode.CHINESE_SIMPLIFIED, LanguageCode.CHINESE_TRADITIONAL, LanguageCode.CHINESE_GENERIC],
                        'is_english': language_result.detected_language == LanguageCode.ENGLISH,
                        'mixed_language_analysis': mixed_language_analysis
                    }
                    
                    logger.info(f"Language detection completed and stored in workflow data")
                
                # Map output data based on configuration
                if node_result.get('post', {}).get('post_status') == 'success':
                    self._map_output_data(node.name, node_result['post'])
                
                logger.info(f"Node {node.name} completed with status: {node_result.get('status')} "
                           f"in {node_metrics.duration:.2f}s" if node_metrics else "")
                
            except Exception as e:
                error_msg = f"Node {node.name} execution failed: {str(e)}"
                logger.error(error_msg)
                
                # Update node metrics for failure
                if node_metrics:
                    node_metrics.status = "failed"
                    node_metrics.error_count += 1
                    node_metrics.end_time = time.time()
                    node_metrics.duration = node_metrics.end_time - node_metrics.start_time
                
                # Record failure in circuit breaker
                if node.name in self.circuit_breakers:
                    self.circuit_breakers[node.name].record_failure()
                
                # Create structured error
                error_info = self._create_enhanced_workflow_error(e, node.name, "execution")
                self.workflow_errors.append(error_info)
                
                # Store partial results
                results[node.name] = {
                    'error': error_info.__dict__,
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
        
        # Update execution duration
        if self.enable_monitoring and self.metrics:
            self.metrics.execution_duration = time.time() - execution_start_time
        
        # Clean up temporary data after all nodes complete
        cleanup_start_time = time.time()
        self._cleanup_store_data()
        
        # Update cleanup duration
        if self.enable_monitoring and self.metrics:
            self.metrics.cleanup_duration = time.time() - cleanup_start_time
        
        return results
    
    def _execute_single_node_with_fallback(self, node: 'Node', node_config: Optional[NodeConfig]) -> Dict[str, Any]:
        """Execute a single node with fallback mechanisms."""
        primary_attempt = True
        fallback_attempt = 0
        
        while True:
            try:
                # Adjust timeout for fallback attempts
                timeout_factor = 1.0
                if not primary_attempt:
                    timeout_factor = self.config.fallback_strategy.fallback_timeout_factor
                
                # Execute node phases
                prep_result = node.prep(self.store)
                exec_result = node.exec(self.store, prep_result)
                post_result = node.post(self.store, prep_result, exec_result)
                
                # Check for success
                if post_result.get('post_status') == 'success':
                    return {
                        'status': 'success',
                        'prep': prep_result,
                        'exec': exec_result,
                        'post': post_result,
                        'primary_attempt': primary_attempt,
                        'fallback_attempts': fallback_attempt
                    }
                else:
                    # Node phases completed but post failed
                    error_msg = post_result.get('error', 'Post-processing failed')
                    raise RuntimeError(error_msg)
                    
            except Exception as e:
                # Create error info for decision making
                error_info = self._create_enhanced_workflow_error(e, node.name, "execution")
                
                # Check if we should attempt fallback
                if primary_attempt and self._should_attempt_fallback(error_info, node.name):
                    logger.info(f"Attempting fallback for {node.name}")
                    primary_attempt = False
                    fallback_attempt += 1
                    
                    # Track fallback attempt
                    self.fallback_attempts[node.name] = fallback_attempt
                    
                    # Try fallback execution
                    try:
                        fallback_result = self._execute_node_fallback(node, error_info)
                        if fallback_result:
                            fallback_result['status'] = 'partial_success'
                            fallback_result['fallback_used'] = True
                            fallback_result['original_error'] = error_info.__dict__
                            return fallback_result
                    except Exception as fallback_error:
                        logger.warning(f"Fallback also failed for {node.name}: {str(fallback_error)}")
                
                # No fallback or fallback failed
                return {
                    'status': 'failed',
                    'error': error_info.__dict__,
                    'primary_attempt': primary_attempt,
                    'fallback_attempts': fallback_attempt,
                    'timestamp': datetime.utcnow().isoformat()
                }
    
    def _execute_node_fallback(self, node: 'Node', original_error: WorkflowError) -> Optional[Dict[str, Any]]:
        """Execute fallback strategy for a failed node."""
        strategy = self.config.fallback_strategy
        
        # Node-specific fallback strategies
        if node.name == "YouTubeTranscriptNode" and strategy.enable_transcript_only:
            return self._fallback_transcript_basic(node)
        elif node.name == "SummarizationNode" and strategy.enable_summary_fallback:
            return self._fallback_summary_simple(node)
        elif node.name in ["TimestampNode", "KeywordExtractionNode"] and strategy.enable_partial_results:
            return self._fallback_optional_node(node)
        
        return None
    
    def _fallback_transcript_basic(self, node: 'Node') -> Optional[Dict[str, Any]]:
        """Basic fallback for transcript node with minimal processing."""
        try:
            # Try with simpler parameters
            logger.info("Attempting basic transcript fallback")
            
            # Simple prep with reduced validation
            prep_result = {'fallback_mode': True, 'prep_status': 'success'}
            
            # Execute with fallback parameters (would need to be implemented in the node)
            # For now, return None to indicate fallback not available
            return None
            
        except Exception as e:
            logger.error(f"Transcript fallback failed: {str(e)}")
            return None
    
    def _fallback_summary_simple(self, node: 'Node') -> Optional[Dict[str, Any]]:
        """Simple fallback for summary node using basic text processing."""
        try:
            logger.info("Attempting simple summary fallback")
            
            # Get transcript data
            transcript_data = self.store.get('transcript_data', {})
            transcript_text = transcript_data.get('transcript_text', '')
            
            if not transcript_text:
                return None
            
            # Create simple summary (first 500 words + last 100 words)
            words = transcript_text.split()
            if len(words) > 600:
                simple_summary = ' '.join(words[:500]) + " ... " + ' '.join(words[-100:])
            else:
                simple_summary = transcript_text
            
            # Create fallback result
            return {
                'prep': {'prep_status': 'success', 'fallback_mode': True},
                'exec': {'exec_status': 'success', 'summary_text': simple_summary, 'fallback_used': True},
                'post': {'post_status': 'success'}
            }
            
        except Exception as e:
            logger.error(f"Summary fallback failed: {str(e)}")
            return None
    
    def _fallback_optional_node(self, node: 'Node') -> Optional[Dict[str, Any]]:
        """Fallback for optional nodes - return empty but successful result."""
        try:
            logger.info(f"Using empty fallback for optional node {node.name}")
            
            return {
                'prep': {'prep_status': 'success', 'fallback_mode': True},
                'exec': {'exec_status': 'success', 'fallback_used': True, 'empty_result': True},
                'post': {'post_status': 'success'}
            }
            
        except Exception as e:
            logger.error(f"Optional node fallback failed: {str(e)}")
            return None
    
    def _create_enhanced_workflow_error(self, error: Exception, node_name: str, phase: str) -> WorkflowError:
        """Create enhanced workflow error with additional context."""
        severity = self._classify_error_severity(error, node_name)
        
        # Determine recovery action
        recovery_action = None
        if severity in [ErrorSeverity.LOW, ErrorSeverity.MEDIUM]:
            if self._should_attempt_fallback(WorkflowError(
                flow_name=self.name,
                error_type=type(error).__name__,
                message=str(error)
            ), node_name):
                recovery_action = "fallback_available"
            else:
                recovery_action = "continue_processing"
        else:
            recovery_action = "stop_processing"
        
        return WorkflowError(
            flow_name=self.name,
            error_type=type(error).__name__,
            message=str(error),
            failed_node=node_name,
            is_recoverable=severity in [ErrorSeverity.LOW, ErrorSeverity.MEDIUM],
            node_phase=phase,
            recovery_action=recovery_action,
            context={
                'severity': severity.value,
                'circuit_breaker_state': self.circuit_breakers.get(node_name, {}).state.value if node_name in self.circuit_breakers else 'none',
                'fallback_attempts': self.fallback_attempts.get(node_name, 0)
            }
        )
    
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
            language_detection_result = self.store.get('language_detection_result', {})
            
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
                    'workflow_version': '1.0',
                    'language_detection': language_detection_result
                },
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Add comprehensive monitoring data if available
            if self.enable_monitoring and self.metrics:
                final_result['performance'] = {
                    # Timing metrics
                    'total_duration': self.metrics.total_duration,
                    'setup_duration': self.metrics.setup_duration,
                    'execution_duration': self.metrics.execution_duration,
                    'cleanup_duration': self.metrics.cleanup_duration,
                    
                    # Legacy compatibility
                    'node_durations': self.metrics.node_durations,
                    'retry_counts': self.metrics.node_retry_counts,
                    
                    # Overall metrics
                    'success_rate': self.metrics.success_rate,
                    'completion_rate': self.metrics.completion_rate,
                    'error_rate': self.metrics.error_rate,
                    'fallback_usage_rate': self.metrics.fallback_usage_rate,
                    'efficiency_score': self.metrics.efficiency_score,
                    'throughput': self.metrics.throughput,
                    
                    # Quality metrics
                    'data_quality_score': self.metrics.data_quality_score,
                    'result_completeness': self.metrics.result_completeness,
                    
                    # Resource usage
                    'memory_usage': self.metrics.memory_usage,
                    'peak_memory': self.metrics.peak_memory,
                    'memory_efficiency': self.metrics.memory_efficiency,
                    
                    # Reliability metrics
                    'total_errors': self.metrics.total_errors,
                    'circuit_breaker_trips': self.metrics.circuit_breaker_trips,
                    'fallback_attempts': self.metrics.fallback_attempts
                }
                
                # Add detailed node metrics
                final_result['node_performance'] = {}
                for node_name, node_metrics in self.metrics.node_metrics.items():
                    final_result['node_performance'][node_name] = {
                        'duration': node_metrics.duration,
                        'status': node_metrics.status,
                        'retry_count': node_metrics.retry_count,
                        'fallback_used': node_metrics.fallback_used,
                        'performance_score': node_metrics.performance_score,
                        'circuit_breaker_state': node_metrics.circuit_breaker_state,
                        'phase_durations': node_metrics.phase_durations,
                        'error_count': node_metrics.error_count
                    }
                    
                    # Add memory metrics if available
                    if node_metrics.memory_before and node_metrics.memory_after:
                        final_result['node_performance'][node_name]['memory_delta'] = (
                            node_metrics.memory_after - node_metrics.memory_before
                        )
                        final_result['node_performance'][node_name]['memory_peak'] = node_metrics.memory_peak
            
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
        """Get comprehensive real-time workflow status and metrics."""
        if not self.metrics:
            return {
                'status': 'not_started',
                'timestamp': datetime.utcnow().isoformat()
            }
        
        current_time = time.time()
        elapsed_time = current_time - self.metrics.start_time
        
        # Calculate current progress
        completed_nodes = len([m for m in self.metrics.node_metrics.values() 
                              if m.status in ['success', 'partial_success', 'failed', 'skipped_circuit_breaker']])
        total_nodes = len(self.config.node_configs)
        progress_percent = (completed_nodes / total_nodes) * 100 if total_nodes > 0 else 0
        
        # Find current node
        current_node = None
        running_nodes = [m for m in self.metrics.node_metrics.values() if m.status == 'running']
        if running_nodes:
            current_node = running_nodes[0].node_name
        elif completed_nodes < total_nodes:
            # Next node to execute
            executed_nodes = set(self.metrics.node_metrics.keys())
            for node in self.nodes:
                if node.name not in executed_nodes:
                    current_node = node.name
                    break
        
        # Calculate performance metrics
        success_nodes = len([m for m in self.metrics.node_metrics.values() if m.status == 'success'])
        failed_nodes = len([m for m in self.metrics.node_metrics.values() if m.status == 'failed'])
        fallback_nodes = len([m for m in self.metrics.node_metrics.values() if m.fallback_used])
        
        # Circuit breaker status
        circuit_breaker_status = {}
        for node_name, cb in self.circuit_breakers.items():
            circuit_breaker_status[node_name] = {
                'state': cb.state.value,
                'failure_count': cb.failure_count,
                'consecutive_failures': cb.consecutive_failures
            }
        
        # Node status summary
        node_status = {}
        for node_name, metrics in self.metrics.node_metrics.items():
            node_status[node_name] = {
                'status': metrics.status,
                'duration': metrics.duration,
                'performance_score': metrics.performance_score,
                'retry_count': metrics.retry_count,
                'fallback_used': metrics.fallback_used,
                'progress_percent': 100 if metrics.end_time else 
                                  ((current_time - metrics.start_time) / 60 * 100) if metrics.status == 'running' else 0
            }
        
        # Resource usage
        resource_usage = {}
        if self.memory_monitoring_available:
            try:
                memory_info = self.process.memory_info()
                resource_usage = {
                    'current_memory': memory_info.rss,
                    'peak_memory': getattr(memory_info, 'peak_wset', memory_info.rss),
                    'memory_percent': self.process.memory_percent()
                }
            except Exception:
                pass
        
        status = {
            # Basic status
            'status': 'running' if not self.metrics.end_time else 'completed',
            'timestamp': datetime.utcnow().isoformat(),
            'elapsed_time': elapsed_time,
            
            # Progress information
            'progress': {
                'completed_nodes': completed_nodes,
                'total_nodes': total_nodes,
                'progress_percent': progress_percent,
                'current_node': current_node,
                'success_nodes': success_nodes,
                'failed_nodes': failed_nodes,
                'fallback_nodes': fallback_nodes
            },
            
            # Performance metrics (calculated in real-time)
            'performance': {
                'efficiency_score': self.metrics.efficiency_score,
                'throughput': completed_nodes / elapsed_time if elapsed_time > 0 else 0,
                'success_rate': (success_nodes / completed_nodes) * 100 if completed_nodes > 0 else 0,
                'error_rate': (failed_nodes / completed_nodes) * 100 if completed_nodes > 0 else 0,
                'fallback_usage_rate': (fallback_nodes / completed_nodes) * 100 if completed_nodes > 0 else 0
            },
            
            # Detailed node status
            'nodes': node_status,
            
            # Circuit breaker status
            'circuit_breakers': circuit_breaker_status,
            
            # Resource usage
            'resources': resource_usage,
            
            # Error summary
            'errors': {
                'total_errors': len(self.workflow_errors),
                'total_retries': sum(self.metrics.node_retry_counts.values()),
                'total_fallback_attempts': sum(self.fallback_attempts.values()),
                'recent_errors': [err.__dict__ for err in self.workflow_errors[-3:]]  # Last 3 errors
            },
            
            # Timing breakdown
            'timing': {
                'setup_duration': self.metrics.setup_duration,
                'execution_duration': current_time - (self.metrics.start_time + self.metrics.setup_duration) if self.metrics.setup_duration > 0 else 0,
                'estimated_remaining': self._estimate_remaining_time(elapsed_time, progress_percent)
            }
        }
        
        return status
    
    def _estimate_remaining_time(self, elapsed_time: float, progress_percent: float) -> Optional[float]:
        """Estimate remaining execution time based on current progress."""
        try:
            if progress_percent <= 0:
                return None
            
            # Simple linear estimation
            total_estimated_time = elapsed_time / (progress_percent / 100)
            remaining_time = total_estimated_time - elapsed_time
            
            return max(0, remaining_time)
            
        except Exception:
            return None
    
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
    
    # Create fallback strategy
    fallback_strategy = FallbackStrategy(
        enable_transcript_only=kwargs.get('enable_transcript_only', True),
        enable_summary_fallback=kwargs.get('enable_summary_fallback', True),
        enable_partial_results=kwargs.get('enable_partial_results', True),
        enable_retry_with_degraded=kwargs.get('enable_retry_with_degraded', True),
        max_fallback_attempts=kwargs.get('max_fallback_attempts', 2),
        fallback_timeout_factor=kwargs.get('fallback_timeout_factor', 0.5)
    )
    
    # Create circuit breaker config
    circuit_breaker_config = CircuitBreakerConfig(
        failure_threshold=kwargs.get('failure_threshold', 3),
        recovery_timeout=kwargs.get('recovery_timeout', 60),
        success_threshold=kwargs.get('success_threshold', 2),
        enabled=kwargs.get('enable_circuit_breaker', True)
    )
    
    # Create language processing config with user overrides
    language_processing_config = LanguageProcessingConfig(
        enable_language_detection=kwargs.get('enable_language_detection', True),
        enable_chinese_optimization=kwargs.get('enable_chinese_optimization', True),
        default_language=kwargs.get('default_language', 'en'),
        preferred_languages=kwargs.get('preferred_languages', ["en", "zh-CN", "zh-TW"]),
        chinese_prompt_optimization=kwargs.get('chinese_prompt_optimization', True),
        language_confidence_threshold=kwargs.get('language_confidence_threshold', 0.5),
        mixed_language_handling=kwargs.get('mixed_language_handling', 'primary'),
        preserve_chinese_encoding=kwargs.get('preserve_chinese_encoding', True),
        enable_transcript_language_preference=kwargs.get('enable_transcript_language_preference', True)
    )

    # Create workflow config
    workflow_config = WorkflowConfig(
        execution_mode=execution_mode,
        node_configs=node_config_objects,
        data_flow_config=data_flow_config,
        fallback_strategy=fallback_strategy,
        circuit_breaker_config=circuit_breaker_config,
        language_processing=language_processing_config,
        enable_monitoring=kwargs.get('enable_monitoring', True),
        enable_fallbacks=kwargs.get('enable_fallbacks', True),
        max_retries=kwargs.get('max_retries', 2),
        timeout_seconds=kwargs.get('timeout_seconds', 300),
        store_cleanup=kwargs.get('store_cleanup', True)
    )
    
    logger.info(f"Created custom workflow config with {len(node_config_objects)} nodes")
    return workflow_config


def create_chinese_optimized_config(**kwargs) -> WorkflowConfig:
    """
    Create a workflow configuration optimized for Chinese video content.
    
    Args:
        **kwargs: Additional configuration parameters
        
    Returns:
        WorkflowConfig optimized for Chinese content processing
    """
    # Chinese-specific defaults
    chinese_defaults = {
        'enable_language_detection': True,
        'enable_chinese_optimization': True,
        'default_language': 'zh-CN',
        'preferred_languages': ['zh-CN', 'zh-TW', 'zh', 'en'],
        'chinese_prompt_optimization': True,
        'language_confidence_threshold': 0.3,  # Lower threshold for better Chinese detection
        'mixed_language_handling': 'primary',
        'preserve_chinese_encoding': True,
        'enable_transcript_language_preference': True
    }
    
    # Merge with user overrides
    config_params = {**chinese_defaults, **kwargs}
    
    return create_custom_workflow_config(**config_params)


def create_english_optimized_config(**kwargs) -> WorkflowConfig:
    """
    Create a workflow configuration optimized for English video content.
    
    Args:
        **kwargs: Additional configuration parameters
        
    Returns:
        WorkflowConfig optimized for English content processing
    """
    # English-specific defaults
    english_defaults = {
        'enable_language_detection': True,
        'enable_chinese_optimization': False,
        'default_language': 'en',
        'preferred_languages': ['en', 'en-US', 'en-GB', 'en-CA'],
        'chinese_prompt_optimization': False,
        'language_confidence_threshold': 0.5,
        'mixed_language_handling': 'primary',
        'preserve_chinese_encoding': False,
        'enable_transcript_language_preference': True
    }
    
    # Merge with user overrides
    config_params = {**english_defaults, **kwargs}
    
    return create_custom_workflow_config(**config_params)


def create_multilingual_config(**kwargs) -> WorkflowConfig:
    """
    Create a workflow configuration optimized for multilingual video content.
    
    Args:
        **kwargs: Additional configuration parameters
        
    Returns:
        WorkflowConfig optimized for mixed-language content processing
    """
    # Multilingual-specific defaults
    multilingual_defaults = {
        'enable_language_detection': True,
        'enable_chinese_optimization': True,
        'default_language': 'en',
        'preferred_languages': ['en', 'zh-CN', 'zh-TW', 'zh', 'ja', 'ko', 'es', 'fr'],
        'chinese_prompt_optimization': True,
        'language_confidence_threshold': 0.4,
        'mixed_language_handling': 'segment',  # Use segmentation for mixed content
        'preserve_chinese_encoding': True,
        'enable_transcript_language_preference': True
    }
    
    # Merge with user overrides
    config_params = {**multilingual_defaults, **kwargs}
    
    return create_custom_workflow_config(**config_params)


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