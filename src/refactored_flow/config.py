"""
Configuration classes and management for workflow orchestration.

This module contains all configuration classes used to set up and control
the YouTube summarization workflow execution.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


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
class FallbackStrategy:
    """Configuration for fallback strategies."""
    enable_transcript_only: bool = True      # Fallback to transcript only
    enable_summary_fallback: bool = True     # Use simpler summary if AI fails
    enable_partial_results: bool = True      # Return partial results on failure


@dataclass
class DuplicateHandlingConfig:
    """Configuration for duplicate video detection and handling."""
    enable_duplicate_detection: bool = True
    reprocess_policy: str = "never"  # "never", "always", "if_failed"
    cache_expiry_hours: int = 24  # Hours after which to consider reprocessing
    allow_policy_override: bool = True  # Allow per-request policy override
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
    duplicate_handling: DuplicateHandlingConfig = field(default_factory=DuplicateHandlingConfig)
    enable_monitoring: bool = True
    enable_fallbacks: bool = True
    max_retries: int = 2
    timeout_seconds: int = 300
    store_cleanup: bool = True

    def create_default_node_configs(self) -> Dict[str, NodeConfig]:
        """Create default node configurations for all workflow nodes."""
        default_configs = {
            'YouTubeDataNode': NodeConfig(
                name='YouTubeDataNode',
                enabled=True,
                required=True,
                max_retries=3,
                retry_delay=1.0,
                timeout_seconds=60,
                output_keys=['transcript_data', 'video_metadata', 'youtube_data']
            ),
            'YouTubeTranscriptNode': NodeConfig(
                name='YouTubeTranscriptNode',
                enabled=False,  # Disabled in favor of YouTubeDataNode
                required=True,
                max_retries=3,
                retry_delay=1.0,
                timeout_seconds=60,
                output_keys=['transcript_data', 'video_metadata']
            ),
            'SummarizationNode': NodeConfig(
                name='SummarizationNode',
                enabled=True,
                required=False,
                max_retries=3,
                retry_delay=2.0,
                timeout_seconds=120,
                dependencies=['YouTubeDataNode'],  # Updated dependency
                output_keys=['summary_data']
            ),
            'TimestampNode': NodeConfig(
                name='TimestampNode',
                enabled=False,  # Disabled in favor of EnhancedTimestampNode
                required=False,
                max_retries=3,
                retry_delay=2.0,
                timeout_seconds=90,
                dependencies=['YouTubeDataNode'],  # Updated dependency
                output_keys=['timestamp_data']
            ),
            'SemanticAnalysisNode': NodeConfig(
                name='SemanticAnalysisNode',
                enabled=True,
                required=False,
                max_retries=3,
                retry_delay=2.0,
                timeout_seconds=180,  # Longer timeout for semantic analysis
                dependencies=['YouTubeDataNode'],
                output_keys=['semantic_analysis_result', 'semantic_clusters', 'semantic_keywords', 'semantic_metrics']
            ),
            'VectorSearchNode': NodeConfig(
                name='VectorSearchNode',
                enabled=True,
                required=False,
                max_retries=3,
                retry_delay=2.0,
                timeout_seconds=120,
                dependencies=['SemanticAnalysisNode'],
                output_keys=['vector_search_result', 'coherence_analysis', 'optimized_timestamps']
            ),
            'EnhancedTimestampNode': NodeConfig(
                name='EnhancedTimestampNode',
                enabled=True,
                required=False,
                max_retries=3,
                retry_delay=2.0,
                timeout_seconds=120,
                dependencies=['SemanticAnalysisNode'],  # Can work with or without VectorSearchNode
                output_keys=['enhanced_timestamps', 'enhanced_timestamp_metadata']
            ),
            'KeywordExtractionNode': NodeConfig(
                name='KeywordExtractionNode',
                enabled=True,
                required=False,
                max_retries=3,
                retry_delay=1.5,
                timeout_seconds=60,
                dependencies=['YouTubeDataNode'],  # Can benefit from semantic analysis but not required
                output_keys=['keywords_data']
            )
        }
        
        # Merge with any existing configs, preferring existing over defaults
        for name, default_config in default_configs.items():
            if name not in self.node_configs:
                self.node_configs[name] = default_config
        
        return self.node_configs

    def get_node_config(self, node_name: str) -> Optional[NodeConfig]:
        """Get configuration for a specific node."""
        return self.node_configs.get(node_name)

    def update_node_config(self, node_name: str, **kwargs) -> None:
        """Update configuration for a specific node."""
        if node_name in self.node_configs:
            config = self.node_configs[node_name]
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)
        else:
            # Create new config with provided values
            self.node_configs[node_name] = NodeConfig(name=node_name, **kwargs)

    def disable_node(self, node_name: str) -> None:
        """Disable a specific node."""
        self.update_node_config(node_name, enabled=False)

    def enable_node(self, node_name: str) -> None:
        """Enable a specific node."""
        self.update_node_config(node_name, enabled=True)

    def set_node_optional(self, node_name: str) -> None:
        """Make a node optional (not required for workflow success)."""
        self.update_node_config(node_name, required=False)

    def set_node_required(self, node_name: str) -> None:
        """Make a node required for workflow success."""
        self.update_node_config(node_name, required=True)

    def validate_config(self) -> List[str]:
        """Validate the workflow configuration and return any issues."""
        issues = []
        
        # Check for circular dependencies
        for node_name, config in self.node_configs.items():
            if self._has_circular_dependency(node_name, config.dependencies):
                issues.append(f"Circular dependency detected for node: {node_name}")
        
        # Check that dependencies exist
        all_node_names = set(self.node_configs.keys())
        for node_name, config in self.node_configs.items():
            for dep in config.dependencies:
                if dep not in all_node_names:
                    issues.append(f"Node {node_name} depends on non-existent node: {dep}")
        
        # Check timeout values
        if self.timeout_seconds <= 0:
            issues.append("Workflow timeout must be positive")
        
        for node_name, config in self.node_configs.items():
            if config.timeout_seconds <= 0:
                issues.append(f"Node {node_name} timeout must be positive")
            if config.timeout_seconds > self.timeout_seconds:
                issues.append(f"Node {node_name} timeout exceeds workflow timeout")
        
        return issues

    def _has_circular_dependency(self, node_name: str, dependencies: List[str], visited: Optional[set] = None) -> bool:
        """Check for circular dependencies recursively."""
        if visited is None:
            visited = set()
        
        if node_name in visited:
            return True
        
        visited.add(node_name)
        
        for dep in dependencies:
            if dep in self.node_configs:
                dep_config = self.node_configs[dep]
                if self._has_circular_dependency(dep, dep_config.dependencies, visited.copy()):
                    return True
        
        return False

    def get_execution_order(self) -> List[str]:
        """Get the execution order of enabled nodes based on dependencies."""
        enabled_nodes = {name: config for name, config in self.node_configs.items() if config.enabled}
        
        # If no nodes are enabled, return default order
        if not enabled_nodes:
            return ['YouTubeDataNode', 'SummarizationNode', 'EnhancedTimestampNode', 'KeywordExtractionNode']
        
        # Simple topological sort for dependency resolution
        ordered_nodes = []
        remaining_nodes = set(enabled_nodes.keys())
        
        while remaining_nodes:
            # Find nodes with no unresolved dependencies
            ready_nodes = []
            for node_name in remaining_nodes:
                config = enabled_nodes[node_name]
                unresolved_deps = [dep for dep in config.dependencies if dep in remaining_nodes]
                if not unresolved_deps:
                    ready_nodes.append(node_name)
            
            if not ready_nodes:
                # If no nodes are ready, we have a circular dependency
                # Add remaining nodes in default order to break the cycle
                ready_nodes = list(remaining_nodes)
            
            # Sort ready nodes by priority (YouTubeDataNode first, etc.)
            priority_order = ['YouTubeDataNode', 'SummarizationNode', 'SemanticAnalysisNode', 
                            'VectorSearchNode', 'EnhancedTimestampNode', 'KeywordExtractionNode', 'TimestampNode']
            
            ready_nodes.sort(key=lambda x: priority_order.index(x) if x in priority_order else len(priority_order))
            
            # Add the first ready node to execution order
            if ready_nodes:
                node_to_add = ready_nodes[0]
                ordered_nodes.append(node_to_add)
                remaining_nodes.remove(node_to_add)
        
        return ordered_nodes

    def get_node_config(self, node_name: str) -> Optional[NodeConfig]:
        """Get configuration for a specific node."""
        return self.node_configs.get(node_name)


def create_default_workflow_config() -> WorkflowConfig:
    """Create a default workflow configuration with sensible defaults."""
    config = WorkflowConfig()
    config.create_default_node_configs()
    return config


def create_fast_workflow_config() -> WorkflowConfig:
    """Create a workflow configuration optimized for speed."""
    config = WorkflowConfig(
        enable_monitoring=False,
        enable_fallbacks=False,
        max_retries=1,
        timeout_seconds=180
    )
    
    # Reduce timeouts and retries for all nodes
    config.create_default_node_configs()
    for node_config in config.node_configs.values():
        node_config.max_retries = 1
        node_config.retry_delay = 0.5
        node_config.timeout_seconds = min(30, node_config.timeout_seconds)
    
    return config


def create_robust_workflow_config() -> WorkflowConfig:
    """Create a workflow configuration optimized for reliability."""
    config = WorkflowConfig(
        enable_monitoring=True,
        enable_fallbacks=True,
        max_retries=3,
        timeout_seconds=600
    )
    
    # Increase timeouts and retries for all nodes
    config.create_default_node_configs()
    for node_config in config.node_configs.values():
        node_config.max_retries = 5
        node_config.retry_delay = 2.0
        node_config.timeout_seconds = min(180, node_config.timeout_seconds * 2)
    
    # Enhanced fallback strategy
    config.fallback_strategy = FallbackStrategy(
        enable_transcript_only=True,
        enable_summary_fallback=True,
        enable_partial_results=True,
        enable_retry_with_degraded=True,
        max_fallback_attempts=3,
        fallback_timeout_factor=0.7
    )
    
    return config


def create_minimal_workflow_config() -> WorkflowConfig:
    """Create a minimal workflow configuration with only essential features."""
    config = WorkflowConfig(
        enable_monitoring=False,
        enable_fallbacks=False,
        max_retries=1,
        timeout_seconds=120
    )
    
    config.create_default_node_configs()
    
    # Disable optional nodes
    config.disable_node('TimestampNode')
    config.disable_node('KeywordExtractionNode')
    
    # Make remaining nodes more lenient
    config.set_node_optional('SummarizationNode')
    
    return config