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
    enable_monitoring: bool = True
    enable_fallbacks: bool = True
    max_retries: int = 2
    timeout_seconds: int = 300
    store_cleanup: bool = True

    def create_default_node_configs(self) -> Dict[str, NodeConfig]:
        """Create default node configurations for all workflow nodes."""
        default_configs = {
            'YouTubeTranscriptNode': NodeConfig(
                name='YouTubeTranscriptNode',
                enabled=True,
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
                dependencies=['YouTubeTranscriptNode'],
                output_keys=['summary_data']
            ),
            'TimestampNode': NodeConfig(
                name='TimestampNode',
                enabled=True,
                required=False,
                max_retries=3,
                retry_delay=2.0,
                timeout_seconds=90,
                dependencies=['YouTubeTranscriptNode'],
                output_keys=['timestamp_data']
            ),
            'KeywordExtractionNode': NodeConfig(
                name='KeywordExtractionNode',
                enabled=True,
                required=False,
                max_retries=3,
                retry_delay=1.5,
                timeout_seconds=60,
                dependencies=['YouTubeTranscriptNode'],
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