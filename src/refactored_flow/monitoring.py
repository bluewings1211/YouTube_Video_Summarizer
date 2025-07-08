"""
Performance monitoring and metrics collection for workflow orchestration.

This module provides comprehensive monitoring and metrics tracking for
workflow execution, including performance, resource usage, and quality metrics.
"""

import logging
import time
import psutil
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


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

    def start_phase(self, phase: str) -> None:
        """Start timing a specific phase of node execution."""
        if phase not in self.phase_durations:
            self.phase_durations[phase] = time.time()

    def end_phase(self, phase: str) -> None:
        """End timing a specific phase of node execution."""
        if phase in self.phase_durations:
            start_time = self.phase_durations[phase]
            self.phase_durations[phase] = time.time() - start_time

    def finish(self, status: str) -> None:
        """Mark the node execution as finished."""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        self.status = status
        
        # Calculate performance score
        self.performance_score = self._calculate_performance_score()

    def _calculate_performance_score(self) -> float:
        """Calculate a performance score for this node execution."""
        score = 100.0
        
        # Penalty for retries
        score -= self.retry_count * 10
        
        # Penalty for errors
        score -= self.error_count * 15
        
        # Penalty for fallback usage
        if self.fallback_used:
            score -= 20
        
        # Penalty for long duration (relative scoring)
        if self.duration > 60:  # More than 1 minute
            score -= min(30, (self.duration - 60) / 60 * 5)
        
        # Bonus for fast execution
        if self.duration < 10:  # Less than 10 seconds
            score += 10
        
        return max(0.0, min(100.0, score))

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for serialization."""
        return {
            'node_name': self.node_name,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration': self.duration,
            'retry_count': self.retry_count,
            'status': self.status,
            'memory_before': self.memory_before,
            'memory_after': self.memory_after,
            'memory_peak': self.memory_peak,
            'error_count': self.error_count,
            'fallback_used': self.fallback_used,
            'circuit_breaker_state': self.circuit_breaker_state,
            'phase_durations': self.phase_durations,
            'performance_score': self.performance_score
        }


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

    def start_node(self, node_name: str) -> NodeMetrics:
        """Start tracking metrics for a node."""
        node_metrics = NodeMetrics(
            node_name=node_name,
            start_time=time.time(),
            memory_before=self._get_current_memory()
        )
        self.node_metrics[node_name] = node_metrics
        return node_metrics

    def finish_node(self, node_name: str, status: str) -> None:
        """Finish tracking metrics for a node."""
        if node_name in self.node_metrics:
            node_metrics = self.node_metrics[node_name]
            node_metrics.memory_after = self._get_current_memory()
            node_metrics.finish(status)
            
            # Update legacy compatibility fields
            self.node_durations[node_name] = node_metrics.duration
            self.node_retry_counts[node_name] = node_metrics.retry_count

    def record_node_error(self, node_name: str) -> None:
        """Record an error for a node."""
        if node_name in self.node_metrics:
            self.node_metrics[node_name].error_count += 1
        self.total_errors += 1

    def record_fallback_usage(self, node_name: str) -> None:
        """Record fallback usage for a node."""
        if node_name in self.node_metrics:
            self.node_metrics[node_name].fallback_used = True
        self.fallback_attempts += 1

    def record_circuit_breaker_trip(self) -> None:
        """Record a circuit breaker trip."""
        self.circuit_breaker_trips += 1

    def finish_workflow(self, success: bool = True) -> None:
        """Finish tracking workflow metrics."""
        self.end_time = time.time()
        self.total_duration = self.end_time - self.start_time
        
        # Calculate derived metrics
        self._calculate_completion_metrics()
        self._calculate_performance_metrics()
        self._calculate_resource_metrics()

    def _calculate_completion_metrics(self) -> None:
        """Calculate completion and success rates."""
        if not self.node_metrics:
            return
        
        completed_nodes = sum(1 for metrics in self.node_metrics.values() 
                             if metrics.status in ['success', 'failed'])
        successful_nodes = sum(1 for metrics in self.node_metrics.values() 
                              if metrics.status == 'success')
        
        total_nodes = len(self.node_metrics)
        
        self.completion_rate = (completed_nodes / total_nodes) * 100 if total_nodes > 0 else 0
        self.success_rate = (successful_nodes / total_nodes) * 100 if total_nodes > 0 else 0
        self.error_rate = (self.total_errors / total_nodes) * 100 if total_nodes > 0 else 0
        self.fallback_usage_rate = (self.fallback_attempts / total_nodes) * 100 if total_nodes > 0 else 0

    def _calculate_performance_metrics(self) -> None:
        """Calculate performance metrics."""
        if not self.node_metrics or self.total_duration <= 0:
            return
        
        # Throughput (nodes per second)
        completed_nodes = sum(1 for metrics in self.node_metrics.values() 
                             if metrics.status in ['success', 'failed'])
        self.throughput = completed_nodes / self.total_duration
        
        # Efficiency score (average of node performance scores)
        scores = [metrics.performance_score for metrics in self.node_metrics.values() 
                 if metrics.performance_score is not None]
        self.efficiency_score = sum(scores) / len(scores) if scores else 0

    def _calculate_resource_metrics(self) -> None:
        """Calculate resource usage metrics."""
        memory_values = [metrics.memory_after for metrics in self.node_metrics.values() 
                        if metrics.memory_after is not None]
        
        if memory_values:
            self.peak_memory = max(memory_values)
            
            # Memory efficiency (lower is better)
            initial_memory = min(metrics.memory_before for metrics in self.node_metrics.values() 
                               if metrics.memory_before is not None)
            if initial_memory and self.peak_memory:
                memory_growth = self.peak_memory - initial_memory
                self.memory_efficiency = memory_growth / len(self.node_metrics)

        self.memory_usage = {
            'peak_memory_mb': self.peak_memory // (1024 * 1024) if self.peak_memory else None,
            'memory_efficiency': self.memory_efficiency,
            'memory_per_node': {
                name: {
                    'before_mb': metrics.memory_before // (1024 * 1024) if metrics.memory_before else None,
                    'after_mb': metrics.memory_after // (1024 * 1024) if metrics.memory_after else None
                }
                for name, metrics in self.node_metrics.items()
            }
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of workflow metrics."""
        return {
            'duration': self.total_duration,
            'success_rate': self.success_rate,
            'completion_rate': self.completion_rate,
            'error_rate': self.error_rate,
            'throughput': self.throughput,
            'efficiency_score': self.efficiency_score,
            'total_nodes': len(self.node_metrics),
            'successful_nodes': sum(1 for m in self.node_metrics.values() if m.status == 'success'),
            'failed_nodes': sum(1 for m in self.node_metrics.values() if m.status == 'failed'),
            'total_errors': self.total_errors,
            'fallback_attempts': self.fallback_attempts,
            'circuit_breaker_trips': self.circuit_breaker_trips,
            'peak_memory_mb': self.peak_memory // (1024 * 1024) if self.peak_memory else None
        }

    def get_detailed_report(self) -> Dict[str, Any]:
        """Get a detailed metrics report."""
        return {
            'workflow_summary': self.get_summary(),
            'node_metrics': {name: metrics.to_dict() for name, metrics in self.node_metrics.items()},
            'performance_analysis': self._generate_performance_analysis(),
            'resource_analysis': self.memory_usage,
            'recommendations': self._generate_recommendations()
        }

    def _generate_performance_analysis(self) -> Dict[str, Any]:
        """Generate performance analysis."""
        if not self.node_metrics:
            return {}
        
        durations = [metrics.duration for metrics in self.node_metrics.values()]
        
        return {
            'fastest_node': min(self.node_metrics.items(), key=lambda x: x[1].duration)[0] if durations else None,
            'slowest_node': max(self.node_metrics.items(), key=lambda x: x[1].duration)[0] if durations else None,
            'average_duration': sum(durations) / len(durations) if durations else 0,
            'total_retries': sum(metrics.retry_count for metrics in self.node_metrics.values()),
            'nodes_with_fallback': [name for name, metrics in self.node_metrics.items() if metrics.fallback_used]
        }

    def _generate_recommendations(self) -> List[str]:
        """Generate performance recommendations."""
        recommendations = []
        
        if self.efficiency_score < 50:
            recommendations.append("Low efficiency score - consider optimizing node configurations")
        
        if self.error_rate > 20:
            recommendations.append("High error rate - investigate node failures and add more error handling")
        
        if self.fallback_usage_rate > 30:
            recommendations.append("High fallback usage - consider improving primary processing paths")
        
        if self.throughput < 0.1:  # Less than 1 node per 10 seconds
            recommendations.append("Low throughput - consider reducing node timeouts or improving processing speed")
        
        return recommendations

    def _get_current_memory(self) -> Optional[int]:
        """Get current memory usage in bytes."""
        try:
            process = psutil.Process()
            return process.memory_info().rss
        except Exception:
            return None


class WorkflowMonitor:
    """Central monitoring system for workflow execution."""
    
    def __init__(self, workflow_name: str):
        self.workflow_name = workflow_name
        self.metrics: Optional[WorkflowMetrics] = None
        self.enabled = True
        
    def start_workflow(self) -> WorkflowMetrics:
        """Start monitoring a workflow execution."""
        if not self.enabled:
            return None
        
        self.metrics = WorkflowMetrics(start_time=time.time())
        logger.debug(f"Started monitoring workflow: {self.workflow_name}")
        return self.metrics
    
    def start_node(self, node_name: str) -> Optional[NodeMetrics]:
        """Start monitoring a node execution."""
        if not self.enabled or not self.metrics:
            return None
        
        return self.metrics.start_node(node_name)
    
    def finish_node(self, node_name: str, status: str) -> None:
        """Finish monitoring a node execution."""
        if not self.enabled or not self.metrics:
            return
        
        self.metrics.finish_node(node_name, status)
    
    def record_error(self, node_name: str) -> None:
        """Record an error for monitoring."""
        if not self.enabled or not self.metrics:
            return
        
        self.metrics.record_node_error(node_name)
    
    def record_fallback(self, node_name: str) -> None:
        """Record fallback usage for monitoring."""
        if not self.enabled or not self.metrics:
            return
        
        self.metrics.record_fallback_usage(node_name)
    
    def finish_workflow(self, success: bool = True) -> Optional[Dict[str, Any]]:
        """Finish monitoring and get final metrics."""
        if not self.enabled or not self.metrics:
            return None
        
        self.metrics.finish_workflow(success)
        summary = self.metrics.get_summary()
        
        logger.info(f"Workflow {self.workflow_name} completed: "
                   f"Duration={summary['duration']:.2f}s, "
                   f"Success Rate={summary['success_rate']:.1f}%, "
                   f"Efficiency={summary['efficiency_score']:.1f}")
        
        return summary
    
    def get_current_metrics(self) -> Optional[Dict[str, Any]]:
        """Get current metrics snapshot."""
        if not self.enabled or not self.metrics:
            return None
        
        return self.metrics.get_summary()
    
    def enable(self) -> None:
        """Enable monitoring."""
        self.enabled = True
    
    def disable(self) -> None:
        """Disable monitoring."""
        self.enabled = False