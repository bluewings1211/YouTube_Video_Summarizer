"""
Error handling and circuit breaker logic for workflow orchestration.

This module contains error handling classes, circuit breaker implementation,
and recovery strategies for robust workflow execution.
"""

import logging
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"           # Minor issues, continue processing
    MEDIUM = "medium"     # Significant issues, try fallback
    HIGH = "high"         # Major issues, stop processing
    CRITICAL = "critical" # System-level issues, immediate failure


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Circuit open, blocking requests
    HALF_OPEN = "half_open"  # Testing if service recovered


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
    severity: ErrorSeverity = ErrorSeverity.MEDIUM

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for serialization."""
        return {
            'flow_name': self.flow_name,
            'error_type': self.error_type,
            'message': self.message,
            'failed_node': self.failed_node,
            'timestamp': self.timestamp,
            'is_recoverable': self.is_recoverable,
            'context': self.context,
            'retry_count': self.retry_count,
            'node_phase': self.node_phase,
            'recovery_action': self.recovery_action,
            'severity': self.severity.value
        }

    def should_retry(self, max_retries: int) -> bool:
        """Determine if this error should trigger a retry."""
        return (
            self.is_recoverable and 
            self.retry_count < max_retries and
            self.severity in [ErrorSeverity.LOW, ErrorSeverity.MEDIUM]
        )

    def get_recovery_delay(self) -> float:
        """Get recommended delay before retry based on error severity."""
        delay_map = {
            ErrorSeverity.LOW: 1.0,
            ErrorSeverity.MEDIUM: 2.0,
            ErrorSeverity.HIGH: 5.0,
            ErrorSeverity.CRITICAL: 10.0
        }
        base_delay = delay_map.get(self.severity, 2.0)
        # Exponential backoff based on retry count
        return base_delay * (2 ** min(self.retry_count, 5))


@dataclass
class CircuitBreaker:
    """Circuit breaker implementation for node failures."""
    node_name: str
    config: 'CircuitBreakerConfig'
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
                logger.info(f"Circuit breaker for {self.node_name} moving to HALF_OPEN state")
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
                logger.info(f"Circuit breaker for {self.node_name} closed after successful recovery")
    
    def record_failure(self) -> None:
        """Record failed execution."""
        self.failure_count += 1
        self.consecutive_failures += 1
        self.last_failure_time = time.time()
        
        if (self.state == CircuitBreakerState.CLOSED and 
            self.consecutive_failures >= self.config.failure_threshold):
            self.state = CircuitBreakerState.OPEN
            logger.warning(f"Circuit breaker for {self.node_name} opened after {self.consecutive_failures} consecutive failures")
        elif self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.OPEN
            logger.warning(f"Circuit breaker for {self.node_name} reopened after failure during recovery")

    def get_status(self) -> Dict[str, Any]:
        """Get current circuit breaker status."""
        return {
            'node_name': self.node_name,
            'state': self.state.value,
            'failure_count': self.failure_count,
            'success_count': self.success_count,
            'consecutive_failures': self.consecutive_failures,
            'last_failure_time': self.last_failure_time,
            'can_execute': self.can_execute()
        }


class ErrorHandler:
    """Centralized error handling for workflow execution."""
    
    def __init__(self, flow_name: str):
        self.flow_name = flow_name
        self.errors: List[WorkflowError] = []
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.fallback_attempts: Dict[str, int] = {}
        
    def create_circuit_breaker(self, node_name: str, config: 'CircuitBreakerConfig') -> CircuitBreaker:
        """Create a circuit breaker for a node."""
        circuit_breaker = CircuitBreaker(node_name, config)
        self.circuit_breakers[node_name] = circuit_breaker
        return circuit_breaker
    
    def get_circuit_breaker(self, node_name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker for a node."""
        return self.circuit_breakers.get(node_name)
    
    def handle_node_error(
        self, 
        error: Exception, 
        node_name: str, 
        phase: str = "exec",
        context: Optional[Dict[str, Any]] = None
    ) -> WorkflowError:
        """Handle an error from a node execution."""
        
        # Determine error severity
        severity = self._determine_error_severity(error, node_name)
        
        # Create workflow error
        workflow_error = WorkflowError(
            flow_name=self.flow_name,
            error_type=type(error).__name__,
            message=str(error),
            failed_node=node_name,
            node_phase=phase,
            context=context or {},
            severity=severity,
            is_recoverable=self._is_error_recoverable(error, severity)
        )
        
        # Store error
        self.errors.append(workflow_error)
        
        # Update circuit breaker
        circuit_breaker = self.get_circuit_breaker(node_name)
        if circuit_breaker:
            circuit_breaker.record_failure()
        
        # Log error
        log_level = logging.ERROR if severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL] else logging.WARNING
        logger.log(log_level, f"Node {node_name} failed in {phase} phase: {workflow_error.message}")
        
        return workflow_error
    
    def handle_node_success(self, node_name: str) -> None:
        """Handle successful node execution."""
        circuit_breaker = self.get_circuit_breaker(node_name)
        if circuit_breaker:
            circuit_breaker.record_success()
    
    def can_node_execute(self, node_name: str) -> bool:
        """Check if a node can execute based on circuit breaker state."""
        circuit_breaker = self.get_circuit_breaker(node_name)
        if circuit_breaker:
            return circuit_breaker.can_execute()
        return True
    
    def should_attempt_fallback(self, node_name: str, max_fallback_attempts: int = 2) -> bool:
        """Determine if fallback should be attempted for a node."""
        attempts = self.fallback_attempts.get(node_name, 0)
        return attempts < max_fallback_attempts
    
    def record_fallback_attempt(self, node_name: str) -> None:
        """Record a fallback attempt for a node."""
        self.fallback_attempts[node_name] = self.fallback_attempts.get(node_name, 0) + 1
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of all errors encountered."""
        if not self.errors:
            return {'total_errors': 0, 'error_by_severity': {}, 'error_by_node': {}}
        
        error_by_severity = {}
        error_by_node = {}
        
        for error in self.errors:
            # Count by severity
            severity_key = error.severity.value
            error_by_severity[severity_key] = error_by_severity.get(severity_key, 0) + 1
            
            # Count by node
            if error.failed_node:
                error_by_node[error.failed_node] = error_by_node.get(error.failed_node, 0) + 1
        
        return {
            'total_errors': len(self.errors),
            'error_by_severity': error_by_severity,
            'error_by_node': error_by_node,
            'latest_error': self.errors[-1].to_dict() if self.errors else None,
            'circuit_breaker_status': {name: cb.get_status() for name, cb in self.circuit_breakers.items()}
        }
    
    def get_recovery_recommendations(self) -> List[str]:
        """Get recommendations for recovering from errors."""
        recommendations = []
        
        # Check for patterns in errors
        if len(self.errors) > 5:
            recommendations.append("High error count detected - consider reducing workflow complexity")
        
        # Check circuit breaker states
        open_breakers = [name for name, cb in self.circuit_breakers.items() 
                        if cb.state == CircuitBreakerState.OPEN]
        if open_breakers:
            recommendations.append(f"Circuit breakers open for: {', '.join(open_breakers)} - investigate node health")
        
        # Check for specific error patterns
        error_types = [error.error_type for error in self.errors]
        if error_types.count('TimeoutError') > 2:
            recommendations.append("Multiple timeout errors - consider increasing timeout values")
        
        if error_types.count('ConnectionError') > 2:
            recommendations.append("Multiple connection errors - check network connectivity and service availability")
        
        return recommendations
    
    def _determine_error_severity(self, error: Exception, node_name: str) -> ErrorSeverity:
        """Determine the severity of an error."""
        error_type = type(error).__name__
        error_message = str(error).lower()
        
        # Critical errors that require immediate failure
        if error_type in ['MemoryError', 'SystemError', 'KeyboardInterrupt']:
            return ErrorSeverity.CRITICAL
        
        # High severity errors
        if error_type in ['ValueError', 'TypeError', 'AttributeError'] and 'config' in error_message:
            return ErrorSeverity.HIGH
        
        # Medium severity errors (default for most processing errors)
        if error_type in ['ConnectionError', 'TimeoutError', 'HTTPError']:
            return ErrorSeverity.MEDIUM
        
        # Low severity errors
        if 'rate limit' in error_message or 'temporary' in error_message:
            return ErrorSeverity.LOW
        
        # Default to medium severity
        return ErrorSeverity.MEDIUM
    
    def _is_error_recoverable(self, error: Exception, severity: ErrorSeverity) -> bool:
        """Determine if an error is recoverable."""
        if severity == ErrorSeverity.CRITICAL:
            return False
        
        error_type = type(error).__name__
        error_message = str(error).lower()
        
        # Non-recoverable errors
        non_recoverable_types = ['ValueError', 'TypeError', 'AttributeError']
        if error_type in non_recoverable_types and 'config' not in error_message:
            return False
        
        # Recoverable errors
        recoverable_types = ['ConnectionError', 'TimeoutError', 'HTTPError']
        if error_type in recoverable_types:
            return True
        
        # Check message for recoverable indicators
        recoverable_indicators = ['rate limit', 'temporary', 'timeout', 'connection']
        if any(indicator in error_message for indicator in recoverable_indicators):
            return True
        
        # Default to non-recoverable for unknown errors
        return False