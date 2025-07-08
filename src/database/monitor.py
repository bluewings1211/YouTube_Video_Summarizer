"""
Database monitoring and alerting system.

This module provides monitoring capabilities for database operations,
including error tracking, performance metrics, and health monitoring.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from collections import defaultdict, deque
from threading import Lock

from .exceptions import DatabaseError, DatabaseErrorSeverity, DatabaseErrorCategory


@dataclass
class DatabaseMetrics:
    """Database operation metrics."""
    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    total_response_time: float = 0.0
    error_count_by_category: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    error_count_by_severity: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    recent_errors: deque = field(default_factory=lambda: deque(maxlen=100))
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.total_operations == 0:
            return 100.0
        return (self.successful_operations / self.total_operations) * 100.0
    
    @property
    def average_response_time(self) -> float:
        """Calculate average response time in seconds."""
        if self.successful_operations == 0:
            return 0.0
        return self.total_response_time / self.successful_operations
    
    @property
    def error_rate(self) -> float:
        """Calculate error rate percentage."""
        if self.total_operations == 0:
            return 0.0
        return (self.failed_operations / self.total_operations) * 100.0


@dataclass
class DatabaseAlert:
    """Database alert information."""
    alert_type: str
    severity: str
    message: str
    timestamp: datetime
    context: Dict[str, Any] = field(default_factory=dict)


class DatabaseMonitor:
    """
    Monitor database operations and track performance metrics.
    
    Provides real-time monitoring of database health, error tracking,
    and alerting for critical issues.
    """
    
    def __init__(
        self,
        alert_threshold_error_rate: float = 10.0,
        alert_threshold_response_time: float = 5.0,
        monitoring_window_minutes: int = 10
    ):
        """
        Initialize database monitor.
        
        Args:
            alert_threshold_error_rate: Error rate threshold for alerts (percentage)
            alert_threshold_response_time: Response time threshold for alerts (seconds)
            monitoring_window_minutes: Window for calculating rolling metrics
        """
        self.logger = logging.getLogger(__name__)
        self.metrics = DatabaseMetrics()
        self.alerts: List[DatabaseAlert] = []
        self.alert_threshold_error_rate = alert_threshold_error_rate
        self.alert_threshold_response_time = alert_threshold_response_time
        self.monitoring_window = timedelta(minutes=monitoring_window_minutes)
        self._lock = Lock()
        
        # Time-based operation tracking
        self.operation_history: deque = deque(maxlen=1000)
        
    def record_operation_start(self, operation_name: str) -> str:
        """
        Record the start of a database operation.
        
        Args:
            operation_name: Name of the operation
            
        Returns:
            Operation ID for tracking
        """
        operation_id = f"{operation_name}_{int(time.time() * 1000000)}"
        start_time = time.time()
        
        with self._lock:
            self.operation_history.append({
                'operation_id': operation_id,
                'operation_name': operation_name,
                'start_time': start_time,
                'end_time': None,
                'success': None,
                'error': None,
                'response_time': None
            })
        
        return operation_id
    
    def record_operation_success(self, operation_id: str) -> None:
        """
        Record successful completion of a database operation.
        
        Args:
            operation_id: Operation ID from record_operation_start
        """
        end_time = time.time()
        
        with self._lock:
            # Find and update operation record
            for op in reversed(self.operation_history):
                if op['operation_id'] == operation_id:
                    op['end_time'] = end_time
                    op['success'] = True
                    op['response_time'] = end_time - op['start_time']
                    
                    # Update metrics
                    self.metrics.total_operations += 1
                    self.metrics.successful_operations += 1
                    self.metrics.total_response_time += op['response_time']
                    
                    # Check for performance alerts
                    self._check_performance_alerts(op['response_time'])
                    break
    
    def record_operation_error(self, operation_id: str, error: Exception) -> None:
        """
        Record failed completion of a database operation.
        
        Args:
            operation_id: Operation ID from record_operation_start
            error: Exception that occurred
        """
        end_time = time.time()
        
        # Classify error if it's not already a DatabaseError
        if isinstance(error, DatabaseError):
            db_error = error
        else:
            from .exceptions import classify_database_error
            db_error = classify_database_error(error)
        
        with self._lock:
            # Find and update operation record
            for op in reversed(self.operation_history):
                if op['operation_id'] == operation_id:
                    op['end_time'] = end_time
                    op['success'] = False
                    op['error'] = db_error.to_dict()
                    op['response_time'] = end_time - op['start_time']
                    
                    # Update metrics
                    self.metrics.total_operations += 1
                    self.metrics.failed_operations += 1
                    self.metrics.error_count_by_category[db_error.category.value] += 1
                    self.metrics.error_count_by_severity[db_error.severity.value] += 1
                    self.metrics.recent_errors.append({
                        'timestamp': datetime.utcnow(),
                        'operation_name': op['operation_name'],
                        'error': db_error.to_dict()
                    })
                    
                    # Check for error rate alerts
                    self._check_error_alerts(db_error)
                    break
    
    def _check_performance_alerts(self, response_time: float) -> None:
        """Check if response time exceeds threshold and generate alert."""
        if response_time > self.alert_threshold_response_time:
            alert = DatabaseAlert(
                alert_type="performance",
                severity="warning",
                message=f"Slow database operation: {response_time:.2f}s (threshold: {self.alert_threshold_response_time:.2f}s)",
                timestamp=datetime.utcnow(),
                context={'response_time': response_time, 'threshold': self.alert_threshold_response_time}
            )
            self._add_alert(alert)
    
    def _check_error_alerts(self, error: DatabaseError) -> None:
        """Check error conditions and generate alerts if needed."""
        # High severity errors always generate alerts
        if error.severity in [DatabaseErrorSeverity.HIGH, DatabaseErrorSeverity.CRITICAL]:
            alert = DatabaseAlert(
                alert_type="error",
                severity=error.severity.value,
                message=f"Database error: {error.message}",
                timestamp=datetime.utcnow(),
                context=error.to_dict()
            )
            self._add_alert(alert)
        
        # Check overall error rate
        if self.metrics.error_rate > self.alert_threshold_error_rate:
            alert = DatabaseAlert(
                alert_type="error_rate",
                severity="warning",
                message=f"High database error rate: {self.metrics.error_rate:.1f}% (threshold: {self.alert_threshold_error_rate:.1f}%)",
                timestamp=datetime.utcnow(),
                context={
                    'error_rate': self.metrics.error_rate,
                    'threshold': self.alert_threshold_error_rate,
                    'total_operations': self.metrics.total_operations,
                    'failed_operations': self.metrics.failed_operations
                }
            )
            self._add_alert(alert)
    
    def _add_alert(self, alert: DatabaseAlert) -> None:
        """Add alert and log it."""
        self.alerts.append(alert)
        
        # Log alert based on severity
        if alert.severity == "critical":
            self.logger.critical(f"Database Alert: {alert.message}", extra=alert.context)
        elif alert.severity == "high":
            self.logger.error(f"Database Alert: {alert.message}", extra=alert.context)
        elif alert.severity == "warning":
            self.logger.warning(f"Database Alert: {alert.message}", extra=alert.context)
        else:
            self.logger.info(f"Database Alert: {alert.message}", extra=alert.context)
        
        # Keep only recent alerts (last 100)
        if len(self.alerts) > 100:
            self.alerts = self.alerts[-100:]
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get current database health status.
        
        Returns:
            Dictionary with health metrics and status
        """
        with self._lock:
            # Calculate recent metrics (last monitoring window)
            cutoff_time = datetime.utcnow() - self.monitoring_window
            recent_operations = [
                op for op in self.operation_history
                if op.get('end_time') and datetime.fromtimestamp(op['end_time']) > cutoff_time
            ]
            
            recent_success = len([op for op in recent_operations if op.get('success')])
            recent_total = len(recent_operations)
            recent_error_rate = 0.0 if recent_total == 0 else ((recent_total - recent_success) / recent_total) * 100.0
            
            # Determine overall health status
            health_status = "healthy"
            if recent_error_rate > self.alert_threshold_error_rate * 2:
                health_status = "critical"
            elif recent_error_rate > self.alert_threshold_error_rate:
                health_status = "degraded"
            elif recent_error_rate > 0:
                health_status = "warning"
            
            return {
                'status': health_status,
                'timestamp': datetime.utcnow().isoformat(),
                'metrics': {
                    'total_operations': self.metrics.total_operations,
                    'success_rate': self.metrics.success_rate,
                    'error_rate': self.metrics.error_rate,
                    'average_response_time': self.metrics.average_response_time,
                    'recent_error_rate': recent_error_rate,
                    'recent_operations': recent_total
                },
                'errors_by_category': dict(self.metrics.error_count_by_category),
                'errors_by_severity': dict(self.metrics.error_count_by_severity),
                'recent_alerts': len([a for a in self.alerts if a.timestamp > cutoff_time]),
                'active_alerts': len(self.alerts)
            }
    
    def get_recent_errors(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent database errors.
        
        Args:
            limit: Maximum number of errors to return
            
        Returns:
            List of recent error information
        """
        with self._lock:
            recent_errors = list(self.metrics.recent_errors)[-limit:]
            return [
                {
                    'timestamp': error['timestamp'].isoformat(),
                    'operation': error['operation_name'],
                    'error_type': error['error']['error_type'],
                    'message': error['error']['message'],
                    'severity': error['error']['severity'],
                    'category': error['error']['category']
                }
                for error in reversed(recent_errors)
            ]
    
    def reset_metrics(self) -> None:
        """Reset all metrics and alerts."""
        with self._lock:
            self.metrics = DatabaseMetrics()
            self.alerts = []
            self.operation_history.clear()
            self.logger.info("Database monitor metrics reset")


# Global database monitor instance
db_monitor = DatabaseMonitor()


# Context manager for operation monitoring
class MonitoredOperation:
    """Context manager for monitoring database operations."""
    
    def __init__(self, operation_name: str, monitor: DatabaseMonitor = None):
        self.operation_name = operation_name
        self.monitor = monitor or db_monitor
        self.operation_id = None
    
    def __enter__(self):
        self.operation_id = self.monitor.record_operation_start(self.operation_name)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.monitor.record_operation_success(self.operation_id)
        else:
            self.monitor.record_operation_error(self.operation_id, exc_val)
        return False  # Don't suppress exceptions