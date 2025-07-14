"""
Comprehensive monitoring utilities for batch processing operations.

This module provides detailed monitoring, metrics collection, and performance
tracking for batch processing operations in the YouTube summarization system.
"""

import logging
import time
import psutil
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import json
import uuid

from ..database.batch_models import BatchStatus, BatchItemStatus, BatchPriority

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics that can be collected."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class BatchProcessingMetrics:
    """Comprehensive metrics for batch processing operations."""
    batch_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    total_duration: float = 0.0
    
    # Processing metrics
    total_items: int = 0
    processed_items: int = 0
    failed_items: int = 0
    completed_items: int = 0
    queued_items: int = 0
    
    # Performance metrics
    processing_rate: float = 0.0  # items per second
    average_item_duration: float = 0.0
    success_rate: float = 0.0
    failure_rate: float = 0.0
    
    # Resource metrics
    memory_usage: Dict[str, Any] = field(default_factory=dict)
    cpu_usage: Dict[str, Any] = field(default_factory=dict)
    peak_memory: Optional[int] = None
    peak_cpu: Optional[float] = None
    
    # Error metrics
    error_count: int = 0
    retry_count: int = 0
    timeout_count: int = 0
    
    # Worker metrics
    active_workers: int = 0
    max_workers: int = 0
    worker_efficiency: float = 0.0
    
    # Queue metrics
    queue_size: int = 0
    queue_wait_time: float = 0.0
    
    # Custom metrics
    custom_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def update_progress(self, processed: int, failed: int, completed: int, queued: int):
        """Update processing progress metrics."""
        self.processed_items = processed
        self.failed_items = failed
        self.completed_items = completed
        self.queued_items = queued
        
        # Calculate derived metrics
        if self.total_items > 0:
            self.success_rate = (completed / self.total_items) * 100
            self.failure_rate = (failed / self.total_items) * 100
        
        # Calculate processing rate
        if self.start_time:
            elapsed = (datetime.utcnow() - self.start_time).total_seconds()
            if elapsed > 0:
                self.processing_rate = (processed + failed) / elapsed
    
    def update_resource_usage(self, memory_mb: int, cpu_percent: float):
        """Update resource usage metrics."""
        self.memory_usage['current_mb'] = memory_mb
        self.cpu_usage['current_percent'] = cpu_percent
        
        # Track peaks
        if self.peak_memory is None or memory_mb > self.peak_memory:
            self.peak_memory = memory_mb
        if self.peak_cpu is None or cpu_percent > self.peak_cpu:
            self.peak_cpu = cpu_percent
    
    def finish(self):
        """Mark batch processing as finished and calculate final metrics."""
        self.end_time = datetime.utcnow()
        if self.start_time:
            self.total_duration = (self.end_time - self.start_time).total_seconds()
        
        # Calculate average item duration
        if self.processed_items > 0 and self.total_duration > 0:
            self.average_item_duration = self.total_duration / self.processed_items
        
        # Calculate worker efficiency
        if self.max_workers > 0:
            self.worker_efficiency = (self.processing_rate / self.max_workers) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for serialization."""
        return {
            'batch_id': self.batch_id,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'total_duration': self.total_duration,
            'total_items': self.total_items,
            'processed_items': self.processed_items,
            'failed_items': self.failed_items,
            'completed_items': self.completed_items,
            'queued_items': self.queued_items,
            'processing_rate': self.processing_rate,
            'average_item_duration': self.average_item_duration,
            'success_rate': self.success_rate,
            'failure_rate': self.failure_rate,
            'memory_usage': self.memory_usage,
            'cpu_usage': self.cpu_usage,
            'peak_memory': self.peak_memory,
            'peak_cpu': self.peak_cpu,
            'error_count': self.error_count,
            'retry_count': self.retry_count,
            'timeout_count': self.timeout_count,
            'active_workers': self.active_workers,
            'max_workers': self.max_workers,
            'worker_efficiency': self.worker_efficiency,
            'queue_size': self.queue_size,
            'queue_wait_time': self.queue_wait_time,
            'custom_metrics': self.custom_metrics
        }


@dataclass
class BatchItemMetrics:
    """Detailed metrics for individual batch item processing."""
    batch_item_id: int
    batch_id: str
    url: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: float = 0.0
    status: BatchItemStatus = BatchItemStatus.QUEUED
    
    # Processing stages
    stage_durations: Dict[str, float] = field(default_factory=dict)
    current_stage: Optional[str] = None
    
    # Worker info
    worker_id: Optional[str] = None
    worker_start_time: Optional[datetime] = None
    
    # Error tracking
    error_count: int = 0
    retry_count: int = 0
    last_error: Optional[str] = None
    
    # Resource usage
    memory_before: Optional[int] = None
    memory_after: Optional[int] = None
    cpu_time: Optional[float] = None
    
    # Quality metrics
    result_quality_score: Optional[float] = None
    processing_efficiency: Optional[float] = None
    
    def start_stage(self, stage_name: str):
        """Start timing a processing stage."""
        if self.current_stage:
            # End previous stage
            self.end_stage(self.current_stage)
        
        self.current_stage = stage_name
        self.stage_durations[stage_name] = time.time()
    
    def end_stage(self, stage_name: str):
        """End timing a processing stage."""
        if stage_name in self.stage_durations:
            start_time = self.stage_durations[stage_name]
            self.stage_durations[stage_name] = time.time() - start_time
    
    def record_error(self, error_message: str):
        """Record an error for this batch item."""
        self.error_count += 1
        self.last_error = error_message
    
    def record_retry(self):
        """Record a retry attempt."""
        self.retry_count += 1
    
    def finish(self, status: BatchItemStatus):
        """Mark item processing as finished."""
        self.end_time = datetime.utcnow()
        self.status = status
        
        if self.start_time:
            self.duration = (self.end_time - self.start_time).total_seconds()
        
        # End current stage if active
        if self.current_stage:
            self.end_stage(self.current_stage)
            self.current_stage = None
        
        # Calculate processing efficiency
        if self.duration > 0:
            expected_duration = 30.0  # Expected processing time in seconds
            self.processing_efficiency = min(100.0, (expected_duration / self.duration) * 100)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for serialization."""
        return {
            'batch_item_id': self.batch_item_id,
            'batch_id': self.batch_id,
            'url': self.url,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration': self.duration,
            'status': self.status.value if self.status else None,
            'stage_durations': self.stage_durations,
            'current_stage': self.current_stage,
            'worker_id': self.worker_id,
            'worker_start_time': self.worker_start_time.isoformat() if self.worker_start_time else None,
            'error_count': self.error_count,
            'retry_count': self.retry_count,
            'last_error': self.last_error,
            'memory_before': self.memory_before,
            'memory_after': self.memory_after,
            'cpu_time': self.cpu_time,
            'result_quality_score': self.result_quality_score,
            'processing_efficiency': self.processing_efficiency
        }


@dataclass
class Alert:
    """Alert for monitoring system."""
    alert_id: str
    level: AlertLevel
    message: str
    component: str
    timestamp: datetime
    details: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    
    def resolve(self):
        """Mark alert as resolved."""
        self.resolved = True
        self.resolved_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary for serialization."""
        return {
            'alert_id': self.alert_id,
            'level': self.level.value,
            'message': self.message,
            'component': self.component,
            'timestamp': self.timestamp.isoformat(),
            'details': self.details,
            'resolved': self.resolved,
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None
        }


class BatchMonitor:
    """
    Comprehensive monitoring system for batch processing operations.
    
    This class provides real-time monitoring, metrics collection, alerting,
    and performance tracking for batch processing operations.
    """
    
    def __init__(self, enabled: bool = True):
        """
        Initialize the batch monitor.
        
        Args:
            enabled: Whether monitoring is enabled
        """
        self.enabled = enabled
        self._logger = logging.getLogger(f"{__name__}.BatchMonitor")
        
        # Metrics storage
        self._batch_metrics: Dict[str, BatchProcessingMetrics] = {}
        self._item_metrics: Dict[int, BatchItemMetrics] = {}
        
        # Alerting
        self._alerts: List[Alert] = []
        self._alert_handlers: List[callable] = []
        
        # Performance tracking
        self._performance_history: deque = deque(maxlen=1000)
        self._metrics_history: deque = deque(maxlen=1000)
        
        # Real-time monitoring
        self._monitoring_thread: Optional[threading.Thread] = None
        self._monitoring_active = False
        
        # Thresholds for alerting
        self._thresholds = {
            'high_failure_rate': 20.0,  # Percentage
            'low_processing_rate': 0.1,  # Items per second
            'high_memory_usage': 80.0,  # Percentage
            'high_cpu_usage': 90.0,  # Percentage
            'long_queue_wait': 300.0,  # Seconds
            'high_error_rate': 10.0,  # Percentage
        }
        
        # Custom metrics
        self._custom_counters: Dict[str, int] = defaultdict(int)
        self._custom_gauges: Dict[str, float] = defaultdict(float)
        self._custom_histograms: Dict[str, List[float]] = defaultdict(list)
        
        if self.enabled:
            self._start_monitoring()
    
    def _start_monitoring(self):
        """Start the monitoring thread."""
        if not self._monitoring_active:
            self._monitoring_active = True
            self._monitoring_thread = threading.Thread(
                target=self._monitoring_loop,
                daemon=True
            )
            self._monitoring_thread.start()
            self._logger.info("Batch monitoring started")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self._monitoring_active:
            try:
                # Update system metrics
                self._update_system_metrics()
                
                # Check thresholds and generate alerts
                self._check_alert_conditions()
                
                # Clean up old metrics
                self._cleanup_old_metrics()
                
                # Sleep for monitoring interval
                time.sleep(30)  # 30 second interval
                
            except Exception as e:
                self._logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)  # Sleep longer on error
    
    def start_batch_monitoring(self, batch_id: str, total_items: int) -> BatchProcessingMetrics:
        """
        Start monitoring a batch processing operation.
        
        Args:
            batch_id: Unique identifier for the batch
            total_items: Total number of items to process
            
        Returns:
            BatchProcessingMetrics object for tracking
        """
        if not self.enabled:
            return None
        
        metrics = BatchProcessingMetrics(
            batch_id=batch_id,
            start_time=datetime.utcnow(),
            total_items=total_items
        )
        
        self._batch_metrics[batch_id] = metrics
        
        self._logger.info(f"Started batch monitoring for {batch_id} with {total_items} items")
        return metrics
    
    def finish_batch_monitoring(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """
        Finish monitoring a batch processing operation.
        
        Args:
            batch_id: Unique identifier for the batch
            
        Returns:
            Final metrics summary or None if not found
        """
        if not self.enabled or batch_id not in self._batch_metrics:
            return None
        
        metrics = self._batch_metrics[batch_id]
        metrics.finish()
        
        # Store in history
        self._metrics_history.append(metrics.to_dict())
        
        # Generate summary
        summary = self._generate_batch_summary(metrics)
        
        self._logger.info(f"Finished batch monitoring for {batch_id}: {summary}")
        return summary
    
    def start_item_monitoring(self, batch_item_id: int, batch_id: str, url: str) -> BatchItemMetrics:
        """
        Start monitoring a batch item processing operation.
        
        Args:
            batch_item_id: Unique identifier for the batch item
            batch_id: Batch identifier
            url: URL being processed
            
        Returns:
            BatchItemMetrics object for tracking
        """
        if not self.enabled:
            return None
        
        metrics = BatchItemMetrics(
            batch_item_id=batch_item_id,
            batch_id=batch_id,
            url=url,
            start_time=datetime.utcnow(),
            memory_before=self._get_memory_usage()
        )
        
        self._item_metrics[batch_item_id] = metrics
        
        self._logger.debug(f"Started item monitoring for {batch_item_id} in batch {batch_id}")
        return metrics
    
    def finish_item_monitoring(self, batch_item_id: int, status: BatchItemStatus) -> Optional[Dict[str, Any]]:
        """
        Finish monitoring a batch item processing operation.
        
        Args:
            batch_item_id: Unique identifier for the batch item
            status: Final processing status
            
        Returns:
            Item metrics summary or None if not found
        """
        if not self.enabled or batch_item_id not in self._item_metrics:
            return None
        
        metrics = self._item_metrics[batch_item_id]
        metrics.memory_after = self._get_memory_usage()
        metrics.finish(status)
        
        # Update batch metrics
        if metrics.batch_id in self._batch_metrics:
            batch_metrics = self._batch_metrics[metrics.batch_id]
            if status == BatchItemStatus.COMPLETED:
                batch_metrics.completed_items += 1
            elif status == BatchItemStatus.FAILED:
                batch_metrics.failed_items += 1
            
            # Update batch progress
            batch_metrics.update_progress(
                processed=batch_metrics.completed_items + batch_metrics.failed_items,
                failed=batch_metrics.failed_items,
                completed=batch_metrics.completed_items,
                queued=batch_metrics.queued_items
            )
        
        summary = metrics.to_dict()
        self._logger.debug(f"Finished item monitoring for {batch_item_id}: {status.value}")
        return summary
    
    def record_error(self, batch_id: str, batch_item_id: int, error_message: str):
        """
        Record an error for monitoring.
        
        Args:
            batch_id: Batch identifier
            batch_item_id: Batch item identifier
            error_message: Error message
        """
        if not self.enabled:
            return
        
        # Update batch metrics
        if batch_id in self._batch_metrics:
            self._batch_metrics[batch_id].error_count += 1
        
        # Update item metrics
        if batch_item_id in self._item_metrics:
            self._item_metrics[batch_item_id].record_error(error_message)
        
        # Generate alert if error rate is high
        self._check_error_rate_alert(batch_id)
        
        self._logger.warning(f"Error recorded for batch {batch_id}, item {batch_item_id}: {error_message}")
    
    def record_retry(self, batch_id: str, batch_item_id: int):
        """
        Record a retry attempt for monitoring.
        
        Args:
            batch_id: Batch identifier
            batch_item_id: Batch item identifier
        """
        if not self.enabled:
            return
        
        # Update batch metrics
        if batch_id in self._batch_metrics:
            self._batch_metrics[batch_id].retry_count += 1
        
        # Update item metrics
        if batch_item_id in self._item_metrics:
            self._item_metrics[batch_item_id].record_retry()
        
        self._logger.debug(f"Retry recorded for batch {batch_id}, item {batch_item_id}")
    
    def update_worker_count(self, batch_id: str, active_workers: int, max_workers: int):
        """
        Update worker count metrics.
        
        Args:
            batch_id: Batch identifier
            active_workers: Number of active workers
            max_workers: Maximum number of workers
        """
        if not self.enabled or batch_id not in self._batch_metrics:
            return
        
        metrics = self._batch_metrics[batch_id]
        metrics.active_workers = active_workers
        metrics.max_workers = max_workers
        
        # Check for worker efficiency alert
        if max_workers > 0:
            efficiency = (active_workers / max_workers) * 100
            if efficiency < 50:  # Less than 50% efficiency
                self._generate_alert(
                    AlertLevel.WARNING,
                    f"Low worker efficiency in batch {batch_id}: {efficiency:.1f}%",
                    "worker_efficiency",
                    {"batch_id": batch_id, "efficiency": efficiency}
                )
    
    def update_queue_metrics(self, batch_id: str, queue_size: int, wait_time: float):
        """
        Update queue metrics.
        
        Args:
            batch_id: Batch identifier
            queue_size: Current queue size
            wait_time: Average wait time in seconds
        """
        if not self.enabled or batch_id not in self._batch_metrics:
            return
        
        metrics = self._batch_metrics[batch_id]
        metrics.queue_size = queue_size
        metrics.queue_wait_time = wait_time
        
        # Check for queue wait time alert
        if wait_time > self._thresholds['long_queue_wait']:
            self._generate_alert(
                AlertLevel.WARNING,
                f"Long queue wait time in batch {batch_id}: {wait_time:.1f}s",
                "queue_wait_time",
                {"batch_id": batch_id, "wait_time": wait_time}
            )
    
    def record_custom_metric(self, name: str, value: Any, metric_type: MetricType):
        """
        Record a custom metric.
        
        Args:
            name: Metric name
            value: Metric value
            metric_type: Type of metric
        """
        if not self.enabled:
            return
        
        if metric_type == MetricType.COUNTER:
            self._custom_counters[name] += value
        elif metric_type == MetricType.GAUGE:
            self._custom_gauges[name] = value
        elif metric_type == MetricType.HISTOGRAM:
            self._custom_histograms[name].append(value)
        
        self._logger.debug(f"Custom metric recorded: {name} = {value} ({metric_type.value})")
    
    def get_batch_metrics(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """
        Get current metrics for a batch.
        
        Args:
            batch_id: Batch identifier
            
        Returns:
            Batch metrics or None if not found
        """
        if not self.enabled or batch_id not in self._batch_metrics:
            return None
        
        return self._batch_metrics[batch_id].to_dict()
    
    def get_item_metrics(self, batch_item_id: int) -> Optional[Dict[str, Any]]:
        """
        Get current metrics for a batch item.
        
        Args:
            batch_item_id: Batch item identifier
            
        Returns:
            Item metrics or None if not found
        """
        if not self.enabled or batch_item_id not in self._item_metrics:
            return None
        
        return self._item_metrics[batch_item_id].to_dict()
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """
        Get current system metrics.
        
        Returns:
            System metrics dictionary
        """
        if not self.enabled:
            return {}
        
        return {
            'memory_usage_mb': self._get_memory_usage(),
            'cpu_usage_percent': self._get_cpu_usage(),
            'active_batches': len(self._batch_metrics),
            'active_items': len(self._item_metrics),
            'total_alerts': len(self._alerts),
            'unresolved_alerts': len([a for a in self._alerts if not a.resolved]),
            'custom_counters': dict(self._custom_counters),
            'custom_gauges': dict(self._custom_gauges),
            'custom_histograms': {k: len(v) for k, v in self._custom_histograms.items()}
        }
    
    def get_alerts(self, level: Optional[AlertLevel] = None, resolved: bool = False) -> List[Dict[str, Any]]:
        """
        Get alerts from the monitoring system.
        
        Args:
            level: Optional alert level filter
            resolved: Whether to include resolved alerts
            
        Returns:
            List of alert dictionaries
        """
        if not self.enabled:
            return []
        
        alerts = self._alerts
        
        if level:
            alerts = [a for a in alerts if a.level == level]
        
        if not resolved:
            alerts = [a for a in alerts if not a.resolved]
        
        return [alert.to_dict() for alert in alerts]
    
    def add_alert_handler(self, handler: callable):
        """
        Add a custom alert handler.
        
        Args:
            handler: Function to handle alerts (takes Alert object as parameter)
        """
        self._alert_handlers.append(handler)
    
    def _generate_alert(self, level: AlertLevel, message: str, component: str, details: Dict[str, Any]):
        """Generate an alert."""
        alert = Alert(
            alert_id=str(uuid.uuid4()),
            level=level,
            message=message,
            component=component,
            timestamp=datetime.utcnow(),
            details=details
        )
        
        self._alerts.append(alert)
        
        # Call alert handlers
        for handler in self._alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                self._logger.error(f"Error in alert handler: {e}")
        
        self._logger.warning(f"Alert generated: {level.value} - {message}")
    
    def _check_error_rate_alert(self, batch_id: str):
        """Check if error rate exceeds threshold."""
        if batch_id not in self._batch_metrics:
            return
        
        metrics = self._batch_metrics[batch_id]
        if metrics.total_items > 0:
            error_rate = (metrics.error_count / metrics.total_items) * 100
            if error_rate > self._thresholds['high_error_rate']:
                self._generate_alert(
                    AlertLevel.ERROR,
                    f"High error rate in batch {batch_id}: {error_rate:.1f}%",
                    "error_rate",
                    {"batch_id": batch_id, "error_rate": error_rate}
                )
    
    def _check_alert_conditions(self):
        """Check all alert conditions."""
        # Check system resources
        memory_usage = self._get_memory_usage_percent()
        cpu_usage = self._get_cpu_usage()
        
        if memory_usage > self._thresholds['high_memory_usage']:
            self._generate_alert(
                AlertLevel.WARNING,
                f"High memory usage: {memory_usage:.1f}%",
                "memory_usage",
                {"memory_usage": memory_usage}
            )
        
        if cpu_usage > self._thresholds['high_cpu_usage']:
            self._generate_alert(
                AlertLevel.WARNING,
                f"High CPU usage: {cpu_usage:.1f}%",
                "cpu_usage",
                {"cpu_usage": cpu_usage}
            )
        
        # Check batch-specific conditions
        for batch_id, metrics in self._batch_metrics.items():
            # Check processing rate
            if metrics.processing_rate < self._thresholds['low_processing_rate']:
                self._generate_alert(
                    AlertLevel.WARNING,
                    f"Low processing rate in batch {batch_id}: {metrics.processing_rate:.3f} items/sec",
                    "processing_rate",
                    {"batch_id": batch_id, "processing_rate": metrics.processing_rate}
                )
            
            # Check failure rate
            if metrics.failure_rate > self._thresholds['high_failure_rate']:
                self._generate_alert(
                    AlertLevel.ERROR,
                    f"High failure rate in batch {batch_id}: {metrics.failure_rate:.1f}%",
                    "failure_rate",
                    {"batch_id": batch_id, "failure_rate": metrics.failure_rate}
                )
    
    def _update_system_metrics(self):
        """Update system-wide metrics."""
        # Update resource usage for all active batches
        memory_mb = self._get_memory_usage()
        cpu_percent = self._get_cpu_usage()
        
        for metrics in self._batch_metrics.values():
            metrics.update_resource_usage(memory_mb, cpu_percent)
    
    def _cleanup_old_metrics(self):
        """Clean up old metrics to prevent memory leaks."""
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        
        # Clean up old batch metrics
        expired_batches = [
            batch_id for batch_id, metrics in self._batch_metrics.items()
            if metrics.end_time and metrics.end_time < cutoff_time
        ]
        for batch_id in expired_batches:
            del self._batch_metrics[batch_id]
        
        # Clean up old item metrics
        expired_items = [
            item_id for item_id, metrics in self._item_metrics.items()
            if metrics.end_time and metrics.end_time < cutoff_time
        ]
        for item_id in expired_items:
            del self._item_metrics[item_id]
        
        # Clean up old alerts
        self._alerts = [alert for alert in self._alerts if alert.timestamp > cutoff_time]
    
    def _generate_batch_summary(self, metrics: BatchProcessingMetrics) -> Dict[str, Any]:
        """Generate a summary of batch processing metrics."""
        return {
            'batch_id': metrics.batch_id,
            'duration': metrics.total_duration,
            'total_items': metrics.total_items,
            'success_rate': metrics.success_rate,
            'failure_rate': metrics.failure_rate,
            'processing_rate': metrics.processing_rate,
            'average_duration': metrics.average_item_duration,
            'peak_memory_mb': metrics.peak_memory,
            'peak_cpu_percent': metrics.peak_cpu,
            'error_count': metrics.error_count,
            'retry_count': metrics.retry_count,
            'worker_efficiency': metrics.worker_efficiency
        }
    
    def _get_memory_usage(self) -> int:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss // (1024 * 1024)
        except Exception:
            return 0
    
    def _get_memory_usage_percent(self) -> float:
        """Get current memory usage as percentage."""
        try:
            return psutil.virtual_memory().percent
        except Exception:
            return 0.0
    
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        try:
            return psutil.cpu_percent(interval=1)
        except Exception:
            return 0.0
    
    def shutdown(self):
        """Shutdown the monitoring system."""
        if self._monitoring_active:
            self._monitoring_active = False
            if self._monitoring_thread:
                self._monitoring_thread.join(timeout=5)
            self._logger.info("Batch monitoring shutdown")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()


# Global monitor instance
_global_monitor: Optional[BatchMonitor] = None


def get_batch_monitor() -> BatchMonitor:
    """Get the global batch monitor instance."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = BatchMonitor()
    return _global_monitor


def initialize_monitoring(enabled: bool = True) -> BatchMonitor:
    """Initialize the global monitoring system."""
    global _global_monitor
    _global_monitor = BatchMonitor(enabled=enabled)
    return _global_monitor