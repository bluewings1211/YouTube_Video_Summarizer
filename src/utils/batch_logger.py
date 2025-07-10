"""
Comprehensive logging utilities for batch processing operations.

This module provides structured logging, audit trails, and performance logging
for batch processing operations in the YouTube summarization system.
"""

import logging
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import threading
from collections import deque
import traceback
import sys
import os

from ..database.batch_models import BatchStatus, BatchItemStatus, BatchPriority

# Configure logging format
BATCH_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - [%(batch_id)s] - %(message)s"
ITEM_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - [%(batch_id)s:%(item_id)s] - %(message)s"


class LogLevel(Enum):
    """Log levels for batch processing."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class EventType(Enum):
    """Types of events that can be logged."""
    BATCH_CREATED = "batch_created"
    BATCH_STARTED = "batch_started"
    BATCH_COMPLETED = "batch_completed"
    BATCH_CANCELLED = "batch_cancelled"
    BATCH_FAILED = "batch_failed"
    
    ITEM_QUEUED = "item_queued"
    ITEM_STARTED = "item_started"
    ITEM_COMPLETED = "item_completed"
    ITEM_FAILED = "item_failed"
    ITEM_RETRIED = "item_retried"
    ITEM_TIMEOUT = "item_timeout"
    
    WORKER_STARTED = "worker_started"
    WORKER_STOPPED = "worker_stopped"
    WORKER_ERROR = "worker_error"
    
    SYSTEM_ERROR = "system_error"
    PERFORMANCE_ALERT = "performance_alert"
    RESOURCE_ALERT = "resource_alert"
    
    CUSTOM_EVENT = "custom_event"


@dataclass
class LogEntry:
    """Structured log entry for batch processing."""
    timestamp: datetime
    level: LogLevel
    event_type: EventType
    message: str
    batch_id: Optional[str] = None
    batch_item_id: Optional[int] = None
    worker_id: Optional[str] = None
    session_id: Optional[str] = None
    component: Optional[str] = None
    
    # Context information
    context: Dict[str, Any] = field(default_factory=dict)
    
    # Performance metrics
    duration: Optional[float] = None
    memory_usage: Optional[int] = None
    cpu_usage: Optional[float] = None
    
    # Error information
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    stack_trace: Optional[str] = None
    
    # Metadata
    log_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    correlation_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert log entry to dictionary."""
        data = asdict(self)
        # Convert datetime and enum values
        data['timestamp'] = self.timestamp.isoformat()
        data['level'] = self.level.value
        data['event_type'] = self.event_type.value
        return data
    
    def to_json(self) -> str:
        """Convert log entry to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class BatchAuditTrail:
    """Audit trail for batch processing operations."""
    batch_id: str
    created_at: datetime
    events: List[LogEntry] = field(default_factory=list)
    
    def add_event(self, event: LogEntry):
        """Add an event to the audit trail."""
        self.events.append(event)
    
    def get_events_by_type(self, event_type: EventType) -> List[LogEntry]:
        """Get events by type."""
        return [event for event in self.events if event.event_type == event_type]
    
    def get_events_by_level(self, level: LogLevel) -> List[LogEntry]:
        """Get events by log level."""
        return [event for event in self.events if event.level == level]
    
    def get_error_events(self) -> List[LogEntry]:
        """Get all error events."""
        return [event for event in self.events if event.level in [LogLevel.ERROR, LogLevel.CRITICAL]]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary from audit trail."""
        durations = [event.duration for event in self.events if event.duration is not None]
        memory_usage = [event.memory_usage for event in self.events if event.memory_usage is not None]
        
        return {
            'total_events': len(self.events),
            'error_events': len(self.get_error_events()),
            'average_duration': sum(durations) / len(durations) if durations else 0,
            'max_duration': max(durations) if durations else 0,
            'peak_memory': max(memory_usage) if memory_usage else 0,
            'event_types': {event_type.value: len(self.get_events_by_type(event_type)) for event_type in EventType}
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert audit trail to dictionary."""
        return {
            'batch_id': self.batch_id,
            'created_at': self.created_at.isoformat(),
            'events': [event.to_dict() for event in self.events],
            'performance_summary': self.get_performance_summary()
        }


class BatchLogger:
    """
    Comprehensive logging system for batch processing operations.
    
    This class provides structured logging, audit trails, performance logging,
    and error tracking for batch processing operations.
    """
    
    def __init__(self, 
                 log_dir: Optional[str] = None,
                 log_level: LogLevel = LogLevel.INFO,
                 enable_file_logging: bool = True,
                 enable_console_logging: bool = True,
                 max_log_files: int = 10,
                 max_audit_entries: int = 10000):
        """
        Initialize the batch logger.
        
        Args:
            log_dir: Directory for log files
            log_level: Minimum log level to record
            enable_file_logging: Whether to log to files
            enable_console_logging: Whether to log to console
            max_log_files: Maximum number of log files to keep
            max_audit_entries: Maximum number of audit trail entries
        """
        self.log_level = log_level
        self.enable_file_logging = enable_file_logging
        self.enable_console_logging = enable_console_logging
        self.max_log_files = max_log_files
        self.max_audit_entries = max_audit_entries
        
        # Set up log directory
        self.log_dir = Path(log_dir) if log_dir else Path("logs")
        self.log_dir.mkdir(exist_ok=True)
        
        # Internal state
        self._loggers: Dict[str, logging.Logger] = {}
        self._audit_trails: Dict[str, BatchAuditTrail] = {}
        self._log_buffer: deque = deque(maxlen=1000)
        self._lock = threading.Lock()
        
        # Performance tracking
        self._performance_logs: deque = deque(maxlen=1000)
        self._error_logs: deque = deque(maxlen=1000)
        
        # Configure root logger
        self._setup_logging()
    
    def _setup_logging(self):
        """Set up logging configuration."""
        # Create formatters
        self.batch_formatter = logging.Formatter(BATCH_LOG_FORMAT)
        self.item_formatter = logging.Formatter(ITEM_LOG_FORMAT)
        
        # Set up file handlers
        if self.enable_file_logging:
            self._setup_file_logging()
        
        # Set up console handlers
        if self.enable_console_logging:
            self._setup_console_logging()
    
    def _setup_file_logging(self):
        """Set up file logging handlers."""
        # Batch processing log
        batch_log_file = self.log_dir / "batch_processing.log"
        batch_handler = logging.FileHandler(batch_log_file)
        batch_handler.setFormatter(self.batch_formatter)
        batch_handler.setLevel(getattr(logging, self.log_level.value))
        
        # Error log
        error_log_file = self.log_dir / "batch_errors.log"
        error_handler = logging.FileHandler(error_log_file)
        error_handler.setFormatter(self.batch_formatter)
        error_handler.setLevel(logging.ERROR)
        
        # Performance log
        performance_log_file = self.log_dir / "batch_performance.log"
        performance_handler = logging.FileHandler(performance_log_file)
        performance_handler.setFormatter(self.batch_formatter)
        performance_handler.setLevel(logging.INFO)
        
        # Store handlers for later use
        self._file_handlers = {
            'batch': batch_handler,
            'error': error_handler,
            'performance': performance_handler
        }
    
    def _setup_console_logging(self):
        """Set up console logging handlers."""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(self.batch_formatter)
        console_handler.setLevel(getattr(logging, self.log_level.value))
        
        self._console_handler = console_handler
    
    def get_logger(self, name: str) -> logging.Logger:
        """
        Get a logger instance for a specific component.
        
        Args:
            name: Logger name
            
        Returns:
            Logger instance
        """
        if name not in self._loggers:
            logger = logging.getLogger(f"batch_processing.{name}")
            logger.setLevel(getattr(logging, self.log_level.value))
            
            # Add handlers
            if self.enable_file_logging:
                logger.addHandler(self._file_handlers['batch'])
                logger.addHandler(self._file_handlers['error'])
            
            if self.enable_console_logging:
                logger.addHandler(self._console_handler)
            
            # Prevent duplicate logs
            logger.propagate = False
            
            self._loggers[name] = logger
        
        return self._loggers[name]
    
    def create_log_entry(self,
                        level: LogLevel,
                        event_type: EventType,
                        message: str,
                        batch_id: Optional[str] = None,
                        batch_item_id: Optional[int] = None,
                        worker_id: Optional[str] = None,
                        session_id: Optional[str] = None,
                        component: Optional[str] = None,
                        context: Optional[Dict[str, Any]] = None,
                        duration: Optional[float] = None,
                        memory_usage: Optional[int] = None,
                        cpu_usage: Optional[float] = None,
                        error_type: Optional[str] = None,
                        error_message: Optional[str] = None,
                        stack_trace: Optional[str] = None,
                        correlation_id: Optional[str] = None) -> LogEntry:
        """
        Create a structured log entry.
        
        Args:
            level: Log level
            event_type: Type of event
            message: Log message
            batch_id: Batch identifier
            batch_item_id: Batch item identifier
            worker_id: Worker identifier
            session_id: Session identifier
            component: Component name
            context: Additional context data
            duration: Operation duration
            memory_usage: Memory usage in bytes
            cpu_usage: CPU usage percentage
            error_type: Error type
            error_message: Error message
            stack_trace: Stack trace
            correlation_id: Correlation identifier
            
        Returns:
            LogEntry object
        """
        return LogEntry(
            timestamp=datetime.utcnow(),
            level=level,
            event_type=event_type,
            message=message,
            batch_id=batch_id,
            batch_item_id=batch_item_id,
            worker_id=worker_id,
            session_id=session_id,
            component=component,
            context=context or {},
            duration=duration,
            memory_usage=memory_usage,
            cpu_usage=cpu_usage,
            error_type=error_type,
            error_message=error_message,
            stack_trace=stack_trace,
            correlation_id=correlation_id
        )
    
    def log_entry(self, entry: LogEntry):
        """
        Log a structured entry.
        
        Args:
            entry: LogEntry object to log
        """
        with self._lock:
            # Add to buffer
            self._log_buffer.append(entry)
            
            # Add to audit trail
            if entry.batch_id:
                self._add_to_audit_trail(entry)
            
            # Track performance and errors
            if entry.duration is not None:
                self._performance_logs.append(entry)
            
            if entry.level in [LogLevel.ERROR, LogLevel.CRITICAL]:
                self._error_logs.append(entry)
            
            # Log to standard logging system
            self._log_to_standard_logger(entry)
    
    def _add_to_audit_trail(self, entry: LogEntry):
        """Add entry to audit trail."""
        batch_id = entry.batch_id
        if batch_id not in self._audit_trails:
            self._audit_trails[batch_id] = BatchAuditTrail(
                batch_id=batch_id,
                created_at=datetime.utcnow()
            )
        
        self._audit_trails[batch_id].add_event(entry)
        
        # Limit audit trail size
        if len(self._audit_trails[batch_id].events) > self.max_audit_entries:
            self._audit_trails[batch_id].events = self._audit_trails[batch_id].events[-self.max_audit_entries:]
    
    def _log_to_standard_logger(self, entry: LogEntry):
        """Log entry to standard logging system."""
        logger = self.get_logger(entry.component or "batch")
        
        # Create log message
        message = entry.message
        if entry.context:
            message += f" | Context: {json.dumps(entry.context)}"
        
        # Add structured data to log record
        extra = {
            'batch_id': entry.batch_id or 'N/A',
            'item_id': entry.batch_item_id or 'N/A',
            'worker_id': entry.worker_id or 'N/A',
            'event_type': entry.event_type.value,
            'log_id': entry.log_id
        }
        
        # Log at appropriate level
        log_level = getattr(logging, entry.level.value)
        logger.log(log_level, message, extra=extra)
    
    def log_batch_event(self,
                       batch_id: str,
                       event_type: EventType,
                       message: str,
                       level: LogLevel = LogLevel.INFO,
                       context: Optional[Dict[str, Any]] = None,
                       duration: Optional[float] = None):
        """
        Log a batch-level event.
        
        Args:
            batch_id: Batch identifier
            event_type: Type of event
            message: Log message
            level: Log level
            context: Additional context data
            duration: Operation duration
        """
        entry = self.create_log_entry(
            level=level,
            event_type=event_type,
            message=message,
            batch_id=batch_id,
            component="batch_service",
            context=context,
            duration=duration
        )
        
        self.log_entry(entry)
    
    def log_item_event(self,
                      batch_id: str,
                      batch_item_id: int,
                      event_type: EventType,
                      message: str,
                      level: LogLevel = LogLevel.INFO,
                      context: Optional[Dict[str, Any]] = None,
                      duration: Optional[float] = None,
                      worker_id: Optional[str] = None):
        """
        Log a batch item event.
        
        Args:
            batch_id: Batch identifier
            batch_item_id: Batch item identifier
            event_type: Type of event
            message: Log message
            level: Log level
            context: Additional context data
            duration: Operation duration
            worker_id: Worker identifier
        """
        entry = self.create_log_entry(
            level=level,
            event_type=event_type,
            message=message,
            batch_id=batch_id,
            batch_item_id=batch_item_id,
            worker_id=worker_id,
            component="batch_processor",
            context=context,
            duration=duration
        )
        
        self.log_entry(entry)
    
    def log_worker_event(self,
                        worker_id: str,
                        event_type: EventType,
                        message: str,
                        level: LogLevel = LogLevel.INFO,
                        batch_id: Optional[str] = None,
                        batch_item_id: Optional[int] = None,
                        context: Optional[Dict[str, Any]] = None):
        """
        Log a worker event.
        
        Args:
            worker_id: Worker identifier
            event_type: Type of event
            message: Log message
            level: Log level
            batch_id: Batch identifier
            batch_item_id: Batch item identifier
            context: Additional context data
        """
        entry = self.create_log_entry(
            level=level,
            event_type=event_type,
            message=message,
            batch_id=batch_id,
            batch_item_id=batch_item_id,
            worker_id=worker_id,
            component="queue_service",
            context=context
        )
        
        self.log_entry(entry)
    
    def log_error(self,
                 message: str,
                 error: Exception,
                 batch_id: Optional[str] = None,
                 batch_item_id: Optional[int] = None,
                 worker_id: Optional[str] = None,
                 component: Optional[str] = None,
                 context: Optional[Dict[str, Any]] = None):
        """
        Log an error with full details.
        
        Args:
            message: Error message
            error: Exception object
            batch_id: Batch identifier
            batch_item_id: Batch item identifier
            worker_id: Worker identifier
            component: Component name
            context: Additional context data
        """
        entry = self.create_log_entry(
            level=LogLevel.ERROR,
            event_type=EventType.SYSTEM_ERROR,
            message=message,
            batch_id=batch_id,
            batch_item_id=batch_item_id,
            worker_id=worker_id,
            component=component or "system",
            context=context,
            error_type=type(error).__name__,
            error_message=str(error),
            stack_trace=traceback.format_exc()
        )
        
        self.log_entry(entry)
    
    def log_performance(self,
                       operation: str,
                       duration: float,
                       batch_id: Optional[str] = None,
                       batch_item_id: Optional[int] = None,
                       worker_id: Optional[str] = None,
                       memory_usage: Optional[int] = None,
                       cpu_usage: Optional[float] = None,
                       context: Optional[Dict[str, Any]] = None):
        """
        Log performance metrics.
        
        Args:
            operation: Operation name
            duration: Operation duration
            batch_id: Batch identifier
            batch_item_id: Batch item identifier
            worker_id: Worker identifier
            memory_usage: Memory usage in bytes
            cpu_usage: CPU usage percentage
            context: Additional context data
        """
        entry = self.create_log_entry(
            level=LogLevel.INFO,
            event_type=EventType.PERFORMANCE_ALERT,
            message=f"Performance: {operation} completed in {duration:.2f}s",
            batch_id=batch_id,
            batch_item_id=batch_item_id,
            worker_id=worker_id,
            component="performance",
            context=context,
            duration=duration,
            memory_usage=memory_usage,
            cpu_usage=cpu_usage
        )
        
        self.log_entry(entry)
    
    def get_audit_trail(self, batch_id: str) -> Optional[BatchAuditTrail]:
        """
        Get audit trail for a batch.
        
        Args:
            batch_id: Batch identifier
            
        Returns:
            BatchAuditTrail object or None if not found
        """
        return self._audit_trails.get(batch_id)
    
    def get_recent_logs(self, 
                       count: int = 100,
                       level: Optional[LogLevel] = None,
                       event_type: Optional[EventType] = None,
                       batch_id: Optional[str] = None) -> List[LogEntry]:
        """
        Get recent log entries.
        
        Args:
            count: Number of entries to return
            level: Optional log level filter
            event_type: Optional event type filter
            batch_id: Optional batch ID filter
            
        Returns:
            List of LogEntry objects
        """
        logs = list(self._log_buffer)
        
        # Apply filters
        if level:
            logs = [log for log in logs if log.level == level]
        
        if event_type:
            logs = [log for log in logs if log.event_type == event_type]
        
        if batch_id:
            logs = [log for log in logs if log.batch_id == batch_id]
        
        # Sort by timestamp (most recent first)
        logs.sort(key=lambda x: x.timestamp, reverse=True)
        
        return logs[:count]
    
    def get_error_summary(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get error summary for the specified time period.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            Error summary dictionary
        """
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        recent_errors = [
            error for error in self._error_logs
            if error.timestamp > cutoff_time
        ]
        
        # Group errors by type
        error_types = {}
        for error in recent_errors:
            error_type = error.error_type or "Unknown"
            if error_type not in error_types:
                error_types[error_type] = []
            error_types[error_type].append(error)
        
        return {
            'total_errors': len(recent_errors),
            'error_types': {
                error_type: len(errors)
                for error_type, errors in error_types.items()
            },
            'recent_errors': [error.to_dict() for error in recent_errors[-10:]]
        }
    
    def get_performance_summary(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get performance summary for the specified time period.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            Performance summary dictionary
        """
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        recent_performance = [
            perf for perf in self._performance_logs
            if perf.timestamp > cutoff_time and perf.duration is not None
        ]
        
        if not recent_performance:
            return {'total_operations': 0}
        
        durations = [perf.duration for perf in recent_performance]
        
        return {
            'total_operations': len(recent_performance),
            'average_duration': sum(durations) / len(durations),
            'min_duration': min(durations),
            'max_duration': max(durations),
            'slow_operations': len([d for d in durations if d > 60]),  # Over 1 minute
            'recent_operations': [perf.to_dict() for perf in recent_performance[-10:]]
        }
    
    def export_audit_trail(self, batch_id: str, file_path: Optional[str] = None) -> str:
        """
        Export audit trail to JSON file.
        
        Args:
            batch_id: Batch identifier
            file_path: Optional file path
            
        Returns:
            Path to exported file
        """
        audit_trail = self.get_audit_trail(batch_id)
        if not audit_trail:
            raise ValueError(f"No audit trail found for batch {batch_id}")
        
        if not file_path:
            file_path = self.log_dir / f"audit_trail_{batch_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(file_path, 'w') as f:
            json.dump(audit_trail.to_dict(), f, indent=2)
        
        return str(file_path)
    
    def cleanup_old_logs(self, days: int = 30):
        """
        Clean up old log files.
        
        Args:
            days: Number of days to keep logs
        """
        cutoff_time = datetime.utcnow() - timedelta(days=days)
        
        # Clean up audit trails
        expired_batches = [
            batch_id for batch_id, trail in self._audit_trails.items()
            if trail.created_at < cutoff_time
        ]
        
        for batch_id in expired_batches:
            del self._audit_trails[batch_id]
        
        # Clean up log files
        if self.log_dir.exists():
            for log_file in self.log_dir.glob("*.log"):
                if log_file.stat().st_mtime < cutoff_time.timestamp():
                    log_file.unlink()
    
    def shutdown(self):
        """Shutdown the logging system."""
        # Close all handlers
        for logger in self._loggers.values():
            for handler in logger.handlers:
                handler.close()
        
        # Clear internal state
        self._loggers.clear()
        self._audit_trails.clear()
        self._log_buffer.clear()
        self._performance_logs.clear()
        self._error_logs.clear()


# Context manager for batch operation logging
class BatchOperationLogger:
    """Context manager for logging batch operations."""
    
    def __init__(self, 
                 logger: BatchLogger,
                 batch_id: str,
                 operation: str,
                 batch_item_id: Optional[int] = None,
                 worker_id: Optional[str] = None,
                 context: Optional[Dict[str, Any]] = None):
        self.logger = logger
        self.batch_id = batch_id
        self.operation = operation
        self.batch_item_id = batch_item_id
        self.worker_id = worker_id
        self.context = context or {}
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.utcnow()
        
        # Log start event
        if self.batch_item_id:
            self.logger.log_item_event(
                batch_id=self.batch_id,
                batch_item_id=self.batch_item_id,
                event_type=EventType.ITEM_STARTED,
                message=f"Started {self.operation}",
                worker_id=self.worker_id,
                context=self.context
            )
        else:
            self.logger.log_batch_event(
                batch_id=self.batch_id,
                event_type=EventType.BATCH_STARTED,
                message=f"Started {self.operation}",
                context=self.context
            )
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (datetime.utcnow() - self.start_time).total_seconds()
        
        if exc_type is None:
            # Success
            if self.batch_item_id:
                self.logger.log_item_event(
                    batch_id=self.batch_id,
                    batch_item_id=self.batch_item_id,
                    event_type=EventType.ITEM_COMPLETED,
                    message=f"Completed {self.operation}",
                    worker_id=self.worker_id,
                    context=self.context,
                    duration=duration
                )
            else:
                self.logger.log_batch_event(
                    batch_id=self.batch_id,
                    event_type=EventType.BATCH_COMPLETED,
                    message=f"Completed {self.operation}",
                    context=self.context,
                    duration=duration
                )
        else:
            # Error
            self.logger.log_error(
                message=f"Failed {self.operation}",
                error=exc_val,
                batch_id=self.batch_id,
                batch_item_id=self.batch_item_id,
                worker_id=self.worker_id,
                context=self.context
            )


# Global logger instance
_global_logger: Optional[BatchLogger] = None


def get_batch_logger() -> BatchLogger:
    """Get the global batch logger instance."""
    global _global_logger
    if _global_logger is None:
        _global_logger = BatchLogger()
    return _global_logger


def initialize_logging(log_dir: Optional[str] = None, 
                      log_level: LogLevel = LogLevel.INFO) -> BatchLogger:
    """Initialize the global logging system."""
    global _global_logger
    _global_logger = BatchLogger(log_dir=log_dir, log_level=log_level)
    return _global_logger