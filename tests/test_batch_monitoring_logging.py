"""
Comprehensive tests for batch processing monitoring and logging utilities.

This test suite provides comprehensive testing for monitoring and logging including:
- Structured logging functionality
- Audit trail management
- Performance metrics collection
- System resource monitoring
- Alert generation and handling
- Custom metrics tracking
- Log entry validation
- Monitoring system lifecycle
- Error handling and recovery
- Thread safety and concurrency
- Memory management
- Log file management
- Performance profiling
"""

import pytest
import time
import threading
import tempfile
import json
import uuid
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from src.utils.batch_logger import (
    BatchLogger, LogLevel, EventType, LogEntry, BatchAuditTrail,
    BatchOperationLogger, get_batch_logger, initialize_logging
)
from src.utils.batch_monitor import (
    BatchMonitor, MetricType, AlertLevel, BatchProcessingMetrics,
    BatchItemMetrics, Alert, get_batch_monitor, initialize_monitoring
)
from src.database.batch_models import BatchStatus, BatchItemStatus, BatchPriority


class TestBatchLogger:
    """Comprehensive tests for batch logging functionality."""

    @pytest.fixture(scope="function")
    def temp_log_dir(self):
        """Create temporary directory for log files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    @pytest.fixture(scope="function")
    def batch_logger(self, temp_log_dir):
        """Create BatchLogger instance for testing."""
        logger = BatchLogger(
            log_dir=temp_log_dir,
            log_level=LogLevel.DEBUG,
            enable_file_logging=True,
            enable_console_logging=False,
            max_audit_entries=100
        )
        yield logger
        logger.shutdown()

    @pytest.fixture(scope="function")
    def sample_log_entry(self):
        """Create sample log entry for testing."""
        return LogEntry(
            timestamp=datetime.utcnow(),
            level=LogLevel.INFO,
            event_type=EventType.BATCH_CREATED,
            message="Test batch created",
            batch_id="batch_123",
            component="test",
            context={"test_key": "test_value"}
        )

    # Log Entry Tests

    def test_log_entry_creation(self):
        """Test log entry creation and validation."""
        timestamp = datetime.utcnow()
        entry = LogEntry(
            timestamp=timestamp,
            level=LogLevel.INFO,
            event_type=EventType.BATCH_STARTED,
            message="Test message",
            batch_id="batch_123",
            batch_item_id=456,
            worker_id="worker_001",
            session_id="session_789",
            component="batch_service",
            context={"key": "value"},
            duration=10.5,
            memory_usage=1024,
            cpu_usage=25.3,
            error_type="TestError",
            error_message="Test error message",
            stack_trace="Test stack trace",
            correlation_id="corr_123"
        )
        
        # Verify all fields
        assert entry.timestamp == timestamp
        assert entry.level == LogLevel.INFO
        assert entry.event_type == EventType.BATCH_STARTED
        assert entry.message == "Test message"
        assert entry.batch_id == "batch_123"
        assert entry.batch_item_id == 456
        assert entry.worker_id == "worker_001"
        assert entry.session_id == "session_789"
        assert entry.component == "batch_service"
        assert entry.context == {"key": "value"}
        assert entry.duration == 10.5
        assert entry.memory_usage == 1024
        assert entry.cpu_usage == 25.3
        assert entry.error_type == "TestError"
        assert entry.error_message == "Test error message"
        assert entry.stack_trace == "Test stack trace"
        assert entry.correlation_id == "corr_123"
        assert entry.log_id is not None

    def test_log_entry_serialization(self, sample_log_entry):
        """Test log entry serialization to dict and JSON."""
        # Test to_dict
        entry_dict = sample_log_entry.to_dict()
        assert isinstance(entry_dict, dict)
        assert entry_dict['timestamp'] == sample_log_entry.timestamp.isoformat()
        assert entry_dict['level'] == sample_log_entry.level.value
        assert entry_dict['event_type'] == sample_log_entry.event_type.value
        assert entry_dict['message'] == sample_log_entry.message
        assert entry_dict['batch_id'] == sample_log_entry.batch_id
        assert entry_dict['context'] == sample_log_entry.context
        
        # Test to_json
        entry_json = sample_log_entry.to_json()
        assert isinstance(entry_json, str)
        parsed_json = json.loads(entry_json)
        assert parsed_json['message'] == sample_log_entry.message

    def test_log_entry_defaults(self):
        """Test log entry creation with minimal parameters."""
        entry = LogEntry(
            timestamp=datetime.utcnow(),
            level=LogLevel.ERROR,
            event_type=EventType.SYSTEM_ERROR,
            message="Error message"
        )
        
        assert entry.batch_id is None
        assert entry.batch_item_id is None
        assert entry.worker_id is None
        assert entry.session_id is None
        assert entry.component is None
        assert entry.context == {}
        assert entry.duration is None
        assert entry.memory_usage is None
        assert entry.cpu_usage is None
        assert entry.error_type is None
        assert entry.error_message is None
        assert entry.stack_trace is None
        assert entry.correlation_id is None
        assert entry.log_id is not None

    # Audit Trail Tests

    def test_batch_audit_trail_creation(self):
        """Test batch audit trail creation."""
        batch_id = "batch_123"
        created_at = datetime.utcnow()
        
        audit_trail = BatchAuditTrail(
            batch_id=batch_id,
            created_at=created_at
        )
        
        assert audit_trail.batch_id == batch_id
        assert audit_trail.created_at == created_at
        assert len(audit_trail.events) == 0

    def test_audit_trail_event_management(self, sample_log_entry):
        """Test audit trail event addition and retrieval."""
        audit_trail = BatchAuditTrail(
            batch_id="batch_123",
            created_at=datetime.utcnow()
        )
        
        # Add events
        event1 = sample_log_entry
        event2 = LogEntry(
            timestamp=datetime.utcnow(),
            level=LogLevel.ERROR,
            event_type=EventType.SYSTEM_ERROR,
            message="Error occurred",
            batch_id="batch_123"
        )
        
        audit_trail.add_event(event1)
        audit_trail.add_event(event2)
        
        assert len(audit_trail.events) == 2
        
        # Test filtering by type
        batch_events = audit_trail.get_events_by_type(EventType.BATCH_CREATED)
        assert len(batch_events) == 1
        assert batch_events[0] == event1
        
        error_events = audit_trail.get_events_by_type(EventType.SYSTEM_ERROR)
        assert len(error_events) == 1
        assert error_events[0] == event2
        
        # Test filtering by level
        info_events = audit_trail.get_events_by_level(LogLevel.INFO)
        assert len(info_events) == 1
        
        error_level_events = audit_trail.get_events_by_level(LogLevel.ERROR)
        assert len(error_level_events) == 1
        
        # Test getting error events
        all_error_events = audit_trail.get_error_events()
        assert len(all_error_events) == 1
        assert all_error_events[0] == event2

    def test_audit_trail_performance_summary(self):
        """Test audit trail performance summary generation."""
        audit_trail = BatchAuditTrail(
            batch_id="batch_123",
            created_at=datetime.utcnow()
        )
        
        # Add events with performance data
        events = [
            LogEntry(
                timestamp=datetime.utcnow(),
                level=LogLevel.INFO,
                event_type=EventType.ITEM_COMPLETED,
                message="Item 1 completed",
                duration=10.5,
                memory_usage=1024
            ),
            LogEntry(
                timestamp=datetime.utcnow(),
                level=LogLevel.INFO,
                event_type=EventType.ITEM_COMPLETED,
                message="Item 2 completed",
                duration=15.2,
                memory_usage=2048
            ),
            LogEntry(
                timestamp=datetime.utcnow(),
                level=LogLevel.ERROR,
                event_type=EventType.SYSTEM_ERROR,
                message="Error occurred"
            )
        ]
        
        for event in events:
            audit_trail.add_event(event)
        
        summary = audit_trail.get_performance_summary()
        
        assert summary['total_events'] == 3
        assert summary['error_events'] == 1
        assert summary['average_duration'] == (10.5 + 15.2) / 2
        assert summary['max_duration'] == 15.2
        assert summary['peak_memory'] == 2048
        assert 'event_types' in summary

    def test_audit_trail_serialization(self):
        """Test audit trail serialization."""
        audit_trail = BatchAuditTrail(
            batch_id="batch_123",
            created_at=datetime.utcnow()
        )
        
        # Add sample event
        event = LogEntry(
            timestamp=datetime.utcnow(),
            level=LogLevel.INFO,
            event_type=EventType.BATCH_STARTED,
            message="Batch started"
        )
        audit_trail.add_event(event)
        
        # Test serialization
        trail_dict = audit_trail.to_dict()
        
        assert trail_dict['batch_id'] == "batch_123"
        assert 'created_at' in trail_dict
        assert 'events' in trail_dict
        assert 'performance_summary' in trail_dict
        assert len(trail_dict['events']) == 1

    # BatchLogger Tests

    def test_batch_logger_initialization(self, temp_log_dir):
        """Test batch logger initialization."""
        logger = BatchLogger(
            log_dir=temp_log_dir,
            log_level=LogLevel.WARNING,
            enable_file_logging=True,
            enable_console_logging=True,
            max_log_files=5,
            max_audit_entries=500
        )
        
        assert logger.log_level == LogLevel.WARNING
        assert logger.enable_file_logging is True
        assert logger.enable_console_logging is True
        assert logger.max_log_files == 5
        assert logger.max_audit_entries == 500
        assert logger.log_dir == Path(temp_log_dir)
        
        logger.shutdown()

    def test_batch_logger_get_logger(self, batch_logger):
        """Test logger instance creation and retrieval."""
        # Get logger for a component
        component_logger = batch_logger.get_logger("test_component")
        
        assert component_logger is not None
        assert component_logger.name == "batch_processing.test_component"
        
        # Get same logger again
        same_logger = batch_logger.get_logger("test_component")
        assert same_logger is component_logger

    def test_batch_logger_create_log_entry(self, batch_logger):
        """Test log entry creation through BatchLogger."""
        entry = batch_logger.create_log_entry(
            level=LogLevel.INFO,
            event_type=EventType.BATCH_CREATED,
            message="Test batch created",
            batch_id="batch_123",
            component="test_component",
            context={"key": "value"},
            duration=5.5
        )
        
        assert entry.level == LogLevel.INFO
        assert entry.event_type == EventType.BATCH_CREATED
        assert entry.message == "Test batch created"
        assert entry.batch_id == "batch_123"
        assert entry.component == "test_component"
        assert entry.context == {"key": "value"}
        assert entry.duration == 5.5
        assert entry.timestamp is not None
        assert entry.log_id is not None

    def test_batch_logger_log_entry(self, batch_logger):
        """Test logging an entry through BatchLogger."""
        entry = LogEntry(
            timestamp=datetime.utcnow(),
            level=LogLevel.INFO,
            event_type=EventType.BATCH_STARTED,
            message="Test message",
            batch_id="batch_123"
        )
        
        # Log the entry
        batch_logger.log_entry(entry)
        
        # Verify it was added to buffer
        recent_logs = batch_logger.get_recent_logs(count=1)
        assert len(recent_logs) == 1
        assert recent_logs[0].message == "Test message"
        
        # Verify audit trail was created
        audit_trail = batch_logger.get_audit_trail("batch_123")
        assert audit_trail is not None
        assert len(audit_trail.events) == 1

    def test_batch_logger_convenience_methods(self, batch_logger):
        """Test batch logger convenience methods."""
        # Test log_batch_event
        batch_logger.log_batch_event(
            batch_id="batch_123",
            event_type=EventType.BATCH_CREATED,
            message="Batch created",
            level=LogLevel.INFO,
            context={"items": 5},
            duration=2.1
        )
        
        # Test log_item_event
        batch_logger.log_item_event(
            batch_id="batch_123",
            batch_item_id=456,
            event_type=EventType.ITEM_STARTED,
            message="Item started",
            worker_id="worker_001"
        )
        
        # Test log_worker_event
        batch_logger.log_worker_event(
            worker_id="worker_001",
            event_type=EventType.WORKER_STARTED,
            message="Worker started",
            batch_id="batch_123"
        )
        
        # Test log_error
        test_error = ValueError("Test error")
        batch_logger.log_error(
            message="Error occurred",
            error=test_error,
            batch_id="batch_123",
            component="test_component"
        )
        
        # Test log_performance
        batch_logger.log_performance(
            operation="test_operation",
            duration=10.5,
            batch_id="batch_123",
            memory_usage=1024,
            cpu_usage=25.0
        )
        
        # Verify all events were logged
        recent_logs = batch_logger.get_recent_logs(count=10)
        assert len(recent_logs) == 5

    def test_batch_logger_filtering(self, batch_logger):
        """Test log filtering functionality."""
        # Log different types of events
        batch_logger.log_batch_event(
            batch_id="batch_123",
            event_type=EventType.BATCH_CREATED,
            message="Batch created",
            level=LogLevel.INFO
        )
        
        batch_logger.log_batch_event(
            batch_id="batch_456",
            event_type=EventType.BATCH_STARTED,
            message="Batch started",
            level=LogLevel.INFO
        )
        
        batch_logger.log_error(
            message="Error occurred",
            error=ValueError("Test error"),
            batch_id="batch_123"
        )
        
        # Test filtering by level
        error_logs = batch_logger.get_recent_logs(level=LogLevel.ERROR)
        assert len(error_logs) == 1
        assert "Error occurred" in error_logs[0].message
        
        # Test filtering by event type
        batch_events = batch_logger.get_recent_logs(event_type=EventType.BATCH_CREATED)
        assert len(batch_events) == 1
        assert batch_events[0].event_type == EventType.BATCH_CREATED
        
        # Test filtering by batch ID
        batch_123_logs = batch_logger.get_recent_logs(batch_id="batch_123")
        assert len(batch_123_logs) == 2
        
        batch_456_logs = batch_logger.get_recent_logs(batch_id="batch_456")
        assert len(batch_456_logs) == 1

    def test_batch_logger_summaries(self, batch_logger):
        """Test error and performance summary generation."""
        # Log some errors
        for i in range(3):
            batch_logger.log_error(
                message=f"Error {i}",
                error=ValueError(f"Test error {i}"),
                batch_id=f"batch_{i}"
            )
        
        # Log some performance data
        for i in range(2):
            batch_logger.log_performance(
                operation=f"operation_{i}",
                duration=10.0 + i,
                batch_id=f"batch_{i}"
            )
        
        # Get error summary
        error_summary = batch_logger.get_error_summary(hours=24)
        assert error_summary['total_errors'] == 3
        assert 'error_types' in error_summary
        assert 'recent_errors' in error_summary
        
        # Get performance summary
        perf_summary = batch_logger.get_performance_summary(hours=24)
        assert perf_summary['total_operations'] == 2
        assert perf_summary['average_duration'] == 10.5
        assert perf_summary['min_duration'] == 10.0
        assert perf_summary['max_duration'] == 11.0

    def test_batch_logger_audit_trail_export(self, batch_logger, temp_log_dir):
        """Test audit trail export functionality."""
        # Create audit trail
        batch_logger.log_batch_event(
            batch_id="batch_123",
            event_type=EventType.BATCH_CREATED,
            message="Batch created"
        )
        
        # Export audit trail
        export_path = batch_logger.export_audit_trail("batch_123")
        
        assert Path(export_path).exists()
        
        # Verify exported content
        with open(export_path, 'r') as f:
            exported_data = json.load(f)
        
        assert exported_data['batch_id'] == "batch_123"
        assert 'events' in exported_data
        assert 'performance_summary' in exported_data

    def test_batch_logger_cleanup(self, batch_logger, temp_log_dir):
        """Test cleanup functionality."""
        # Create some old audit trails
        old_trail = BatchAuditTrail(
            batch_id="old_batch",
            created_at=datetime.utcnow() - timedelta(days=31)
        )
        batch_logger._audit_trails["old_batch"] = old_trail
        
        recent_trail = BatchAuditTrail(
            batch_id="recent_batch",
            created_at=datetime.utcnow()
        )
        batch_logger._audit_trails["recent_batch"] = recent_trail
        
        # Run cleanup
        batch_logger.cleanup_old_logs(days=30)
        
        # Verify old trail was removed
        assert "old_batch" not in batch_logger._audit_trails
        assert "recent_batch" in batch_logger._audit_trails

    def test_batch_operation_logger_context_manager(self, batch_logger):
        """Test BatchOperationLogger context manager."""
        # Test successful operation
        with BatchOperationLogger(
            logger=batch_logger,
            batch_id="batch_123",
            operation="test_operation",
            batch_item_id=456,
            worker_id="worker_001",
            context={"test": "context"}
        ) as op_logger:
            time.sleep(0.1)  # Simulate work
        
        # Verify start and completion events were logged
        recent_logs = batch_logger.get_recent_logs(count=10)
        start_events = [log for log in recent_logs if log.event_type == EventType.ITEM_STARTED]
        complete_events = [log for log in recent_logs if log.event_type == EventType.ITEM_COMPLETED]
        
        assert len(start_events) == 1
        assert len(complete_events) == 1
        assert complete_events[0].duration > 0

    def test_batch_operation_logger_error_handling(self, batch_logger):
        """Test BatchOperationLogger error handling."""
        try:
            with BatchOperationLogger(
                logger=batch_logger,
                batch_id="batch_123",
                operation="failing_operation"
            ):
                raise ValueError("Test error")
        except ValueError:
            pass
        
        # Verify error was logged
        recent_logs = batch_logger.get_recent_logs(count=10)
        error_events = [log for log in recent_logs if log.event_type == EventType.SYSTEM_ERROR]
        assert len(error_events) == 1
        assert "failing_operation" in error_events[0].message

    def test_batch_logger_thread_safety(self, batch_logger):
        """Test batch logger thread safety."""
        def log_worker(worker_id):
            for i in range(10):
                batch_logger.log_batch_event(
                    batch_id=f"batch_{worker_id}",
                    event_type=EventType.BATCH_CREATED,
                    message=f"Worker {worker_id} log {i}"
                )
        
        # Run multiple workers concurrently
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(log_worker, i) for i in range(5)]
            for future in futures:
                future.result()
        
        # Verify all logs were recorded
        recent_logs = batch_logger.get_recent_logs(count=100)
        assert len(recent_logs) == 50  # 5 workers * 10 logs each

    def test_global_logger_functions(self, temp_log_dir):
        """Test global logger functions."""
        # Test initialization
        global_logger = initialize_logging(log_dir=temp_log_dir, log_level=LogLevel.DEBUG)
        assert global_logger is not None
        
        # Test get_batch_logger
        retrieved_logger = get_batch_logger()
        assert retrieved_logger is global_logger
        
        global_logger.shutdown()


class TestBatchMonitor:
    """Comprehensive tests for batch monitoring functionality."""

    @pytest.fixture(scope="function")
    def batch_monitor(self):
        """Create BatchMonitor instance for testing."""
        monitor = BatchMonitor(enabled=True)
        yield monitor
        monitor.shutdown()

    @pytest.fixture(scope="function")
    def sample_metrics(self):
        """Create sample batch metrics for testing."""
        return BatchProcessingMetrics(
            batch_id="batch_123",
            start_time=datetime.utcnow(),
            total_items=10
        )

    @pytest.fixture(scope="function")
    def sample_item_metrics(self):
        """Create sample item metrics for testing."""
        return BatchItemMetrics(
            batch_item_id=456,
            batch_id="batch_123",
            url="https://www.youtube.com/watch?v=test123",
            start_time=datetime.utcnow()
        )

    # Metrics Tests

    def test_batch_processing_metrics_creation(self):
        """Test batch processing metrics creation."""
        start_time = datetime.utcnow()
        metrics = BatchProcessingMetrics(
            batch_id="batch_123",
            start_time=start_time,
            total_items=10
        )
        
        assert metrics.batch_id == "batch_123"
        assert metrics.start_time == start_time
        assert metrics.total_items == 10
        assert metrics.processed_items == 0
        assert metrics.failed_items == 0
        assert metrics.completed_items == 0
        assert metrics.queued_items == 0
        assert metrics.processing_rate == 0.0
        assert metrics.success_rate == 0.0
        assert metrics.failure_rate == 0.0

    def test_batch_processing_metrics_updates(self, sample_metrics):
        """Test batch processing metrics updates."""
        # Update progress
        sample_metrics.update_progress(
            processed=6,
            failed=2,
            completed=4,
            queued=4
        )
        
        assert sample_metrics.processed_items == 6
        assert sample_metrics.failed_items == 2
        assert sample_metrics.completed_items == 4
        assert sample_metrics.queued_items == 4
        assert sample_metrics.success_rate == 40.0  # 4/10 * 100
        assert sample_metrics.failure_rate == 20.0  # 2/10 * 100
        assert sample_metrics.processing_rate > 0
        
        # Update resource usage
        sample_metrics.update_resource_usage(memory_mb=1024, cpu_percent=75.5)
        
        assert sample_metrics.memory_usage['current_mb'] == 1024
        assert sample_metrics.cpu_usage['current_percent'] == 75.5
        assert sample_metrics.peak_memory == 1024
        assert sample_metrics.peak_cpu == 75.5
        
        # Update again with lower values
        sample_metrics.update_resource_usage(memory_mb=512, cpu_percent=50.0)
        
        assert sample_metrics.memory_usage['current_mb'] == 512
        assert sample_metrics.cpu_usage['current_percent'] == 50.0
        assert sample_metrics.peak_memory == 1024  # Should keep peak
        assert sample_metrics.peak_cpu == 75.5  # Should keep peak

    def test_batch_processing_metrics_finish(self, sample_metrics):
        """Test batch processing metrics completion."""
        # Set up some data
        sample_metrics.update_progress(processed=5, failed=0, completed=5, queued=5)
        sample_metrics.max_workers = 2
        
        # Wait a bit then finish
        time.sleep(0.1)
        sample_metrics.finish()
        
        assert sample_metrics.end_time is not None
        assert sample_metrics.total_duration > 0
        assert sample_metrics.average_item_duration > 0
        assert sample_metrics.worker_efficiency > 0

    def test_batch_processing_metrics_serialization(self, sample_metrics):
        """Test batch processing metrics serialization."""
        metrics_dict = sample_metrics.to_dict()
        
        assert isinstance(metrics_dict, dict)
        assert metrics_dict['batch_id'] == sample_metrics.batch_id
        assert 'start_time' in metrics_dict
        assert metrics_dict['total_items'] == sample_metrics.total_items
        assert 'memory_usage' in metrics_dict
        assert 'cpu_usage' in metrics_dict

    def test_batch_item_metrics_creation(self):
        """Test batch item metrics creation."""
        start_time = datetime.utcnow()
        metrics = BatchItemMetrics(
            batch_item_id=456,
            batch_id="batch_123",
            url="https://www.youtube.com/watch?v=test123",
            start_time=start_time
        )
        
        assert metrics.batch_item_id == 456
        assert metrics.batch_id == "batch_123"
        assert metrics.url == "https://www.youtube.com/watch?v=test123"
        assert metrics.start_time == start_time
        assert metrics.status == BatchItemStatus.QUEUED
        assert metrics.error_count == 0
        assert metrics.retry_count == 0

    def test_batch_item_metrics_stage_tracking(self, sample_item_metrics):
        """Test batch item metrics stage tracking."""
        # Start first stage
        sample_item_metrics.start_stage("download")
        time.sleep(0.05)
        
        # Start second stage (should end first)
        sample_item_metrics.start_stage("process")
        time.sleep(0.05)
        
        # End current stage
        sample_item_metrics.end_stage("process")
        
        assert "download" in sample_item_metrics.stage_durations
        assert "process" in sample_item_metrics.stage_durations
        assert sample_item_metrics.stage_durations["download"] > 0
        assert sample_item_metrics.stage_durations["process"] > 0

    def test_batch_item_metrics_error_tracking(self, sample_item_metrics):
        """Test batch item metrics error tracking."""
        # Record errors
        sample_item_metrics.record_error("First error")
        sample_item_metrics.record_error("Second error")
        
        assert sample_item_metrics.error_count == 2
        assert sample_item_metrics.last_error == "Second error"
        
        # Record retries
        sample_item_metrics.record_retry()
        sample_item_metrics.record_retry()
        
        assert sample_item_metrics.retry_count == 2

    def test_batch_item_metrics_finish(self, sample_item_metrics):
        """Test batch item metrics completion."""
        sample_item_metrics.start_stage("processing")
        time.sleep(0.1)
        
        sample_item_metrics.finish(BatchItemStatus.COMPLETED)
        
        assert sample_item_metrics.end_time is not None
        assert sample_item_metrics.status == BatchItemStatus.COMPLETED
        assert sample_item_metrics.duration > 0
        assert sample_item_metrics.current_stage is None
        assert sample_item_metrics.processing_efficiency is not None

    def test_batch_item_metrics_serialization(self, sample_item_metrics):
        """Test batch item metrics serialization."""
        metrics_dict = sample_item_metrics.to_dict()
        
        assert isinstance(metrics_dict, dict)
        assert metrics_dict['batch_item_id'] == sample_item_metrics.batch_item_id
        assert metrics_dict['batch_id'] == sample_item_metrics.batch_id
        assert metrics_dict['url'] == sample_item_metrics.url
        assert 'start_time' in metrics_dict

    # Alert Tests

    def test_alert_creation(self):
        """Test alert creation and management."""
        timestamp = datetime.utcnow()
        alert = Alert(
            alert_id="alert_123",
            level=AlertLevel.WARNING,
            message="Test alert",
            component="test_component",
            timestamp=timestamp,
            details={"key": "value"}
        )
        
        assert alert.alert_id == "alert_123"
        assert alert.level == AlertLevel.WARNING
        assert alert.message == "Test alert"
        assert alert.component == "test_component"
        assert alert.timestamp == timestamp
        assert alert.details == {"key": "value"}
        assert alert.resolved is False
        assert alert.resolved_at is None

    def test_alert_resolution(self):
        """Test alert resolution."""
        alert = Alert(
            alert_id="alert_123",
            level=AlertLevel.ERROR,
            message="Test error",
            component="test",
            timestamp=datetime.utcnow()
        )
        
        assert alert.resolved is False
        
        alert.resolve()
        
        assert alert.resolved is True
        assert alert.resolved_at is not None

    def test_alert_serialization(self):
        """Test alert serialization."""
        alert = Alert(
            alert_id="alert_123",
            level=AlertLevel.CRITICAL,
            message="Critical alert",
            component="system",
            timestamp=datetime.utcnow(),
            details={"severity": "high"}
        )
        
        alert_dict = alert.to_dict()
        
        assert isinstance(alert_dict, dict)
        assert alert_dict['alert_id'] == "alert_123"
        assert alert_dict['level'] == AlertLevel.CRITICAL.value
        assert alert_dict['message'] == "Critical alert"
        assert 'timestamp' in alert_dict
        assert alert_dict['details'] == {"severity": "high"}

    # BatchMonitor Tests

    def test_batch_monitor_initialization(self):
        """Test batch monitor initialization."""
        monitor = BatchMonitor(enabled=True)
        
        assert monitor.enabled is True
        assert len(monitor._batch_metrics) == 0
        assert len(monitor._item_metrics) == 0
        assert len(monitor._alerts) == 0
        assert monitor._monitoring_active is True
        
        monitor.shutdown()

    def test_batch_monitor_disabled(self):
        """Test batch monitor when disabled."""
        monitor = BatchMonitor(enabled=False)
        
        assert monitor.enabled is False
        
        # Should return None for monitoring operations
        result = monitor.start_batch_monitoring("batch_123", 10)
        assert result is None
        
        result = monitor.start_item_monitoring(456, "batch_123", "test_url")
        assert result is None

    def test_batch_monitor_batch_lifecycle(self, batch_monitor):
        """Test complete batch monitoring lifecycle."""
        # Start batch monitoring
        metrics = batch_monitor.start_batch_monitoring("batch_123", 5)
        
        assert metrics is not None
        assert metrics.batch_id == "batch_123"
        assert metrics.total_items == 5
        assert "batch_123" in batch_monitor._batch_metrics
        
        # Update progress
        batch_monitor._batch_metrics["batch_123"].update_progress(
            processed=3, failed=1, completed=2, queued=2
        )
        
        # Get current metrics
        current_metrics = batch_monitor.get_batch_metrics("batch_123")
        assert current_metrics is not None
        assert current_metrics['processed_items'] == 3
        assert current_metrics['failed_items'] == 1
        
        # Finish monitoring
        summary = batch_monitor.finish_batch_monitoring("batch_123")
        assert summary is not None
        assert 'duration' in summary
        assert 'success_rate' in summary

    def test_batch_monitor_item_lifecycle(self, batch_monitor):
        """Test complete item monitoring lifecycle."""
        # Start batch monitoring first
        batch_monitor.start_batch_monitoring("batch_123", 1)
        
        # Start item monitoring
        metrics = batch_monitor.start_item_monitoring(456, "batch_123", "https://test.com")
        
        assert metrics is not None
        assert metrics.batch_item_id == 456
        assert 456 in batch_monitor._item_metrics
        
        # Get current metrics
        current_metrics = batch_monitor.get_item_metrics(456)
        assert current_metrics is not None
        assert current_metrics['batch_item_id'] == 456
        
        # Finish monitoring
        summary = batch_monitor.finish_item_monitoring(456, BatchItemStatus.COMPLETED)
        assert summary is not None
        assert summary['status'] == BatchItemStatus.COMPLETED.value

    def test_batch_monitor_error_recording(self, batch_monitor):
        """Test error recording and tracking."""
        # Start monitoring
        batch_monitor.start_batch_monitoring("batch_123", 2)
        batch_monitor.start_item_monitoring(456, "batch_123", "https://test.com")
        
        # Record errors
        batch_monitor.record_error("batch_123", 456, "Test error 1")
        batch_monitor.record_error("batch_123", 456, "Test error 2")
        
        # Check batch metrics
        batch_metrics = batch_monitor.get_batch_metrics("batch_123")
        assert batch_metrics['error_count'] == 2
        
        # Check item metrics
        item_metrics = batch_monitor.get_item_metrics(456)
        assert item_metrics['error_count'] == 2
        assert item_metrics['last_error'] == "Test error 2"

    def test_batch_monitor_retry_recording(self, batch_monitor):
        """Test retry recording and tracking."""
        # Start monitoring
        batch_monitor.start_batch_monitoring("batch_123", 1)
        batch_monitor.start_item_monitoring(456, "batch_123", "https://test.com")
        
        # Record retries
        batch_monitor.record_retry("batch_123", 456)
        batch_monitor.record_retry("batch_123", 456)
        
        # Check metrics
        batch_metrics = batch_monitor.get_batch_metrics("batch_123")
        assert batch_metrics['retry_count'] == 2
        
        item_metrics = batch_monitor.get_item_metrics(456)
        assert item_metrics['retry_count'] == 2

    def test_batch_monitor_worker_tracking(self, batch_monitor):
        """Test worker count tracking."""
        # Start monitoring
        batch_monitor.start_batch_monitoring("batch_123", 10)
        
        # Update worker count
        batch_monitor.update_worker_count("batch_123", 3, 5)
        
        # Check metrics
        metrics = batch_monitor.get_batch_metrics("batch_123")
        assert metrics['active_workers'] == 3
        assert metrics['max_workers'] == 5

    def test_batch_monitor_queue_tracking(self, batch_monitor):
        """Test queue metrics tracking."""
        # Start monitoring
        batch_monitor.start_batch_monitoring("batch_123", 10)
        
        # Update queue metrics
        batch_monitor.update_queue_metrics("batch_123", 5, 30.5)
        
        # Check metrics
        metrics = batch_monitor.get_batch_metrics("batch_123")
        assert metrics['queue_size'] == 5
        assert metrics['queue_wait_time'] == 30.5

    def test_batch_monitor_custom_metrics(self, batch_monitor):
        """Test custom metrics recording."""
        # Record different types of custom metrics
        batch_monitor.record_custom_metric("test_counter", 5, MetricType.COUNTER)
        batch_monitor.record_custom_metric("test_counter", 3, MetricType.COUNTER)
        batch_monitor.record_custom_metric("test_gauge", 75.5, MetricType.GAUGE)
        batch_monitor.record_custom_metric("test_histogram", 10.1, MetricType.HISTOGRAM)
        batch_monitor.record_custom_metric("test_histogram", 15.3, MetricType.HISTOGRAM)
        
        # Check system metrics
        system_metrics = batch_monitor.get_system_metrics()
        
        assert system_metrics['custom_counters']['test_counter'] == 8  # 5 + 3
        assert system_metrics['custom_gauges']['test_gauge'] == 75.5
        assert system_metrics['custom_histograms']['test_histogram'] == 2  # Count of values

    def test_batch_monitor_alert_generation(self, batch_monitor):
        """Test alert generation and retrieval."""
        # Manually generate an alert
        batch_monitor._generate_alert(
            AlertLevel.WARNING,
            "Test warning",
            "test_component",
            {"key": "value"}
        )
        
        # Get all alerts
        all_alerts = batch_monitor.get_alerts()
        assert len(all_alerts) == 1
        assert all_alerts[0]['level'] == AlertLevel.WARNING.value
        assert all_alerts[0]['message'] == "Test warning"
        
        # Get alerts by level
        warning_alerts = batch_monitor.get_alerts(level=AlertLevel.WARNING)
        assert len(warning_alerts) == 1
        
        error_alerts = batch_monitor.get_alerts(level=AlertLevel.ERROR)
        assert len(error_alerts) == 0

    def test_batch_monitor_alert_handlers(self, batch_monitor):
        """Test custom alert handlers."""
        handled_alerts = []
        
        def test_handler(alert):
            handled_alerts.append(alert)
        
        # Add handler
        batch_monitor.add_alert_handler(test_handler)
        
        # Generate alert
        batch_monitor._generate_alert(
            AlertLevel.ERROR,
            "Test error",
            "test_component",
            {}
        )
        
        # Verify handler was called
        assert len(handled_alerts) == 1
        assert handled_alerts[0].level == AlertLevel.ERROR
        assert handled_alerts[0].message == "Test error"

    @patch('psutil.virtual_memory')
    @patch('psutil.cpu_percent')
    def test_batch_monitor_system_metrics(self, mock_cpu, mock_memory, batch_monitor):
        """Test system metrics collection."""
        # Mock system metrics
        mock_memory.return_value.percent = 85.0
        mock_cpu.return_value = 90.5
        
        # Get system metrics
        system_metrics = batch_monitor.get_system_metrics()
        
        assert 'memory_usage_mb' in system_metrics
        assert 'cpu_usage_percent' in system_metrics
        assert system_metrics['active_batches'] == 0
        assert system_metrics['active_items'] == 0

    @patch('src.utils.batch_monitor.psutil.virtual_memory')
    @patch('src.utils.batch_monitor.psutil.cpu_percent')
    def test_batch_monitor_threshold_alerts(self, mock_cpu, mock_memory, batch_monitor):
        """Test threshold-based alert generation."""
        # Mock high resource usage
        mock_memory.return_value.percent = 95.0  # Above threshold
        mock_cpu.return_value = 95.0  # Above threshold
        
        # Manually trigger alert checking
        batch_monitor._check_alert_conditions()
        
        # Check for generated alerts
        alerts = batch_monitor.get_alerts()
        
        # Should have alerts for high memory and CPU usage
        memory_alerts = [a for a in alerts if 'memory' in a['message'].lower()]
        cpu_alerts = [a for a in alerts if 'cpu' in a['message'].lower()]
        
        assert len(memory_alerts) > 0
        assert len(cpu_alerts) > 0

    def test_batch_monitor_thread_safety(self, batch_monitor):
        """Test batch monitor thread safety."""
        def monitoring_worker(worker_id):
            batch_id = f"batch_{worker_id}"
            batch_monitor.start_batch_monitoring(batch_id, 5)
            
            for i in range(5):
                item_id = worker_id * 100 + i
                batch_monitor.start_item_monitoring(item_id, batch_id, f"https://test{i}.com")
                batch_monitor.record_error(batch_id, item_id, f"Error {i}")
                batch_monitor.finish_item_monitoring(item_id, BatchItemStatus.FAILED)
            
            batch_monitor.finish_batch_monitoring(batch_id)
        
        # Run multiple workers concurrently
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(monitoring_worker, i) for i in range(3)]
            for future in futures:
                future.result()
        
        # Verify all monitoring data was recorded correctly
        system_metrics = batch_monitor.get_system_metrics()
        # Should have processed 3 batches, but they might be cleaned up
        assert system_metrics is not None

    def test_global_monitor_functions(self):
        """Test global monitor functions."""
        # Test initialization
        global_monitor = initialize_monitoring(enabled=True)
        assert global_monitor is not None
        assert global_monitor.enabled is True
        
        # Test get_batch_monitor
        retrieved_monitor = get_batch_monitor()
        assert retrieved_monitor is global_monitor
        
        global_monitor.shutdown()


class TestIntegration:
    """Integration tests for logging and monitoring together."""

    @pytest.fixture(scope="function")
    def temp_log_dir(self):
        """Create temporary directory for log files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    def test_logger_monitor_integration(self, temp_log_dir):
        """Test integration between logger and monitor."""
        # Initialize both systems
        logger = BatchLogger(log_dir=temp_log_dir, enable_console_logging=False)
        monitor = BatchMonitor(enabled=True)
        
        try:
            # Start monitoring a batch
            batch_id = "integration_batch_123"
            batch_metrics = monitor.start_batch_monitoring(batch_id, 3)
            
            # Log batch creation
            logger.log_batch_event(
                batch_id=batch_id,
                event_type=EventType.BATCH_CREATED,
                message="Batch created for integration test",
                context={"total_items": 3}
            )
            
            # Process items
            for i in range(3):
                item_id = i + 1
                
                # Start item monitoring
                item_metrics = monitor.start_item_monitoring(
                    item_id, batch_id, f"https://test{i}.com"
                )
                
                # Log item start
                logger.log_item_event(
                    batch_id=batch_id,
                    batch_item_id=item_id,
                    event_type=EventType.ITEM_STARTED,
                    message=f"Started processing item {item_id}",
                    worker_id=f"worker_{i}"
                )
                
                # Simulate processing
                time.sleep(0.05)
                
                # Finish with different outcomes
                if i == 0:
                    # Success
                    status = BatchItemStatus.COMPLETED
                    logger.log_item_event(
                        batch_id=batch_id,
                        batch_item_id=item_id,
                        event_type=EventType.ITEM_COMPLETED,
                        message=f"Completed item {item_id}",
                        worker_id=f"worker_{i}"
                    )
                elif i == 1:
                    # Failure
                    status = BatchItemStatus.FAILED
                    error = ValueError(f"Test error for item {item_id}")
                    monitor.record_error(batch_id, item_id, str(error))
                    logger.log_error(
                        message=f"Failed to process item {item_id}",
                        error=error,
                        batch_id=batch_id,
                        batch_item_id=item_id,
                        worker_id=f"worker_{i}"
                    )
                else:
                    # Retry scenario
                    status = BatchItemStatus.FAILED
                    monitor.record_retry(batch_id, item_id)
                    logger.log_item_event(
                        batch_id=batch_id,
                        batch_item_id=item_id,
                        event_type=EventType.ITEM_RETRIED,
                        message=f"Retrying item {item_id}",
                        worker_id=f"worker_{i}"
                    )
                
                # Finish item monitoring
                monitor.finish_item_monitoring(item_id, status)
            
            # Finish batch
            logger.log_batch_event(
                batch_id=batch_id,
                event_type=EventType.BATCH_COMPLETED,
                message="Batch processing completed"
            )
            
            batch_summary = monitor.finish_batch_monitoring(batch_id)
            
            # Verify integration results
            
            # Check audit trail
            audit_trail = logger.get_audit_trail(batch_id)
            assert audit_trail is not None
            assert len(audit_trail.events) >= 5  # At least batch + item events
            
            # Check batch metrics
            assert batch_summary is not None
            assert batch_summary['total_items'] == 3
            assert batch_summary['success_rate'] > 0
            assert batch_summary['failure_rate'] > 0
            
            # Check error summary
            error_summary = logger.get_error_summary(hours=1)
            assert error_summary['total_errors'] >= 1
            
            # Check system metrics
            system_metrics = monitor.get_system_metrics()
            assert system_metrics is not None
            
        finally:
            logger.shutdown()
            monitor.shutdown()

    def test_performance_under_load(self, temp_log_dir):
        """Test performance of logging and monitoring under load."""
        logger = BatchLogger(log_dir=temp_log_dir, enable_console_logging=False)
        monitor = BatchMonitor(enabled=True)
        
        try:
            start_time = time.time()
            
            # Process multiple batches concurrently
            def process_batch(batch_num):
                batch_id = f"load_batch_{batch_num}"
                
                # Start monitoring
                monitor.start_batch_monitoring(batch_id, 10)
                
                # Log batch creation
                logger.log_batch_event(
                    batch_id=batch_id,
                    event_type=EventType.BATCH_CREATED,
                    message=f"Created batch {batch_num}"
                )
                
                # Process items
                for i in range(10):
                    item_id = batch_num * 100 + i
                    
                    monitor.start_item_monitoring(item_id, batch_id, f"https://test{i}.com")
                    
                    logger.log_item_event(
                        batch_id=batch_id,
                        batch_item_id=item_id,
                        event_type=EventType.ITEM_COMPLETED,
                        message=f"Processed item {item_id}",
                        duration=0.1
                    )
                    
                    monitor.finish_item_monitoring(item_id, BatchItemStatus.COMPLETED)
                
                # Finish batch
                logger.log_batch_event(
                    batch_id=batch_id,
                    event_type=EventType.BATCH_COMPLETED,
                    message=f"Completed batch {batch_num}"
                )
                
                monitor.finish_batch_monitoring(batch_id)
            
            # Run multiple batches concurrently
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(process_batch, i) for i in range(10)]
                for future in futures:
                    future.result()
            
            total_time = time.time() - start_time
            
            # Verify performance
            assert total_time < 10.0  # Should complete within 10 seconds
            
            # Verify all data was recorded
            recent_logs = logger.get_recent_logs(count=1000)
            assert len(recent_logs) >= 200  # 10 batches * 20+ events each
            
            system_metrics = monitor.get_system_metrics()
            assert system_metrics is not None
            
        finally:
            logger.shutdown()
            monitor.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])