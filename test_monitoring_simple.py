#!/usr/bin/env python3
"""
Simple test for batch monitoring and logging functionality.

This script tests the core monitoring and logging capabilities
without requiring the full application dependencies.
"""

import sys
import os
import time
from datetime import datetime
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Test imports
try:
    from utils.batch_monitor import BatchMonitor, BatchProcessingMetrics, BatchItemMetrics, AlertLevel
    from utils.batch_logger import BatchLogger, LogLevel, EventType, LogEntry
    print("✓ Successfully imported monitoring and logging modules")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


def test_batch_monitor_standalone():
    """Test BatchMonitor in standalone mode."""
    print("\nTesting BatchMonitor standalone functionality...")
    
    # Create monitor instance
    monitor = BatchMonitor(enabled=True)
    
    # Test batch metrics creation
    batch_id = "test_batch_001"
    metrics = BatchProcessingMetrics(
        batch_id=batch_id,
        start_time=datetime.utcnow(),
        total_items=5
    )
    
    # Test metrics updates
    metrics.update_progress(3, 1, 2, 1)
    metrics.update_resource_usage(512, 75.5)
    
    print(f"✓ Batch metrics: {metrics.success_rate:.1f}% success rate")
    print(f"✓ Processing rate: {metrics.processing_rate:.2f} items/sec")
    
    # Test item metrics
    item_metrics = BatchItemMetrics(
        batch_item_id=1,
        batch_id=batch_id,
        url="https://youtube.com/watch?v=test1",
        start_time=datetime.utcnow()
    )
    
    # Test stage tracking
    item_metrics.start_stage("preparation")
    time.sleep(0.1)
    item_metrics.end_stage("preparation")
    
    item_metrics.start_stage("processing")
    time.sleep(0.1)
    item_metrics.end_stage("processing")
    
    # Test error recording
    item_metrics.record_error("Test error message")
    item_metrics.record_retry()
    
    # Finish item
    from database.batch_models import BatchItemStatus
    item_metrics.finish(BatchItemStatus.COMPLETED)
    
    print(f"✓ Item metrics: {item_metrics.duration:.2f}s duration")
    print(f"✓ Stage durations: {item_metrics.stage_durations}")
    
    # Test metrics serialization
    metrics_dict = metrics.to_dict()
    item_dict = item_metrics.to_dict()
    
    print(f"✓ Metrics serialization: {len(metrics_dict)} fields")
    print(f"✓ Item serialization: {len(item_dict)} fields")
    
    print("✓ BatchMonitor standalone test passed!")


def test_batch_logger_standalone():
    """Test BatchLogger in standalone mode."""
    print("\nTesting BatchLogger standalone functionality...")
    
    # Create logger instance
    temp_dir = Path("/tmp/batch_test_logs")
    temp_dir.mkdir(exist_ok=True)
    
    logger = BatchLogger(
        log_dir=str(temp_dir),
        log_level=LogLevel.INFO,
        enable_file_logging=True,
        enable_console_logging=False
    )
    
    # Test log entry creation
    entry = logger.create_log_entry(
        level=LogLevel.INFO,
        event_type=EventType.BATCH_CREATED,
        message="Test batch created",
        batch_id="test_batch_002",
        context={"total_items": 3, "priority": "normal"}
    )
    
    print(f"✓ Created log entry: {entry.log_id}")
    print(f"✓ Entry timestamp: {entry.timestamp}")
    
    # Test logging
    logger.log_entry(entry)
    
    # Test batch events
    batch_id = "test_batch_002"
    logger.log_batch_event(
        batch_id=batch_id,
        event_type=EventType.BATCH_STARTED,
        message="Started test batch processing",
        context={"worker_count": 2}
    )
    
    # Test item events
    logger.log_item_event(
        batch_id=batch_id,
        batch_item_id=1,
        event_type=EventType.ITEM_STARTED,
        message="Started processing item 1",
        worker_id="worker_1",
        context={"url": "https://youtube.com/watch?v=test1"}
    )
    
    # Test error logging
    try:
        raise ValueError("Test error for logging")
    except Exception as e:
        logger.log_error(
            message="Test error occurred",
            error=e,
            batch_id=batch_id,
            batch_item_id=1,
            worker_id="worker_1"
        )
    
    # Test performance logging
    logger.log_performance(
        operation="test_operation",
        duration=2.5,
        batch_id=batch_id,
        batch_item_id=1,
        memory_usage=128 * 1024 * 1024,  # 128MB
        cpu_usage=45.0
    )
    
    # Test audit trail
    audit_trail = logger.get_audit_trail(batch_id)
    if audit_trail:
        print(f"✓ Audit trail has {len(audit_trail.events)} events")
        performance_summary = audit_trail.get_performance_summary()
        print(f"✓ Performance summary: {performance_summary}")
    
    # Test recent logs
    recent_logs = logger.get_recent_logs(count=10)
    print(f"✓ Retrieved {len(recent_logs)} recent logs")
    
    # Test error summary
    error_summary = logger.get_error_summary(hours=1)
    print(f"✓ Error summary: {error_summary['total_errors']} errors")
    
    # Test performance summary
    perf_summary = logger.get_performance_summary(hours=1)
    print(f"✓ Performance summary: {perf_summary['total_operations']} operations")
    
    # Test log entry serialization
    entry_dict = entry.to_dict()
    entry_json = entry.to_json()
    print(f"✓ Log entry serialization: {len(entry_dict)} fields")
    
    print("✓ BatchLogger standalone test passed!")


def test_data_structures():
    """Test data structures and enums."""
    print("\nTesting data structures and enums...")
    
    # Test enums
    from database.batch_models import BatchStatus, BatchItemStatus, BatchPriority
    
    print(f"✓ BatchStatus values: {[status.value for status in BatchStatus]}")
    print(f"✓ BatchItemStatus values: {[status.value for status in BatchItemStatus]}")
    print(f"✓ BatchPriority values: {[priority.value for priority in BatchPriority]}")
    
    # Test EventType
    print(f"✓ EventType values: {[event.value for event in EventType]}")
    
    # Test LogLevel
    print(f"✓ LogLevel values: {[level.value for level in LogLevel]}")
    
    # Test AlertLevel
    print(f"✓ AlertLevel values: {[level.value for level in AlertLevel]}")
    
    print("✓ Data structures test passed!")


def main():
    """Run all standalone tests."""
    print("Starting standalone monitoring and logging tests...")
    print("=" * 60)
    
    try:
        test_data_structures()
        test_batch_monitor_standalone()
        test_batch_logger_standalone()
        
        print("\n" + "=" * 60)
        print("✅ All standalone tests passed successfully!")
        print("\nNote: These tests validate the core functionality of the monitoring")
        print("and logging utilities. Full integration tests require the complete")
        print("application environment with database connections.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()