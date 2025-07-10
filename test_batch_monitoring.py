#!/usr/bin/env python3
"""
Test script for batch monitoring and logging functionality.

This script tests the core monitoring and logging capabilities
for batch processing operations.
"""

import sys
import os
import time
from datetime import datetime
from unittest.mock import Mock, patch

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from utils.batch_monitor import BatchMonitor, initialize_monitoring, get_batch_monitor
from utils.batch_logger import BatchLogger, initialize_logging, get_batch_logger, EventType, LogLevel
from database.batch_models import BatchStatus, BatchItemStatus, BatchPriority


def test_batch_monitoring():
    """Test batch monitoring functionality."""
    print("Testing batch monitoring functionality...")
    
    # Initialize monitoring
    monitor = initialize_monitoring(enabled=True)
    
    # Test batch monitoring
    batch_id = "test_batch_001"
    total_items = 5
    
    # Start batch monitoring
    metrics = monitor.start_batch_monitoring(batch_id, total_items)
    print(f"✓ Started monitoring batch {batch_id} with {total_items} items")
    
    # Simulate item processing
    for i in range(total_items):
        item_id = i + 1
        url = f"https://youtube.com/watch?v=test{i}"
        
        # Start item monitoring
        item_metrics = monitor.start_item_monitoring(item_id, batch_id, url)
        print(f"✓ Started monitoring item {item_id}: {url}")
        
        # Simulate processing time
        time.sleep(0.1)
        
        # Finish item monitoring
        status = BatchItemStatus.COMPLETED if i < 4 else BatchItemStatus.FAILED
        monitor.finish_item_monitoring(item_id, status)
        print(f"✓ Finished monitoring item {item_id} with status {status.value}")
        
        # Record error for failed item
        if status == BatchItemStatus.FAILED:
            monitor.record_error(batch_id, item_id, "Test error message")
    
    # Update worker count
    monitor.update_worker_count(batch_id, 2, 3)
    print("✓ Updated worker count metrics")
    
    # Update queue metrics
    monitor.update_queue_metrics(batch_id, 0, 5.0)
    print("✓ Updated queue metrics")
    
    # Get batch metrics
    batch_metrics = monitor.get_batch_metrics(batch_id)
    print(f"✓ Retrieved batch metrics: {batch_metrics['success_rate']:.1f}% success rate")
    
    # Get system metrics
    system_metrics = monitor.get_system_metrics()
    print(f"✓ Retrieved system metrics: {system_metrics['active_batches']} active batches")
    
    # Finish batch monitoring
    summary = monitor.finish_batch_monitoring(batch_id)
    print(f"✓ Finished batch monitoring: {summary}")
    
    # Get alerts
    alerts = monitor.get_alerts()
    print(f"✓ Retrieved {len(alerts)} alerts")
    
    print("✓ Batch monitoring tests passed!")


def test_batch_logging():
    """Test batch logging functionality."""
    print("\nTesting batch logging functionality...")
    
    # Initialize logging
    logger = initialize_logging(log_level=LogLevel.INFO)
    
    # Test batch logging
    batch_id = "test_batch_002"
    
    # Log batch creation
    logger.log_batch_event(
        batch_id=batch_id,
        event_type=EventType.BATCH_CREATED,
        message="Created test batch",
        context={"total_items": 3, "priority": "normal"}
    )
    print("✓ Logged batch creation event")
    
    # Log batch started
    logger.log_batch_event(
        batch_id=batch_id,
        event_type=EventType.BATCH_STARTED,
        message="Started batch processing",
        context={"worker_count": 2}
    )
    print("✓ Logged batch started event")
    
    # Log item events
    for i in range(3):
        item_id = i + 1
        worker_id = f"worker_{i % 2 + 1}"
        
        # Log item started
        logger.log_item_event(
            batch_id=batch_id,
            batch_item_id=item_id,
            event_type=EventType.ITEM_STARTED,
            message=f"Started processing item {item_id}",
            worker_id=worker_id,
            context={"url": f"https://youtube.com/watch?v=test{i}"}
        )
        
        # Log item completed
        status = EventType.ITEM_COMPLETED if i < 2 else EventType.ITEM_FAILED
        logger.log_item_event(
            batch_id=batch_id,
            batch_item_id=item_id,
            event_type=status,
            message=f"Completed processing item {item_id}",
            worker_id=worker_id,
            duration=1.5 + i * 0.5
        )
        
        print(f"✓ Logged item {item_id} events")
    
    # Log worker events
    worker_id = "worker_1"
    logger.log_worker_event(
        worker_id=worker_id,
        event_type=EventType.WORKER_STARTED,
        message="Worker started",
        context={"queue_name": "video_processing"}
    )
    print("✓ Logged worker event")
    
    # Log error
    try:
        raise ValueError("Test error")
    except Exception as e:
        logger.log_error(
            message="Test error occurred",
            error=e,
            batch_id=batch_id,
            batch_item_id=3,
            worker_id=worker_id,
            context={"operation": "video_processing"}
        )
    print("✓ Logged error event")
    
    # Log performance
    logger.log_performance(
        operation="video_summarization",
        duration=45.2,
        batch_id=batch_id,
        batch_item_id=1,
        worker_id=worker_id,
        memory_usage=512 * 1024 * 1024,  # 512MB
        cpu_usage=75.5
    )
    print("✓ Logged performance metrics")
    
    # Get audit trail
    audit_trail = logger.get_audit_trail(batch_id)
    if audit_trail:
        print(f"✓ Retrieved audit trail with {len(audit_trail.events)} events")
        performance_summary = audit_trail.get_performance_summary()
        print(f"✓ Performance summary: {performance_summary}")
    
    # Get recent logs
    recent_logs = logger.get_recent_logs(count=10, batch_id=batch_id)
    print(f"✓ Retrieved {len(recent_logs)} recent logs")
    
    # Get error summary
    error_summary = logger.get_error_summary(hours=1)
    print(f"✓ Error summary: {error_summary['total_errors']} errors")
    
    # Get performance summary
    perf_summary = logger.get_performance_summary(hours=1)
    print(f"✓ Performance summary: {perf_summary['total_operations']} operations")
    
    print("✓ Batch logging tests passed!")


def test_integration():
    """Test integration between monitoring and logging."""
    print("\nTesting monitoring and logging integration...")
    
    # Get global instances
    monitor = get_batch_monitor()
    logger = get_batch_logger()
    
    batch_id = "integration_test_batch"
    
    # Start monitoring
    monitor.start_batch_monitoring(batch_id, 2)
    
    # Log batch events
    logger.log_batch_event(
        batch_id=batch_id,
        event_type=EventType.BATCH_CREATED,
        message="Integration test batch created"
    )
    
    # Process items
    for i in range(2):
        item_id = i + 1
        url = f"https://youtube.com/watch?v=integration_test_{i}"
        
        # Start monitoring and logging
        monitor.start_item_monitoring(item_id, batch_id, url)
        logger.log_item_event(
            batch_id=batch_id,
            batch_item_id=item_id,
            event_type=EventType.ITEM_STARTED,
            message=f"Started integration test item {item_id}",
            context={"url": url}
        )
        
        # Simulate processing
        time.sleep(0.1)
        
        # Complete monitoring and logging
        status = BatchItemStatus.COMPLETED
        monitor.finish_item_monitoring(item_id, status)
        logger.log_item_event(
            batch_id=batch_id,
            batch_item_id=item_id,
            event_type=EventType.ITEM_COMPLETED,
            message=f"Completed integration test item {item_id}",
            duration=0.1
        )
        
        print(f"✓ Processed integration test item {item_id}")
    
    # Finish monitoring
    summary = monitor.finish_batch_monitoring(batch_id)
    logger.log_batch_event(
        batch_id=batch_id,
        event_type=EventType.BATCH_COMPLETED,
        message="Integration test batch completed",
        context=summary
    )
    
    print("✓ Integration test completed successfully!")


def main():
    """Run all tests."""
    print("Starting batch monitoring and logging tests...")
    print("=" * 60)
    
    try:
        test_batch_monitoring()
        test_batch_logging()
        test_integration()
        
        print("\n" + "=" * 60)
        print("✅ All tests passed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()