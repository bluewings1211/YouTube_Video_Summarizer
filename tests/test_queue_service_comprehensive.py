"""
Comprehensive test suite for QueueService functionality.

This test suite provides complete coverage for QueueService including:
- Worker registration and lifecycle management
- Queue item processing and priority handling
- Locking mechanisms and timeout handling
- Queue statistics and health monitoring
- Error handling and recovery
- Concurrency and thread safety
- Performance optimization
- Integration with batch processing
- Edge cases and stress testing
"""

import pytest
import threading
import time
import asyncio
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from sqlalchemy.exc import SQLAlchemyError
from concurrent.futures import ThreadPoolExecutor, as_completed
import uuid
import random

from src.database.models import Base, Video
from src.database.batch_models import (
    Batch, BatchItem, QueueItem, ProcessingSession,
    BatchStatus, BatchItemStatus, BatchPriority
)
from src.services.queue_service import (
    QueueService, QueueServiceError, QueueWorkerStatus, QueueHealthStatus,
    QueueProcessingOptions, WorkerInfo, QueueStatistics
)


class TestQueueServiceComprehensive:
    """Comprehensive test suite for QueueService."""

    @pytest.fixture(scope="function")
    def db_engine(self):
        """Create in-memory database engine."""
        engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
            echo=False
        )
        Base.metadata.create_all(engine)
        return engine

    @pytest.fixture(scope="function")
    def db_session(self, db_engine):
        """Create database session for testing."""
        TestingSessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=db_engine
        )
        session = TestingSessionLocal()
        yield session
        session.close()

    @pytest.fixture(scope="function")
    def queue_options(self):
        """Create queue processing options for testing."""
        return QueueProcessingOptions(
            max_workers=5,
            worker_timeout_minutes=10,
            lock_timeout_minutes=5,
            heartbeat_interval_seconds=30,
            stale_lock_cleanup_interval_minutes=2,
            enable_automatic_cleanup=False  # Disable for controlled testing
        )

    @pytest.fixture(scope="function")
    def queue_service(self, db_session, queue_options):
        """Create QueueService instance."""
        service = QueueService(db_session, queue_options)
        yield service
        service.shutdown()

    @pytest.fixture
    def sample_batch(self, db_session):
        """Create sample batch for testing."""
        batch = Batch(
            batch_id="test_batch_001",
            name="Test Batch",
            description="Test batch for queue service",
            status=BatchStatus.PROCESSING,
            priority=BatchPriority.NORMAL,
            total_items=10
        )
        db_session.add(batch)
        db_session.commit()
        return batch

    @pytest.fixture
    def sample_batch_items(self, db_session, sample_batch):
        """Create sample batch items."""
        items = []
        priorities = [BatchPriority.HIGH, BatchPriority.NORMAL, BatchPriority.LOW]
        
        for i in range(10):
            item = BatchItem(
                batch_id=sample_batch.id,
                url=f"https://youtube.com/watch?v=test{i:03d}",
                status=BatchItemStatus.QUEUED,
                priority=priorities[i % 3],
                processing_order=i,
                max_retries=3
            )
            items.append(item)
        
        db_session.add_all(items)
        db_session.commit()
        return items

    @pytest.fixture
    def sample_queue_items(self, db_session, sample_batch_items):
        """Create sample queue items."""
        items = []
        for batch_item in sample_batch_items:
            item = QueueItem(
                batch_item_id=batch_item.id,
                queue_name='test_queue',
                priority=batch_item.priority,
                max_retries=3
            )
            items.append(item)
        
        db_session.add_all(items)
        db_session.commit()
        return items

    @pytest.fixture
    def large_queue_items(self, db_session):
        """Create large number of queue items for performance testing."""
        # Create batch
        batch = Batch(
            batch_id="large_test_batch",
            name="Large Test Batch",
            description="Large batch for performance testing",
            status=BatchStatus.PROCESSING,
            priority=BatchPriority.NORMAL,
            total_items=500
        )
        db_session.add(batch)
        db_session.commit()
        
        # Create batch items
        batch_items = []
        for i in range(500):
            item = BatchItem(
                batch_id=batch.id,
                url=f"https://youtube.com/watch?v=perf{i:04d}",
                status=BatchItemStatus.QUEUED,
                priority=BatchPriority.NORMAL,
                processing_order=i
            )
            batch_items.append(item)
        
        db_session.add_all(batch_items)
        db_session.commit()
        
        # Create queue items
        queue_items = []
        for batch_item in batch_items:
            item = QueueItem(
                batch_item_id=batch_item.id,
                queue_name='performance_queue',
                priority=batch_item.priority
            )
            queue_items.append(item)
        
        db_session.add_all(queue_items)
        db_session.commit()
        return queue_items

    # Worker Management Tests

    def test_worker_registration_comprehensive(self, queue_service):
        """Test comprehensive worker registration scenarios."""
        # Test basic worker registration
        worker_info = queue_service.register_worker("test_queue")
        assert worker_info.queue_name == "test_queue"
        assert worker_info.status == QueueWorkerStatus.IDLE
        assert worker_info.worker_id.startswith("worker_test_queue_")
        assert worker_info.registered_at is not None
        assert worker_info.last_heartbeat is not None
        
        # Test custom worker ID
        custom_worker = queue_service.register_worker("test_queue", "custom_worker_001")
        assert custom_worker.worker_id == "custom_worker_001"
        assert custom_worker.queue_name == "test_queue"
        
        # Test duplicate worker registration
        duplicate_worker = queue_service.register_worker("test_queue", "custom_worker_001")
        assert duplicate_worker.worker_id == "custom_worker_001"  # Should return existing worker
        
        # Test worker registration with different queues
        worker_queue2 = queue_service.register_worker("different_queue", "worker_queue2")
        assert worker_queue2.queue_name == "different_queue"
        
        # Verify worker registry
        assert len(queue_service._worker_registry) == 3
        assert "custom_worker_001" in queue_service._worker_registry
        assert "worker_queue2" in queue_service._worker_registry

    def test_worker_lifecycle_management(self, queue_service):
        """Test complete worker lifecycle management."""
        # Register worker
        worker_info = queue_service.register_worker("lifecycle_queue", "lifecycle_worker")
        
        # Test worker status updates
        result = queue_service.update_worker_heartbeat("lifecycle_worker", QueueWorkerStatus.PROCESSING)
        assert result is True
        
        worker = queue_service._worker_registry["lifecycle_worker"]
        assert worker.status == QueueWorkerStatus.PROCESSING
        
        # Test worker statistics
        stats = queue_service.get_worker_statistics("lifecycle_worker")
        assert stats["worker_id"] == "lifecycle_worker"
        assert stats["queue_name"] == "lifecycle_queue"
        assert stats["status"] == QueueWorkerStatus.PROCESSING.value
        assert stats["processed_items"] == 0
        assert stats["failed_items"] == 0
        
        # Test worker unregistration
        result = queue_service.unregister_worker("lifecycle_worker")
        assert result is True
        assert "lifecycle_worker" not in queue_service._worker_registry
        
        # Test unregistering non-existent worker
        result = queue_service.unregister_worker("non_existent")
        assert result is False

    def test_worker_heartbeat_mechanisms(self, queue_service):
        """Test worker heartbeat mechanisms."""
        # Register worker
        worker_info = queue_service.register_worker("heartbeat_queue", "heartbeat_worker")
        original_heartbeat = worker_info.last_heartbeat
        
        # Small delay to ensure timestamp difference
        time.sleep(0.1)
        
        # Update heartbeat
        result = queue_service.update_worker_heartbeat("heartbeat_worker", QueueWorkerStatus.PROCESSING)
        assert result is True
        
        # Verify heartbeat was updated
        updated_worker = queue_service._worker_registry["heartbeat_worker"]
        assert updated_worker.last_heartbeat > original_heartbeat
        assert updated_worker.status == QueueWorkerStatus.PROCESSING
        
        # Test multiple heartbeat updates
        for i in range(5):
            time.sleep(0.05)
            result = queue_service.update_worker_heartbeat("heartbeat_worker", QueueWorkerStatus.PROCESSING)
            assert result is True
            
            current_worker = queue_service._worker_registry["heartbeat_worker"]
            assert current_worker.last_heartbeat > original_heartbeat
        
        # Test heartbeat for non-existent worker
        result = queue_service.update_worker_heartbeat("non_existent")
        assert result is False

    def test_worker_status_transitions(self, queue_service):
        """Test worker status transitions."""
        worker_info = queue_service.register_worker("status_queue", "status_worker")
        
        # Test all status transitions
        statuses = [
            QueueWorkerStatus.IDLE,
            QueueWorkerStatus.PROCESSING,
            QueueWorkerStatus.PAUSED,
            QueueWorkerStatus.STOPPED,
            QueueWorkerStatus.ERROR
        ]
        
        for status in statuses:
            result = queue_service.update_worker_heartbeat("status_worker", status)
            assert result is True
            
            worker = queue_service._worker_registry["status_worker"]
            assert worker.status == status
            
            # Verify status in statistics
            stats = queue_service.get_worker_statistics("status_worker")
            assert stats["status"] == status.value

    def test_worker_error_handling(self, queue_service):
        """Test worker error handling."""
        worker_info = queue_service.register_worker("error_queue", "error_worker")
        
        # Test error status with message
        queue_service.update_worker_heartbeat("error_worker", QueueWorkerStatus.ERROR)
        
        # Simulate error increment
        worker = queue_service._worker_registry["error_worker"]
        worker.error_count += 1
        worker.last_error = "Test error message"
        
        # Verify error tracking
        stats = queue_service.get_worker_statistics("error_worker")
        assert stats["error_count"] == 1
        assert stats["last_error"] == "Test error message"
        
        # Test recovery
        queue_service.update_worker_heartbeat("error_worker", QueueWorkerStatus.IDLE)
        assert queue_service._worker_registry["error_worker"].status == QueueWorkerStatus.IDLE

    # Queue Processing Tests

    def test_queue_item_processing_comprehensive(self, queue_service, sample_queue_items):
        """Test comprehensive queue item processing."""
        # Register worker
        worker_info = queue_service.register_worker("test_queue", "comprehensive_worker")
        
        # Test priority-based processing
        queue_items = []
        for i in range(5):
            item = queue_service.get_next_queue_item("test_queue", "comprehensive_worker")
            if item:
                queue_items.append(item)
        
        # Verify items are retrieved in priority order
        assert len(queue_items) == 5
        high_priority_items = [item for item in queue_items if item.priority == BatchPriority.HIGH]
        normal_priority_items = [item for item in queue_items if item.priority == BatchPriority.NORMAL]
        low_priority_items = [item for item in queue_items if item.priority == BatchPriority.LOW]
        
        # Should get high priority items first
        assert len(high_priority_items) > 0
        
        # Process items
        for i, item in enumerate(queue_items):
            # Verify item properties
            assert item.locked_by == "comprehensive_worker"
            assert item.locked_at is not None
            assert item.lock_expires_at is not None
            assert item.lock_expires_at > datetime.utcnow()
            assert item.batch_item.status == BatchItemStatus.PROCESSING
            
            # Complete item
            result = queue_service.complete_queue_item(
                item.id,
                "comprehensive_worker",
                BatchItemStatus.COMPLETED,
                result_data={"test": f"result_{i}"}
            )
            assert result is True
            
            # Verify completion
            queue_service.session.refresh(item.batch_item)
            assert item.batch_item.status == BatchItemStatus.COMPLETED
            assert item.batch_item.result_data == {"test": f"result_{i}"}
            assert item.batch_item.completed_at is not None

    def test_queue_item_locking_mechanisms(self, queue_service, sample_queue_items):
        """Test queue item locking mechanisms."""
        # Register multiple workers
        worker1 = queue_service.register_worker("test_queue", "worker1")
        worker2 = queue_service.register_worker("test_queue", "worker2")
        
        # Worker1 gets first item
        item1 = queue_service.get_next_queue_item("test_queue", "worker1")
        assert item1 is not None
        assert item1.locked_by == "worker1"
        
        # Worker2 gets different item
        item2 = queue_service.get_next_queue_item("test_queue", "worker2")
        assert item2 is not None
        assert item2.locked_by == "worker2"
        assert item2.id != item1.id
        
        # Worker1 cannot get item2
        with pytest.raises(QueueServiceError):
            queue_service.release_queue_item(item2.id, "worker1")
        
        # Worker2 cannot complete item1
        with pytest.raises(QueueServiceError):
            queue_service.complete_queue_item(item1.id, "worker2", BatchItemStatus.COMPLETED)
        
        # Test lock expiration
        item1.lock_expires_at = datetime.utcnow() - timedelta(minutes=1)
        queue_service.session.commit()
        
        # Run cleanup
        cleaned_count = queue_service._cleanup_stale_locks()
        assert cleaned_count >= 1
        
        # Item should be available again
        queue_service.session.refresh(item1)
        assert item1.locked_by is None

    def test_queue_item_release_mechanisms(self, queue_service, sample_queue_items):
        """Test queue item release mechanisms."""
        worker_info = queue_service.register_worker("test_queue", "release_worker")
        
        # Get queue item
        item = queue_service.get_next_queue_item("test_queue", "release_worker")
        assert item is not None
        assert item.locked_by == "release_worker"
        
        # Release item
        result = queue_service.release_queue_item(item.id, "release_worker")
        assert result is True
        
        # Verify release
        queue_service.session.refresh(item)
        assert item.locked_by is None
        assert item.locked_at is None
        assert item.lock_expires_at is None
        
        # Verify batch item status
        queue_service.session.refresh(item.batch_item)
        assert item.batch_item.status == BatchItemStatus.QUEUED
        
        # Item should be available again
        reacquired_item = queue_service.get_next_queue_item("test_queue", "release_worker")
        assert reacquired_item is not None
        assert reacquired_item.id == item.id

    def test_queue_item_retry_mechanisms(self, queue_service, sample_queue_items):
        """Test queue item retry mechanisms."""
        worker_info = queue_service.register_worker("test_queue", "retry_worker")
        
        # Get and fail item
        item = queue_service.get_next_queue_item("test_queue", "retry_worker")
        assert item is not None
        
        # Mark as failed
        result = queue_service.complete_queue_item(
            item.id,
            "retry_worker",
            BatchItemStatus.FAILED,
            error_message="Test failure"
        )
        assert result is True
        
        # Verify failure
        queue_service.session.refresh(item.batch_item)
        assert item.batch_item.status == BatchItemStatus.FAILED
        assert item.batch_item.error_info == "Test failure"
        
        # Retry item
        result = queue_service.retry_queue_item(item.id, delay_minutes=0)
        assert result is True
        
        # Verify retry
        queue_service.session.refresh(item)
        assert item.retry_count == 1
        assert item.error_info is None
        assert item.scheduled_at is not None
        
        # Verify batch item status
        queue_service.session.refresh(item.batch_item)
        assert item.batch_item.status == BatchItemStatus.QUEUED
        
        # Test multiple retries
        for retry_count in range(2, 4):
            # Get and fail item again
            retry_item = queue_service.get_next_queue_item("test_queue", "retry_worker")
            if retry_item and retry_item.id == item.id:
                queue_service.complete_queue_item(
                    retry_item.id,
                    "retry_worker",
                    BatchItemStatus.FAILED,
                    error_message=f"Retry {retry_count} failed"
                )
                
                if retry_count < 3:
                    result = queue_service.retry_queue_item(retry_item.id, delay_minutes=0)
                    assert result is True
                    
                    queue_service.session.refresh(retry_item)
                    assert retry_item.retry_count == retry_count

    def test_queue_priority_filtering(self, queue_service, sample_queue_items):
        """Test queue priority filtering."""
        worker_info = queue_service.register_worker("test_queue", "priority_worker")
        
        # Get high priority items only
        high_priority_items = []
        for i in range(5):
            item = queue_service.get_next_queue_item(
                "test_queue", 
                "priority_worker",
                priority_filter=[BatchPriority.HIGH]
            )
            if item:
                high_priority_items.append(item)
                queue_service.complete_queue_item(
                    item.id,
                    "priority_worker",
                    BatchItemStatus.COMPLETED
                )
        
        # All retrieved items should be high priority
        for item in high_priority_items:
            assert item.priority == BatchPriority.HIGH
        
        # Try to get another high priority item (should be None)
        no_item = queue_service.get_next_queue_item(
            "test_queue",
            "priority_worker",
            priority_filter=[BatchPriority.HIGH]
        )
        assert no_item is None
        
        # Get normal priority items
        normal_item = queue_service.get_next_queue_item(
            "test_queue",
            "priority_worker",
            priority_filter=[BatchPriority.NORMAL]
        )
        assert normal_item is not None
        assert normal_item.priority == BatchPriority.NORMAL

    # Queue Statistics and Monitoring Tests

    def test_queue_statistics_comprehensive(self, queue_service, sample_queue_items):
        """Test comprehensive queue statistics."""
        # Register workers
        worker1 = queue_service.register_worker("test_queue", "stats_worker1")
        worker2 = queue_service.register_worker("test_queue", "stats_worker2")
        
        # Initial statistics
        stats = queue_service.get_queue_statistics("test_queue")
        assert isinstance(stats, QueueStatistics)
        assert stats.queue_name == "test_queue"
        assert stats.total_items == 10
        assert stats.pending_items == 10
        assert stats.processing_items == 0
        assert stats.completed_items == 0
        assert stats.failed_items == 0
        assert stats.active_workers == 2
        assert stats.health_status == QueueHealthStatus.HEALTHY
        
        # Process some items
        processed_items = []
        for i in range(5):
            item = queue_service.get_next_queue_item("test_queue", f"stats_worker{(i % 2) + 1}")
            if item:
                processed_items.append(item)
        
        # Statistics after processing starts
        stats = queue_service.get_queue_statistics("test_queue")
        assert stats.processing_items == 5
        assert stats.pending_items == 5
        
        # Complete some items
        for i, item in enumerate(processed_items[:3]):
            queue_service.complete_queue_item(
                item.id,
                item.locked_by,
                BatchItemStatus.COMPLETED
            )
        
        # Fail some items
        for i, item in enumerate(processed_items[3:]):
            queue_service.complete_queue_item(
                item.id,
                item.locked_by,
                BatchItemStatus.FAILED,
                error_message="Test failure"
            )
        
        # Final statistics
        stats = queue_service.get_queue_statistics("test_queue")
        assert stats.completed_items == 3
        assert stats.failed_items == 2
        assert stats.processing_items == 0
        assert stats.pending_items == 5

    def test_queue_health_monitoring(self, queue_service, sample_queue_items):
        """Test queue health monitoring."""
        # Test healthy status
        stats = queue_service.get_queue_statistics("test_queue")
        assert stats.health_status == QueueHealthStatus.HEALTHY
        
        # Test offline status (no workers)
        for worker_id in list(queue_service._worker_registry.keys()):
            queue_service.unregister_worker(worker_id)
        
        stats = queue_service.get_queue_statistics("test_queue")
        assert stats.health_status == QueueHealthStatus.OFFLINE
        
        # Register worker and create high pending load
        worker_info = queue_service.register_worker("test_queue", "health_worker")
        
        # Create warning condition (high pending ratio)
        # Process just 1 item to create imbalance
        item = queue_service.get_next_queue_item("test_queue", "health_worker")
        if item:
            queue_service.complete_queue_item(
                item.id,
                "health_worker",
                BatchItemStatus.COMPLETED
            )
        
        # The queue should still be healthy with good worker ratio
        stats = queue_service.get_queue_statistics("test_queue")
        assert stats.health_status in [QueueHealthStatus.HEALTHY, QueueHealthStatus.WARNING]

    def test_worker_statistics_comprehensive(self, queue_service, sample_queue_items):
        """Test comprehensive worker statistics."""
        # Register multiple workers
        workers = []
        for i in range(3):
            worker = queue_service.register_worker("test_queue", f"stats_worker_{i}")
            workers.append(worker)
        
        # Get all worker statistics
        all_stats = queue_service.get_worker_statistics()
        assert all_stats["total_workers"] == 3
        assert all_stats["active_workers"] == 3
        assert len(all_stats["workers"]) == 3
        
        # Process items with different workers
        for i, worker in enumerate(workers):
            for j in range(i + 1):  # Different number of items per worker
                item = queue_service.get_next_queue_item("test_queue", worker.worker_id)
                if item:
                    status = BatchItemStatus.COMPLETED if j % 2 == 0 else BatchItemStatus.FAILED
                    queue_service.complete_queue_item(
                        item.id,
                        worker.worker_id,
                        status,
                        error_message="Test error" if status == BatchItemStatus.FAILED else None
                    )
                    
                    # Update worker stats
                    if status == BatchItemStatus.COMPLETED:
                        worker.processed_items += 1
                    else:
                        worker.failed_items += 1
        
        # Get individual worker statistics
        for i, worker in enumerate(workers):
            stats = queue_service.get_worker_statistics(worker.worker_id)
            assert stats["worker_id"] == worker.worker_id
            assert stats["queue_name"] == "test_queue"
            assert stats["processed_items"] == worker.processed_items
            assert stats["failed_items"] == worker.failed_items

    # Queue Control Tests

    def test_queue_pause_resume_mechanisms(self, queue_service, sample_queue_items):
        """Test queue pause and resume mechanisms."""
        # Register workers
        workers = []
        for i in range(3):
            worker = queue_service.register_worker("test_queue", f"pause_worker_{i}")
            workers.append(worker)
        
        # Verify workers are active
        for worker in workers:
            assert queue_service._worker_registry[worker.worker_id].status == QueueWorkerStatus.IDLE
        
        # Pause queue
        result = queue_service.pause_queue("test_queue")
        assert result is True
        
        # Verify all workers are paused
        for worker in workers:
            assert queue_service._worker_registry[worker.worker_id].status == QueueWorkerStatus.PAUSED
        
        # Try to get queue item while paused (should still work but workers are paused)
        item = queue_service.get_next_queue_item("test_queue", "pause_worker_0")
        # The item retrieval might still work, but workers are marked as paused
        
        # Resume queue
        result = queue_service.resume_queue("test_queue")
        assert result is True
        
        # Verify all workers are resumed
        for worker in workers:
            assert queue_service._worker_registry[worker.worker_id].status == QueueWorkerStatus.IDLE
        
        # Test pausing non-existent queue
        result = queue_service.pause_queue("non_existent_queue")
        assert result is False
        
        # Test resuming non-existent queue
        result = queue_service.resume_queue("non_existent_queue")
        assert result is False

    def test_queue_shutdown_mechanisms(self, queue_service, sample_queue_items):
        """Test queue shutdown mechanisms."""
        # Register workers
        workers = []
        for i in range(3):
            worker = queue_service.register_worker("test_queue", f"shutdown_worker_{i}")
            workers.append(worker)
        
        # Verify workers are registered
        assert len(queue_service._worker_registry) == 3
        
        # Shutdown queue service
        queue_service.shutdown()
        
        # Verify workers are unregistered
        assert len(queue_service._worker_registry) == 0
        
        # Verify cleanup thread is stopped
        assert not queue_service._cleanup_running

    # Performance and Concurrency Tests

    def test_concurrent_queue_operations(self, queue_service, large_queue_items):
        """Test concurrent queue operations."""
        # Register multiple workers
        workers = []
        for i in range(10):
            worker = queue_service.register_worker("performance_queue", f"concurrent_worker_{i}")
            workers.append(worker)
        
        # Track processed items
        processed_items = []
        processed_lock = threading.Lock()
        
        def worker_function(worker_id):
            local_processed = 0
            for _ in range(10):  # Each worker processes up to 10 items
                item = queue_service.get_next_queue_item("performance_queue", worker_id)
                if item:
                    # Simulate processing time
                    time.sleep(0.001)
                    
                    # Complete item
                    queue_service.complete_queue_item(
                        item.id,
                        worker_id,
                        BatchItemStatus.COMPLETED
                    )
                    
                    with processed_lock:
                        processed_items.append(item.id)
                    
                    local_processed += 1
                else:
                    break
            return local_processed
        
        # Start concurrent processing
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(worker_function, worker.worker_id)
                for worker in workers
            ]
            
            results = [future.result() for future in futures]
        
        processing_time = time.time() - start_time
        
        # Verify results
        total_processed = sum(results)
        assert total_processed > 0
        assert total_processed <= 100  # Each worker processes max 10 items
        assert len(set(processed_items)) == len(processed_items)  # No duplicates
        
        # Performance should be reasonable
        assert processing_time < 30.0  # Should complete within 30 seconds
        
        # Verify statistics
        stats = queue_service.get_queue_statistics("performance_queue")
        assert stats.completed_items == total_processed

    def test_queue_performance_large_scale(self, queue_service, large_queue_items):
        """Test queue performance with large scale operations."""
        # Register workers
        workers = []
        for i in range(5):
            worker = queue_service.register_worker("performance_queue", f"perf_worker_{i}")
            workers.append(worker)
        
        # Measure queue statistics calculation time
        start_time = time.time()
        stats = queue_service.get_queue_statistics("performance_queue")
        stats_time = time.time() - start_time
        
        assert stats_time < 2.0  # Should calculate stats quickly
        assert stats.total_items == 500
        
        # Measure item retrieval performance
        start_time = time.time()
        items = []
        for i in range(50):  # Get 50 items
            item = queue_service.get_next_queue_item("performance_queue", f"perf_worker_{i % 5}")
            if item:
                items.append(item)
        
        retrieval_time = time.time() - start_time
        assert retrieval_time < 5.0  # Should retrieve items quickly
        assert len(items) == 50

    def test_queue_memory_usage_optimization(self, queue_service, large_queue_items):
        """Test queue memory usage optimization."""
        import psutil
        import os
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Register workers and process items
        workers = []
        for i in range(5):
            worker = queue_service.register_worker("performance_queue", f"memory_worker_{i}")
            workers.append(worker)
        
        # Process items in chunks to avoid memory buildup
        processed_count = 0
        for chunk in range(10):  # Process 10 chunks of 10 items each
            chunk_items = []
            for i in range(10):
                item = queue_service.get_next_queue_item("performance_queue", f"memory_worker_{i % 5}")
                if item:
                    chunk_items.append(item)
            
            # Complete chunk items
            for item in chunk_items:
                queue_service.complete_queue_item(
                    item.id,
                    item.locked_by,
                    BatchItemStatus.COMPLETED
                )
                processed_count += 1
        
        # Check memory usage
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 50MB)
        assert memory_increase < 50 * 1024 * 1024
        assert processed_count > 0

    # Integration Tests

    def test_queue_batch_integration(self, queue_service, sample_queue_items):
        """Test queue integration with batch processing."""
        # Register worker
        worker_info = queue_service.register_worker("test_queue", "integration_worker")
        
        # Process all items in queue
        processed_items = []
        while True:
            item = queue_service.get_next_queue_item("test_queue", "integration_worker")
            if not item:
                break
            
            processed_items.append(item)
            
            # Complete item
            queue_service.complete_queue_item(
                item.id,
                "integration_worker",
                BatchItemStatus.COMPLETED,
                result_data={"processed": True}
            )
        
        # Verify all items were processed
        assert len(processed_items) == 10
        
        # Verify batch completion
        stats = queue_service.get_queue_statistics("test_queue")
        assert stats.completed_items == 10
        assert stats.pending_items == 0
        assert stats.processing_items == 0
        
        # Verify batch status is updated
        batch_item = processed_items[0].batch_item
        queue_service.session.refresh(batch_item)
        assert batch_item.batch.status == BatchStatus.COMPLETED

    def test_queue_resilience_with_failures(self, queue_service, sample_queue_items):
        """Test queue resilience with failures."""
        worker_info = queue_service.register_worker("test_queue", "resilience_worker")
        
        # Process items with mixed success/failure
        processed_count = 0
        for i in range(10):
            item = queue_service.get_next_queue_item("test_queue", "resilience_worker")
            if item:
                # Alternate between success and failure
                if i % 2 == 0:
                    queue_service.complete_queue_item(
                        item.id,
                        "resilience_worker",
                        BatchItemStatus.COMPLETED
                    )
                else:
                    queue_service.complete_queue_item(
                        item.id,
                        "resilience_worker",
                        BatchItemStatus.FAILED,
                        error_message="Simulated failure"
                    )
                
                processed_count += 1
        
        # Verify queue statistics
        stats = queue_service.get_queue_statistics("test_queue")
        assert stats.completed_items == 5  # Half succeeded
        assert stats.failed_items == 5    # Half failed
        assert stats.health_status in [QueueHealthStatus.HEALTHY, QueueHealthStatus.WARNING]

    # Error Handling and Edge Cases

    def test_queue_error_handling_comprehensive(self, queue_service):
        """Test comprehensive error handling."""
        # Test invalid worker operations
        with pytest.raises(QueueServiceError):
            queue_service.get_next_queue_item("non_existent_queue", "invalid_worker")
        
        # Test completing non-existent item
        with pytest.raises(QueueServiceError):
            queue_service.complete_queue_item(
                99999, "worker1", BatchItemStatus.COMPLETED
            )
        
        # Test releasing item with wrong worker
        # First create a valid scenario
        worker1 = queue_service.register_worker("test_queue", "worker1")
        worker2 = queue_service.register_worker("test_queue", "worker2")
        
        # Create a test item
        batch = Batch(
            batch_id="error_test",
            status=BatchStatus.PROCESSING,
            total_items=1
        )
        queue_service.session.add(batch)
        queue_service.session.commit()
        
        batch_item = BatchItem(
            batch_id=batch.id,
            url="https://youtube.com/watch?v=error_test",
            status=BatchItemStatus.PROCESSING
        )
        queue_service.session.add(batch_item)
        queue_service.session.commit()
        
        queue_item = QueueItem(
            batch_item_id=batch_item.id,
            queue_name='test_queue',
            locked_by="worker1",
            locked_at=datetime.utcnow()
        )
        queue_service.session.add(queue_item)
        queue_service.session.commit()
        
        # Try to release with wrong worker
        with pytest.raises(QueueServiceError):
            queue_service.release_queue_item(queue_item.id, "worker2")

    def test_queue_edge_cases(self, queue_service):
        """Test queue edge cases."""
        # Test empty queue
        worker_info = queue_service.register_worker("empty_queue", "empty_worker")
        
        item = queue_service.get_next_queue_item("empty_queue", "empty_worker")
        assert item is None
        
        # Test queue with no workers
        stats = queue_service.get_queue_statistics("empty_queue")
        assert stats.total_items == 0
        assert stats.active_workers == 1  # Worker was registered
        
        # Test unregistering all workers
        queue_service.unregister_worker("empty_worker")
        stats = queue_service.get_queue_statistics("empty_queue")
        assert stats.active_workers == 0
        assert stats.health_status == QueueHealthStatus.OFFLINE

    def test_queue_cleanup_operations(self, queue_service, sample_queue_items):
        """Test queue cleanup operations."""
        # Register worker and get items
        worker_info = queue_service.register_worker("test_queue", "cleanup_worker")
        
        # Get items and make them stale
        stale_items = []
        for i in range(3):
            item = queue_service.get_next_queue_item("test_queue", "cleanup_worker")
            if item:
                # Make lock stale
                item.lock_expires_at = datetime.utcnow() - timedelta(minutes=10)
                stale_items.append(item)
        
        queue_service.session.commit()
        
        # Run cleanup
        cleaned_count = queue_service._cleanup_stale_locks()
        assert cleaned_count == 3
        
        # Verify items are unlocked
        for item in stale_items:
            queue_service.session.refresh(item)
            assert item.locked_by is None
            assert item.locked_at is None
            assert item.lock_expires_at is None

    def test_queue_stress_testing(self, queue_service):
        """Test queue under stress conditions."""
        # Create many workers
        workers = []
        for i in range(20):
            worker = queue_service.register_worker("stress_queue", f"stress_worker_{i}")
            workers.append(worker)
        
        # Create many items
        batch = Batch(
            batch_id="stress_test",
            status=BatchStatus.PROCESSING,
            total_items=100
        )
        queue_service.session.add(batch)
        queue_service.session.commit()
        
        batch_items = []
        for i in range(100):
            item = BatchItem(
                batch_id=batch.id,
                url=f"https://youtube.com/watch?v=stress{i:03d}",
                status=BatchItemStatus.QUEUED
            )
            batch_items.append(item)
        
        queue_service.session.add_all(batch_items)
        queue_service.session.commit()
        
        queue_items = []
        for batch_item in batch_items:
            item = QueueItem(
                batch_item_id=batch_item.id,
                queue_name='stress_queue',
                priority=random.choice(list(BatchPriority))
            )
            queue_items.append(item)
        
        queue_service.session.add_all(queue_items)
        queue_service.session.commit()
        
        # Stress test with rapid operations
        def stress_worker(worker_id):
            items_processed = 0
            for _ in range(10):
                item = queue_service.get_next_queue_item("stress_queue", worker_id)
                if item:
                    # Rapid completion
                    queue_service.complete_queue_item(
                        item.id,
                        worker_id,
                        BatchItemStatus.COMPLETED
                    )
                    items_processed += 1
                else:
                    break
            return items_processed
        
        # Run stress test
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [
                executor.submit(stress_worker, worker.worker_id)
                for worker in workers
            ]
            
            results = [future.result() for future in futures]
        
        stress_time = time.time() - start_time
        
        # Verify stress test results
        total_processed = sum(results)
        assert total_processed > 0
        assert total_processed <= 100
        assert stress_time < 60.0  # Should complete within 60 seconds
        
        # Verify final statistics
        stats = queue_service.get_queue_statistics("stress_queue")
        assert stats.completed_items == total_processed
        assert stats.health_status in [QueueHealthStatus.HEALTHY, QueueHealthStatus.WARNING]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])