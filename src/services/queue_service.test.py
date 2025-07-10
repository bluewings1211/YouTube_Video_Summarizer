"""
Test suite for QueueService.

This test suite covers:
- Queue service initialization and configuration
- Worker registration and management
- Queue item processing and locking
- Priority-based queue processing
- Queue monitoring and statistics
- Error handling and edge cases
- Integration with batch processing system
"""

import pytest
import threading
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from ..database.models import Base, Video
from ..database.batch_models import (
    Batch, BatchItem, QueueItem, ProcessingSession,
    BatchStatus, BatchItemStatus, BatchPriority
)
from .queue_service import (
    QueueService, QueueServiceError, QueueWorkerStatus, QueueHealthStatus,
    QueueProcessingOptions, WorkerInfo, QueueStatistics
)


class TestQueueService:
    """Test suite for QueueService class."""
    
    @pytest.fixture(autouse=True)
    def setup_database(self):
        """Set up test database."""
        # Create in-memory SQLite database
        self.engine = create_engine("sqlite:///:memory:", echo=False)
        Base.metadata.create_all(self.engine)
        
        # Create session factory
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
        
        # Create test options
        self.options = QueueProcessingOptions(
            max_workers=3,
            worker_timeout_minutes=5,
            lock_timeout_minutes=2,
            heartbeat_interval_seconds=10,
            stale_lock_cleanup_interval_minutes=1,
            enable_automatic_cleanup=False  # Disable for testing
        )
        
        # Create queue service
        self.queue_service = QueueService(self.session, self.options)
        
        yield
        
        # Cleanup
        self.queue_service.shutdown()
        self.session.close()
    
    @pytest.fixture
    def sample_batch(self):
        """Create a sample batch for testing."""
        batch = Batch(
            batch_id="test_batch_001",
            name="Test Batch",
            description="Test batch for queue service",
            status=BatchStatus.PROCESSING,
            priority=BatchPriority.NORMAL,
            total_items=3
        )
        self.session.add(batch)
        self.session.commit()
        return batch
    
    @pytest.fixture
    def sample_batch_items(self, sample_batch):
        """Create sample batch items for testing."""
        items = []
        for i in range(3):
            item = BatchItem(
                batch_id=sample_batch.id,
                url=f"https://youtube.com/watch?v=test{i}",
                status=BatchItemStatus.QUEUED,
                priority=BatchPriority.NORMAL,
                processing_order=i
            )
            items.append(item)
        
        self.session.add_all(items)
        self.session.commit()
        return items
    
    @pytest.fixture
    def sample_queue_items(self, sample_batch_items):
        """Create sample queue items for testing."""
        items = []
        for batch_item in sample_batch_items:
            item = QueueItem(
                batch_item_id=batch_item.id,
                queue_name='test_queue',
                priority=BatchPriority.NORMAL
            )
            items.append(item)
        
        self.session.add_all(items)
        self.session.commit()
        return items
    
    def test_queue_service_initialization(self):
        """Test queue service initialization."""
        # Test with default options
        service = QueueService(self.session)
        assert service is not None
        assert service.options.max_workers == 5
        assert service.options.worker_timeout_minutes == 30
        
        # Test with custom options
        custom_options = QueueProcessingOptions(max_workers=10)
        service = QueueService(self.session, custom_options)
        assert service.options.max_workers == 10
        
        service.shutdown()
    
    def test_worker_registration(self):
        """Test worker registration and management."""
        # Test worker registration
        worker_info = self.queue_service.register_worker("test_queue")
        assert worker_info.queue_name == "test_queue"
        assert worker_info.status == QueueWorkerStatus.IDLE
        assert worker_info.worker_id.startswith("worker_test_queue_")
        
        # Test custom worker ID
        custom_worker = self.queue_service.register_worker("test_queue", "custom_worker_001")
        assert custom_worker.worker_id == "custom_worker_001"
        
        # Test worker retrieval
        stats = self.queue_service.get_worker_statistics("custom_worker_001")
        assert stats["worker_id"] == "custom_worker_001"
        assert stats["queue_name"] == "test_queue"
        
        # Test worker unregistration
        result = self.queue_service.unregister_worker("custom_worker_001")
        assert result is True
        
        # Test getting non-existent worker
        stats = self.queue_service.get_worker_statistics("non_existent")
        assert stats == {}
    
    def test_worker_heartbeat(self):
        """Test worker heartbeat functionality."""
        # Register worker
        worker_info = self.queue_service.register_worker("test_queue")
        worker_id = worker_info.worker_id
        
        # Test heartbeat update
        original_heartbeat = worker_info.last_heartbeat
        time.sleep(0.1)  # Small delay to ensure timestamp difference
        
        result = self.queue_service.update_worker_heartbeat(worker_id, QueueWorkerStatus.PROCESSING)
        assert result is True
        
        # Verify heartbeat was updated
        updated_worker = self.queue_service._worker_registry[worker_id]
        assert updated_worker.last_heartbeat > original_heartbeat
        assert updated_worker.status == QueueWorkerStatus.PROCESSING
        
        # Test heartbeat for non-existent worker
        result = self.queue_service.update_worker_heartbeat("non_existent")
        assert result is False
    
    def test_queue_item_processing(self, sample_queue_items):
        """Test queue item processing lifecycle."""
        # Register worker
        worker_info = self.queue_service.register_worker("test_queue")
        worker_id = worker_info.worker_id
        
        # Get next queue item
        queue_item = self.queue_service.get_next_queue_item("test_queue", worker_id)
        assert queue_item is not None
        assert queue_item.locked_by == worker_id
        assert queue_item.locked_at is not None
        assert queue_item.lock_expires_at is not None
        
        # Verify batch item status updated
        assert queue_item.batch_item.status == BatchItemStatus.PROCESSING
        
        # Test getting another item (should get different item)
        worker_info2 = self.queue_service.register_worker("test_queue")
        queue_item2 = self.queue_service.get_next_queue_item("test_queue", worker_info2.worker_id)
        assert queue_item2 is not None
        assert queue_item2.id != queue_item.id
        
        # Complete the first item
        result = self.queue_service.complete_queue_item(
            queue_item.id, 
            worker_id, 
            BatchItemStatus.COMPLETED,
            result_data={"test": "data"}
        )
        assert result is True
        
        # Verify item was removed from queue
        self.session.refresh(queue_item.batch_item)
        assert queue_item.batch_item.status == BatchItemStatus.COMPLETED
        assert queue_item.batch_item.result_data == {"test": "data"}
    
    def test_queue_item_release(self, sample_queue_items):
        """Test queue item release functionality."""
        # Register worker and get item
        worker_info = self.queue_service.register_worker("test_queue")
        worker_id = worker_info.worker_id
        
        queue_item = self.queue_service.get_next_queue_item("test_queue", worker_id)
        assert queue_item is not None
        
        # Release the item
        result = self.queue_service.release_queue_item(queue_item.id, worker_id)
        assert result is True
        
        # Verify item was released
        self.session.refresh(queue_item)
        assert queue_item.locked_by is None
        assert queue_item.locked_at is None
        assert queue_item.lock_expires_at is None
        
        # Verify batch item status
        self.session.refresh(queue_item.batch_item)
        assert queue_item.batch_item.status == BatchItemStatus.QUEUED
    
    def test_priority_based_processing(self):
        """Test priority-based queue processing."""
        # Create batch items with different priorities
        batch = Batch(
            batch_id="priority_test_batch",
            name="Priority Test",
            status=BatchStatus.PROCESSING,
            total_items=3
        )
        self.session.add(batch)
        self.session.commit()
        
        # Create items with different priorities
        priorities = [BatchPriority.LOW, BatchPriority.HIGH, BatchPriority.NORMAL]
        queue_items = []
        
        for i, priority in enumerate(priorities):
            batch_item = BatchItem(
                batch_id=batch.id,
                url=f"https://youtube.com/watch?v=priority{i}",
                status=BatchItemStatus.QUEUED,
                priority=priority,
                processing_order=i
            )
            self.session.add(batch_item)
            self.session.commit()
            
            queue_item = QueueItem(
                batch_item_id=batch_item.id,
                queue_name='priority_queue',
                priority=priority
            )
            queue_items.append(queue_item)
        
        self.session.add_all(queue_items)
        self.session.commit()
        
        # Register worker
        worker_info = self.queue_service.register_worker("priority_queue")
        worker_id = worker_info.worker_id
        
        # Get items - should get HIGH priority first
        first_item = self.queue_service.get_next_queue_item("priority_queue", worker_id)
        assert first_item is not None
        assert first_item.priority == BatchPriority.HIGH
        
        # Complete first item and get next
        self.queue_service.complete_queue_item(
            first_item.id, worker_id, BatchItemStatus.COMPLETED
        )
        
        second_item = self.queue_service.get_next_queue_item("priority_queue", worker_id)
        assert second_item is not None
        assert second_item.priority == BatchPriority.NORMAL
    
    def test_queue_item_retry(self, sample_queue_items):
        """Test queue item retry functionality."""
        queue_item = sample_queue_items[0]
        
        # Set item as failed
        queue_item.retry_count = 1
        queue_item.error_info = "Test error"
        self.session.commit()
        
        # Retry the item
        result = self.queue_service.retry_queue_item(queue_item.id, delay_minutes=1)
        assert result is True
        
        # Verify retry was scheduled
        self.session.refresh(queue_item)
        assert queue_item.retry_count == 2
        assert queue_item.error_info is None
        assert queue_item.scheduled_at > datetime.utcnow()
        
        # Verify batch item status
        self.session.refresh(queue_item.batch_item)
        assert queue_item.batch_item.status == BatchItemStatus.QUEUED
    
    def test_queue_statistics(self, sample_queue_items):
        """Test queue statistics functionality."""
        # Get initial statistics
        stats = self.queue_service.get_queue_statistics("test_queue")
        assert isinstance(stats, QueueStatistics)
        assert stats.queue_name == "test_queue"
        assert stats.total_items == 3
        assert stats.pending_items == 3
        assert stats.processing_items == 0
        assert stats.completed_items == 0
        assert stats.failed_items == 0
        
        # Register worker and process an item
        worker_info = self.queue_service.register_worker("test_queue")
        worker_id = worker_info.worker_id
        
        queue_item = self.queue_service.get_next_queue_item("test_queue", worker_id)
        
        # Get updated statistics
        stats = self.queue_service.get_queue_statistics("test_queue")
        assert stats.processing_items == 1
        assert stats.pending_items == 2
        assert stats.active_workers == 1
        
        # Complete the item
        self.queue_service.complete_queue_item(
            queue_item.id, worker_id, BatchItemStatus.COMPLETED
        )
        
        # Get final statistics
        stats = self.queue_service.get_queue_statistics("test_queue")
        assert stats.completed_items == 1
        assert stats.processing_items == 0
    
    def test_worker_statistics(self):
        """Test worker statistics functionality."""
        # Register multiple workers
        worker1 = self.queue_service.register_worker("test_queue", "worker1")
        worker2 = self.queue_service.register_worker("test_queue", "worker2")
        
        # Get all worker statistics
        all_stats = self.queue_service.get_worker_statistics()
        assert all_stats["total_workers"] == 2
        assert all_stats["active_workers"] == 2
        assert len(all_stats["workers"]) == 2
        
        # Get specific worker statistics
        worker1_stats = self.queue_service.get_worker_statistics("worker1")
        assert worker1_stats["worker_id"] == "worker1"
        assert worker1_stats["queue_name"] == "test_queue"
        assert worker1_stats["status"] == QueueWorkerStatus.IDLE.value
        assert worker1_stats["processed_items"] == 0
        assert worker1_stats["failed_items"] == 0
    
    def test_queue_pause_resume(self):
        """Test queue pause and resume functionality."""
        # Register workers
        worker1 = self.queue_service.register_worker("test_queue", "worker1")
        worker2 = self.queue_service.register_worker("test_queue", "worker2")
        
        # Pause the queue
        result = self.queue_service.pause_queue("test_queue")
        assert result is True
        
        # Verify workers are paused
        assert self.queue_service._worker_registry["worker1"].status == QueueWorkerStatus.PAUSED
        assert self.queue_service._worker_registry["worker2"].status == QueueWorkerStatus.PAUSED
        
        # Resume the queue
        result = self.queue_service.resume_queue("test_queue")
        assert result is True
        
        # Verify workers are resumed
        assert self.queue_service._worker_registry["worker1"].status == QueueWorkerStatus.IDLE
        assert self.queue_service._worker_registry["worker2"].status == QueueWorkerStatus.IDLE
    
    def test_stale_lock_cleanup(self, sample_queue_items):
        """Test stale lock cleanup functionality."""
        # Register worker and get item
        worker_info = self.queue_service.register_worker("test_queue")
        worker_id = worker_info.worker_id
        
        queue_item = self.queue_service.get_next_queue_item("test_queue", worker_id)
        assert queue_item is not None
        
        # Manually expire the lock
        queue_item.lock_expires_at = datetime.utcnow() - timedelta(minutes=1)
        self.session.commit()
        
        # Run cleanup
        cleaned_count = self.queue_service._cleanup_stale_locks()
        assert cleaned_count == 1
        
        # Verify lock was released
        self.session.refresh(queue_item)
        assert queue_item.locked_by is None
        assert queue_item.locked_at is None
        assert queue_item.lock_expires_at is None
    
    def test_queue_health_calculation(self):
        """Test queue health status calculation."""
        # Test healthy status
        health = self.queue_service._calculate_queue_health(
            total_items=10, pending_items=2, processing_items=1, stale_locks=0, active_workers=3
        )
        assert health == QueueHealthStatus.HEALTHY
        
        # Test offline status
        health = self.queue_service._calculate_queue_health(
            total_items=10, pending_items=5, processing_items=0, stale_locks=0, active_workers=0
        )
        assert health == QueueHealthStatus.OFFLINE
        
        # Test warning status (high pending ratio)
        health = self.queue_service._calculate_queue_health(
            total_items=10, pending_items=9, processing_items=0, stale_locks=0, active_workers=1
        )
        assert health == QueueHealthStatus.WARNING
        
        # Test critical status (high stale locks)
        health = self.queue_service._calculate_queue_health(
            total_items=10, pending_items=2, processing_items=1, stale_locks=2, active_workers=1
        )
        assert health == QueueHealthStatus.CRITICAL
    
    def test_error_handling(self):
        """Test error handling in queue service."""
        # Test getting queue item with invalid worker
        with pytest.raises(QueueServiceError):
            self.queue_service.get_next_queue_item("non_existent_queue", "invalid_worker")
        
        # Test completing non-existent queue item
        with pytest.raises(QueueServiceError):
            self.queue_service.complete_queue_item(
                99999, "worker1", BatchItemStatus.COMPLETED
            )
        
        # Test releasing queue item with wrong worker
        worker_info = self.queue_service.register_worker("test_queue")
        worker_id = worker_info.worker_id
        
        # Create a queue item locked by different worker
        batch = Batch(batch_id="error_test", status=BatchStatus.PROCESSING, total_items=1)
        self.session.add(batch)
        self.session.commit()
        
        batch_item = BatchItem(
            batch_id=batch.id,
            url="https://youtube.com/watch?v=error_test",
            status=BatchItemStatus.PROCESSING
        )
        self.session.add(batch_item)
        self.session.commit()
        
        queue_item = QueueItem(
            batch_item_id=batch_item.id,
            queue_name='test_queue',
            locked_by="other_worker",
            locked_at=datetime.utcnow()
        )
        self.session.add(queue_item)
        self.session.commit()
        
        # Try to release with wrong worker
        with pytest.raises(QueueServiceError):
            self.queue_service.release_queue_item(queue_item.id, worker_id)
    
    def test_priority_filtering(self, sample_queue_items):
        """Test priority filtering in queue processing."""
        # Set different priorities
        sample_queue_items[0].priority = BatchPriority.HIGH
        sample_queue_items[1].priority = BatchPriority.NORMAL
        sample_queue_items[2].priority = BatchPriority.LOW
        self.session.commit()
        
        # Register worker
        worker_info = self.queue_service.register_worker("test_queue")
        worker_id = worker_info.worker_id
        
        # Get item with HIGH priority filter
        queue_item = self.queue_service.get_next_queue_item(
            "test_queue", worker_id, priority_filter=[BatchPriority.HIGH]
        )
        assert queue_item is not None
        assert queue_item.priority == BatchPriority.HIGH
        
        # Complete the item
        self.queue_service.complete_queue_item(
            queue_item.id, worker_id, BatchItemStatus.COMPLETED
        )
        
        # Try to get HIGH priority item again (should be None)
        queue_item = self.queue_service.get_next_queue_item(
            "test_queue", worker_id, priority_filter=[BatchPriority.HIGH]
        )
        assert queue_item is None
    
    def test_batch_completion_trigger(self, sample_batch, sample_batch_items, sample_queue_items):
        """Test that batch completion is triggered when all items are processed."""
        # Register worker
        worker_info = self.queue_service.register_worker("test_queue")
        worker_id = worker_info.worker_id
        
        # Process all items
        for _ in range(3):
            queue_item = self.queue_service.get_next_queue_item("test_queue", worker_id)
            assert queue_item is not None
            
            self.queue_service.complete_queue_item(
                queue_item.id, worker_id, BatchItemStatus.COMPLETED
            )
        
        # Verify batch is completed
        self.session.refresh(sample_batch)
        assert sample_batch.status == BatchStatus.COMPLETED
        assert sample_batch.completed_items == 3
        assert sample_batch.completed_at is not None
    
    def test_context_manager(self):
        """Test queue service as context manager."""
        with QueueService(self.session, self.options) as service:
            worker_info = service.register_worker("test_queue")
            assert worker_info is not None
        
        # Service should be shut down after context exit
        # This is primarily for testing the context manager interface
    
    def test_concurrent_workers(self, sample_queue_items):
        """Test concurrent worker operations."""
        # Register multiple workers
        workers = []
        for i in range(3):
            worker_info = self.queue_service.register_worker("test_queue", f"worker_{i}")
            workers.append(worker_info)
        
        # Function to process items concurrently
        def worker_process(worker_id):
            queue_item = self.queue_service.get_next_queue_item("test_queue", worker_id)
            if queue_item:
                self.queue_service.complete_queue_item(
                    queue_item.id, worker_id, BatchItemStatus.COMPLETED
                )
        
        # Start concurrent processing
        threads = []
        for worker in workers:
            thread = threading.Thread(target=worker_process, args=(worker.worker_id,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify all items were processed
        stats = self.queue_service.get_queue_statistics("test_queue")
        assert stats.completed_items == 3
        assert stats.total_items == 3
    
    def test_queue_service_shutdown(self):
        """Test queue service shutdown functionality."""
        # Register workers
        worker1 = self.queue_service.register_worker("test_queue", "worker1")
        worker2 = self.queue_service.register_worker("test_queue", "worker2")
        
        # Verify workers are registered
        assert len(self.queue_service._worker_registry) == 2
        
        # Shutdown the service
        self.queue_service.shutdown()
        
        # Verify workers are unregistered
        assert len(self.queue_service._worker_registry) == 0
        
        # Verify cleanup thread is stopped
        assert not self.queue_service._cleanup_running


class TestQueueServiceIntegration:
    """Integration tests for QueueService with other components."""
    
    @pytest.fixture(autouse=True)
    def setup_integration_database(self):
        """Set up integration test database."""
        # Create in-memory SQLite database
        self.engine = create_engine("sqlite:///:memory:", echo=False)
        Base.metadata.create_all(self.engine)
        
        # Create session factory
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
        
        # Create queue service with minimal cleanup interval for testing
        self.options = QueueProcessingOptions(
            stale_lock_cleanup_interval_minutes=0.1,  # 6 seconds
            enable_automatic_cleanup=False  # Keep disabled for controlled testing
        )
        self.queue_service = QueueService(self.session, self.options)
        
        yield
        
        # Cleanup
        self.queue_service.shutdown()
        self.session.close()
    
    def test_integration_with_batch_service(self):
        """Test integration with batch processing workflow."""
        # Create a complete batch workflow
        batch = Batch(
            batch_id="integration_test_batch",
            name="Integration Test",
            status=BatchStatus.PROCESSING,
            total_items=2,
            priority=BatchPriority.HIGH
        )
        self.session.add(batch)
        self.session.commit()
        
        # Create batch items
        batch_items = []
        for i in range(2):
            item = BatchItem(
                batch_id=batch.id,
                url=f"https://youtube.com/watch?v=integration{i}",
                status=BatchItemStatus.QUEUED,
                priority=BatchPriority.HIGH,
                processing_order=i
            )
            batch_items.append(item)
        
        self.session.add_all(batch_items)
        self.session.commit()
        
        # Create queue items
        queue_items = []
        for batch_item in batch_items:
            item = QueueItem(
                batch_item_id=batch_item.id,
                queue_name='video_processing',
                priority=BatchPriority.HIGH
            )
            queue_items.append(item)
        
        self.session.add_all(queue_items)
        self.session.commit()
        
        # Process through queue service
        worker_info = self.queue_service.register_worker("video_processing")
        worker_id = worker_info.worker_id
        
        # Process first item
        queue_item = self.queue_service.get_next_queue_item("video_processing", worker_id)
        assert queue_item is not None
        assert queue_item.priority == BatchPriority.HIGH
        
        # Complete first item
        self.queue_service.complete_queue_item(
            queue_item.id, worker_id, BatchItemStatus.COMPLETED,
            result_data={"video_id": 1, "summary": "Test summary"}
        )
        
        # Process second item
        queue_item = self.queue_service.get_next_queue_item("video_processing", worker_id)
        assert queue_item is not None
        
        # Complete second item
        self.queue_service.complete_queue_item(
            queue_item.id, worker_id, BatchItemStatus.COMPLETED,
            result_data={"video_id": 2, "summary": "Test summary 2"}
        )
        
        # Verify batch is completed
        self.session.refresh(batch)
        assert batch.status == BatchStatus.COMPLETED
        assert batch.completed_items == 2
        assert batch.completed_at is not None
        
        # Verify queue statistics
        stats = self.queue_service.get_queue_statistics("video_processing")
        assert stats.completed_items == 2
        assert stats.total_items == 2
        assert stats.health_status == QueueHealthStatus.HEALTHY
    
    def test_queue_resilience_with_failures(self):
        """Test queue resilience when processing fails."""
        # Create batch with items
        batch = Batch(
            batch_id="resilience_test_batch",
            name="Resilience Test",
            status=BatchStatus.PROCESSING,
            total_items=3
        )
        self.session.add(batch)
        self.session.commit()
        
        # Create items with different retry limits
        batch_items = []
        for i in range(3):
            item = BatchItem(
                batch_id=batch.id,
                url=f"https://youtube.com/watch?v=resilience{i}",
                status=BatchItemStatus.QUEUED,
                max_retries=2,
                processing_order=i
            )
            batch_items.append(item)
        
        self.session.add_all(batch_items)
        self.session.commit()
        
        # Create queue items
        queue_items = []
        for batch_item in batch_items:
            item = QueueItem(
                batch_item_id=batch_item.id,
                queue_name='resilience_queue',
                max_retries=2
            )
            queue_items.append(item)
        
        self.session.add_all(queue_items)
        self.session.commit()
        
        # Register worker
        worker_info = self.queue_service.register_worker("resilience_queue")
        worker_id = worker_info.worker_id
        
        # Process items with failures and retries
        for attempt in range(3):  # Max retries + 1
            queue_item = self.queue_service.get_next_queue_item("resilience_queue", worker_id)
            if queue_item:
                if attempt < 2:  # Fail first two attempts
                    self.queue_service.complete_queue_item(
                        queue_item.id, worker_id, BatchItemStatus.FAILED,
                        error_message=f"Simulated failure attempt {attempt + 1}"
                    )
                    
                    # Retry the item
                    self.queue_service.retry_queue_item(queue_item.id, delay_minutes=0)
                else:  # Succeed on third attempt
                    self.queue_service.complete_queue_item(
                        queue_item.id, worker_id, BatchItemStatus.COMPLETED,
                        result_data={"success": True}
                    )
        
        # Verify queue statistics show resilience
        stats = self.queue_service.get_queue_statistics("resilience_queue")
        assert stats.completed_items >= 1  # At least one item should succeed
        assert stats.health_status in [QueueHealthStatus.HEALTHY, QueueHealthStatus.WARNING]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])