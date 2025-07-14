"""
Unit tests for concurrent batch processing service.

This module tests the advanced concurrency control for batch processing operations including:
- Thread-safe batch operations with proper locking
- Concurrent worker management with resource allocation
- Rate limiting and throttling for API calls
- Data consistency across concurrent operations
- Deadlock prevention and recovery
- Performance monitoring and optimization
"""

import unittest
import asyncio
import threading
import time
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import uuid

from ..database.batch_models import (
    Batch, BatchItem, QueueItem, ProcessingSession,
    BatchStatus, BatchItemStatus, BatchPriority
)
from ..services.batch_service import BatchCreateRequest
from ..utils.concurrency_manager import ResourceType, ResourceQuota, ConcurrencyManager

from .concurrent_batch_service import (
    ConcurrentBatchService, ConcurrentBatchConfig, ConcurrentBatchMode,
    WorkerState, WorkerInfo, ConcurrentBatchStatistics,
    ConcurrentBatchError, ConcurrentWorkerError, ConcurrentBatchTimeoutError,
    create_concurrent_batch_service, get_concurrent_batch_service
)


class TestConcurrentBatchConfig(unittest.TestCase):
    """Test cases for ConcurrentBatchConfig class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ConcurrentBatchConfig()
        
        self.assertEqual(config.max_concurrent_batches, 5)
        self.assertEqual(config.max_concurrent_items_per_batch, 10)
        self.assertEqual(config.max_total_concurrent_items, 50)
        self.assertEqual(config.max_workers_per_batch, 3)
        self.assertEqual(config.max_api_calls_per_second, 2.0)
        self.assertEqual(config.max_database_connections, 10)
        self.assertTrue(config.enable_rate_limiting)
        self.assertTrue(config.enable_resource_throttling)
        self.assertTrue(config.enable_deadlock_detection)
        self.assertTrue(config.enable_performance_monitoring)
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = ConcurrentBatchConfig(
            max_concurrent_batches=10,
            max_concurrent_items_per_batch=20,
            max_total_concurrent_items=100,
            max_workers_per_batch=5,
            max_api_calls_per_second=5.0,
            max_database_connections=20,
            enable_rate_limiting=False,
            enable_resource_throttling=False,
            enable_deadlock_detection=False,
            enable_performance_monitoring=False
        )
        
        self.assertEqual(config.max_concurrent_batches, 10)
        self.assertEqual(config.max_concurrent_items_per_batch, 20)
        self.assertEqual(config.max_total_concurrent_items, 100)
        self.assertEqual(config.max_workers_per_batch, 5)
        self.assertEqual(config.max_api_calls_per_second, 5.0)
        self.assertEqual(config.max_database_connections, 20)
        self.assertFalse(config.enable_rate_limiting)
        self.assertFalse(config.enable_resource_throttling)
        self.assertFalse(config.enable_deadlock_detection)
        self.assertFalse(config.enable_performance_monitoring)


class TestWorkerInfo(unittest.TestCase):
    """Test cases for WorkerInfo dataclass."""
    
    def test_worker_info_creation(self):
        """Test WorkerInfo creation with default values."""
        worker = WorkerInfo(
            worker_id="test_worker",
            batch_id="test_batch",
            worker_state=WorkerState.IDLE
        )
        
        self.assertEqual(worker.worker_id, "test_worker")
        self.assertEqual(worker.batch_id, "test_batch")
        self.assertEqual(worker.worker_state, WorkerState.IDLE)
        self.assertIsNone(worker.current_item_id)
        self.assertIsInstance(worker.started_at, datetime)
        self.assertIsInstance(worker.last_heartbeat, datetime)
        self.assertEqual(worker.processed_items, 0)
        self.assertEqual(worker.failed_items, 0)
        self.assertEqual(worker.total_processing_time, 0.0)
        self.assertIsNone(worker.current_operation)
        self.assertEqual(worker.error_count, 0)
        self.assertIsNone(worker.last_error)
        self.assertIsInstance(worker.metadata, dict)
    
    def test_worker_info_with_values(self):
        """Test WorkerInfo with custom values."""
        start_time = datetime.utcnow()
        
        worker = WorkerInfo(
            worker_id="test_worker",
            batch_id="test_batch",
            worker_state=WorkerState.PROCESSING,
            current_item_id=123,
            started_at=start_time,
            last_heartbeat=start_time,
            processed_items=5,
            failed_items=2,
            total_processing_time=30.5,
            current_operation="processing_video",
            error_count=1,
            last_error="Test error",
            metadata={"key": "value"}
        )
        
        self.assertEqual(worker.worker_id, "test_worker")
        self.assertEqual(worker.batch_id, "test_batch")
        self.assertEqual(worker.worker_state, WorkerState.PROCESSING)
        self.assertEqual(worker.current_item_id, 123)
        self.assertEqual(worker.started_at, start_time)
        self.assertEqual(worker.last_heartbeat, start_time)
        self.assertEqual(worker.processed_items, 5)
        self.assertEqual(worker.failed_items, 2)
        self.assertEqual(worker.total_processing_time, 30.5)
        self.assertEqual(worker.current_operation, "processing_video")
        self.assertEqual(worker.error_count, 1)
        self.assertEqual(worker.last_error, "Test error")
        self.assertEqual(worker.metadata, {"key": "value"})


class TestConcurrentBatchService(unittest.TestCase):
    """Test cases for ConcurrentBatchService class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = ConcurrentBatchConfig(
            max_concurrent_batches=2,
            max_concurrent_items_per_batch=3,
            max_total_concurrent_items=10,
            max_workers_per_batch=2,
            max_api_calls_per_second=5.0,
            max_database_connections=5,
            enable_performance_monitoring=False,  # Disable for testing
            cleanup_interval_seconds=1,
            heartbeat_interval_seconds=1
        )
        
        # Mock database session
        self.mock_session = Mock()
        
        # Mock concurrency manager
        self.mock_concurrency_manager = Mock()
        self.mock_concurrency_manager.configure_resource_quota = Mock()
        self.mock_concurrency_manager.allocate_resource = Mock()
        self.mock_concurrency_manager.get_lock = Mock()
        self.mock_concurrency_manager.get_comprehensive_statistics = Mock(return_value={})
        
        # Create service instance
        self.service = ConcurrentBatchService(
            config=self.config,
            session=self.mock_session,
            concurrency_manager=self.mock_concurrency_manager
        )
    
    def tearDown(self):
        """Clean up after tests."""
        self.service.shutdown()
    
    def test_service_initialization(self):
        """Test service initialization."""
        self.assertIsInstance(self.service.config, ConcurrentBatchConfig)
        self.assertEqual(self.service._session, self.mock_session)
        self.assertEqual(self.service._concurrency_manager, self.mock_concurrency_manager)
        self.assertIsInstance(self.service._statistics, ConcurrentBatchStatistics)
        self.assertIsInstance(self.service._worker_registry, dict)
        self.assertIsInstance(self.service._active_batches, set)
        self.assertIsInstance(self.service._batch_workers, dict)
        self.assertIsInstance(self.service._batch_locks, dict)
    
    def test_resource_quota_configuration(self):
        """Test resource quota configuration during initialization."""
        # Verify that configure_resource_quota was called for each resource type
        self.mock_concurrency_manager.configure_resource_quota.assert_any_call(
            ResourceType.DATABASE_CONNECTION, 
            unittest.mock.ANY
        )
        self.mock_concurrency_manager.configure_resource_quota.assert_any_call(
            ResourceType.API_REQUEST, 
            unittest.mock.ANY
        )
        self.mock_concurrency_manager.configure_resource_quota.assert_any_call(
            ResourceType.WORKER_THREAD, 
            unittest.mock.ANY
        )
    
    @patch('src.services.concurrent_batch_service.get_database_session')
    @patch('src.services.concurrent_batch_service.BatchService')
    @patch('src.services.concurrent_batch_service.allocate_resource')
    async def test_create_concurrent_batch(self, mock_allocate, mock_batch_service_class, mock_get_session):
        """Test concurrent batch creation."""
        # Mock dependencies
        mock_get_session.return_value = self.mock_session
        mock_batch_service = Mock()
        mock_batch_service_class.return_value = mock_batch_service
        
        # Mock batch creation
        mock_batch = Mock()
        mock_batch.batch_id = "test_batch_123"
        mock_batch.total_items = 5
        mock_batch.batch_metadata = {}
        mock_batch_service.create_batch.return_value = mock_batch
        
        # Mock resource allocation
        mock_allocate.return_value.__enter__ = Mock(return_value=None)
        mock_allocate.return_value.__exit__ = Mock(return_value=None)
        
        # Create batch request
        request = BatchCreateRequest(
            name="Test Batch",
            description="Test batch description",
            urls=["https://youtube.com/watch?v=123", "https://youtube.com/watch?v=456"],
            priority=BatchPriority.NORMAL
        )
        
        # Test batch creation
        result = await self.service.create_concurrent_batch(
            request, 
            ConcurrentBatchMode.PARALLEL,
            max_concurrent_items=3
        )
        
        self.assertEqual(result, mock_batch)
        self.assertIn("test_batch_123", self.service._active_batches)
        self.assertEqual(self.service._statistics.total_batches, 1)
        self.assertEqual(self.service._statistics.active_batches, 1)
        self.assertEqual(self.service._statistics.total_items, 5)
    
    async def test_create_concurrent_batch_quota_exceeded(self):
        """Test batch creation when quota is exceeded."""
        # Fill up the active batches to max capacity
        for i in range(self.config.max_concurrent_batches):
            self.service._active_batches.add(f"batch_{i}")
        
        request = BatchCreateRequest(
            name="Test Batch",
            urls=["https://youtube.com/watch?v=123"],
            priority=BatchPriority.NORMAL
        )
        
        # Should raise error when quota exceeded
        with self.assertRaises(ConcurrentBatchError):
            await self.service.create_concurrent_batch(request)
    
    def test_worker_creation(self):
        """Test worker creation."""
        batch_id = "test_batch"
        worker_id = "test_worker"
        
        # Add batch to active batches
        self.service._active_batches.add(batch_id)
        self.service._batch_workers[batch_id] = set()
        
        # Create worker
        worker = self.service._create_batch_worker(batch_id, worker_id)
        
        self.assertEqual(worker.worker_id, worker_id)
        self.assertEqual(worker.batch_id, batch_id)
        self.assertEqual(worker.worker_state, WorkerState.INITIALIZING)
        self.assertIn(worker_id, self.service._worker_registry)
        self.assertIn(worker_id, self.service._batch_workers[batch_id])
        self.assertEqual(self.service._statistics.total_workers, 1)
        self.assertEqual(self.service._statistics.active_workers, 1)
    
    def test_worker_cleanup(self):
        """Test worker cleanup."""
        batch_id = "test_batch"
        worker_id = "test_worker"
        
        # Set up worker
        self.service._active_batches.add(batch_id)
        self.service._batch_workers[batch_id] = {worker_id}
        worker = WorkerInfo(worker_id=worker_id, batch_id=batch_id, worker_state=WorkerState.IDLE)
        self.service._worker_registry[worker_id] = worker
        self.service._statistics.active_workers = 1
        
        # Clean up worker
        self.service._cleanup_worker(worker_id)
        
        self.assertNotIn(worker_id, self.service._worker_registry)
        self.assertNotIn(worker_id, self.service._batch_workers[batch_id])
        self.assertEqual(self.service._statistics.active_workers, 0)
    
    def test_get_batch_lock(self):
        """Test batch lock retrieval."""
        batch_id = "test_batch"
        
        # Get lock for first time
        lock1 = self.service._get_batch_lock(batch_id)
        self.assertIsNotNone(lock1)
        
        # Get same lock again
        lock2 = self.service._get_batch_lock(batch_id)
        self.assertIs(lock1, lock2)  # Should be same instance
        
        # Get different lock
        lock3 = self.service._get_batch_lock("different_batch")
        self.assertIsNot(lock1, lock3)
    
    def test_pause_batch_processing(self):
        """Test pausing batch processing."""
        batch_id = "test_batch"
        
        # Create some workers
        self.service._batch_workers[batch_id] = set()
        for i in range(3):
            worker_id = f"worker_{i}"
            worker = WorkerInfo(
                worker_id=worker_id,
                batch_id=batch_id,
                worker_state=WorkerState.IDLE
            )
            self.service._worker_registry[worker_id] = worker
            self.service._batch_workers[batch_id].add(worker_id)
        
        # Pause batch processing
        result = self.service.pause_batch_processing(batch_id)
        
        self.assertTrue(result)
        
        # Check that all workers are paused
        for worker_id in self.service._batch_workers[batch_id]:
            worker = self.service._worker_registry[worker_id]
            self.assertEqual(worker.worker_state, WorkerState.PAUSED)
    
    def test_resume_batch_processing(self):
        """Test resuming batch processing."""
        batch_id = "test_batch"
        
        # Create some paused workers
        self.service._batch_workers[batch_id] = set()
        for i in range(3):
            worker_id = f"worker_{i}"
            worker = WorkerInfo(
                worker_id=worker_id,
                batch_id=batch_id,
                worker_state=WorkerState.PAUSED
            )
            self.service._worker_registry[worker_id] = worker
            self.service._batch_workers[batch_id].add(worker_id)
        
        # Resume batch processing
        result = self.service.resume_batch_processing(batch_id)
        
        self.assertTrue(result)
        
        # Check that all workers are resumed
        for worker_id in self.service._batch_workers[batch_id]:
            worker = self.service._worker_registry[worker_id]
            self.assertEqual(worker.worker_state, WorkerState.IDLE)
    
    def test_cancel_batch_processing(self):
        """Test cancelling batch processing."""
        batch_id = "test_batch"
        
        # Add batch to active batches
        self.service._active_batches.add(batch_id)
        self.service._batch_workers[batch_id] = set()
        
        # Create some workers
        for i in range(3):
            worker_id = f"worker_{i}"
            worker = WorkerInfo(
                worker_id=worker_id,
                batch_id=batch_id,
                worker_state=WorkerState.PROCESSING
            )
            self.service._worker_registry[worker_id] = worker
            self.service._batch_workers[batch_id].add(worker_id)
        
        # Cancel batch processing
        result = self.service.cancel_batch_processing(batch_id)
        
        self.assertTrue(result)
        self.assertNotIn(batch_id, self.service._active_batches)
        self.assertNotIn(batch_id, self.service._batch_workers)
    
    def test_get_concurrent_statistics(self):
        """Test getting concurrent statistics."""
        # Set up some test data
        batch_id = "test_batch"
        worker_id = "test_worker"
        
        self.service._active_batches.add(batch_id)
        self.service._batch_workers[batch_id] = {worker_id}
        
        worker = WorkerInfo(
            worker_id=worker_id,
            batch_id=batch_id,
            worker_state=WorkerState.PROCESSING,
            current_item_id=123,
            processed_items=5,
            failed_items=2,
            error_count=1,
            last_error="Test error"
        )
        self.service._worker_registry[worker_id] = worker
        
        # Update statistics
        self.service._statistics.total_batches = 10
        self.service._statistics.active_batches = 1
        self.service._statistics.completed_batches = 8
        self.service._statistics.failed_batches = 1
        
        # Get statistics
        stats = self.service.get_concurrent_statistics()
        
        self.assertIn('service_statistics', stats)
        self.assertIn('active_batches', stats)
        self.assertIn('worker_details', stats)
        self.assertIn('concurrency_statistics', stats)
        
        # Check service statistics
        service_stats = stats['service_statistics']
        self.assertEqual(service_stats['total_batches'], 10)
        self.assertEqual(service_stats['active_batches'], 1)
        self.assertEqual(service_stats['completed_batches'], 8)
        self.assertEqual(service_stats['failed_batches'], 1)
        
        # Check active batches
        self.assertEqual(stats['active_batches'], [batch_id])
        
        # Check worker details
        worker_details = stats['worker_details']
        self.assertIn(worker_id, worker_details)
        worker_detail = worker_details[worker_id]
        self.assertEqual(worker_detail['batch_id'], batch_id)
        self.assertEqual(worker_detail['state'], WorkerState.PROCESSING.value)
        self.assertEqual(worker_detail['current_item_id'], 123)
        self.assertEqual(worker_detail['processed_items'], 5)
        self.assertEqual(worker_detail['failed_items'], 2)
        self.assertEqual(worker_detail['error_count'], 1)
        self.assertEqual(worker_detail['last_error'], "Test error")
    
    def test_context_manager(self):
        """Test context manager functionality."""
        config = ConcurrentBatchConfig(enable_performance_monitoring=False)
        
        with ConcurrentBatchService(config=config) as service:
            self.assertIsInstance(service, ConcurrentBatchService)
            self.assertIsInstance(service._statistics, ConcurrentBatchStatistics)
        
        # Service should be shut down after context exit
        self.assertTrue(service._shutdown_event.is_set())
    
    def test_stale_worker_cleanup(self):
        """Test cleanup of stale workers."""
        # Create worker with old heartbeat
        worker_id = "stale_worker"
        batch_id = "test_batch"
        
        old_time = datetime.utcnow() - timedelta(seconds=self.config.worker_timeout_seconds + 10)
        
        worker = WorkerInfo(
            worker_id=worker_id,
            batch_id=batch_id,
            worker_state=WorkerState.IDLE,
            last_heartbeat=old_time
        )
        
        self.service._worker_registry[worker_id] = worker
        self.service._batch_workers[batch_id] = {worker_id}
        self.service._statistics.active_workers = 1
        
        # Run cleanup
        self.service._cleanup_stale_workers()
        
        # Worker should be cleaned up
        self.assertNotIn(worker_id, self.service._worker_registry)
        self.assertNotIn(worker_id, self.service._batch_workers.get(batch_id, set()))
        self.assertEqual(self.service._statistics.active_workers, 0)
    
    def test_unused_lock_cleanup(self):
        """Test cleanup of unused locks."""
        batch_id = "test_batch"
        
        # Create a lock for inactive batch
        lock = self.service._get_batch_lock(batch_id)
        self.assertIn(batch_id, self.service._batch_locks)
        
        # Run cleanup (batch is not in active batches)
        self.service._cleanup_unused_locks()
        
        # Lock should be cleaned up
        self.assertNotIn(batch_id, self.service._batch_locks)
    
    def test_statistics_update(self):
        """Test statistics update."""
        # Create some workers
        batch_id = "test_batch"
        
        for i in range(3):
            worker_id = f"worker_{i}"
            worker = WorkerInfo(
                worker_id=worker_id,
                batch_id=batch_id,
                worker_state=WorkerState.PROCESSING if i < 2 else WorkerState.STOPPED,
                total_processing_time=10.0 + i
            )
            self.service._worker_registry[worker_id] = worker
        
        # Set some completed items
        self.service._statistics.completed_items = 5
        self.service._statistics.failed_items = 2
        
        # Update statistics
        self.service._update_statistics()
        
        # Check updated values
        self.assertEqual(self.service._statistics.active_workers, 2)  # 2 processing workers
        self.assertEqual(self.service._statistics.average_processing_time, 6.2)  # (10+11+12)/5
        self.assertEqual(self.service._statistics.error_rate, 2/7)  # 2 failed out of 7 total
        self.assertIsInstance(self.service._statistics.last_updated, datetime)


class TestConcurrentBatchIntegration(unittest.TestCase):
    """Integration tests for concurrent batch processing."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = ConcurrentBatchConfig(
            max_concurrent_batches=2,
            max_concurrent_items_per_batch=3,
            max_total_concurrent_items=5,
            max_workers_per_batch=2,
            enable_performance_monitoring=False,
            cleanup_interval_seconds=60,  # Disable frequent cleanup
            heartbeat_interval_seconds=60
        )
    
    def tearDown(self):
        """Clean up after tests."""
        pass
    
    @patch('src.services.concurrent_batch_service.get_database_session')
    @patch('src.services.concurrent_batch_service.BatchService')
    @patch('src.services.concurrent_batch_service.QueueService')
    @patch('src.services.concurrent_batch_service.BatchProcessor')
    @patch('src.services.concurrent_batch_service.allocate_resource')
    @patch('src.services.concurrent_batch_service.acquire_exclusive_lock')
    async def test_concurrent_batch_processing_flow(self, mock_acquire_lock, mock_allocate, 
                                                   mock_batch_processor_class, mock_queue_service_class,
                                                   mock_batch_service_class, mock_get_session):
        """Test complete concurrent batch processing flow."""
        # Set up mocks
        mock_session = Mock()
        mock_get_session.return_value = mock_session
        
        # Mock BatchService
        mock_batch_service = Mock()
        mock_batch_service_class.return_value = mock_batch_service
        
        mock_batch = Mock()
        mock_batch.batch_id = "test_batch"
        mock_batch.status = BatchStatus.PENDING
        mock_batch.total_items = 3
        mock_batch_service.get_batch.return_value = mock_batch
        mock_batch_service.start_batch_processing.return_value = True
        
        # Mock progress updates
        progress_updates = [
            Mock(status=BatchStatus.PROCESSING, progress_percentage=0.0),
            Mock(status=BatchStatus.PROCESSING, progress_percentage=33.3),
            Mock(status=BatchStatus.PROCESSING, progress_percentage=66.6),
            Mock(status=BatchStatus.COMPLETED, progress_percentage=100.0,
                 total_items=3, completed_items=3, failed_items=0)
        ]
        mock_batch_service.get_batch_progress.side_effect = progress_updates
        
        # Mock QueueService
        mock_queue_service = Mock()
        mock_queue_service_class.return_value = mock_queue_service
        
        # Mock queue items
        mock_queue_items = [
            Mock(id=1, batch_item_id=1, batch_item=Mock(batch=mock_batch)),
            Mock(id=2, batch_item_id=2, batch_item=Mock(batch=mock_batch)),
            Mock(id=3, batch_item_id=3, batch_item=Mock(batch=mock_batch)),
            None, None, None  # No more items
        ]
        mock_queue_service.get_next_queue_item.side_effect = mock_queue_items
        mock_queue_service.complete_queue_item.return_value = True
        
        # Mock BatchProcessor
        mock_batch_processor = Mock()
        mock_batch_processor_class.return_value = mock_batch_processor
        
        # Mock successful processing results
        mock_result = Mock()
        mock_result.status = BatchItemStatus.COMPLETED
        mock_result.result_data = {"success": True}
        mock_result.error_message = None
        
        async def mock_process_batch_item(batch_item_id, worker_id):
            await asyncio.sleep(0.01)  # Simulate processing time
            return mock_result
        
        mock_batch_processor.process_batch_item = mock_process_batch_item
        
        # Mock resource allocation
        mock_allocate.return_value.__enter__ = Mock(return_value=None)
        mock_allocate.return_value.__exit__ = Mock(return_value=None)
        
        # Mock lock acquisition
        mock_acquire_lock.return_value.__enter__ = Mock(return_value=None)
        mock_acquire_lock.return_value.__exit__ = Mock(return_value=None)
        
        # Create service and process batch
        with ConcurrentBatchService(config=self.config) as service:
            result = await service.process_batch_concurrently(
                "test_batch",
                max_workers=2
            )
            
            # Verify results
            self.assertEqual(result['batch_id'], "test_batch")
            self.assertEqual(result['status'], BatchStatus.COMPLETED.value)
            self.assertEqual(result['total_items'], 3)
            self.assertEqual(result['completed_items'], 3)
            self.assertEqual(result['failed_items'], 0)
            self.assertEqual(result['progress_percentage'], 100.0)
            self.assertEqual(result['workers_used'], 2)
            self.assertGreater(result['processing_time_seconds'], 0)
            
            # Verify service calls
            mock_batch_service.get_batch.assert_called_with("test_batch")
            mock_batch_service.start_batch_processing.assert_called_with("test_batch")
            self.assertTrue(mock_queue_service.get_next_queue_item.called)
            self.assertTrue(mock_queue_service.complete_queue_item.called)
    
    def test_concurrent_worker_management(self):
        """Test concurrent worker management."""
        with ConcurrentBatchService(config=self.config) as service:
            batch_id = "test_batch"
            
            # Create multiple workers
            workers = []
            for i in range(3):
                worker_id = f"worker_{i}"
                worker = service._create_batch_worker(batch_id, worker_id)
                workers.append(worker)
            
            # Verify workers are created
            self.assertEqual(len(service._worker_registry), 3)
            self.assertEqual(len(service._batch_workers[batch_id]), 3)
            
            # Test worker state management
            for worker in workers:
                self.assertEqual(worker.worker_state, WorkerState.INITIALIZING)
            
            # Test pause/resume
            service.pause_batch_processing(batch_id)
            for worker_id in service._batch_workers[batch_id]:
                worker = service._worker_registry[worker_id]
                self.assertEqual(worker.worker_state, WorkerState.PAUSED)
            
            service.resume_batch_processing(batch_id)
            for worker_id in service._batch_workers[batch_id]:
                worker = service._worker_registry[worker_id]
                self.assertEqual(worker.worker_state, WorkerState.IDLE)
            
            # Test cancellation
            service.cancel_batch_processing(batch_id)
            self.assertNotIn(batch_id, service._active_batches)
            self.assertNotIn(batch_id, service._batch_workers)
    
    def test_error_handling_and_recovery(self):
        """Test error handling and recovery mechanisms."""
        with ConcurrentBatchService(config=self.config) as service:
            batch_id = "test_batch"
            worker_id = "test_worker"
            
            # Create worker
            worker = service._create_batch_worker(batch_id, worker_id)
            
            # Simulate error condition
            worker.worker_state = WorkerState.ERROR
            worker.error_count = 3
            worker.last_error = "Test error"
            
            # Test error recovery
            worker.worker_state = WorkerState.IDLE
            worker.error_count = 0
            worker.last_error = None
            
            self.assertEqual(worker.worker_state, WorkerState.IDLE)
            self.assertEqual(worker.error_count, 0)
            self.assertIsNone(worker.last_error)
    
    def test_resource_contention_handling(self):
        """Test handling of resource contention."""
        # Create tight resource limits
        config = ConcurrentBatchConfig(
            max_concurrent_batches=1,
            max_concurrent_items_per_batch=1,
            max_total_concurrent_items=2,
            max_workers_per_batch=1,
            enable_performance_monitoring=False
        )
        
        with ConcurrentBatchService(config=config) as service:
            # Try to create more batches than allowed
            service._active_batches.add("existing_batch")
            
            # This should fail due to resource limits
            with self.assertRaises(ConcurrentBatchError):
                async def test_batch_creation():
                    request = BatchCreateRequest(
                        name="Test Batch",
                        urls=["https://youtube.com/watch?v=123"],
                        priority=BatchPriority.NORMAL
                    )
                    await service.create_concurrent_batch(request)
                
                asyncio.run(test_batch_creation())


class TestFactoryFunctions(unittest.TestCase):
    """Test cases for factory functions."""
    
    def test_create_concurrent_batch_service(self):
        """Test create_concurrent_batch_service factory function."""
        config = ConcurrentBatchConfig(max_concurrent_batches=3)
        
        service = create_concurrent_batch_service(config)
        
        self.assertIsInstance(service, ConcurrentBatchService)
        self.assertEqual(service.config.max_concurrent_batches, 3)
        
        service.shutdown()
    
    def test_create_concurrent_batch_service_default_config(self):
        """Test create_concurrent_batch_service with default config."""
        service = create_concurrent_batch_service()
        
        self.assertIsInstance(service, ConcurrentBatchService)
        self.assertIsInstance(service.config, ConcurrentBatchConfig)
        
        service.shutdown()
    
    def test_get_concurrent_batch_service(self):
        """Test get_concurrent_batch_service factory function."""
        mock_session = Mock()
        config = ConcurrentBatchConfig(max_concurrent_batches=5)
        
        service = get_concurrent_batch_service(mock_session, config)
        
        self.assertIsInstance(service, ConcurrentBatchService)
        self.assertEqual(service._session, mock_session)
        self.assertEqual(service.config.max_concurrent_batches, 5)
        
        service.shutdown()


class TestPerformanceAndStress(unittest.TestCase):
    """Performance and stress tests for concurrent batch processing."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = ConcurrentBatchConfig(
            max_concurrent_batches=10,
            max_concurrent_items_per_batch=20,
            max_total_concurrent_items=100,
            max_workers_per_batch=5,
            enable_performance_monitoring=False,
            cleanup_interval_seconds=60,
            heartbeat_interval_seconds=60
        )
    
    def tearDown(self):
        """Clean up after tests."""
        pass
    
    def test_high_concurrency_worker_management(self):
        """Test worker management under high concurrency."""
        with ConcurrentBatchService(config=self.config) as service:
            batch_id = "stress_test_batch"
            num_workers = 50
            
            # Create many workers
            workers = []
            for i in range(num_workers):
                worker_id = f"worker_{i}"
                worker = service._create_batch_worker(batch_id, worker_id)
                workers.append(worker)
            
            # Verify all workers are created
            self.assertEqual(len(service._worker_registry), num_workers)
            self.assertEqual(len(service._batch_workers[batch_id]), num_workers)
            
            # Test concurrent state changes
            def change_worker_state(worker_id, state):
                if worker_id in service._worker_registry:
                    service._worker_registry[worker_id].worker_state = state
            
            # Use ThreadPoolExecutor to simulate concurrent operations
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = []
                
                # Submit state change tasks
                for i, worker in enumerate(workers):
                    state = WorkerState.PROCESSING if i % 2 == 0 else WorkerState.IDLE
                    future = executor.submit(change_worker_state, worker.worker_id, state)
                    futures.append(future)
                
                # Wait for all tasks to complete
                for future in as_completed(futures):
                    future.result()
            
            # Verify final states
            processing_count = sum(1 for w in service._worker_registry.values() 
                                 if w.worker_state == WorkerState.PROCESSING)
            idle_count = sum(1 for w in service._worker_registry.values() 
                           if w.worker_state == WorkerState.IDLE)
            
            self.assertEqual(processing_count + idle_count, num_workers)
            
            # Clean up all workers
            service.cancel_batch_processing(batch_id)
            
            # Verify cleanup
            self.assertNotIn(batch_id, service._active_batches)
            self.assertNotIn(batch_id, service._batch_workers)
    
    def test_rapid_batch_creation_and_cleanup(self):
        """Test rapid batch creation and cleanup."""
        with ConcurrentBatchService(config=self.config) as service:
            batch_ids = []
            
            # Rapidly create batches
            for i in range(5):
                batch_id = f"rapid_batch_{i}"
                service._active_batches.add(batch_id)
                service._batch_workers[batch_id] = set()
                batch_ids.append(batch_id)
            
            # Verify batches are created
            self.assertEqual(len(service._active_batches), 5)
            
            # Rapidly clean up batches
            for batch_id in batch_ids:
                service.cancel_batch_processing(batch_id)
            
            # Verify cleanup
            self.assertEqual(len(service._active_batches), 0)
            self.assertEqual(len(service._batch_workers), 0)
    
    def test_statistics_under_load(self):
        """Test statistics collection under load."""
        with ConcurrentBatchService(config=self.config) as service:
            # Create many workers across multiple batches
            for batch_i in range(5):
                batch_id = f"batch_{batch_i}"
                service._batch_workers[batch_id] = set()
                
                for worker_i in range(10):
                    worker_id = f"worker_{batch_i}_{worker_i}"
                    worker = WorkerInfo(
                        worker_id=worker_id,
                        batch_id=batch_id,
                        worker_state=WorkerState.PROCESSING,
                        processed_items=worker_i * 2,
                        failed_items=worker_i,
                        total_processing_time=worker_i * 5.0
                    )
                    service._worker_registry[worker_id] = worker
                    service._batch_workers[batch_id].add(worker_id)
            
            # Update statistics
            service._statistics.total_batches = 20
            service._statistics.completed_items = 100
            service._statistics.failed_items = 20
            
            # Collect statistics multiple times concurrently
            def collect_stats():
                return service.get_concurrent_statistics()
            
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(collect_stats) for _ in range(10)]
                
                results = []
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
            
            # Verify all statistics collections succeeded
            self.assertEqual(len(results), 10)
            
            # Verify statistics content
            for stats in results:
                self.assertIn('service_statistics', stats)
                self.assertIn('worker_details', stats)
                self.assertEqual(len(stats['worker_details']), 50)  # 5 batches * 10 workers


if __name__ == '__main__':
    unittest.main()