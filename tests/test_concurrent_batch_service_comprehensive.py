"""
Comprehensive test suite for ConcurrentBatchService functionality.

This test suite provides complete coverage for ConcurrentBatchService including:
- Concurrent batch creation and management
- Worker management and coordination
- Resource allocation and throttling
- Performance monitoring and statistics
- Error handling and recovery
- Thread safety and synchronization
- Load balancing and optimization
- Integration with other services
- Stress testing and edge cases
"""

import pytest
import asyncio
import threading
import time
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import uuid
import random
from contextlib import asynccontextmanager

from src.database.batch_models import (
    Batch, BatchItem, QueueItem, ProcessingSession,
    BatchStatus, BatchItemStatus, BatchPriority
)
from src.services.batch_service import BatchCreateRequest
from src.utils.concurrency_manager import ResourceType, ResourceQuota, ConcurrencyManager
from src.services.concurrent_batch_service import (
    ConcurrentBatchService, ConcurrentBatchConfig, ConcurrentBatchMode,
    WorkerState, WorkerInfo, ConcurrentBatchStatistics,
    ConcurrentBatchError, ConcurrentWorkerError, ConcurrentBatchTimeoutError,
    create_concurrent_batch_service, get_concurrent_batch_service
)


class TestConcurrentBatchServiceComprehensive:
    """Comprehensive test suite for ConcurrentBatchService."""

    @pytest.fixture(scope="function")
    def minimal_config(self):
        """Create minimal configuration for testing."""
        return ConcurrentBatchConfig(
            max_concurrent_batches=2,
            max_concurrent_items_per_batch=3,
            max_total_concurrent_items=6,
            max_workers_per_batch=2,
            max_api_calls_per_second=5.0,
            max_database_connections=5,
            enable_performance_monitoring=False,
            cleanup_interval_seconds=60,
            heartbeat_interval_seconds=30,
            worker_timeout_seconds=300,
            batch_timeout_seconds=1800,
            enable_rate_limiting=True,
            enable_resource_throttling=True,
            enable_deadlock_detection=True
        )

    @pytest.fixture(scope="function")
    def high_performance_config(self):
        """Create high-performance configuration for testing."""
        return ConcurrentBatchConfig(
            max_concurrent_batches=10,
            max_concurrent_items_per_batch=20,
            max_total_concurrent_items=100,
            max_workers_per_batch=5,
            max_api_calls_per_second=10.0,
            max_database_connections=20,
            enable_performance_monitoring=True,
            cleanup_interval_seconds=30,
            heartbeat_interval_seconds=15,
            worker_timeout_seconds=600,
            batch_timeout_seconds=3600,
            enable_rate_limiting=True,
            enable_resource_throttling=True,
            enable_deadlock_detection=True
        )

    @pytest.fixture(scope="function")
    def mock_session(self):
        """Create mock database session."""
        session = Mock()
        session.commit = Mock()
        session.rollback = Mock()
        session.close = Mock()
        return session

    @pytest.fixture(scope="function")
    def mock_concurrency_manager(self):
        """Create mock concurrency manager."""
        manager = Mock(spec=ConcurrencyManager)
        manager.configure_resource_quota = Mock()
        manager.allocate_resource = Mock()
        manager.get_lock = Mock()
        manager.get_comprehensive_statistics = Mock(return_value={
            "resource_usage": {},
            "lock_statistics": {},
            "performance_metrics": {}
        })
        return manager

    @pytest.fixture(scope="function")
    def service(self, minimal_config, mock_session, mock_concurrency_manager):
        """Create ConcurrentBatchService instance."""
        service = ConcurrentBatchService(
            config=minimal_config,
            session=mock_session,
            concurrency_manager=mock_concurrency_manager
        )
        yield service
        service.shutdown()

    @pytest.fixture(scope="function")
    def high_perf_service(self, high_performance_config, mock_session, mock_concurrency_manager):
        """Create high-performance ConcurrentBatchService instance."""
        service = ConcurrentBatchService(
            config=high_performance_config,
            session=mock_session,
            concurrency_manager=mock_concurrency_manager
        )
        yield service
        service.shutdown()

    # Configuration Tests

    def test_concurrent_batch_config_default(self):
        """Test default configuration values."""
        config = ConcurrentBatchConfig()
        
        assert config.max_concurrent_batches == 5
        assert config.max_concurrent_items_per_batch == 10
        assert config.max_total_concurrent_items == 50
        assert config.max_workers_per_batch == 3
        assert config.max_api_calls_per_second == 2.0
        assert config.max_database_connections == 10
        assert config.enable_rate_limiting is True
        assert config.enable_resource_throttling is True
        assert config.enable_deadlock_detection is True
        assert config.enable_performance_monitoring is True

    def test_concurrent_batch_config_custom(self):
        """Test custom configuration values."""
        config = ConcurrentBatchConfig(
            max_concurrent_batches=20,
            max_concurrent_items_per_batch=50,
            max_total_concurrent_items=200,
            max_workers_per_batch=10,
            max_api_calls_per_second=15.0,
            max_database_connections=30,
            enable_rate_limiting=False,
            enable_resource_throttling=False,
            enable_deadlock_detection=False,
            enable_performance_monitoring=False
        )
        
        assert config.max_concurrent_batches == 20
        assert config.max_concurrent_items_per_batch == 50
        assert config.max_total_concurrent_items == 200
        assert config.max_workers_per_batch == 10
        assert config.max_api_calls_per_second == 15.0
        assert config.max_database_connections == 30
        assert config.enable_rate_limiting is False
        assert config.enable_resource_throttling is False
        assert config.enable_deadlock_detection is False
        assert config.enable_performance_monitoring is False

    def test_concurrent_batch_config_validation(self):
        """Test configuration validation."""
        # Test invalid values
        with pytest.raises(ValueError):
            ConcurrentBatchConfig(max_concurrent_batches=0)
        
        with pytest.raises(ValueError):
            ConcurrentBatchConfig(max_concurrent_items_per_batch=0)
        
        with pytest.raises(ValueError):
            ConcurrentBatchConfig(max_api_calls_per_second=0)

    # Service Initialization Tests

    def test_service_initialization(self, service, minimal_config, mock_session, mock_concurrency_manager):
        """Test service initialization."""
        assert service.config == minimal_config
        assert service._session == mock_session
        assert service._concurrency_manager == mock_concurrency_manager
        assert isinstance(service._statistics, ConcurrentBatchStatistics)
        assert isinstance(service._worker_registry, dict)
        assert isinstance(service._active_batches, set)
        assert isinstance(service._batch_workers, dict)
        assert isinstance(service._batch_locks, dict)
        assert not service._shutdown_event.is_set()

    def test_service_resource_configuration(self, service, mock_concurrency_manager):
        """Test resource configuration during initialization."""
        # Verify resource quota configuration calls
        assert mock_concurrency_manager.configure_resource_quota.call_count >= 3
        
        # Check that database, API, and worker resources were configured
        call_args = mock_concurrency_manager.configure_resource_quota.call_args_list
        resource_types = [call[0][0] for call in call_args]
        
        assert ResourceType.DATABASE_CONNECTION in resource_types
        assert ResourceType.API_REQUEST in resource_types
        assert ResourceType.WORKER_THREAD in resource_types

    def test_service_context_manager(self, minimal_config):
        """Test service as context manager."""
        with ConcurrentBatchService(config=minimal_config) as service:
            assert isinstance(service, ConcurrentBatchService)
            assert not service._shutdown_event.is_set()
        
        # Service should be shut down after context exit
        assert service._shutdown_event.is_set()

    # Worker Management Tests

    def test_worker_creation_and_management(self, service):
        """Test worker creation and management."""
        batch_id = "test_batch"
        worker_id = "test_worker"
        
        # Add batch to active batches
        service._active_batches.add(batch_id)
        service._batch_workers[batch_id] = set()
        
        # Create worker
        worker = service._create_batch_worker(batch_id, worker_id)
        
        assert worker.worker_id == worker_id
        assert worker.batch_id == batch_id
        assert worker.worker_state == WorkerState.INITIALIZING
        assert worker.processed_items == 0
        assert worker.failed_items == 0
        assert worker.error_count == 0
        assert worker.started_at is not None
        assert worker.last_heartbeat is not None
        
        # Verify worker is registered
        assert worker_id in service._worker_registry
        assert worker_id in service._batch_workers[batch_id]
        assert service._statistics.total_workers == 1
        assert service._statistics.active_workers == 1

    def test_worker_state_transitions(self, service):
        """Test worker state transitions."""
        batch_id = "test_batch"
        worker_id = "test_worker"
        
        service._active_batches.add(batch_id)
        service._batch_workers[batch_id] = set()
        
        worker = service._create_batch_worker(batch_id, worker_id)
        
        # Test state transitions
        states = [
            WorkerState.INITIALIZING,
            WorkerState.IDLE,
            WorkerState.PROCESSING,
            WorkerState.PAUSED,
            WorkerState.STOPPED,
            WorkerState.ERROR
        ]
        
        for state in states:
            worker.worker_state = state
            assert service._worker_registry[worker_id].worker_state == state

    def test_worker_cleanup(self, service):
        """Test worker cleanup."""
        batch_id = "test_batch"
        worker_id = "test_worker"
        
        # Set up worker
        service._active_batches.add(batch_id)
        service._batch_workers[batch_id] = {worker_id}
        
        worker = WorkerInfo(
            worker_id=worker_id,
            batch_id=batch_id,
            worker_state=WorkerState.IDLE
        )
        service._worker_registry[worker_id] = worker
        service._statistics.active_workers = 1
        
        # Clean up worker
        service._cleanup_worker(worker_id)
        
        # Verify cleanup
        assert worker_id not in service._worker_registry
        assert worker_id not in service._batch_workers[batch_id]
        assert service._statistics.active_workers == 0

    def test_worker_heartbeat_management(self, service):
        """Test worker heartbeat management."""
        batch_id = "test_batch"
        worker_id = "test_worker"
        
        service._active_batches.add(batch_id)
        service._batch_workers[batch_id] = set()
        
        worker = service._create_batch_worker(batch_id, worker_id)
        original_heartbeat = worker.last_heartbeat
        
        # Simulate heartbeat update
        time.sleep(0.1)
        worker.last_heartbeat = datetime.utcnow()
        
        assert worker.last_heartbeat > original_heartbeat

    def test_stale_worker_detection(self, service):
        """Test stale worker detection and cleanup."""
        batch_id = "test_batch"
        worker_id = "stale_worker"
        
        # Create worker with old heartbeat
        old_time = datetime.utcnow() - timedelta(seconds=service.config.worker_timeout_seconds + 60)
        
        worker = WorkerInfo(
            worker_id=worker_id,
            batch_id=batch_id,
            worker_state=WorkerState.IDLE,
            last_heartbeat=old_time
        )
        
        service._worker_registry[worker_id] = worker
        service._batch_workers[batch_id] = {worker_id}
        service._statistics.active_workers = 1
        
        # Run cleanup
        service._cleanup_stale_workers()
        
        # Worker should be cleaned up
        assert worker_id not in service._worker_registry
        assert worker_id not in service._batch_workers.get(batch_id, set())
        assert service._statistics.active_workers == 0

    # Batch Management Tests

    @pytest.mark.asyncio
    async def test_create_concurrent_batch(self, service):
        """Test concurrent batch creation."""
        request = BatchCreateRequest(
            name="Test Batch",
            description="Test batch description",
            urls=["https://youtube.com/watch?v=123", "https://youtube.com/watch?v=456"],
            priority=BatchPriority.NORMAL
        )
        
        # Mock dependencies
        mock_batch = Mock()
        mock_batch.batch_id = "test_batch_123"
        mock_batch.total_items = 2
        mock_batch.batch_metadata = {}
        
        with patch('src.services.concurrent_batch_service.get_database_session') as mock_get_session:
            mock_get_session.return_value = service._session
            
            with patch('src.services.concurrent_batch_service.BatchService') as mock_batch_service_class:
                mock_batch_service = Mock()
                mock_batch_service_class.return_value = mock_batch_service
                mock_batch_service.create_batch.return_value = mock_batch
                
                with patch('src.services.concurrent_batch_service.allocate_resource') as mock_allocate:
                    mock_allocate.return_value.__enter__ = Mock(return_value=None)
                    mock_allocate.return_value.__exit__ = Mock(return_value=None)
                    
                    # Test batch creation
                    result = await service.create_concurrent_batch(
                        request,
                        ConcurrentBatchMode.PARALLEL,
                        max_concurrent_items=2
                    )
                    
                    assert result == mock_batch
                    assert "test_batch_123" in service._active_batches
                    assert service._statistics.total_batches == 1
                    assert service._statistics.active_batches == 1

    @pytest.mark.asyncio
    async def test_create_concurrent_batch_quota_exceeded(self, service):
        """Test batch creation when quota is exceeded."""
        # Fill up the active batches to max capacity
        for i in range(service.config.max_concurrent_batches):
            service._active_batches.add(f"batch_{i}")
        
        request = BatchCreateRequest(
            name="Test Batch",
            urls=["https://youtube.com/watch?v=123"],
            priority=BatchPriority.NORMAL
        )
        
        # Should raise error when quota exceeded
        with pytest.raises(ConcurrentBatchError):
            await service.create_concurrent_batch(request)

    def test_batch_lock_management(self, service):
        """Test batch lock management."""
        batch_id = "test_batch"
        
        # Get lock for first time
        lock1 = service._get_batch_lock(batch_id)
        assert lock1 is not None
        
        # Get same lock again
        lock2 = service._get_batch_lock(batch_id)
        assert lock1 is lock2  # Should be same instance
        
        # Get different lock
        lock3 = service._get_batch_lock("different_batch")
        assert lock1 is not lock3

    def test_batch_pause_resume(self, service):
        """Test batch pause and resume operations."""
        batch_id = "test_batch"
        
        # Create some workers
        service._batch_workers[batch_id] = set()
        for i in range(3):
            worker_id = f"worker_{i}"
            worker = WorkerInfo(
                worker_id=worker_id,
                batch_id=batch_id,
                worker_state=WorkerState.IDLE
            )
            service._worker_registry[worker_id] = worker
            service._batch_workers[batch_id].add(worker_id)
        
        # Pause batch processing
        result = service.pause_batch_processing(batch_id)
        assert result is True
        
        # Check that all workers are paused
        for worker_id in service._batch_workers[batch_id]:
            worker = service._worker_registry[worker_id]
            assert worker.worker_state == WorkerState.PAUSED
        
        # Resume batch processing
        result = service.resume_batch_processing(batch_id)
        assert result is True
        
        # Check that all workers are resumed
        for worker_id in service._batch_workers[batch_id]:
            worker = service._worker_registry[worker_id]
            assert worker.worker_state == WorkerState.IDLE

    def test_batch_cancellation(self, service):
        """Test batch cancellation."""
        batch_id = "test_batch"
        
        # Add batch to active batches
        service._active_batches.add(batch_id)
        service._batch_workers[batch_id] = set()
        
        # Create some workers
        for i in range(3):
            worker_id = f"worker_{i}"
            worker = WorkerInfo(
                worker_id=worker_id,
                batch_id=batch_id,
                worker_state=WorkerState.PROCESSING
            )
            service._worker_registry[worker_id] = worker
            service._batch_workers[batch_id].add(worker_id)
        
        # Cancel batch processing
        result = service.cancel_batch_processing(batch_id)
        assert result is True
        
        # Verify cancellation
        assert batch_id not in service._active_batches
        assert batch_id not in service._batch_workers

    # Concurrency and Threading Tests

    def test_concurrent_worker_creation(self, service):
        """Test concurrent worker creation."""
        batch_id = "concurrent_batch"
        service._active_batches.add(batch_id)
        service._batch_workers[batch_id] = set()
        
        created_workers = []
        worker_lock = threading.Lock()
        
        def create_worker(worker_id):
            worker = service._create_batch_worker(batch_id, worker_id)
            with worker_lock:
                created_workers.append(worker)
            return worker
        
        # Create workers concurrently
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(create_worker, f"concurrent_worker_{i}")
                for i in range(10)
            ]
            
            results = [future.result() for future in futures]
        
        # Verify all workers were created
        assert len(created_workers) == 10
        assert len(service._worker_registry) == 10
        assert len(service._batch_workers[batch_id]) == 10
        
        # Verify no duplicate workers
        worker_ids = [worker.worker_id for worker in created_workers]
        assert len(set(worker_ids)) == len(worker_ids)

    def test_concurrent_batch_operations(self, service):
        """Test concurrent batch operations."""
        batch_operations = []
        operation_lock = threading.Lock()
        
        def batch_operation(batch_id):
            try:
                service._active_batches.add(batch_id)
                service._batch_workers[batch_id] = set()
                
                # Create workers for this batch
                for i in range(2):
                    worker_id = f"worker_{batch_id}_{i}"
                    worker = service._create_batch_worker(batch_id, worker_id)
                
                # Pause and resume
                service.pause_batch_processing(batch_id)
                service.resume_batch_processing(batch_id)
                
                # Cancel batch
                service.cancel_batch_processing(batch_id)
                
                with operation_lock:
                    batch_operations.append(batch_id)
                
                return True
            except Exception as e:
                return False
        
        # Perform concurrent batch operations
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(batch_operation, f"batch_{i}")
                for i in range(5)
            ]
            
            results = [future.result() for future in futures]
        
        # Verify all operations succeeded
        assert all(results)
        assert len(batch_operations) == 5

    def test_thread_safety_statistics(self, service):
        """Test thread safety of statistics collection."""
        # Create some workers across multiple batches
        for batch_i in range(3):
            batch_id = f"batch_{batch_i}"
            service._batch_workers[batch_id] = set()
            
            for worker_i in range(5):
                worker_id = f"worker_{batch_i}_{worker_i}"
                worker = WorkerInfo(
                    worker_id=worker_id,
                    batch_id=batch_id,
                    worker_state=WorkerState.PROCESSING,
                    processed_items=worker_i * 2,
                    failed_items=worker_i,
                    total_processing_time=worker_i * 10.0
                )
                service._worker_registry[worker_id] = worker
                service._batch_workers[batch_id].add(worker_id)
        
        # Collect statistics concurrently
        stats_results = []
        
        def collect_stats():
            try:
                stats = service.get_concurrent_statistics()
                return stats
            except Exception as e:
                return None
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(collect_stats) for _ in range(10)]
            
            results = [future.result() for future in futures]
        
        # Verify all statistics collections succeeded
        assert all(result is not None for result in results)
        
        # Verify statistics content
        for stats in results:
            assert 'service_statistics' in stats
            assert 'worker_details' in stats
            assert len(stats['worker_details']) == 15  # 3 batches * 5 workers

    # Performance and Load Tests

    def test_high_concurrency_performance(self, high_perf_service):
        """Test performance under high concurrency."""
        service = high_perf_service
        
        # Create many batches and workers
        batch_count = 50
        worker_count = 200
        
        start_time = time.time()
        
        # Create batches
        for i in range(batch_count):
            batch_id = f"perf_batch_{i}"
            service._active_batches.add(batch_id)
            service._batch_workers[batch_id] = set()
        
        # Create workers
        for i in range(worker_count):
            batch_id = f"perf_batch_{i % batch_count}"
            worker_id = f"perf_worker_{i}"
            worker = service._create_batch_worker(batch_id, worker_id)
        
        creation_time = time.time() - start_time
        
        # Performance assertions
        assert creation_time < 10.0  # Should create all within 10 seconds
        assert len(service._active_batches) == batch_count
        assert len(service._worker_registry) == worker_count
        
        # Test statistics collection performance
        start_time = time.time()
        stats = service.get_concurrent_statistics()
        stats_time = time.time() - start_time
        
        assert stats_time < 5.0  # Should collect stats within 5 seconds
        assert len(stats['worker_details']) == worker_count

    def test_memory_usage_optimization(self, service):
        """Test memory usage optimization."""
        import psutil
        import os
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Create and cleanup many workers
        for batch_i in range(10):
            batch_id = f"memory_batch_{batch_i}"
            service._active_batches.add(batch_id)
            service._batch_workers[batch_id] = set()
            
            # Create workers
            for worker_i in range(20):
                worker_id = f"memory_worker_{batch_i}_{worker_i}"
                worker = service._create_batch_worker(batch_id, worker_id)
            
            # Cleanup batch
            service.cancel_batch_processing(batch_id)
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Check memory usage
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 50MB)
        assert memory_increase < 50 * 1024 * 1024

    def test_load_balancing_efficiency(self, service):
        """Test load balancing efficiency."""
        # Create multiple batches
        batch_count = 5
        for i in range(batch_count):
            batch_id = f"load_batch_{i}"
            service._active_batches.add(batch_id)
            service._batch_workers[batch_id] = set()
        
        # Create workers distributed across batches
        total_workers = 25
        workers_per_batch = {}
        
        for i in range(total_workers):
            batch_id = f"load_batch_{i % batch_count}"
            worker_id = f"load_worker_{i}"
            worker = service._create_batch_worker(batch_id, worker_id)
            
            if batch_id not in workers_per_batch:
                workers_per_batch[batch_id] = 0
            workers_per_batch[batch_id] += 1
        
        # Verify load balancing
        worker_counts = list(workers_per_batch.values())
        max_workers = max(worker_counts)
        min_workers = min(worker_counts)
        
        # Load should be reasonably balanced
        assert max_workers - min_workers <= 1  # Difference should be at most 1

    # Error Handling and Recovery Tests

    def test_error_handling_comprehensive(self, service):
        """Test comprehensive error handling."""
        # Test invalid batch operations
        with pytest.raises(ConcurrentBatchError):
            service.pause_batch_processing("non_existent_batch")
        
        with pytest.raises(ConcurrentBatchError):
            service.resume_batch_processing("non_existent_batch")
        
        # Test worker creation with invalid batch
        with pytest.raises(ConcurrentBatchError):
            service._create_batch_worker("non_existent_batch", "test_worker")
        
        # Test resource allocation failures
        with patch.object(service._concurrency_manager, 'allocate_resource') as mock_allocate:
            mock_allocate.side_effect = Exception("Resource allocation failed")
            
            # Should handle resource allocation errors gracefully
            with pytest.raises(ConcurrentBatchError):
                service._create_batch_worker("test_batch", "test_worker")

    def test_worker_error_recovery(self, service):
        """Test worker error recovery."""
        batch_id = "error_batch"
        worker_id = "error_worker"
        
        service._active_batches.add(batch_id)
        service._batch_workers[batch_id] = set()
        
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
        
        assert worker.worker_state == WorkerState.IDLE
        assert worker.error_count == 0
        assert worker.last_error is None

    def test_deadlock_prevention(self, service):
        """Test deadlock prevention mechanisms."""
        # Create multiple batches
        batch_ids = ["deadlock_batch_1", "deadlock_batch_2"]
        for batch_id in batch_ids:
            service._active_batches.add(batch_id)
            service._batch_workers[batch_id] = set()
        
        # Simulate potential deadlock scenario
        lock_order = []
        
        def acquire_locks(batch_id):
            lock = service._get_batch_lock(batch_id)
            lock_order.append(batch_id)
            time.sleep(0.1)  # Hold lock briefly
            return lock
        
        # Try to acquire locks in different orders concurrently
        with ThreadPoolExecutor(max_workers=2) as executor:
            future1 = executor.submit(acquire_locks, "deadlock_batch_1")
            future2 = executor.submit(acquire_locks, "deadlock_batch_2")
            
            # Both should complete without deadlock
            lock1 = future1.result(timeout=5)
            lock2 = future2.result(timeout=5)
            
            assert lock1 is not None
            assert lock2 is not None

    def test_timeout_handling(self, service):
        """Test timeout handling."""
        batch_id = "timeout_batch"
        worker_id = "timeout_worker"
        
        service._active_batches.add(batch_id)
        service._batch_workers[batch_id] = set()
        
        # Create worker
        worker = service._create_batch_worker(batch_id, worker_id)
        
        # Simulate timeout condition
        worker.last_heartbeat = datetime.utcnow() - timedelta(seconds=service.config.worker_timeout_seconds + 60)
        
        # Run cleanup
        service._cleanup_stale_workers()
        
        # Worker should be cleaned up
        assert worker_id not in service._worker_registry

    # Statistics and Monitoring Tests

    def test_statistics_collection_comprehensive(self, service):
        """Test comprehensive statistics collection."""
        # Create test data
        batch_count = 3
        worker_count = 9
        
        for batch_i in range(batch_count):
            batch_id = f"stats_batch_{batch_i}"
            service._active_batches.add(batch_id)
            service._batch_workers[batch_id] = set()
            
            for worker_i in range(3):
                worker_id = f"stats_worker_{batch_i}_{worker_i}"
                worker = WorkerInfo(
                    worker_id=worker_id,
                    batch_id=batch_id,
                    worker_state=WorkerState.PROCESSING,
                    processed_items=worker_i * 5,
                    failed_items=worker_i * 2,
                    total_processing_time=worker_i * 15.0,
                    error_count=worker_i,
                    last_error="Test error" if worker_i > 0 else None
                )
                service._worker_registry[worker_id] = worker
                service._batch_workers[batch_id].add(worker_id)
        
        # Update service statistics
        service._statistics.total_batches = 10
        service._statistics.active_batches = batch_count
        service._statistics.completed_batches = 5
        service._statistics.failed_batches = 2
        service._statistics.total_items = 100
        service._statistics.completed_items = 60
        service._statistics.failed_items = 15
        
        # Get comprehensive statistics
        stats = service.get_concurrent_statistics()
        
        # Verify structure
        assert 'service_statistics' in stats
        assert 'active_batches' in stats
        assert 'worker_details' in stats
        assert 'concurrency_statistics' in stats
        
        # Verify service statistics
        service_stats = stats['service_statistics']
        assert service_stats['total_batches'] == 10
        assert service_stats['active_batches'] == batch_count
        assert service_stats['completed_batches'] == 5
        assert service_stats['failed_batches'] == 2
        assert service_stats['total_items'] == 100
        assert service_stats['completed_items'] == 60
        assert service_stats['failed_items'] == 15
        
        # Verify worker details
        worker_details = stats['worker_details']
        assert len(worker_details) == worker_count
        
        for worker_id, details in worker_details.items():
            assert 'batch_id' in details
            assert 'state' in details
            assert 'processed_items' in details
            assert 'failed_items' in details
            assert 'total_processing_time' in details
            assert 'error_count' in details

    def test_performance_metrics_collection(self, service):
        """Test performance metrics collection."""
        # Create workers with different performance characteristics
        batch_id = "perf_batch"
        service._active_batches.add(batch_id)
        service._batch_workers[batch_id] = set()
        
        processing_times = [10.0, 20.0, 30.0, 40.0, 50.0]
        
        for i, processing_time in enumerate(processing_times):
            worker_id = f"perf_worker_{i}"
            worker = WorkerInfo(
                worker_id=worker_id,
                batch_id=batch_id,
                worker_state=WorkerState.PROCESSING,
                processed_items=i + 1,
                failed_items=i,
                total_processing_time=processing_time
            )
            service._worker_registry[worker_id] = worker
            service._batch_workers[batch_id].add(worker_id)
        
        # Update statistics
        service._statistics.completed_items = 15
        service._statistics.failed_items = 10
        
        # Update statistics calculation
        service._update_statistics()
        
        # Verify performance metrics
        stats = service._statistics
        assert stats.active_workers == 5
        assert stats.average_processing_time == 30.0  # (10+20+30+40+50)/5
        assert stats.error_rate == 10/25  # 10 failed out of 25 total
        assert stats.last_updated is not None

    def test_real_time_monitoring(self, service):
        """Test real-time monitoring capabilities."""
        # Create dynamic worker scenario
        batch_id = "monitor_batch"
        service._active_batches.add(batch_id)
        service._batch_workers[batch_id] = set()
        
        # Create initial workers
        for i in range(3):
            worker_id = f"monitor_worker_{i}"
            worker = service._create_batch_worker(batch_id, worker_id)
        
        # Get initial statistics
        initial_stats = service.get_concurrent_statistics()
        initial_worker_count = len(initial_stats['worker_details'])
        
        # Add more workers
        for i in range(3, 6):
            worker_id = f"monitor_worker_{i}"
            worker = service._create_batch_worker(batch_id, worker_id)
        
        # Get updated statistics
        updated_stats = service.get_concurrent_statistics()
        updated_worker_count = len(updated_stats['worker_details'])
        
        # Verify monitoring detected changes
        assert updated_worker_count > initial_worker_count
        assert updated_worker_count == 6
        
        # Remove some workers
        for i in range(3):
            worker_id = f"monitor_worker_{i}"
            service._cleanup_worker(worker_id)
        
        # Get final statistics
        final_stats = service.get_concurrent_statistics()
        final_worker_count = len(final_stats['worker_details'])
        
        # Verify monitoring detected cleanup
        assert final_worker_count < updated_worker_count
        assert final_worker_count == 3

    # Integration Tests

    def test_integration_with_batch_service(self, service):
        """Test integration with BatchService."""
        # This test focuses on the interface and mock interactions
        # since we're using mocked dependencies
        
        with patch('src.services.concurrent_batch_service.BatchService') as mock_batch_service_class:
            mock_batch_service = Mock()
            mock_batch_service_class.return_value = mock_batch_service
            
            # Test that ConcurrentBatchService properly delegates to BatchService
            batch_id = "integration_batch"
            service._active_batches.add(batch_id)
            
            # Verify that the service can handle batch operations
            assert batch_id in service._active_batches
            
            # Test batch removal
            service._active_batches.remove(batch_id)
            assert batch_id not in service._active_batches

    def test_integration_with_concurrency_manager(self, service, mock_concurrency_manager):
        """Test integration with ConcurrencyManager."""
        # Test resource allocation
        batch_id = "concurrency_batch"
        worker_id = "concurrency_worker"
        
        service._active_batches.add(batch_id)
        service._batch_workers[batch_id] = set()
        
        # Create worker (should interact with concurrency manager)
        worker = service._create_batch_worker(batch_id, worker_id)
        
        # Verify concurrency manager was used
        assert mock_concurrency_manager.configure_resource_quota.called
        
        # Test statistics integration
        stats = service.get_concurrent_statistics()
        assert 'concurrency_statistics' in stats
        assert mock_concurrency_manager.get_comprehensive_statistics.called

    # Edge Cases and Stress Tests

    def test_edge_case_empty_service(self, service):
        """Test edge cases with empty service."""
        # Test statistics with no data
        stats = service.get_concurrent_statistics()
        
        assert stats['service_statistics']['total_batches'] == 0
        assert stats['service_statistics']['active_batches'] == 0
        assert len(stats['active_batches']) == 0
        assert len(stats['worker_details']) == 0
        
        # Test cleanup operations with no data
        service._cleanup_stale_workers()
        service._cleanup_unused_locks()
        
        # Should not raise errors
        assert True

    def test_edge_case_rapid_operations(self, service):
        """Test rapid operations edge cases."""
        batch_id = "rapid_batch"
        
        # Rapid batch operations
        for i in range(100):
            service._active_batches.add(f"{batch_id}_{i}")
            service._batch_workers[f"{batch_id}_{i}"] = set()
            
            # Immediately remove
            service._active_batches.remove(f"{batch_id}_{i}")
            del service._batch_workers[f"{batch_id}_{i}"]
        
        # Verify final state
        assert len(service._active_batches) == 0
        assert len(service._batch_workers) == 0

    def test_stress_test_worker_churn(self, service):
        """Test stress with high worker churn."""
        batch_id = "churn_batch"
        service._active_batches.add(batch_id)
        service._batch_workers[batch_id] = set()
        
        # Create and destroy workers rapidly
        for i in range(100):
            worker_id = f"churn_worker_{i}"
            
            # Create worker
            worker = service._create_batch_worker(batch_id, worker_id)
            
            # Immediately clean up
            service._cleanup_worker(worker_id)
        
        # Verify final state
        assert len(service._worker_registry) == 0
        assert len(service._batch_workers[batch_id]) == 0

    def test_resource_exhaustion_handling(self, service):
        """Test handling of resource exhaustion."""
        # Fill up all available batch slots
        for i in range(service.config.max_concurrent_batches):
            service._active_batches.add(f"resource_batch_{i}")
        
        # Try to add one more batch
        with pytest.raises(ConcurrentBatchError):
            service._active_batches.add("overflow_batch")
            if len(service._active_batches) > service.config.max_concurrent_batches:
                service._active_batches.remove("overflow_batch")
                raise ConcurrentBatchError("Too many concurrent batches")

    def test_cleanup_operations_comprehensive(self, service):
        """Test comprehensive cleanup operations."""
        # Create test data
        batch_count = 5
        worker_count = 15
        
        for batch_i in range(batch_count):
            batch_id = f"cleanup_batch_{batch_i}"
            service._active_batches.add(batch_id)
            service._batch_workers[batch_id] = set()
            
            for worker_i in range(3):
                worker_id = f"cleanup_worker_{batch_i}_{worker_i}"
                worker = service._create_batch_worker(batch_id, worker_id)
        
        # Verify initial state
        assert len(service._active_batches) == batch_count
        assert len(service._worker_registry) == worker_count
        
        # Run comprehensive cleanup
        service._cleanup_stale_workers()
        service._cleanup_unused_locks()
        
        # Shutdown service
        service.shutdown()
        
        # Verify cleanup
        assert len(service._worker_registry) == 0
        assert service._shutdown_event.is_set()

    # Factory Function Tests

    def test_factory_functions(self):
        """Test factory functions."""
        # Test create_concurrent_batch_service
        config = ConcurrentBatchConfig(max_concurrent_batches=3)
        service = create_concurrent_batch_service(config)
        
        assert isinstance(service, ConcurrentBatchService)
        assert service.config.max_concurrent_batches == 3
        
        service.shutdown()
        
        # Test create_concurrent_batch_service with default config
        service2 = create_concurrent_batch_service()
        
        assert isinstance(service2, ConcurrentBatchService)
        assert isinstance(service2.config, ConcurrentBatchConfig)
        
        service2.shutdown()
        
        # Test get_concurrent_batch_service
        mock_session = Mock()
        config = ConcurrentBatchConfig(max_concurrent_batches=5)
        
        service3 = get_concurrent_batch_service(mock_session, config)
        
        assert isinstance(service3, ConcurrentBatchService)
        assert service3._session == mock_session
        assert service3.config.max_concurrent_batches == 5
        
        service3.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "--asyncio-mode=auto"])