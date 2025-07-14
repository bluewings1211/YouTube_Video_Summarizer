"""
Stress tests and load tests for batch processing operations.

This test suite provides comprehensive stress and load testing including:
- High-volume batch processing stress tests
- System resource exhaustion testing
- Extreme concurrency load tests
- Memory pressure stress testing
- Database connection pool stress tests
- Error cascade and recovery testing
- System stability under extreme conditions
- Performance degradation analysis
- Breaking point identification
"""

import pytest
import time
import threading
import asyncio
import psutil
import os
import random
import gc
from unittest.mock import Mock, patch
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from contextlib import contextmanager
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from multiprocessing import Process, Queue as MPQueue, Event

from src.database.models import Base, Video
from src.database.batch_models import (
    Batch, BatchItem, QueueItem, ProcessingSession,
    BatchStatus, BatchItemStatus, BatchPriority
)
from src.services.batch_service import (
    BatchService, BatchCreateRequest, BatchItemResult
)
from src.services.queue_service import (
    QueueService, QueueProcessingOptions
)
from src.services.concurrent_batch_service import (
    ConcurrentBatchService, ConcurrentBatchConfig
)


class SystemResourceMonitor:
    """Monitor system resources during stress tests."""
    
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.monitoring = False
        self.samples = []
        self.monitor_thread = None
        
    def start_monitoring(self, interval=1.0):
        """Start resource monitoring."""
        self.monitoring = True
        self.samples = []
        
        def monitor():
            while self.monitoring:
                try:
                    sample = {
                        'timestamp': time.time(),
                        'memory_rss': self.process.memory_info().rss,
                        'memory_vms': self.process.memory_info().vms,
                        'cpu_percent': self.process.cpu_percent(),
                        'num_threads': self.process.num_threads(),
                        'open_files': len(self.process.open_files()),
                        'connections': len(self.process.connections())
                    }
                    self.samples.append(sample)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    break
                time.sleep(interval)
        
        self.monitor_thread = threading.Thread(target=monitor, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop resource monitoring and return analysis."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        
        if not self.samples:
            return {}
        
        # Analyze samples
        memory_values = [s['memory_rss'] for s in self.samples]
        cpu_values = [s['cpu_percent'] for s in self.samples]
        thread_values = [s['num_threads'] for s in self.samples]
        
        return {
            'duration': self.samples[-1]['timestamp'] - self.samples[0]['timestamp'],
            'memory_peak': max(memory_values),
            'memory_average': sum(memory_values) / len(memory_values),
            'memory_growth': memory_values[-1] - memory_values[0],
            'cpu_peak': max(cpu_values),
            'cpu_average': sum(cpu_values) / len(cpu_values),
            'threads_peak': max(thread_values),
            'threads_average': sum(thread_values) / len(thread_values),
            'sample_count': len(self.samples)
        }


@contextmanager
def resource_monitoring():
    """Context manager for resource monitoring."""
    monitor = SystemResourceMonitor()
    monitor.start_monitoring()
    try:
        yield monitor
    finally:
        analysis = monitor.stop_monitoring()
        return analysis


class TestBatchStressLoad:
    """Stress and load tests for batch processing operations."""

    @pytest.fixture(scope="function")
    def db_engine(self):
        """Create database engine with stress test optimizations."""
        engine = create_engine(
            "sqlite:///:memory:",
            connect_args={
                "check_same_thread": False,
                "timeout": 60,
                "cache_size": -64000,  # 64MB cache
                "synchronous": "NORMAL"
            },
            poolclass=StaticPool,
            echo=False,
            pool_pre_ping=True,
            pool_recycle=3600
        )
        Base.metadata.create_all(engine)
        return engine

    @pytest.fixture(scope="function")
    def db_session(self, db_engine):
        """Create database session for stress testing."""
        TestingSessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=db_engine,
            expire_on_commit=False
        )
        session = TestingSessionLocal()
        yield session
        session.close()

    @pytest.fixture(scope="function")
    def batch_service(self, db_session):
        """Create BatchService instance."""
        return BatchService(session=db_session)

    @pytest.fixture(scope="function")
    def queue_service(self, db_session):
        """Create QueueService instance for stress testing."""
        options = QueueProcessingOptions(
            max_workers=50,
            worker_timeout_minutes=10,
            lock_timeout_minutes=5,
            heartbeat_interval_seconds=30,
            stale_lock_cleanup_interval_minutes=2,
            enable_automatic_cleanup=False
        )
        service = QueueService(db_session, options)
        yield service
        service.shutdown()

    @pytest.fixture
    def massive_url_list(self):
        """Massive list of URLs for stress testing."""
        return [f"https://www.youtube.com/watch?v=stress{i:07d}" for i in range(10000)]

    @pytest.fixture
    def extreme_url_list(self):
        """Extreme list of URLs for breaking point tests."""
        return [f"https://www.youtube.com/watch?v=extreme{i:08d}" for i in range(50000)]

    # High-Volume Stress Tests

    def test_massive_batch_creation_stress(self, batch_service, massive_url_list):
        """Stress test with massive batch creation."""
        with resource_monitoring() as monitor:
            with patch('src.services.batch_service.extract_video_id_from_url') as mock_extract:
                mock_extract.side_effect = lambda url: url.split('=')[-1]
                
                # Create extremely large batch
                request = BatchCreateRequest(
                    name="Massive Stress Test",
                    urls=massive_url_list,
                    priority=BatchPriority.NORMAL,
                    batch_metadata={"stress_test": True, "size": "massive"}
                )
                
                start_time = time.time()
                batch = batch_service.create_batch(request)
                creation_time = time.time() - start_time
                
                # Verify creation succeeded
                assert batch is not None
                assert batch.total_items == 10000
                assert batch.status == BatchStatus.PENDING
                
                # Performance under stress
                assert creation_time < 300.0  # Should complete within 5 minutes
                
        analysis = monitor.stop_monitoring()
        
        # Resource usage analysis
        assert analysis['memory_peak'] < 1024 * 1024 * 1024  # Less than 1GB
        print(f"Massive batch creation: {creation_time:.3f}s, "
              f"Memory peak: {analysis['memory_peak']//1024//1024}MB")

    def test_multiple_large_batches_stress(self, batch_service, massive_url_list):
        """Stress test with multiple large batches."""
        batch_count = 20
        items_per_batch = 500
        
        with resource_monitoring() as monitor:
            with patch('src.services.batch_service.extract_video_id_from_url') as mock_extract:
                mock_extract.side_effect = lambda url: url.split('=')[-1]
                
                batches = []
                creation_times = []
                
                for i in range(batch_count):
                    urls = massive_url_list[i*items_per_batch:(i+1)*items_per_batch]
                    request = BatchCreateRequest(
                        name=f"Multi-Batch Stress {i}",
                        urls=urls,
                        priority=random.choice(list(BatchPriority))
                    )
                    
                    start_time = time.time()
                    batch = batch_service.create_batch(request)
                    creation_time = time.time() - start_time
                    
                    batches.append(batch)
                    creation_times.append(creation_time)
                    
                    # Verify batch creation
                    assert batch.total_items == items_per_batch
                    
                    # Check for performance degradation
                    if i > 0:
                        # Creation time shouldn't increase dramatically
                        assert creation_time < creation_times[0] * 3
                
                # Verify all batches
                assert len(batches) == batch_count
                total_items = sum(b.total_items for b in batches)
                assert total_items == batch_count * items_per_batch
        
        analysis = monitor.stop_monitoring()
        
        avg_creation_time = sum(creation_times) / len(creation_times)
        print(f"Multiple large batches: {avg_creation_time:.3f}s avg creation, "
              f"Memory growth: {analysis['memory_growth']//1024//1024}MB")

    def test_extreme_concurrency_stress(self, batch_service, queue_service, massive_url_list):
        """Stress test with extreme concurrency."""
        # Create large batch
        request = BatchCreateRequest(
            name="Extreme Concurrency Stress",
            urls=massive_url_list[:5000],
            priority=BatchPriority.NORMAL
        )
        
        with patch('src.services.batch_service.extract_video_id_from_url') as mock_extract:
            mock_extract.side_effect = lambda url: url.split('=')[-1]
            
            batch = batch_service.create_batch(request)
            batch_service.start_batch_processing(batch.batch_id)
            
            # Create extreme number of workers
            num_workers = 100
            workers = []
            
            with resource_monitoring() as monitor:
                # Register workers
                for i in range(num_workers):
                    worker = queue_service.register_worker("video_processing", f"extreme_worker_{i}")
                    workers.append(worker)
                
                # Process items with extreme concurrency
                def worker_function(worker_id):
                    processed = 0
                    errors = 0
                    
                    try:
                        for _ in range(50):  # Each worker processes up to 50 items
                            item = queue_service.get_next_queue_item("video_processing", worker_id)
                            if item:
                                try:
                                    queue_service.complete_queue_item(
                                        item.id,
                                        worker_id,
                                        BatchItemStatus.COMPLETED
                                    )
                                    processed += 1
                                except Exception as e:
                                    errors += 1
                            else:
                                break
                    except Exception as e:
                        errors += 1
                    
                    return processed, errors
                
                # Run with extreme concurrency
                start_time = time.time()
                
                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    futures = [
                        executor.submit(worker_function, worker.worker_id)
                        for worker in workers
                    ]
                    
                    results = []
                    for future in as_completed(futures):
                        try:
                            result = future.result(timeout=30)  # 30 second timeout per worker
                            results.append(result)
                        except Exception as e:
                            results.append((0, 1))  # Count as error
                
                processing_time = time.time() - start_time
        
        analysis = monitor.stop_monitoring()
        
        # Analyze results
        total_processed = sum(r[0] for r in results)
        total_errors = sum(r[1] for r in results)
        error_rate = total_errors / (total_processed + total_errors) if (total_processed + total_errors) > 0 else 1.0
        
        # Stress test assertions
        assert total_processed > 1000  # Should process a significant number
        assert error_rate < 0.5  # Error rate should be reasonable under stress
        assert processing_time < 300.0  # Should complete within 5 minutes
        
        print(f"Extreme concurrency: {total_processed} items, {error_rate:.2%} error rate, "
              f"Threads peak: {analysis['threads_peak']}")

    # Memory Pressure Stress Tests

    def test_memory_pressure_stress(self, batch_service, extreme_url_list):
        """Stress test under memory pressure conditions."""
        # Create memory pressure by processing very large batches
        batch_size = 25000
        
        with patch('src.services.batch_service.extract_video_id_from_url') as mock_extract:
            mock_extract.side_effect = lambda url: url.split('=')[-1]
            
            with resource_monitoring() as monitor:
                # Create batch under memory pressure
                request = BatchCreateRequest(
                    name="Memory Pressure Stress",
                    urls=extreme_url_list[:batch_size],
                    priority=BatchPriority.NORMAL
                )
                
                # Force garbage collection before test
                gc.collect()
                initial_memory = psutil.Process(os.getpid()).memory_info().rss
                
                batch = batch_service.create_batch(request)
                
                # Verify batch creation under pressure
                assert batch.total_items == batch_size
                
                # Perform memory-intensive operations
                for i in range(10):
                    stats = batch_service.get_batch_statistics()
                    progress = batch_service.get_batch_progress(batch.batch_id)
                    batches = batch_service.list_batches()
                    
                    # Force memory cleanup periodically
                    if i % 3 == 0:
                        gc.collect()
        
        analysis = monitor.stop_monitoring()
        
        # Memory pressure analysis
        final_memory = psutil.Process(os.getpid()).memory_info().rss
        memory_growth = final_memory - initial_memory
        
        # Should handle memory pressure gracefully
        assert memory_growth < 2 * 1024 * 1024 * 1024  # Less than 2GB growth
        assert analysis['memory_peak'] < 3 * 1024 * 1024 * 1024  # Less than 3GB peak
        
        print(f"Memory pressure: {memory_growth//1024//1024}MB growth, "
              f"Peak: {analysis['memory_peak']//1024//1024}MB")

    def test_memory_leak_stress(self, batch_service, massive_url_list):
        """Stress test for memory leaks under continuous operation."""
        batch_count = 50
        items_per_batch = 200
        
        memory_samples = []
        
        with patch('src.services.batch_service.extract_video_id_from_url') as mock_extract:
            mock_extract.side_effect = lambda url: url.split('=')[-1]
            
            for i in range(batch_count):
                # Sample memory before batch creation
                gc.collect()
                memory_before = psutil.Process(os.getpid()).memory_info().rss
                
                # Create batch
                urls = massive_url_list[i*items_per_batch:(i+1)*items_per_batch]
                request = BatchCreateRequest(
                    name=f"Memory Leak Test {i}",
                    urls=urls,
                    priority=BatchPriority.NORMAL
                )
                
                batch = batch_service.create_batch(request)
                
                # Perform operations that might leak memory
                batch_service.get_batch(batch.batch_id)
                batch_service.get_batch_progress(batch.batch_id)
                batch_service.list_batches()
                batch_service.get_batch_statistics()
                
                # Sample memory after operations
                memory_after = psutil.Process(os.getpid()).memory_info().rss
                memory_samples.append({
                    'iteration': i,
                    'memory_before': memory_before,
                    'memory_after': memory_after,
                    'memory_delta': memory_after - memory_before
                })
                
                # Force cleanup every 10 iterations
                if i % 10 == 0:
                    gc.collect()
        
        # Analyze memory leak patterns
        total_growth = memory_samples[-1]['memory_after'] - memory_samples[0]['memory_before']
        avg_growth_per_batch = total_growth / batch_count
        
        # Check for memory leak trends
        recent_samples = memory_samples[-10:]  # Last 10 samples
        early_samples = memory_samples[:10]    # First 10 samples
        
        recent_avg_delta = sum(s['memory_delta'] for s in recent_samples) / len(recent_samples)
        early_avg_delta = sum(s['memory_delta'] for s in early_samples) / len(early_samples)
        
        # Memory leak assertions
        assert avg_growth_per_batch < 10 * 1024 * 1024  # Less than 10MB per batch
        assert total_growth < 500 * 1024 * 1024  # Less than 500MB total growth
        
        # Growth should not accelerate (indicating leak)
        assert recent_avg_delta < early_avg_delta * 2
        
        print(f"Memory leak stress: {total_growth//1024//1024}MB total growth, "
              f"{avg_growth_per_batch//1024}KB per batch")

    # Database Stress Tests

    def test_database_connection_stress(self, db_engine, massive_url_list):
        """Stress test database connections under heavy load."""
        connection_count = 50
        operations_per_connection = 100
        
        def database_worker(worker_id, results_queue):
            """Worker function for database stress testing."""
            try:
                # Create separate session for each worker
                TestingSessionLocal = sessionmaker(bind=db_engine)
                session = TestingSessionLocal()
                batch_service = BatchService(session=session)
                
                operations_completed = 0
                errors = 0
                
                try:
                    for i in range(operations_per_connection):
                        try:
                            # Perform various database operations
                            if i % 4 == 0:
                                # Create batch
                                urls = massive_url_list[i*10:(i+1)*10]
                                request = BatchCreateRequest(
                                    name=f"DB Stress {worker_id}_{i}",
                                    urls=urls,
                                    priority=BatchPriority.NORMAL
                                )
                                with patch('src.services.batch_service.extract_video_id_from_url') as mock_extract:
                                    mock_extract.side_effect = lambda url: url.split('=')[-1]
                                    batch = batch_service.create_batch(request)
                            
                            elif i % 4 == 1:
                                # Get statistics
                                stats = batch_service.get_batch_statistics()
                            
                            elif i % 4 == 2:
                                # List batches
                                batches = batch_service.list_batches()
                            
                            else:
                                # Query operations
                                session.execute(text("SELECT COUNT(*) FROM batches"))
                                session.execute(text("SELECT COUNT(*) FROM batch_items"))
                            
                            operations_completed += 1
                            
                        except Exception as e:
                            errors += 1
                
                finally:
                    session.close()
                
                results_queue.put({
                    'worker_id': worker_id,
                    'operations_completed': operations_completed,
                    'errors': errors
                })
                
            except Exception as e:
                results_queue.put({
                    'worker_id': worker_id,
                    'operations_completed': 0,
                    'errors': operations_per_connection
                })
        
        # Run database stress test
        with resource_monitoring() as monitor:
            results_queue = MPQueue()
            processes = []
            
            start_time = time.time()
            
            # Start worker processes
            for i in range(connection_count):
                process = Process(
                    target=database_worker,
                    args=(i, results_queue)
                )
                processes.append(process)
                process.start()
            
            # Wait for all processes to complete
            for process in processes:
                process.join(timeout=300)  # 5 minute timeout
                if process.is_alive():
                    process.terminate()
            
            processing_time = time.time() - start_time
            
            # Collect results
            results = []
            while not results_queue.empty():
                results.append(results_queue.get())
        
        analysis = monitor.stop_monitoring()
        
        # Analyze database stress results
        total_operations = sum(r['operations_completed'] for r in results)
        total_errors = sum(r['errors'] for r in results)
        error_rate = total_errors / (total_operations + total_errors) if (total_operations + total_errors) > 0 else 1.0
        
        # Database stress assertions
        assert len(results) >= connection_count * 0.8  # At least 80% of workers completed
        assert error_rate < 0.3  # Error rate should be reasonable
        assert processing_time < 600.0  # Should complete within 10 minutes
        
        print(f"Database stress: {total_operations} operations, {error_rate:.2%} error rate, "
              f"{processing_time:.1f}s")

    def test_database_deadlock_stress(self, batch_service, queue_service, massive_url_list):
        """Stress test for database deadlock scenarios."""
        # Create scenario prone to deadlocks
        batch_count = 10
        worker_count = 20
        
        with patch('src.services.batch_service.extract_video_id_from_url') as mock_extract:
            mock_extract.side_effect = lambda url: url.split('=')[-1]
            
            # Create multiple batches
            batches = []
            for i in range(batch_count):
                urls = massive_url_list[i*100:(i+1)*100]
                request = BatchCreateRequest(
                    name=f"Deadlock Stress {i}",
                    urls=urls,
                    priority=BatchPriority.NORMAL
                )
                batch = batch_service.create_batch(request)
                batch_service.start_batch_processing(batch.batch_id)
                batches.append(batch)
            
            # Create workers that might cause deadlocks
            workers = []
            for i in range(worker_count):
                worker = queue_service.register_worker("video_processing", f"deadlock_worker_{i}")
                workers.append(worker)
            
            def deadlock_prone_worker(worker_id):
                """Worker that performs operations prone to deadlocks."""
                operations = 0
                deadlocks = 0
                
                try:
                    for _ in range(50):
                        try:
                            # Mix of operations that might cause deadlocks
                            if random.random() < 0.7:
                                # Process queue item
                                item = queue_service.get_next_queue_item("video_processing", worker_id)
                                if item:
                                    queue_service.complete_queue_item(
                                        item.id,
                                        worker_id,
                                        random.choice([BatchItemStatus.COMPLETED, BatchItemStatus.FAILED])
                                    )
                                    operations += 1
                            else:
                                # Statistics operations
                                stats = batch_service.get_batch_statistics()
                                operations += 1
                                
                        except Exception as e:
                            if "deadlock" in str(e).lower() or "locked" in str(e).lower():
                                deadlocks += 1
                            else:
                                # Other errors
                                pass
                
                except Exception as e:
                    pass
                
                return operations, deadlocks
            
            # Run deadlock stress test
            start_time = time.time()
            
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = [
                    executor.submit(deadlock_prone_worker, worker.worker_id)
                    for worker in workers
                ]
                
                results = []
                for future in as_completed(futures):
                    try:
                        result = future.result(timeout=60)
                        results.append(result)
                    except Exception as e:
                        results.append((0, 1))
            
            stress_time = time.time() - start_time
        
        # Analyze deadlock stress results
        total_operations = sum(r[0] for r in results)
        total_deadlocks = sum(r[1] for r in results)
        deadlock_rate = total_deadlocks / (total_operations + total_deadlocks) if (total_operations + total_deadlocks) > 0 else 0
        
        # Deadlock stress assertions
        assert total_operations > 200  # Should complete substantial operations
        assert deadlock_rate < 0.1  # Deadlock rate should be low
        assert stress_time < 300.0  # Should complete within 5 minutes
        
        print(f"Deadlock stress: {total_operations} operations, {deadlock_rate:.2%} deadlock rate")

    # System Stability Stress Tests

    def test_continuous_operation_stress(self, batch_service, queue_service, massive_url_list):
        """Stress test for continuous operation stability."""
        duration_minutes = 5  # 5 minute continuous operation
        end_time = time.time() + (duration_minutes * 60)
        
        operations_log = []
        error_log = []
        
        with patch('src.services.batch_service.extract_video_id_from_url') as mock_extract:
            mock_extract.side_effect = lambda url: url.split('=')[-1]
            
            # Register workers
            workers = []
            for i in range(10):
                worker = queue_service.register_worker("video_processing", f"continuous_worker_{i}")
                workers.append(worker)
            
            operation_counter = 0
            
            with resource_monitoring() as monitor:
                while time.time() < end_time:
                    try:
                        operation_start = time.time()
                        
                        # Rotate between different operations
                        operation_type = operation_counter % 5
                        
                        if operation_type == 0:
                            # Create batch
                            urls = massive_url_list[operation_counter*10:(operation_counter+1)*10]
                            request = BatchCreateRequest(
                                name=f"Continuous Batch {operation_counter}",
                                urls=urls,
                                priority=random.choice(list(BatchPriority))
                            )
                            batch = batch_service.create_batch(request)
                            batch_service.start_batch_processing(batch.batch_id)
                        
                        elif operation_type == 1:
                            # Process queue items
                            for worker in workers[:3]:  # Use subset of workers
                                item = queue_service.get_next_queue_item("video_processing", worker.worker_id)
                                if item:
                                    queue_service.complete_queue_item(
                                        item.id,
                                        worker.worker_id,
                                        BatchItemStatus.COMPLETED
                                    )
                        
                        elif operation_type == 2:
                            # Statistics operations
                            batch_stats = batch_service.get_batch_statistics()
                            queue_stats = queue_service.get_queue_statistics("video_processing")
                        
                        elif operation_type == 3:
                            # List and query operations
                            batches = batch_service.list_batches()
                            worker_stats = queue_service.get_worker_statistics()
                        
                        else:
                            # Cleanup operations
                            queue_service._cleanup_stale_locks()
                            batch_service.cleanup_stale_sessions(timeout_minutes=1)
                        
                        operation_duration = time.time() - operation_start
                        operations_log.append({
                            'operation': operation_type,
                            'duration': operation_duration,
                            'timestamp': time.time()
                        })
                        
                        operation_counter += 1
                        
                        # Brief pause to prevent overwhelming
                        time.sleep(0.1)
                    
                    except Exception as e:
                        error_log.append({
                            'error': str(e),
                            'operation': operation_type,
                            'timestamp': time.time()
                        })
        
        analysis = monitor.stop_monitoring()
        
        # Analyze continuous operation results
        total_operations = len(operations_log)
        total_errors = len(error_log)
        error_rate = total_errors / (total_operations + total_errors) if (total_operations + total_errors) > 0 else 1.0
        
        avg_operation_time = sum(op['duration'] for op in operations_log) / len(operations_log) if operations_log else 0
        
        # Stability assertions
        assert total_operations > 100  # Should complete many operations
        assert error_rate < 0.2  # Error rate should be acceptable
        assert avg_operation_time < 5.0  # Operations should remain fast
        
        # Check for performance degradation over time
        if len(operations_log) > 20:
            early_ops = operations_log[:10]
            late_ops = operations_log[-10:]
            
            early_avg = sum(op['duration'] for op in early_ops) / len(early_ops)
            late_avg = sum(op['duration'] for op in late_ops) / len(late_ops)
            
            # Performance shouldn't degrade significantly
            assert late_avg < early_avg * 3
        
        print(f"Continuous operation: {total_operations} ops in {duration_minutes}min, "
              f"{error_rate:.2%} error rate, {avg_operation_time:.3f}s avg time")

    def test_resource_exhaustion_recovery(self, batch_service, queue_service, extreme_url_list):
        """Test system recovery from resource exhaustion."""
        # Deliberately exhaust resources then test recovery
        
        with patch('src.services.batch_service.extract_video_id_from_url') as mock_extract:
            mock_extract.side_effect = lambda url: url.split('=')[-1]
            
            # Phase 1: Resource exhaustion
            print("Phase 1: Exhausting resources...")
            
            exhaustion_batches = []
            try:
                # Create many large batches to exhaust resources
                for i in range(50):
                    urls = extreme_url_list[i*1000:(i+1)*1000]
                    request = BatchCreateRequest(
                        name=f"Exhaustion Batch {i}",
                        urls=urls,
                        priority=BatchPriority.NORMAL
                    )
                    
                    batch = batch_service.create_batch(request)
                    exhaustion_batches.append(batch)
                    
                    # Stop if we hit resource limits
                    if i > 10 and i % 5 == 0:
                        try:
                            stats = batch_service.get_batch_statistics()
                            if stats["total_batch_items"] > 30000:
                                break
                        except:
                            break
            
            except Exception as e:
                # Expected to hit resource limits
                print(f"Resource exhaustion achieved: {e}")
            
            # Phase 2: Recovery testing
            print("Phase 2: Testing recovery...")
            
            # Force cleanup
            gc.collect()
            time.sleep(2)
            
            recovery_operations = 0
            recovery_errors = 0
            
            # Test system recovery
            for i in range(20):
                try:
                    # Try basic operations during recovery
                    stats = batch_service.get_batch_statistics()
                    batches = batch_service.list_batches()
                    
                    # Try creating small batch
                    if i % 5 == 0:
                        small_urls = extreme_url_list[:5]
                        recovery_request = BatchCreateRequest(
                            name=f"Recovery Test {i}",
                            urls=small_urls,
                            priority=BatchPriority.LOW
                        )
                        recovery_batch = batch_service.create_batch(recovery_request)
                    
                    recovery_operations += 1
                    
                except Exception as e:
                    recovery_errors += 1
                    time.sleep(1)  # Wait before retry
            
            # Phase 3: Normal operation test
            print("Phase 3: Testing normal operation...")
            
            normal_operations = 0
            normal_errors = 0
            
            # Test that system returns to normal operation
            for i in range(10):
                try:
                    urls = extreme_url_list[:50]  # Small batches
                    normal_request = BatchCreateRequest(
                        name=f"Normal Operation {i}",
                        urls=urls,
                        priority=BatchPriority.NORMAL
                    )
                    
                    normal_batch = batch_service.create_batch(normal_request)
                    normal_operations += 1
                    
                except Exception as e:
                    normal_errors += 1
        
        # Recovery assertions
        recovery_rate = recovery_operations / (recovery_operations + recovery_errors) if (recovery_operations + recovery_errors) > 0 else 0
        normal_rate = normal_operations / (normal_operations + normal_errors) if (normal_operations + normal_errors) > 0 else 0
        
        assert recovery_rate > 0.5  # Should recover from at least 50% of operations
        assert normal_rate > 0.8   # Should return to normal operation
        
        print(f"Resource recovery: {recovery_rate:.1%} recovery rate, {normal_rate:.1%} normal rate")

    # Breaking Point Tests

    def test_find_breaking_point_batch_size(self, batch_service):
        """Find the breaking point for batch size."""
        sizes_to_test = [1000, 5000, 10000, 25000, 50000, 100000]
        breaking_point = None
        
        with patch('src.services.batch_service.extract_video_id_from_url') as mock_extract:
            mock_extract.side_effect = lambda url: url.split('=')[-1]
            
            for size in sizes_to_test:
                try:
                    print(f"Testing batch size: {size}")
                    
                    urls = [f"https://www.youtube.com/watch?v=break{i:08d}" for i in range(size)]
                    request = BatchCreateRequest(
                        name=f"Breaking Point Test {size}",
                        urls=urls,
                        priority=BatchPriority.NORMAL
                    )
                    
                    start_time = time.time()
                    batch = batch_service.create_batch(request)
                    creation_time = time.time() - start_time
                    
                    # Check if performance is still acceptable
                    if creation_time > 600:  # 10 minutes
                        breaking_point = size
                        print(f"Breaking point found at size {size} (took {creation_time:.1f}s)")
                        break
                    
                    print(f"Size {size}: {creation_time:.3f}s - OK")
                    
                except Exception as e:
                    breaking_point = size
                    print(f"Breaking point found at size {size}: {e}")
                    break
        
        # Document breaking point
        if breaking_point:
            print(f"System breaking point: {breaking_point} items per batch")
            assert breaking_point > 1000  # Should handle at least 1000 items
        else:
            print("No breaking point found within tested range")

    def test_find_breaking_point_concurrency(self, batch_service, queue_service, massive_url_list):
        """Find the breaking point for concurrency."""
        worker_counts = [10, 25, 50, 100, 200, 500]
        breaking_point = None
        
        # Create test batch
        with patch('src.services.batch_service.extract_video_id_from_url') as mock_extract:
            mock_extract.side_effect = lambda url: url.split('=')[-1]
            
            request = BatchCreateRequest(
                name="Concurrency Breaking Point Test",
                urls=massive_url_list[:1000],
                priority=BatchPriority.NORMAL
            )
            
            batch = batch_service.create_batch(request)
            batch_service.start_batch_processing(batch.batch_id)
            
            for worker_count in worker_counts:
                try:
                    print(f"Testing concurrency: {worker_count} workers")
                    
                    # Register workers
                    workers = []
                    for i in range(worker_count):
                        worker = queue_service.register_worker("video_processing", f"break_worker_{i}")
                        workers.append(worker)
                    
                    # Test concurrent processing
                    def worker_function(worker_id):
                        try:
                            item = queue_service.get_next_queue_item("video_processing", worker_id)
                            if item:
                                queue_service.complete_queue_item(
                                    item.id,
                                    worker_id,
                                    BatchItemStatus.COMPLETED
                                )
                                return 1
                        except Exception as e:
                            return -1
                        return 0
                    
                    start_time = time.time()
                    
                    with ThreadPoolExecutor(max_workers=worker_count) as executor:
                        futures = [
                            executor.submit(worker_function, worker.worker_id)
                            for worker in workers[:min(100, worker_count)]  # Limit test workers
                        ]
                        
                        results = []
                        for future in as_completed(futures, timeout=60):
                            try:
                                result = future.result()
                                results.append(result)
                            except Exception as e:
                                results.append(-1)
                    
                    test_time = time.time() - start_time
                    success_count = sum(1 for r in results if r > 0)
                    error_count = sum(1 for r in results if r < 0)
                    error_rate = error_count / len(results) if results else 1.0
                    
                    # Check if system is breaking down
                    if error_rate > 0.5 or test_time > 120:
                        breaking_point = worker_count
                        print(f"Concurrency breaking point: {worker_count} workers "
                              f"({error_rate:.1%} error rate, {test_time:.1f}s)")
                        break
                    
                    print(f"Workers {worker_count}: {error_rate:.1%} error rate - OK")
                    
                    # Cleanup workers for next test
                    for worker in workers:
                        queue_service.unregister_worker(worker.worker_id)
                
                except Exception as e:
                    breaking_point = worker_count
                    print(f"Concurrency breaking point: {worker_count} workers - {e}")
                    break
        
        # Document concurrency breaking point
        if breaking_point:
            print(f"System concurrency breaking point: {breaking_point} workers")
            assert breaking_point > 10  # Should handle at least 10 concurrent workers
        else:
            print("No concurrency breaking point found within tested range")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-s"])