"""
Performance tests for batch processing operations.

This test suite provides comprehensive performance testing including:
- Throughput testing for batch operations
- Latency measurement for critical operations
- Scalability testing with increasing loads
- Memory usage profiling
- Database performance optimization
- Concurrent processing efficiency
- Resource utilization analysis
- Bottleneck identification
"""

import pytest
import time
import threading
import psutil
import os
import asyncio
from unittest.mock import Mock, patch
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

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
    ConcurrentBatchService, ConcurrentBatchConfig, ConcurrentBatchMode
)
from src.services.batch_processor import BatchProcessor
from src.flow import WorkflowConfig


class PerformanceProfiler:
    """Helper class for performance profiling."""
    
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.start_time = None
        self.start_memory = None
        self.start_cpu_percent = None
        
    def start(self):
        """Start performance monitoring."""
        self.start_time = time.time()
        self.start_memory = self.process.memory_info().rss
        self.start_cpu_percent = self.process.cpu_percent()
        
    def stop(self):
        """Stop performance monitoring and return metrics."""
        end_time = time.time()
        end_memory = self.process.memory_info().rss
        end_cpu_percent = self.process.cpu_percent()
        
        return {
            'duration': end_time - self.start_time,
            'memory_delta': end_memory - self.start_memory,
            'memory_peak': end_memory,
            'cpu_usage': end_cpu_percent
        }


@contextmanager
def performance_monitor():
    """Context manager for performance monitoring."""
    profiler = PerformanceProfiler()
    profiler.start()
    try:
        yield profiler
    finally:
        metrics = profiler.stop()
        return metrics


class TestBatchPerformance:
    """Performance tests for batch processing operations."""

    @pytest.fixture(scope="function")
    def db_engine(self):
        """Create in-memory database engine optimized for performance."""
        engine = create_engine(
            "sqlite:///:memory:",
            connect_args={
                "check_same_thread": False,
                "timeout": 30
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
        """Create optimized database session."""
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
        """Create QueueService instance with performance optimizations."""
        options = QueueProcessingOptions(
            max_workers=10,
            worker_timeout_minutes=5,
            lock_timeout_minutes=2,
            heartbeat_interval_seconds=10,
            stale_lock_cleanup_interval_minutes=5,
            enable_automatic_cleanup=False
        )
        service = QueueService(db_session, options)
        yield service
        service.shutdown()

    @pytest.fixture(scope="function")
    def concurrent_batch_service(self, db_session):
        """Create ConcurrentBatchService with performance config."""
        config = ConcurrentBatchConfig(
            max_concurrent_batches=10,
            max_concurrent_items_per_batch=20,
            max_total_concurrent_items=100,
            max_workers_per_batch=5,
            max_api_calls_per_second=10.0,
            max_database_connections=20,
            enable_performance_monitoring=True,
            cleanup_interval_seconds=60,
            heartbeat_interval_seconds=30
        )
        service = ConcurrentBatchService(config=config, session=db_session)
        yield service
        service.shutdown()

    @pytest.fixture
    def small_url_list(self):
        """Small list of URLs for basic performance tests."""
        return [f"https://www.youtube.com/watch?v=small{i:03d}" for i in range(10)]

    @pytest.fixture
    def medium_url_list(self):
        """Medium list of URLs for moderate performance tests."""
        return [f"https://www.youtube.com/watch?v=med{i:04d}" for i in range(100)]

    @pytest.fixture
    def large_url_list(self):
        """Large list of URLs for stress performance tests."""
        return [f"https://www.youtube.com/watch?v=large{i:05d}" for i in range(1000)]

    @pytest.fixture
    def xlarge_url_list(self):
        """Extra large list of URLs for extreme performance tests."""
        return [f"https://www.youtube.com/watch?v=xl{i:06d}" for i in range(5000)]

    # Batch Creation Performance Tests

    def test_batch_creation_performance_small(self, batch_service, small_url_list):
        """Test batch creation performance with small datasets."""
        request = BatchCreateRequest(
            name="Small Performance Test",
            urls=small_url_list,
            priority=BatchPriority.NORMAL
        )
        
        with patch('src.services.batch_service.extract_video_id_from_url') as mock_extract:
            mock_extract.side_effect = lambda url: url.split('=')[-1]
            
            with performance_monitor() as profiler:
                batch = batch_service.create_batch(request)
            
            metrics = profiler.stop()
            
            # Performance assertions
            assert metrics['duration'] < 1.0  # Should complete within 1 second
            assert metrics['memory_delta'] < 10 * 1024 * 1024  # Less than 10MB
            assert batch.total_items == 10
            
            print(f"Small batch creation: {metrics['duration']:.3f}s, {metrics['memory_delta']//1024}KB")

    def test_batch_creation_performance_medium(self, batch_service, medium_url_list):
        """Test batch creation performance with medium datasets."""
        request = BatchCreateRequest(
            name="Medium Performance Test",
            urls=medium_url_list,
            priority=BatchPriority.NORMAL
        )
        
        with patch('src.services.batch_service.extract_video_id_from_url') as mock_extract:
            mock_extract.side_effect = lambda url: url.split('=')[-1]
            
            with performance_monitor() as profiler:
                batch = batch_service.create_batch(request)
            
            metrics = profiler.stop()
            
            # Performance assertions
            assert metrics['duration'] < 5.0  # Should complete within 5 seconds
            assert metrics['memory_delta'] < 50 * 1024 * 1024  # Less than 50MB
            assert batch.total_items == 100
            
            print(f"Medium batch creation: {metrics['duration']:.3f}s, {metrics['memory_delta']//1024}KB")

    def test_batch_creation_performance_large(self, batch_service, large_url_list):
        """Test batch creation performance with large datasets."""
        request = BatchCreateRequest(
            name="Large Performance Test",
            urls=large_url_list,
            priority=BatchPriority.NORMAL
        )
        
        with patch('src.services.batch_service.extract_video_id_from_url') as mock_extract:
            mock_extract.side_effect = lambda url: url.split('=')[-1]
            
            with performance_monitor() as profiler:
                batch = batch_service.create_batch(request)
            
            metrics = profiler.stop()
            
            # Performance assertions
            assert metrics['duration'] < 30.0  # Should complete within 30 seconds
            assert metrics['memory_delta'] < 200 * 1024 * 1024  # Less than 200MB
            assert batch.total_items == 1000
            
            print(f"Large batch creation: {metrics['duration']:.3f}s, {metrics['memory_delta']//1024}KB")

    def test_batch_creation_scalability(self, batch_service):
        """Test batch creation scalability with increasing sizes."""
        sizes = [10, 50, 100, 250, 500]
        results = []
        
        with patch('src.services.batch_service.extract_video_id_from_url') as mock_extract:
            mock_extract.side_effect = lambda url: url.split('=')[-1]
            
            for size in sizes:
                urls = [f"https://www.youtube.com/watch?v=scale{i:04d}" for i in range(size)]
                request = BatchCreateRequest(
                    name=f"Scalability Test {size}",
                    urls=urls,
                    priority=BatchPriority.NORMAL
                )
                
                start_time = time.time()
                batch = batch_service.create_batch(request)
                duration = time.time() - start_time
                
                results.append({
                    'size': size,
                    'duration': duration,
                    'items_per_second': size / duration
                })
                
                assert batch.total_items == size
                print(f"Size {size}: {duration:.3f}s ({size/duration:.1f} items/s)")
        
        # Verify scalability - duration should scale reasonably
        for i in range(1, len(results)):
            prev_result = results[i-1]
            curr_result = results[i]
            
            # Performance should not degrade drastically
            scale_factor = curr_result['size'] / prev_result['size']
            time_factor = curr_result['duration'] / prev_result['duration']
            
            # Time increase should not be more than 3x the scale factor
            assert time_factor < scale_factor * 3

    # Queue Processing Performance Tests

    def test_queue_processing_throughput(self, batch_service, queue_service, medium_url_list):
        """Test queue processing throughput."""
        # Create batch
        request = BatchCreateRequest(
            name="Throughput Test",
            urls=medium_url_list,
            priority=BatchPriority.NORMAL
        )
        
        with patch('src.services.batch_service.extract_video_id_from_url') as mock_extract:
            mock_extract.side_effect = lambda url: url.split('=')[-1]
            
            batch = batch_service.create_batch(request)
            batch_service.start_batch_processing(batch.batch_id)
            
            # Register workers
            num_workers = 5
            workers = []
            for i in range(num_workers):
                worker = queue_service.register_worker("video_processing", f"throughput_worker_{i}")
                workers.append(worker)
            
            # Measure processing throughput
            start_time = time.time()
            processed_count = 0
            
            while processed_count < 100:
                for worker in workers:
                    if processed_count >= 100:
                        break
                    
                    item = queue_service.get_next_queue_item("video_processing", worker.worker_id)
                    if item:
                        queue_service.complete_queue_item(
                            item.id,
                            worker.worker_id,
                            BatchItemStatus.COMPLETED
                        )
                        processed_count += 1
                    else:
                        break
            
            duration = time.time() - start_time
            throughput = processed_count / duration
            
            # Performance assertions
            assert throughput > 10.0  # Should process at least 10 items per second
            assert duration < 30.0    # Should complete within 30 seconds
            
            print(f"Queue throughput: {throughput:.1f} items/s, {duration:.3f}s total")

    def test_queue_latency_measurement(self, batch_service, queue_service, small_url_list):
        """Test queue operation latency."""
        # Create batch
        request = BatchCreateRequest(
            name="Latency Test",
            urls=small_url_list,
            priority=BatchPriority.NORMAL
        )
        
        with patch('src.services.batch_service.extract_video_id_from_url') as mock_extract:
            mock_extract.side_effect = lambda url: url.split('=')[-1]
            
            batch = batch_service.create_batch(request)
            batch_service.start_batch_processing(batch.batch_id)
            
            # Register worker
            worker = queue_service.register_worker("video_processing", "latency_worker")
            
            # Measure individual operation latencies
            latencies = {
                'get_item': [],
                'complete_item': [],
                'get_statistics': []
            }
            
            for i in range(10):
                # Measure get_next_queue_item latency
                start_time = time.time()
                item = queue_service.get_next_queue_item("video_processing", "latency_worker")
                get_latency = time.time() - start_time
                latencies['get_item'].append(get_latency)
                
                if item:
                    # Measure complete_queue_item latency
                    start_time = time.time()
                    queue_service.complete_queue_item(
                        item.id,
                        "latency_worker",
                        BatchItemStatus.COMPLETED
                    )
                    complete_latency = time.time() - start_time
                    latencies['complete_item'].append(complete_latency)
                
                # Measure get_statistics latency
                start_time = time.time()
                stats = queue_service.get_queue_statistics("video_processing")
                stats_latency = time.time() - start_time
                latencies['get_statistics'].append(stats_latency)
            
            # Calculate average latencies
            avg_latencies = {
                operation: sum(times) / len(times)
                for operation, times in latencies.items()
                if times
            }
            
            # Performance assertions
            assert avg_latencies['get_item'] < 0.1      # Less than 100ms
            assert avg_latencies['complete_item'] < 0.1 # Less than 100ms
            assert avg_latencies['get_statistics'] < 0.2 # Less than 200ms
            
            print(f"Latencies - Get: {avg_latencies['get_item']*1000:.1f}ms, "
                  f"Complete: {avg_latencies['complete_item']*1000:.1f}ms, "
                  f"Stats: {avg_latencies['get_statistics']*1000:.1f}ms")

    def test_concurrent_queue_performance(self, batch_service, queue_service, large_url_list):
        """Test concurrent queue processing performance."""
        # Create batch
        request = BatchCreateRequest(
            name="Concurrent Performance Test",
            urls=large_url_list,
            priority=BatchPriority.NORMAL
        )
        
        with patch('src.services.batch_service.extract_video_id_from_url') as mock_extract:
            mock_extract.side_effect = lambda url: url.split('=')[-1]
            
            batch = batch_service.create_batch(request)
            batch_service.start_batch_processing(batch.batch_id)
            
            # Register multiple workers
            num_workers = 10
            workers = []
            for i in range(num_workers):
                worker = queue_service.register_worker("video_processing", f"concurrent_worker_{i}")
                workers.append(worker)
            
            # Process items concurrently
            processed_count = 0
            target_items = 500  # Process 500 items
            
            def worker_function(worker_id):
                local_processed = 0
                while True:
                    item = queue_service.get_next_queue_item("video_processing", worker_id)
                    if item:
                        queue_service.complete_queue_item(
                            item.id,
                            worker_id,
                            BatchItemStatus.COMPLETED
                        )
                        local_processed += 1
                        
                        nonlocal processed_count
                        processed_count += 1
                        
                        if processed_count >= target_items:
                            break
                    else:
                        break
                return local_processed
            
            # Run concurrent processing
            start_time = time.time()
            
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [
                    executor.submit(worker_function, worker.worker_id)
                    for worker in workers
                ]
                
                results = [future.result() for future in futures]
            
            duration = time.time() - start_time
            total_processed = sum(results)
            throughput = total_processed / duration
            
            # Performance assertions
            assert total_processed >= target_items
            assert throughput > 20.0  # Should achieve higher throughput with concurrency
            assert duration < 60.0    # Should complete within 60 seconds
            
            print(f"Concurrent throughput: {throughput:.1f} items/s, "
                  f"{total_processed} items in {duration:.3f}s")

    # Database Performance Tests

    def test_database_bulk_operations_performance(self, batch_service, xlarge_url_list):
        """Test database bulk operations performance."""
        request = BatchCreateRequest(
            name="Database Bulk Test",
            urls=xlarge_url_list,
            priority=BatchPriority.NORMAL
        )
        
        with patch('src.services.batch_service.extract_video_id_from_url') as mock_extract:
            mock_extract.side_effect = lambda url: url.split('=')[-1]
            
            # Measure bulk insert performance
            with performance_monitor() as profiler:
                batch = batch_service.create_batch(request)
            
            metrics = profiler.stop()
            
            # Calculate insertion rate
            items_per_second = batch.total_items / metrics['duration']
            
            # Performance assertions
            assert items_per_second > 100  # Should insert at least 100 items/sec
            assert metrics['duration'] < 120  # Should complete within 2 minutes
            assert batch.total_items == 5000
            
            print(f"Bulk insert: {items_per_second:.1f} items/s, "
                  f"{metrics['duration']:.3f}s for {batch.total_items} items")

    def test_database_query_performance(self, batch_service, queue_service, large_url_list):
        """Test database query performance under load."""
        # Create multiple batches
        batches = []
        num_batches = 10
        
        with patch('src.services.batch_service.extract_video_id_from_url') as mock_extract:
            mock_extract.side_effect = lambda url: url.split('=')[-1]
            
            # Create batches
            for i in range(num_batches):
                request = BatchCreateRequest(
                    name=f"Query Performance Batch {i}",
                    urls=large_url_list[:100],  # 100 items per batch
                    priority=BatchPriority.NORMAL
                )
                batch = batch_service.create_batch(request)
                batches.append(batch)
            
            # Measure query performance
            query_times = []
            
            # Test various query operations
            operations = [
                ('list_batches', lambda: batch_service.list_batches()),
                ('get_statistics', lambda: batch_service.get_batch_statistics()),
                ('get_progress', lambda: batch_service.get_batch_progress(batches[0].batch_id)),
                ('queue_stats', lambda: queue_service.get_queue_statistics("video_processing"))
            ]
            
            for operation_name, operation in operations:
                times = []
                for _ in range(10):  # Run each operation 10 times
                    start_time = time.time()
                    result = operation()
                    duration = time.time() - start_time
                    times.append(duration)
                
                avg_time = sum(times) / len(times)
                query_times.append((operation_name, avg_time))
                
                # Performance assertions based on operation type
                if operation_name == 'list_batches':
                    assert avg_time < 0.5  # Should list batches in less than 500ms
                elif operation_name == 'get_statistics':
                    assert avg_time < 1.0  # Should calculate stats in less than 1s
                elif operation_name == 'get_progress':
                    assert avg_time < 0.2  # Should get progress in less than 200ms
                elif operation_name == 'queue_stats':
                    assert avg_time < 0.3  # Should get queue stats in less than 300ms
            
            print("Query performance:")
            for operation, avg_time in query_times:
                print(f"  {operation}: {avg_time*1000:.1f}ms")

    # Memory Performance Tests

    def test_memory_usage_scaling(self, batch_service):
        """Test memory usage scaling with increasing batch sizes."""
        sizes = [100, 500, 1000, 2000]
        memory_usage = []
        
        with patch('src.services.batch_service.extract_video_id_from_url') as mock_extract:
            mock_extract.side_effect = lambda url: url.split('=')[-1]
            
            for size in sizes:
                # Force garbage collection before measurement
                import gc
                gc.collect()
                
                # Measure initial memory
                process = psutil.Process(os.getpid())
                initial_memory = process.memory_info().rss
                
                # Create batch
                urls = [f"https://www.youtube.com/watch?v=mem{i:05d}" for i in range(size)]
                request = BatchCreateRequest(
                    name=f"Memory Test {size}",
                    urls=urls,
                    priority=BatchPriority.NORMAL
                )
                
                batch = batch_service.create_batch(request)
                
                # Measure final memory
                final_memory = process.memory_info().rss
                memory_delta = final_memory - initial_memory
                memory_per_item = memory_delta / size
                
                memory_usage.append({
                    'size': size,
                    'memory_delta': memory_delta,
                    'memory_per_item': memory_per_item
                })
                
                print(f"Size {size}: {memory_delta//1024}KB total, "
                      f"{memory_per_item:.1f}B per item")
                
                # Memory usage should be reasonable
                assert memory_per_item < 10000  # Less than 10KB per item
        
        # Verify memory usage scales linearly (not exponentially)
        for i in range(1, len(memory_usage)):
            prev = memory_usage[i-1]
            curr = memory_usage[i]
            
            size_ratio = curr['size'] / prev['size']
            memory_ratio = curr['memory_delta'] / prev['memory_delta']
            
            # Memory usage should scale roughly linearly
            assert memory_ratio < size_ratio * 2

    def test_memory_leak_detection(self, batch_service, medium_url_list):
        """Test for memory leaks in batch operations."""
        import gc
        
        # Get baseline memory usage
        gc.collect()
        process = psutil.Process(os.getpid())
        baseline_memory = process.memory_info().rss
        
        with patch('src.services.batch_service.extract_video_id_from_url') as mock_extract:
            mock_extract.side_effect = lambda url: url.split('=')[-1]
            
            # Perform many batch operations
            for i in range(20):
                request = BatchCreateRequest(
                    name=f"Leak Test {i}",
                    urls=medium_url_list,
                    priority=BatchPriority.NORMAL
                )
                
                batch = batch_service.create_batch(request)
                
                # Simulate some operations
                batch_service.get_batch(batch.batch_id)
                batch_service.get_batch_progress(batch.batch_id)
                batch_service.list_batches()
                
                # Force garbage collection periodically
                if i % 5 == 0:
                    gc.collect()
        
        # Final garbage collection
        gc.collect()
        final_memory = process.memory_info().rss
        memory_increase = final_memory - baseline_memory
        
        # Memory increase should be reasonable (less than 100MB)
        assert memory_increase < 100 * 1024 * 1024
        
        print(f"Memory leak test: {memory_increase//1024}KB increase after 20 batches")

    # Concurrent Service Performance Tests

    @pytest.mark.asyncio
    async def test_concurrent_service_performance(self, concurrent_batch_service, medium_url_list):
        """Test ConcurrentBatchService performance."""
        request = BatchCreateRequest(
            name="Concurrent Service Performance",
            urls=medium_url_list,
            priority=BatchPriority.NORMAL
        )
        
        # Mock dependencies
        with patch('src.services.concurrent_batch_service.get_database_session') as mock_get_session:
            mock_session = Mock()
            mock_get_session.return_value = mock_session
            
            with patch('src.services.concurrent_batch_service.BatchService') as mock_batch_service_class:
                mock_batch_service = Mock()
                mock_batch_service_class.return_value = mock_batch_service
                
                mock_batch = Mock()
                mock_batch.batch_id = "concurrent_perf_batch"
                mock_batch.total_items = 100
                mock_batch.batch_metadata = {}
                mock_batch_service.create_batch.return_value = mock_batch
                
                with patch('src.services.concurrent_batch_service.allocate_resource') as mock_allocate:
                    mock_allocate.return_value.__enter__ = Mock(return_value=None)
                    mock_allocate.return_value.__exit__ = Mock(return_value=None)
                    
                    # Measure concurrent batch creation performance
                    start_time = time.time()
                    
                    # Create multiple batches concurrently
                    tasks = []
                    for i in range(5):
                        task = concurrent_batch_service.create_concurrent_batch(
                            request,
                            ConcurrentBatchMode.PARALLEL,
                            max_concurrent_items=20
                        )
                        tasks.append(task)
                    
                    results = await asyncio.gather(*tasks)
                    
                    creation_time = time.time() - start_time
                    
                    # Performance assertions
                    assert creation_time < 10.0  # Should create 5 batches within 10 seconds
                    assert len(results) == 5
                    assert len(concurrent_batch_service._active_batches) == 5
                    
                    # Test worker creation performance
                    start_time = time.time()
                    
                    workers = []
                    for i, batch_id in enumerate(concurrent_batch_service._active_batches):
                        for j in range(3):  # 3 workers per batch
                            worker = concurrent_batch_service._create_batch_worker(
                                batch_id,
                                f"perf_worker_{i}_{j}"
                            )
                            workers.append(worker)
                    
                    worker_creation_time = time.time() - start_time
                    
                    # Performance assertions
                    assert worker_creation_time < 5.0  # Should create 15 workers within 5 seconds
                    assert len(workers) == 15
                    assert len(concurrent_batch_service._worker_registry) == 15
                    
                    print(f"Concurrent service: {creation_time:.3f}s batch creation, "
                          f"{worker_creation_time:.3f}s worker creation")

    # Stress and Load Performance Tests

    def test_high_load_performance(self, batch_service, queue_service):
        """Test performance under high load conditions."""
        # Create high load scenario
        num_batches = 20
        items_per_batch = 50
        num_workers = 15
        
        # Create multiple batches rapidly
        with patch('src.services.batch_service.extract_video_id_from_url') as mock_extract:
            mock_extract.side_effect = lambda url: url.split('=')[-1]
            
            # Batch creation phase
            batch_creation_start = time.time()
            batches = []
            
            for i in range(num_batches):
                urls = [f"https://www.youtube.com/watch?v=load{i:02d}{j:03d}" 
                       for j in range(items_per_batch)]
                request = BatchCreateRequest(
                    name=f"High Load Batch {i}",
                    urls=urls,
                    priority=BatchPriority.NORMAL
                )
                
                batch = batch_service.create_batch(request)
                batch_service.start_batch_processing(batch.batch_id)
                batches.append(batch)
            
            batch_creation_time = time.time() - batch_creation_start
            
            # Worker registration phase
            worker_registration_start = time.time()
            workers = []
            
            for i in range(num_workers):
                worker = queue_service.register_worker("video_processing", f"load_worker_{i}")
                workers.append(worker)
            
            worker_registration_time = time.time() - worker_registration_start
            
            # Processing phase
            processing_start = time.time()
            total_items = num_batches * items_per_batch
            processed_count = 0
            
            while processed_count < total_items:
                batch_processed = 0
                
                for worker in workers:
                    if processed_count >= total_items:
                        break
                    
                    item = queue_service.get_next_queue_item("video_processing", worker.worker_id)
                    if item:
                        queue_service.complete_queue_item(
                            item.id,
                            worker.worker_id,
                            BatchItemStatus.COMPLETED
                        )
                        processed_count += 1
                        batch_processed += 1
                
                # Break if no items processed in this round
                if batch_processed == 0:
                    break
            
            processing_time = time.time() - processing_start
            total_time = time.time() - batch_creation_start
            
            # Performance assertions
            assert batch_creation_time < 60.0    # Batch creation within 60s
            assert worker_registration_time < 5.0 # Worker registration within 5s
            assert processing_time < 300.0       # Processing within 5 minutes
            assert processed_count >= total_items * 0.95  # Process at least 95% of items
            
            # Calculate throughput
            overall_throughput = processed_count / total_time
            processing_throughput = processed_count / processing_time
            
            print(f"High load performance:")
            print(f"  Batch creation: {batch_creation_time:.3f}s for {num_batches} batches")
            print(f"  Worker registration: {worker_registration_time:.3f}s for {num_workers} workers")
            print(f"  Processing: {processing_time:.3f}s for {processed_count} items")
            print(f"  Overall throughput: {overall_throughput:.1f} items/s")
            print(f"  Processing throughput: {processing_throughput:.1f} items/s")
            
            # Verify final state
            final_stats = batch_service.get_batch_statistics()
            assert final_stats["item_status_counts"]["completed"] >= processed_count

    def test_resource_contention_performance(self, batch_service, queue_service, large_url_list):
        """Test performance under resource contention."""
        # Create scenario with high resource contention
        batch_count = 10
        worker_count = 20
        
        with patch('src.services.batch_service.extract_video_id_from_url') as mock_extract:
            mock_extract.side_effect = lambda url: url.split('=')[-1]
            
            # Create batches
            batches = []
            for i in range(batch_count):
                request = BatchCreateRequest(
                    name=f"Contention Batch {i}",
                    urls=large_url_list[:50],  # 50 items per batch
                    priority=BatchPriority.NORMAL
                )
                batch = batch_service.create_batch(request)
                batch_service.start_batch_processing(batch.batch_id)
                batches.append(batch)
            
            # Create many workers competing for resources
            workers = []
            for i in range(worker_count):
                worker = queue_service.register_worker("video_processing", f"contention_worker_{i}")
                workers.append(worker)
            
            # Measure contention impact
            def worker_function(worker_id):
                processed = 0
                contention_delays = []
                
                for _ in range(25):  # Each worker tries to process 25 items
                    start_time = time.time()
                    item = queue_service.get_next_queue_item("video_processing", worker_id)
                    get_time = time.time() - start_time
                    
                    if item:
                        start_time = time.time()
                        queue_service.complete_queue_item(
                            item.id,
                            worker_id,
                            BatchItemStatus.COMPLETED
                        )
                        complete_time = time.time() - start_time
                        
                        contention_delays.append(get_time + complete_time)
                        processed += 1
                    else:
                        break
                
                avg_delay = sum(contention_delays) / len(contention_delays) if contention_delays else 0
                return processed, avg_delay
            
            # Run workers concurrently to create contention
            start_time = time.time()
            
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = [
                    executor.submit(worker_function, worker.worker_id)
                    for worker in workers
                ]
                
                results = [future.result() for future in futures]
            
            total_time = time.time() - start_time
            
            # Analyze results
            total_processed = sum(result[0] for result in results)
            avg_delays = [result[1] for result in results if result[1] > 0]
            overall_avg_delay = sum(avg_delays) / len(avg_delays) if avg_delays else 0
            
            # Performance assertions under contention
            assert total_processed > 400  # Should still process a good number of items
            assert overall_avg_delay < 1.0  # Average delay should be reasonable
            assert total_time < 180.0  # Should complete within 3 minutes
            
            print(f"Resource contention performance:")
            print(f"  Total processed: {total_processed} items")
            print(f"  Total time: {total_time:.3f}s")
            print(f"  Average delay per operation: {overall_avg_delay*1000:.1f}ms")
            print(f"  Throughput under contention: {total_processed/total_time:.1f} items/s")

    # Benchmark Comparison Tests

    def test_performance_benchmarks(self, batch_service, queue_service):
        """Test performance benchmarks for comparison."""
        benchmark_results = {}
        
        # Benchmark 1: Small batch processing
        small_urls = [f"https://www.youtube.com/watch?v=bench{i:03d}" for i in range(50)]
        
        with patch('src.services.batch_service.extract_video_id_from_url') as mock_extract:
            mock_extract.side_effect = lambda url: url.split('=')[-1]
            
            # Small batch benchmark
            start_time = time.time()
            
            request = BatchCreateRequest(name="Benchmark Small", urls=small_urls)
            batch = batch_service.create_batch(request)
            batch_service.start_batch_processing(batch.batch_id)
            
            worker = queue_service.register_worker("video_processing", "benchmark_worker")
            
            processed = 0
            while processed < 50:
                item = queue_service.get_next_queue_item("video_processing", "benchmark_worker")
                if item:
                    queue_service.complete_queue_item(
                        item.id,
                        "benchmark_worker",
                        BatchItemStatus.COMPLETED
                    )
                    processed += 1
                else:
                    break
            
            small_batch_time = time.time() - start_time
            benchmark_results['small_batch'] = {
                'items': 50,
                'time': small_batch_time,
                'throughput': 50 / small_batch_time
            }
            
            # Benchmark 2: Statistics calculation
            start_time = time.time()
            for _ in range(100):
                stats = batch_service.get_batch_statistics()
            stats_time = time.time() - start_time
            
            benchmark_results['statistics'] = {
                'operations': 100,
                'time': stats_time,
                'ops_per_second': 100 / stats_time
            }
            
            # Benchmark 3: Queue operations
            start_time = time.time()
            for _ in range(100):
                queue_stats = queue_service.get_queue_statistics("video_processing")
            queue_stats_time = time.time() - start_time
            
            benchmark_results['queue_stats'] = {
                'operations': 100,
                'time': queue_stats_time,
                'ops_per_second': 100 / queue_stats_time
            }
        
        # Print benchmark results
        print("\nPerformance Benchmarks:")
        for benchmark, results in benchmark_results.items():
            if 'throughput' in results:
                print(f"  {benchmark}: {results['throughput']:.1f} items/s")
            else:
                print(f"  {benchmark}: {results['ops_per_second']:.1f} ops/s")
        
        # Performance assertions (baseline expectations)
        assert benchmark_results['small_batch']['throughput'] > 5.0
        assert benchmark_results['statistics']['ops_per_second'] > 50.0
        assert benchmark_results['queue_stats']['ops_per_second'] > 100.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])