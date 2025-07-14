"""
Performance and load tests for the status tracking system.

This module provides performance tests to ensure the status tracking
system can handle high loads and concurrent operations efficiently.
"""

import pytest
import asyncio
import time
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import Mock, patch
import statistics

from src.services.status_service import StatusService
from src.services.status_updater import StatusUpdater, StatusUpdate, ProgressUpdate
from src.services.status_filtering import StatusFilterService, FilterQuery, FilterCondition, FilterOperator
from src.services.status_events import StatusEventManager, StatusEvent, EventType
from src.database.status_models import ProcessingStatusType, ProcessingPriority


class TestStatusServicePerformance:
    """Performance tests for StatusService."""
    
    @pytest.fixture
    def mock_db_session(self):
        """Create optimized mock database session."""
        session = Mock()
        # Mock bulk operations
        session.bulk_insert_mappings = Mock()
        session.bulk_update_mappings = Mock()
        return session
    
    @pytest.fixture
    def status_service(self, mock_db_session):
        """Create StatusService with performance mocks."""
        service = StatusService(db_session=mock_db_session)
        
        # Mock time-consuming operations
        service.db_session.add = Mock()
        service.db_session.commit = Mock()
        service._create_status_history = Mock()
        
        return service
    
    def test_bulk_status_creation_performance(self, status_service):
        """Test performance of creating multiple statuses."""
        num_statuses = 1000
        start_time = time.time()
        
        # Create multiple statuses
        status_ids = []
        for i in range(num_statuses):
            status = status_service.create_processing_status(
                video_id=i,
                priority=ProcessingPriority.NORMAL,
                total_steps=5
            )
            status_ids.append(status.status_id)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Performance assertions
        assert len(status_ids) == num_statuses
        assert duration < 5.0  # Should complete within 5 seconds
        assert (duration / num_statuses) < 0.01  # Less than 10ms per status
        
        print(f"Created {num_statuses} statuses in {duration:.2f}s ({duration/num_statuses*1000:.2f}ms per status)")
    
    def test_concurrent_status_updates_performance(self, status_service):
        """Test performance of concurrent status updates."""
        num_updates = 500
        num_threads = 10
        
        # Mock existing statuses
        def mock_get_status(status_id):
            status = Mock()
            status.status_id = status_id
            status.status = ProcessingStatusType.STARTING
            status.progress_percentage = 0.0
            return status
        
        status_service.get_processing_status = mock_get_status
        
        def update_status(i):
            start = time.time()
            status_service.update_status(
                status_id=f"status_{i}",
                new_status=ProcessingStatusType.COMPLETED,
                progress_percentage=100.0
            )
            return time.time() - start
        
        # Execute concurrent updates
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(update_status, i) for i in range(num_updates)]
            update_times = [future.result() for future in as_completed(futures)]
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        # Performance assertions
        assert len(update_times) == num_updates
        assert total_duration < 10.0  # Should complete within 10 seconds
        assert max(update_times) < 0.1  # No single update should take more than 100ms
        assert statistics.mean(update_times) < 0.05  # Average under 50ms
        
        print(f"Completed {num_updates} concurrent updates in {total_duration:.2f}s")
        print(f"Average update time: {statistics.mean(update_times)*1000:.2f}ms")
        print(f"Max update time: {max(update_times)*1000:.2f}ms")
    
    def test_status_query_performance(self, status_service):
        """Test performance of status queries."""
        num_queries = 1000
        
        # Mock query result
        mock_status = Mock()
        mock_status.status_id = "test_status"
        status_service.db_session.query.return_value.filter.return_value.first.return_value = mock_status
        
        # Execute queries
        start_time = time.time()
        for i in range(num_queries):
            status = status_service.get_processing_status(f"status_{i}")
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Performance assertions
        assert duration < 2.0  # Should complete within 2 seconds
        assert (duration / num_queries) < 0.005  # Less than 5ms per query
        
        print(f"Executed {num_queries} status queries in {duration:.2f}s ({duration/num_queries*1000:.2f}ms per query)")


class TestStatusUpdaterPerformance:
    """Performance tests for StatusUpdater."""
    
    @pytest.fixture
    def mock_db_session(self):
        """Create mock database session."""
        return Mock()
    
    @pytest.fixture
    def status_updater(self, mock_db_session):
        """Create StatusUpdater with performance mocks."""
        updater = StatusUpdater(db_session=mock_db_session)
        updater.status_service = Mock()
        return updater
    
    @pytest.mark.asyncio
    async def test_batch_update_performance(self, status_updater):
        """Test performance of batch status updates."""
        num_updates = 5000
        batch_size = 100
        
        # Queue many updates
        for i in range(num_updates):
            update = StatusUpdate(
                status_id=f"status_{i}",
                new_status=ProcessingStatusType.COMPLETED,
                progress_percentage=100.0
            )
            status_updater.queue_status_update(update)
        
        # Process updates
        start_time = time.time()
        await status_updater.process_update_queues()
        end_time = time.time()
        
        duration = end_time - start_time
        
        # Performance assertions
        assert len(status_updater.status_update_queue) == 0  # All updates processed
        assert duration < 5.0  # Should complete within 5 seconds
        
        # Calculate throughput
        throughput = num_updates / duration
        assert throughput > 1000  # Should process at least 1000 updates per second
        
        print(f"Processed {num_updates} batch updates in {duration:.2f}s ({throughput:.0f} updates/sec)")
    
    @pytest.mark.asyncio
    async def test_mixed_update_types_performance(self, status_updater):
        """Test performance with mixed update types."""
        num_each_type = 1000
        
        # Queue different types of updates
        for i in range(num_each_type):
            # Status updates
            status_update = StatusUpdate(
                status_id=f"status_{i}",
                new_status=ProcessingStatusType.COMPLETED
            )
            status_updater.queue_status_update(status_update)
            
            # Progress updates
            progress_update = ProgressUpdate(
                status_id=f"status_{i}",
                progress_percentage=float(i % 101)
            )
            status_updater.queue_progress_update(progress_update)
        
        total_updates = num_each_type * 2
        
        # Process all updates
        start_time = time.time()
        await status_updater.process_update_queues()
        end_time = time.time()
        
        duration = end_time - start_time
        
        # Performance assertions
        assert len(status_updater.status_update_queue) == 0
        assert len(status_updater.progress_update_queue) == 0
        assert duration < 8.0  # Should complete within 8 seconds
        
        throughput = total_updates / duration
        print(f"Processed {total_updates} mixed updates in {duration:.2f}s ({throughput:.0f} updates/sec)")


class TestStatusFilteringPerformance:
    """Performance tests for StatusFilterService."""
    
    @pytest.fixture
    def mock_db_session(self):
        """Create mock database session."""
        session = Mock()
        
        # Mock large query result
        mock_query = Mock()
        mock_query.count.return_value = 100000  # Large dataset
        mock_query.all.return_value = [Mock() for _ in range(100)]  # Page of results
        mock_query.offset.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.filter.return_value = mock_query
        
        session.query.return_value = mock_query
        return session
    
    @pytest.fixture
    def filter_service(self, mock_db_session):
        """Create StatusFilterService with performance mocks."""
        return StatusFilterService(db_session=mock_db_session)
    
    def test_complex_filter_performance(self, filter_service):
        """Test performance of complex filtering operations."""
        from src.services.status_filtering import FilterQuery, PaginationParams
        
        # Create complex filter query
        filter_query = FilterQuery(
            filters=[
                FilterCondition("status", FilterOperator.IN, [
                    ProcessingStatusType.COMPLETED,
                    ProcessingStatusType.FAILED,
                    ProcessingStatusType.CANCELLED
                ]),
                FilterCondition("created_at", FilterOperator.GTE, datetime.utcnow() - timedelta(days=30)),
                FilterCondition("priority", FilterOperator.EQ, ProcessingPriority.HIGH),
                FilterCondition("progress_percentage", FilterOperator.GTE, 50.0)
            ],
            pagination=PaginationParams(page=1, page_size=100)
        )
        
        # Execute multiple complex queries
        num_queries = 100
        start_time = time.time()
        
        for _ in range(num_queries):
            result = filter_service.filter_statuses(filter_query)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Performance assertions
        assert duration < 5.0  # Should complete within 5 seconds
        query_time = duration / num_queries
        assert query_time < 0.1  # Less than 100ms per complex query
        
        print(f"Executed {num_queries} complex filter queries in {duration:.2f}s ({query_time*1000:.2f}ms per query)")
    
    def test_pagination_performance(self, filter_service):
        """Test performance of pagination with large datasets."""
        from src.services.status_filtering import FilterQuery, PaginationParams
        
        # Test pagination through large dataset
        page_size = 50
        num_pages = 100
        
        start_time = time.time()
        
        for page in range(1, num_pages + 1):
            filter_query = FilterQuery(
                pagination=PaginationParams(page=page, page_size=page_size)
            )
            result = filter_service.filter_statuses(filter_query)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Performance assertions
        assert duration < 10.0  # Should complete within 10 seconds
        page_time = duration / num_pages
        assert page_time < 0.2  # Less than 200ms per page
        
        print(f"Paginated through {num_pages} pages in {duration:.2f}s ({page_time*1000:.2f}ms per page)")
    
    def test_aggregation_performance(self, filter_service):
        """Test performance of aggregation queries."""
        # Mock aggregation query results
        filter_service.db_session.query.return_value.with_entities.return_value.group_by.return_value.all.return_value = [
            (ProcessingStatusType.COMPLETED, 80000),
            (ProcessingStatusType.FAILED, 15000),
            (ProcessingStatusType.CANCELLED, 5000)
        ]
        filter_service.db_session.query.return_value.with_entities.return_value.first.return_value = Mock(
            avg_progress=75.5,
            min_progress=0.0,
            max_progress=100.0,
            unique_workers=50,
            total_with_workers=95000
        )
        
        # Execute multiple aggregation queries
        num_queries = 50
        start_time = time.time()
        
        for _ in range(num_queries):
            result = filter_service.get_status_aggregates()
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Performance assertions
        assert duration < 5.0  # Should complete within 5 seconds
        query_time = duration / num_queries
        assert query_time < 0.2  # Less than 200ms per aggregation query
        
        print(f"Executed {num_queries} aggregation queries in {duration:.2f}s ({query_time*1000:.2f}ms per query)")


class TestEventSystemPerformance:
    """Performance tests for the event system."""
    
    @pytest.mark.asyncio
    async def test_high_volume_event_processing(self):
        """Test processing high volume of events."""
        num_events = 10000
        max_workers = 5
        
        # Create event manager
        manager = StatusEventManager(max_workers=max_workers)
        
        # Track processed events
        processed_events = []
        
        class PerformanceEventHandler:
            async def handle_event(self, event):
                processed_events.append(event.event_id)
                return True
            
            def get_handled_event_types(self):
                return {EventType.STATUS_UPDATED, EventType.PROGRESS_UPDATED}
            
            def should_handle_event(self, event):
                return event.event_type in self.get_handled_event_types()
        
        handler = PerformanceEventHandler()
        manager.add_handler(handler)
        
        await manager.start()
        
        try:
            # Emit many events rapidly
            start_time = time.time()
            
            for i in range(num_events):
                event = StatusEvent(
                    event_type=EventType.STATUS_UPDATED if i % 2 == 0 else EventType.PROGRESS_UPDATED,
                    status_id=f"status_{i}",
                    progress_percentage=float(i % 101)
                )
                await manager.emit_event(event)
            
            # Wait for processing to complete
            while manager.event_queue.qsize() > 0:
                await asyncio.sleep(0.1)
            
            # Give extra time for final processing
            await asyncio.sleep(1.0)
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Performance assertions
            assert len(processed_events) >= num_events * 0.95  # At least 95% processed
            assert duration < 30.0  # Should complete within 30 seconds
            
            throughput = len(processed_events) / duration
            assert throughput > 500  # Should process at least 500 events per second
            
            print(f"Processed {len(processed_events)}/{num_events} events in {duration:.2f}s ({throughput:.0f} events/sec)")
            
        finally:
            await manager.stop()
    
    @pytest.mark.asyncio
    async def test_concurrent_event_emission(self):
        """Test concurrent event emission performance."""
        num_emitters = 10
        events_per_emitter = 500
        total_events = num_emitters * events_per_emitter
        
        manager = StatusEventManager(max_workers=8)
        
        processed_count = 0
        
        class ConcurrentEventHandler:
            async def handle_event(self, event):
                nonlocal processed_count
                processed_count += 1
                return True
            
            def get_handled_event_types(self):
                return set(EventType)
            
            def should_handle_event(self, event):
                return True
        
        handler = ConcurrentEventHandler()
        manager.add_handler(handler)
        
        await manager.start()
        
        try:
            async def emit_events(emitter_id):
                for i in range(events_per_emitter):
                    event = StatusEvent(
                        event_type=EventType.STATUS_UPDATED,
                        status_id=f"emitter_{emitter_id}_status_{i}"
                    )
                    await manager.emit_event(event)
            
            # Run concurrent emitters
            start_time = time.time()
            tasks = [emit_events(i) for i in range(num_emitters)]
            await asyncio.gather(*tasks)
            
            # Wait for processing
            while manager.event_queue.qsize() > 0:
                await asyncio.sleep(0.1)
            
            await asyncio.sleep(2.0)  # Extra time for processing
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Performance assertions
            assert processed_count >= total_events * 0.95  # At least 95% processed
            assert duration < 20.0  # Should complete within 20 seconds
            
            throughput = processed_count / duration
            print(f"Concurrent emission: {processed_count}/{total_events} events in {duration:.2f}s ({throughput:.0f} events/sec)")
            
        finally:
            await manager.stop()


class TestMemoryUsagePerformance:
    """Performance tests focusing on memory usage."""
    
    def test_status_service_memory_efficiency(self):
        """Test memory efficiency of status service operations."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create many status objects
        statuses = []
        with patch('src.services.status_service.get_db_session'):
            service = StatusService()
            service.db_session = Mock()
            service.db_session.add = Mock()
            service.db_session.commit = Mock()
            service._create_status_history = Mock()
            
            for i in range(10000):
                status = service.create_processing_status(
                    video_id=i,
                    total_steps=5
                )
                if i % 1000 == 0:  # Keep some references to prevent garbage collection
                    statuses.append(status)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory assertions (allowing for some overhead)
        assert memory_increase < 100  # Should not increase by more than 100MB
        
        print(f"Memory usage: {initial_memory:.1f}MB -> {final_memory:.1f}MB (increase: {memory_increase:.1f}MB)")
    
    @pytest.mark.asyncio
    async def test_event_queue_memory_efficiency(self):
        """Test memory efficiency of event queue under load."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        manager = StatusEventManager(max_workers=4)
        
        # Handler that introduces small delay to build up queue
        class SlowEventHandler:
            async def handle_event(self, event):
                await asyncio.sleep(0.001)  # 1ms delay
                return True
            
            def get_handled_event_types(self):
                return set(EventType)
            
            def should_handle_event(self, event):
                return True
        
        handler = SlowEventHandler()
        manager.add_handler(handler)
        
        await manager.start()
        
        try:
            # Emit events faster than they can be processed
            for i in range(5000):
                event = StatusEvent(
                    event_type=EventType.STATUS_UPDATED,
                    status_id=f"status_{i}"
                )
                await manager.emit_event(event)
                
                if i % 1000 == 0:
                    current_memory = process.memory_info().rss / 1024 / 1024
                    queue_size = manager.event_queue.qsize()
                    print(f"Events emitted: {i}, Queue size: {queue_size}, Memory: {current_memory:.1f}MB")
            
            # Wait for queue to drain
            while manager.event_queue.qsize() > 0:
                await asyncio.sleep(0.1)
            
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory
            
            # Memory assertions
            assert memory_increase < 50  # Should not increase by more than 50MB
            
            print(f"Event queue memory usage: {initial_memory:.1f}MB -> {final_memory:.1f}MB (increase: {memory_increase:.1f}MB)")
            
        finally:
            await manager.stop()


if __name__ == "__main__":
    # Run performance tests with detailed output
    pytest.main([
        __file__,
        "-v",
        "-s",  # Don't capture output so we can see performance metrics
        "--tb=short"
    ])