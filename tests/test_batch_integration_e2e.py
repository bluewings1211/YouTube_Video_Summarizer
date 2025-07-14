"""
Comprehensive integration tests for end-to-end batch processing workflows.

This test suite provides complete integration testing including:
- Full batch processing workflows from creation to completion
- Integration between all batch processing components
- Real database operations with proper cleanup
- API endpoint integration testing
- PocketFlow workflow integration
- Error handling and recovery scenarios
- Performance testing under realistic conditions
- Multi-component interaction testing
"""

import pytest
import asyncio
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
import tempfile
import os

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
from src.services.video_service import VideoService
from src.flow import WorkflowConfig
from src.utils.concurrency_manager import ConcurrencyManager


class TestBatchIntegrationE2E:
    """End-to-end integration tests for batch processing workflows."""

    @pytest.fixture(scope="function")
    def db_engine(self):
        """Create in-memory database engine for testing."""
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
        session.rollback()  # Ensure cleanup
        session.close()

    @pytest.fixture(scope="function")
    def batch_service(self, db_session):
        """Create BatchService instance."""
        return BatchService(session=db_session)

    @pytest.fixture(scope="function")
    def queue_service(self, db_session):
        """Create QueueService instance."""
        options = QueueProcessingOptions(
            max_workers=3,
            worker_timeout_minutes=5,
            enable_automatic_cleanup=False
        )
        service = QueueService(db_session, options)
        yield service
        service.shutdown()

    @pytest.fixture(scope="function")
    def video_service(self, db_session):
        """Create VideoService instance."""
        return VideoService(session=db_session)

    @pytest.fixture(scope="function")
    def batch_processor(self, batch_service, video_service):
        """Create BatchProcessor instance."""
        config = WorkflowConfig()
        config.enable_monitoring = True
        config.max_retries = 2
        config.timeout_seconds = 300
        return BatchProcessor(
            batch_service=batch_service,
            video_service=video_service,
            workflow_config=config
        )

    @pytest.fixture(scope="function")
    def concurrent_batch_service(self, db_session):
        """Create ConcurrentBatchService instance."""
        config = ConcurrentBatchConfig(
            max_concurrent_batches=3,
            max_concurrent_items_per_batch=5,
            max_total_concurrent_items=10,
            max_workers_per_batch=2,
            enable_performance_monitoring=False
        )
        service = ConcurrentBatchService(config=config, session=db_session)
        yield service
        service.shutdown()

    @pytest.fixture
    def sample_urls(self):
        """Sample YouTube URLs for testing."""
        return [
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://www.youtube.com/watch?v=9bZkp7q19f0",
            "https://youtu.be/oHg5SJYRHA0"
        ]

    @pytest.fixture
    def large_url_list(self):
        """Large list of URLs for performance testing."""
        return [f"https://www.youtube.com/watch?v=test{i:04d}" for i in range(50)]

    # Complete End-to-End Workflow Tests

    def test_complete_batch_processing_workflow(self, batch_service, queue_service, sample_urls):
        """Test complete batch processing workflow from creation to completion."""
        # Step 1: Create batch
        request = BatchCreateRequest(
            name="E2E Test Batch",
            description="End-to-end test batch",
            urls=sample_urls,
            priority=BatchPriority.NORMAL,
            batch_metadata={"test_type": "e2e"}
        )
        
        with patch('src.services.batch_service.extract_video_id_from_url') as mock_extract:
            mock_extract.side_effect = lambda url: url.split('=')[-1] if '=' in url else url.split('/')[-1]
            
            batch = batch_service.create_batch(request)
            
            # Verify batch creation
            assert batch is not None
            assert batch.name == "E2E Test Batch"
            assert batch.total_items == 3
            assert batch.status == BatchStatus.PENDING
            
            # Step 2: Start batch processing
            result = batch_service.start_batch_processing(batch.batch_id)
            assert result is True
            
            # Verify batch status
            updated_batch = batch_service.get_batch(batch.batch_id)
            assert updated_batch.status == BatchStatus.PROCESSING
            
            # Step 3: Register workers and process items
            worker1 = queue_service.register_worker("video_processing", "e2e_worker_1")
            worker2 = queue_service.register_worker("video_processing", "e2e_worker_2")
            
            processed_items = []
            
            # Process all items
            while True:
                # Worker 1 gets item
                item1 = queue_service.get_next_queue_item("video_processing", "e2e_worker_1")
                if item1:
                    processed_items.append(item1)
                    queue_service.complete_queue_item(
                        item1.id,
                        "e2e_worker_1",
                        BatchItemStatus.COMPLETED,
                        result_data={"video_id": f"video_{item1.id}", "summary": "Test summary"}
                    )
                
                # Worker 2 gets item
                item2 = queue_service.get_next_queue_item("video_processing", "e2e_worker_2")
                if item2:
                    processed_items.append(item2)
                    queue_service.complete_queue_item(
                        item2.id,
                        "e2e_worker_2",
                        BatchItemStatus.COMPLETED,
                        result_data={"video_id": f"video_{item2.id}", "summary": "Test summary"}
                    )
                
                # Check if all items processed
                if not item1 and not item2:
                    break
            
            # Step 4: Verify completion
            final_batch = batch_service.get_batch(batch.batch_id)
            assert final_batch.status == BatchStatus.COMPLETED
            assert final_batch.completed_items == 3
            assert final_batch.failed_items == 0
            assert final_batch.completed_at is not None
            
            # Verify all items were processed
            assert len(processed_items) == 3
            
            # Step 5: Verify statistics
            batch_stats = batch_service.get_batch_statistics()
            assert batch_stats["total_batches"] == 1
            assert batch_stats["total_batch_items"] == 3
            assert batch_stats["item_status_counts"]["completed"] == 3
            
            queue_stats = queue_service.get_queue_statistics("video_processing")
            assert queue_stats.completed_items == 3
            assert queue_stats.total_items == 3

    def test_batch_processing_with_failures_and_retries(self, batch_service, queue_service, sample_urls):
        """Test batch processing with failures and retry mechanisms."""
        # Create batch
        request = BatchCreateRequest(
            name="Failure Test Batch",
            urls=sample_urls,
            priority=BatchPriority.HIGH
        )
        
        with patch('src.services.batch_service.extract_video_id_from_url') as mock_extract:
            mock_extract.side_effect = lambda url: url.split('=')[-1] if '=' in url else url.split('/')[-1]
            
            batch = batch_service.create_batch(request)
            batch_service.start_batch_processing(batch.batch_id)
            
            # Register worker
            worker = queue_service.register_worker("video_processing", "failure_worker")
            
            # Process items with mixed success/failure
            processed_count = 0
            retry_attempts = {}
            
            while processed_count < 3:
                item = queue_service.get_next_queue_item("video_processing", "failure_worker")
                if item:
                    retry_count = retry_attempts.get(item.id, 0)
                    
                    # Fail first attempt, succeed on retry
                    if retry_count == 0:
                        # First attempt - fail
                        queue_service.complete_queue_item(
                            item.id,
                            "failure_worker",
                            BatchItemStatus.FAILED,
                            error_message=f"Simulated failure for item {item.id}"
                        )
                        
                        # Retry the item
                        retry_result = queue_service.retry_queue_item(item.id, delay_minutes=0)
                        assert retry_result is True
                        
                        retry_attempts[item.id] = 1
                    else:
                        # Retry attempt - succeed
                        queue_service.complete_queue_item(
                            item.id,
                            "failure_worker",
                            BatchItemStatus.COMPLETED,
                            result_data={"retry_success": True}
                        )
                        processed_count += 1
                else:
                    break
            
            # Verify final state
            final_batch = batch_service.get_batch(batch.batch_id)
            assert final_batch.status == BatchStatus.COMPLETED
            assert final_batch.completed_items == 3
            assert final_batch.failed_items == 0

    @pytest.mark.asyncio
    async def test_concurrent_batch_processing_workflow(self, concurrent_batch_service, sample_urls):
        """Test concurrent batch processing workflow."""
        request = BatchCreateRequest(
            name="Concurrent Test Batch",
            description="Test concurrent processing",
            urls=sample_urls,
            priority=BatchPriority.NORMAL
        )
        
        # Mock dependencies for concurrent service
        with patch('src.services.concurrent_batch_service.get_database_session') as mock_get_session:
            mock_session = Mock()
            mock_get_session.return_value = mock_session
            
            with patch('src.services.concurrent_batch_service.BatchService') as mock_batch_service_class:
                mock_batch_service = Mock()
                mock_batch_service_class.return_value = mock_batch_service
                
                # Mock batch creation
                mock_batch = Mock()
                mock_batch.batch_id = "concurrent_test_batch"
                mock_batch.total_items = 3
                mock_batch.batch_metadata = {}
                mock_batch_service.create_batch.return_value = mock_batch
                
                with patch('src.services.concurrent_batch_service.allocate_resource') as mock_allocate:
                    mock_allocate.return_value.__enter__ = Mock(return_value=None)
                    mock_allocate.return_value.__exit__ = Mock(return_value=None)
                    
                    # Create concurrent batch
                    result = await concurrent_batch_service.create_concurrent_batch(
                        request,
                        ConcurrentBatchMode.PARALLEL,
                        max_concurrent_items=3
                    )
                    
                    assert result == mock_batch
                    assert "concurrent_test_batch" in concurrent_batch_service._active_batches
                    
                    # Test worker management
                    worker = concurrent_batch_service._create_batch_worker(
                        "concurrent_test_batch",
                        "concurrent_worker_1"
                    )
                    
                    assert worker.worker_id == "concurrent_worker_1"
                    assert worker.batch_id == "concurrent_test_batch"
                    
                    # Test batch operations
                    pause_result = concurrent_batch_service.pause_batch_processing("concurrent_test_batch")
                    assert pause_result is True
                    
                    resume_result = concurrent_batch_service.resume_batch_processing("concurrent_test_batch")
                    assert resume_result is True
                    
                    # Get statistics
                    stats = concurrent_batch_service.get_concurrent_statistics()
                    assert 'service_statistics' in stats
                    assert stats['service_statistics']['active_batches'] == 1

    @pytest.mark.asyncio
    async def test_batch_processor_integration_workflow(self, batch_processor, batch_service, sample_urls):
        """Test BatchProcessor integration workflow."""
        # Create batch through BatchService
        request = BatchCreateRequest(
            name="Processor Integration Test",
            urls=sample_urls[:2],  # Use smaller batch for test
            priority=BatchPriority.NORMAL
        )
        
        with patch('src.services.batch_service.extract_video_id_from_url') as mock_extract:
            mock_extract.side_effect = lambda url: url.split('=')[-1] if '=' in url else url.split('/')[-1]
            
            batch = batch_service.create_batch(request)
            
            # Mock database session for processor
            with patch.object(batch_processor, '_get_database_session') as mock_get_session:
                @asyncio.coroutine
                async def mock_session_context():
                    yield batch_service._get_session()
                
                mock_get_session.return_value.__aenter__ = lambda self: mock_session_context().__anext__()
                mock_get_session.return_value.__aexit__ = Mock(return_value=False)
                
                # Mock workflow processing
                with patch.object(batch_processor, '_process_video_workflow') as mock_workflow:
                    mock_workflow.return_value = {
                        'success': True,
                        'workflow_result': {
                            'video_data': {'title': 'Integration Test Video', 'duration': 120},
                            'transcript_data': {'content': 'Test transcript'},
                            'summary_data': {'content': 'Test summary'}
                        },
                        'video_id': 'test123',
                        'url': 'https://www.youtube.com/watch?v=test123'
                    }
                    
                    with patch.object(batch_processor, '_save_video_results') as mock_save:
                        mock_video = Mock()
                        mock_video.id = 123
                        mock_save.return_value = mock_video
                        
                        # Process batch by ID
                        result = await batch_processor.process_batch_by_id(batch.batch_id, num_workers=1)
                        
                        assert result['batch_id'] == batch.batch_id
                        assert result['num_workers_used'] == 1

    # Multi-Component Integration Tests

    def test_full_stack_integration(self, batch_service, queue_service, video_service, sample_urls):
        """Test full stack integration with all components."""
        # Step 1: Create batch with BatchService
        request = BatchCreateRequest(
            name="Full Stack Test",
            description="Test all components together",
            urls=sample_urls,
            priority=BatchPriority.HIGH,
            batch_metadata={"integration_test": True}
        )
        
        with patch('src.services.batch_service.extract_video_id_from_url') as mock_extract:
            mock_extract.side_effect = lambda url: url.split('=')[-1] if '=' in url else url.split('/')[-1]
            
            batch = batch_service.create_batch(request)
            
            # Step 2: Start processing
            batch_service.start_batch_processing(batch.batch_id)
            
            # Step 3: Create processing sessions for tracking
            sessions = []
            for i in range(3):
                queue_item = batch_service.get_next_queue_item(worker_id=f"stack_worker_{i}")
                if queue_item:
                    session = batch_service.create_processing_session(
                        queue_item.batch_item_id,
                        f"stack_worker_{i}"
                    )
                    sessions.append((queue_item, session))
            
            # Step 4: Simulate video processing and saving
            for i, (queue_item, session) in enumerate(sessions):
                # Update progress
                batch_service.update_processing_session(
                    session.session_id,
                    50.0,
                    "Processing video"
                )
                
                # Mock video creation
                with patch.object(video_service, 'create_video_record') as mock_create:
                    mock_video = Mock()
                    mock_video.id = 100 + i
                    mock_create.return_value = mock_video
                    
                    # Create video record
                    video_record = video_service.create_video_record(
                        video_id=f"test{i}",
                        title=f"Test Video {i}",
                        duration=180,
                        url=queue_item.batch_item.url
                    )
                    
                    # Complete processing
                    batch_service.update_processing_session(
                        session.session_id,
                        100.0,
                        "Completed"
                    )
                    
                    # Complete batch item
                    result = BatchItemResult(
                        batch_item_id=queue_item.batch_item_id,
                        status=BatchItemStatus.COMPLETED,
                        video_id=video_record.id,
                        result_data={
                            "video_title": f"Test Video {i}",
                            "processing_time": 60 + i * 10
                        }
                    )
                    
                    batch_service.complete_batch_item(queue_item.batch_item_id, result)
            
            # Step 5: Verify final state
            final_batch = batch_service.get_batch(batch.batch_id)
            assert final_batch.status == BatchStatus.COMPLETED
            assert final_batch.completed_items == 3
            
            # Verify batch progress
            progress = batch_service.get_batch_progress(batch.batch_id)
            assert progress.progress_percentage == 100.0
            assert progress.status == BatchStatus.COMPLETED

    def test_database_transaction_integration(self, batch_service, queue_service, sample_urls):
        """Test database transaction integration across components."""
        # Test transaction rollback scenario
        request = BatchCreateRequest(
            name="Transaction Test",
            urls=sample_urls,
            priority=BatchPriority.NORMAL
        )
        
        with patch('src.services.batch_service.extract_video_id_from_url') as mock_extract:
            mock_extract.side_effect = lambda url: url.split('=')[-1] if '=' in url else url.split('/')[-1]
            
            batch = batch_service.create_batch(request)
            
            # Start processing
            batch_service.start_batch_processing(batch.batch_id)
            
            # Get queue item
            worker = queue_service.register_worker("video_processing", "transaction_worker")
            queue_item = queue_service.get_next_queue_item("video_processing", "transaction_worker")
            
            assert queue_item is not None
            
            # Verify transactional consistency
            # The queue item should be locked in the database
            session = batch_service._get_session()
            db_queue_item = session.get(QueueItem, queue_item.id)
            assert db_queue_item.locked_by == "transaction_worker"
            assert db_queue_item.locked_at is not None
            
            # Complete the item
            queue_service.complete_queue_item(
                queue_item.id,
                "transaction_worker",
                BatchItemStatus.COMPLETED
            )
            
            # Verify transaction was committed
            session.refresh(db_queue_item.batch_item)
            assert db_queue_item.batch_item.status == BatchItemStatus.COMPLETED

    # Performance Integration Tests

    def test_performance_integration_large_batch(self, batch_service, queue_service, large_url_list):
        """Test performance integration with large batch."""
        start_time = time.time()
        
        # Create large batch
        request = BatchCreateRequest(
            name="Performance Test Batch",
            description="Large batch for performance testing",
            urls=large_url_list,
            priority=BatchPriority.NORMAL
        )
        
        with patch('src.services.batch_service.extract_video_id_from_url') as mock_extract:
            mock_extract.side_effect = lambda url: url.split('=')[-1] if '=' in url else url.split('/')[-1]
            
            batch = batch_service.create_batch(request)
            creation_time = time.time() - start_time
            
            # Should create batch efficiently
            assert creation_time < 5.0
            assert batch.total_items == 50
            
            # Start processing
            start_processing_time = time.time()
            batch_service.start_batch_processing(batch.batch_id)
            
            # Register multiple workers for parallel processing
            workers = []
            for i in range(5):
                worker = queue_service.register_worker("video_processing", f"perf_worker_{i}")
                workers.append(worker)
            
            # Process items in parallel batches
            processed_count = 0
            batch_size = 10
            
            while processed_count < 50:
                # Get batch of items
                current_batch = []
                for worker in workers:
                    if processed_count >= 50:
                        break
                    item = queue_service.get_next_queue_item("video_processing", worker.worker_id)
                    if item:
                        current_batch.append((item, worker.worker_id))
                
                # Process current batch
                for item, worker_id in current_batch:
                    queue_service.complete_queue_item(
                        item.id,
                        worker_id,
                        BatchItemStatus.COMPLETED,
                        result_data={"processed_at": time.time()}
                    )
                    processed_count += 1
                
                if not current_batch:
                    break
            
            processing_time = time.time() - start_processing_time
            
            # Verify performance
            assert processing_time < 30.0  # Should process 50 items within 30 seconds
            
            # Verify final state
            final_batch = batch_service.get_batch(batch.batch_id)
            assert final_batch.status == BatchStatus.COMPLETED
            assert final_batch.completed_items == 50

    def test_concurrent_processing_performance(self, batch_service, queue_service, large_url_list):
        """Test concurrent processing performance."""
        # Create multiple batches for concurrent processing
        batches = []
        batch_count = 3
        
        with patch('src.services.batch_service.extract_video_id_from_url') as mock_extract:
            mock_extract.side_effect = lambda url: url.split('=')[-1] if '=' in url else url.split('/')[-1]
            
            # Create multiple batches
            for i in range(batch_count):
                request = BatchCreateRequest(
                    name=f"Concurrent Batch {i}",
                    urls=large_url_list[:10],  # 10 items per batch
                    priority=BatchPriority.NORMAL
                )
                
                batch = batch_service.create_batch(request)
                batch_service.start_batch_processing(batch.batch_id)
                batches.append(batch)
            
            # Process batches concurrently using multiple workers
            def process_batch_worker(worker_id, target_items):
                processed = 0
                while processed < target_items:
                    item = queue_service.get_next_queue_item("video_processing", worker_id)
                    if item:
                        queue_service.complete_queue_item(
                            item.id,
                            worker_id,
                            BatchItemStatus.COMPLETED
                        )
                        processed += 1
                    else:
                        break
                return processed
            
            # Start concurrent workers
            import threading
            threads = []
            results = {}
            
            for i in range(6):  # 6 workers for 30 total items
                worker_id = f"concurrent_worker_{i}"
                queue_service.register_worker("video_processing", worker_id)
                
                def worker_thread(wid=worker_id):
                    results[wid] = process_batch_worker(wid, 10)
                
                thread = threading.Thread(target=worker_thread)
                threads.append(thread)
                thread.start()
            
            # Wait for all threads to complete
            for thread in threads:
                thread.join()
            
            # Verify all items were processed
            total_processed = sum(results.values())
            assert total_processed == 30  # 3 batches * 10 items each
            
            # Verify all batches completed
            for batch in batches:
                final_batch = batch_service.get_batch(batch.batch_id)
                assert final_batch.status == BatchStatus.COMPLETED
                assert final_batch.completed_items == 10

    # Error Handling Integration Tests

    def test_error_recovery_integration(self, batch_service, queue_service, sample_urls):
        """Test error recovery integration across components."""
        # Create batch
        request = BatchCreateRequest(
            name="Error Recovery Test",
            urls=sample_urls,
            priority=BatchPriority.NORMAL
        )
        
        with patch('src.services.batch_service.extract_video_id_from_url') as mock_extract:
            mock_extract.side_effect = lambda url: url.split('=')[-1] if '=' in url else url.split('/')[-1]
            
            batch = batch_service.create_batch(request)
            batch_service.start_batch_processing(batch.batch_id)
            
            # Register worker
            worker = queue_service.register_worker("video_processing", "error_worker")
            
            # Simulate various error scenarios
            error_scenarios = [
                ("Network timeout", False),  # Retryable
                ("Invalid video", True),     # Non-retryable
                ("Processing error", False)  # Retryable
            ]
            
            for i, (error_msg, permanent) in enumerate(error_scenarios):
                queue_item = queue_service.get_next_queue_item("video_processing", "error_worker")
                if queue_item:
                    # Create processing session
                    session = batch_service.create_processing_session(
                        queue_item.batch_item_id,
                        "error_worker"
                    )
                    
                    # Simulate error during processing
                    batch_service.update_processing_session(
                        session.session_id,
                        30.0,
                        f"Error: {error_msg}"
                    )
                    
                    if permanent:
                        # Mark as permanently failed
                        queue_service.complete_queue_item(
                            queue_item.id,
                            "error_worker",
                            BatchItemStatus.FAILED,
                            error_message=error_msg
                        )
                    else:
                        # Mark as failed but retryable
                        queue_service.complete_queue_item(
                            queue_item.id,
                            "error_worker",
                            BatchItemStatus.FAILED,
                            error_message=error_msg
                        )
                        
                        # Retry the item
                        retry_result = queue_service.retry_queue_item(queue_item.id, delay_minutes=0)
                        assert retry_result is True
                        
                        # Process retry successfully
                        retry_item = queue_service.get_next_queue_item("video_processing", "error_worker")
                        if retry_item and retry_item.id == queue_item.id:
                            queue_service.complete_queue_item(
                                retry_item.id,
                                "error_worker",
                                BatchItemStatus.COMPLETED,
                                result_data={"retry_success": True}
                            )
            
            # Verify final state
            final_batch = batch_service.get_batch(batch.batch_id)
            assert final_batch.completed_items == 2  # 2 successful (including retries)
            assert final_batch.failed_items == 1     # 1 permanent failure

    def test_cleanup_integration(self, batch_service, queue_service, sample_urls):
        """Test cleanup integration across components."""
        # Create batch with stale sessions
        request = BatchCreateRequest(
            name="Cleanup Test",
            urls=sample_urls,
            priority=BatchPriority.NORMAL
        )
        
        with patch('src.services.batch_service.extract_video_id_from_url') as mock_extract:
            mock_extract.side_effect = lambda url: url.split('=')[-1] if '=' in url else url.split('/')[-1]
            
            batch = batch_service.create_batch(request)
            batch_service.start_batch_processing(batch.batch_id)
            
            # Create workers and sessions
            workers = []
            sessions = []
            
            for i in range(3):
                worker_id = f"cleanup_worker_{i}"
                worker = queue_service.register_worker("video_processing", worker_id)
                workers.append(worker)
                
                queue_item = queue_service.get_next_queue_item("video_processing", worker_id)
                if queue_item:
                    session = batch_service.create_processing_session(
                        queue_item.batch_item_id,
                        worker_id
                    )
                    sessions.append((queue_item, session))
            
            # Make sessions stale
            db_session = batch_service._get_session()
            for queue_item, session in sessions:
                processing_session = db_session.get(ProcessingSession, session.id)
                processing_session.heartbeat_at = datetime.utcnow() - timedelta(hours=1)
                
                # Make queue locks stale
                queue_item.lock_expires_at = datetime.utcnow() - timedelta(minutes=30)
            
            db_session.commit()
            
            # Run cleanup operations
            batch_cleanup_count = batch_service.cleanup_stale_sessions(timeout_minutes=30)
            queue_cleanup_count = queue_service._cleanup_stale_locks()
            
            # Verify cleanup
            assert batch_cleanup_count == 3
            assert queue_cleanup_count == 3
            
            # Verify items are available for processing again
            new_worker = queue_service.register_worker("video_processing", "cleanup_new_worker")
            recovered_item = queue_service.get_next_queue_item("video_processing", "cleanup_new_worker")
            assert recovered_item is not None

    # API Integration Tests (Mock)

    def test_api_integration_simulation(self, batch_service, sample_urls):
        """Test API integration simulation."""
        # Simulate API request to create batch
        api_request_data = {
            "name": "API Test Batch",
            "description": "Test batch created via API",
            "urls": sample_urls,
            "priority": "HIGH",
            "metadata": {"source": "api", "user_id": "test_user"}
        }
        
        # Convert API data to service request
        request = BatchCreateRequest(
            name=api_request_data["name"],
            description=api_request_data["description"],
            urls=api_request_data["urls"],
            priority=BatchPriority.HIGH,
            batch_metadata=api_request_data["metadata"]
        )
        
        with patch('src.services.batch_service.extract_video_id_from_url') as mock_extract:
            mock_extract.side_effect = lambda url: url.split('=')[-1] if '=' in url else url.split('/')[-1]
            
            # Create batch
            batch = batch_service.create_batch(request)
            
            # Simulate API response
            api_response = {
                "batch_id": batch.batch_id,
                "name": batch.name,
                "status": batch.status.value,
                "total_items": batch.total_items,
                "created_at": batch.created_at.isoformat(),
                "metadata": batch.batch_metadata
            }
            
            # Verify API response structure
            assert api_response["batch_id"] == batch.batch_id
            assert api_response["name"] == "API Test Batch"
            assert api_response["status"] == "PENDING"
            assert api_response["total_items"] == 3
            assert api_response["metadata"]["source"] == "api"
            
            # Simulate API status check
            progress = batch_service.get_batch_progress(batch.batch_id)
            
            status_response = {
                "batch_id": progress.batch_id,
                "status": progress.status.value,
                "progress_percentage": progress.progress_percentage,
                "total_items": progress.total_items,
                "completed_items": progress.completed_items,
                "failed_items": progress.failed_items,
                "pending_items": progress.pending_items
            }
            
            assert status_response["status"] == "PENDING"
            assert status_response["progress_percentage"] == 0.0

    # Resource Management Integration Tests

    def test_resource_management_integration(self, batch_service, queue_service, large_url_list):
        """Test resource management integration."""
        # Create multiple batches to test resource limits
        batches = []
        max_batches = 3
        
        with patch('src.services.batch_service.extract_video_id_from_url') as mock_extract:
            mock_extract.side_effect = lambda url: url.split('=')[-1] if '=' in url else url.split('/')[-1]
            
            # Create batches up to limit
            for i in range(max_batches):
                request = BatchCreateRequest(
                    name=f"Resource Test Batch {i}",
                    urls=large_url_list[:5],  # 5 items per batch
                    priority=BatchPriority.NORMAL
                )
                
                batch = batch_service.create_batch(request)
                batches.append(batch)
            
            # Register workers with limits
            max_workers_per_queue = 10
            workers = []
            
            for i in range(max_workers_per_queue):
                worker = queue_service.register_worker("video_processing", f"resource_worker_{i}")
                workers.append(worker)
            
            # Test that workers are properly distributed
            worker_stats = queue_service.get_worker_statistics()
            assert worker_stats["total_workers"] == max_workers_per_queue
            assert worker_stats["active_workers"] == max_workers_per_queue
            
            # Start processing with resource monitoring
            for batch in batches:
                batch_service.start_batch_processing(batch.batch_id)
            
            # Monitor resource usage during processing
            initial_stats = batch_service.get_batch_statistics()
            assert initial_stats["batch_status_counts"]["processing"] == max_batches
            
            # Process a few items to verify resource management
            processed_items = 0
            target_items = 5  # Process 5 items total
            
            while processed_items < target_items:
                for worker in workers[:3]:  # Use only 3 workers
                    if processed_items >= target_items:
                        break
                    
                    item = queue_service.get_next_queue_item("video_processing", worker.worker_id)
                    if item:
                        queue_service.complete_queue_item(
                            item.id,
                            worker.worker_id,
                            BatchItemStatus.COMPLETED
                        )
                        processed_items += 1
            
            # Verify resource usage
            final_stats = batch_service.get_batch_statistics()
            assert final_stats["item_status_counts"]["completed"] == target_items

    # Edge Cases Integration Tests

    def test_edge_cases_integration(self, batch_service, queue_service):
        """Test edge cases integration."""
        # Test empty batch handling
        empty_request = BatchCreateRequest(
            name="Empty Batch Test",
            urls=[],
            priority=BatchPriority.LOW
        )
        
        try:
            batch_service.create_batch(empty_request)
            assert False, "Should have raised error for empty URLs"
        except Exception as e:
            assert "cannot be empty" in str(e)
        
        # Test single item batch
        single_url = ["https://www.youtube.com/watch?v=single"]
        
        with patch('src.services.batch_service.extract_video_id_from_url') as mock_extract:
            mock_extract.return_value = "single"
            
            single_request = BatchCreateRequest(
                name="Single Item Batch",
                urls=single_url,
                priority=BatchPriority.NORMAL
            )
            
            batch = batch_service.create_batch(single_request)
            assert batch.total_items == 1
            
            # Process single item
            batch_service.start_batch_processing(batch.batch_id)
            worker = queue_service.register_worker("video_processing", "single_worker")
            
            item = queue_service.get_next_queue_item("video_processing", "single_worker")
            assert item is not None
            
            queue_service.complete_queue_item(
                item.id,
                "single_worker",
                BatchItemStatus.COMPLETED
            )
            
            # Verify completion
            final_batch = batch_service.get_batch(batch.batch_id)
            assert final_batch.status == BatchStatus.COMPLETED
            assert final_batch.completed_items == 1
        
        # Test no workers scenario
        no_worker_request = BatchCreateRequest(
            name="No Worker Test",
            urls=["https://www.youtube.com/watch?v=noworker"],
            priority=BatchPriority.NORMAL
        )
        
        with patch('src.services.batch_service.extract_video_id_from_url') as mock_extract:
            mock_extract.return_value = "noworker"
            
            no_worker_batch = batch_service.create_batch(no_worker_request)
            batch_service.start_batch_processing(no_worker_batch.batch_id)
            
            # Try to get item without registered worker
            item = queue_service.get_next_queue_item("video_processing", "unregistered_worker")
            # Should handle gracefully (might return None or raise error depending on implementation)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])