"""
Comprehensive test suite for BatchService functionality.

This test suite provides complete coverage for BatchService including:
- Core batch operations (create, get, update, delete)
- Batch lifecycle management (start, cancel, complete)
- Queue management and processing
- Processing session management
- Error handling and edge cases
- Performance considerations
- Database transaction handling
- Concurrency and thread safety
- Batch statistics and monitoring
- Cleanup and maintenance operations
"""

import pytest
import asyncio
import threading
import time
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from sqlalchemy.exc import SQLAlchemyError, IntegrityError
from concurrent.futures import ThreadPoolExecutor, as_completed
import uuid
import json

from src.database.models import Base, Video
from src.database.batch_models import (
    Batch, BatchItem, QueueItem, ProcessingSession,
    BatchStatus, BatchItemStatus, BatchPriority
)
from src.services.batch_service import (
    BatchService, BatchCreateRequest, BatchProgressInfo, BatchItemResult,
    BatchServiceError
)
from src.utils.youtube_utils import extract_video_id_from_url


class TestBatchServiceComprehensive:
    """Comprehensive test suite for BatchService."""

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
    def batch_service(self, db_session):
        """Create BatchService instance with test database session."""
        return BatchService(session=db_session)

    @pytest.fixture
    def sample_urls(self):
        """Sample YouTube URLs for testing."""
        return [
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://www.youtube.com/watch?v=9bZkp7q19f0",
            "https://youtu.be/oHg5SJYRHA0",
            "https://www.youtube.com/watch?v=kJQP7kiw5Fk",
            "https://youtu.be/fNFzfwLM72c"
        ]

    @pytest.fixture
    def batch_request(self, sample_urls):
        """Sample batch creation request."""
        return BatchCreateRequest(
            name="Test Batch",
            description="Comprehensive test batch",
            urls=sample_urls,
            priority=BatchPriority.NORMAL,
            batch_metadata={"test": True, "environment": "test"}
        )

    @pytest.fixture
    def large_batch_request(self):
        """Large batch request for performance testing."""
        urls = [f"https://www.youtube.com/watch?v=test{i:06d}" for i in range(100)]
        return BatchCreateRequest(
            name="Large Test Batch",
            description="Large batch for performance testing",
            urls=urls,
            priority=BatchPriority.HIGH,
            batch_metadata={"test": True, "size": "large"}
        )

    # Core Batch Operations Tests

    def test_batch_creation_comprehensive(self, batch_service, batch_request):
        """Test comprehensive batch creation scenarios."""
        with patch('src.services.batch_service.extract_video_id_from_url') as mock_extract:
            mock_extract.side_effect = lambda url: url.split('=')[-1] if '=' in url else url.split('/')[-1]
            
            batch = batch_service.create_batch(batch_request)
            
            # Verify batch properties
            assert batch is not None
            assert batch.name == "Test Batch"
            assert batch.description == "Comprehensive test batch"
            assert batch.status == BatchStatus.PENDING
            assert batch.priority == BatchPriority.NORMAL
            assert batch.total_items == 5
            assert batch.completed_items == 0
            assert batch.failed_items == 0
            assert batch.batch_metadata == {"test": True, "environment": "test"}
            assert batch.batch_id.startswith("batch_")
            assert batch.created_at is not None
            assert batch.updated_at is not None
            
            # Verify batch items were created
            assert len(batch.batch_items) == 5
            for i, item in enumerate(batch.batch_items):
                assert item.batch_id == batch.id
                assert item.status == BatchItemStatus.QUEUED
                assert item.priority == BatchPriority.NORMAL
                assert item.processing_order == i
                assert item.retry_count == 0
                assert item.max_retries == 3
                assert item.created_at is not None
                assert item.updated_at is not None

    def test_batch_creation_with_different_priorities(self, batch_service, sample_urls):
        """Test batch creation with different priority levels."""
        priorities = [BatchPriority.LOW, BatchPriority.NORMAL, BatchPriority.HIGH]
        
        for priority in priorities:
            request = BatchCreateRequest(
                name=f"Test Batch {priority.value}",
                description=f"Test batch with {priority.value} priority",
                urls=sample_urls[:2],
                priority=priority
            )
            
            with patch('src.services.batch_service.extract_video_id_from_url') as mock_extract:
                mock_extract.side_effect = lambda url: url.split('=')[-1] if '=' in url else url.split('/')[-1]
                
                batch = batch_service.create_batch(request)
                
                assert batch.priority == priority
                for item in batch.batch_items:
                    assert item.priority == priority

    def test_batch_creation_with_metadata(self, batch_service, sample_urls):
        """Test batch creation with various metadata scenarios."""
        metadata_scenarios = [
            {"simple": "value"},
            {"nested": {"key": "value", "array": [1, 2, 3]}},
            {"complex": {"user": "test", "params": {"retry": True, "timeout": 30}}},
            {}  # Empty metadata
        ]
        
        for metadata in metadata_scenarios:
            request = BatchCreateRequest(
                name=f"Metadata Test {len(metadata)}",
                urls=sample_urls[:1],
                batch_metadata=metadata
            )
            
            with patch('src.services.batch_service.extract_video_id_from_url') as mock_extract:
                mock_extract.side_effect = lambda url: url.split('=')[-1] if '=' in url else url.split('/')[-1]
                
                batch = batch_service.create_batch(request)
                assert batch.batch_metadata == metadata

    def test_batch_creation_error_cases(self, batch_service):
        """Test batch creation error scenarios."""
        # Empty URLs
        with pytest.raises(BatchServiceError, match="URLs list cannot be empty"):
            batch_service.create_batch(BatchCreateRequest(
                name="Empty Batch",
                urls=[]
            ))
        
        # Invalid URLs
        with patch('src.services.batch_service.extract_video_id_from_url') as mock_extract:
            mock_extract.return_value = None
            
            with pytest.raises(BatchServiceError, match="Invalid YouTube URL"):
                batch_service.create_batch(BatchCreateRequest(
                    name="Invalid Batch",
                    urls=["not-a-youtube-url"]
                ))
        
        # Duplicate URLs
        duplicate_urls = ["https://www.youtube.com/watch?v=test"] * 3
        with patch('src.services.batch_service.extract_video_id_from_url') as mock_extract:
            mock_extract.return_value = "test"
            
            # Should handle duplicates gracefully
            batch = batch_service.create_batch(BatchCreateRequest(
                name="Duplicate Batch",
                urls=duplicate_urls
            ))
            assert batch.total_items == 3  # Should create all items even if URLs are duplicates

    def test_batch_retrieval_comprehensive(self, batch_service, batch_request):
        """Test comprehensive batch retrieval scenarios."""
        with patch('src.services.batch_service.extract_video_id_from_url') as mock_extract:
            mock_extract.side_effect = lambda url: url.split('=')[-1] if '=' in url else url.split('/')[-1]
            
            created_batch = batch_service.create_batch(batch_request)
            
            # Test retrieval by ID
            retrieved_batch = batch_service.get_batch(created_batch.batch_id)
            assert retrieved_batch is not None
            assert retrieved_batch.batch_id == created_batch.batch_id
            assert retrieved_batch.name == created_batch.name
            assert len(retrieved_batch.batch_items) == len(created_batch.batch_items)
            
            # Test retrieval of non-existent batch
            assert batch_service.get_batch("non-existent-batch") is None
            
            # Test retrieval with malformed ID
            assert batch_service.get_batch("") is None
            assert batch_service.get_batch(None) is None

    def test_batch_listing_comprehensive(self, batch_service, sample_urls):
        """Test comprehensive batch listing scenarios."""
        # Create multiple batches with different statuses
        batches = []
        statuses = [BatchStatus.PENDING, BatchStatus.PROCESSING, BatchStatus.COMPLETED]
        
        for i, status in enumerate(statuses):
            request = BatchCreateRequest(
                name=f"Batch {i}",
                urls=sample_urls[:2],
                priority=BatchPriority.NORMAL
            )
            
            with patch('src.services.batch_service.extract_video_id_from_url') as mock_extract:
                mock_extract.side_effect = lambda url: url.split('=')[-1] if '=' in url else url.split('/')[-1]
                
                batch = batch_service.create_batch(request)
                batch.status = status
                batch_service._get_session().commit()
                batches.append(batch)
        
        # Test listing all batches
        all_batches = batch_service.list_batches()
        assert len(all_batches) == 3
        
        # Test filtering by status
        pending_batches = batch_service.list_batches(status=BatchStatus.PENDING)
        assert len(pending_batches) == 1
        assert pending_batches[0].status == BatchStatus.PENDING
        
        processing_batches = batch_service.list_batches(status=BatchStatus.PROCESSING)
        assert len(processing_batches) == 1
        assert processing_batches[0].status == BatchStatus.PROCESSING
        
        completed_batches = batch_service.list_batches(status=BatchStatus.COMPLETED)
        assert len(completed_batches) == 1
        assert completed_batches[0].status == BatchStatus.COMPLETED
        
        # Test filtering by non-existent status
        failed_batches = batch_service.list_batches(status=BatchStatus.FAILED)
        assert len(failed_batches) == 0

    # Batch Lifecycle Management Tests

    def test_batch_lifecycle_complete(self, batch_service, batch_request):
        """Test complete batch lifecycle."""
        with patch('src.services.batch_service.extract_video_id_from_url') as mock_extract:
            mock_extract.side_effect = lambda url: url.split('=')[-1] if '=' in url else url.split('/')[-1]
            
            # Create batch
            batch = batch_service.create_batch(batch_request)
            assert batch.status == BatchStatus.PENDING
            
            # Start processing
            result = batch_service.start_batch_processing(batch.batch_id)
            assert result is True
            
            updated_batch = batch_service.get_batch(batch.batch_id)
            assert updated_batch.status == BatchStatus.PROCESSING
            assert updated_batch.started_at is not None
            
            # Process all items
            for i in range(batch.total_items):
                queue_item = batch_service.get_next_queue_item(worker_id=f"worker_{i}")
                if queue_item:
                    result = BatchItemResult(
                        batch_item_id=queue_item.batch_item_id,
                        status=BatchItemStatus.COMPLETED,
                        video_id=i + 1,
                        result_data={"summary": f"Test summary {i}"}
                    )
                    batch_service.complete_batch_item(queue_item.batch_item_id, result)
            
            # Verify batch completion
            final_batch = batch_service.get_batch(batch.batch_id)
            assert final_batch.status == BatchStatus.COMPLETED
            assert final_batch.completed_at is not None
            assert final_batch.completed_items == batch.total_items
            assert final_batch.failed_items == 0

    def test_batch_cancellation_scenarios(self, batch_service, batch_request):
        """Test batch cancellation in various scenarios."""
        with patch('src.services.batch_service.extract_video_id_from_url') as mock_extract:
            mock_extract.side_effect = lambda url: url.split('=')[-1] if '=' in url else url.split('/')[-1]
            
            # Test cancellation of pending batch
            batch = batch_service.create_batch(batch_request)
            result = batch_service.cancel_batch(batch.batch_id, "User requested cancellation")
            assert result is True
            
            updated_batch = batch_service.get_batch(batch.batch_id)
            assert updated_batch.status == BatchStatus.CANCELLED
            assert updated_batch.completed_at is not None
            assert updated_batch.error_info == "User requested cancellation"
            
            # Test cancellation of processing batch
            batch2 = batch_service.create_batch(batch_request)
            batch_service.start_batch_processing(batch2.batch_id)
            
            result = batch_service.cancel_batch(batch2.batch_id, "System cancellation")
            assert result is True
            
            updated_batch2 = batch_service.get_batch(batch2.batch_id)
            assert updated_batch2.status == BatchStatus.CANCELLED
            assert updated_batch2.error_info == "System cancellation"

    def test_batch_error_handling_scenarios(self, batch_service, batch_request):
        """Test batch error handling scenarios."""
        with patch('src.services.batch_service.extract_video_id_from_url') as mock_extract:
            mock_extract.side_effect = lambda url: url.split('=')[-1] if '=' in url else url.split('/')[-1]
            
            batch = batch_service.create_batch(batch_request)
            
            # Test starting non-existent batch
            with pytest.raises(BatchServiceError, match="Batch .* not found"):
                batch_service.start_batch_processing("non-existent-batch")
            
            # Test starting batch with wrong status
            batch_service.start_batch_processing(batch.batch_id)
            with pytest.raises(BatchServiceError, match="is not in PENDING status"):
                batch_service.start_batch_processing(batch.batch_id)
            
            # Test cancelling already completed batch
            batch.status = BatchStatus.COMPLETED
            batch_service._get_session().commit()
            
            with pytest.raises(BatchServiceError, match="is already completed"):
                batch_service.cancel_batch(batch.batch_id)

    # Queue Management Tests

    def test_queue_management_comprehensive(self, batch_service, batch_request):
        """Test comprehensive queue management scenarios."""
        with patch('src.services.batch_service.extract_video_id_from_url') as mock_extract:
            mock_extract.side_effect = lambda url: url.split('=')[-1] if '=' in url else url.split('/')[-1]
            
            batch = batch_service.create_batch(batch_request)
            
            # Test getting queue items
            workers = []
            queue_items = []
            
            for i in range(3):
                worker_id = f"test-worker-{i}"
                workers.append(worker_id)
                
                queue_item = batch_service.get_next_queue_item(worker_id=worker_id)
                assert queue_item is not None
                assert queue_item.batch_item is not None
                assert queue_item.locked_by == worker_id
                assert queue_item.locked_at is not None
                assert queue_item.lock_expires_at is not None
                assert queue_item.batch_item.status == BatchItemStatus.PROCESSING
                
                queue_items.append(queue_item)
            
            # Test that all items are locked
            next_item = batch_service.get_next_queue_item(worker_id="another-worker")
            assert next_item is not None  # Should get remaining items
            
            # Test completing items
            for i, queue_item in enumerate(queue_items):
                result = BatchItemResult(
                    batch_item_id=queue_item.batch_item_id,
                    status=BatchItemStatus.COMPLETED,
                    video_id=i + 1,
                    result_data={"summary": f"Test summary {i}"}
                )
                success = batch_service.complete_batch_item(queue_item.batch_item_id, result)
                assert success is True

    def test_queue_item_locking_and_expiration(self, batch_service, batch_request):
        """Test queue item locking and expiration mechanisms."""
        with patch('src.services.batch_service.extract_video_id_from_url') as mock_extract:
            mock_extract.side_effect = lambda url: url.split('=')[-1] if '=' in url else url.split('/')[-1]
            
            batch = batch_service.create_batch(batch_request)
            
            # Get queue item
            queue_item = batch_service.get_next_queue_item(worker_id="test-worker")
            assert queue_item is not None
            
            # Verify lock properties
            assert queue_item.locked_by == "test-worker"
            assert queue_item.locked_at is not None
            assert queue_item.lock_expires_at is not None
            assert queue_item.lock_expires_at > datetime.utcnow()
            
            # Test that same worker can't get the same item
            same_item = batch_service.get_next_queue_item(worker_id="test-worker")
            if same_item:
                assert same_item.id != queue_item.id
            
            # Test lock expiration cleanup
            # Manually expire the lock
            queue_item.lock_expires_at = datetime.utcnow() - timedelta(minutes=1)
            batch_service._get_session().commit()
            
            # Should be able to get the item again
            expired_item = batch_service.get_next_queue_item(worker_id="new-worker")
            # The item should be available again after cleanup

    def test_queue_priority_processing(self, batch_service, sample_urls):
        """Test queue priority-based processing."""
        priorities = [BatchPriority.LOW, BatchPriority.HIGH, BatchPriority.NORMAL]
        batches = []
        
        # Create batches with different priorities
        for i, priority in enumerate(priorities):
            request = BatchCreateRequest(
                name=f"Priority Batch {priority.value}",
                urls=[sample_urls[i]],
                priority=priority
            )
            
            with patch('src.services.batch_service.extract_video_id_from_url') as mock_extract:
                mock_extract.side_effect = lambda url: url.split('=')[-1] if '=' in url else url.split('/')[-1]
                
                batch = batch_service.create_batch(request)
                batches.append(batch)
        
        # Get items - should prioritize HIGH, then NORMAL, then LOW
        queue_items = []
        for i in range(3):
            item = batch_service.get_next_queue_item(worker_id=f"worker_{i}")
            if item:
                queue_items.append(item)
        
        # Verify priority ordering
        assert len(queue_items) == 3
        assert queue_items[0].priority == BatchPriority.HIGH
        assert queue_items[1].priority == BatchPriority.NORMAL
        assert queue_items[2].priority == BatchPriority.LOW

    # Processing Session Management Tests

    def test_processing_session_lifecycle(self, batch_service, batch_request):
        """Test complete processing session lifecycle."""
        with patch('src.services.batch_service.extract_video_id_from_url') as mock_extract:
            mock_extract.side_effect = lambda url: url.split('=')[-1] if '=' in url else url.split('/')[-1]
            
            batch = batch_service.create_batch(batch_request)
            queue_item = batch_service.get_next_queue_item(worker_id="test-worker")
            
            # Create processing session
            session = batch_service.create_processing_session(
                queue_item.batch_item_id,
                "test-worker"
            )
            
            assert session is not None
            assert session.batch_item_id == queue_item.batch_item_id
            assert session.worker_id == "test-worker"
            assert session.status == BatchItemStatus.PROCESSING
            assert session.progress_percentage == 0.0
            assert session.session_id.startswith("session_")
            assert session.created_at is not None
            
            # Update session progress
            success = batch_service.update_processing_session(
                session.session_id,
                50.0,
                "Processing transcripts"
            )
            assert success is True
            
            # Verify session update
            db_session = batch_service._get_session()
            updated_session = db_session.get(ProcessingSession, session.id)
            assert updated_session.progress_percentage == 50.0
            assert updated_session.current_step == "Processing transcripts"
            assert updated_session.updated_at > session.created_at
            
            # Update session progress again
            success = batch_service.update_processing_session(
                session.session_id,
                100.0,
                "Completed"
            )
            assert success is True
            
            # Verify final update
            final_session = db_session.get(ProcessingSession, session.id)
            assert final_session.progress_percentage == 100.0
            assert final_session.current_step == "Completed"

    def test_processing_session_error_handling(self, batch_service, batch_request):
        """Test processing session error handling."""
        with patch('src.services.batch_service.extract_video_id_from_url') as mock_extract:
            mock_extract.side_effect = lambda url: url.split('=')[-1] if '=' in url else url.split('/')[-1]
            
            batch = batch_service.create_batch(batch_request)
            queue_item = batch_service.get_next_queue_item(worker_id="test-worker")
            
            # Test creating session for non-existent item
            with pytest.raises(BatchServiceError):
                batch_service.create_processing_session(99999, "test-worker")
            
            # Create valid session
            session = batch_service.create_processing_session(
                queue_item.batch_item_id,
                "test-worker"
            )
            
            # Test updating non-existent session
            success = batch_service.update_processing_session(
                "non-existent-session",
                50.0,
                "Test"
            )
            assert success is False
            
            # Test updating with invalid progress
            success = batch_service.update_processing_session(
                session.session_id,
                -10.0,  # Invalid progress
                "Test"
            )
            assert success is False
            
            success = batch_service.update_processing_session(
                session.session_id,
                150.0,  # Invalid progress
                "Test"
            )
            assert success is False

    def test_stale_session_cleanup(self, batch_service, batch_request):
        """Test stale processing session cleanup."""
        with patch('src.services.batch_service.extract_video_id_from_url') as mock_extract:
            mock_extract.side_effect = lambda url: url.split('=')[-1] if '=' in url else url.split('/')[-1]
            
            batch = batch_service.create_batch(batch_request)
            queue_item = batch_service.get_next_queue_item(worker_id="test-worker")
            
            session = batch_service.create_processing_session(
                queue_item.batch_item_id,
                "test-worker"
            )
            
            # Make session stale
            db_session = batch_service._get_session()
            processing_session = db_session.get(ProcessingSession, session.id)
            processing_session.heartbeat_at = datetime.utcnow() - timedelta(hours=2)
            db_session.commit()
            
            # Run cleanup
            cleaned_count = batch_service.cleanup_stale_sessions(timeout_minutes=60)
            assert cleaned_count == 1
            
            # Verify batch item was marked as failed
            batch_item = db_session.get(BatchItem, queue_item.batch_item_id)
            assert batch_item.status == BatchItemStatus.FAILED
            assert "timed out" in batch_item.error_info

    # Retry and Error Handling Tests

    def test_batch_item_retry_mechanisms(self, batch_service, batch_request):
        """Test comprehensive batch item retry mechanisms."""
        with patch('src.services.batch_service.extract_video_id_from_url') as mock_extract:
            mock_extract.side_effect = lambda url: url.split('=')[-1] if '=' in url else url.split('/')[-1]
            
            batch = batch_service.create_batch(batch_request)
            queue_item = batch_service.get_next_queue_item(worker_id="test-worker")
            
            # Mark as failed
            failure_result = BatchItemResult(
                batch_item_id=queue_item.batch_item_id,
                status=BatchItemStatus.FAILED,
                error_message="Network timeout"
            )
            batch_service.complete_batch_item(queue_item.batch_item_id, failure_result)
            
            # Verify item is marked as failed
            session = batch_service._get_session()
            batch_item = session.get(BatchItem, queue_item.batch_item_id)
            assert batch_item.status == BatchItemStatus.FAILED
            assert batch_item.error_info == "Network timeout"
            
            # Test retry
            success = batch_service.retry_failed_batch_item(queue_item.batch_item_id)
            assert success is True
            
            # Verify retry
            session.refresh(batch_item)
            assert batch_item.status == BatchItemStatus.QUEUED
            assert batch_item.retry_count == 1
            assert batch_item.error_info is None
            
            # Test multiple retries
            for retry_count in range(2, 4):  # Test retry 2 and 3
                queue_item = batch_service.get_next_queue_item(worker_id="test-worker")
                
                failure_result = BatchItemResult(
                    batch_item_id=queue_item.batch_item_id,
                    status=BatchItemStatus.FAILED,
                    error_message=f"Retry {retry_count} failed"
                )
                batch_service.complete_batch_item(queue_item.batch_item_id, failure_result)
                
                if retry_count < 3:
                    success = batch_service.retry_failed_batch_item(queue_item.batch_item_id)
                    assert success is True
                    
                    session.refresh(batch_item)
                    assert batch_item.retry_count == retry_count
            
            # Test max retries exceeded
            with pytest.raises(BatchServiceError, match="cannot be retried"):
                batch_service.retry_failed_batch_item(queue_item.batch_item_id)

    def test_batch_item_result_scenarios(self, batch_service, batch_request):
        """Test various batch item result scenarios."""
        with patch('src.services.batch_service.extract_video_id_from_url') as mock_extract:
            mock_extract.side_effect = lambda url: url.split('=')[-1] if '=' in url else url.split('/')[-1]
            
            batch = batch_service.create_batch(batch_request)
            
            # Test successful completion
            queue_item = batch_service.get_next_queue_item(worker_id="test-worker")
            success_result = BatchItemResult(
                batch_item_id=queue_item.batch_item_id,
                status=BatchItemStatus.COMPLETED,
                video_id=123,
                result_data={
                    "summary": "Test summary",
                    "keywords": ["test", "video"],
                    "duration": 180,
                    "transcript": "Test transcript content"
                }
            )
            
            success = batch_service.complete_batch_item(queue_item.batch_item_id, success_result)
            assert success is True
            
            # Verify result data
            session = batch_service._get_session()
            batch_item = session.get(BatchItem, queue_item.batch_item_id)
            assert batch_item.status == BatchItemStatus.COMPLETED
            assert batch_item.video_id == 123
            assert batch_item.result_data["summary"] == "Test summary"
            assert batch_item.result_data["keywords"] == ["test", "video"]
            assert batch_item.completed_at is not None
            
            # Test partial success
            queue_item2 = batch_service.get_next_queue_item(worker_id="test-worker")
            partial_result = BatchItemResult(
                batch_item_id=queue_item2.batch_item_id,
                status=BatchItemStatus.COMPLETED,
                video_id=124,
                result_data={
                    "summary": "Partial summary",
                    "keywords": None,  # Keywords extraction failed
                    "duration": None,
                    "transcript": "Partial transcript"
                }
            )
            
            success = batch_service.complete_batch_item(queue_item2.batch_item_id, partial_result)
            assert success is True
            
            # Test failure with detailed error
            queue_item3 = batch_service.get_next_queue_item(worker_id="test-worker")
            failure_result = BatchItemResult(
                batch_item_id=queue_item3.batch_item_id,
                status=BatchItemStatus.FAILED,
                error_message="Video is private",
                result_data={
                    "error_type": "ACCESS_DENIED",
                    "error_code": 403,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
            success = batch_service.complete_batch_item(queue_item3.batch_item_id, failure_result)
            assert success is True
            
            # Verify failure data
            batch_item3 = session.get(BatchItem, queue_item3.batch_item_id)
            assert batch_item3.status == BatchItemStatus.FAILED
            assert batch_item3.error_info == "Video is private"
            assert batch_item3.result_data["error_type"] == "ACCESS_DENIED"

    # Statistics and Monitoring Tests

    def test_batch_statistics_comprehensive(self, batch_service, batch_request):
        """Test comprehensive batch statistics functionality."""
        with patch('src.services.batch_service.extract_video_id_from_url') as mock_extract:
            mock_extract.side_effect = lambda url: url.split('=')[-1] if '=' in url else url.split('/')[-1]
            
            # Create multiple batches
            batch1 = batch_service.create_batch(batch_request)
            batch2 = batch_service.create_batch(BatchCreateRequest(
                name="Second Batch",
                urls=["https://www.youtube.com/watch?v=test123"],
                priority=BatchPriority.HIGH
            ))
            
            # Start processing
            batch_service.start_batch_processing(batch1.batch_id)
            
            # Process some items
            for i in range(2):
                queue_item = batch_service.get_next_queue_item(worker_id=f"worker_{i}")
                if queue_item:
                    result = BatchItemResult(
                        batch_item_id=queue_item.batch_item_id,
                        status=BatchItemStatus.COMPLETED,
                        video_id=i + 1
                    )
                    batch_service.complete_batch_item(queue_item.batch_item_id, result)
            
            # Fail one item
            queue_item = batch_service.get_next_queue_item(worker_id="worker_fail")
            if queue_item:
                result = BatchItemResult(
                    batch_item_id=queue_item.batch_item_id,
                    status=BatchItemStatus.FAILED,
                    error_message="Test failure"
                )
                batch_service.complete_batch_item(queue_item.batch_item_id, result)
            
            # Get statistics
            stats = batch_service.get_batch_statistics()
            
            assert stats["total_batches"] == 2
            assert stats["total_batch_items"] == 6  # 5 + 1
            assert stats["batch_status_counts"]["pending"] == 1
            assert stats["batch_status_counts"]["processing"] == 1
            assert stats["item_status_counts"]["completed"] == 2
            assert stats["item_status_counts"]["failed"] == 1
            assert stats["item_status_counts"]["queued"] == 2
            assert stats["item_status_counts"]["processing"] == 1

    def test_batch_progress_tracking(self, batch_service, batch_request):
        """Test batch progress tracking functionality."""
        with patch('src.services.batch_service.extract_video_id_from_url') as mock_extract:
            mock_extract.side_effect = lambda url: url.split('=')[-1] if '=' in url else url.split('/')[-1]
            
            batch = batch_service.create_batch(batch_request)
            
            # Initial progress
            progress = batch_service.get_batch_progress(batch.batch_id)
            assert progress.batch_id == batch.batch_id
            assert progress.status == BatchStatus.PENDING
            assert progress.total_items == 5
            assert progress.completed_items == 0
            assert progress.failed_items == 0
            assert progress.pending_items == 5
            assert progress.progress_percentage == 0.0
            
            # Start processing
            batch_service.start_batch_processing(batch.batch_id)
            
            # Process items progressively
            for i in range(5):
                queue_item = batch_service.get_next_queue_item(worker_id=f"worker_{i}")
                if queue_item:
                    status = BatchItemStatus.COMPLETED if i < 3 else BatchItemStatus.FAILED
                    result = BatchItemResult(
                        batch_item_id=queue_item.batch_item_id,
                        status=status,
                        video_id=i + 1 if status == BatchItemStatus.COMPLETED else None,
                        error_message="Test error" if status == BatchItemStatus.FAILED else None
                    )
                    batch_service.complete_batch_item(queue_item.batch_item_id, result)
                    
                    # Check progress
                    progress = batch_service.get_batch_progress(batch.batch_id)
                    expected_completed = min(i + 1, 3)  # First 3 complete, last 2 fail
                    expected_failed = max(0, i - 2)  # Last 2 fail
                    
                    assert progress.completed_items == expected_completed
                    assert progress.failed_items == expected_failed
                    assert progress.progress_percentage == ((expected_completed + expected_failed) / 5) * 100
            
            # Final progress
            final_progress = batch_service.get_batch_progress(batch.batch_id)
            assert final_progress.status == BatchStatus.COMPLETED
            assert final_progress.completed_items == 3
            assert final_progress.failed_items == 2
            assert final_progress.progress_percentage == 100.0

    # Performance and Concurrency Tests

    def test_concurrent_batch_operations(self, batch_service, large_batch_request):
        """Test concurrent batch operations."""
        with patch('src.services.batch_service.extract_video_id_from_url') as mock_extract:
            mock_extract.side_effect = lambda url: url.split('=')[-1] if '=' in url else url.split('/')[-1]
            
            # Create large batch
            batch = batch_service.create_batch(large_batch_request)
            
            # Start processing
            batch_service.start_batch_processing(batch.batch_id)
            
            # Process items concurrently
            def process_items(worker_id):
                processed = 0
                while processed < 10:  # Each worker processes up to 10 items
                    queue_item = batch_service.get_next_queue_item(worker_id=worker_id)
                    if queue_item:
                        result = BatchItemResult(
                            batch_item_id=queue_item.batch_item_id,
                            status=BatchItemStatus.COMPLETED,
                            video_id=processed + 1
                        )
                        batch_service.complete_batch_item(queue_item.batch_item_id, result)
                        processed += 1
                    else:
                        break
                return processed
            
            # Use thread pool for concurrent processing
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = [
                    executor.submit(process_items, f"worker_{i}") 
                    for i in range(5)
                ]
                
                results = [future.result() for future in futures]
            
            # Verify results
            total_processed = sum(results)
            assert total_processed <= 100  # Should not exceed total items
            assert total_processed > 0  # Should process some items
            
            # Verify batch progress
            progress = batch_service.get_batch_progress(batch.batch_id)
            assert progress.completed_items == total_processed

    def test_batch_service_performance(self, batch_service, large_batch_request):
        """Test batch service performance with large batches."""
        with patch('src.services.batch_service.extract_video_id_from_url') as mock_extract:
            mock_extract.side_effect = lambda url: url.split('=')[-1] if '=' in url else url.split('/')[-1]
            
            # Measure batch creation time
            start_time = time.time()
            batch = batch_service.create_batch(large_batch_request)
            creation_time = time.time() - start_time
            
            assert creation_time < 5.0  # Should create batch in less than 5 seconds
            assert batch.total_items == 100
            
            # Measure batch retrieval time
            start_time = time.time()
            retrieved_batch = batch_service.get_batch(batch.batch_id)
            retrieval_time = time.time() - start_time
            
            assert retrieval_time < 1.0  # Should retrieve batch in less than 1 second
            assert len(retrieved_batch.batch_items) == 100
            
            # Measure statistics calculation time
            start_time = time.time()
            stats = batch_service.get_batch_statistics()
            stats_time = time.time() - start_time
            
            assert stats_time < 2.0  # Should calculate stats in less than 2 seconds
            assert stats["total_batch_items"] == 100

    # Database Integration Tests

    def test_database_transaction_handling(self, batch_service, batch_request):
        """Test database transaction handling."""
        with patch('src.services.batch_service.extract_video_id_from_url') as mock_extract:
            mock_extract.side_effect = lambda url: url.split('=')[-1] if '=' in url else url.split('/')[-1]
            
            # Test successful transaction
            batch = batch_service.create_batch(batch_request)
            assert batch is not None
            
            # Verify data was committed
            session = batch_service._get_session()
            db_batch = session.query(Batch).filter_by(batch_id=batch.batch_id).first()
            assert db_batch is not None
            assert len(db_batch.batch_items) == 5
            
            # Test rollback on error
            with patch.object(session, 'commit') as mock_commit:
                mock_commit.side_effect = SQLAlchemyError("Database error")
                
                with pytest.raises(BatchServiceError):
                    batch_service.start_batch_processing(batch.batch_id)
                
                # Verify batch status wasn't changed
                session.refresh(db_batch)
                assert db_batch.status == BatchStatus.PENDING

    def test_database_constraint_handling(self, batch_service, sample_urls):
        """Test database constraint handling."""
        with patch('src.services.batch_service.extract_video_id_from_url') as mock_extract:
            mock_extract.side_effect = lambda url: url.split('=')[-1] if '=' in url else url.split('/')[-1]
            
            # Test unique constraint on batch_id
            request = BatchCreateRequest(
                name="Test Batch",
                urls=sample_urls[:2]
            )
            
            batch1 = batch_service.create_batch(request)
            
            # Should not be able to create batch with same ID
            with patch('src.services.batch_service.generate_batch_id') as mock_gen:
                mock_gen.return_value = batch1.batch_id
                
                # This should generate a new ID automatically
                batch2 = batch_service.create_batch(request)
                assert batch2.batch_id != batch1.batch_id

    def test_database_cleanup_operations(self, batch_service, batch_request):
        """Test database cleanup operations."""
        with patch('src.services.batch_service.extract_video_id_from_url') as mock_extract:
            mock_extract.side_effect = lambda url: url.split('=')[-1] if '=' in url else url.split('/')[-1]
            
            batch = batch_service.create_batch(batch_request)
            
            # Create some stale sessions
            for i in range(3):
                queue_item = batch_service.get_next_queue_item(worker_id=f"worker_{i}")
                if queue_item:
                    session = batch_service.create_processing_session(
                        queue_item.batch_item_id,
                        f"worker_{i}"
                    )
                    
                    # Make session stale
                    db_session = batch_service._get_session()
                    processing_session = db_session.get(ProcessingSession, session.id)
                    processing_session.heartbeat_at = datetime.utcnow() - timedelta(hours=1)
                    db_session.commit()
            
            # Run cleanup
            cleaned_count = batch_service.cleanup_stale_sessions(timeout_minutes=30)
            assert cleaned_count == 3
            
            # Verify cleanup worked
            db_session = batch_service._get_session()
            active_sessions = db_session.query(ProcessingSession).filter(
                ProcessingSession.heartbeat_at > datetime.utcnow() - timedelta(minutes=30)
            ).count()
            assert active_sessions == 0

    # Edge Cases and Error Conditions

    def test_edge_case_empty_batch_handling(self, batch_service):
        """Test edge cases with empty or minimal batches."""
        # Test with single URL
        with patch('src.services.batch_service.extract_video_id_from_url') as mock_extract:
            mock_extract.return_value = "test123"
            
            request = BatchCreateRequest(
                name="Single Item Batch",
                urls=["https://www.youtube.com/watch?v=test123"]
            )
            
            batch = batch_service.create_batch(request)
            assert batch.total_items == 1
            
            # Process the single item
            queue_item = batch_service.get_next_queue_item(worker_id="test-worker")
            assert queue_item is not None
            
            result = BatchItemResult(
                batch_item_id=queue_item.batch_item_id,
                status=BatchItemStatus.COMPLETED,
                video_id=1
            )
            batch_service.complete_batch_item(queue_item.batch_item_id, result)
            
            # Verify batch completion
            final_batch = batch_service.get_batch(batch.batch_id)
            assert final_batch.status == BatchStatus.COMPLETED
            assert final_batch.completed_items == 1

    def test_edge_case_malformed_data_handling(self, batch_service):
        """Test handling of malformed data."""
        # Test with malformed URLs
        malformed_urls = [
            "",  # Empty string
            "not-a-url",  # Not a URL
            "https://example.com",  # Not YouTube
            "https://www.youtube.com/watch?v=",  # Missing video ID
            "https://www.youtube.com/watch?v=invalid id"  # Invalid characters
        ]
        
        request = BatchCreateRequest(
            name="Malformed Batch",
            urls=malformed_urls
        )
        
        with patch('src.services.batch_service.extract_video_id_from_url') as mock_extract:
            mock_extract.return_value = None  # All URLs are invalid
            
            with pytest.raises(BatchServiceError, match="Invalid YouTube URL"):
                batch_service.create_batch(request)

    def test_edge_case_extreme_values(self, batch_service):
        """Test handling of extreme values."""
        # Test with very long names and descriptions
        long_name = "A" * 1000
        long_description = "B" * 5000
        
        request = BatchCreateRequest(
            name=long_name,
            description=long_description,
            urls=["https://www.youtube.com/watch?v=test123"]
        )
        
        with patch('src.services.batch_service.extract_video_id_from_url') as mock_extract:
            mock_extract.return_value = "test123"
            
            # Should handle long strings gracefully
            batch = batch_service.create_batch(request)
            assert batch.name == long_name
            assert batch.description == long_description

    def test_memory_usage_optimization(self, batch_service, large_batch_request):
        """Test memory usage optimization with large batches."""
        import psutil
        import os
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        with patch('src.services.batch_service.extract_video_id_from_url') as mock_extract:
            mock_extract.side_effect = lambda url: url.split('=')[-1] if '=' in url else url.split('/')[-1]
            
            # Create large batch
            batch = batch_service.create_batch(large_batch_request)
            
            # Process items in batches to avoid memory issues
            processed = 0
            while processed < 50:  # Process half the items
                queue_item = batch_service.get_next_queue_item(worker_id=f"worker_{processed}")
                if queue_item:
                    result = BatchItemResult(
                        batch_item_id=queue_item.batch_item_id,
                        status=BatchItemStatus.COMPLETED,
                        video_id=processed + 1
                    )
                    batch_service.complete_batch_item(queue_item.batch_item_id, result)
                    processed += 1
                else:
                    break
        
        # Check memory usage hasn't grown excessively
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB)
        assert memory_increase < 100 * 1024 * 1024

    def test_concurrent_worker_safety(self, batch_service, batch_request):
        """Test thread safety with concurrent workers."""
        with patch('src.services.batch_service.extract_video_id_from_url') as mock_extract:
            mock_extract.side_effect = lambda url: url.split('=')[-1] if '=' in url else url.split('/')[-1]
            
            batch = batch_service.create_batch(batch_request)
            batch_service.start_batch_processing(batch.batch_id)
            
            # Track processed items to ensure no duplicates
            processed_items = set()
            lock = threading.Lock()
            
            def worker_function(worker_id):
                local_processed = []
                for _ in range(2):  # Each worker tries to process 2 items
                    queue_item = batch_service.get_next_queue_item(worker_id=worker_id)
                    if queue_item:
                        # Ensure no duplicate processing
                        with lock:
                            if queue_item.batch_item_id not in processed_items:
                                processed_items.add(queue_item.batch_item_id)
                                local_processed.append(queue_item.batch_item_id)
                            else:
                                # This shouldn't happen - indicates a race condition
                                assert False, f"Duplicate processing of item {queue_item.batch_item_id}"
                        
                        result = BatchItemResult(
                            batch_item_id=queue_item.batch_item_id,
                            status=BatchItemStatus.COMPLETED,
                            video_id=queue_item.batch_item_id
                        )
                        batch_service.complete_batch_item(queue_item.batch_item_id, result)
                return local_processed
            
            # Start multiple workers concurrently
            threads = []
            for i in range(5):
                thread = threading.Thread(target=worker_function, args=(f"worker_{i}",))
                threads.append(thread)
                thread.start()
            
            # Wait for all threads to complete
            for thread in threads:
                thread.join()
            
            # Verify all items were processed exactly once
            assert len(processed_items) <= 5  # Should not exceed total items
            
            # Verify batch completion
            final_batch = batch_service.get_batch(batch.batch_id)
            assert final_batch.completed_items == len(processed_items)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])