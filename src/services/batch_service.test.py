"""
Test suite for BatchService functionality.

This test suite covers:
- Batch creation and management
- Batch lifecycle operations (start, cancel, complete)
- Queue management
- Processing session management
- Error handling and validation
- Integration with database models
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

from ..database.models import Base, Video
from ..database.batch_models import (
    Batch, BatchItem, QueueItem, ProcessingSession,
    BatchStatus, BatchItemStatus, BatchPriority
)
from .batch_service import (
    BatchService, BatchCreateRequest, BatchProgressInfo, BatchItemResult,
    BatchServiceError
)


class TestBatchService:
    """Test suite for BatchService."""

    @pytest.fixture
    def db_session(self):
        """Create in-memory database session for testing."""
        engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool
        )
        
        # Create all tables
        Base.metadata.create_all(engine)
        
        TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        session = TestingSessionLocal()
        
        yield session
        
        session.close()

    @pytest.fixture
    def batch_service(self, db_session):
        """Create BatchService instance with test database session."""
        return BatchService(session=db_session)

    @pytest.fixture
    def sample_urls(self):
        """Sample YouTube URLs for testing."""
        return [
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://www.youtube.com/watch?v=9bZkp7q19f0",
            "https://youtu.be/oHg5SJYRHA0"
        ]

    @pytest.fixture
    def batch_request(self, sample_urls):
        """Sample batch creation request."""
        return BatchCreateRequest(
            name="Test Batch",
            description="Test batch for unit testing",
            urls=sample_urls,
            priority=BatchPriority.NORMAL,
            batch_metadata={"test": True}
        )

    def test_create_batch_success(self, batch_service, batch_request):
        """Test successful batch creation."""
        # Mock URL validation
        with patch('src.services.batch_service.extract_video_id_from_url') as mock_extract:
            mock_extract.side_effect = lambda url: url.split('=')[-1] if '=' in url else url.split('/')[-1]
            
            batch = batch_service.create_batch(batch_request)
            
            assert batch is not None
            assert batch.name == "Test Batch"
            assert batch.description == "Test batch for unit testing"
            assert batch.status == BatchStatus.PENDING
            assert batch.priority == BatchPriority.NORMAL
            assert batch.total_items == 3
            assert batch.completed_items == 0
            assert batch.failed_items == 0
            assert batch.batch_metadata == {"test": True}
            assert batch.batch_id.startswith("batch_")

    def test_create_batch_empty_urls(self, batch_service):
        """Test batch creation with empty URLs."""
        request = BatchCreateRequest(
            name="Empty Batch",
            urls=[]
        )
        
        with pytest.raises(BatchServiceError, match="URLs list cannot be empty"):
            batch_service.create_batch(request)

    def test_create_batch_invalid_urls(self, batch_service):
        """Test batch creation with invalid URLs."""
        request = BatchCreateRequest(
            name="Invalid Batch",
            urls=["not-a-valid-url", "https://example.com"]
        )
        
        with patch('src.services.batch_service.extract_video_id_from_url') as mock_extract:
            mock_extract.return_value = None
            
            with pytest.raises(BatchServiceError, match="Invalid YouTube URL"):
                batch_service.create_batch(request)

    def test_get_batch_success(self, batch_service, batch_request):
        """Test successful batch retrieval."""
        with patch('src.services.batch_service.extract_video_id_from_url') as mock_extract:
            mock_extract.side_effect = lambda url: url.split('=')[-1] if '=' in url else url.split('/')[-1]
            
            created_batch = batch_service.create_batch(batch_request)
            retrieved_batch = batch_service.get_batch(created_batch.batch_id)
            
            assert retrieved_batch is not None
            assert retrieved_batch.batch_id == created_batch.batch_id
            assert retrieved_batch.name == "Test Batch"
            assert len(retrieved_batch.batch_items) == 3

    def test_get_batch_not_found(self, batch_service):
        """Test batch retrieval for non-existent batch."""
        result = batch_service.get_batch("non-existent-batch")
        assert result is None

    def test_get_batch_progress(self, batch_service, batch_request):
        """Test batch progress retrieval."""
        with patch('src.services.batch_service.extract_video_id_from_url') as mock_extract:
            mock_extract.side_effect = lambda url: url.split('=')[-1] if '=' in url else url.split('/')[-1]
            
            batch = batch_service.create_batch(batch_request)
            progress = batch_service.get_batch_progress(batch.batch_id)
            
            assert progress is not None
            assert progress.batch_id == batch.batch_id
            assert progress.status == BatchStatus.PENDING
            assert progress.total_items == 3
            assert progress.completed_items == 0
            assert progress.failed_items == 0
            assert progress.pending_items == 3
            assert progress.progress_percentage == 0.0

    def test_start_batch_processing(self, batch_service, batch_request):
        """Test starting batch processing."""
        with patch('src.services.batch_service.extract_video_id_from_url') as mock_extract:
            mock_extract.side_effect = lambda url: url.split('=')[-1] if '=' in url else url.split('/')[-1]
            
            batch = batch_service.create_batch(batch_request)
            result = batch_service.start_batch_processing(batch.batch_id)
            
            assert result is True
            
            # Check batch status was updated
            updated_batch = batch_service.get_batch(batch.batch_id)
            assert updated_batch.status == BatchStatus.PROCESSING
            assert updated_batch.started_at is not None

    def test_start_batch_processing_not_found(self, batch_service):
        """Test starting batch processing for non-existent batch."""
        with pytest.raises(BatchServiceError, match="Batch .* not found"):
            batch_service.start_batch_processing("non-existent-batch")

    def test_start_batch_processing_wrong_status(self, batch_service, batch_request):
        """Test starting batch processing for batch not in PENDING status."""
        with patch('src.services.batch_service.extract_video_id_from_url') as mock_extract:
            mock_extract.side_effect = lambda url: url.split('=')[-1] if '=' in url else url.split('/')[-1]
            
            batch = batch_service.create_batch(batch_request)
            
            # Change status to PROCESSING
            batch.status = BatchStatus.PROCESSING
            batch_service._get_session().commit()
            
            with pytest.raises(BatchServiceError, match="is not in PENDING status"):
                batch_service.start_batch_processing(batch.batch_id)

    def test_cancel_batch(self, batch_service, batch_request):
        """Test cancelling a batch."""
        with patch('src.services.batch_service.extract_video_id_from_url') as mock_extract:
            mock_extract.side_effect = lambda url: url.split('=')[-1] if '=' in url else url.split('/')[-1]
            
            batch = batch_service.create_batch(batch_request)
            batch_service.start_batch_processing(batch.batch_id)
            
            result = batch_service.cancel_batch(batch.batch_id, "Test cancellation")
            
            assert result is True
            
            # Check batch status was updated
            updated_batch = batch_service.get_batch(batch.batch_id)
            assert updated_batch.status == BatchStatus.CANCELLED
            assert updated_batch.completed_at is not None
            assert updated_batch.error_info == "Test cancellation"

    def test_cancel_batch_already_completed(self, batch_service, batch_request):
        """Test cancelling an already completed batch."""
        with patch('src.services.batch_service.extract_video_id_from_url') as mock_extract:
            mock_extract.side_effect = lambda url: url.split('=')[-1] if '=' in url else url.split('/')[-1]
            
            batch = batch_service.create_batch(batch_request)
            
            # Mark batch as completed
            batch.status = BatchStatus.COMPLETED
            batch_service._get_session().commit()
            
            with pytest.raises(BatchServiceError, match="is already completed"):
                batch_service.cancel_batch(batch.batch_id)

    def test_get_next_queue_item(self, batch_service, batch_request):
        """Test getting next queue item."""
        with patch('src.services.batch_service.extract_video_id_from_url') as mock_extract:
            mock_extract.side_effect = lambda url: url.split('=')[-1] if '=' in url else url.split('/')[-1]
            
            batch = batch_service.create_batch(batch_request)
            worker_id = "test-worker-123"
            
            queue_item = batch_service.get_next_queue_item(worker_id=worker_id)
            
            assert queue_item is not None
            assert queue_item.batch_item is not None
            assert queue_item.locked_by == worker_id
            assert queue_item.locked_at is not None
            assert queue_item.lock_expires_at is not None
            assert queue_item.batch_item.status == BatchItemStatus.PROCESSING

    def test_get_next_queue_item_no_items(self, batch_service):
        """Test getting next queue item when no items are available."""
        queue_item = batch_service.get_next_queue_item()
        assert queue_item is None

    def test_complete_batch_item_success(self, batch_service, batch_request):
        """Test completing a batch item successfully."""
        with patch('src.services.batch_service.extract_video_id_from_url') as mock_extract:
            mock_extract.side_effect = lambda url: url.split('=')[-1] if '=' in url else url.split('/')[-1]
            
            batch = batch_service.create_batch(batch_request)
            queue_item = batch_service.get_next_queue_item(worker_id="test-worker")
            
            result = BatchItemResult(
                batch_item_id=queue_item.batch_item_id,
                status=BatchItemStatus.COMPLETED,
                video_id=123,
                result_data={"summary": "Test summary"}
            )
            
            success = batch_service.complete_batch_item(queue_item.batch_item_id, result)
            
            assert success is True
            
            # Check batch item was updated
            session = batch_service._get_session()
            batch_item = session.get(BatchItem, queue_item.batch_item_id)
            assert batch_item.status == BatchItemStatus.COMPLETED
            assert batch_item.video_id == 123
            assert batch_item.result_data == {"summary": "Test summary"}
            assert batch_item.completed_at is not None

    def test_complete_batch_item_failure(self, batch_service, batch_request):
        """Test completing a batch item with failure."""
        with patch('src.services.batch_service.extract_video_id_from_url') as mock_extract:
            mock_extract.side_effect = lambda url: url.split('=')[-1] if '=' in url else url.split('/')[-1]
            
            batch = batch_service.create_batch(batch_request)
            queue_item = batch_service.get_next_queue_item(worker_id="test-worker")
            
            result = BatchItemResult(
                batch_item_id=queue_item.batch_item_id,
                status=BatchItemStatus.FAILED,
                error_message="Processing failed"
            )
            
            success = batch_service.complete_batch_item(queue_item.batch_item_id, result)
            
            assert success is True
            
            # Check batch item was updated
            session = batch_service._get_session()
            batch_item = session.get(BatchItem, queue_item.batch_item_id)
            assert batch_item.status == BatchItemStatus.FAILED
            assert batch_item.error_info == "Processing failed"
            assert batch_item.completed_at is not None

    def test_retry_failed_batch_item(self, batch_service, batch_request):
        """Test retrying a failed batch item."""
        with patch('src.services.batch_service.extract_video_id_from_url') as mock_extract:
            mock_extract.side_effect = lambda url: url.split('=')[-1] if '=' in url else url.split('/')[-1]
            
            batch = batch_service.create_batch(batch_request)
            queue_item = batch_service.get_next_queue_item(worker_id="test-worker")
            
            # Mark as failed
            failure_result = BatchItemResult(
                batch_item_id=queue_item.batch_item_id,
                status=BatchItemStatus.FAILED,
                error_message="Processing failed"
            )
            batch_service.complete_batch_item(queue_item.batch_item_id, failure_result)
            
            # Retry the item
            success = batch_service.retry_failed_batch_item(queue_item.batch_item_id)
            
            assert success is True
            
            # Check batch item was updated
            session = batch_service._get_session()
            batch_item = session.get(BatchItem, queue_item.batch_item_id)
            assert batch_item.status == BatchItemStatus.QUEUED
            assert batch_item.retry_count == 1
            assert batch_item.error_info is None

    def test_retry_batch_item_max_retries(self, batch_service, batch_request):
        """Test retrying a batch item that has reached max retries."""
        with patch('src.services.batch_service.extract_video_id_from_url') as mock_extract:
            mock_extract.side_effect = lambda url: url.split('=')[-1] if '=' in url else url.split('/')[-1]
            
            batch = batch_service.create_batch(batch_request)
            queue_item = batch_service.get_next_queue_item(worker_id="test-worker")
            
            # Set retry count to max
            session = batch_service._get_session()
            batch_item = session.get(BatchItem, queue_item.batch_item_id)
            batch_item.retry_count = batch_item.max_retries
            batch_item.status = BatchItemStatus.FAILED
            session.commit()
            
            with pytest.raises(BatchServiceError, match="cannot be retried"):
                batch_service.retry_failed_batch_item(queue_item.batch_item_id)

    def test_create_processing_session(self, batch_service, batch_request):
        """Test creating a processing session."""
        with patch('src.services.batch_service.extract_video_id_from_url') as mock_extract:
            mock_extract.side_effect = lambda url: url.split('=')[-1] if '=' in url else url.split('/')[-1]
            
            batch = batch_service.create_batch(batch_request)
            queue_item = batch_service.get_next_queue_item(worker_id="test-worker")
            
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

    def test_update_processing_session(self, batch_service, batch_request):
        """Test updating processing session progress."""
        with patch('src.services.batch_service.extract_video_id_from_url') as mock_extract:
            mock_extract.side_effect = lambda url: url.split('=')[-1] if '=' in url else url.split('/')[-1]
            
            batch = batch_service.create_batch(batch_request)
            queue_item = batch_service.get_next_queue_item(worker_id="test-worker")
            
            session = batch_service.create_processing_session(
                queue_item.batch_item_id,
                "test-worker"
            )
            
            success = batch_service.update_processing_session(
                session.session_id,
                50.0,
                "Processing transcripts"
            )
            
            assert success is True
            
            # Check session was updated
            db_session = batch_service._get_session()
            updated_session = db_session.get(ProcessingSession, session.id)
            assert updated_session.progress_percentage == 50.0
            assert updated_session.current_step == "Processing transcripts"

    def test_get_batch_statistics(self, batch_service, batch_request):
        """Test getting batch statistics."""
        with patch('src.services.batch_service.extract_video_id_from_url') as mock_extract:
            mock_extract.side_effect = lambda url: url.split('=')[-1] if '=' in url else url.split('/')[-1]
            
            # Create some test data
            batch = batch_service.create_batch(batch_request)
            queue_item = batch_service.get_next_queue_item(worker_id="test-worker")
            
            stats = batch_service.get_batch_statistics()
            
            assert stats is not None
            assert stats["total_batches"] == 1
            assert stats["total_batch_items"] == 3
            assert stats["batch_status_counts"]["pending"] == 1
            assert stats["item_status_counts"]["processing"] == 1
            assert stats["item_status_counts"]["queued"] == 2

    def test_cleanup_stale_sessions(self, batch_service, batch_request):
        """Test cleaning up stale processing sessions."""
        with patch('src.services.batch_service.extract_video_id_from_url') as mock_extract:
            mock_extract.side_effect = lambda url: url.split('=')[-1] if '=' in url else url.split('/')[-1]
            
            batch = batch_service.create_batch(batch_request)
            queue_item = batch_service.get_next_queue_item(worker_id="test-worker")
            
            session = batch_service.create_processing_session(
                queue_item.batch_item_id,
                "test-worker"
            )
            
            # Make session stale by modifying heartbeat
            db_session = batch_service._get_session()
            processing_session = db_session.get(ProcessingSession, session.id)
            processing_session.heartbeat_at = datetime.utcnow() - timedelta(hours=1)
            db_session.commit()
            
            cleaned_count = batch_service.cleanup_stale_sessions(timeout_minutes=30)
            
            assert cleaned_count == 1
            
            # Check that batch item was marked as failed
            batch_item = db_session.get(BatchItem, queue_item.batch_item_id)
            assert batch_item.status == BatchItemStatus.FAILED
            assert "timed out" in batch_item.error_info

    def test_list_batches(self, batch_service, batch_request):
        """Test listing batches."""
        with patch('src.services.batch_service.extract_video_id_from_url') as mock_extract:
            mock_extract.side_effect = lambda url: url.split('=')[-1] if '=' in url else url.split('/')[-1]
            
            # Create multiple batches
            batch1 = batch_service.create_batch(batch_request)
            
            batch_request2 = BatchCreateRequest(
                name="Second Batch",
                urls=["https://www.youtube.com/watch?v=test123"],
                priority=BatchPriority.HIGH
            )
            batch2 = batch_service.create_batch(batch_request2)
            
            # Test listing all batches
            batches = batch_service.list_batches()
            assert len(batches) == 2
            
            # Test filtering by status
            pending_batches = batch_service.list_batches(status=BatchStatus.PENDING)
            assert len(pending_batches) == 2
            
            completed_batches = batch_service.list_batches(status=BatchStatus.COMPLETED)
            assert len(completed_batches) == 0

    def test_batch_completion_detection(self, batch_service, batch_request):
        """Test that batch is marked as completed when all items are processed."""
        with patch('src.services.batch_service.extract_video_id_from_url') as mock_extract:
            mock_extract.side_effect = lambda url: url.split('=')[-1] if '=' in url else url.split('/')[-1]
            
            batch = batch_service.create_batch(batch_request)
            
            # Complete all items
            for i in range(3):
                queue_item = batch_service.get_next_queue_item(worker_id=f"worker-{i}")
                if queue_item:
                    result = BatchItemResult(
                        batch_item_id=queue_item.batch_item_id,
                        status=BatchItemStatus.COMPLETED,
                        video_id=i + 1
                    )
                    batch_service.complete_batch_item(queue_item.batch_item_id, result)
            
            # Check batch is marked as completed
            final_batch = batch_service.get_batch(batch.batch_id)
            assert final_batch.status == BatchStatus.COMPLETED
            assert final_batch.completed_at is not None
            assert final_batch.completed_items == 3

    def test_database_error_handling(self, batch_service):
        """Test database error handling."""
        # Mock database error
        with patch.object(batch_service, '_get_session') as mock_session:
            mock_session.side_effect = Exception("Database connection failed")
            
            with pytest.raises(BatchServiceError):
                batch_service.get_batch_statistics()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])