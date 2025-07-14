"""
Comprehensive database integration tests for batch processing models.

This test suite provides comprehensive database integration testing including:
- Model creation and validation
- Database schema integrity
- Foreign key relationships
- Indexes and constraints
- Data persistence and retrieval
- Transaction management
- Cascading deletes
- Model property calculations
- Enum validation
- JSON field handling
- Concurrent access patterns
- Database migration scenarios
"""

import pytest
import uuid
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from concurrent.futures import ThreadPoolExecutor

from src.database.models import Base, Video
from src.database.batch_models import (
    Batch, BatchItem, QueueItem, ProcessingSession,
    BatchStatus, BatchItemStatus, BatchPriority,
    get_batch_model_by_name, get_all_batch_models,
    create_batch_tables, drop_batch_tables
)


class TestBatchDatabaseIntegration:
    """Comprehensive database integration tests for batch processing models."""

    @pytest.fixture(scope="function")
    def db_engine(self):
        """Create in-memory database engine with all tables."""
        engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
            echo=False
        )
        
        # Create all tables
        Base.metadata.create_all(engine)
        
        yield engine
        
        # Cleanup
        engine.dispose()

    @pytest.fixture(scope="function")
    def db_session(self, db_engine):
        """Create database session."""
        TestingSessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=db_engine
        )
        session = TestingSessionLocal()
        
        yield session
        
        session.rollback()
        session.close()

    @pytest.fixture(scope="function")
    def sample_video(self, db_session):
        """Create a sample video for testing."""
        video = Video(
            video_id="test123",
            title="Test Video",
            description="Test description",
            duration=300,
            upload_date=datetime.utcnow(),
            thumbnail_url="https://example.com/thumb.jpg",
            channel_name="Test Channel",
            url="https://www.youtube.com/watch?v=test123"
        )
        db_session.add(video)
        db_session.commit()
        db_session.refresh(video)
        return video

    # Model Creation and Validation Tests

    def test_batch_model_creation(self, db_session):
        """Test basic batch model creation and validation."""
        # Create batch with minimal required fields
        batch = Batch(
            batch_id="batch_" + str(uuid.uuid4()),
            name="Test Batch",
            description="Test batch description",
            total_items=5
        )
        
        db_session.add(batch)
        db_session.commit()
        db_session.refresh(batch)
        
        # Verify batch was created
        assert batch.id is not None
        assert batch.batch_id is not None
        assert batch.name == "Test Batch"
        assert batch.description == "Test batch description"
        assert batch.status == BatchStatus.PENDING
        assert batch.priority == BatchPriority.NORMAL
        assert batch.total_items == 5
        assert batch.completed_items == 0
        assert batch.failed_items == 0
        assert batch.pending_items == 5
        assert batch.progress_percentage == 0.0
        assert batch.created_at is not None
        assert batch.updated_at is not None
        assert batch.is_completed is False
        assert batch.is_failed is False

    def test_batch_model_with_all_fields(self, db_session):
        """Test batch model creation with all fields."""
        metadata = {
            "source": "api",
            "priority_reason": "urgent_request",
            "estimated_duration": 300
        }
        
        batch = Batch(
            batch_id="batch_" + str(uuid.uuid4()),
            name="Complete Test Batch",
            description="Test batch with all fields",
            status=BatchStatus.PROCESSING,
            priority=BatchPriority.HIGH,
            total_items=10,
            completed_items=3,
            failed_items=1,
            started_at=datetime.utcnow(),
            webhook_url="https://webhook.example.com/batch",
            batch_metadata=metadata,
            error_info="Test error info"
        )
        
        db_session.add(batch)
        db_session.commit()
        db_session.refresh(batch)
        
        # Verify all fields
        assert batch.name == "Complete Test Batch"
        assert batch.status == BatchStatus.PROCESSING
        assert batch.priority == BatchPriority.HIGH
        assert batch.total_items == 10
        assert batch.completed_items == 3
        assert batch.failed_items == 1
        assert batch.pending_items == 6
        assert batch.progress_percentage == 30.0
        assert batch.webhook_url == "https://webhook.example.com/batch"
        assert batch.batch_metadata == metadata
        assert batch.error_info == "Test error info"

    def test_batch_item_model_creation(self, db_session, sample_video):
        """Test batch item model creation."""
        # Create batch first
        batch = Batch(
            batch_id="batch_" + str(uuid.uuid4()),
            total_items=1
        )
        db_session.add(batch)
        db_session.commit()
        
        # Create batch item
        batch_item = BatchItem(
            batch_id=batch.id,
            video_id=sample_video.id,
            url="https://www.youtube.com/watch?v=test123",
            status=BatchItemStatus.QUEUED,
            priority=BatchPriority.HIGH,
            processing_order=1,
            max_retries=5
        )
        
        db_session.add(batch_item)
        db_session.commit()
        db_session.refresh(batch_item)
        
        # Verify batch item
        assert batch_item.id is not None
        assert batch_item.batch_id == batch.id
        assert batch_item.video_id == sample_video.id
        assert batch_item.url == "https://www.youtube.com/watch?v=test123"
        assert batch_item.status == BatchItemStatus.QUEUED
        assert batch_item.priority == BatchPriority.HIGH
        assert batch_item.processing_order == 1
        assert batch_item.retry_count == 0
        assert batch_item.max_retries == 5
        assert batch_item.can_retry is False  # Not failed yet
        assert batch_item.is_completed is False
        assert batch_item.is_failed is False
        assert batch_item.is_processing is False

    def test_queue_item_model_creation(self, db_session, sample_video):
        """Test queue item model creation."""
        # Create batch and batch item
        batch = Batch(
            batch_id="batch_" + str(uuid.uuid4()),
            total_items=1
        )
        db_session.add(batch)
        db_session.commit()
        
        batch_item = BatchItem(
            batch_id=batch.id,
            url="https://www.youtube.com/watch?v=test123"
        )
        db_session.add(batch_item)
        db_session.commit()
        
        # Create queue item
        scheduled_time = datetime.utcnow() + timedelta(minutes=5)
        queue_item = QueueItem(
            batch_item_id=batch_item.id,
            queue_name="test_queue",
            priority=BatchPriority.URGENT,
            scheduled_at=scheduled_time,
            max_retries=2
        )
        
        db_session.add(queue_item)
        db_session.commit()
        db_session.refresh(queue_item)
        
        # Verify queue item
        assert queue_item.id is not None
        assert queue_item.batch_item_id == batch_item.id
        assert queue_item.queue_name == "test_queue"
        assert queue_item.priority == BatchPriority.URGENT
        assert queue_item.scheduled_at == scheduled_time
        assert queue_item.retry_count == 0
        assert queue_item.max_retries == 2
        assert queue_item.is_locked is False
        assert queue_item.is_available is False  # Scheduled in future
        assert queue_item.can_retry is True

    def test_processing_session_model_creation(self, db_session):
        """Test processing session model creation."""
        # Create batch and batch item
        batch = Batch(
            batch_id="batch_" + str(uuid.uuid4()),
            total_items=1
        )
        db_session.add(batch)
        db_session.commit()
        
        batch_item = BatchItem(
            batch_id=batch.id,
            url="https://www.youtube.com/watch?v=test123"
        )
        db_session.add(batch_item)
        db_session.commit()
        
        # Create processing session
        session_metadata = {
            "worker_type": "batch_processor",
            "version": "1.0.0"
        }
        
        session = ProcessingSession(
            session_id="session_" + str(uuid.uuid4()),
            batch_item_id=batch_item.id,
            worker_id="worker_123",
            status=BatchItemStatus.PROCESSING,
            progress_percentage=25.5,
            current_step="metadata_extraction",
            session_metadata=session_metadata
        )
        
        db_session.add(session)
        db_session.commit()
        db_session.refresh(session)
        
        # Verify processing session
        assert session.id is not None
        assert session.session_id is not None
        assert session.batch_item_id == batch_item.id
        assert session.worker_id == "worker_123"
        assert session.status == BatchItemStatus.PROCESSING
        assert session.progress_percentage == 25.5
        assert session.current_step == "metadata_extraction"
        assert session.session_metadata == session_metadata
        assert session.is_stale(timeout_seconds=300) is False

    # Relationship Tests

    def test_batch_to_batch_items_relationship(self, db_session):
        """Test batch to batch items relationship."""
        # Create batch
        batch = Batch(
            batch_id="batch_" + str(uuid.uuid4()),
            total_items=3
        )
        db_session.add(batch)
        db_session.commit()
        
        # Create batch items
        items = []
        for i in range(3):
            item = BatchItem(
                batch_id=batch.id,
                url=f"https://www.youtube.com/watch?v=test{i}",
                processing_order=i
            )
            items.append(item)
            db_session.add(item)
        
        db_session.commit()
        
        # Test relationship
        db_session.refresh(batch)
        assert len(batch.batch_items) == 3
        assert all(item.batch_id == batch.id for item in batch.batch_items)
        assert batch.batch_items[0].processing_order == 0
        assert batch.batch_items[1].processing_order == 1
        assert batch.batch_items[2].processing_order == 2

    def test_batch_item_to_video_relationship(self, db_session, sample_video):
        """Test batch item to video relationship."""
        # Create batch and batch item
        batch = Batch(
            batch_id="batch_" + str(uuid.uuid4()),
            total_items=1
        )
        db_session.add(batch)
        db_session.commit()
        
        batch_item = BatchItem(
            batch_id=batch.id,
            video_id=sample_video.id,
            url="https://www.youtube.com/watch?v=test123"
        )
        db_session.add(batch_item)
        db_session.commit()
        
        # Test relationship
        db_session.refresh(batch_item)
        assert batch_item.video is not None
        assert batch_item.video.id == sample_video.id
        assert batch_item.video.video_id == "test123"

    def test_queue_item_to_batch_item_relationship(self, db_session):
        """Test queue item to batch item relationship."""
        # Create batch and batch item
        batch = Batch(
            batch_id="batch_" + str(uuid.uuid4()),
            total_items=1
        )
        db_session.add(batch)
        db_session.commit()
        
        batch_item = BatchItem(
            batch_id=batch.id,
            url="https://www.youtube.com/watch?v=test123"
        )
        db_session.add(batch_item)
        db_session.commit()
        
        # Create queue item
        queue_item = QueueItem(
            batch_item_id=batch_item.id,
            queue_name="test_queue"
        )
        db_session.add(queue_item)
        db_session.commit()
        
        # Test relationship
        db_session.refresh(queue_item)
        assert queue_item.batch_item is not None
        assert queue_item.batch_item.id == batch_item.id
        assert queue_item.batch_item.url == "https://www.youtube.com/watch?v=test123"

    def test_processing_session_to_batch_item_relationship(self, db_session):
        """Test processing session to batch item relationship."""
        # Create batch and batch item
        batch = Batch(
            batch_id="batch_" + str(uuid.uuid4()),
            total_items=1
        )
        db_session.add(batch)
        db_session.commit()
        
        batch_item = BatchItem(
            batch_id=batch.id,
            url="https://www.youtube.com/watch?v=test123"
        )
        db_session.add(batch_item)
        db_session.commit()
        
        # Create processing session
        session = ProcessingSession(
            session_id="session_" + str(uuid.uuid4()),
            batch_item_id=batch_item.id,
            worker_id="worker_123"
        )
        db_session.add(session)
        db_session.commit()
        
        # Test relationship
        db_session.refresh(session)
        assert session.batch_item is not None
        assert session.batch_item.id == batch_item.id
        assert session.batch_item.url == "https://www.youtube.com/watch?v=test123"

    # Cascading Delete Tests

    def test_batch_cascade_delete(self, db_session):
        """Test batch cascade delete removes batch items."""
        # Create batch with items
        batch = Batch(
            batch_id="batch_" + str(uuid.uuid4()),
            total_items=2
        )
        db_session.add(batch)
        db_session.commit()
        
        # Create batch items
        item1 = BatchItem(
            batch_id=batch.id,
            url="https://www.youtube.com/watch?v=test1"
        )
        item2 = BatchItem(
            batch_id=batch.id,
            url="https://www.youtube.com/watch?v=test2"
        )
        db_session.add(item1)
        db_session.add(item2)
        db_session.commit()
        
        # Create queue items
        queue1 = QueueItem(
            batch_item_id=item1.id,
            queue_name="test_queue"
        )
        queue2 = QueueItem(
            batch_item_id=item2.id,
            queue_name="test_queue"
        )
        db_session.add(queue1)
        db_session.add(queue2)
        db_session.commit()
        
        # Verify items exist
        assert db_session.query(BatchItem).filter_by(batch_id=batch.id).count() == 2
        assert db_session.query(QueueItem).filter_by(batch_item_id=item1.id).count() == 1
        assert db_session.query(QueueItem).filter_by(batch_item_id=item2.id).count() == 1
        
        # Delete batch
        db_session.delete(batch)
        db_session.commit()
        
        # Verify cascade delete
        assert db_session.query(BatchItem).filter_by(batch_id=batch.id).count() == 0
        assert db_session.query(QueueItem).filter_by(batch_item_id=item1.id).count() == 0
        assert db_session.query(QueueItem).filter_by(batch_item_id=item2.id).count() == 0

    def test_batch_item_cascade_delete(self, db_session):
        """Test batch item cascade delete removes related items."""
        # Create batch and batch item
        batch = Batch(
            batch_id="batch_" + str(uuid.uuid4()),
            total_items=1
        )
        db_session.add(batch)
        db_session.commit()
        
        batch_item = BatchItem(
            batch_id=batch.id,
            url="https://www.youtube.com/watch?v=test123"
        )
        db_session.add(batch_item)
        db_session.commit()
        
        # Create related items
        queue_item = QueueItem(
            batch_item_id=batch_item.id,
            queue_name="test_queue"
        )
        session = ProcessingSession(
            session_id="session_" + str(uuid.uuid4()),
            batch_item_id=batch_item.id,
            worker_id="worker_123"
        )
        db_session.add(queue_item)
        db_session.add(session)
        db_session.commit()
        
        # Verify items exist
        assert db_session.query(QueueItem).filter_by(batch_item_id=batch_item.id).count() == 1
        assert db_session.query(ProcessingSession).filter_by(batch_item_id=batch_item.id).count() == 1
        
        # Delete batch item
        db_session.delete(batch_item)
        db_session.commit()
        
        # Verify cascade delete
        assert db_session.query(QueueItem).filter_by(batch_item_id=batch_item.id).count() == 0
        assert db_session.query(ProcessingSession).filter_by(batch_item_id=batch_item.id).count() == 0

    # Validation Tests

    def test_batch_validation_empty_batch_id(self, db_session):
        """Test batch validation for empty batch ID."""
        batch = Batch(
            batch_id="",  # Empty batch ID
            total_items=1
        )
        
        with pytest.raises(ValueError, match="Batch ID cannot be empty"):
            db_session.add(batch)
            db_session.commit()

    def test_batch_validation_negative_counts(self, db_session):
        """Test batch validation for negative counts."""
        # Test negative total_items
        batch = Batch(
            batch_id="batch_" + str(uuid.uuid4()),
            total_items=-1
        )
        
        with pytest.raises(ValueError, match="Total items cannot be negative"):
            db_session.add(batch)
            db_session.commit()

    def test_batch_item_validation_empty_url(self, db_session):
        """Test batch item validation for empty URL."""
        batch = Batch(
            batch_id="batch_" + str(uuid.uuid4()),
            total_items=1
        )
        db_session.add(batch)
        db_session.commit()
        
        batch_item = BatchItem(
            batch_id=batch.id,
            url=""  # Empty URL
        )
        
        with pytest.raises(ValueError, match="URL cannot be empty"):
            db_session.add(batch_item)
            db_session.commit()

    def test_batch_item_validation_negative_retry_count(self, db_session):
        """Test batch item validation for negative retry count."""
        batch = Batch(
            batch_id="batch_" + str(uuid.uuid4()),
            total_items=1
        )
        db_session.add(batch)
        db_session.commit()
        
        batch_item = BatchItem(
            batch_id=batch.id,
            url="https://www.youtube.com/watch?v=test123",
            retry_count=-1  # Negative retry count
        )
        
        with pytest.raises(ValueError, match="Retry count cannot be negative"):
            db_session.add(batch_item)
            db_session.commit()

    def test_queue_item_validation_empty_queue_name(self, db_session):
        """Test queue item validation for empty queue name."""
        batch = Batch(
            batch_id="batch_" + str(uuid.uuid4()),
            total_items=1
        )
        db_session.add(batch)
        db_session.commit()
        
        batch_item = BatchItem(
            batch_id=batch.id,
            url="https://www.youtube.com/watch?v=test123"
        )
        db_session.add(batch_item)
        db_session.commit()
        
        queue_item = QueueItem(
            batch_item_id=batch_item.id,
            queue_name=""  # Empty queue name
        )
        
        with pytest.raises(ValueError, match="Queue name cannot be empty"):
            db_session.add(queue_item)
            db_session.commit()

    def test_processing_session_validation_empty_session_id(self, db_session):
        """Test processing session validation for empty session ID."""
        batch = Batch(
            batch_id="batch_" + str(uuid.uuid4()),
            total_items=1
        )
        db_session.add(batch)
        db_session.commit()
        
        batch_item = BatchItem(
            batch_id=batch.id,
            url="https://www.youtube.com/watch?v=test123"
        )
        db_session.add(batch_item)
        db_session.commit()
        
        session = ProcessingSession(
            session_id="",  # Empty session ID
            batch_item_id=batch_item.id,
            worker_id="worker_123"
        )
        
        with pytest.raises(ValueError, match="Session ID cannot be empty"):
            db_session.add(session)
            db_session.commit()

    def test_processing_session_validation_invalid_progress(self, db_session):
        """Test processing session validation for invalid progress."""
        batch = Batch(
            batch_id="batch_" + str(uuid.uuid4()),
            total_items=1
        )
        db_session.add(batch)
        db_session.commit()
        
        batch_item = BatchItem(
            batch_id=batch.id,
            url="https://www.youtube.com/watch?v=test123"
        )
        db_session.add(batch_item)
        db_session.commit()
        
        session = ProcessingSession(
            session_id="session_" + str(uuid.uuid4()),
            batch_item_id=batch_item.id,
            worker_id="worker_123",
            progress_percentage=150.0  # Invalid progress > 100
        )
        
        with pytest.raises(ValueError, match="Progress percentage must be between 0 and 100"):
            db_session.add(session)
            db_session.commit()

    # Property Tests

    def test_batch_property_calculations(self, db_session):
        """Test batch property calculations."""
        batch = Batch(
            batch_id="batch_" + str(uuid.uuid4()),
            total_items=10,
            completed_items=6,
            failed_items=2
        )
        db_session.add(batch)
        db_session.commit()
        
        # Test properties
        assert batch.pending_items == 2  # 10 - 6 - 2
        assert batch.progress_percentage == 60.0  # 6/10 * 100
        assert batch.is_completed is False
        assert batch.is_failed is False
        
        # Test completed batch
        batch.status = BatchStatus.COMPLETED
        batch.completed_items = 10
        db_session.commit()
        
        assert batch.pending_items == 0
        assert batch.progress_percentage == 100.0
        assert batch.is_completed is True
        assert batch.is_failed is False

    def test_batch_item_property_calculations(self, db_session):
        """Test batch item property calculations."""
        batch = Batch(
            batch_id="batch_" + str(uuid.uuid4()),
            total_items=1
        )
        db_session.add(batch)
        db_session.commit()
        
        # Test queued item
        batch_item = BatchItem(
            batch_id=batch.id,
            url="https://www.youtube.com/watch?v=test123",
            status=BatchItemStatus.QUEUED
        )
        db_session.add(batch_item)
        db_session.commit()
        
        assert batch_item.can_retry is False
        assert batch_item.is_completed is False
        assert batch_item.is_failed is False
        assert batch_item.is_processing is False
        
        # Test failed item
        batch_item.status = BatchItemStatus.FAILED
        batch_item.retry_count = 1
        db_session.commit()
        
        assert batch_item.can_retry is True
        assert batch_item.is_completed is False
        assert batch_item.is_failed is True
        assert batch_item.is_processing is False
        
        # Test processing item
        batch_item.status = BatchItemStatus.PROCESSING
        db_session.commit()
        
        assert batch_item.is_processing is True

    def test_queue_item_property_calculations(self, db_session):
        """Test queue item property calculations."""
        batch = Batch(
            batch_id="batch_" + str(uuid.uuid4()),
            total_items=1
        )
        db_session.add(batch)
        db_session.commit()
        
        batch_item = BatchItem(
            batch_id=batch.id,
            url="https://www.youtube.com/watch?v=test123"
        )
        db_session.add(batch_item)
        db_session.commit()
        
        # Test unlocked item
        queue_item = QueueItem(
            batch_item_id=batch_item.id,
            queue_name="test_queue",
            scheduled_at=datetime.utcnow() - timedelta(minutes=1)  # Past time
        )
        db_session.add(queue_item)
        db_session.commit()
        
        assert queue_item.is_locked is False
        assert queue_item.is_available is True
        assert queue_item.can_retry is True
        
        # Test locked item
        queue_item.locked_at = datetime.utcnow()
        queue_item.locked_by = "worker_123"
        queue_item.lock_expires_at = datetime.utcnow() + timedelta(minutes=30)
        db_session.commit()
        
        assert queue_item.is_locked is True
        assert queue_item.is_available is False

    def test_processing_session_stale_detection(self, db_session):
        """Test processing session stale detection."""
        batch = Batch(
            batch_id="batch_" + str(uuid.uuid4()),
            total_items=1
        )
        db_session.add(batch)
        db_session.commit()
        
        batch_item = BatchItem(
            batch_id=batch.id,
            url="https://www.youtube.com/watch?v=test123"
        )
        db_session.add(batch_item)
        db_session.commit()
        
        # Test fresh session
        session = ProcessingSession(
            session_id="session_" + str(uuid.uuid4()),
            batch_item_id=batch_item.id,
            worker_id="worker_123",
            heartbeat_at=datetime.utcnow()
        )
        db_session.add(session)
        db_session.commit()
        
        assert session.is_stale(timeout_seconds=300) is False
        
        # Test stale session
        session.heartbeat_at = datetime.utcnow() - timedelta(minutes=10)
        db_session.commit()
        
        assert session.is_stale(timeout_seconds=300) is True

    # JSON Field Tests

    def test_batch_metadata_json_field(self, db_session):
        """Test batch metadata JSON field handling."""
        complex_metadata = {
            "source": "api",
            "settings": {
                "quality": "high",
                "format": "mp4",
                "options": ["transcribe", "summarize"]
            },
            "timestamps": {
                "created": "2023-01-01T00:00:00Z",
                "updated": "2023-01-01T01:00:00Z"
            },
            "numbers": [1, 2, 3, 4, 5],
            "boolean_flag": True,
            "null_value": None
        }
        
        batch = Batch(
            batch_id="batch_" + str(uuid.uuid4()),
            batch_metadata=complex_metadata
        )
        db_session.add(batch)
        db_session.commit()
        db_session.refresh(batch)
        
        # Verify JSON serialization/deserialization
        assert batch.batch_metadata == complex_metadata
        assert batch.batch_metadata["source"] == "api"
        assert batch.batch_metadata["settings"]["quality"] == "high"
        assert batch.batch_metadata["numbers"] == [1, 2, 3, 4, 5]
        assert batch.batch_metadata["boolean_flag"] is True
        assert batch.batch_metadata["null_value"] is None

    def test_batch_item_processing_data_json_field(self, db_session):
        """Test batch item processing data JSON field handling."""
        batch = Batch(
            batch_id="batch_" + str(uuid.uuid4()),
            total_items=1
        )
        db_session.add(batch)
        db_session.commit()
        
        processing_data = {
            "extraction_method": "yt-dlp",
            "video_info": {
                "duration": 300,
                "resolution": "1080p",
                "fps": 30
            },
            "steps_completed": ["download", "extract_audio"],
            "current_step": "transcribe",
            "progress_details": {
                "transcription_progress": 0.75,
                "chunks_processed": 15,
                "total_chunks": 20
            }
        }
        
        batch_item = BatchItem(
            batch_id=batch.id,
            url="https://www.youtube.com/watch?v=test123",
            processing_data=processing_data
        )
        db_session.add(batch_item)
        db_session.commit()
        db_session.refresh(batch_item)
        
        # Verify JSON field
        assert batch_item.processing_data == processing_data
        assert batch_item.processing_data["extraction_method"] == "yt-dlp"
        assert batch_item.processing_data["video_info"]["duration"] == 300
        assert batch_item.processing_data["progress_details"]["transcription_progress"] == 0.75

    # Uniqueness Constraint Tests

    def test_batch_unique_batch_id_constraint(self, db_session):
        """Test batch unique batch_id constraint."""
        batch_id = "batch_" + str(uuid.uuid4())
        
        # Create first batch
        batch1 = Batch(
            batch_id=batch_id,
            total_items=1
        )
        db_session.add(batch1)
        db_session.commit()
        
        # Try to create second batch with same ID
        batch2 = Batch(
            batch_id=batch_id,  # Same ID
            total_items=2
        )
        db_session.add(batch2)
        
        with pytest.raises(IntegrityError):
            db_session.commit()

    def test_processing_session_unique_session_id_constraint(self, db_session):
        """Test processing session unique session_id constraint."""
        batch = Batch(
            batch_id="batch_" + str(uuid.uuid4()),
            total_items=2
        )
        db_session.add(batch)
        db_session.commit()
        
        batch_item1 = BatchItem(
            batch_id=batch.id,
            url="https://www.youtube.com/watch?v=test1"
        )
        batch_item2 = BatchItem(
            batch_id=batch.id,
            url="https://www.youtube.com/watch?v=test2"
        )
        db_session.add(batch_item1)
        db_session.add(batch_item2)
        db_session.commit()
        
        session_id = "session_" + str(uuid.uuid4())
        
        # Create first session
        session1 = ProcessingSession(
            session_id=session_id,
            batch_item_id=batch_item1.id,
            worker_id="worker_123"
        )
        db_session.add(session1)
        db_session.commit()
        
        # Try to create second session with same ID
        session2 = ProcessingSession(
            session_id=session_id,  # Same ID
            batch_item_id=batch_item2.id,
            worker_id="worker_456"
        )
        db_session.add(session2)
        
        with pytest.raises(IntegrityError):
            db_session.commit()

    # Index Tests

    def test_database_indexes_exist(self, db_engine):
        """Test that database indexes are created."""
        inspector = inspect(db_engine)
        
        # Check batch indexes
        batch_indexes = inspector.get_indexes('batches')
        index_names = [idx['name'] for idx in batch_indexes]
        
        assert 'idx_batch_id' in index_names
        assert 'idx_batch_status' in index_names
        assert 'idx_batch_priority' in index_names
        assert 'idx_batch_created_at' in index_names
        assert 'idx_batch_status_priority' in index_names
        
        # Check batch_items indexes
        batch_item_indexes = inspector.get_indexes('batch_items')
        batch_item_index_names = [idx['name'] for idx in batch_item_indexes]
        
        assert 'idx_batch_item_batch_id' in batch_item_index_names
        assert 'idx_batch_item_status' in batch_item_index_names
        assert 'idx_batch_item_priority' in batch_item_index_names
        
        # Check queue_items indexes
        queue_indexes = inspector.get_indexes('queue_items')
        queue_index_names = [idx['name'] for idx in queue_indexes]
        
        assert 'idx_queue_item_queue_name' in queue_index_names
        assert 'idx_queue_item_priority' in queue_index_names
        assert 'idx_queue_item_scheduled_at' in queue_index_names

    # Concurrent Access Tests

    def test_concurrent_batch_creation(self, db_engine):
        """Test concurrent batch creation."""
        def create_batch(worker_id):
            # Create separate session for each worker
            SessionLocal = sessionmaker(bind=db_engine)
            session = SessionLocal()
            
            try:
                batch = Batch(
                    batch_id=f"batch_{worker_id}_{uuid.uuid4()}",
                    name=f"Worker {worker_id} Batch",
                    total_items=1
                )
                session.add(batch)
                session.commit()
                return batch.id
            except Exception as e:
                session.rollback()
                raise e
            finally:
                session.close()
        
        # Create batches concurrently
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(create_batch, i) for i in range(10)]
            results = [future.result() for future in futures]
        
        # Verify all batches were created
        assert len(results) == 10
        assert all(isinstance(batch_id, int) for batch_id in results)
        assert len(set(results)) == 10  # All unique IDs

    def test_concurrent_queue_item_locking(self, db_engine):
        """Test concurrent queue item locking."""
        # Setup initial data
        SessionLocal = sessionmaker(bind=db_engine)
        session = SessionLocal()
        
        batch = Batch(
            batch_id="batch_" + str(uuid.uuid4()),
            total_items=1
        )
        session.add(batch)
        session.commit()
        
        batch_item = BatchItem(
            batch_id=batch.id,
            url="https://www.youtube.com/watch?v=test123"
        )
        session.add(batch_item)
        session.commit()
        
        queue_item = QueueItem(
            batch_item_id=batch_item.id,
            queue_name="test_queue",
            scheduled_at=datetime.utcnow() - timedelta(minutes=1)
        )
        session.add(queue_item)
        session.commit()
        
        queue_item_id = queue_item.id
        session.close()
        
        def try_lock_item(worker_id):
            """Try to lock the queue item."""
            session = SessionLocal()
            try:
                # Try to lock the item
                item = session.query(QueueItem).filter_by(id=queue_item_id).first()
                if item and not item.is_locked:
                    item.locked_at = datetime.utcnow()
                    item.locked_by = f"worker_{worker_id}"
                    item.lock_expires_at = datetime.utcnow() + timedelta(minutes=30)
                    session.commit()
                    return f"worker_{worker_id}"
                else:
                    return None
            except Exception as e:
                session.rollback()
                return None
            finally:
                session.close()
        
        # Try to lock concurrently
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(try_lock_item, i) for i in range(5)]
            results = [future.result() for future in futures]
        
        # Only one worker should successfully lock the item
        successful_locks = [r for r in results if r is not None]
        assert len(successful_locks) == 1

    # Utility Function Tests

    def test_get_batch_model_by_name(self):
        """Test get_batch_model_by_name utility function."""
        assert get_batch_model_by_name('Batch') == Batch
        assert get_batch_model_by_name('BatchItem') == BatchItem
        assert get_batch_model_by_name('QueueItem') == QueueItem
        assert get_batch_model_by_name('ProcessingSession') == ProcessingSession
        assert get_batch_model_by_name('NonExistent') is None

    def test_get_all_batch_models(self):
        """Test get_all_batch_models utility function."""
        models = get_all_batch_models()
        assert len(models) == 4
        assert Batch in models
        assert BatchItem in models
        assert QueueItem in models
        assert ProcessingSession in models

    def test_create_batch_tables(self, db_engine):
        """Test create_batch_tables utility function."""
        # Drop existing tables
        drop_batch_tables(db_engine)
        
        # Create tables
        create_batch_tables(db_engine)
        
        # Verify tables exist
        inspector = inspect(db_engine)
        table_names = inspector.get_table_names()
        
        assert 'batches' in table_names
        assert 'batch_items' in table_names
        assert 'queue_items' in table_names
        assert 'processing_sessions' in table_names

    def test_drop_batch_tables(self, db_engine):
        """Test drop_batch_tables utility function."""
        # Ensure tables exist
        create_batch_tables(db_engine)
        
        # Drop tables
        drop_batch_tables(db_engine)
        
        # Verify tables are gone
        inspector = inspect(db_engine)
        table_names = inspector.get_table_names()
        
        assert 'batches' not in table_names
        assert 'batch_items' not in table_names
        assert 'queue_items' not in table_names
        assert 'processing_sessions' not in table_names

    # Transaction Tests

    def test_transaction_rollback_on_error(self, db_session):
        """Test transaction rollback on error."""
        batch = Batch(
            batch_id="batch_" + str(uuid.uuid4()),
            total_items=1
        )
        db_session.add(batch)
        db_session.commit()
        
        # Start transaction
        try:
            batch_item = BatchItem(
                batch_id=batch.id,
                url="https://www.youtube.com/watch?v=test123"
            )
            db_session.add(batch_item)
            
            # This should cause an error
            bad_batch_item = BatchItem(
                batch_id=batch.id,
                url=""  # Empty URL should cause validation error
            )
            db_session.add(bad_batch_item)
            db_session.commit()
        except Exception:
            db_session.rollback()
        
        # Verify rollback - no batch items should exist
        assert db_session.query(BatchItem).filter_by(batch_id=batch.id).count() == 0

    def test_savepoint_rollback(self, db_session):
        """Test savepoint rollback functionality."""
        batch = Batch(
            batch_id="batch_" + str(uuid.uuid4()),
            total_items=2
        )
        db_session.add(batch)
        db_session.commit()
        
        # Create first item successfully
        batch_item1 = BatchItem(
            batch_id=batch.id,
            url="https://www.youtube.com/watch?v=test1"
        )
        db_session.add(batch_item1)
        db_session.commit()
        
        # Create savepoint
        savepoint = db_session.begin_nested()
        
        try:
            # This should succeed
            batch_item2 = BatchItem(
                batch_id=batch.id,
                url="https://www.youtube.com/watch?v=test2"
            )
            db_session.add(batch_item2)
            
            # This should fail
            bad_batch_item = BatchItem(
                batch_id=batch.id,
                url=""  # Empty URL
            )
            db_session.add(bad_batch_item)
            db_session.commit()
        except Exception:
            savepoint.rollback()
        
        # Verify first item still exists, but second doesn't
        items = db_session.query(BatchItem).filter_by(batch_id=batch.id).all()
        assert len(items) == 1
        assert items[0].url == "https://www.youtube.com/watch?v=test1"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])