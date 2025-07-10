"""
Tests for the status service module.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from ..database.status_models import (
    ProcessingStatus, StatusHistory, ProcessingStatusType, 
    ProcessingPriority, StatusChangeType
)
from ..services.status_service import StatusService


class TestStatusService:
    """Test cases for StatusService."""
    
    @pytest.fixture
    def mock_db_session(self):
        """Mock database session fixture."""
        session = Mock(spec=Session)
        session.add = Mock()
        session.commit = Mock()
        session.rollback = Mock()
        session.close = Mock()
        session.query = Mock()
        return session
    
    @pytest.fixture
    def status_service(self, mock_db_session):
        """StatusService fixture."""
        return StatusService(db_session=mock_db_session)
    
    def test_init_with_session(self, mock_db_session):
        """Test StatusService initialization with session."""
        service = StatusService(db_session=mock_db_session)
        assert service.db_session == mock_db_session
        assert service._should_close_session is False
    
    def test_init_without_session(self):
        """Test StatusService initialization without session."""
        service = StatusService()
        assert service.db_session is None
        assert service._should_close_session is True
    
    def test_context_manager_with_session(self, mock_db_session):
        """Test context manager with provided session."""
        service = StatusService(db_session=mock_db_session)
        
        with service as s:
            assert s.db_session == mock_db_session
        
        mock_db_session.close.assert_not_called()
    
    @patch('src.services.status_service.get_db_session')
    def test_context_manager_without_session(self, mock_get_db_session):
        """Test context manager without provided session."""
        mock_session = Mock(spec=Session)
        mock_get_db_session.return_value = mock_session
        
        service = StatusService()
        
        with service as s:
            assert s.db_session == mock_session
        
        mock_session.close.assert_called_once()
    
    def test_create_processing_status_success(self, status_service, mock_db_session):
        """Test successful processing status creation."""
        # Setup
        mock_status = Mock(spec=ProcessingStatus)
        mock_status.id = 1
        
        # Mock the session.add to capture the added object
        added_status = None
        def capture_add(obj):
            nonlocal added_status
            added_status = obj
        
        mock_db_session.add = capture_add
        
        # Execute
        result = status_service.create_processing_status(
            video_id=123,
            priority=ProcessingPriority.HIGH,
            total_steps=5,
            max_retries=3,
            external_id="test-123"
        )
        
        # Verify
        assert result is not None
        assert result.video_id == 123
        assert result.priority == ProcessingPriority.HIGH
        assert result.total_steps == 5
        assert result.max_retries == 3
        assert result.external_id == "test-123"
        assert result.status == ProcessingStatusType.QUEUED
        assert result.progress_percentage == 0.0
        assert result.completed_steps == 0
        assert result.status_id.startswith("status_")
        
        mock_db_session.commit.assert_called_once()
    
    def test_create_processing_status_integrity_error(self, status_service, mock_db_session):
        """Test processing status creation with integrity error."""
        # Setup
        mock_db_session.add.side_effect = IntegrityError("Test error", None, None)
        
        # Execute & Verify
        with pytest.raises(ValueError, match="Failed to create processing status"):
            status_service.create_processing_status(video_id=123)
        
        mock_db_session.rollback.assert_called_once()
    
    def test_create_processing_status_sqlalchemy_error(self, status_service, mock_db_session):
        """Test processing status creation with SQLAlchemy error."""
        # Setup
        mock_db_session.add.side_effect = SQLAlchemyError("Database error")
        
        # Execute & Verify
        with pytest.raises(SQLAlchemyError):
            status_service.create_processing_status(video_id=123)
        
        mock_db_session.rollback.assert_called_once()
    
    def test_update_status_success(self, status_service, mock_db_session):
        """Test successful status update."""
        # Setup
        mock_status = Mock(spec=ProcessingStatus)
        mock_status.id = 1
        mock_status.status = ProcessingStatusType.QUEUED
        mock_status.progress_percentage = 0.0
        
        with patch.object(status_service, 'get_processing_status', return_value=mock_status):
            with patch.object(status_service, '_create_status_history'):
                # Execute
                result = status_service.update_status(
                    status_id="test-status-123",
                    new_status=ProcessingStatusType.YOUTUBE_METADATA,
                    progress_percentage=25.0,
                    current_step="Processing metadata",
                    worker_id="worker-1"
                )
                
                # Verify
                assert result == mock_status
                assert mock_status.status == ProcessingStatusType.YOUTUBE_METADATA
                assert mock_status.progress_percentage == 25.0
                assert mock_status.current_step == "Processing metadata"
                assert mock_status.worker_id == "worker-1"
                assert mock_status.started_at is not None
                
                mock_db_session.commit.assert_called_once()
    
    def test_update_status_not_found(self, status_service, mock_db_session):
        """Test status update with non-existent status."""
        # Setup
        with patch.object(status_service, 'get_processing_status', return_value=None):
            # Execute & Verify
            with pytest.raises(ValueError, match="Processing status not found"):
                status_service.update_status(
                    status_id="non-existent",
                    new_status=ProcessingStatusType.COMPLETED
                )
    
    def test_update_status_completion(self, status_service, mock_db_session):
        """Test status update to completion."""
        # Setup
        mock_status = Mock(spec=ProcessingStatus)
        mock_status.id = 1
        mock_status.status = ProcessingStatusType.FINALIZING
        mock_status.progress_percentage = 95.0
        
        with patch.object(status_service, 'get_processing_status', return_value=mock_status):
            with patch.object(status_service, '_create_status_history'):
                # Execute
                result = status_service.update_status(
                    status_id="test-status-123",
                    new_status=ProcessingStatusType.COMPLETED,
                    progress_percentage=100.0
                )
                
                # Verify
                assert result == mock_status
                assert mock_status.status == ProcessingStatusType.COMPLETED
                assert mock_status.progress_percentage == 100.0
                assert mock_status.completed_at is not None
    
    def test_update_progress_success(self, status_service, mock_db_session):
        """Test successful progress update."""
        # Setup
        mock_status = Mock(spec=ProcessingStatus)
        mock_status.id = 1
        mock_status.status = ProcessingStatusType.YOUTUBE_METADATA
        mock_status.progress_percentage = 25.0
        mock_status.processing_metadata = {"step": "metadata"}
        
        with patch.object(status_service, 'get_processing_status', return_value=mock_status):
            with patch.object(status_service, '_create_status_history'):
                # Execute
                result = status_service.update_progress(
                    status_id="test-status-123",
                    progress_percentage=50.0,
                    current_step="Processing transcript",
                    completed_steps=2,
                    processing_metadata={"step": "transcript"}
                )
                
                # Verify
                assert result == mock_status
                assert mock_status.progress_percentage == 50.0
                assert mock_status.current_step == "Processing transcript"
                assert mock_status.completed_steps == 2
                assert mock_status.processing_metadata == {"step": "transcript"}
                assert mock_status.heartbeat_at is not None
                
                mock_db_session.commit.assert_called_once()
    
    def test_record_error_with_retry(self, status_service, mock_db_session):
        """Test error recording with retry capability."""
        # Setup
        mock_status = Mock(spec=ProcessingStatus)
        mock_status.id = 1
        mock_status.status = ProcessingStatusType.YOUTUBE_METADATA
        mock_status.progress_percentage = 25.0
        mock_status.retry_count = 0
        mock_status.can_retry = True
        
        with patch.object(status_service, 'get_processing_status', return_value=mock_status):
            with patch.object(status_service, '_create_status_history'):
                # Execute
                result = status_service.record_error(
                    status_id="test-status-123",
                    error_info="Connection timeout",
                    worker_id="worker-1"
                )
                
                # Verify
                assert result == mock_status
                assert mock_status.error_info == "Connection timeout"
                assert mock_status.retry_count == 1
                assert mock_status.status == ProcessingStatusType.RETRY_PENDING
                
                mock_db_session.commit.assert_called_once()
    
    def test_record_error_no_retry(self, status_service, mock_db_session):
        """Test error recording without retry capability."""
        # Setup
        mock_status = Mock(spec=ProcessingStatus)
        mock_status.id = 1
        mock_status.status = ProcessingStatusType.YOUTUBE_METADATA
        mock_status.progress_percentage = 25.0
        mock_status.retry_count = 3
        mock_status.can_retry = False
        
        with patch.object(status_service, 'get_processing_status', return_value=mock_status):
            with patch.object(status_service, '_create_status_history'):
                # Execute
                result = status_service.record_error(
                    status_id="test-status-123",
                    error_info="Fatal error",
                    worker_id="worker-1"
                )
                
                # Verify
                assert result == mock_status
                assert mock_status.error_info == "Fatal error"
                assert mock_status.retry_count == 4
                assert mock_status.status == ProcessingStatusType.FAILED
                assert mock_status.completed_at is not None
    
    def test_heartbeat_success(self, status_service, mock_db_session):
        """Test successful heartbeat update."""
        # Setup
        mock_status = Mock(spec=ProcessingStatus)
        mock_status.id = 1
        mock_status.status = ProcessingStatusType.YOUTUBE_METADATA
        mock_status.progress_percentage = 25.0
        
        with patch.object(status_service, 'get_processing_status', return_value=mock_status):
            # Execute
            result = status_service.heartbeat(
                status_id="test-status-123",
                worker_id="worker-1",
                progress_percentage=30.0,
                current_step="Still processing"
            )
            
            # Verify
            assert result == mock_status
            assert mock_status.worker_id == "worker-1"
            assert mock_status.progress_percentage == 30.0
            assert mock_status.current_step == "Still processing"
            assert mock_status.heartbeat_at is not None
            
            mock_db_session.commit.assert_called_once()
    
    def test_get_processing_status_found(self, status_service, mock_db_session):
        """Test getting processing status that exists."""
        # Setup
        mock_status = Mock(spec=ProcessingStatus)
        mock_query = Mock()
        mock_query.filter.return_value.first.return_value = mock_status
        mock_db_session.query.return_value = mock_query
        
        # Execute
        result = status_service.get_processing_status("test-status-123")
        
        # Verify
        assert result == mock_status
        mock_db_session.query.assert_called_once_with(ProcessingStatus)
    
    def test_get_processing_status_not_found(self, status_service, mock_db_session):
        """Test getting processing status that doesn't exist."""
        # Setup
        mock_query = Mock()
        mock_query.filter.return_value.first.return_value = None
        mock_db_session.query.return_value = mock_query
        
        # Execute
        result = status_service.get_processing_status("non-existent")
        
        # Verify
        assert result is None
    
    def test_get_status_by_video_id(self, status_service, mock_db_session):
        """Test getting status by video ID."""
        # Setup
        mock_status = Mock(spec=ProcessingStatus)
        mock_query = Mock()
        mock_query.filter.return_value.order_by.return_value.first.return_value = mock_status
        mock_db_session.query.return_value = mock_query
        
        # Execute
        result = status_service.get_status_by_video_id(123)
        
        # Verify
        assert result == mock_status
        mock_db_session.query.assert_called_once_with(ProcessingStatus)
    
    def test_get_active_statuses(self, status_service, mock_db_session):
        """Test getting active statuses."""
        # Setup
        mock_statuses = [Mock(spec=ProcessingStatus) for _ in range(3)]
        mock_query = Mock()
        mock_query.filter.return_value.order_by.return_value.all.return_value = mock_statuses
        mock_db_session.query.return_value = mock_query
        
        # Execute
        result = status_service.get_active_statuses()
        
        # Verify
        assert result == mock_statuses
        assert len(result) == 3
        mock_db_session.query.assert_called_once_with(ProcessingStatus)
    
    def test_get_active_statuses_with_worker_filter(self, status_service, mock_db_session):
        """Test getting active statuses with worker filter."""
        # Setup
        mock_statuses = [Mock(spec=ProcessingStatus)]
        mock_query = Mock()
        mock_query.filter.return_value.filter.return_value.order_by.return_value.all.return_value = mock_statuses
        mock_db_session.query.return_value = mock_query
        
        # Execute
        result = status_service.get_active_statuses(worker_id="worker-1")
        
        # Verify
        assert result == mock_statuses
        assert len(result) == 1
    
    def test_get_stale_statuses(self, status_service, mock_db_session):
        """Test getting stale statuses."""
        # Setup
        mock_statuses = [Mock(spec=ProcessingStatus) for _ in range(2)]
        mock_query = Mock()
        mock_query.filter.return_value.all.return_value = mock_statuses
        mock_db_session.query.return_value = mock_query
        
        # Execute
        result = status_service.get_stale_statuses(timeout_seconds=300)
        
        # Verify
        assert result == mock_statuses
        assert len(result) == 2
        mock_db_session.query.assert_called_once_with(ProcessingStatus)
    
    def test_get_status_history(self, status_service, mock_db_session):
        """Test getting status history."""
        # Setup
        mock_status = Mock(spec=ProcessingStatus)
        mock_status.id = 1
        mock_history = [Mock(spec=StatusHistory) for _ in range(5)]
        
        mock_query = Mock()
        mock_query.filter.return_value.order_by.return_value.all.return_value = mock_history
        mock_db_session.query.return_value = mock_query
        
        with patch.object(status_service, 'get_processing_status', return_value=mock_status):
            # Execute
            result = status_service.get_status_history("test-status-123")
            
            # Verify
            assert result == mock_history
            assert len(result) == 5
    
    def test_get_status_history_not_found(self, status_service, mock_db_session):
        """Test getting history for non-existent status."""
        # Setup
        with patch.object(status_service, 'get_processing_status', return_value=None):
            # Execute & Verify
            with pytest.raises(ValueError, match="Processing status not found"):
                status_service.get_status_history("non-existent")
    
    def test_cleanup_old_statuses(self, status_service, mock_db_session):
        """Test cleaning up old statuses."""
        # Setup
        mock_query = Mock()
        mock_query.filter.return_value.count.return_value = 5
        mock_query.filter.return_value.delete.return_value = 5
        mock_db_session.query.return_value = mock_query
        
        # Execute
        result = status_service.cleanup_old_statuses(days_old=30)
        
        # Verify
        assert result == 5
        mock_db_session.commit.assert_called_once()
    
    def test_cleanup_old_statuses_error(self, status_service, mock_db_session):
        """Test cleanup with database error."""
        # Setup
        mock_query = Mock()
        mock_query.filter.return_value.count.side_effect = SQLAlchemyError("Database error")
        mock_db_session.query.return_value = mock_query
        
        # Execute & Verify
        with pytest.raises(SQLAlchemyError):
            status_service.cleanup_old_statuses(days_old=30)
        
        mock_db_session.rollback.assert_called_once()
    
    def test_create_status_history(self, status_service, mock_db_session):
        """Test creating status history entry."""
        # Setup
        mock_last_history = Mock(spec=StatusHistory)
        mock_last_history.created_at = datetime.utcnow() - timedelta(minutes=5)
        
        mock_query = Mock()
        mock_query.filter.return_value.order_by.return_value.first.return_value = mock_last_history
        mock_db_session.query.return_value = mock_query
        
        # Execute
        result = status_service._create_status_history(
            processing_status_id=1,
            change_type=StatusChangeType.STATUS_UPDATE,
            previous_status=ProcessingStatusType.QUEUED,
            new_status=ProcessingStatusType.YOUTUBE_METADATA,
            previous_progress=0.0,
            new_progress=25.0,
            change_reason="Status updated"
        )
        
        # Verify
        assert result is not None
        assert result.processing_status_id == 1
        assert result.change_type == StatusChangeType.STATUS_UPDATE
        assert result.previous_status == ProcessingStatusType.QUEUED
        assert result.new_status == ProcessingStatusType.YOUTUBE_METADATA
        assert result.previous_progress == 0.0
        assert result.new_progress == 25.0
        assert result.change_reason == "Status updated"
        assert result.duration_seconds is not None
        
        mock_db_session.add.assert_called_once()
        mock_db_session.commit.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])