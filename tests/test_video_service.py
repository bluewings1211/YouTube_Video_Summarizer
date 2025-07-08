"""
Comprehensive tests for VideoService class.

This test suite covers:
- Video service initialization
- Video creation and retrieval
- Transcript, summary, keyword, and segment operations
- Processing metadata operations
- Error handling and retry logic
- Database transaction management
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timezone
from sqlalchemy import create_engine, text, event
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.exc import IntegrityError, TimeoutError as SQLTimeoutError
from sqlalchemy.pool import StaticPool

from src.services.video_service import VideoService, VideoServiceError
from src.database.models import (
    Base, Video, Transcript, Summary, Keyword, TimestampedSegment, ProcessingMetadata
)
from src.database.exceptions import (
    DatabaseError, DatabaseConnectionError, DatabaseQueryError,
    DatabaseConstraintError, DatabaseTimeoutError
)


class TestAsyncSession:
    """Mock async session for testing."""
    
    def __init__(self):
        self.add = Mock()
        self.commit = AsyncMock()
        self.rollback = AsyncMock()
        self.close = AsyncMock()
        self.refresh = AsyncMock()
        self.execute = AsyncMock()
        self.delete = AsyncMock()
        self._queries = []
    
    def query(self, model):
        """Mock query method."""
        mock_query = Mock()
        mock_query.filter_by = Mock(return_value=mock_query)
        mock_query.where = Mock(return_value=mock_query)
        mock_query.first = Mock(return_value=None)
        mock_query.one_or_none = Mock(return_value=None)
        mock_query.scalar_one_or_none = Mock(return_value=None)
        mock_query.order_by = Mock(return_value=mock_query)
        return mock_query


@pytest.fixture
def mock_session():
    """Create a mock async session."""
    session = TestAsyncSession()
    return session


@pytest.fixture
def video_service(mock_session):
    """Create VideoService instance with mock session."""
    return VideoService(session=mock_session)


@pytest.fixture
def sample_video_data():
    """Sample video data for testing."""
    return {
        "video_id": "dQw4w9WgXcQ",
        "title": "Never Gonna Give You Up",
        "duration": 212,
        "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    }


@pytest.fixture
def sample_video_record():
    """Sample video record for testing."""
    video = Video(
        id=1,
        video_id="dQw4w9WgXcQ",
        title="Never Gonna Give You Up",
        duration=212,
        url="https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    )
    return video


class TestVideoServiceInitialization:
    """Test VideoService initialization."""
    
    def test_init_with_session(self, mock_session):
        """Test initialization with provided session."""
        service = VideoService(session=mock_session)
        assert service._session == mock_session
        assert service.max_retries == 3
        assert service.retry_delay == 1.0
    
    def test_init_without_session(self):
        """Test initialization without session."""
        service = VideoService()
        assert service._session is None
        assert service.max_retries == 3
        assert service.retry_delay == 1.0
    
    def test_init_with_custom_retry_params(self, mock_session):
        """Test initialization with custom retry parameters."""
        service = VideoService(session=mock_session, max_retries=5, retry_delay=2.0)
        assert service.max_retries == 5
        assert service.retry_delay == 2.0
    
    @pytest.mark.asyncio
    async def test_get_session_with_provided_session(self, mock_session):
        """Test getting session when provided."""
        service = VideoService(session=mock_session)
        session = await service._get_session()
        assert session == mock_session
    
    @pytest.mark.asyncio
    async def test_get_session_without_provided_session(self):
        """Test getting session when not provided."""
        service = VideoService()
        with pytest.raises(VideoServiceError, match="No database session provided"):
            await service._get_session()


class TestVideoCreation:
    """Test video creation functionality."""
    
    @pytest.mark.asyncio
    async def test_create_video_record_success(self, video_service, mock_session, sample_video_data):
        """Test successful video creation."""
        # Mock get_video_by_video_id to return None (video doesn't exist)
        with patch.object(video_service, 'get_video_by_video_id', return_value=None):
            result = await video_service.create_video_record(**sample_video_data)
        
        # Verify session operations
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()
        mock_session.refresh.assert_called_once()
        
        # Verify video object was created with correct data
        added_video = mock_session.add.call_args[0][0]
        assert isinstance(added_video, Video)
        assert added_video.video_id == sample_video_data["video_id"]
        assert added_video.title == sample_video_data["title"]
        assert added_video.duration == sample_video_data["duration"]
        assert added_video.url == sample_video_data["url"]
    
    @pytest.mark.asyncio
    async def test_create_video_record_already_exists(self, video_service, sample_video_data, sample_video_record):
        """Test creating video when it already exists."""
        # Mock get_video_by_video_id to return existing video
        with patch.object(video_service, 'get_video_by_video_id', return_value=sample_video_record):
            result = await video_service.create_video_record(**sample_video_data)
        
        # Should return existing video
        assert result == sample_video_record
    
    @pytest.mark.asyncio
    async def test_create_video_record_with_default_url(self, video_service, mock_session):
        """Test creating video with default URL generation."""
        video_data = {
            "video_id": "dQw4w9WgXcQ",
            "title": "Test Video"
        }
        
        with patch.object(video_service, 'get_video_by_video_id', return_value=None):
            await video_service.create_video_record(**video_data)
        
        # Verify URL was generated
        added_video = mock_session.add.call_args[0][0]
        assert added_video.url == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    
    @pytest.mark.asyncio
    async def test_create_video_record_integrity_error(self, video_service, mock_session, sample_video_data):
        """Test handling integrity error during video creation."""
        # Mock commit to raise IntegrityError
        mock_session.commit.side_effect = IntegrityError("UNIQUE constraint failed", None, None)
        
        with patch.object(video_service, 'get_video_by_video_id', return_value=None):
            with pytest.raises(VideoServiceError, match="Failed to create video record"):
                await video_service.create_video_record(**sample_video_data)
    
    @pytest.mark.asyncio
    async def test_create_video_record_timeout_error(self, video_service, mock_session, sample_video_data):
        """Test handling timeout error during video creation."""
        # Mock commit to raise TimeoutError
        mock_session.commit.side_effect = SQLTimeoutError("Connection timeout", None, None)
        
        with patch.object(video_service, 'get_video_by_video_id', return_value=None):
            with pytest.raises(VideoServiceError, match="Failed to create video record"):
                await video_service.create_video_record(**sample_video_data)


class TestVideoRetrieval:
    """Test video retrieval functionality."""
    
    @pytest.mark.asyncio
    async def test_get_video_by_video_id_success(self, video_service, mock_session, sample_video_record):
        """Test successful video retrieval by video ID."""
        # Mock execute to return video
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = sample_video_record
        mock_session.execute.return_value = mock_result
        
        result = await video_service.get_video_by_video_id("dQw4w9WgXcQ")
        
        assert result == sample_video_record
        mock_session.execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_video_by_video_id_not_found(self, video_service, mock_session):
        """Test video retrieval when video doesn't exist."""
        # Mock execute to return None
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result
        
        result = await video_service.get_video_by_video_id("nonexistent")
        
        assert result is None
        mock_session.execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_video_by_id_success(self, video_service, mock_session, sample_video_record):
        """Test successful video retrieval by database ID."""
        # Mock execute to return video with relationships
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = sample_video_record
        mock_session.execute.return_value = mock_result
        
        result = await video_service.get_video_by_id(1)
        
        assert result == sample_video_record
        mock_session.execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_video_exists_true(self, video_service, sample_video_record):
        """Test video_exists when video exists."""
        with patch.object(video_service, 'get_video_by_video_id', return_value=sample_video_record):
            result = await video_service.video_exists("dQw4w9WgXcQ")
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_video_exists_false(self, video_service):
        """Test video_exists when video doesn't exist."""
        with patch.object(video_service, 'get_video_by_video_id', return_value=None):
            result = await video_service.video_exists("nonexistent")
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_video_exists_error_handling(self, video_service):
        """Test video_exists error handling."""
        with patch.object(video_service, 'get_video_by_video_id', side_effect=VideoServiceError("Database error")):
            result = await video_service.video_exists("dQw4w9WgXcQ")
        
        assert result is False


class TestTranscriptOperations:
    """Test transcript-related operations."""
    
    @pytest.mark.asyncio
    async def test_save_transcript_success(self, video_service, mock_session, sample_video_record):
        """Test successful transcript saving."""
        with patch.object(video_service, 'get_video_by_video_id', return_value=sample_video_record):
            result = await video_service.save_transcript(
                video_id="dQw4w9WgXcQ",
                content="This is a test transcript",
                language="en"
            )
        
        # Verify session operations
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()
        mock_session.refresh.assert_called_once()
        
        # Verify transcript object
        added_transcript = mock_session.add.call_args[0][0]
        assert isinstance(added_transcript, Transcript)
        assert added_transcript.video_id == sample_video_record.id
        assert added_transcript.content == "This is a test transcript"
        assert added_transcript.language == "en"
    
    @pytest.mark.asyncio
    async def test_save_transcript_video_not_found(self, video_service, mock_session):
        """Test saving transcript when video doesn't exist."""
        with patch.object(video_service, 'get_video_by_video_id', return_value=None):
            with pytest.raises(VideoServiceError, match="Video dQw4w9WgXcQ not found"):
                await video_service.save_transcript(
                    video_id="dQw4w9WgXcQ",
                    content="Test transcript"
                )
    
    @pytest.mark.asyncio
    async def test_save_transcript_without_language(self, video_service, mock_session, sample_video_record):
        """Test saving transcript without language."""
        with patch.object(video_service, 'get_video_by_video_id', return_value=sample_video_record):
            await video_service.save_transcript(
                video_id="dQw4w9WgXcQ",
                content="Test transcript"
            )
        
        added_transcript = mock_session.add.call_args[0][0]
        assert added_transcript.language is None


class TestSummaryOperations:
    """Test summary-related operations."""
    
    @pytest.mark.asyncio
    async def test_save_summary_success(self, video_service, mock_session, sample_video_record):
        """Test successful summary saving."""
        with patch.object(video_service, 'get_video_by_video_id', return_value=sample_video_record):
            result = await video_service.save_summary(
                video_id="dQw4w9WgXcQ",
                content="This is a test summary",
                processing_time=15.5
            )
        
        # Verify session operations
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()
        mock_session.refresh.assert_called_once()
        
        # Verify summary object
        added_summary = mock_session.add.call_args[0][0]
        assert isinstance(added_summary, Summary)
        assert added_summary.video_id == sample_video_record.id
        assert added_summary.content == "This is a test summary"
        assert added_summary.processing_time == 15.5
    
    @pytest.mark.asyncio
    async def test_save_summary_without_processing_time(self, video_service, mock_session, sample_video_record):
        """Test saving summary without processing time."""
        with patch.object(video_service, 'get_video_by_video_id', return_value=sample_video_record):
            await video_service.save_summary(
                video_id="dQw4w9WgXcQ",
                content="Test summary"
            )
        
        added_summary = mock_session.add.call_args[0][0]
        assert added_summary.processing_time is None


class TestKeywordOperations:
    """Test keyword-related operations."""
    
    @pytest.mark.asyncio
    async def test_save_keywords_success(self, video_service, mock_session, sample_video_record):
        """Test successful keywords saving."""
        keywords_data = [
            {"keyword": "test", "score": 0.9},
            {"keyword": "video", "score": 0.8}
        ]
        
        with patch.object(video_service, 'get_video_by_video_id', return_value=sample_video_record):
            result = await video_service.save_keywords(
                video_id="dQw4w9WgXcQ",
                keywords_data=keywords_data
            )
        
        # Verify session operations
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()
        mock_session.refresh.assert_called_once()
        
        # Verify keyword object
        added_keyword = mock_session.add.call_args[0][0]
        assert isinstance(added_keyword, Keyword)
        assert added_keyword.video_id == sample_video_record.id
        assert added_keyword.keywords_json == keywords_data


class TestTimestampedSegmentOperations:
    """Test timestamped segment operations."""
    
    @pytest.mark.asyncio
    async def test_save_timestamped_segments_success(self, video_service, mock_session, sample_video_record):
        """Test successful timestamped segments saving."""
        segments_data = [
            {"start": 0, "end": 30, "text": "First segment"},
            {"start": 30, "end": 60, "text": "Second segment"}
        ]
        
        with patch.object(video_service, 'get_video_by_video_id', return_value=sample_video_record):
            result = await video_service.save_timestamped_segments(
                video_id="dQw4w9WgXcQ",
                segments_data=segments_data
            )
        
        # Verify session operations
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()
        mock_session.refresh.assert_called_once()
        
        # Verify segment object
        added_segment = mock_session.add.call_args[0][0]
        assert isinstance(added_segment, TimestampedSegment)
        assert added_segment.video_id == sample_video_record.id
        assert added_segment.segments_json == segments_data


class TestProcessingMetadataOperations:
    """Test processing metadata operations."""
    
    @pytest.mark.asyncio
    async def test_save_processing_metadata_success(self, video_service, mock_session, sample_video_record):
        """Test successful processing metadata saving."""
        workflow_params = {
            "model": "gpt-3.5-turbo",
            "temperature": 0.7
        }
        
        with patch.object(video_service, 'get_video_by_video_id', return_value=sample_video_record):
            result = await video_service.save_processing_metadata(
                video_id="dQw4w9WgXcQ",
                workflow_params=workflow_params,
                status="completed",
                error_info="No errors"
            )
        
        # Verify session operations
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()
        mock_session.refresh.assert_called_once()
        
        # Verify metadata object
        added_metadata = mock_session.add.call_args[0][0]
        assert isinstance(added_metadata, ProcessingMetadata)
        assert added_metadata.video_id == sample_video_record.id
        assert added_metadata.workflow_params == workflow_params
        assert added_metadata.status == "completed"
        assert added_metadata.error_info == "No errors"
    
    @pytest.mark.asyncio
    async def test_save_processing_metadata_with_defaults(self, video_service, mock_session, sample_video_record):
        """Test saving processing metadata with default values."""
        with patch.object(video_service, 'get_video_by_video_id', return_value=sample_video_record):
            await video_service.save_processing_metadata(video_id="dQw4w9WgXcQ")
        
        added_metadata = mock_session.add.call_args[0][0]
        assert added_metadata.workflow_params is None
        assert added_metadata.status == "pending"
        assert added_metadata.error_info is None
    
    @pytest.mark.asyncio
    async def test_update_processing_status_success(self, video_service, mock_session, sample_video_record):
        """Test successful processing status update."""
        # Mock the execute calls for update and select
        mock_session.execute.return_value = Mock()
        
        # Mock the select result
        mock_metadata = ProcessingMetadata(
            id=1,
            video_id=sample_video_record.id,
            status="completed"
        )
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = mock_metadata
        mock_session.execute.return_value = mock_result
        
        with patch.object(video_service, 'get_video_by_video_id', return_value=sample_video_record):
            result = await video_service.update_processing_status(
                video_id="dQw4w9WgXcQ",
                status="completed",
                error_info="Processing completed successfully"
            )
        
        # Verify session operations
        assert mock_session.execute.call_count == 2  # Update and select
        mock_session.commit.assert_called_once()
        assert result == mock_metadata
    
    @pytest.mark.asyncio
    async def test_get_processing_status_success(self, video_service, mock_session, sample_video_record):
        """Test successful processing status retrieval."""
        mock_metadata = ProcessingMetadata(
            id=1,
            video_id=sample_video_record.id,
            status="completed"
        )
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = mock_metadata
        mock_session.execute.return_value = mock_result
        
        with patch.object(video_service, 'get_video_by_video_id', return_value=sample_video_record):
            result = await video_service.get_processing_status("dQw4w9WgXcQ")
        
        assert result == "completed"
    
    @pytest.mark.asyncio
    async def test_get_processing_status_not_found(self, video_service, mock_session, sample_video_record):
        """Test processing status retrieval when not found."""
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result
        
        with patch.object(video_service, 'get_video_by_video_id', return_value=sample_video_record):
            result = await video_service.get_processing_status("dQw4w9WgXcQ")
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_get_processing_status_video_not_found(self, video_service):
        """Test processing status retrieval when video doesn't exist."""
        with patch.object(video_service, 'get_video_by_video_id', return_value=None):
            result = await video_service.get_processing_status("nonexistent")
        
        assert result is None


class TestVideoOperations:
    """Test video-level operations."""
    
    @pytest.mark.asyncio
    async def test_delete_video_data_success(self, video_service, mock_session, sample_video_record):
        """Test successful video deletion."""
        with patch.object(video_service, 'get_video_by_video_id', return_value=sample_video_record):
            result = await video_service.delete_video_data("dQw4w9WgXcQ")
        
        # Verify session operations
        mock_session.delete.assert_called_once_with(sample_video_record)
        mock_session.commit.assert_called_once()
        assert result is True
    
    @pytest.mark.asyncio
    async def test_delete_video_data_not_found(self, video_service, mock_session):
        """Test video deletion when video doesn't exist."""
        with patch.object(video_service, 'get_video_by_video_id', return_value=None):
            result = await video_service.delete_video_data("nonexistent")
        
        # Should not call delete or commit
        mock_session.delete.assert_not_called()
        mock_session.commit.assert_not_called()
        assert result is False


class TestRetryLogic:
    """Test retry logic and error handling."""
    
    @pytest.mark.asyncio
    async def test_retry_database_operation_success(self, video_service):
        """Test successful operation on first try."""
        mock_operation = AsyncMock(return_value="success")
        
        result = await video_service._retry_database_operation(mock_operation, "arg1", key="value")
        
        assert result == "success"
        mock_operation.assert_called_once_with("arg1", key="value")
    
    @pytest.mark.asyncio
    async def test_retry_database_operation_success_after_retries(self, video_service):
        """Test successful operation after retries."""
        mock_operation = AsyncMock(side_effect=[
            DatabaseConnectionError("Connection failed"),
            DatabaseConnectionError("Connection failed"),
            "success"
        ])
        
        # Reduce retry delay for faster testing
        video_service.retry_delay = 0.01
        
        result = await video_service._retry_database_operation(mock_operation)
        
        assert result == "success"
        assert mock_operation.call_count == 3
    
    @pytest.mark.asyncio
    async def test_retry_database_operation_max_retries_exceeded(self, video_service):
        """Test operation failing after max retries."""
        mock_operation = AsyncMock(side_effect=DatabaseConnectionError("Connection failed"))
        
        # Reduce retry delay for faster testing
        video_service.retry_delay = 0.01
        video_service.max_retries = 2
        
        with pytest.raises(DatabaseConnectionError, match="Connection failed"):
            await video_service._retry_database_operation(mock_operation)
        
        # Should try initial + 2 retries = 3 total
        assert mock_operation.call_count == 3
    
    @pytest.mark.asyncio
    async def test_retry_database_operation_non_retryable_error(self, video_service):
        """Test non-retryable error handling."""
        mock_operation = AsyncMock(side_effect=DatabaseConstraintError("Constraint violation"))
        
        with pytest.raises(DatabaseConstraintError, match="Constraint violation"):
            await video_service._retry_database_operation(mock_operation)
        
        # Should not retry for constraint errors
        assert mock_operation.call_count == 1


class TestServiceUtilities:
    """Test service utility functions."""
    
    def test_get_video_service_dependency_injection(self, mock_session):
        """Test dependency injection function."""
        from src.services.video_service import get_video_service
        
        service = get_video_service(mock_session)
        
        assert isinstance(service, VideoService)
        assert service._session == mock_session
    
    @pytest.mark.asyncio
    async def test_create_video_service_with_session(self):
        """Test creating video service with session."""
        from src.services.video_service import create_video_service_with_session
        
        service = await create_video_service_with_session()
        
        assert isinstance(service, VideoService)
        assert service._session is None  # Should be None when created this way


class TestErrorHandling:
    """Test comprehensive error handling scenarios."""
    
    @pytest.mark.asyncio
    async def test_handle_sqlalchemy_error_in_create_video(self, video_service, mock_session):
        """Test SQLAlchemy error handling in create_video_record."""
        from sqlalchemy.exc import SQLAlchemyError
        
        mock_session.commit.side_effect = SQLAlchemyError("Database error")
        
        with patch.object(video_service, 'get_video_by_video_id', return_value=None):
            with pytest.raises(VideoServiceError, match="Failed to create video record"):
                await video_service.create_video_record(
                    video_id="dQw4w9WgXcQ",
                    title="Test Video"
                )
    
    @pytest.mark.asyncio
    async def test_handle_unexpected_error_in_create_video(self, video_service, mock_session):
        """Test unexpected error handling in create_video_record."""
        mock_session.commit.side_effect = Exception("Unexpected error")
        
        with patch.object(video_service, 'get_video_by_video_id', return_value=None):
            with pytest.raises(VideoServiceError, match="Unexpected error"):
                await video_service.create_video_record(
                    video_id="dQw4w9WgXcQ",
                    title="Test Video"
                )
    
    @pytest.mark.asyncio
    async def test_handle_error_in_get_video_by_video_id(self, video_service, mock_session):
        """Test error handling in get_video_by_video_id."""
        from sqlalchemy.exc import SQLAlchemyError
        
        mock_session.execute.side_effect = SQLAlchemyError("Database error")
        
        with pytest.raises(VideoServiceError, match="Failed to get video"):
            await video_service.get_video_by_video_id("dQw4w9WgXcQ")
    
    @pytest.mark.asyncio
    async def test_handle_error_in_save_transcript(self, video_service, mock_session, sample_video_record):
        """Test error handling in save_transcript."""
        from sqlalchemy.exc import SQLAlchemyError
        
        mock_session.commit.side_effect = SQLAlchemyError("Database error")
        
        with patch.object(video_service, 'get_video_by_video_id', return_value=sample_video_record):
            with pytest.raises(VideoServiceError, match="Failed to save transcript"):
                await video_service.save_transcript(
                    video_id="dQw4w9WgXcQ",
                    content="Test transcript"
                )
    
    @pytest.mark.asyncio
    async def test_handle_error_in_delete_video_data(self, video_service, mock_session, sample_video_record):
        """Test error handling in delete_video_data."""
        from sqlalchemy.exc import SQLAlchemyError
        
        mock_session.commit.side_effect = SQLAlchemyError("Database error")
        
        with patch.object(video_service, 'get_video_by_video_id', return_value=sample_video_record):
            with pytest.raises(VideoServiceError, match="Failed to delete video"):
                await video_service.delete_video_data("dQw4w9WgXcQ")