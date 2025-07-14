"""
Comprehensive API endpoint tests for batch processing operations.

This test suite provides comprehensive testing for all batch processing API endpoints including:
- Batch creation and management endpoints
- Batch listing and pagination
- Batch progress and statistics
- Queue management endpoints
- Processing session management
- Error handling and validation
- Authentication and authorization
- HTTP status codes and response formats
- Request/response serialization
- Edge cases and error conditions
"""

import pytest
import json
import uuid
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List
from fastapi.testclient import TestClient
from fastapi import FastAPI, status
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.database.models import Base, Video
from src.database.batch_models import (
    Batch, BatchItem, QueueItem, ProcessingSession,
    BatchStatus, BatchItemStatus, BatchPriority
)
from src.services.batch_service import (
    BatchService, BatchCreateRequest, BatchItemResult, BatchServiceError,
    BatchProgressInfo
)
from src.api.batch import router as batch_router
from src.api.batch import (
    BatchStatusEnum, BatchItemStatusEnum, BatchPriorityEnum,
    CreateBatchRequest, StartBatchRequest, CancelBatchRequest,
    RetryBatchItemRequest, UpdateBatchRequest,
    BatchResponse, BatchListResponse, BatchProgressResponse,
    BatchStatisticsResponse, ErrorResponse
)


class TestBatchAPIEndpoints:
    """Comprehensive API endpoint tests for batch processing."""

    @pytest.fixture(scope="function")
    def app(self):
        """Create FastAPI application with batch router."""
        app = FastAPI()
        app.include_router(batch_router)
        return app

    @pytest.fixture(scope="function")
    def client(self, app):
        """Create test client."""
        return TestClient(app)

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
    def mock_batch_service(self, db_session):
        """Create mock batch service."""
        service = Mock(spec=BatchService)
        service._session = db_session
        return service

    @pytest.fixture(scope="function")
    def sample_batch_data(self):
        """Create sample batch data for testing."""
        return {
            "name": "Test Batch",
            "description": "Test batch description",
            "urls": [
                "https://www.youtube.com/watch?v=test123",
                "https://www.youtube.com/watch?v=test456"
            ],
            "priority": "normal",
            "webhook_url": "https://webhook.example.com/batch",
            "batch_metadata": {"test_key": "test_value"}
        }

    @pytest.fixture(scope="function")
    def sample_batch_response(self):
        """Create sample batch response for testing."""
        return {
            "id": 1,
            "batch_id": "batch_123",
            "name": "Test Batch",
            "description": "Test batch description",
            "status": "pending",
            "priority": "normal",
            "total_items": 2,
            "completed_items": 0,
            "failed_items": 0,
            "pending_items": 2,
            "progress_percentage": 0.0,
            "created_at": "2023-01-01T00:00:00Z",
            "updated_at": "2023-01-01T00:00:00Z",
            "started_at": None,
            "completed_at": None,
            "webhook_url": "https://webhook.example.com/batch",
            "batch_metadata": {"test_key": "test_value"},
            "error_info": None,
            "is_completed": False,
            "is_failed": False,
            "batch_items": []
        }

    # Batch Creation Tests

    @patch('src.api.batch.get_batch_service')
    @patch('src.api.batch.get_database_session_dependency')
    def test_create_batch_success(self, mock_get_db, mock_get_service, 
                                 client, mock_batch_service, sample_batch_data):
        """Test successful batch creation."""
        # Setup mocks
        mock_get_service.return_value = mock_batch_service
        mock_get_db.return_value = mock_batch_service._session
        
        # Mock batch service response
        mock_batch = Mock()
        mock_batch.id = 1
        mock_batch.batch_id = "batch_123"
        mock_batch.name = "Test Batch"
        mock_batch.description = "Test batch description"
        mock_batch.status = BatchStatus.PENDING
        mock_batch.priority = BatchPriority.NORMAL
        mock_batch.total_items = 2
        mock_batch.completed_items = 0
        mock_batch.failed_items = 0
        mock_batch.pending_items = 2
        mock_batch.progress_percentage = 0.0
        mock_batch.created_at = datetime(2023, 1, 1)
        mock_batch.updated_at = datetime(2023, 1, 1)
        mock_batch.started_at = None
        mock_batch.completed_at = None
        mock_batch.webhook_url = "https://webhook.example.com/batch"
        mock_batch.batch_metadata = {"test_key": "test_value"}
        mock_batch.error_info = None
        mock_batch.is_completed = False
        mock_batch.is_failed = False
        mock_batch.batch_items = []
        
        mock_batch_service.create_batch.return_value = mock_batch
        
        # Mock URL validation
        with patch('src.api.batch.validate_youtube_url_detailed') as mock_validate:
            mock_validate.return_value = Mock(is_valid=True, error_message=None)
            
            # Make request
            response = client.post("/api/v1/batch/batches", json=sample_batch_data)
            
            # Verify response
            assert response.status_code == status.HTTP_201_CREATED
            response_data = response.json()
            assert response_data["batch_id"] == "batch_123"
            assert response_data["name"] == "Test Batch"
            assert response_data["status"] == "pending"
            assert response_data["total_items"] == 2
            
            # Verify service was called correctly
            mock_batch_service.create_batch.assert_called_once()
            call_args = mock_batch_service.create_batch.call_args[0][0]
            assert call_args.name == "Test Batch"
            assert call_args.urls == sample_batch_data["urls"]

    @patch('src.api.batch.get_batch_service')
    @patch('src.api.batch.get_database_session_dependency')
    def test_create_batch_invalid_urls(self, mock_get_db, mock_get_service,
                                      client, mock_batch_service):
        """Test batch creation with invalid URLs."""
        # Setup mocks
        mock_get_service.return_value = mock_batch_service
        mock_get_db.return_value = mock_batch_service._session
        
        # Test data with invalid URLs
        invalid_data = {
            "name": "Test Batch",
            "urls": [
                "not-a-url",
                "https://example.com/not-youtube"
            ]
        }
        
        # Mock URL validation to return invalid
        with patch('src.api.batch.validate_youtube_url_detailed') as mock_validate:
            mock_validate.return_value = Mock(is_valid=False, error_message="Invalid YouTube URL")
            
            # Make request
            response = client.post("/api/v1/batch/batches", json=invalid_data)
            
            # Verify error response
            assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
            response_data = response.json()
            assert "detail" in response_data
            assert "Invalid YouTube URL" in str(response_data["detail"])

    @patch('src.api.batch.get_batch_service')
    @patch('src.api.batch.get_database_session_dependency')
    def test_create_batch_empty_urls(self, mock_get_db, mock_get_service,
                                    client, mock_batch_service):
        """Test batch creation with empty URLs list."""
        # Setup mocks
        mock_get_service.return_value = mock_batch_service
        mock_get_db.return_value = mock_batch_service._session
        
        # Test data with empty URLs
        invalid_data = {
            "name": "Test Batch",
            "urls": []
        }
        
        # Make request
        response = client.post("/api/v1/batch/batches", json=invalid_data)
        
        # Verify error response
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        response_data = response.json()
        assert "detail" in response_data

    @patch('src.api.batch.get_batch_service')
    @patch('src.api.batch.get_database_session_dependency')
    def test_create_batch_service_error(self, mock_get_db, mock_get_service,
                                       client, mock_batch_service, sample_batch_data):
        """Test batch creation with service error."""
        # Setup mocks
        mock_get_service.return_value = mock_batch_service
        mock_get_db.return_value = mock_batch_service._session
        
        # Mock service to raise error
        mock_batch_service.create_batch.side_effect = BatchServiceError("Service error")
        
        # Mock URL validation
        with patch('src.api.batch.validate_youtube_url_detailed') as mock_validate:
            mock_validate.return_value = Mock(is_valid=True, error_message=None)
            
            # Make request
            response = client.post("/api/v1/batch/batches", json=sample_batch_data)
            
            # Verify error response
            assert response.status_code == status.HTTP_400_BAD_REQUEST
            response_data = response.json()
            assert "Failed to create batch" in response_data["detail"]

    # Batch Listing Tests

    @patch('src.api.batch.get_batch_service')
    @patch('src.api.batch.get_database_session_dependency')
    def test_list_batches_success(self, mock_get_db, mock_get_service,
                                 client, mock_batch_service):
        """Test successful batch listing."""
        # Setup mocks
        mock_get_service.return_value = mock_batch_service
        mock_get_db.return_value = mock_batch_service._session
        
        # Mock batch service response
        mock_batches = [
            Mock(
                id=1, batch_id="batch_123", name="Test Batch 1",
                description="Description 1", status=BatchStatus.PENDING,
                priority=BatchPriority.NORMAL, total_items=2,
                completed_items=0, failed_items=0, pending_items=2,
                progress_percentage=0.0, created_at=datetime(2023, 1, 1),
                updated_at=datetime(2023, 1, 1), started_at=None,
                completed_at=None, is_completed=False, is_failed=False
            ),
            Mock(
                id=2, batch_id="batch_456", name="Test Batch 2",
                description="Description 2", status=BatchStatus.PROCESSING,
                priority=BatchPriority.HIGH, total_items=1,
                completed_items=0, failed_items=0, pending_items=1,
                progress_percentage=50.0, created_at=datetime(2023, 1, 2),
                updated_at=datetime(2023, 1, 2), started_at=datetime(2023, 1, 2),
                completed_at=None, is_completed=False, is_failed=False
            )
        ]
        
        mock_batch_service.list_batches.return_value = mock_batches
        
        # Make request
        response = client.get("/api/v1/batch/batches")
        
        # Verify response
        assert response.status_code == status.HTTP_200_OK
        response_data = response.json()
        assert len(response_data["batches"]) == 2
        assert response_data["total_count"] == 2
        assert response_data["page"] == 1
        assert response_data["page_size"] == 20
        assert response_data["has_next"] is False
        assert response_data["has_previous"] is False
        
        # Verify first batch
        first_batch = response_data["batches"][0]
        assert first_batch["batch_id"] == "batch_123"
        assert first_batch["name"] == "Test Batch 1"
        assert first_batch["status"] == "pending"

    @patch('src.api.batch.get_batch_service')
    @patch('src.api.batch.get_database_session_dependency')
    def test_list_batches_with_status_filter(self, mock_get_db, mock_get_service,
                                           client, mock_batch_service):
        """Test batch listing with status filter."""
        # Setup mocks
        mock_get_service.return_value = mock_batch_service
        mock_get_db.return_value = mock_batch_service._session
        
        # Mock filtered results
        mock_batches = [
            Mock(
                id=1, batch_id="batch_123", name="Test Batch 1",
                description="Description 1", status=BatchStatus.PROCESSING,
                priority=BatchPriority.NORMAL, total_items=2,
                completed_items=1, failed_items=0, pending_items=1,
                progress_percentage=50.0, created_at=datetime(2023, 1, 1),
                updated_at=datetime(2023, 1, 1), started_at=datetime(2023, 1, 1),
                completed_at=None, is_completed=False, is_failed=False
            )
        ]
        
        mock_batch_service.list_batches.return_value = mock_batches
        
        # Make request with status filter
        response = client.get("/api/v1/batch/batches?batch_status=processing")
        
        # Verify response
        assert response.status_code == status.HTTP_200_OK
        response_data = response.json()
        assert len(response_data["batches"]) == 1
        assert response_data["batches"][0]["status"] == "processing"
        
        # Verify service was called with correct filter
        mock_batch_service.list_batches.assert_called()
        call_args = mock_batch_service.list_batches.call_args
        assert call_args[1]["status"] == BatchStatus.PROCESSING

    @patch('src.api.batch.get_batch_service')
    @patch('src.api.batch.get_database_session_dependency')
    def test_list_batches_with_pagination(self, mock_get_db, mock_get_service,
                                         client, mock_batch_service):
        """Test batch listing with pagination."""
        # Setup mocks
        mock_get_service.return_value = mock_batch_service
        mock_get_db.return_value = mock_batch_service._session
        
        # Mock paginated results
        mock_batches = [
            Mock(
                id=3, batch_id="batch_789", name="Test Batch 3",
                description="Description 3", status=BatchStatus.COMPLETED,
                priority=BatchPriority.LOW, total_items=1,
                completed_items=1, failed_items=0, pending_items=0,
                progress_percentage=100.0, created_at=datetime(2023, 1, 3),
                updated_at=datetime(2023, 1, 3), started_at=datetime(2023, 1, 3),
                completed_at=datetime(2023, 1, 3), is_completed=True, is_failed=False
            )
        ]
        
        # Mock service calls for pagination
        mock_batch_service.list_batches.side_effect = [
            mock_batches,  # First call for page data
            [Mock()] * 25  # Second call for total count (simulating 25 total items)
        ]
        
        # Make request with pagination
        response = client.get("/api/v1/batch/batches?page=2&page_size=10")
        
        # Verify response
        assert response.status_code == status.HTTP_200_OK
        response_data = response.json()
        assert response_data["page"] == 2
        assert response_data["page_size"] == 10
        assert response_data["total_count"] == 25
        assert response_data["has_next"] is True
        assert response_data["has_previous"] is True

    # Batch Detail Tests

    @patch('src.api.batch.get_batch_service')
    @patch('src.api.batch.get_database_session_dependency')
    def test_get_batch_success(self, mock_get_db, mock_get_service,
                              client, mock_batch_service):
        """Test successful batch retrieval."""
        # Setup mocks
        mock_get_service.return_value = mock_batch_service
        mock_get_db.return_value = mock_batch_service._session
        
        # Mock batch with items
        mock_batch_items = [
            Mock(
                id=1, batch_id=1, video_id=None, url="https://www.youtube.com/watch?v=test123",
                status=BatchItemStatus.QUEUED, priority=BatchPriority.NORMAL,
                processing_order=1, created_at=datetime(2023, 1, 1),
                updated_at=datetime(2023, 1, 1), started_at=None,
                completed_at=None, retry_count=0, max_retries=3,
                error_info=None, processing_data=None, result_data=None,
                can_retry=True
            )
        ]
        
        mock_batch = Mock(
            id=1, batch_id="batch_123", name="Test Batch",
            description="Test batch description", status=BatchStatus.PENDING,
            priority=BatchPriority.NORMAL, total_items=1,
            completed_items=0, failed_items=0, pending_items=1,
            progress_percentage=0.0, created_at=datetime(2023, 1, 1),
            updated_at=datetime(2023, 1, 1), started_at=None,
            completed_at=None, webhook_url="https://webhook.example.com/batch",
            batch_metadata={"test_key": "test_value"}, error_info=None,
            is_completed=False, is_failed=False, batch_items=mock_batch_items
        )
        
        mock_batch_service.get_batch.return_value = mock_batch
        
        # Make request
        response = client.get("/api/v1/batch/batches/batch_123")
        
        # Verify response
        assert response.status_code == status.HTTP_200_OK
        response_data = response.json()
        assert response_data["batch_id"] == "batch_123"
        assert response_data["name"] == "Test Batch"
        assert response_data["status"] == "pending"
        assert len(response_data["batch_items"]) == 1
        assert response_data["batch_items"][0]["url"] == "https://www.youtube.com/watch?v=test123"

    @patch('src.api.batch.get_batch_service')
    @patch('src.api.batch.get_database_session_dependency')
    def test_get_batch_not_found(self, mock_get_db, mock_get_service,
                                client, mock_batch_service):
        """Test batch retrieval when batch not found."""
        # Setup mocks
        mock_get_service.return_value = mock_batch_service
        mock_get_db.return_value = mock_batch_service._session
        
        # Mock service to return None
        mock_batch_service.get_batch.return_value = None
        
        # Make request
        response = client.get("/api/v1/batch/batches/nonexistent_123")
        
        # Verify error response
        assert response.status_code == status.HTTP_404_NOT_FOUND
        response_data = response.json()
        assert "not found" in response_data["detail"]

    # Batch Progress Tests

    @patch('src.api.batch.get_batch_service')
    @patch('src.api.batch.get_database_session_dependency')
    def test_get_batch_progress_success(self, mock_get_db, mock_get_service,
                                       client, mock_batch_service):
        """Test successful batch progress retrieval."""
        # Setup mocks
        mock_get_service.return_value = mock_batch_service
        mock_get_db.return_value = mock_batch_service._session
        
        # Mock progress info
        mock_progress = Mock(
            batch_id="batch_123", status=BatchStatus.PROCESSING,
            total_items=10, completed_items=6, failed_items=1,
            pending_items=3, progress_percentage=60.0,
            started_at=datetime(2023, 1, 1),
            estimated_completion=datetime(2023, 1, 1, 2, 0)
        )
        
        mock_batch_service.get_batch_progress.return_value = mock_progress
        
        # Make request
        response = client.get("/api/v1/batch/batches/batch_123/progress")
        
        # Verify response
        assert response.status_code == status.HTTP_200_OK
        response_data = response.json()
        assert response_data["batch_id"] == "batch_123"
        assert response_data["status"] == "processing"
        assert response_data["total_items"] == 10
        assert response_data["completed_items"] == 6
        assert response_data["failed_items"] == 1
        assert response_data["pending_items"] == 3
        assert response_data["progress_percentage"] == 60.0

    @patch('src.api.batch.get_batch_service')
    @patch('src.api.batch.get_database_session_dependency')
    def test_get_batch_progress_not_found(self, mock_get_db, mock_get_service,
                                         client, mock_batch_service):
        """Test batch progress retrieval when batch not found."""
        # Setup mocks
        mock_get_service.return_value = mock_batch_service
        mock_get_db.return_value = mock_batch_service._session
        
        # Mock service to return None
        mock_batch_service.get_batch_progress.return_value = None
        
        # Make request
        response = client.get("/api/v1/batch/batches/nonexistent_123/progress")
        
        # Verify error response
        assert response.status_code == status.HTTP_404_NOT_FOUND
        response_data = response.json()
        assert "not found" in response_data["detail"]

    # Batch Control Tests

    @patch('src.api.batch.get_batch_service')
    @patch('src.api.batch.get_database_session_dependency')
    def test_start_batch_success(self, mock_get_db, mock_get_service,
                                client, mock_batch_service):
        """Test successful batch start."""
        # Setup mocks
        mock_get_service.return_value = mock_batch_service
        mock_get_db.return_value = mock_batch_service._session
        
        # Mock service to return success
        mock_batch_service.start_batch_processing.return_value = True
        
        # Make request
        response = client.post("/api/v1/batch/batches/batch_123/start")
        
        # Verify response
        assert response.status_code == status.HTTP_200_OK
        response_data = response.json()
        assert response_data["success"] is True
        assert response_data["batch_id"] == "batch_123"
        assert "started successfully" in response_data["message"]

    @patch('src.api.batch.get_batch_service')
    @patch('src.api.batch.get_database_session_dependency')
    def test_start_batch_failure(self, mock_get_db, mock_get_service,
                                client, mock_batch_service):
        """Test batch start failure."""
        # Setup mocks
        mock_get_service.return_value = mock_batch_service
        mock_get_db.return_value = mock_batch_service._session
        
        # Mock service to return failure
        mock_batch_service.start_batch_processing.return_value = False
        
        # Make request
        response = client.post("/api/v1/batch/batches/batch_123/start")
        
        # Verify error response
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        response_data = response.json()
        assert "Failed to start batch" in response_data["detail"]

    @patch('src.api.batch.get_batch_service')
    @patch('src.api.batch.get_database_session_dependency')
    def test_cancel_batch_success(self, mock_get_db, mock_get_service,
                                 client, mock_batch_service):
        """Test successful batch cancellation."""
        # Setup mocks
        mock_get_service.return_value = mock_batch_service
        mock_get_db.return_value = mock_batch_service._session
        
        # Mock service to return success
        mock_batch_service.cancel_batch.return_value = True
        
        # Make request
        cancel_data = {"reason": "User requested cancellation"}
        response = client.post("/api/v1/batch/batches/batch_123/cancel", json=cancel_data)
        
        # Verify response
        assert response.status_code == status.HTTP_200_OK
        response_data = response.json()
        assert response_data["success"] is True
        assert response_data["batch_id"] == "batch_123"
        assert response_data["reason"] == "User requested cancellation"

    @patch('src.api.batch.get_batch_service')
    @patch('src.api.batch.get_database_session_dependency')
    def test_cancel_batch_failure(self, mock_get_db, mock_get_service,
                                 client, mock_batch_service):
        """Test batch cancellation failure."""
        # Setup mocks
        mock_get_service.return_value = mock_batch_service
        mock_get_db.return_value = mock_batch_service._session
        
        # Mock service to return failure
        mock_batch_service.cancel_batch.return_value = False
        
        # Make request
        response = client.post("/api/v1/batch/batches/batch_123/cancel")
        
        # Verify error response
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        response_data = response.json()
        assert "Failed to cancel batch" in response_data["detail"]

    @patch('src.api.batch.get_batch_service')
    @patch('src.api.batch.get_database_session_dependency')
    def test_retry_batch_item_success(self, mock_get_db, mock_get_service,
                                     client, mock_batch_service):
        """Test successful batch item retry."""
        # Setup mocks
        mock_get_service.return_value = mock_batch_service
        mock_get_db.return_value = mock_batch_service._session
        
        # Mock service to return success
        mock_batch_service.retry_failed_batch_item.return_value = True
        
        # Make request
        response = client.post("/api/v1/batch/batches/batch_123/items/1/retry")
        
        # Verify response
        assert response.status_code == status.HTTP_200_OK
        response_data = response.json()
        assert response_data["success"] is True
        assert response_data["batch_id"] == "batch_123"
        assert response_data["item_id"] == 1

    @patch('src.api.batch.get_batch_service')
    @patch('src.api.batch.get_database_session_dependency')
    def test_retry_batch_item_failure(self, mock_get_db, mock_get_service,
                                     client, mock_batch_service):
        """Test batch item retry failure."""
        # Setup mocks
        mock_get_service.return_value = mock_batch_service
        mock_get_db.return_value = mock_batch_service._session
        
        # Mock service to return failure
        mock_batch_service.retry_failed_batch_item.return_value = False
        
        # Make request
        response = client.post("/api/v1/batch/batches/batch_123/items/1/retry")
        
        # Verify error response
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        response_data = response.json()
        assert "Failed to retry batch item" in response_data["detail"]

    # Queue Management Tests

    @patch('src.api.batch.get_batch_service')
    @patch('src.api.batch.get_database_session_dependency')
    def test_get_next_queue_item_success(self, mock_get_db, mock_get_service,
                                        client, mock_batch_service):
        """Test successful queue item retrieval."""
        # Setup mocks
        mock_get_service.return_value = mock_batch_service
        mock_get_db.return_value = mock_batch_service._session
        
        # Mock queue item with batch item
        mock_batch_item = Mock(
            id=1, batch_id=1, url="https://www.youtube.com/watch?v=test123",
            status=BatchItemStatus.PROCESSING, retry_count=0, max_retries=3
        )
        
        mock_queue_item = Mock(
            id=1, batch_item_id=1, queue_name="video_processing",
            priority=BatchPriority.NORMAL, locked_by="worker_123",
            lock_expires_at=datetime(2023, 1, 1, 1, 0),
            batch_item=mock_batch_item
        )
        
        mock_batch_service.get_next_queue_item.return_value = mock_queue_item
        
        # Make request
        response = client.get("/api/v1/batch/queue/next?worker_id=worker_123")
        
        # Verify response
        assert response.status_code == status.HTTP_200_OK
        response_data = response.json()
        assert response_data["available"] is True
        assert response_data["queue_item"]["id"] == 1
        assert response_data["queue_item"]["batch_item_id"] == 1
        assert response_data["queue_item"]["locked_by"] == "worker_123"
        assert response_data["queue_item"]["batch_item"]["url"] == "https://www.youtube.com/watch?v=test123"

    @patch('src.api.batch.get_batch_service')
    @patch('src.api.batch.get_database_session_dependency')
    def test_get_next_queue_item_none_available(self, mock_get_db, mock_get_service,
                                               client, mock_batch_service):
        """Test queue item retrieval when none available."""
        # Setup mocks
        mock_get_service.return_value = mock_batch_service
        mock_get_db.return_value = mock_batch_service._session
        
        # Mock service to return None
        mock_batch_service.get_next_queue_item.return_value = None
        
        # Make request
        response = client.get("/api/v1/batch/queue/next?worker_id=worker_123")
        
        # Verify response
        assert response.status_code == status.HTTP_200_OK
        response_data = response.json()
        assert response_data["available"] is False
        assert "No items available" in response_data["message"]

    @patch('src.api.batch.get_batch_service')
    @patch('src.api.batch.get_database_session_dependency')
    def test_complete_batch_item_success(self, mock_get_db, mock_get_service,
                                        client, mock_batch_service):
        """Test successful batch item completion."""
        # Setup mocks
        mock_get_service.return_value = mock_batch_service
        mock_get_db.return_value = mock_batch_service._session
        
        # Mock service to return success
        mock_batch_service.complete_batch_item.return_value = True
        
        # Make request
        completion_data = {
            "status": "completed",
            "video_id": 123,
            "processing_time_seconds": 45.5,
            "result_data": {"summary": "Test summary"}
        }
        response = client.post("/api/v1/batch/queue/complete/1", json=completion_data)
        
        # Verify response
        assert response.status_code == status.HTTP_200_OK
        response_data = response.json()
        assert response_data["success"] is True
        assert response_data["batch_item_id"] == 1
        assert response_data["status"] == "completed"

    @patch('src.api.batch.get_batch_service')
    @patch('src.api.batch.get_database_session_dependency')
    def test_complete_batch_item_failure(self, mock_get_db, mock_get_service,
                                        client, mock_batch_service):
        """Test batch item completion failure."""
        # Setup mocks
        mock_get_service.return_value = mock_batch_service
        mock_get_db.return_value = mock_batch_service._session
        
        # Mock service to return failure
        mock_batch_service.complete_batch_item.return_value = False
        
        # Make request
        completion_data = {
            "status": "failed",
            "error_message": "Processing failed"
        }
        response = client.post("/api/v1/batch/queue/complete/1", json=completion_data)
        
        # Verify error response
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        response_data = response.json()
        assert "Failed to complete batch item" in response_data["detail"]

    # Statistics Tests

    @patch('src.api.batch.get_batch_service')
    @patch('src.api.batch.get_database_session_dependency')
    def test_get_batch_statistics_success(self, mock_get_db, mock_get_service,
                                         client, mock_batch_service):
        """Test successful batch statistics retrieval."""
        # Setup mocks
        mock_get_service.return_value = mock_batch_service
        mock_get_db.return_value = mock_batch_service._session
        
        # Mock statistics
        mock_stats = {
            "total_batches": 10,
            "batch_status_counts": {
                "pending": 2,
                "processing": 3,
                "completed": 4,
                "failed": 1
            },
            "total_batch_items": 50,
            "item_status_counts": {
                "queued": 10,
                "processing": 5,
                "completed": 30,
                "failed": 5
            },
            "active_processing_sessions": 5
        }
        
        mock_batch_service.get_batch_statistics.return_value = mock_stats
        
        # Make request
        response = client.get("/api/v1/batch/statistics")
        
        # Verify response
        assert response.status_code == status.HTTP_200_OK
        response_data = response.json()
        assert response_data["total_batches"] == 10
        assert response_data["batch_status_counts"]["pending"] == 2
        assert response_data["total_batch_items"] == 50
        assert response_data["active_processing_sessions"] == 5

    # Cleanup Tests

    @patch('src.api.batch.get_batch_service')
    @patch('src.api.batch.get_database_session_dependency')
    def test_cleanup_stale_sessions_success(self, mock_get_db, mock_get_service,
                                           client, mock_batch_service):
        """Test successful stale session cleanup."""
        # Setup mocks
        mock_get_service.return_value = mock_batch_service
        mock_get_db.return_value = mock_batch_service._session
        
        # Mock service to return cleanup count
        mock_batch_service.cleanup_stale_sessions.return_value = 3
        
        # Make request
        response = client.post("/api/v1/batch/cleanup/stale-sessions?timeout_minutes=30")
        
        # Verify response
        assert response.status_code == status.HTTP_200_OK
        response_data = response.json()
        assert response_data["success"] is True
        assert response_data["cleaned_count"] == 3
        assert response_data["timeout_minutes"] == 30

    # Health Check Tests

    def test_batch_health_check(self, client):
        """Test batch processing health check endpoint."""
        # Make request
        response = client.get("/api/v1/batch/health")
        
        # Verify response
        assert response.status_code == status.HTTP_200_OK
        response_data = response.json()
        assert response_data["status"] == "healthy"
        assert response_data["service"] == "batch_processing"
        assert "timestamp" in response_data
        assert response_data["components"]["batch_service"] == "ready"

    # Processing Session Tests

    @patch('src.api.batch.get_batch_service')
    @patch('src.api.batch.get_database_session_dependency')
    def test_create_processing_session_success(self, mock_get_db, mock_get_service,
                                              client, mock_batch_service):
        """Test successful processing session creation."""
        # Setup mocks
        mock_get_service.return_value = mock_batch_service
        mock_get_db.return_value = mock_batch_service._session
        
        # Mock session
        mock_session = Mock(
            session_id="session_123",
            started_at=datetime(2023, 1, 1)
        )
        
        mock_batch_service.create_processing_session.return_value = mock_session
        
        # Make request
        response = client.post("/api/v1/batch/sessions/1?worker_id=worker_123")
        
        # Verify response
        assert response.status_code == status.HTTP_200_OK
        response_data = response.json()
        assert response_data["success"] is True
        assert response_data["session_id"] == "session_123"
        assert response_data["batch_item_id"] == 1
        assert response_data["worker_id"] == "worker_123"

    @patch('src.api.batch.get_batch_service')
    @patch('src.api.batch.get_database_session_dependency')
    def test_update_session_progress_success(self, mock_get_db, mock_get_service,
                                            client, mock_batch_service):
        """Test successful session progress update."""
        # Setup mocks
        mock_get_service.return_value = mock_batch_service
        mock_get_db.return_value = mock_batch_service._session
        
        # Mock service to return success
        mock_batch_service.update_processing_session.return_value = True
        
        # Make request
        response = client.put("/api/v1/batch/sessions/session_123/progress?progress=75.5&current_step=summarizing")
        
        # Verify response
        assert response.status_code == status.HTTP_200_OK
        response_data = response.json()
        assert response_data["success"] is True
        assert response_data["session_id"] == "session_123"
        assert response_data["progress"] == 75.5
        assert response_data["current_step"] == "summarizing"

    @patch('src.api.batch.get_batch_service')
    @patch('src.api.batch.get_database_session_dependency')
    def test_update_session_progress_not_found(self, mock_get_db, mock_get_service,
                                              client, mock_batch_service):
        """Test session progress update when session not found."""
        # Setup mocks
        mock_get_service.return_value = mock_batch_service
        mock_get_db.return_value = mock_batch_service._session
        
        # Mock service to return failure
        mock_batch_service.update_processing_session.return_value = False
        
        # Make request
        response = client.put("/api/v1/batch/sessions/nonexistent_session/progress?progress=50.0")
        
        # Verify error response
        assert response.status_code == status.HTTP_404_NOT_FOUND
        response_data = response.json()
        assert "not found" in response_data["detail"]

    # Error Handling Tests

    @patch('src.api.batch.get_batch_service')
    @patch('src.api.batch.get_database_session_dependency')
    def test_unexpected_error_handling(self, mock_get_db, mock_get_service,
                                      client, mock_batch_service):
        """Test handling of unexpected errors."""
        # Setup mocks
        mock_get_service.return_value = mock_batch_service
        mock_get_db.return_value = mock_batch_service._session
        
        # Mock service to raise unexpected error
        mock_batch_service.get_batch.side_effect = Exception("Unexpected error")
        
        # Make request
        response = client.get("/api/v1/batch/batches/batch_123")
        
        # Verify error response
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        response_data = response.json()
        assert "Internal server error" in response_data["detail"]

    # Validation Tests

    def test_invalid_batch_id_format(self, client):
        """Test handling of invalid batch ID format."""
        # Make request with invalid batch ID
        response = client.get("/api/v1/batch/batches/")
        
        # Verify error response
        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_invalid_page_parameters(self, client):
        """Test handling of invalid pagination parameters."""
        # Make request with invalid page parameters
        response = client.get("/api/v1/batch/batches?page=0&page_size=0")
        
        # Verify error response
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_invalid_progress_percentage(self, client):
        """Test handling of invalid progress percentage."""
        # Make request with invalid progress
        response = client.put("/api/v1/batch/sessions/session_123/progress?progress=150.0")
        
        # Verify error response
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_missing_required_parameters(self, client):
        """Test handling of missing required parameters."""
        # Make request without required worker_id
        response = client.get("/api/v1/batch/queue/next")
        
        # Verify error response
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    # Integration Tests

    @patch('src.api.batch.get_batch_service')
    @patch('src.api.batch.get_database_session_dependency')
    def test_complete_batch_workflow(self, mock_get_db, mock_get_service,
                                    client, mock_batch_service, sample_batch_data):
        """Test complete batch processing workflow through API."""
        # Setup mocks
        mock_get_service.return_value = mock_batch_service
        mock_get_db.return_value = mock_batch_service._session
        
        # Mock batch creation
        mock_batch = Mock(
            id=1, batch_id="batch_123", name="Test Batch",
            description="Test batch description", status=BatchStatus.PENDING,
            priority=BatchPriority.NORMAL, total_items=2,
            completed_items=0, failed_items=0, pending_items=2,
            progress_percentage=0.0, created_at=datetime(2023, 1, 1),
            updated_at=datetime(2023, 1, 1), started_at=None,
            completed_at=None, webhook_url="https://webhook.example.com/batch",
            batch_metadata={"test_key": "test_value"}, error_info=None,
            is_completed=False, is_failed=False, batch_items=[]
        )
        
        mock_batch_service.create_batch.return_value = mock_batch
        mock_batch_service.start_batch_processing.return_value = True
        mock_batch_service.get_batch.return_value = mock_batch
        
        # Mock URL validation
        with patch('src.api.batch.validate_youtube_url_detailed') as mock_validate:
            mock_validate.return_value = Mock(is_valid=True, error_message=None)
            
            # Step 1: Create batch
            response = client.post("/api/v1/batch/batches", json=sample_batch_data)
            assert response.status_code == status.HTTP_201_CREATED
            batch_id = response.json()["batch_id"]
            
            # Step 2: Start batch
            response = client.post(f"/api/v1/batch/batches/{batch_id}/start")
            assert response.status_code == status.HTTP_200_OK
            
            # Step 3: Get batch details
            response = client.get(f"/api/v1/batch/batches/{batch_id}")
            assert response.status_code == status.HTTP_200_OK
            assert response.json()["batch_id"] == batch_id


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])