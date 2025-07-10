"""
Test suite for batch processing API endpoints.

This module provides comprehensive tests for the batch processing REST API,
including request validation, response formatting, and error handling.
"""

import pytest
import json
import asyncio
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient
from fastapi import HTTPException
from sqlalchemy.orm import Session

from ..app import app
from ..database.batch_models import BatchStatus, BatchItemStatus, BatchPriority
from ..services.batch_service import BatchService, BatchServiceError, BatchCreateRequest, BatchProgressInfo
from .batch import (
    BatchStatusEnum, BatchItemStatusEnum, BatchPriorityEnum,
    CreateBatchRequest, StartBatchRequest, CancelBatchRequest
)


class TestBatchAPI:
    """Test suite for batch processing API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    @pytest.fixture
    def mock_batch_service(self):
        """Create mock batch service."""
        return Mock(spec=BatchService)
    
    @pytest.fixture
    def mock_db_session(self):
        """Create mock database session."""
        return Mock(spec=Session)
    
    @pytest.fixture
    def sample_batch_data(self):
        """Sample batch data for testing."""
        return {
            "id": 1,
            "batch_id": "batch_20240101_120000_abcd1234",
            "name": "Test Batch",
            "description": "Test batch description",
            "status": BatchStatus.PENDING,
            "priority": BatchPriority.NORMAL,
            "total_items": 3,
            "completed_items": 0,
            "failed_items": 0,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "started_at": None,
            "completed_at": None,
            "webhook_url": None,
            "batch_metadata": {},
            "error_info": None,
            "batch_items": []
        }
    
    @pytest.fixture
    def sample_batch_item_data(self):
        """Sample batch item data for testing."""
        return {
            "id": 1,
            "batch_id": 1,
            "video_id": None,
            "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "status": BatchItemStatus.QUEUED,
            "priority": BatchPriority.NORMAL,
            "processing_order": 0,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "started_at": None,
            "completed_at": None,
            "retry_count": 0,
            "max_retries": 3,
            "error_info": None,
            "processing_data": None,
            "result_data": None
        }
    
    def test_create_batch_success(self, client, mock_batch_service, mock_db_session):
        """Test successful batch creation."""
        # Mock batch object
        mock_batch = Mock()
        mock_batch.id = 1
        mock_batch.batch_id = "batch_20240101_120000_abcd1234"
        mock_batch.name = "Test Batch"
        mock_batch.description = "Test batch description"
        mock_batch.status = BatchStatus.PENDING
        mock_batch.priority = BatchPriority.NORMAL
        mock_batch.total_items = 2
        mock_batch.completed_items = 0
        mock_batch.failed_items = 0
        mock_batch.pending_items = 2
        mock_batch.progress_percentage = 0.0
        mock_batch.created_at = datetime.utcnow()
        mock_batch.updated_at = datetime.utcnow()
        mock_batch.started_at = None
        mock_batch.completed_at = None
        mock_batch.webhook_url = None
        mock_batch.batch_metadata = {}
        mock_batch.error_info = None
        mock_batch.is_completed = False
        mock_batch.is_failed = False
        mock_batch.batch_items = []
        
        # Mock service method
        mock_batch_service.create_batch.return_value = mock_batch
        
        # Test data
        request_data = {
            "name": "Test Batch",
            "description": "Test batch description",
            "urls": [
                "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                "https://www.youtube.com/watch?v=abcd1234567"
            ],
            "priority": "normal"
        }
        
        # Mock dependencies
        with patch('src.api.batch.get_batch_service', return_value=mock_batch_service):
            with patch('src.api.batch.get_database_session_dependency', return_value=mock_db_session):
                response = client.post("/api/v1/batch/batches", json=request_data)
        
        assert response.status_code == 201
        response_data = response.json()
        assert response_data["batch_id"] == "batch_20240101_120000_abcd1234"
        assert response_data["name"] == "Test Batch"
        assert response_data["status"] == "pending"
        assert response_data["total_items"] == 2
        assert response_data["progress_percentage"] == 0.0
    
    def test_create_batch_invalid_urls(self, client):
        """Test batch creation with invalid URLs."""
        request_data = {
            "name": "Test Batch",
            "urls": [
                "https://invalid-url.com",
                "not-a-url"
            ],
            "priority": "normal"
        }
        
        response = client.post("/api/v1/batch/batches", json=request_data)
        assert response.status_code == 422
        response_data = response.json()
        assert "Invalid YouTube URL" in str(response_data["detail"])
    
    def test_create_batch_empty_urls(self, client):
        """Test batch creation with empty URLs list."""
        request_data = {
            "name": "Test Batch",
            "urls": [],
            "priority": "normal"
        }
        
        response = client.post("/api/v1/batch/batches", json=request_data)
        assert response.status_code == 422
        response_data = response.json()
        assert "at least 1 item" in str(response_data["detail"])
    
    def test_create_batch_duplicate_urls(self, client):
        """Test batch creation with duplicate URLs."""
        request_data = {
            "name": "Test Batch",
            "urls": [
                "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
            ],
            "priority": "normal"
        }
        
        response = client.post("/api/v1/batch/batches", json=request_data)
        assert response.status_code == 422
        response_data = response.json()
        assert "Duplicate URLs" in str(response_data["detail"])
    
    def test_create_batch_service_error(self, client, mock_batch_service, mock_db_session):
        """Test batch creation with service error."""
        # Mock service to raise error
        mock_batch_service.create_batch.side_effect = BatchServiceError("Service error")
        
        request_data = {
            "name": "Test Batch",
            "urls": ["https://www.youtube.com/watch?v=dQw4w9WgXcQ"],
            "priority": "normal"
        }
        
        with patch('src.api.batch.get_batch_service', return_value=mock_batch_service):
            with patch('src.api.batch.get_database_session_dependency', return_value=mock_db_session):
                response = client.post("/api/v1/batch/batches", json=request_data)
        
        assert response.status_code == 400
        response_data = response.json()
        assert "Failed to create batch" in response_data["detail"]
    
    def test_list_batches_success(self, client, mock_batch_service, mock_db_session):
        """Test successful batch listing."""
        # Mock batch objects
        mock_batch1 = Mock()
        mock_batch1.id = 1
        mock_batch1.batch_id = "batch_1"
        mock_batch1.name = "Batch 1"
        mock_batch1.description = "First batch"
        mock_batch1.status = BatchStatus.PENDING
        mock_batch1.priority = BatchPriority.NORMAL
        mock_batch1.total_items = 2
        mock_batch1.completed_items = 0
        mock_batch1.failed_items = 0
        mock_batch1.pending_items = 2
        mock_batch1.progress_percentage = 0.0
        mock_batch1.created_at = datetime.utcnow()
        mock_batch1.updated_at = datetime.utcnow()
        mock_batch1.started_at = None
        mock_batch1.completed_at = None
        mock_batch1.is_completed = False
        mock_batch1.is_failed = False
        
        mock_batch2 = Mock()
        mock_batch2.id = 2
        mock_batch2.batch_id = "batch_2"
        mock_batch2.name = "Batch 2"
        mock_batch2.description = "Second batch"
        mock_batch2.status = BatchStatus.PROCESSING
        mock_batch2.priority = BatchPriority.HIGH
        mock_batch2.total_items = 3
        mock_batch2.completed_items = 1
        mock_batch2.failed_items = 0
        mock_batch2.pending_items = 2
        mock_batch2.progress_percentage = 33.3
        mock_batch2.created_at = datetime.utcnow()
        mock_batch2.updated_at = datetime.utcnow()
        mock_batch2.started_at = datetime.utcnow()
        mock_batch2.completed_at = None
        mock_batch2.is_completed = False
        mock_batch2.is_failed = False
        
        # Mock service method
        mock_batch_service.list_batches.return_value = [mock_batch1, mock_batch2]
        
        with patch('src.api.batch.get_batch_service', return_value=mock_batch_service):
            with patch('src.api.batch.get_database_session_dependency', return_value=mock_db_session):
                response = client.get("/api/v1/batch/batches")
        
        assert response.status_code == 200
        response_data = response.json()
        assert len(response_data["batches"]) == 2
        assert response_data["batches"][0]["batch_id"] == "batch_1"
        assert response_data["batches"][1]["batch_id"] == "batch_2"
        assert response_data["total_count"] == 2
        assert response_data["page"] == 1
        assert response_data["page_size"] == 20
    
    def test_list_batches_with_status_filter(self, client, mock_batch_service, mock_db_session):
        """Test batch listing with status filter."""
        mock_batch = Mock()
        mock_batch.id = 1
        mock_batch.batch_id = "batch_1"
        mock_batch.name = "Batch 1"
        mock_batch.description = "First batch"
        mock_batch.status = BatchStatus.PROCESSING
        mock_batch.priority = BatchPriority.NORMAL
        mock_batch.total_items = 2
        mock_batch.completed_items = 1
        mock_batch.failed_items = 0
        mock_batch.pending_items = 1
        mock_batch.progress_percentage = 50.0
        mock_batch.created_at = datetime.utcnow()
        mock_batch.updated_at = datetime.utcnow()
        mock_batch.started_at = datetime.utcnow()
        mock_batch.completed_at = None
        mock_batch.is_completed = False
        mock_batch.is_failed = False
        
        mock_batch_service.list_batches.return_value = [mock_batch]
        
        with patch('src.api.batch.get_batch_service', return_value=mock_batch_service):
            with patch('src.api.batch.get_database_session_dependency', return_value=mock_db_session):
                response = client.get("/api/v1/batch/batches?batch_status=processing")
        
        assert response.status_code == 200
        response_data = response.json()
        assert len(response_data["batches"]) == 1
        assert response_data["batches"][0]["status"] == "processing"
        
        # Verify service was called with correct status filter
        mock_batch_service.list_batches.assert_called_with(
            status=BatchStatus.PROCESSING,
            limit=20,
            offset=0
        )
    
    def test_get_batch_success(self, client, mock_batch_service, mock_db_session):
        """Test successful batch retrieval."""
        mock_batch = Mock()
        mock_batch.id = 1
        mock_batch.batch_id = "batch_20240101_120000_abcd1234"
        mock_batch.name = "Test Batch"
        mock_batch.description = "Test batch description"
        mock_batch.status = BatchStatus.PENDING
        mock_batch.priority = BatchPriority.NORMAL
        mock_batch.total_items = 2
        mock_batch.completed_items = 0
        mock_batch.failed_items = 0
        mock_batch.pending_items = 2
        mock_batch.progress_percentage = 0.0
        mock_batch.created_at = datetime.utcnow()
        mock_batch.updated_at = datetime.utcnow()
        mock_batch.started_at = None
        mock_batch.completed_at = None
        mock_batch.webhook_url = None
        mock_batch.batch_metadata = {}
        mock_batch.error_info = None
        mock_batch.is_completed = False
        mock_batch.is_failed = False
        mock_batch.batch_items = []
        
        mock_batch_service.get_batch.return_value = mock_batch
        
        with patch('src.api.batch.get_batch_service', return_value=mock_batch_service):
            with patch('src.api.batch.get_database_session_dependency', return_value=mock_db_session):
                response = client.get("/api/v1/batch/batches/batch_20240101_120000_abcd1234")
        
        assert response.status_code == 200
        response_data = response.json()
        assert response_data["batch_id"] == "batch_20240101_120000_abcd1234"
        assert response_data["name"] == "Test Batch"
        assert response_data["status"] == "pending"
    
    def test_get_batch_not_found(self, client, mock_batch_service, mock_db_session):
        """Test batch retrieval with non-existent batch."""
        mock_batch_service.get_batch.return_value = None
        
        with patch('src.api.batch.get_batch_service', return_value=mock_batch_service):
            with patch('src.api.batch.get_database_session_dependency', return_value=mock_db_session):
                response = client.get("/api/v1/batch/batches/nonexistent_batch")
        
        assert response.status_code == 404
        response_data = response.json()
        assert "not found" in response_data["detail"]
    
    def test_get_batch_progress_success(self, client, mock_batch_service, mock_db_session):
        """Test successful batch progress retrieval."""
        mock_progress = BatchProgressInfo(
            batch_id="batch_20240101_120000_abcd1234",
            status=BatchStatus.PROCESSING,
            total_items=10,
            completed_items=3,
            failed_items=1,
            pending_items=6,
            progress_percentage=30.0,
            started_at=datetime.utcnow(),
            estimated_completion=datetime.utcnow()
        )
        
        mock_batch_service.get_batch_progress.return_value = mock_progress
        
        with patch('src.api.batch.get_batch_service', return_value=mock_batch_service):
            with patch('src.api.batch.get_database_session_dependency', return_value=mock_db_session):
                response = client.get("/api/v1/batch/batches/batch_20240101_120000_abcd1234/progress")
        
        assert response.status_code == 200
        response_data = response.json()
        assert response_data["batch_id"] == "batch_20240101_120000_abcd1234"
        assert response_data["status"] == "processing"
        assert response_data["total_items"] == 10
        assert response_data["completed_items"] == 3
        assert response_data["failed_items"] == 1
        assert response_data["pending_items"] == 6
        assert response_data["progress_percentage"] == 30.0
    
    def test_start_batch_success(self, client, mock_batch_service, mock_db_session):
        """Test successful batch start."""
        mock_batch_service.start_batch_processing.return_value = True
        
        with patch('src.api.batch.get_batch_service', return_value=mock_batch_service):
            with patch('src.api.batch.get_database_session_dependency', return_value=mock_db_session):
                response = client.post("/api/v1/batch/batches/batch_20240101_120000_abcd1234/start")
        
        assert response.status_code == 200
        response_data = response.json()
        assert response_data["success"] == True
        assert response_data["batch_id"] == "batch_20240101_120000_abcd1234"
        assert "started successfully" in response_data["message"]
    
    def test_start_batch_failure(self, client, mock_batch_service, mock_db_session):
        """Test batch start failure."""
        mock_batch_service.start_batch_processing.return_value = False
        
        with patch('src.api.batch.get_batch_service', return_value=mock_batch_service):
            with patch('src.api.batch.get_database_session_dependency', return_value=mock_db_session):
                response = client.post("/api/v1/batch/batches/batch_20240101_120000_abcd1234/start")
        
        assert response.status_code == 400
        response_data = response.json()
        assert "Failed to start batch" in response_data["detail"]
    
    def test_cancel_batch_success(self, client, mock_batch_service, mock_db_session):
        """Test successful batch cancellation."""
        mock_batch_service.cancel_batch.return_value = True
        
        request_data = {
            "reason": "User requested cancellation"
        }
        
        with patch('src.api.batch.get_batch_service', return_value=mock_batch_service):
            with patch('src.api.batch.get_database_session_dependency', return_value=mock_db_session):
                response = client.post("/api/v1/batch/batches/batch_20240101_120000_abcd1234/cancel", json=request_data)
        
        assert response.status_code == 200
        response_data = response.json()
        assert response_data["success"] == True
        assert response_data["batch_id"] == "batch_20240101_120000_abcd1234"
        assert response_data["reason"] == "User requested cancellation"
    
    def test_retry_batch_item_success(self, client, mock_batch_service, mock_db_session):
        """Test successful batch item retry."""
        mock_batch_service.retry_failed_batch_item.return_value = True
        
        with patch('src.api.batch.get_batch_service', return_value=mock_batch_service):
            with patch('src.api.batch.get_database_session_dependency', return_value=mock_db_session):
                response = client.post("/api/v1/batch/batches/batch_20240101_120000_abcd1234/items/1/retry")
        
        assert response.status_code == 200
        response_data = response.json()
        assert response_data["success"] == True
        assert response_data["batch_id"] == "batch_20240101_120000_abcd1234"
        assert response_data["item_id"] == 1
    
    def test_get_next_queue_item_success(self, client, mock_batch_service, mock_db_session):
        """Test successful queue item retrieval."""
        mock_queue_item = Mock()
        mock_queue_item.id = 1
        mock_queue_item.batch_item_id = 1
        mock_queue_item.queue_name = "video_processing"
        mock_queue_item.priority = BatchPriority.NORMAL
        mock_queue_item.locked_by = "worker_1"
        mock_queue_item.lock_expires_at = datetime.utcnow()
        
        mock_batch_item = Mock()
        mock_batch_item.id = 1
        mock_batch_item.batch_id = 1
        mock_batch_item.url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        mock_batch_item.status = BatchItemStatus.PROCESSING
        mock_batch_item.retry_count = 0
        mock_batch_item.max_retries = 3
        
        mock_queue_item.batch_item = mock_batch_item
        
        mock_batch_service.get_next_queue_item.return_value = mock_queue_item
        
        with patch('src.api.batch.get_batch_service', return_value=mock_batch_service):
            with patch('src.api.batch.get_database_session_dependency', return_value=mock_db_session):
                response = client.get("/api/v1/batch/queue/next?worker_id=worker_1")
        
        assert response.status_code == 200
        response_data = response.json()
        assert response_data["available"] == True
        assert response_data["queue_item"]["id"] == 1
        assert response_data["queue_item"]["batch_item"]["url"] == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    
    def test_get_next_queue_item_none_available(self, client, mock_batch_service, mock_db_session):
        """Test queue item retrieval when none available."""
        mock_batch_service.get_next_queue_item.return_value = None
        
        with patch('src.api.batch.get_batch_service', return_value=mock_batch_service):
            with patch('src.api.batch.get_database_session_dependency', return_value=mock_db_session):
                response = client.get("/api/v1/batch/queue/next?worker_id=worker_1")
        
        assert response.status_code == 200
        response_data = response.json()
        assert response_data["available"] == False
        assert "No items available" in response_data["message"]
    
    def test_get_batch_statistics_success(self, client, mock_batch_service, mock_db_session):
        """Test successful batch statistics retrieval."""
        mock_stats = {
            "total_batches": 10,
            "batch_status_counts": {
                "pending": 3,
                "processing": 2,
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
            "active_processing_sessions": 3
        }
        
        mock_batch_service.get_batch_statistics.return_value = mock_stats
        
        with patch('src.api.batch.get_batch_service', return_value=mock_batch_service):
            with patch('src.api.batch.get_database_session_dependency', return_value=mock_db_session):
                response = client.get("/api/v1/batch/statistics")
        
        assert response.status_code == 200
        response_data = response.json()
        assert response_data["total_batches"] == 10
        assert response_data["batch_status_counts"]["pending"] == 3
        assert response_data["total_batch_items"] == 50
        assert response_data["active_processing_sessions"] == 3
    
    def test_cleanup_stale_sessions_success(self, client, mock_batch_service, mock_db_session):
        """Test successful stale session cleanup."""
        mock_batch_service.cleanup_stale_sessions.return_value = 5
        
        with patch('src.api.batch.get_batch_service', return_value=mock_batch_service):
            with patch('src.api.batch.get_database_session_dependency', return_value=mock_db_session):
                response = client.post("/api/v1/batch/cleanup/stale-sessions?timeout_minutes=30")
        
        assert response.status_code == 200
        response_data = response.json()
        assert response_data["success"] == True
        assert response_data["cleaned_count"] == 5
        assert response_data["timeout_minutes"] == 30
    
    def test_batch_health_check(self, client):
        """Test batch health check endpoint."""
        response = client.get("/api/v1/batch/health")
        
        assert response.status_code == 200
        response_data = response.json()
        assert response_data["status"] == "healthy"
        assert response_data["service"] == "batch_processing"
        assert "components" in response_data
    
    def test_create_processing_session_success(self, client, mock_batch_service, mock_db_session):
        """Test successful processing session creation."""
        mock_session = Mock()
        mock_session.session_id = "session_20240101_120000_abcd1234"
        mock_session.started_at = datetime.utcnow()
        
        mock_batch_service.create_processing_session.return_value = mock_session
        
        with patch('src.api.batch.get_batch_service', return_value=mock_batch_service):
            with patch('src.api.batch.get_database_session_dependency', return_value=mock_db_session):
                response = client.post("/api/v1/batch/sessions/1?worker_id=worker_1")
        
        assert response.status_code == 200
        response_data = response.json()
        assert response_data["success"] == True
        assert response_data["session_id"] == "session_20240101_120000_abcd1234"
        assert response_data["batch_item_id"] == 1
        assert response_data["worker_id"] == "worker_1"
    
    def test_update_session_progress_success(self, client, mock_batch_service, mock_db_session):
        """Test successful session progress update."""
        mock_batch_service.update_processing_session.return_value = True
        
        with patch('src.api.batch.get_batch_service', return_value=mock_batch_service):
            with patch('src.api.batch.get_database_session_dependency', return_value=mock_db_session):
                response = client.put("/api/v1/batch/sessions/session_20240101_120000_abcd1234/progress?progress=50.0&current_step=Processing")
        
        assert response.status_code == 200
        response_data = response.json()
        assert response_data["success"] == True
        assert response_data["session_id"] == "session_20240101_120000_abcd1234"
        assert response_data["progress"] == 50.0
        assert response_data["current_step"] == "Processing"
    
    def test_update_session_progress_not_found(self, client, mock_batch_service, mock_db_session):
        """Test session progress update with non-existent session."""
        mock_batch_service.update_processing_session.return_value = False
        
        with patch('src.api.batch.get_batch_service', return_value=mock_batch_service):
            with patch('src.api.batch.get_database_session_dependency', return_value=mock_db_session):
                response = client.put("/api/v1/batch/sessions/nonexistent_session/progress?progress=50.0")
        
        assert response.status_code == 404
        response_data = response.json()
        assert "not found" in response_data["detail"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])