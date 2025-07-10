"""
Tests for the status API endpoints.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient
from fastapi import FastAPI

from ..database.status_models import ProcessingStatus, ProcessingStatusType, ProcessingPriority
from ..services.status_service import StatusService
from ..services.status_metrics_service import StatusMetricsService
from .status import router


# Test fixtures
@pytest.fixture
def app():
    """Create FastAPI app for testing."""
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_status_service():
    """Mock StatusService."""
    return Mock(spec=StatusService)


@pytest.fixture
def mock_metrics_service():
    """Mock StatusMetricsService."""
    return Mock(spec=StatusMetricsService)


@pytest.fixture
def sample_processing_status():
    """Sample processing status for testing."""
    return ProcessingStatus(
        id=1,
        status_id="status_test123",
        video_id=123,
        batch_item_id=456,
        processing_session_id=789,
        status=ProcessingStatusType.YOUTUBE_METADATA,
        priority=ProcessingPriority.HIGH,
        progress_percentage=25.0,
        current_step="Processing metadata",
        total_steps=5,
        completed_steps=1,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
        started_at=datetime.utcnow(),
        worker_id="worker-1",
        heartbeat_at=datetime.utcnow(),
        retry_count=0,
        max_retries=3,
        processing_metadata={"step": "metadata"},
        tags=["test", "video"],
        external_id="ext-123"
    )


class TestStatusEndpoints:
    """Test cases for status API endpoints."""
    
    def test_get_status_success(self, client, mock_status_service, sample_processing_status):
        """Test successful status retrieval."""
        mock_status_service.get_processing_status.return_value = sample_processing_status
        
        with patch('src.api.status.get_status_service', return_value=mock_status_service):
            response = client.get("/api/status/status_test123")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status_id"] == "status_test123"
            assert data["video_id"] == 123
            assert data["status"] == "YOUTUBE_METADATA"
            assert data["progress_percentage"] == 25.0
            assert data["current_step"] == "Processing metadata"
    
    def test_get_status_not_found(self, client, mock_status_service):
        """Test status retrieval when status doesn't exist."""
        mock_status_service.get_processing_status.return_value = None
        
        with patch('src.api.status.get_status_service', return_value=mock_status_service):
            response = client.get("/api/status/nonexistent")
            
            assert response.status_code == 404
            assert "not found" in response.json()["detail"]
    
    def test_get_status_by_video_success(self, client, mock_status_service, sample_processing_status):
        """Test successful status retrieval by video ID."""
        mock_status_service.get_status_by_video_id.return_value = sample_processing_status
        
        with patch('src.api.status.get_status_service', return_value=mock_status_service):
            response = client.get("/api/status/video/123")
            
            assert response.status_code == 200
            data = response.json()
            assert data["video_id"] == 123
            assert data["status"] == "YOUTUBE_METADATA"
    
    def test_get_status_by_video_not_found(self, client, mock_status_service):
        """Test status retrieval by video ID when not found."""
        mock_status_service.get_status_by_video_id.return_value = None
        
        with patch('src.api.status.get_status_service', return_value=mock_status_service):
            response = client.get("/api/status/video/999")
            
            assert response.status_code == 404
            assert "not found" in response.json()["detail"]
    
    def test_get_status_by_batch_item_success(self, client, mock_status_service, sample_processing_status):
        """Test successful status retrieval by batch item ID."""
        mock_status_service.get_status_by_batch_item_id.return_value = sample_processing_status
        
        with patch('src.api.status.get_status_service', return_value=mock_status_service):
            response = client.get("/api/status/batch/456")
            
            assert response.status_code == 200
            data = response.json()
            assert data["batch_item_id"] == 456
            assert data["status"] == "YOUTUBE_METADATA"
    
    def test_list_statuses_success(self, client, mock_status_service, sample_processing_status):
        """Test successful status listing."""
        mock_status_service.get_active_statuses.return_value = [sample_processing_status]
        
        with patch('src.api.status.get_status_service', return_value=mock_status_service):
            response = client.get("/api/status/?page=1&page_size=10")
            
            assert response.status_code == 200
            data = response.json()
            assert len(data["statuses"]) == 1
            assert data["page"] == 1
            assert data["page_size"] == 10
            assert data["total_count"] == 1
            assert data["has_next"] is False
            assert data["has_previous"] is False
    
    def test_list_statuses_with_filters(self, client, mock_status_service, sample_processing_status):
        """Test status listing with filters."""
        mock_status_service.get_active_statuses.return_value = [sample_processing_status]
        
        with patch('src.api.status.get_status_service', return_value=mock_status_service):
            response = client.get("/api/status/?worker_id=worker-1&active_only=true")
            
            assert response.status_code == 200
            mock_status_service.get_active_statuses.assert_called_with(
                worker_id="worker-1", 
                limit=21  # page_size + 1
            )
    
    def test_get_active_count_success(self, client, mock_status_service, sample_processing_status):
        """Test getting active count."""
        mock_status_service.get_active_statuses.return_value = [sample_processing_status]
        
        with patch('src.api.status.get_status_service', return_value=mock_status_service):
            response = client.get("/api/status/active/count")
            
            assert response.status_code == 200
            data = response.json()
            assert data["active_count"] == 1
    
    def test_get_stale_statuses_success(self, client, mock_status_service, sample_processing_status):
        """Test getting stale statuses."""
        stale_status = sample_processing_status
        stale_status.heartbeat_at = datetime.utcnow() - timedelta(minutes=10)
        mock_status_service.get_stale_statuses.return_value = [stale_status]
        
        with patch('src.api.status.get_status_service', return_value=mock_status_service):
            response = client.get("/api/status/stale/list?timeout_seconds=300")
            
            assert response.status_code == 200
            data = response.json()
            assert len(data) == 1
            assert data[0]["status_id"] == "status_test123"
    
    def test_update_status_success(self, client, mock_status_service, sample_processing_status):
        """Test successful status update."""
        updated_status = sample_processing_status
        updated_status.status = ProcessingStatusType.TRANSCRIPT_EXTRACTION
        updated_status.progress_percentage = 50.0
        
        mock_status_service.update_status.return_value = updated_status
        
        with patch('src.api.status.get_status_service', return_value=mock_status_service):
            update_data = {
                "new_status": "TRANSCRIPT_EXTRACTION",
                "progress_percentage": 50.0,
                "current_step": "Extracting transcript",
                "worker_id": "worker-1"
            }
            
            response = client.put("/api/status/status_test123", json=update_data)
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "TRANSCRIPT_EXTRACTION"
            assert data["progress_percentage"] == 50.0
    
    def test_update_status_invalid_status(self, client, mock_status_service):
        """Test status update with invalid status."""
        with patch('src.api.status.get_status_service', return_value=mock_status_service):
            update_data = {
                "new_status": "INVALID_STATUS",
                "progress_percentage": 50.0
            }
            
            response = client.put("/api/status/status_test123", json=update_data)
            
            assert response.status_code == 422  # Validation error
    
    def test_update_progress_success(self, client, mock_status_service, sample_processing_status):
        """Test successful progress update."""
        updated_status = sample_processing_status
        updated_status.progress_percentage = 75.0
        
        mock_status_service.update_progress.return_value = updated_status
        
        with patch('src.api.status.get_status_service', return_value=mock_status_service):
            progress_data = {
                "progress_percentage": 75.0,
                "current_step": "Processing transcript",
                "completed_steps": 3,
                "worker_id": "worker-1"
            }
            
            response = client.patch("/api/status/status_test123/progress", json=progress_data)
            
            assert response.status_code == 200
            data = response.json()
            assert data["progress_percentage"] == 75.0
    
    def test_update_progress_invalid_percentage(self, client, mock_status_service):
        """Test progress update with invalid percentage."""
        with patch('src.api.status.get_status_service', return_value=mock_status_service):
            progress_data = {
                "progress_percentage": 150.0  # Invalid: > 100
            }
            
            response = client.patch("/api/status/status_test123/progress", json=progress_data)
            
            assert response.status_code == 422  # Validation error
    
    def test_report_error_success(self, client, mock_status_service, sample_processing_status):
        """Test successful error reporting."""
        failed_status = sample_processing_status
        failed_status.status = ProcessingStatusType.RETRY_PENDING
        failed_status.error_info = "Connection timeout"
        
        mock_status_service.record_error.return_value = failed_status
        
        with patch('src.api.status.get_status_service', return_value=mock_status_service):
            error_data = {
                "error_info": "Connection timeout",
                "worker_id": "worker-1",
                "should_retry": True
            }
            
            response = client.post("/api/status/status_test123/error", json=error_data)
            
            assert response.status_code == 200
            data = response.json()
            assert data["error_info"] == "Connection timeout"
            assert data["status"] == "RETRY_PENDING"
    
    def test_update_heartbeat_success(self, client, mock_status_service, sample_processing_status):
        """Test successful heartbeat update."""
        mock_status_service.heartbeat.return_value = sample_processing_status
        
        with patch('src.api.status.get_status_service', return_value=mock_status_service):
            heartbeat_data = {
                "worker_id": "worker-1",
                "progress_percentage": 30.0,
                "current_step": "Still processing"
            }
            
            response = client.post("/api/status/status_test123/heartbeat", json=heartbeat_data)
            
            assert response.status_code == 200
            data = response.json()
            assert data["worker_id"] == "worker-1"
    
    def test_get_status_history_success(self, client, mock_status_service):
        """Test successful status history retrieval."""
        from ..database.status_models import StatusHistory, StatusChangeType
        
        mock_history = [
            StatusHistory(
                id=1,
                processing_status_id=1,
                change_type=StatusChangeType.STATUS_UPDATE,
                previous_status=ProcessingStatusType.QUEUED,
                new_status=ProcessingStatusType.YOUTUBE_METADATA,
                previous_progress=0.0,
                new_progress=25.0,
                change_reason="Status updated",
                created_at=datetime.utcnow(),
                duration_seconds=60
            )
        ]
        
        mock_status_service.get_status_history.return_value = mock_history
        
        with patch('src.api.status.get_status_service', return_value=mock_status_service):
            response = client.get("/api/status/status_test123/history")
            
            assert response.status_code == 200
            data = response.json()
            assert len(data) == 1
            assert data[0]["change_type"] == "STATUS_UPDATE"
            assert data[0]["new_status"] == "YOUTUBE_METADATA"
            assert data[0]["duration_seconds"] == 60
    
    def test_get_status_history_with_filter(self, client, mock_status_service):
        """Test status history retrieval with filter."""
        mock_status_service.get_status_history.return_value = []
        
        with patch('src.api.status.get_status_service', return_value=mock_status_service):
            response = client.get("/api/status/status_test123/history?change_type=STATUS_UPDATE&limit=10")
            
            assert response.status_code == 200
            from ..database.status_models import StatusChangeType
            mock_status_service.get_status_history.assert_called_with(
                status_id="status_test123",
                limit=10,
                change_type=StatusChangeType.STATUS_UPDATE
            )
    
    def test_get_status_history_invalid_change_type(self, client, mock_status_service):
        """Test status history with invalid change type."""
        with patch('src.api.status.get_status_service', return_value=mock_status_service):
            response = client.get("/api/status/status_test123/history?change_type=INVALID_TYPE")
            
            assert response.status_code == 400
            assert "Invalid change type" in response.json()["detail"]


class TestMetricsEndpoints:
    """Test cases for metrics API endpoints."""
    
    def test_get_performance_summary_success(self, client, mock_metrics_service):
        """Test successful performance summary retrieval."""
        mock_summary = {
            "active_processing_count": 5,
            "active_workers": ["worker-1", "worker-2"],
            "worker_count": 2,
            "average_progress": 45.5,
            "today_total": 20,
            "today_completed": 18,
            "today_failed": 2,
            "today_success_rate": 90.0,
            "success_rate_trend": 5.0,
            "recent_metrics_count": 7,
            "timestamp": "2023-01-01T12:00:00"
        }
        
        mock_metrics_service.get_current_performance_summary.return_value = mock_summary
        
        with patch('src.api.status.get_metrics_service', return_value=mock_metrics_service):
            response = client.get("/api/status/metrics/summary")
            
            assert response.status_code == 200
            data = response.json()
            assert data["active_processing_count"] == 5
            assert data["worker_count"] == 2
            assert data["today_success_rate"] == 90.0
    
    def test_get_worker_performance_success(self, client, mock_metrics_service):
        """Test successful worker performance retrieval."""
        mock_performance = {
            "worker_id": "worker-1",
            "total_processed": 100,
            "completed_count": 95,
            "failed_count": 5,
            "success_rate": 95.0,
            "error_rate": 5.0,
            "average_processing_time": 120.5,
            "processing_time_samples": 95,
            "days_analyzed": 7,
            "timestamp": "2023-01-01T12:00:00"
        }
        
        mock_metrics_service.get_worker_performance.return_value = mock_performance
        
        with patch('src.api.status.get_metrics_service', return_value=mock_metrics_service):
            response = client.get("/api/status/metrics/worker/worker-1?days=7")
            
            assert response.status_code == 200
            data = response.json()
            assert data["worker_id"] == "worker-1"
            assert data["success_rate"] == 95.0
            assert data["total_processed"] == 100
    
    def test_get_status_distribution_success(self, client, mock_metrics_service):
        """Test successful status distribution retrieval."""
        mock_distribution = {
            "COMPLETED": 150,
            "FAILED": 10,
            "QUEUED": 5,
            "YOUTUBE_METADATA": 3
        }
        
        mock_metrics_service.get_status_distribution.return_value = mock_distribution
        
        with patch('src.api.status.get_metrics_service', return_value=mock_metrics_service):
            response = client.get("/api/status/metrics/distribution?days=7")
            
            assert response.status_code == 200
            data = response.json()
            assert data["distribution"]["COMPLETED"] == 150
            assert data["distribution"]["FAILED"] == 10
            assert data["days_analyzed"] == 7
    
    def test_get_hourly_metrics_success(self, client, mock_metrics_service):
        """Test successful hourly metrics retrieval."""
        from ..database.status_models import StatusMetrics
        
        mock_metrics = [
            StatusMetrics(
                id=1,
                metric_date=datetime.utcnow(),
                metric_hour=12,
                total_items=50,
                completed_items=45,
                failed_items=3,
                cancelled_items=2,
                success_rate_percentage=90.0,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
        ]
        
        mock_metrics_service.get_metrics_for_period.return_value = mock_metrics
        
        with patch('src.api.status.get_metrics_service', return_value=mock_metrics_service):
            start_date = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
            end_date = start_date + timedelta(hours=23, minutes=59, seconds=59)
            
            response = client.get(
                f"/api/status/metrics/hourly?start_date={start_date.isoformat()}&end_date={end_date.isoformat()}"
            )
            
            assert response.status_code == 200
            data = response.json()
            assert len(data) == 1
            assert data[0]["metric_hour"] == 12
            assert data[0]["total_items"] == 50
            assert data[0]["success_rate_percentage"] == 90.0
    
    def test_get_hourly_metrics_date_range_too_large(self, client, mock_metrics_service):
        """Test hourly metrics with date range too large."""
        with patch('src.api.status.get_metrics_service', return_value=mock_metrics_service):
            start_date = datetime.utcnow()
            end_date = start_date + timedelta(days=10)  # Too large for hourly
            
            response = client.get(
                f"/api/status/metrics/hourly?start_date={start_date.isoformat()}&end_date={end_date.isoformat()}"
            )
            
            assert response.status_code == 400
            assert "cannot exceed 7 days" in response.json()["detail"]
    
    def test_get_daily_metrics_success(self, client, mock_metrics_service):
        """Test successful daily metrics retrieval."""
        from ..database.status_models import StatusMetrics
        
        mock_metrics = [
            StatusMetrics(
                id=1,
                metric_date=datetime.utcnow(),
                metric_hour=None,  # Daily metric
                total_items=500,
                completed_items=450,
                failed_items=30,
                cancelled_items=20,
                success_rate_percentage=90.0,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
        ]
        
        mock_metrics_service.get_metrics_for_period.return_value = mock_metrics
        
        with patch('src.api.status.get_metrics_service', return_value=mock_metrics_service):
            start_date = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
            end_date = start_date + timedelta(days=6)
            
            response = client.get(
                f"/api/status/metrics/daily?start_date={start_date.isoformat()}&end_date={end_date.isoformat()}"
            )
            
            assert response.status_code == 200
            data = response.json()
            assert len(data) == 1
            assert data[0]["metric_hour"] is None
            assert data[0]["total_items"] == 500
            assert data[0]["success_rate_percentage"] == 90.0
    
    def test_get_daily_metrics_date_range_too_large(self, client, mock_metrics_service):
        """Test daily metrics with date range too large."""
        with patch('src.api.status.get_metrics_service', return_value=mock_metrics_service):
            start_date = datetime.utcnow()
            end_date = start_date + timedelta(days=100)  # Too large for daily
            
            response = client.get(
                f"/api/status/metrics/daily?start_date={start_date.isoformat()}&end_date={end_date.isoformat()}"
            )
            
            assert response.status_code == 400
            assert "cannot exceed 90 days" in response.json()["detail"]


class TestUtilityEndpoints:
    """Test cases for utility API endpoints."""
    
    def test_cleanup_old_statuses_success(self, client, mock_status_service):
        """Test successful cleanup of old statuses."""
        mock_status_service.cleanup_old_statuses.return_value = 25
        
        with patch('src.api.status.get_status_service', return_value=mock_status_service):
            response = client.delete("/api/status/cleanup?days_old=30&keep_failed=true")
            
            assert response.status_code == 200
            data = response.json()
            assert data["cleaned_up_count"] == 25
            
            mock_status_service.cleanup_old_statuses.assert_called_with(30, True)
    
    def test_health_check_success(self, client):
        """Test health check endpoint."""
        response = client.get("/api/status/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
    
    def test_get_status_types_success(self, client):
        """Test getting available status types."""
        response = client.get("/api/status/enums/status-types")
        
        assert response.status_code == 200
        data = response.json()
        assert "status_types" in data
        assert "QUEUED" in data["status_types"]
        assert "COMPLETED" in data["status_types"]
    
    def test_get_priorities_success(self, client):
        """Test getting available priorities."""
        response = client.get("/api/status/enums/priorities")
        
        assert response.status_code == 200
        data = response.json()
        assert "priorities" in data
        assert "LOW" in data["priorities"]
        assert "HIGH" in data["priorities"]
    
    def test_get_change_types_success(self, client):
        """Test getting available change types."""
        response = client.get("/api/status/enums/change-types")
        
        assert response.status_code == 200
        data = response.json()
        assert "change_types" in data
        assert "STATUS_UPDATE" in data["change_types"]
        assert "PROGRESS_UPDATE" in data["change_types"]


class TestErrorHandling:
    """Test cases for error handling in API endpoints."""
    
    def test_get_status_service_error(self, client, mock_status_service):
        """Test service error handling."""
        mock_status_service.get_processing_status.side_effect = Exception("Database error")
        
        with patch('src.api.status.get_status_service', return_value=mock_status_service):
            response = client.get("/api/status/status_test123")
            
            assert response.status_code == 500
            assert "Database error" in response.json()["detail"]
    
    def test_update_status_service_error(self, client, mock_status_service):
        """Test service error in status update."""
        mock_status_service.update_status.side_effect = Exception("Update failed")
        
        with patch('src.api.status.get_status_service', return_value=mock_status_service):
            update_data = {
                "new_status": "COMPLETED",
                "progress_percentage": 100.0
            }
            
            response = client.put("/api/status/status_test123", json=update_data)
            
            assert response.status_code == 500
            assert "Update failed" in response.json()["detail"]
    
    def test_metrics_service_error(self, client, mock_metrics_service):
        """Test service error in metrics endpoint."""
        mock_metrics_service.get_current_performance_summary.side_effect = Exception("Metrics error")
        
        with patch('src.api.status.get_metrics_service', return_value=mock_metrics_service):
            response = client.get("/api/status/metrics/summary")
            
            assert response.status_code == 500
            assert "Metrics error" in response.json()["detail"]


if __name__ == "__main__":
    pytest.main([__file__])