"""
Focused tests for status tracking API endpoints.

This module provides specific tests for all status tracking API endpoints
including request validation, response formatting, and error handling.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient
from fastapi import HTTPException

from src.api.status import router as status_router, StatusResponse, StatusListResponse
from src.api.status_enhanced import router as enhanced_router, EnhancedStatusListResponse
from src.database.status_models import ProcessingStatusType, ProcessingPriority
from src.services.status_filtering import FilterResult


class TestBasicStatusEndpoints:
    """Test cases for basic status API endpoints."""
    
    @pytest.fixture
    def mock_status_service(self):
        """Create mock status service."""
        return Mock()
    
    def test_get_status_success(self, mock_status_service):
        """Test successful status retrieval."""
        from src.api.status import get_status
        
        # Create mock status
        mock_status = Mock()
        mock_status.status_id = "test_123"
        mock_status.status = ProcessingStatusType.COMPLETED
        mock_status.progress_percentage = 100.0
        mock_status_service.get_processing_status.return_value = mock_status
        
        # Test endpoint
        result = get_status("test_123", mock_status_service)
        
        # Verify service call
        mock_status_service.get_processing_status.assert_called_once_with("test_123")
    
    def test_get_status_not_found(self, mock_status_service):
        """Test status not found error."""
        from src.api.status import get_status
        
        mock_status_service.get_processing_status.return_value = None
        
        with pytest.raises(HTTPException) as exc_info:
            get_status("nonexistent", mock_status_service)
        
        assert exc_info.value.status_code == 404
        assert "not found" in exc_info.value.detail
    
    def test_get_status_by_video_success(self, mock_status_service):
        """Test successful status retrieval by video ID."""
        from src.api.status import get_status_by_video
        
        mock_status = Mock()
        mock_status.video_id = 123
        mock_status_service.get_status_by_video_id.return_value = mock_status
        
        result = get_status_by_video(123, mock_status_service)
        
        mock_status_service.get_status_by_video_id.assert_called_once_with(123)
    
    def test_get_status_by_batch_item_success(self, mock_status_service):
        """Test successful status retrieval by batch item ID."""
        from src.api.status import get_status_by_batch_item
        
        mock_status = Mock()
        mock_status.batch_item_id = 456
        mock_status_service.get_status_by_batch_item_id.return_value = mock_status
        
        result = get_status_by_batch_item(456, mock_status_service)
        
        mock_status_service.get_status_by_batch_item_id.assert_called_once_with(456)
    
    def test_list_statuses_active_only(self, mock_status_service):
        """Test listing active statuses only."""
        from src.api.status import list_statuses
        
        mock_statuses = [Mock(), Mock(), Mock()]
        mock_status_service.get_active_statuses.return_value = mock_statuses
        
        result = list_statuses(
            page=1,
            page_size=20,
            active_only=True,
            service=mock_status_service
        )
        
        mock_status_service.get_active_statuses.assert_called_once_with(
            worker_id=None,
            limit=21  # page_size + 1 for has_next check
        )
    
    def test_get_active_count(self, mock_status_service):
        """Test getting active status count."""
        from src.api.status import get_active_count
        
        mock_statuses = [Mock(), Mock(), Mock()]
        mock_status_service.get_active_statuses.return_value = mock_statuses
        
        result = get_active_count(worker_id="worker_123", service=mock_status_service)
        
        assert result["active_count"] == 3
        mock_status_service.get_active_statuses.assert_called_once_with(worker_id="worker_123")
    
    def test_get_stale_statuses(self, mock_status_service):
        """Test getting stale statuses."""
        from src.api.status import get_stale_statuses
        
        mock_statuses = [Mock()]
        mock_status_service.get_stale_statuses.return_value = mock_statuses
        
        result = get_stale_statuses(
            timeout_seconds=600,
            limit=10,
            service=mock_status_service
        )
        
        assert len(result) == 1
        mock_status_service.get_stale_statuses.assert_called_once_with(
            timeout_seconds=600,
            limit=10
        )


class TestStatusUpdateEndpoints:
    """Test cases for status update API endpoints."""
    
    @pytest.fixture
    def mock_status_service(self):
        """Create mock status service."""
        return Mock()
    
    def test_update_status_success(self, mock_status_service):
        """Test successful status update."""
        from src.api.status import update_status, StatusUpdateRequest
        
        # Create update request
        update_request = StatusUpdateRequest(
            new_status="completed",
            progress_percentage=100.0,
            current_step="Finished",
            change_reason="Processing completed"
        )
        
        # Mock updated status
        mock_status = Mock()
        mock_status_service.update_status.return_value = mock_status
        
        result = update_status("test_123", update_request, mock_status_service)
        
        # Verify service call
        mock_status_service.update_status.assert_called_once()
        call_args = mock_status_service.update_status.call_args
        assert call_args.kwargs['status_id'] == "test_123"
        assert call_args.kwargs['new_status'] == ProcessingStatusType.COMPLETED
        assert call_args.kwargs['progress_percentage'] == 100.0
    
    def test_update_status_invalid_status(self, mock_status_service):
        """Test status update with invalid status."""
        from src.api.status import update_status, StatusUpdateRequest
        
        update_request = StatusUpdateRequest(
            new_status="invalid_status",
            progress_percentage=50.0
        )
        
        with pytest.raises(HTTPException) as exc_info:
            update_status("test_123", update_request, mock_status_service)
        
        assert exc_info.value.status_code == 400
    
    def test_update_progress_success(self, mock_status_service):
        """Test successful progress update."""
        from src.api.status import update_progress, ProgressUpdateRequest
        
        update_request = ProgressUpdateRequest(
            progress_percentage=75.0,
            current_step="Processing data",
            worker_id="worker_abc"
        )
        
        mock_status = Mock()
        mock_status_service.update_progress.return_value = mock_status
        
        result = update_progress("test_123", update_request, mock_status_service)
        
        mock_status_service.update_progress.assert_called_once()
    
    def test_report_error_success(self, mock_status_service):
        """Test successful error reporting."""
        from src.api.status import report_error, ErrorReportRequest
        
        error_request = ErrorReportRequest(
            error_info="Network timeout occurred",
            worker_id="worker_abc",
            should_retry=True
        )
        
        mock_status = Mock()
        mock_status_service.record_error.return_value = mock_status
        
        result = report_error("test_123", error_request, mock_status_service)
        
        mock_status_service.record_error.assert_called_once()
    
    def test_send_heartbeat_success(self, mock_status_service):
        """Test successful heartbeat."""
        from src.api.status import send_heartbeat, HeartbeatRequest
        
        heartbeat_request = HeartbeatRequest(
            worker_id="worker_abc",
            progress_percentage=60.0,
            current_step="Still processing"
        )
        
        mock_status = Mock()
        mock_status_service.heartbeat.return_value = mock_status
        
        result = send_heartbeat("test_123", heartbeat_request, mock_status_service)
        
        mock_status_service.heartbeat.assert_called_once()


class TestEnhancedStatusEndpoints:
    """Test cases for enhanced status API endpoints."""
    
    @pytest.fixture
    def mock_filter_service(self):
        """Create mock filter service."""
        return Mock()
    
    def test_filter_statuses_success(self, mock_filter_service):
        """Test successful status filtering."""
        from src.api.status_enhanced import filter_statuses, AdvancedFilterRequest, FilterConditionModel
        
        # Create filter request
        filter_request = AdvancedFilterRequest(
            filters=[
                FilterConditionModel(
                    field="status",
                    operator="eq",
                    value="completed"
                ),
                FilterConditionModel(
                    field="priority",
                    operator="in",
                    value=["high", "normal"]
                )
            ],
            sorts=[
                {"field": "updated_at", "order": "desc"}
            ],
            page=1,
            page_size=25
        )
        
        # Mock filter result
        mock_result = FilterResult(
            items=[Mock(), Mock()],
            total_count=50,
            page=1,
            page_size=25,
            total_pages=2,
            has_next=True,
            has_previous=False,
            filters_applied=2,
            query_time_ms=15.5,
            metadata={"search_applied": False}
        )
        mock_filter_service.filter_statuses.return_value = mock_result
        
        result = filter_statuses(filter_request, mock_filter_service)
        
        # Verify result
        assert result.total_count == 50
        assert result.filters_applied == 2
        assert result.query_time_ms == 15.5
        assert result.has_next is True
        
        # Verify service call
        mock_filter_service.filter_statuses.assert_called_once()
    
    def test_get_preset_filter_active(self, mock_filter_service):
        """Test getting active statuses preset."""
        from src.api.status_enhanced import get_preset_filter, PresetFilterType
        
        mock_result = FilterResult(
            items=[Mock()],
            total_count=10,
            page=1,
            page_size=20,
            total_pages=1,
            has_next=False,
            has_previous=False,
            filters_applied=1,
            query_time_ms=8.0
        )
        mock_filter_service.filter_statuses.return_value = mock_result
        
        result = get_preset_filter(
            PresetFilterType.ACTIVE,
            page=1,
            page_size=20,
            filter_service=mock_filter_service
        )
        
        assert result.total_count == 10
        assert result.metadata['preset_type'] == 'active'
    
    def test_get_preset_filter_failed(self, mock_filter_service):
        """Test getting failed statuses preset."""
        from src.api.status_enhanced import get_preset_filter, PresetFilterType
        
        mock_result = FilterResult(
            items=[],
            total_count=0,
            page=1,
            page_size=20,
            total_pages=0,
            has_next=False,
            has_previous=False,
            filters_applied=2,
            query_time_ms=5.0
        )
        mock_filter_service.filter_statuses.return_value = mock_result
        
        result = get_preset_filter(
            PresetFilterType.FAILED,
            filter_service=mock_filter_service
        )
        
        assert result.total_count == 0
        assert result.filters_applied == 2
    
    def test_search_statuses_success(self, mock_filter_service):
        """Test successful status search."""
        from src.api.status_enhanced import search_statuses
        
        mock_result = FilterResult(
            items=[Mock(), Mock()],
            total_count=15,
            page=1,
            page_size=20,
            total_pages=1,
            has_next=False,
            has_previous=False,
            filters_applied=0,
            query_time_ms=12.0,
            metadata={"search_applied": True}
        )
        mock_filter_service.filter_statuses.return_value = mock_result
        
        result = search_statuses(
            q="error timeout",
            fields=["error_info", "current_step"],
            exact_match=False,
            case_sensitive=False,
            filter_service=mock_filter_service
        )
        
        assert result.total_count == 15
        assert result.metadata['search_query'] == "error timeout"
        assert result.metadata['search_fields'] == ["error_info", "current_step"]
    
    def test_get_status_aggregates_success(self, mock_filter_service):
        """Test getting status aggregates."""
        from src.api.status_enhanced import get_status_aggregates
        
        mock_aggregates = {
            'total_count': 1000,
            'status_distribution': {
                'completed': 800,
                'failed': 150,
                'starting': 50
            },
            'priority_distribution': {
                'normal': 900,
                'high': 100
            },
            'progress_statistics': {
                'average': 75.5,
                'minimum': 0.0,
                'maximum': 100.0
            },
            'worker_statistics': {
                'unique_workers': 10,
                'total_with_workers': 950
            },
            'time_statistics': {
                'last_hour': 45,
                'last_24_hours': 200,
                'last_7_days': 800
            },
            'timestamp': '2023-12-01T12:00:00Z'
        }
        mock_filter_service.get_status_aggregates.return_value = mock_aggregates
        
        result = get_status_aggregates(filter_service=mock_filter_service)
        
        assert result.total_count == 1000
        assert result.status_distribution['completed'] == 800
        assert result.worker_statistics['unique_workers'] == 10
    
    def test_filter_status_history_success(self, mock_filter_service):
        """Test filtering status history."""
        from src.api.status_enhanced import filter_status_history, AdvancedFilterRequest
        
        filter_request = AdvancedFilterRequest(
            filters=[],
            page=1,
            page_size=10
        )
        
        mock_result = FilterResult(
            items=[Mock(), Mock()],
            total_count=5,
            page=1,
            page_size=10,
            total_pages=1,
            has_next=False,
            has_previous=False,
            filters_applied=0,
            query_time_ms=3.0
        )
        mock_filter_service.filter_status_history.return_value = mock_result
        
        result = filter_status_history(
            status_id="test_123",
            filter_request=filter_request,
            filter_service=mock_filter_service
        )
        
        assert len(result) == 2
        mock_filter_service.filter_status_history.assert_called_once_with(
            status_id="test_123",
            filter_query=pytest.any  # Any FilterQuery object
        )
    
    def test_get_available_fields(self):
        """Test getting available fields schema."""
        from src.api.status_enhanced import get_available_fields
        
        result = get_available_fields()
        
        assert 'filterable_fields' in result
        assert 'sortable_fields' in result
        assert 'searchable_fields' in result
        assert 'status_id' in result['filterable_fields']
        assert 'created_at' in result['sortable_fields']
        assert 'error_info' in result['searchable_fields']
    
    def test_get_available_operators(self):
        """Test getting available operators schema."""
        from src.api.status_enhanced import get_available_operators
        
        result = get_available_operators()
        
        assert 'operators' in result
        assert 'sort_orders' in result
        
        # Check some expected operators
        operator_names = [op['name'] for op in result['operators']]
        assert 'eq' in operator_names
        assert 'in' in operator_names
        assert 'like' in operator_names
        assert 'between' in operator_names
        
        # Check sort orders
        assert 'asc' in result['sort_orders']
        assert 'desc' in result['sort_orders']


class TestErrorHandling:
    """Test cases for API error handling."""
    
    def test_invalid_filter_operator(self):
        """Test invalid filter operator error."""
        from src.api.status_enhanced import FilterConditionModel
        
        with pytest.raises(ValueError, match="Invalid filter operator"):
            FilterConditionModel(
                field="status",
                operator="invalid_operator",
                value="test"
            )
    
    def test_invalid_sort_order(self):
        """Test invalid sort order error."""
        from src.api.status_enhanced import SortConditionModel
        
        with pytest.raises(ValueError, match="Sort order must be"):
            SortConditionModel(
                field="created_at",
                order="invalid_order"
            )
    
    def test_service_error_handling(self, mock_status_service):
        """Test service error handling."""
        from src.api.status import get_status
        
        # Mock service to raise exception
        mock_status_service.get_processing_status.side_effect = Exception("Database error")
        
        with pytest.raises(HTTPException) as exc_info:
            get_status("test_123", mock_status_service)
        
        assert exc_info.value.status_code == 500
        assert "Error retrieving status" in exc_info.value.detail


class TestRequestValidation:
    """Test cases for API request validation."""
    
    def test_status_update_request_validation(self):
        """Test status update request validation."""
        from src.api.status import StatusUpdateRequest
        
        # Valid request
        request = StatusUpdateRequest(
            new_status="completed",
            progress_percentage=100.0,
            current_step="Done"
        )
        assert request.new_status == "completed"
        assert request.progress_percentage == 100.0
        
        # Invalid progress percentage
        with pytest.raises(ValueError):
            StatusUpdateRequest(
                new_status="completed",
                progress_percentage=150.0  # > 100
            )
    
    def test_progress_update_request_validation(self):
        """Test progress update request validation."""
        from src.api.status import ProgressUpdateRequest
        
        # Valid request
        request = ProgressUpdateRequest(
            progress_percentage=75.0,
            current_step="Processing"
        )
        assert request.progress_percentage == 75.0
        
        # Invalid progress percentage (negative)
        with pytest.raises(ValueError):
            ProgressUpdateRequest(progress_percentage=-10.0)
    
    def test_heartbeat_request_validation(self):
        """Test heartbeat request validation."""
        from src.api.status import HeartbeatRequest
        
        # Valid request
        request = HeartbeatRequest(
            worker_id="worker_123",
            progress_percentage=50.0
        )
        assert request.worker_id == "worker_123"
        assert request.progress_percentage == 50.0
        
        # Missing worker_id should fail
        with pytest.raises(ValueError):
            HeartbeatRequest(progress_percentage=50.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])