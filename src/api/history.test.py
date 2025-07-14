"""
Comprehensive tests for History API endpoints.

This test suite covers all the API endpoints for:
- Video deletion operations
- Reprocessing operations  
- Transaction management
- Error handling
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from fastapi.testclient import TestClient
from fastapi import HTTPException

from ..api.history import router
from ..services.history_service import HistoryService, HistoryServiceError
from ..services.reprocessing_service import (
    ReprocessingService, ReprocessingServiceError, ReprocessingMode, 
    ReprocessingValidation, ReprocessingResult, ReprocessingStatus
)
from ..database.cascade_delete import CascadeDeleteValidation, CascadeDeleteResult
from ..database.transaction_manager import TransactionResult, TransactionStatus


@pytest.fixture
def client():
    """Create test client."""
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


@pytest.fixture
def mock_history_service():
    """Create mock HistoryService."""
    return Mock(spec=HistoryService)


@pytest.fixture
def mock_reprocessing_service():
    """Create mock ReprocessingService."""
    return Mock(spec=ReprocessingService)


class TestVideoDeletionEndpoints:
    """Test suite for video deletion API endpoints."""
    
    def test_get_video_deletion_info_success(self, client, mock_history_service):
        """Test successful video deletion info retrieval."""
        # Setup
        deletion_info = {
            "video": {
                "id": 1,
                "video_id": "test_video_123",
                "title": "Test Video",
                "url": "https://youtube.com/watch?v=test_video_123",
                "created_at": datetime.now(),
                "updated_at": datetime.now()
            },
            "related_data_counts": {
                "transcripts": 1,
                "summaries": 1,
                "keywords": 1,
                "timestamped_segments": 0,
                "processing_metadata": 2
            },
            "total_related_records": 5
        }
        
        mock_history_service.get_video_deletion_info.return_value = deletion_info
        
        with patch('..api.history.get_history_service', return_value=mock_history_service):
            # Execute
            response = client.get("/videos/1/deletion-info")
            
            # Verify
            assert response.status_code == 200
            data = response.json()
            assert data["video"]["id"] == 1
            assert data["total_related_records"] == 5
    
    def test_get_video_deletion_info_not_found(self, client, mock_history_service):
        """Test video deletion info for non-existent video."""
        # Setup
        mock_history_service.get_video_deletion_info.return_value = None
        
        with patch('..api.history.get_history_service', return_value=mock_history_service):
            # Execute
            response = client.get("/videos/999/deletion-info")
            
            # Verify
            assert response.status_code == 404
    
    def test_validate_video_deletion_success(self, client, mock_history_service):
        """Test successful video deletion validation."""
        # Setup
        validation = CascadeDeleteValidation(
            can_delete=True,
            video_exists=True,
            related_counts={"transcripts": 1, "summaries": 1},
            potential_issues=[],
            total_related_records=2
        )
        
        mock_history_service.validate_video_deletion.return_value = validation
        
        with patch('..api.history.get_history_service', return_value=mock_history_service):
            # Execute
            response = client.get("/videos/1/validate-deletion")
            
            # Verify
            assert response.status_code == 200
            data = response.json()
            assert data["can_delete"] is True
            assert data["video_exists"] is True
            assert data["total_related_records"] == 2
    
    def test_delete_video_success(self, client, mock_history_service):
        """Test successful video deletion."""
        # Setup
        delete_result = CascadeDeleteResult(
            success=True,
            video_id=1,
            deleted_counts={"transcripts": 1, "summaries": 1, "videos": 1},
            total_deleted=3,
            execution_time_ms=150.5
        )
        
        mock_history_service.enhanced_delete_video_by_id.return_value = delete_result
        
        with patch('..api.history.get_history_service', return_value=mock_history_service):
            # Execute
            response = client.delete("/videos/1", json={"force": False, "audit_user": "test_user"})
            
            # Verify
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["video_id"] == 1
            assert data["total_deleted"] == 3
    
    def test_delete_video_failure(self, client, mock_history_service):
        """Test video deletion failure."""
        # Setup
        delete_result = CascadeDeleteResult(
            success=False,
            video_id=1,
            deleted_counts={},
            total_deleted=0,
            error_message="Video is currently being processed"
        )
        
        mock_history_service.enhanced_delete_video_by_id.return_value = delete_result
        
        with patch('..api.history.get_history_service', return_value=mock_history_service):
            # Execute
            response = client.delete("/videos/1", json={"force": False})
            
            # Verify
            assert response.status_code == 400
            assert "Video is currently being processed" in response.json()["detail"]
    
    def test_delete_video_by_youtube_id_success(self, client, mock_history_service):
        """Test successful video deletion by YouTube ID."""
        # Setup
        mock_history_service.delete_video_by_video_id.return_value = True
        
        with patch('..api.history.get_history_service', return_value=mock_history_service):
            # Execute
            response = client.delete("/videos/test_video_123/by-youtube-id", json={"force": False})
            
            # Verify
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
    
    def test_delete_video_by_youtube_id_not_found(self, client, mock_history_service):
        """Test video deletion by YouTube ID when video doesn't exist."""
        # Setup
        mock_history_service.delete_video_by_video_id.return_value = False
        
        with patch('..api.history.get_history_service', return_value=mock_history_service):
            # Execute
            response = client.delete("/videos/nonexistent/by-youtube-id", json={"force": False})
            
            # Verify
            assert response.status_code == 404
    
    def test_batch_delete_videos_success(self, client, mock_history_service):
        """Test successful batch video deletion."""
        # Setup
        delete_results = [
            CascadeDeleteResult(
                success=True,
                video_id=1,
                deleted_counts={"videos": 1},
                total_deleted=1,
                execution_time_ms=100.0
            ),
            CascadeDeleteResult(
                success=True,
                video_id=2,
                deleted_counts={"videos": 1},
                total_deleted=1,
                execution_time_ms=120.0
            )
        ]
        
        mock_history_service.enhanced_batch_delete_videos.return_value = delete_results
        
        with patch('..api.history.get_history_service', return_value=mock_history_service):
            # Execute
            response = client.post("/videos/batch-delete", json={
                "video_ids": [1, 2],
                "force": False,
                "audit_user": "test_user"
            })
            
            # Verify
            assert response.status_code == 200
            data = response.json()
            assert len(data["results"]) == 2
            assert data["summary"]["total_videos"] == 2
            assert data["summary"]["successful_deletions"] == 2
            assert data["summary"]["failed_deletions"] == 0
    
    def test_batch_delete_videos_empty_list(self, client, mock_history_service):
        """Test batch deletion with empty video list."""
        with patch('..api.history.get_history_service', return_value=mock_history_service):
            # Execute
            response = client.post("/videos/batch-delete", json={
                "video_ids": [],
                "force": False
            })
            
            # Verify
            assert response.status_code == 400
            assert "No video IDs provided" in response.json()["detail"]
    
    def test_batch_delete_videos_too_many(self, client, mock_history_service):
        """Test batch deletion with too many videos."""
        with patch('..api.history.get_history_service', return_value=mock_history_service):
            # Execute
            response = client.post("/videos/batch-delete", json={
                "video_ids": list(range(101)),  # 101 videos
                "force": False
            })
            
            # Verify
            assert response.status_code == 400
            assert "limited to 100 videos" in response.json()["detail"]
    
    def test_check_deletion_integrity_success(self, client, mock_history_service):
        """Test successful deletion integrity check."""
        # Setup
        integrity_result = {
            "video_exists": False,
            "has_orphaned_records": False,
            "orphaned_records": {},
            "integrity_check_passed": True
        }
        
        mock_history_service.verify_cascade_delete_integrity.return_value = integrity_result
        
        with patch('..api.history.get_history_service', return_value=mock_history_service):
            # Execute
            response = client.get("/videos/1/integrity-check")
            
            # Verify
            assert response.status_code == 200
            data = response.json()
            assert data["video_exists"] is False
            assert data["has_orphaned_records"] is False
            assert data["integrity_check_passed"] is True
    
    def test_cleanup_orphaned_records_success(self, client, mock_history_service):
        """Test successful orphaned records cleanup."""
        # Setup
        cleaned_counts = {"transcripts": 2, "summaries": 1}
        mock_history_service.cleanup_orphaned_records.return_value = cleaned_counts
        
        with patch('..api.history.get_history_service', return_value=mock_history_service):
            # Execute
            response = client.post("/videos/1/cleanup-orphans")
            
            # Verify
            assert response.status_code == 200
            data = response.json()
            assert data["video_id"] == 1
            assert data["total_cleaned"] == 3
            assert "transcripts" in data["cleaned_records"]
    
    def test_get_cascade_delete_statistics_success(self, client, mock_history_service):
        """Test successful cascade delete statistics retrieval."""
        # Setup
        stats = {
            "total_videos": 100,
            "average_related_records": {
                "transcripts": {"total_records": 100, "avg_per_video": 1.0},
                "summaries": {"total_records": 80, "avg_per_video": 0.8}
            },
            "videos_with_most_related": [
                {"id": 1, "video_id": "test1", "title": "Test Video 1", "total_related": 10}
            ]
        }
        
        mock_history_service.get_cascade_delete_statistics.return_value = stats
        
        with patch('..api.history.get_history_service', return_value=mock_history_service):
            # Execute
            response = client.get("/deletion-statistics")
            
            # Verify
            assert response.status_code == 200
            data = response.json()
            assert data["total_videos"] == 100
            assert "average_related_records" in data
            assert "videos_with_most_related" in data


class TestReprocessingEndpoints:
    """Test suite for reprocessing API endpoints."""
    
    def test_validate_video_reprocessing_success(self, client, mock_reprocessing_service):
        """Test successful video reprocessing validation."""
        # Setup
        validation = ReprocessingValidation(
            can_reprocess=True,
            video_exists=True,
            current_status="completed",
            existing_components={"transcripts": 1, "summaries": 1},
            potential_issues=[],
            recommendations=["Consider incremental mode"]
        )
        
        mock_reprocessing_service.validate_reprocessing_request.return_value = validation
        
        with patch('..api.history.get_reprocessing_service', return_value=mock_reprocessing_service):
            # Execute
            response = client.get("/videos/1/validate-reprocessing?mode=full")
            
            # Verify
            assert response.status_code == 200
            data = response.json()
            assert data["can_reprocess"] is True
            assert data["video_exists"] is True
            assert len(data["recommendations"]) == 1
    
    def test_reprocess_video_success(self, client, mock_reprocessing_service):
        """Test successful video reprocessing."""
        # Setup
        reprocess_result = ReprocessingResult(
            success=True,
            video_id=1,
            mode=ReprocessingMode.FULL,
            status=ReprocessingStatus.PENDING,
            message="Reprocessing initiated successfully",
            cleared_components=["transcripts", "summaries"],
            processing_metadata_id=123,
            start_time=datetime.now(),
            end_time=datetime.now(),
            execution_time_seconds=0.5
        )
        
        mock_reprocessing_service.initiate_reprocessing.return_value = reprocess_result
        
        with patch('..api.history.get_reprocessing_service', return_value=mock_reprocessing_service):
            # Execute
            response = client.post("/videos/1/reprocess", json={
                "mode": "full",
                "force": False,
                "clear_cache": True,
                "preserve_metadata": True,
                "requested_by": "test_user"
            })
            
            # Verify
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["video_id"] == 1
            assert data["mode"] == "full"
            assert data["status"] == "pending"
    
    def test_reprocess_video_failure(self, client, mock_reprocessing_service):
        """Test video reprocessing failure."""
        # Setup
        reprocess_result = ReprocessingResult(
            success=False,
            video_id=1,
            mode=ReprocessingMode.FULL,
            status=ReprocessingStatus.FAILED,
            message="Validation failed: Video is being processed",
            cleared_components=[]
        )
        
        mock_reprocessing_service.initiate_reprocessing.return_value = reprocess_result
        
        with patch('..api.history.get_reprocessing_service', return_value=mock_reprocessing_service):
            # Execute
            response = client.post("/videos/1/reprocess", json={
                "mode": "full",
                "force": False
            })
            
            # Verify
            assert response.status_code == 400
            assert "Validation failed" in response.json()["detail"]
    
    def test_get_video_reprocessing_status_success(self, client, mock_reprocessing_service):
        """Test successful reprocessing status retrieval."""
        # Setup
        status_info = {
            "video_id": 1,
            "video_title": "Test Video",
            "video_youtube_id": "test_video_123",
            "processing_metadata_id": 123,
            "status": "processing",
            "workflow_params": {"reprocessing_mode": "full"},
            "error_info": None,
            "created_at": datetime.now(),
            "is_reprocessing": True
        }
        
        mock_reprocessing_service.get_reprocessing_status.return_value = status_info
        
        with patch('..api.history.get_reprocessing_service', return_value=mock_reprocessing_service):
            # Execute
            response = client.get("/videos/1/reprocessing-status")
            
            # Verify
            assert response.status_code == 200
            data = response.json()
            assert data["video_id"] == 1
            assert data["status"] == "processing"
            assert data["is_reprocessing"] is True
    
    def test_get_video_reprocessing_status_not_found(self, client, mock_reprocessing_service):
        """Test reprocessing status for non-existent video."""
        # Setup
        mock_reprocessing_service.get_reprocessing_status.return_value = None
        
        with patch('..api.history.get_reprocessing_service', return_value=mock_reprocessing_service):
            # Execute
            response = client.get("/videos/999/reprocessing-status")
            
            # Verify
            assert response.status_code == 404
    
    def test_cancel_video_reprocessing_success(self, client, mock_reprocessing_service):
        """Test successful reprocessing cancellation."""
        # Setup
        mock_reprocessing_service.cancel_reprocessing.return_value = True
        
        with patch('..api.history.get_reprocessing_service', return_value=mock_reprocessing_service):
            # Execute
            response = client.post("/videos/1/cancel-reprocessing", json={
                "reason": "User requested cancellation"
            })
            
            # Verify
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["video_id"] == 1
    
    def test_cancel_video_reprocessing_not_found(self, client, mock_reprocessing_service):
        """Test reprocessing cancellation when no active reprocessing exists."""
        # Setup
        mock_reprocessing_service.cancel_reprocessing.return_value = False
        
        with patch('..api.history.get_reprocessing_service', return_value=mock_reprocessing_service):
            # Execute
            response = client.post("/videos/1/cancel-reprocessing", json={
                "reason": "User requested cancellation"
            })
            
            # Verify
            assert response.status_code == 404
    
    def test_get_video_reprocessing_history_success(self, client, mock_reprocessing_service):
        """Test successful reprocessing history retrieval."""
        # Setup
        history = [
            {
                "id": 1,
                "status": "completed",
                "workflow_params": {"reprocessing_mode": "full"},
                "error_info": None,
                "created_at": datetime.now(),
                "is_reprocessing": True
            },
            {
                "id": 2,
                "status": "failed",
                "workflow_params": {"reprocessing_mode": "transcript_only"},
                "error_info": "Transcript extraction failed",
                "created_at": datetime.now(),
                "is_reprocessing": True
            }
        ]
        
        mock_reprocessing_service.get_reprocessing_history.return_value = history
        
        with patch('..api.history.get_reprocessing_service', return_value=mock_reprocessing_service):
            # Execute
            response = client.get("/videos/1/reprocessing-history?limit=10")
            
            # Verify
            assert response.status_code == 200
            data = response.json()
            assert data["total_count"] == 2
            assert len(data["history"]) == 2
            assert data["history"][0]["status"] == "completed"
            assert data["history"][1]["status"] == "failed"
    
    def test_clear_video_cache_success(self, client, mock_reprocessing_service):
        """Test successful video cache clearing."""
        # Setup
        cleared_components = ["transcripts", "summaries"]
        mock_reprocessing_service.clear_video_cache.return_value = cleared_components
        
        with patch('..api.history.get_reprocessing_service', return_value=mock_reprocessing_service):
            # Execute
            response = client.post("/videos/1/clear-cache?mode=full&preserve_metadata=true")
            
            # Verify
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["video_id"] == 1
            assert data["mode"] == "full"
            assert data["cleared_components"] == cleared_components


class TestTransactionalDeletionEndpoints:
    """Test suite for transactional deletion API endpoints."""
    
    def test_transactional_delete_video_success(self, client, mock_history_service):
        """Test successful transactional video deletion."""
        # Setup
        from ..database.transaction_manager import TransactionStatus, OperationType
        
        transaction_result = TransactionResult(
            success=True,
            transaction_id="txn_123",
            status=TransactionStatus.COMMITTED,
            operations=[
                Mock(
                    id="op_1",
                    operation_type=OperationType.DELETE,
                    description="Delete video record",
                    target_table="videos",
                    target_id=1,
                    success=True,
                    affected_rows=1,
                    executed_at=datetime.now(),
                    error_message=None
                )
            ],
            savepoints=[
                Mock(
                    name="sp_1",
                    created_at=datetime.now(),
                    operations_count=0,
                    description="before_deletion"
                )
            ],
            start_time=datetime.now(),
            end_time=datetime.now(),
            execution_time_seconds=0.5
        )
        
        mock_history_service.transactional_delete_video_by_id.return_value = transaction_result
        
        with patch('..api.history.get_history_service', return_value=mock_history_service):
            # Execute
            response = client.delete("/videos/1/transactional", json={
                "create_savepoints": True,
                "audit_user": "test_user"
            })
            
            # Verify
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["transaction_id"] == "txn_123"
            assert data["status"] == "committed"
            assert len(data["operations"]) == 1
            assert len(data["savepoints"]) == 1
    
    def test_transactional_batch_delete_videos_success(self, client, mock_history_service):
        """Test successful transactional batch video deletion."""
        # Setup
        from ..database.transaction_manager import TransactionStatus, OperationType
        
        transaction_result = TransactionResult(
            success=True,
            transaction_id="txn_batch_123",
            status=TransactionStatus.COMMITTED,
            operations=[
                Mock(
                    id="op_1",
                    operation_type=OperationType.BATCH_DELETE,
                    description="Delete video 1",
                    target_table="videos",
                    target_id=1,
                    target_ids=None,
                    success=True,
                    affected_rows=1,
                    executed_at=datetime.now(),
                    error_message=None,
                    parameters={}
                ),
                Mock(
                    id="op_2", 
                    operation_type=OperationType.BATCH_DELETE,
                    description="Delete video 2",
                    target_table="videos",
                    target_id=2,
                    target_ids=None,
                    success=True,
                    affected_rows=1,
                    executed_at=datetime.now(),
                    error_message=None,
                    parameters={}
                )
            ],
            savepoints=[],
            start_time=datetime.now(),
            end_time=datetime.now(),
            execution_time_seconds=1.0
        )
        
        mock_history_service.transactional_batch_delete_videos.return_value = transaction_result
        
        with patch('..api.history.get_history_service', return_value=mock_history_service):
            # Execute
            response = client.post("/videos/batch-delete-transactional", json={
                "video_ids": [1, 2],
                "create_savepoints": True,
                "audit_user": "test_user"
            })
            
            # Verify
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["transaction_id"] == "txn_batch_123"
            assert len(data["operations"]) == 2
    
    def test_transactional_batch_delete_videos_too_many(self, client, mock_history_service):
        """Test transactional batch deletion with too many videos."""
        with patch('..api.history.get_history_service', return_value=mock_history_service):
            # Execute
            response = client.post("/videos/batch-delete-transactional", json={
                "video_ids": list(range(51)),  # 51 videos
                "create_savepoints": True
            })
            
            # Verify
            assert response.status_code == 400
            assert "limited to 50 videos" in response.json()["detail"]
    
    def test_test_transaction_rollback_success(self, client, mock_history_service):
        """Test successful transaction rollback test."""
        # Setup
        from ..database.transaction_manager import TransactionStatus, OperationType
        
        transaction_result = TransactionResult(
            success=True,
            transaction_id="txn_test_123",
            status=TransactionStatus.COMMITTED,
            operations=[
                Mock(
                    id="op_1",
                    operation_type=OperationType.UPDATE,
                    description="Update video title to test rollback",
                    target_table="videos",
                    target_id=1,
                    success=True,
                    executed_at=datetime.now(),
                    error_message=None,
                    parameters={"test_result": "success"}
                ),
                Mock(
                    id="op_2",
                    operation_type=OperationType.UPDATE,
                    description="Rollback test completed successfully",
                    target_table="test",
                    target_id=1,
                    success=True,
                    executed_at=datetime.now(),
                    error_message=None,
                    parameters={"test_result": "success"}
                )
            ],
            savepoints=[
                Mock(
                    name="sp_test",
                    created_at=datetime.now(),
                    operations_count=0,
                    description="test_rollback_point"
                )
            ],
            start_time=datetime.now(),
            end_time=datetime.now(),
            execution_time_seconds=0.1
        )
        
        mock_history_service.test_transaction_rollback.return_value = transaction_result
        
        with patch('..api.history.get_history_service', return_value=mock_history_service):
            # Execute
            response = client.post("/videos/1/test-rollback")
            
            # Verify
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["transaction_id"] == "txn_test_123"
            assert len(data["operations"]) == 2
            assert len(data["savepoints"]) == 1


class TestErrorHandling:
    """Test suite for API error handling."""
    
    def test_history_service_error_handling(self, client, mock_history_service):
        """Test handling of HistoryServiceError."""
        # Setup
        mock_history_service.get_video_deletion_info.side_effect = HistoryServiceError("Database error")
        
        with patch('..api.history.get_history_service', return_value=mock_history_service):
            # Execute
            response = client.get("/videos/1/deletion-info")
            
            # Verify
            assert response.status_code == 500
            assert "Database error" in response.json()["detail"]
    
    def test_reprocessing_service_error_handling(self, client, mock_reprocessing_service):
        """Test handling of ReprocessingServiceError."""
        # Setup
        mock_reprocessing_service.validate_reprocessing_request.side_effect = ReprocessingServiceError("Validation error")
        
        with patch('..api.history.get_reprocessing_service', return_value=mock_reprocessing_service):
            # Execute
            response = client.get("/videos/1/validate-reprocessing")
            
            # Verify
            assert response.status_code == 500
            assert "Validation error" in response.json()["detail"]
    
    def test_unexpected_error_handling(self, client, mock_history_service):
        """Test handling of unexpected errors."""
        # Setup
        mock_history_service.get_video_deletion_info.side_effect = Exception("Unexpected error")
        
        with patch('..api.history.get_history_service', return_value=mock_history_service):
            # Execute
            response = client.get("/videos/1/deletion-info")
            
            # Verify
            assert response.status_code == 500
            assert "Internal server error" in response.json()["detail"]


class TestHealthEndpoint:
    """Test suite for health check endpoint."""
    
    def test_health_check_success(self, client):
        """Test successful health check."""
        # Execute
        response = client.get("/health")
        
        # Verify
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "history_api"


@pytest.mark.integration
class TestIntegrationEndpoints:
    """Integration tests for complete API workflows."""
    
    def test_complete_deletion_workflow(self, client, mock_history_service):
        """Test complete video deletion workflow through API."""
        # Setup deletion info
        deletion_info = {
            "video": {"id": 1, "video_id": "test_video", "title": "Test Video"},
            "related_data_counts": {"transcripts": 1, "summaries": 1},
            "total_related_records": 2
        }
        
        # Setup validation
        validation = CascadeDeleteValidation(
            can_delete=True,
            video_exists=True,
            related_counts={"transcripts": 1, "summaries": 1},
            potential_issues=[],
            total_related_records=2
        )
        
        # Setup deletion result
        delete_result = CascadeDeleteResult(
            success=True,
            video_id=1,
            deleted_counts={"transcripts": 1, "summaries": 1, "videos": 1},
            total_deleted=3
        )
        
        # Setup integrity check
        integrity_result = {
            "video_exists": False,
            "has_orphaned_records": False,
            "orphaned_records": {},
            "integrity_check_passed": True
        }
        
        mock_history_service.get_video_deletion_info.return_value = deletion_info
        mock_history_service.validate_video_deletion.return_value = validation
        mock_history_service.enhanced_delete_video_by_id.return_value = delete_result
        mock_history_service.verify_cascade_delete_integrity.return_value = integrity_result
        
        with patch('..api.history.get_history_service', return_value=mock_history_service):
            # Execute workflow
            
            # 1. Get deletion info
            response = client.get("/videos/1/deletion-info")
            assert response.status_code == 200
            
            # 2. Validate deletion
            response = client.get("/videos/1/validate-deletion")
            assert response.status_code == 200
            assert response.json()["can_delete"] is True
            
            # 3. Delete video
            response = client.delete("/videos/1", json={"force": False})
            assert response.status_code == 200
            assert response.json()["success"] is True
            
            # 4. Verify integrity
            response = client.get("/videos/1/integrity-check")
            assert response.status_code == 200
            assert response.json()["integrity_check_passed"] is True
    
    def test_complete_reprocessing_workflow(self, client, mock_reprocessing_service):
        """Test complete video reprocessing workflow through API."""
        # Setup validation
        validation = ReprocessingValidation(
            can_reprocess=True,
            video_exists=True,
            current_status="completed",
            existing_components={"transcripts": 1, "summaries": 1},
            potential_issues=[],
            recommendations=[]
        )
        
        # Setup reprocessing result
        reprocess_result = ReprocessingResult(
            success=True,
            video_id=1,
            mode=ReprocessingMode.FULL,
            status=ReprocessingStatus.PENDING,
            message="Reprocessing initiated",
            cleared_components=["transcripts", "summaries"]
        )
        
        # Setup status info
        status_info = {
            "video_id": 1,
            "video_title": "Test Video",
            "video_youtube_id": "test_video",
            "processing_metadata_id": 123,
            "status": "completed",
            "workflow_params": {"reprocessing_mode": "full"},
            "error_info": None,
            "created_at": datetime.now(),
            "is_reprocessing": True
        }
        
        mock_reprocessing_service.validate_reprocessing_request.return_value = validation
        mock_reprocessing_service.initiate_reprocessing.return_value = reprocess_result
        mock_reprocessing_service.get_reprocessing_status.return_value = status_info
        
        with patch('..api.history.get_reprocessing_service', return_value=mock_reprocessing_service):
            # Execute workflow
            
            # 1. Validate reprocessing
            response = client.get("/videos/1/validate-reprocessing")
            assert response.status_code == 200
            assert response.json()["can_reprocess"] is True
            
            # 2. Initiate reprocessing
            response = client.post("/videos/1/reprocess", json={
                "mode": "full",
                "force": False,
                "clear_cache": True
            })
            assert response.status_code == 200
            assert response.json()["success"] is True
            
            # 3. Check status
            response = client.get("/videos/1/reprocessing-status")
            assert response.status_code == 200
            assert response.json()["is_reprocessing"] is True