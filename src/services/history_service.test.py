"""
Comprehensive tests for HistoryService deletion and reprocessing functionality.

This test suite covers all aspects of the enhanced history service including:
- Video deletion operations
- Cascade delete functionality  
- Reprocessing operations
- Transaction management
- Error handling
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError, IntegrityError

from ..services.history_service import HistoryService, HistoryServiceError
from ..services.reprocessing_service import ReprocessingService, ReprocessingServiceError, ReprocessingMode
from ..database.models import Video, Transcript, Summary, Keyword, TimestampedSegment, ProcessingMetadata
from ..database.cascade_delete import CascadeDeleteManager, CascadeDeleteResult, CascadeDeleteValidation
from ..database.transaction_manager import TransactionManager, TransactionResult, TransactionStatus
from ..database.exceptions import DatabaseError, DatabaseQueryError


class TestHistoryServiceDeletion:
    """Test suite for HistoryService deletion functionality."""
    
    @pytest.fixture
    def mock_session(self):
        """Create mock database session."""
        return Mock(spec=Session)
    
    @pytest.fixture
    def history_service(self, mock_session):
        """Create HistoryService instance with mock session."""
        return HistoryService(session=mock_session)
    
    @pytest.fixture
    def sample_video(self):
        """Create sample video for testing."""
        video = Mock(spec=Video)
        video.id = 1
        video.video_id = "test_video_123"
        video.title = "Test Video"
        video.url = "https://youtube.com/watch?v=test_video_123"
        video.created_at = datetime.now()
        video.updated_at = datetime.now()
        video.transcripts = []
        video.summaries = []
        video.keywords = []
        video.timestamped_segments = []
        video.processing_metadata = []
        return video
    
    def test_delete_video_by_id_success(self, history_service, mock_session, sample_video):
        """Test successful video deletion by ID."""
        # Setup
        mock_session.get.return_value = sample_video
        mock_session.delete.return_value = None
        mock_session.commit.return_value = None
        
        # Execute
        result = history_service.delete_video_by_id(1)
        
        # Verify
        assert result is True
        mock_session.get.assert_called_once_with(Video, 1)
        mock_session.delete.assert_called_once_with(sample_video)
        mock_session.commit.assert_called_once()
    
    def test_delete_video_by_id_not_found(self, history_service, mock_session):
        """Test video deletion when video doesn't exist."""
        # Setup
        mock_session.get.return_value = None
        
        # Execute
        result = history_service.delete_video_by_id(999)
        
        # Verify
        assert result is False
        mock_session.get.assert_called_once_with(Video, 999)
        mock_session.delete.assert_not_called()
        mock_session.commit.assert_not_called()
    
    def test_delete_video_by_id_database_error(self, history_service, mock_session):
        """Test video deletion with database error."""
        # Setup
        mock_session.get.side_effect = SQLAlchemyError("Database error")
        
        # Execute and verify
        with pytest.raises(HistoryServiceError):
            history_service.delete_video_by_id(1)
    
    def test_delete_video_by_video_id_success(self, history_service, mock_session, sample_video):
        """Test successful video deletion by YouTube video ID."""
        # Setup
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = sample_video
        mock_session.execute.return_value = mock_result
        
        # Mock the delete_video_by_id method
        with patch.object(history_service, 'delete_video_by_id', return_value=True) as mock_delete:
            # Execute
            result = history_service.delete_video_by_video_id("test_video_123")
            
            # Verify
            assert result is True
            mock_delete.assert_called_once_with(sample_video.id)
    
    def test_delete_multiple_videos_success(self, history_service, mock_session):
        """Test successful multiple video deletion."""
        # Setup
        video1 = Mock(spec=Video)
        video1.id = 1
        video1.video_id = "test1"
        video1.title = "Test Video 1"
        video1.url = "https://youtube.com/watch?v=test1"
        
        video2 = Mock(spec=Video)
        video2.id = 2
        video2.video_id = "test2"
        video2.title = "Test Video 2"
        video2.url = "https://youtube.com/watch?v=test2"
        
        mock_session.get.side_effect = [video1, video2]
        mock_session.delete.return_value = None
        mock_session.commit.return_value = None
        
        # Execute
        result = history_service.delete_multiple_videos([1, 2])
        
        # Verify
        assert result["deleted_count"] == 2
        assert result["failed_count"] == 0
        assert result["not_found_count"] == 0
        assert len(result["deleted_videos"]) == 2
        assert len(result["failed_videos"]) == 0
    
    def test_delete_multiple_videos_partial_failure(self, history_service, mock_session):
        """Test multiple video deletion with partial failure."""
        # Setup
        video1 = Mock(spec=Video)
        video1.id = 1
        video1.video_id = "test1"
        video1.title = "Test Video 1"
        video1.url = "https://youtube.com/watch?v=test1"
        
        mock_session.get.side_effect = [video1, None]  # Second video not found
        mock_session.delete.return_value = None
        mock_session.commit.return_value = None
        
        # Execute
        result = history_service.delete_multiple_videos([1, 999])
        
        # Verify
        assert result["deleted_count"] == 1
        assert result["failed_count"] == 0
        assert result["not_found_count"] == 1
        assert len(result["deleted_videos"]) == 1
        assert len(result["failed_videos"]) == 1
    
    def test_get_video_deletion_info_success(self, history_service, mock_session, sample_video):
        """Test getting video deletion info."""
        # Setup
        sample_video.transcripts = [Mock(), Mock()]  # 2 transcripts
        sample_video.summaries = [Mock()]  # 1 summary
        sample_video.keywords = [Mock()]  # 1 keyword
        sample_video.timestamped_segments = []  # 0 segments
        sample_video.processing_metadata = [Mock(), Mock(), Mock()]  # 3 metadata
        
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = sample_video
        mock_session.execute.return_value = mock_result
        
        # Execute
        result = history_service.get_video_deletion_info(1)
        
        # Verify
        assert result is not None
        assert result["video"]["id"] == 1
        assert result["related_data_counts"]["transcripts"] == 2
        assert result["related_data_counts"]["summaries"] == 1
        assert result["related_data_counts"]["keywords"] == 1
        assert result["related_data_counts"]["timestamped_segments"] == 0
        assert result["related_data_counts"]["processing_metadata"] == 3
        assert result["total_related_records"] == 7  # 2+1+1+0+3
    
    def test_get_video_deletion_info_not_found(self, history_service, mock_session):
        """Test getting deletion info for non-existent video."""
        # Setup
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result
        
        # Execute
        result = history_service.get_video_deletion_info(999)
        
        # Verify
        assert result is None


class TestHistoryServiceEnhancedDeletion:
    """Test suite for enhanced deletion functionality."""
    
    @pytest.fixture
    def mock_session(self):
        """Create mock database session."""
        return Mock(spec=Session)
    
    @pytest.fixture
    def history_service(self, mock_session):
        """Create HistoryService instance with mock session."""
        return HistoryService(session=mock_session)
    
    @pytest.fixture
    def mock_cascade_manager(self):
        """Create mock cascade delete manager."""
        return Mock(spec=CascadeDeleteManager)
    
    def test_validate_video_deletion_success(self, history_service, mock_session):
        """Test video deletion validation."""
        # Setup
        mock_validation = CascadeDeleteValidation(
            can_delete=True,
            video_exists=True,
            related_counts={"transcripts": 1, "summaries": 1},
            potential_issues=[],
            total_related_records=2
        )
        
        with patch('..services.history_service.validate_video_deletion', return_value=mock_validation):
            # Execute
            result = history_service.validate_video_deletion(1)
            
            # Verify
            assert result.can_delete is True
            assert result.video_exists is True
            assert result.total_related_records == 2
    
    def test_enhanced_delete_video_by_id_success(self, history_service, mock_session):
        """Test enhanced video deletion."""
        # Setup
        mock_result = CascadeDeleteResult(
            success=True,
            video_id=1,
            deleted_counts={"transcripts": 1, "summaries": 1, "videos": 1},
            total_deleted=3,
            execution_time_ms=250.5
        )
        
        with patch('..services.history_service.execute_enhanced_cascade_delete', return_value=mock_result):
            # Execute
            result = history_service.enhanced_delete_video_by_id(1)
            
            # Verify
            assert result.success is True
            assert result.video_id == 1
            assert result.total_deleted == 3
    
    def test_enhanced_batch_delete_videos_success(self, history_service, mock_session):
        """Test enhanced batch video deletion."""
        # Setup
        mock_results = [
            CascadeDeleteResult(
                success=True,
                video_id=1,
                deleted_counts={"videos": 1},
                total_deleted=1
            ),
            CascadeDeleteResult(
                success=True,
                video_id=2,
                deleted_counts={"videos": 1},
                total_deleted=1
            )
        ]
        
        with patch.object(history_service, '_get_session', return_value=mock_session):
            with patch('..services.history_service.create_cascade_delete_manager') as mock_create_manager:
                mock_manager = Mock()
                mock_manager.batch_cascade_delete.return_value = mock_results
                mock_create_manager.return_value = mock_manager
                
                # Execute
                result = history_service.enhanced_batch_delete_videos([1, 2])
                
                # Verify
                assert len(result) == 2
                assert all(r.success for r in result)


class TestReprocessingService:
    """Test suite for ReprocessingService functionality."""
    
    @pytest.fixture
    def mock_session(self):
        """Create mock database session."""
        return Mock(spec=Session)
    
    @pytest.fixture
    def reprocessing_service(self, mock_session):
        """Create ReprocessingService instance with mock session."""
        return ReprocessingService(session=mock_session)
    
    @pytest.fixture
    def sample_video(self):
        """Create sample video for testing."""
        video = Mock(spec=Video)
        video.id = 1
        video.video_id = "test_video_123"
        video.title = "Test Video"
        return video
    
    def test_validate_reprocessing_request_success(self, reprocessing_service, mock_session, sample_video):
        """Test reprocessing request validation."""
        # Setup
        from ..services.reprocessing_service import ReprocessingRequest
        
        mock_session.get.return_value = sample_video
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = None  # No active processing
        mock_session.execute.return_value = mock_result
        
        request = ReprocessingRequest(
            video_id=1,
            mode=ReprocessingMode.FULL,
            force=False
        )
        
        # Execute
        result = reprocessing_service.validate_reprocessing_request(request)
        
        # Verify
        assert result.can_reprocess is True
        assert result.video_exists is True
        assert len(result.potential_issues) == 0
    
    def test_validate_reprocessing_request_video_not_found(self, reprocessing_service, mock_session):
        """Test reprocessing validation when video doesn't exist."""
        # Setup
        from ..services.reprocessing_service import ReprocessingRequest
        
        mock_session.get.return_value = None
        
        request = ReprocessingRequest(
            video_id=999,
            mode=ReprocessingMode.FULL,
            force=False
        )
        
        # Execute
        result = reprocessing_service.validate_reprocessing_request(request)
        
        # Verify
        assert result.can_reprocess is False
        assert result.video_exists is False
        assert "Video not found" in result.potential_issues
    
    def test_clear_video_cache_full_mode(self, reprocessing_service, mock_session):
        """Test video cache clearing in full mode."""
        # Setup
        mock_result = Mock()
        mock_result.rowcount = 1
        mock_session.execute.return_value = mock_result
        mock_session.commit.return_value = None
        
        # Execute
        result = reprocessing_service.clear_video_cache(1, ReprocessingMode.FULL)
        
        # Verify
        assert 'transcripts' in result
        assert 'summaries' in result
        assert 'keywords' in result
        assert 'timestamped_segments' in result
        assert len(result) == 4  # All components cleared
    
    def test_clear_video_cache_transcript_only(self, reprocessing_service, mock_session):
        """Test video cache clearing in transcript-only mode."""
        # Setup
        mock_result = Mock()
        mock_result.rowcount = 1
        mock_session.execute.return_value = mock_result
        mock_session.commit.return_value = None
        
        # Execute
        result = reprocessing_service.clear_video_cache(1, ReprocessingMode.TRANSCRIPT_ONLY)
        
        # Verify
        assert 'transcripts' in result
        assert len(result) == 1  # Only transcripts cleared
    
    def test_initiate_reprocessing_success(self, reprocessing_service, mock_session, sample_video):
        """Test successful reprocessing initiation."""
        # Setup
        from ..services.reprocessing_service import ReprocessingRequest
        
        mock_session.get.return_value = sample_video
        mock_session.add.return_value = None
        mock_session.commit.return_value = None
        mock_session.refresh.return_value = None
        
        # Mock validation
        with patch.object(reprocessing_service, 'validate_reprocessing_request') as mock_validate:
            mock_validate.return_value = Mock(can_reprocess=True)
            
            # Mock cache clearing
            with patch.object(reprocessing_service, 'clear_video_cache') as mock_clear:
                mock_clear.return_value = ['transcripts', 'summaries']
                
                request = ReprocessingRequest(
                    video_id=1,
                    mode=ReprocessingMode.FULL,
                    clear_cache=True,
                    force=False
                )
                
                # Execute
                result = reprocessing_service.initiate_reprocessing(request)
                
                # Verify
                assert result.success is True
                assert result.video_id == 1
                assert result.mode == ReprocessingMode.FULL
                assert 'transcripts' in result.cleared_components
                assert 'summaries' in result.cleared_components
    
    def test_get_reprocessing_status_success(self, reprocessing_service, mock_session, sample_video):
        """Test getting reprocessing status."""
        # Setup
        mock_metadata = Mock(spec=ProcessingMetadata)
        mock_metadata.id = 1
        mock_metadata.status = "processing"
        mock_metadata.workflow_params = {"reprocessing_mode": "full"}
        mock_metadata.error_info = None
        mock_metadata.created_at = datetime.now()
        
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = mock_metadata
        mock_session.execute.return_value = mock_result
        mock_session.get.return_value = sample_video
        
        # Execute
        result = reprocessing_service.get_reprocessing_status(1)
        
        # Verify
        assert result is not None
        assert result["video_id"] == 1
        assert result["status"] == "processing"
        assert result["is_reprocessing"] is True
    
    def test_cancel_reprocessing_success(self, reprocessing_service, mock_session):
        """Test successful reprocessing cancellation."""
        # Setup
        mock_metadata = Mock(spec=ProcessingMetadata)
        mock_metadata.status = "processing"
        
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = mock_metadata
        mock_session.execute.return_value = mock_result
        mock_session.commit.return_value = None
        
        # Execute
        result = reprocessing_service.cancel_reprocessing(1, "User requested")
        
        # Verify
        assert result is True
        assert mock_metadata.status == "cancelled"
        assert "User requested" in mock_metadata.error_info


class TestTransactionManager:
    """Test suite for TransactionManager functionality."""
    
    @pytest.fixture
    def mock_session(self):
        """Create mock database session."""
        return Mock(spec=Session)
    
    @pytest.fixture
    def transaction_manager(self, mock_session):
        """Create TransactionManager instance with mock session."""
        return TransactionManager(session=mock_session)
    
    def test_create_savepoint_success(self, transaction_manager, mock_session):
        """Test successful savepoint creation."""
        # Setup
        mock_session.execute.return_value = None
        
        # Execute
        savepoint_name = transaction_manager.create_savepoint("test_savepoint")
        
        # Verify
        assert savepoint_name.startswith("sp_")
        assert len(transaction_manager.savepoints) == 1
        assert transaction_manager.savepoints[0].name == savepoint_name
    
    def test_rollback_to_savepoint_success(self, transaction_manager, mock_session):
        """Test successful rollback to savepoint."""
        # Setup
        mock_session.execute.return_value = None
        savepoint_name = transaction_manager.create_savepoint("test_savepoint")
        
        # Add some operations after savepoint
        transaction_manager.operations.append(Mock())
        transaction_manager.operations.append(Mock())
        
        # Execute
        result = transaction_manager.rollback_to_savepoint(savepoint_name, "Test rollback")
        
        # Verify
        assert result is True
        # Operations should be truncated to savepoint
        assert len(transaction_manager.operations) == 0
    
    def test_execute_operation_success(self, transaction_manager, mock_session):
        """Test successful operation execution."""
        # Setup
        from ..database.transaction_manager import OperationType
        
        def test_operation():
            return Mock(rowcount=1)
        
        # Execute
        result = transaction_manager.execute_operation(
            OperationType.DELETE,
            "Test operation",
            "test_table",
            test_operation,
            target_id=1
        )
        
        # Verify
        assert result.success is True
        assert result.operation_type == OperationType.DELETE
        assert result.description == "Test operation"
        assert result.target_table == "test_table"
        assert result.target_id == 1
        assert result.affected_rows == 1
    
    def test_execute_operation_failure(self, transaction_manager, mock_session):
        """Test operation execution failure."""
        # Setup
        from ..database.transaction_manager import OperationType
        
        def failing_operation():
            raise Exception("Operation failed")
        
        # Execute and verify
        with pytest.raises(Exception):
            transaction_manager.execute_operation(
                OperationType.DELETE,
                "Failing operation",
                "test_table",
                failing_operation
            )
        
        # Verify operation was recorded as failed
        assert len(transaction_manager.operations) == 1
        assert transaction_manager.operations[0].success is False
        assert transaction_manager.operations[0].error_message == "Operation failed"
    
    def test_commit_transaction_success(self, transaction_manager, mock_session):
        """Test successful transaction commit."""
        # Setup
        mock_session.commit.return_value = None
        
        # Execute
        result = transaction_manager.commit_transaction()
        
        # Verify
        assert result.success is True
        assert result.status == TransactionStatus.COMMITTED
        mock_session.commit.assert_called_once()
    
    def test_rollback_transaction_success(self, transaction_manager, mock_session):
        """Test successful transaction rollback."""
        # Setup
        mock_session.rollback.return_value = None
        
        # Execute
        result = transaction_manager.rollback_transaction("Test rollback")
        
        # Verify
        assert result.success is False
        assert result.status == TransactionStatus.ROLLED_BACK
        assert result.rollback_reason == "Test rollback"
        mock_session.rollback.assert_called_once()


class TestHistoryServiceTransactionalDeletion:
    """Test suite for transactional deletion functionality."""
    
    @pytest.fixture
    def mock_session(self):
        """Create mock database session."""
        return Mock(spec=Session)
    
    @pytest.fixture
    def history_service(self, mock_session):
        """Create HistoryService instance with mock session."""
        return HistoryService(session=mock_session)
    
    @pytest.fixture
    def sample_video(self):
        """Create sample video for testing."""
        video = Mock(spec=Video)
        video.id = 1
        video.video_id = "test_video_123"
        video.title = "Test Video"
        video.url = "https://youtube.com/watch?v=test_video_123"
        video.created_at = datetime.now()
        video.updated_at = datetime.now()
        return video
    
    def test_transactional_delete_video_by_id_success(self, history_service, mock_session, sample_video):
        """Test successful transactional video deletion."""
        # Setup
        mock_session.get.return_value = sample_video
        mock_session.execute.return_value = Mock(scalar=Mock(return_value=1))
        mock_session.delete.return_value = None
        mock_session.commit.return_value = None
        
        with patch('..services.history_service.managed_transaction') as mock_transaction:
            mock_txn = Mock()
            mock_txn.create_savepoint.return_value = "test_savepoint"
            mock_txn.execute_operation.return_value = Mock(affected_rows=1)
            mock_txn.commit_transaction.return_value = Mock(success=True)
            mock_transaction.return_value.__enter__.return_value = mock_txn
            
            # Execute
            result = history_service.transactional_delete_video_by_id(1)
            
            # Verify
            assert result.success is True
    
    def test_transactional_batch_delete_videos_success(self, history_service, mock_session):
        """Test successful transactional batch video deletion."""
        # Setup
        video1 = Mock(spec=Video)
        video1.id = 1
        video1.video_id = "test1"
        video1.title = "Test Video 1"
        
        video2 = Mock(spec=Video)
        video2.id = 2
        video2.video_id = "test2"
        video2.title = "Test Video 2"
        
        mock_session.get.side_effect = [video1, video2]
        mock_session.execute.return_value = Mock(scalar=Mock(return_value=0))
        mock_session.delete.return_value = None
        mock_session.commit.return_value = None
        mock_session.flush.return_value = None
        
        with patch('..services.history_service.managed_transaction') as mock_transaction:
            mock_txn = Mock()
            mock_txn.create_savepoint.return_value = "test_savepoint"
            mock_txn.execute_operation.return_value = Mock(affected_rows=1)
            mock_txn.commit_transaction.return_value = Mock(success=True)
            mock_transaction.return_value.__enter__.return_value = mock_txn
            
            # Execute
            result = history_service.transactional_batch_delete_videos([1, 2])
            
            # Verify
            assert result.success is True
    
    def test_test_transaction_rollback_success(self, history_service, mock_session, sample_video):
        """Test transaction rollback test functionality."""
        # Setup
        sample_video.title = "Original Title"
        mock_session.get.return_value = sample_video
        mock_session.flush.return_value = None
        
        with patch('..services.history_service.managed_transaction') as mock_transaction:
            mock_txn = Mock()
            mock_txn.create_savepoint.return_value = "test_savepoint"
            mock_txn.execute_operation.return_value = Mock(success=True)
            mock_txn.rollback_to_savepoint.return_value = True
            mock_txn.commit_transaction.return_value = Mock(success=True)
            mock_transaction.return_value.__enter__.return_value = mock_txn
            
            # Execute
            result = history_service.test_transaction_rollback(1)
            
            # Verify
            assert result.success is True
            mock_txn.rollback_to_savepoint.assert_called_once()


class TestErrorHandling:
    """Test suite for error handling across all services."""
    
    @pytest.fixture
    def mock_session(self):
        """Create mock database session."""
        return Mock(spec=Session)
    
    def test_database_error_handling_in_history_service(self, mock_session):
        """Test database error handling in HistoryService."""
        # Setup
        history_service = HistoryService(session=mock_session)
        mock_session.get.side_effect = SQLAlchemyError("Database connection failed")
        
        # Execute and verify
        with pytest.raises(HistoryServiceError):
            history_service.delete_video_by_id(1)
    
    def test_integrity_error_handling_in_reprocessing_service(self, mock_session):
        """Test integrity error handling in ReprocessingService."""
        # Setup
        reprocessing_service = ReprocessingService(session=mock_session)
        mock_session.get.side_effect = IntegrityError("statement", "params", "orig")
        
        # Execute and verify
        with pytest.raises(ReprocessingServiceError):
            from ..services.reprocessing_service import ReprocessingRequest
            request = ReprocessingRequest(video_id=1, mode=ReprocessingMode.FULL)
            reprocessing_service.validate_reprocessing_request(request)
    
    def test_transaction_rollback_on_error(self, mock_session):
        """Test automatic transaction rollback on error."""
        # Setup
        from ..database.transaction_manager import TransactionManager
        
        manager = TransactionManager(session=mock_session)
        mock_session.commit.side_effect = SQLAlchemyError("Commit failed")
        mock_session.rollback.return_value = None
        
        # Execute and verify
        with pytest.raises(DatabaseError):
            manager.commit_transaction()
        
        # Verify rollback was called
        mock_session.rollback.assert_called_once()
        assert manager.status == TransactionStatus.ROLLED_BACK


@pytest.mark.integration
class TestIntegration:
    """Integration tests for complete workflows."""
    
    @pytest.fixture
    def mock_session(self):
        """Create mock database session."""
        return Mock(spec=Session)
    
    def test_complete_deletion_workflow(self, mock_session):
        """Test complete video deletion workflow."""
        # Setup services
        history_service = HistoryService(session=mock_session)
        
        # Setup video
        video = Mock(spec=Video)
        video.id = 1
        video.video_id = "test_video"
        video.title = "Test Video"
        video.transcripts = [Mock()]
        video.summaries = [Mock()]
        video.keywords = []
        video.timestamped_segments = []
        video.processing_metadata = [Mock()]
        
        mock_session.get.return_value = video
        mock_session.delete.return_value = None
        mock_session.commit.return_value = None
        
        # Execute deletion workflow
        deletion_info = history_service.get_video_deletion_info(1)
        assert deletion_info is not None
        
        result = history_service.delete_video_by_id(1)
        assert result is True
    
    def test_complete_reprocessing_workflow(self, mock_session):
        """Test complete video reprocessing workflow."""
        # Setup services
        reprocessing_service = ReprocessingService(session=mock_session)
        
        # Setup video and metadata
        video = Mock(spec=Video)
        video.id = 1
        video.video_id = "test_video"
        video.title = "Test Video"
        
        mock_session.get.return_value = video
        mock_session.add.return_value = None
        mock_session.commit.return_value = None
        mock_session.refresh.return_value = None
        
        # Mock query results
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = None
        mock_result.rowcount = 1
        mock_session.execute.return_value = mock_result
        
        # Execute reprocessing workflow
        from ..services.reprocessing_service import ReprocessingRequest
        
        request = ReprocessingRequest(
            video_id=1,
            mode=ReprocessingMode.FULL,
            clear_cache=True,
            force=False
        )
        
        # Validate request
        validation = reprocessing_service.validate_reprocessing_request(request)
        assert validation.video_exists is True
        
        # Clear cache
        cleared = reprocessing_service.clear_video_cache(1, ReprocessingMode.FULL)
        assert len(cleared) > 0
        
        # Initiate reprocessing
        result = reprocessing_service.initiate_reprocessing(request)
        assert result.success is True