"""
Tests for status tracking integration.

This module tests the integration of status tracking with workflows and nodes.
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from .status_integration import StatusTrackingMixin, WorkflowStatusManager
from ..database.status_models import ProcessingStatusType, ProcessingPriority


class MockNode(StatusTrackingMixin):
    """Mock node for testing status tracking mixin."""
    
    def __init__(self, name: str = "TestNode", *args, **kwargs):
        self.name = name
        super().__init__(*args, **kwargs)


class TestStatusTrackingMixin:
    """Test cases for StatusTrackingMixin."""
    
    @patch('src.services.status_integration.get_db_session')
    def test_initialization_success(self, mock_get_db_session):
        """Test successful initialization of status tracking."""
        mock_session = Mock()
        mock_get_db_session.return_value = mock_session
        
        node = MockNode("TestNode")
        
        assert node._status_tracking_enabled is True
        assert node._status_service is not None
        assert node._status_updater is not None
        assert node._worker_id.startswith("MockNode_")
    
    @patch('src.services.status_integration.get_db_session')
    def test_initialization_failure(self, mock_get_db_session):
        """Test initialization failure handling."""
        mock_get_db_session.side_effect = Exception("Database connection failed")
        
        node = MockNode("TestNode")
        
        assert node._status_tracking_enabled is False
        assert node._status_service is None
        assert node._status_updater is None
    
    @patch('src.services.status_integration.get_db_session')
    def test_create_processing_status(self, mock_get_db_session):
        """Test creating a processing status."""
        mock_session = Mock()
        mock_get_db_session.return_value = mock_session
        
        mock_status = Mock()
        mock_status.status_id = "test_status_123"
        
        node = MockNode("TestNode")
        node._status_service.create_processing_status.return_value = mock_status
        
        status_id = node._create_processing_status(
            video_id=1,
            total_steps=3,
            tags=["test"]
        )
        
        assert status_id == "test_status_123"
        assert node._current_status_id == "test_status_123"
        
        node._status_service.create_processing_status.assert_called_once()
    
    @patch('src.services.status_integration.get_db_session')
    def test_update_status_immediate(self, mock_get_db_session):
        """Test immediate status update."""
        mock_session = Mock()
        mock_get_db_session.return_value = mock_session
        
        node = MockNode("TestNode")
        node._current_status_id = "test_status_123"
        
        result = node._update_status(
            new_status=ProcessingStatusType.STARTING,
            progress_percentage=25.0,
            current_step="Test step",
            immediate=True
        )
        
        assert result is True
        node._status_service.update_status.assert_called_once()
    
    @patch('src.services.status_integration.get_db_session')
    def test_update_status_queued(self, mock_get_db_session):
        """Test queued status update."""
        mock_session = Mock()
        mock_get_db_session.return_value = mock_session
        
        node = MockNode("TestNode")
        node._current_status_id = "test_status_123"
        
        result = node._update_status(
            new_status=ProcessingStatusType.STARTING,
            progress_percentage=25.0,
            current_step="Test step",
            immediate=False
        )
        
        assert result is True
        node._status_updater.queue_status_update.assert_called_once()
    
    @patch('src.services.status_integration.get_db_session')
    def test_update_progress(self, mock_get_db_session):
        """Test progress update."""
        mock_session = Mock()
        mock_get_db_session.return_value = mock_session
        
        node = MockNode("TestNode")
        node._current_status_id = "test_status_123"
        
        result = node._update_progress(
            progress_percentage=50.0,
            current_step="Processing data"
        )
        
        assert result is True
        node._status_updater.queue_progress_update.assert_called_once()
    
    @patch('src.services.status_integration.get_db_session')
    def test_record_error(self, mock_get_db_session):
        """Test error recording."""
        mock_session = Mock()
        mock_get_db_session.return_value = mock_session
        
        node = MockNode("TestNode")
        node._current_status_id = "test_status_123"
        
        result = node._record_error(
            error_info="Test error occurred",
            should_retry=True
        )
        
        assert result is True
        node._status_service.record_error.assert_called_once()
    
    @patch('src.services.status_integration.get_db_session')
    def test_send_heartbeat(self, mock_get_db_session):
        """Test heartbeat sending."""
        mock_session = Mock()
        mock_get_db_session.return_value = mock_session
        
        node = MockNode("TestNode")
        node._current_status_id = "test_status_123"
        
        result = node._send_heartbeat(
            progress_percentage=75.0,
            current_step="Almost done"
        )
        
        assert result is True
        node._status_service.heartbeat.assert_called_once()
    
    @patch('src.services.status_integration.get_db_session')
    def test_status_tracking_context_success(self, mock_get_db_session):
        """Test status tracking context manager with successful operation."""
        mock_session = Mock()
        mock_get_db_session.return_value = mock_session
        
        mock_status = Mock()
        mock_status.status_id = "test_status_123"
        
        node = MockNode("TestNode")
        node._status_service.create_processing_status.return_value = mock_status
        
        with node._status_tracking_context("test_operation") as status_id:
            assert status_id == "test_status_123"
            # Simulate some work
            time.sleep(0.1)
        
        # Verify status updates were called
        assert node._status_service.update_status.call_count >= 2  # Start and complete
    
    @patch('src.services.status_integration.get_db_session')
    def test_status_tracking_context_error(self, mock_get_db_session):
        """Test status tracking context manager with error."""
        mock_session = Mock()
        mock_get_db_session.return_value = mock_session
        
        mock_status = Mock()
        mock_status.status_id = "test_status_123"
        
        node = MockNode("TestNode")
        node._status_service.create_processing_status.return_value = mock_status
        
        with pytest.raises(ValueError, match="Test error"):
            with node._status_tracking_context("test_operation"):
                raise ValueError("Test error")
        
        # Verify error was recorded
        node._status_service.record_error.assert_called_once()
    
    def test_disabled_status_tracking(self):
        """Test behavior when status tracking is disabled."""
        node = MockNode("TestNode", enable_status_tracking=False)
        
        assert node._status_tracking_enabled is False
        
        # All operations should return False or None when disabled
        assert node._create_processing_status() is None
        assert node._update_status(ProcessingStatusType.STARTING) is False
        assert node._update_progress(50.0) is False
        assert node._record_error("Error") is False
        assert node._send_heartbeat() is False


class TestWorkflowStatusManager:
    """Test cases for WorkflowStatusManager."""
    
    @patch('src.services.status_integration.get_db_session')
    def test_initialization(self, mock_get_db_session):
        """Test WorkflowStatusManager initialization."""
        mock_session = Mock()
        mock_get_db_session.return_value = mock_session
        
        manager = WorkflowStatusManager("TestWorkflow")
        
        assert manager.workflow_name == "TestWorkflow"
        assert manager.workflow_status_id is None
        assert manager.node_status_mapping == {}
        assert manager.current_node_index == 0
        assert manager.total_nodes == 0
    
    @patch('src.services.status_integration.get_db_session')
    def test_start_workflow(self, mock_get_db_session):
        """Test starting workflow status tracking."""
        mock_session = Mock()
        mock_get_db_session.return_value = mock_session
        
        mock_status = Mock()
        mock_status.status_id = "workflow_status_123"
        
        manager = WorkflowStatusManager("TestWorkflow")
        manager.status_service.create_processing_status.return_value = mock_status
        
        status_id = manager.start_workflow(
            video_id=1,
            node_names=["Node1", "Node2", "Node3"]
        )
        
        assert status_id == "workflow_status_123"
        assert manager.workflow_status_id == "workflow_status_123"
        assert manager.total_nodes == 3
        
        manager.status_service.create_processing_status.assert_called_once()
        manager.status_service.update_status.assert_called_once()
    
    @patch('src.services.status_integration.get_db_session')
    def test_start_node(self, mock_get_db_session):
        """Test starting node status tracking."""
        mock_session = Mock()
        mock_get_db_session.return_value = mock_session
        
        mock_status = Mock()
        mock_status.status_id = "node_status_123"
        
        manager = WorkflowStatusManager("TestWorkflow")
        manager.workflow_status_id = "workflow_status_123"
        manager.status_service.create_processing_status.return_value = mock_status
        
        node_status_id = manager.start_node("TestNode", video_id=1)
        
        assert node_status_id == "node_status_123"
        assert manager.node_status_mapping["TestNode"] == "node_status_123"
        
        manager.status_service.create_processing_status.assert_called_once()
        manager.status_service.update_status.assert_called_once()
    
    @patch('src.services.status_integration.get_db_session')
    def test_finish_node_success(self, mock_get_db_session):
        """Test finishing node with success."""
        mock_session = Mock()
        mock_get_db_session.return_value = mock_session
        
        manager = WorkflowStatusManager("TestWorkflow")
        manager.node_status_mapping["TestNode"] = "node_status_123"
        manager.total_nodes = 3
        
        manager.finish_node("TestNode", success=True)
        
        assert manager.current_node_index == 1
        manager.status_service.update_status.assert_called_once()
    
    @patch('src.services.status_integration.get_db_session')
    def test_finish_node_failure(self, mock_get_db_session):
        """Test finishing node with failure."""
        mock_session = Mock()
        mock_get_db_session.return_value = mock_session
        
        manager = WorkflowStatusManager("TestWorkflow")
        manager.workflow_status_id = "workflow_status_123"
        manager.node_status_mapping["TestNode"] = "node_status_123"
        
        manager.finish_node("TestNode", success=False, error_info="Node failed")
        
        manager.status_service.record_error.assert_called()
        manager.status_service.update_status.assert_called()
    
    @patch('src.services.status_integration.get_db_session')
    def test_finish_workflow_success(self, mock_get_db_session):
        """Test finishing workflow with success."""
        mock_session = Mock()
        mock_get_db_session.return_value = mock_session
        
        manager = WorkflowStatusManager("TestWorkflow")
        manager.workflow_status_id = "workflow_status_123"
        manager.total_nodes = 3
        
        manager.finish_workflow(success=True)
        
        manager.status_service.update_status.assert_called_once()
        args, kwargs = manager.status_service.update_status.call_args
        assert kwargs['new_status'] == ProcessingStatusType.COMPLETED
        assert kwargs['progress_percentage'] == 100.0
    
    @patch('src.services.status_integration.get_db_session')
    def test_finish_workflow_failure(self, mock_get_db_session):
        """Test finishing workflow with failure."""
        mock_session = Mock()
        mock_get_db_session.return_value = mock_session
        
        manager = WorkflowStatusManager("TestWorkflow")
        manager.workflow_status_id = "workflow_status_123"
        
        manager.finish_workflow(success=False, error_info="Workflow failed")
        
        manager.status_service.record_error.assert_called_once()
        args, kwargs = manager.status_service.record_error.call_args
        assert kwargs['error_info'] == "Workflow failed"
        assert kwargs['should_retry'] is False
    
    @patch('src.services.status_integration.get_db_session')
    def test_cleanup(self, mock_get_db_session):
        """Test cleanup of resources."""
        mock_session = Mock()
        mock_get_db_session.return_value = mock_session
        
        manager = WorkflowStatusManager("TestWorkflow")
        
        # Should not raise any exceptions
        manager.cleanup()
        
        mock_session.close.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])