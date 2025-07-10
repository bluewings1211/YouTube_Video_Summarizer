"""
Comprehensive test suite for the status tracking system.

This module provides comprehensive tests for all status tracking components
including services, models, APIs, events, and integrations.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.database.status_models import (
    ProcessingStatus, StatusHistory, StatusMetrics,
    ProcessingStatusType, ProcessingPriority, StatusChangeType
)
from src.services.status_service import StatusService
from src.services.status_updater import StatusUpdater, StatusUpdate, ProgressUpdate, ErrorUpdate
from src.services.status_metrics_service import StatusMetricsService
from src.services.status_integration import StatusTrackingMixin, WorkflowStatusManager
from src.services.status_filtering import (
    StatusFilterService, FilterQuery, FilterCondition, SortCondition,
    PaginationParams, SearchParams, FilterOperator, SortOrder
)
from src.services.status_events import StatusEventManager, StatusEvent, EventType
from src.services.status_event_integration import EventAwareStatusService, StatusEventIntegrator


class TestStatusTrackingModels:
    """Test cases for status tracking database models."""
    
    def test_processing_status_creation(self):
        """Test creating a ProcessingStatus instance."""
        status = ProcessingStatus(
            status_id="test_status_123",
            status=ProcessingStatusType.STARTING,
            priority=ProcessingPriority.NORMAL,
            progress_percentage=0.0,
            completed_steps=0,
            retry_count=0,
            max_retries=3
        )
        
        assert status.status_id == "test_status_123"
        assert status.status == ProcessingStatusType.STARTING
        assert status.priority == ProcessingPriority.NORMAL
        assert status.progress_percentage == 0.0
        assert status.can_retry is True
    
    def test_processing_status_can_retry(self):
        """Test retry logic for ProcessingStatus."""
        # Should be able to retry
        status = ProcessingStatus(
            status_id="test_123",
            status=ProcessingStatusType.FAILED,
            retry_count=1,
            max_retries=3
        )
        assert status.can_retry is True
        
        # Should not be able to retry
        status.retry_count = 3
        assert status.can_retry is False
    
    def test_status_history_creation(self):
        """Test creating a StatusHistory instance."""
        history = StatusHistory(
            processing_status_id=1,
            change_type=StatusChangeType.STATUS_UPDATE,
            previous_status=ProcessingStatusType.STARTING,
            new_status=ProcessingStatusType.YOUTUBE_METADATA,
            previous_progress=0.0,
            new_progress=25.0,
            change_reason="Progress update"
        )
        
        assert history.processing_status_id == 1
        assert history.change_type == StatusChangeType.STATUS_UPDATE
        assert history.previous_status == ProcessingStatusType.STARTING
        assert history.new_status == ProcessingStatusType.YOUTUBE_METADATA
    
    def test_status_metrics_creation(self):
        """Test creating a StatusMetrics instance."""
        metrics = StatusMetrics(
            metric_date=datetime.utcnow().date(),
            total_items=100,
            completed_items=80,
            failed_items=15,
            cancelled_items=5,
            pending_items=0,
            success_rate_percentage=80.0
        )
        
        assert metrics.total_items == 100
        assert metrics.completed_items == 80
        assert metrics.success_rate_percentage == 80.0


class TestStatusService:
    """Test cases for StatusService."""
    
    @pytest.fixture
    def mock_db_session(self):
        """Create mock database session."""
        session = Mock()
        return session
    
    @pytest.fixture
    def status_service(self, mock_db_session):
        """Create StatusService with mock session."""
        return StatusService(db_session=mock_db_session)
    
    def test_status_service_initialization(self, status_service):
        """Test StatusService initialization."""
        assert status_service.db_session is not None
        assert status_service._should_close_session is False
    
    def test_create_processing_status(self, status_service):
        """Test creating a processing status."""
        mock_status = Mock()
        mock_status.status_id = "test_status_123"
        
        status_service.db_session.add = Mock()
        status_service.db_session.commit = Mock()
        
        with patch.object(status_service, '_create_status_history'):
            result = status_service.create_processing_status(
                video_id=1,
                priority=ProcessingPriority.HIGH,
                total_steps=5
            )
            
            assert result.status_id.startswith("status_")
            assert result.status == ProcessingStatusType.QUEUED
            assert result.priority == ProcessingPriority.HIGH
            assert result.total_steps == 5
    
    def test_update_status(self, status_service):
        """Test updating a processing status."""
        # Mock existing status
        mock_status = Mock()
        mock_status.status = ProcessingStatusType.STARTING
        mock_status.progress_percentage = 0.0
        
        status_service.get_processing_status = Mock(return_value=mock_status)
        status_service.db_session.commit = Mock()
        
        with patch.object(status_service, '_create_status_history'):
            result = status_service.update_status(
                status_id="test_status_123",
                new_status=ProcessingStatusType.YOUTUBE_METADATA,
                progress_percentage=25.0,
                current_step="Extracting metadata"
            )
            
            assert result.status == ProcessingStatusType.YOUTUBE_METADATA
            assert result.progress_percentage == 25.0
            assert result.current_step == "Extracting metadata"
    
    def test_update_progress(self, status_service):
        """Test updating progress."""
        mock_status = Mock()
        mock_status.progress_percentage = 25.0
        
        status_service.get_processing_status = Mock(return_value=mock_status)
        status_service.db_session.commit = Mock()
        
        with patch.object(status_service, '_create_status_history'):
            result = status_service.update_progress(
                status_id="test_status_123",
                progress_percentage=50.0,
                current_step="Processing transcript"
            )
            
            assert result.progress_percentage == 50.0
            assert result.current_step == "Processing transcript"
    
    def test_record_error(self, status_service):
        """Test recording an error."""
        mock_status = Mock()
        mock_status.retry_count = 1
        mock_status.can_retry = True
        mock_status.status = ProcessingStatusType.STARTING
        
        status_service.get_processing_status = Mock(return_value=mock_status)
        status_service.db_session.commit = Mock()
        
        with patch.object(status_service, '_create_status_history'):
            result = status_service.record_error(
                status_id="test_status_123",
                error_info="Network timeout error",
                should_retry=True
            )
            
            assert result.error_info == "Network timeout error"
            assert result.retry_count == 2
            assert result.status == ProcessingStatusType.RETRY_PENDING
    
    def test_heartbeat(self, status_service):
        """Test sending heartbeat."""
        mock_status = Mock()
        
        status_service.get_processing_status = Mock(return_value=mock_status)
        status_service.db_session.commit = Mock()
        
        result = status_service.heartbeat(
            status_id="test_status_123",
            worker_id="worker_abc",
            progress_percentage=75.0
        )
        
        assert result.worker_id == "worker_abc"
        assert result.progress_percentage == 75.0
        assert result.heartbeat_at is not None
    
    def test_get_active_statuses(self, status_service):
        """Test getting active statuses."""
        mock_query = Mock()
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.all.return_value = []
        
        status_service.db_session.query.return_value = mock_query
        
        result = status_service.get_active_statuses(worker_id="worker_abc", limit=10)
        
        assert result == []
        mock_query.filter.assert_called()
        mock_query.limit.assert_called_with(10)
    
    def test_get_stale_statuses(self, status_service):
        """Test getting stale statuses."""
        mock_query = Mock()
        mock_query.filter.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.all.return_value = []
        
        status_service.db_session.query.return_value = mock_query
        
        result = status_service.get_stale_statuses(timeout_seconds=300, limit=5)
        
        assert result == []
        mock_query.filter.assert_called()
        mock_query.limit.assert_called_with(5)


class TestStatusUpdater:
    """Test cases for StatusUpdater."""
    
    @pytest.fixture
    def mock_db_session(self):
        """Create mock database session."""
        return Mock()
    
    @pytest.fixture
    def status_updater(self, mock_db_session):
        """Create StatusUpdater with mock session."""
        updater = StatusUpdater(db_session=mock_db_session)
        updater.status_service = Mock()
        return updater
    
    def test_status_updater_initialization(self, status_updater):
        """Test StatusUpdater initialization."""
        assert status_updater.db_session is not None
        assert status_updater.status_service is not None
        assert status_updater.event_handler is not None
        assert status_updater._batch_size == 100
        assert status_updater._batch_interval == 5
    
    def test_queue_status_update(self, status_updater):
        """Test queuing status updates."""
        update = StatusUpdate(
            status_id="test_123",
            new_status=ProcessingStatusType.COMPLETED,
            progress_percentage=100.0
        )
        
        status_updater.queue_status_update(update)
        
        assert len(status_updater.status_update_queue) == 1
        assert status_updater.status_update_queue[0] == update
    
    def test_queue_progress_update(self, status_updater):
        """Test queuing progress updates."""
        update = ProgressUpdate(
            status_id="test_123",
            progress_percentage=50.0,
            current_step="Processing"
        )
        
        status_updater.queue_progress_update(update)
        
        assert len(status_updater.progress_update_queue) == 1
        assert status_updater.progress_update_queue[0] == update
    
    def test_queue_error_update(self, status_updater):
        """Test queuing error updates."""
        update = ErrorUpdate(
            status_id="test_123",
            error_info="Test error",
            should_retry=True
        )
        
        status_updater.queue_error_update(update)
        
        assert len(status_updater.error_update_queue) == 1
        assert status_updater.error_update_queue[0] == update
    
    @pytest.mark.asyncio
    async def test_process_status_updates(self, status_updater):
        """Test processing status update queue."""
        # Add test update to queue
        update = StatusUpdate(
            status_id="test_123",
            new_status=ProcessingStatusType.COMPLETED
        )
        status_updater.status_update_queue.append(update)
        
        # Mock status service update
        mock_status = Mock()
        status_updater.status_service.update_status.return_value = mock_status
        
        # Process updates
        await status_updater._process_status_updates()
        
        # Verify queue is empty and service was called
        assert len(status_updater.status_update_queue) == 0
        status_updater.status_service.update_status.assert_called_once()


class TestStatusMetricsService:
    """Test cases for StatusMetricsService."""
    
    @pytest.fixture
    def mock_db_session(self):
        """Create mock database session."""
        return Mock()
    
    @pytest.fixture
    def metrics_service(self, mock_db_session):
        """Create StatusMetricsService with mock session."""
        return StatusMetricsService(db_session=mock_db_session)
    
    def test_metrics_service_initialization(self, metrics_service):
        """Test StatusMetricsService initialization."""
        assert metrics_service.db_session is not None
        assert metrics_service._should_close_session is False
    
    def test_calculate_daily_metrics(self, metrics_service):
        """Test calculating daily metrics."""
        target_date = datetime.utcnow().date()
        
        # Mock query results
        mock_query = Mock()
        mock_query.filter.return_value = mock_query
        mock_query.with_entities.return_value = mock_query
        mock_query.group_by.return_value = mock_query
        mock_query.all.return_value = [
            (ProcessingStatusType.COMPLETED, 80),
            (ProcessingStatusType.FAILED, 15),
            (ProcessingStatusType.CANCELLED, 5)
        ]
        mock_query.first.return_value = Mock(
            avg_time=120.5,
            median_time=110.0,
            max_time=300.0,
            min_time=45.0
        )
        
        metrics_service.db_session.query.return_value = mock_query
        metrics_service.db_session.add = Mock()
        metrics_service.db_session.commit = Mock()
        
        result = metrics_service.calculate_daily_metrics(target_date)
        
        assert result.metric_date == target_date
        assert result.total_items == 100
        assert result.completed_items == 80
        assert result.failed_items == 15
        assert result.success_rate_percentage == 80.0


class TestStatusIntegration:
    """Test cases for status integration components."""
    
    def test_status_tracking_mixin_initialization(self):
        """Test StatusTrackingMixin initialization."""
        class TestClass(StatusTrackingMixin):
            def __init__(self):
                super().__init__()
        
        with patch('src.services.status_integration.get_db_session'):
            test_obj = TestClass()
            
            assert hasattr(test_obj, '_status_service')
            assert hasattr(test_obj, '_status_updater')
            assert hasattr(test_obj, '_current_status_id')
            assert hasattr(test_obj, '_worker_id')
    
    def test_workflow_status_manager_initialization(self):
        """Test WorkflowStatusManager initialization."""
        with patch('src.services.status_integration.get_db_session'):
            manager = WorkflowStatusManager("TestWorkflow")
            
            assert manager.workflow_name == "TestWorkflow"
            assert manager.workflow_status_id is None
            assert manager.node_status_mapping == {}
            assert manager.current_node_index == 0
    
    def test_workflow_status_manager_start_workflow(self):
        """Test starting workflow status tracking."""
        with patch('src.services.status_integration.get_db_session'):
            manager = WorkflowStatusManager("TestWorkflow")
            
            # Mock status service
            mock_status = Mock()
            mock_status.status_id = "workflow_123"
            manager.status_service.create_processing_status.return_value = mock_status
            manager.status_service.update_status = Mock()
            
            status_id = manager.start_workflow(
                video_id=1,
                node_names=["Node1", "Node2"]
            )
            
            assert status_id == "workflow_123"
            assert manager.workflow_status_id == "workflow_123"
            assert manager.total_nodes == 2


class TestStatusFiltering:
    """Test cases for status filtering components."""
    
    @pytest.fixture
    def mock_db_session(self):
        """Create mock database session."""
        return Mock()
    
    @pytest.fixture
    def filter_service(self, mock_db_session):
        """Create StatusFilterService with mock session."""
        return StatusFilterService(db_session=mock_db_session)
    
    def test_filter_condition_creation(self):
        """Test creating filter conditions."""
        condition = FilterCondition(
            field="status",
            operator=FilterOperator.EQ,
            value=ProcessingStatusType.COMPLETED
        )
        
        assert condition.field == "status"
        assert condition.operator == FilterOperator.EQ
        assert condition.value == ProcessingStatusType.COMPLETED
    
    def test_pagination_params(self):
        """Test pagination parameters."""
        params = PaginationParams(page=2, page_size=50)
        
        assert params.page == 2
        assert params.page_size == 50
        assert params.offset == 50
        assert params.limit == 50
    
    def test_filter_query_construction(self):
        """Test constructing filter queries."""
        filters = [FilterCondition("status", FilterOperator.IN, [ProcessingStatusType.COMPLETED])]
        sorts = [SortCondition("created_at", SortOrder.DESC)]
        pagination = PaginationParams(page=1, page_size=20)
        
        query = FilterQuery(
            filters=filters,
            sorts=sorts,
            pagination=pagination
        )
        
        assert len(query.filters) == 1
        assert len(query.sorts) == 1
        assert query.pagination.page == 1
    
    def test_filter_statuses(self, filter_service):
        """Test filtering statuses."""
        # Mock query setup
        mock_query = Mock()
        mock_query.count.return_value = 5
        mock_query.all.return_value = []
        mock_query.offset.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.filter.return_value = mock_query
        
        filter_service.db_session.query.return_value = mock_query
        
        # Create filter query
        filter_query = FilterQuery(
            filters=[FilterCondition("status", FilterOperator.EQ, ProcessingStatusType.COMPLETED)]
        )
        
        result = filter_service.filter_statuses(filter_query)
        
        assert result.total_count == 5
        assert result.items == []
        assert result.filters_applied == 1


class TestStatusEvents:
    """Test cases for status event system."""
    
    def test_status_event_creation(self):
        """Test creating status events."""
        event = StatusEvent(
            event_type=EventType.STATUS_UPDATED,
            status_id="test_123",
            new_status=ProcessingStatusType.COMPLETED,
            progress_percentage=100.0
        )
        
        assert event.event_type == EventType.STATUS_UPDATED
        assert event.status_id == "test_123"
        assert event.new_status == ProcessingStatusType.COMPLETED
        assert event.progress_percentage == 100.0
    
    def test_status_event_serialization(self):
        """Test status event serialization."""
        event = StatusEvent(
            event_type=EventType.ERROR_OCCURRED,
            status_id="test_123",
            error_info="Test error"
        )
        
        data = event.to_dict()
        
        assert data['event_type'] == "error_occurred"
        assert data['status_id'] == "test_123"
        assert data['error_info'] == "Test error"
        
        # Test round-trip
        restored_event = StatusEvent.from_dict(data)
        assert restored_event.event_type == EventType.ERROR_OCCURRED
        assert restored_event.status_id == "test_123"
    
    @pytest.mark.asyncio
    async def test_status_event_manager(self):
        """Test status event manager."""
        manager = StatusEventManager(max_workers=2)
        
        # Add mock handler
        mock_handler = AsyncMock()
        mock_handler.should_handle_event.return_value = True
        mock_handler.handle_event.return_value = True
        mock_handler.get_handled_event_types.return_value = {EventType.STATUS_UPDATED}
        
        manager.add_handler(mock_handler)
        
        # Create and emit event
        event = StatusEvent(
            event_type=EventType.STATUS_UPDATED,
            status_id="test_123"
        )
        
        await manager.start()
        try:
            await manager.emit_event(event)
            
            # Wait for processing
            await asyncio.sleep(0.1)
            
            # Verify handler was called
            mock_handler.handle_event.assert_called_once()
        finally:
            await manager.stop()


class TestStatusAPIEndpoints:
    """Test cases for status API endpoints."""
    
    @pytest.fixture
    def mock_status_service(self):
        """Create mock status service."""
        return Mock()
    
    @pytest.fixture
    def mock_filter_service(self):
        """Create mock filter service."""
        return Mock()
    
    def test_get_status_endpoint(self, mock_status_service):
        """Test GET /api/status/{status_id} endpoint."""
        from src.api.status import get_status
        
        # Mock status
        mock_status = Mock()
        mock_status.status_id = "test_123"
        mock_status_service.get_processing_status.return_value = mock_status
        
        # Test successful retrieval
        result = get_status("test_123", mock_status_service)
        
        # Verify service was called correctly
        mock_status_service.get_processing_status.assert_called_with("test_123")
    
    def test_filter_statuses_endpoint(self, mock_filter_service):
        """Test POST /api/status/enhanced/filter endpoint."""
        from src.api.status_enhanced import filter_statuses
        from src.api.status_enhanced import AdvancedFilterRequest, FilterConditionModel
        
        # Create test request
        filter_request = AdvancedFilterRequest(
            filters=[
                FilterConditionModel(
                    field="status",
                    operator="eq",
                    value="completed"
                )
            ],
            page=1,
            page_size=20
        )
        
        # Mock filter service result
        from src.services.status_filtering import FilterResult
        mock_result = FilterResult(
            items=[],
            total_count=0,
            page=1,
            page_size=20,
            total_pages=0,
            has_next=False,
            has_previous=False,
            filters_applied=1,
            query_time_ms=10.0
        )
        mock_filter_service.filter_statuses.return_value = mock_result
        
        # Test endpoint
        result = filter_statuses(filter_request, mock_filter_service)
        
        # Verify result
        assert result.total_count == 0
        assert result.filters_applied == 1
        assert result.query_time_ms == 10.0


class TestStatusTrackingIntegration:
    """Integration tests for the complete status tracking system."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_status_tracking(self):
        """Test complete status tracking workflow."""
        with patch('src.services.status_integration.get_db_session'):
            # Create workflow manager
            manager = WorkflowStatusManager("IntegrationTest")
            
            # Mock database interactions
            mock_status = Mock()
            mock_status.status_id = "integration_test_123"
            manager.status_service.create_processing_status.return_value = mock_status
            manager.status_service.update_status = Mock()
            
            # Start workflow
            workflow_status_id = manager.start_workflow(
                video_id=1,
                node_names=["Node1", "Node2", "Node3"]
            )
            
            # Start nodes
            node1_status_id = manager.start_node("Node1", video_id=1)
            node2_status_id = manager.start_node("Node2", video_id=1)
            
            # Finish nodes
            manager.finish_node("Node1", success=True)
            manager.finish_node("Node2", success=False, error_info="Node2 failed")
            
            # Finish workflow
            manager.finish_workflow(success=False, error_info="Workflow failed due to Node2")
            
            # Verify interactions
            assert manager.workflow_status_id == workflow_status_id
            assert "Node1" in manager.node_status_mapping
            assert "Node2" in manager.node_status_mapping
    
    def test_status_filtering_integration(self):
        """Test status filtering integration."""
        with patch('src.services.status_filtering.get_db_session'):
            filter_service = StatusFilterService()
            
            # Mock query setup
            mock_query = Mock()
            mock_query.count.return_value = 100
            mock_query.all.return_value = []
            mock_query.offset.return_value = mock_query
            mock_query.limit.return_value = mock_query
            mock_query.order_by.return_value = mock_query
            mock_query.filter.return_value = mock_query
            
            filter_service.db_session.query.return_value = mock_query
            
            # Create complex filter query
            filter_query = FilterQuery(
                filters=[
                    FilterCondition("status", FilterOperator.IN, [
                        ProcessingStatusType.COMPLETED,
                        ProcessingStatusType.FAILED
                    ]),
                    FilterCondition("created_at", FilterOperator.GTE, datetime.utcnow() - timedelta(days=7))
                ],
                sorts=[
                    SortCondition("updated_at", SortOrder.DESC),
                    SortCondition("priority", SortOrder.ASC)
                ],
                pagination=PaginationParams(page=2, page_size=25),
                search=SearchParams(
                    query="error",
                    fields=["error_info", "current_step"]
                )
            )
            
            # Execute filter
            result = filter_service.filter_statuses(filter_query)
            
            # Verify result
            assert result.total_count == 100
            assert result.page == 2
            assert result.page_size == 25
            assert result.filters_applied == 2
    
    @pytest.mark.asyncio
    async def test_event_system_integration(self):
        """Test event system integration."""
        # Create event manager
        manager = StatusEventManager(max_workers=1)
        
        # Track events
        received_events = []
        
        class TestEventHandler:
            async def handle_event(self, event):
                received_events.append(event)
                return True
            
            def get_handled_event_types(self):
                return {EventType.STATUS_UPDATED, EventType.PROGRESS_UPDATED}
            
            def should_handle_event(self, event):
                return event.event_type in self.get_handled_event_types()
        
        handler = TestEventHandler()
        manager.add_handler(handler)
        
        # Start manager
        await manager.start()
        
        try:
            # Emit events
            await manager.emit_status_updated(
                status_id="test_123",
                previous_status=ProcessingStatusType.STARTING,
                new_status=ProcessingStatusType.COMPLETED,
                progress_percentage=100.0
            )
            
            await manager.emit_progress_updated(
                status_id="test_123",
                progress_percentage=75.0,
                current_step="Almost done"
            )
            
            # Wait for processing
            await asyncio.sleep(0.2)
            
            # Verify events were received
            assert len(received_events) == 2
            assert received_events[0].event_type == EventType.STATUS_UPDATED
            assert received_events[1].event_type == EventType.PROGRESS_UPDATED
            
        finally:
            await manager.stop()


if __name__ == "__main__":
    # Run with coverage
    pytest.main([
        __file__,
        "--cov=src.services",
        "--cov=src.api",
        "--cov=src.database.status_models",
        "--cov-report=html",
        "--cov-report=term-missing",
        "-v"
    ])