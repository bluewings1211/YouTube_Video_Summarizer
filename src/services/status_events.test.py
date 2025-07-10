"""
Tests for the status event system.

This module tests the event system components including event handlers,
event manager, and event-aware status services.
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from .status_events import (
    StatusEvent, EventType, EventPriority,
    EventHandler, LoggingEventHandler, DatabaseEventHandler, WebhookEventHandler,
    StatusEventManager, get_event_manager, initialize_event_system, shutdown_event_system
)
from .status_event_integration import (
    EventAwareStatusService, EventAwareStatusUpdater, StatusEventIntegrator,
    get_status_event_integrator
)
from ..database.status_models import ProcessingStatusType


class TestStatusEvent:
    """Test cases for StatusEvent."""
    
    def test_status_event_creation(self):
        """Test creating a status event."""
        event = StatusEvent(
            event_id="test_event_123",
            event_type=EventType.STATUS_UPDATED,
            timestamp=datetime.utcnow(),
            status_id="status_123",
            previous_status=ProcessingStatusType.STARTING,
            new_status=ProcessingStatusType.COMPLETED,
            progress_percentage=100.0
        )
        
        assert event.event_id == "test_event_123"
        assert event.event_type == EventType.STATUS_UPDATED
        assert event.status_id == "status_123"
        assert event.previous_status == ProcessingStatusType.STARTING
        assert event.new_status == ProcessingStatusType.COMPLETED
        assert event.progress_percentage == 100.0
    
    def test_status_event_auto_fields(self):
        """Test automatic field generation."""
        event = StatusEvent(
            event_type=EventType.STATUS_CREATED,
            status_id="status_123"
        )
        
        assert event.event_id.startswith("event_")
        assert event.timestamp is not None
        assert event.tags == []
        assert event.metadata == {}
    
    def test_status_event_to_dict(self):
        """Test converting event to dictionary."""
        timestamp = datetime.utcnow()
        event = StatusEvent(
            event_id="test_event_123",
            event_type=EventType.STATUS_UPDATED,
            timestamp=timestamp,
            status_id="status_123",
            new_status=ProcessingStatusType.COMPLETED
        )
        
        data = event.to_dict()
        
        assert data['event_id'] == "test_event_123"
        assert data['event_type'] == "status_updated"
        assert data['timestamp'] == timestamp.isoformat()
        assert data['status_id'] == "status_123"
        assert data['new_status'] == "completed"
    
    def test_status_event_to_json(self):
        """Test converting event to JSON."""
        event = StatusEvent(
            event_type=EventType.ERROR_OCCURRED,
            status_id="status_123",
            error_info="Test error"
        )
        
        json_str = event.to_json()
        parsed = json.loads(json_str)
        
        assert parsed['event_type'] == "error_occurred"
        assert parsed['status_id'] == "status_123"
        assert parsed['error_info'] == "Test error"
    
    def test_status_event_from_dict(self):
        """Test creating event from dictionary."""
        data = {
            'event_id': 'test_event_123',
            'event_type': 'status_updated',
            'timestamp': '2023-01-01T12:00:00',
            'status_id': 'status_123',
            'new_status': 'completed',
            'priority': 'high'
        }
        
        event = StatusEvent.from_dict(data)
        
        assert event.event_id == "test_event_123"
        assert event.event_type == EventType.STATUS_UPDATED
        assert event.status_id == "status_123"
        assert event.new_status == ProcessingStatusType.COMPLETED
        assert event.priority == EventPriority.HIGH


class MockEventHandler(EventHandler):
    """Mock event handler for testing."""
    
    def __init__(self, handled_event_types=None):
        self.handled_event_types = handled_event_types or {EventType.STATUS_UPDATED}
        self.handled_events = []
        self.should_fail = False
    
    async def handle_event(self, event: StatusEvent) -> bool:
        if self.should_fail:
            raise Exception("Handler failed")
        
        self.handled_events.append(event)
        return True
    
    def get_handled_event_types(self):
        return self.handled_event_types


class TestEventHandlers:
    """Test cases for event handlers."""
    
    @pytest.mark.asyncio
    async def test_logging_event_handler(self):
        """Test logging event handler."""
        handler = LoggingEventHandler()
        
        event = StatusEvent(
            event_type=EventType.STATUS_UPDATED,
            status_id="status_123",
            new_status=ProcessingStatusType.COMPLETED,
            progress_percentage=100.0
        )
        
        with patch.object(handler.logger, 'log') as mock_log:
            result = await handler.handle_event(event)
            
            assert result is True
            mock_log.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_database_event_handler(self):
        """Test database event handler."""
        handler = DatabaseEventHandler()
        
        event = StatusEvent(
            event_type=EventType.STATUS_CREATED,
            status_id="status_123"
        )
        
        with patch('src.services.status_events.get_db_session') as mock_get_session:
            mock_session = Mock()
            mock_get_session.return_value.__enter__.return_value = mock_session
            
            result = await handler.handle_event(event)
            
            assert result is True
    
    @pytest.mark.asyncio
    async def test_webhook_event_handler(self):
        """Test webhook event handler."""
        webhook_urls = ["http://example.com/webhook1", "http://example.com/webhook2"]
        handler = WebhookEventHandler(webhook_urls)
        
        event = StatusEvent(
            event_type=EventType.ERROR_OCCURRED,
            status_id="status_123",
            error_info="Test error"
        )
        
        # Mock successful webhook responses
        mock_response = Mock()
        mock_response.status = 200
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value = mock_response
            
            result = await handler.handle_event(event)
            
            assert result is True
            assert mock_post.call_count == 2  # Two webhook URLs
    
    def test_event_handler_should_handle_event(self):
        """Test event handler filtering."""
        handler = MockEventHandler({EventType.STATUS_UPDATED, EventType.ERROR_OCCURRED})
        
        # Should handle
        event1 = StatusEvent(event_type=EventType.STATUS_UPDATED, status_id="status_123")
        assert handler.should_handle_event(event1) is True
        
        # Should not handle
        event2 = StatusEvent(event_type=EventType.WORKFLOW_STARTED, status_id="status_123")
        assert handler.should_handle_event(event2) is False


class TestStatusEventManager:
    """Test cases for StatusEventManager."""
    
    @pytest.fixture
    def event_manager(self):
        """Create event manager for testing."""
        manager = StatusEventManager(max_workers=2)
        return manager
    
    def test_event_manager_initialization(self, event_manager):
        """Test event manager initialization."""
        assert event_manager.max_workers == 2
        assert len(event_manager.handlers) >= 1  # Default logging handler
        assert event_manager.running is False
        assert event_manager.events_processed == 0
    
    def test_add_remove_handlers(self, event_manager):
        """Test adding and removing event handlers."""
        handler = MockEventHandler()
        initial_count = len(event_manager.handlers)
        
        # Add handler
        event_manager.add_handler(handler)
        assert len(event_manager.handlers) == initial_count + 1
        assert handler in event_manager.handlers
        
        # Remove handler
        event_manager.remove_handler(handler)
        assert len(event_manager.handlers) == initial_count
        assert handler not in event_manager.handlers
    
    @pytest.mark.asyncio
    async def test_emit_event(self, event_manager):
        """Test emitting events."""
        handler = MockEventHandler()
        event_manager.add_handler(handler)
        
        event = StatusEvent(
            event_type=EventType.STATUS_UPDATED,
            status_id="status_123"
        )
        
        # Start manager
        await event_manager.start()
        
        try:
            # Emit event
            await event_manager.emit_event(event)
            
            # Wait for processing
            await asyncio.sleep(0.1)
            
            # Check that event was handled
            assert len(handler.handled_events) == 1
            assert handler.handled_events[0].status_id == "status_123"
            
        finally:
            await event_manager.stop()
    
    @pytest.mark.asyncio
    async def test_emit_convenience_methods(self, event_manager):
        """Test convenience methods for emitting events."""
        handler = MockEventHandler({EventType.STATUS_CREATED, EventType.PROGRESS_UPDATED, EventType.ERROR_OCCURRED})
        event_manager.add_handler(handler)
        
        await event_manager.start()
        
        try:
            # Test status created
            await event_manager.emit_status_created(
                status_id="status_123",
                initial_status=ProcessingStatusType.STARTING
            )
            
            # Test progress updated
            await event_manager.emit_progress_updated(
                status_id="status_123",
                progress_percentage=50.0,
                current_step="Processing"
            )
            
            # Test error occurred
            await event_manager.emit_error_occurred(
                status_id="status_123",
                error_info="Test error"
            )
            
            # Wait for processing
            await asyncio.sleep(0.1)
            
            # Check events were handled
            assert len(handler.handled_events) == 3
            
            # Check event types
            event_types = [event.event_type for event in handler.handled_events]
            assert EventType.STATUS_CREATED in event_types
            assert EventType.PROGRESS_UPDATED in event_types
            assert EventType.ERROR_OCCURRED in event_types
            
        finally:
            await event_manager.stop()
    
    @pytest.mark.asyncio
    async def test_handler_failure(self, event_manager):
        """Test handling of handler failures."""
        handler = MockEventHandler()
        handler.should_fail = True
        event_manager.add_handler(handler)
        
        event = StatusEvent(
            event_type=EventType.STATUS_UPDATED,
            status_id="status_123"
        )
        
        await event_manager.start()
        
        try:
            # Emit event
            await event_manager.emit_event(event)
            
            # Wait for processing
            await asyncio.sleep(0.1)
            
            # Check statistics
            stats = event_manager.get_statistics()
            assert stats['handlers_failed'] > 0
            
        finally:
            await event_manager.stop()
    
    def test_get_statistics(self, event_manager):
        """Test getting event manager statistics."""
        stats = event_manager.get_statistics()
        
        assert 'events_processed' in stats
        assert 'events_failed' in stats
        assert 'handlers_failed' in stats
        assert 'handlers_count' in stats
        assert 'queue_size' in stats
        assert 'workers_running' in stats
        assert 'is_running' in stats


class TestEventAwareServices:
    """Test cases for event-aware status services."""
    
    @pytest.mark.asyncio
    async def test_event_aware_status_service(self):
        """Test event-aware status service."""
        # Mock the event manager
        mock_event_manager = AsyncMock()
        
        with patch('src.services.status_event_integration.get_db_session'), \
             patch.object(EventAwareStatusService, '_emit_status_created_event') as mock_emit:
            
            service = EventAwareStatusService(event_manager=mock_event_manager)
            
            # Mock the parent create_processing_status method
            mock_status = Mock()
            mock_status.status_id = "status_123"
            mock_status.status = ProcessingStatusType.STARTING
            
            with patch.object(service.__class__.__bases__[0], 'create_processing_status', return_value=mock_status):
                # Create status
                status = service.create_processing_status(video_id=1)
                
                assert status.status_id == "status_123"
    
    @pytest.mark.asyncio
    async def test_status_event_integrator(self):
        """Test status event integrator."""
        integrator = StatusEventIntegrator()
        
        # Test that services are created
        assert integrator.status_service is not None
        assert integrator.status_updater is not None
        assert isinstance(integrator.status_service, EventAwareStatusService)
        assert isinstance(integrator.status_updater, EventAwareStatusUpdater)
        
        # Test convenience methods
        with patch.object(integrator.event_manager, 'emit_workflow_started') as mock_emit:
            await integrator.emit_workflow_started(
                workflow_status_id="workflow_123",
                workflow_name="TestWorkflow"
            )
            mock_emit.assert_called_once()


class TestGlobalFunctions:
    """Test cases for global functions."""
    
    def test_get_event_manager(self):
        """Test getting global event manager."""
        manager1 = get_event_manager()
        manager2 = get_event_manager()
        
        # Should return the same instance
        assert manager1 is manager2
    
    @pytest.mark.asyncio
    async def test_initialize_shutdown_event_system(self):
        """Test initializing and shutting down event system."""
        # Initialize
        await initialize_event_system()
        
        manager = get_event_manager()
        assert manager.running is True
        
        # Shutdown
        await shutdown_event_system()
        
        # Manager should be reset
        # Note: This test might need adjustment based on actual implementation


if __name__ == "__main__":
    pytest.main([__file__])