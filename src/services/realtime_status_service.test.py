"""
Tests for the real-time status service.
"""

import pytest
import asyncio
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from fastapi import WebSocket

from ..database.status_models import ProcessingStatus, ProcessingStatusType, ProcessingPriority
from ..services.realtime_status_service import (
    RealTimeStatusService, ConnectionManager, Subscription, SubscriptionType,
    WebSocketMessage
)


class MockWebSocket:
    """Mock WebSocket for testing."""
    
    def __init__(self):
        self.messages_sent = []
        self.is_connected = True
        self.ping_called = False
        
    async def accept(self):
        """Mock accept method."""
        pass
    
    async def send_text(self, data: str):
        """Mock send_text method."""
        if not self.is_connected:
            raise Exception("WebSocket disconnected")
        self.messages_sent.append(data)
    
    async def receive_text(self):
        """Mock receive_text method."""
        if not self.is_connected:
            raise Exception("WebSocket disconnected")
        return '{"type": "ping"}'
    
    async def ping(self):
        """Mock ping method."""
        self.ping_called = True
        if not self.is_connected:
            raise Exception("WebSocket disconnected")
    
    def disconnect(self):
        """Simulate disconnection."""
        self.is_connected = False


class TestWebSocketMessage:
    """Test cases for WebSocketMessage."""
    
    def test_websocket_message_creation(self):
        """Test WebSocketMessage creation."""
        timestamp = datetime.utcnow()
        message = WebSocketMessage(
            type="status_update",
            event="status_changed",
            data={"status_id": "test123"},
            timestamp=timestamp,
            subscription_id="sub123"
        )
        
        assert message.type == "status_update"
        assert message.event == "status_changed"
        assert message.data == {"status_id": "test123"}
        assert message.timestamp == timestamp
        assert message.subscription_id == "sub123"
    
    def test_websocket_message_to_dict(self):
        """Test WebSocketMessage to_dict conversion."""
        timestamp = datetime.utcnow()
        message = WebSocketMessage(
            type="status_update",
            event="status_changed",
            data={"status_id": "test123"},
            timestamp=timestamp
        )
        
        result = message.to_dict()
        
        assert result["type"] == "status_update"
        assert result["event"] == "status_changed"
        assert result["data"] == {"status_id": "test123"}
        assert result["timestamp"] == timestamp.isoformat()
        assert result["subscription_id"] is None


class TestSubscription:
    """Test cases for Subscription."""
    
    def test_subscription_creation(self):
        """Test Subscription creation."""
        mock_websocket = MockWebSocket()
        subscription = Subscription(
            websocket=mock_websocket,
            subscription_type=SubscriptionType.SPECIFIC_STATUS,
            filter_value="status123",
            subscription_id="sub123"
        )
        
        assert subscription.websocket == mock_websocket
        assert subscription.subscription_type == SubscriptionType.SPECIFIC_STATUS
        assert subscription.filter_value == "status123"
        assert subscription.subscription_id == "sub123"
        assert subscription.created_at is not None
    
    def test_subscription_matches_all_statuses(self):
        """Test subscription matching for all statuses."""
        mock_websocket = MockWebSocket()
        subscription = Subscription(
            websocket=mock_websocket,
            subscription_type=SubscriptionType.ALL_STATUSES
        )
        
        assert subscription.matches({"status_id": "any"})
        assert subscription.matches({"video_id": 123})
        assert subscription.matches({})
    
    def test_subscription_matches_specific_status(self):
        """Test subscription matching for specific status."""
        mock_websocket = MockWebSocket()
        subscription = Subscription(
            websocket=mock_websocket,
            subscription_type=SubscriptionType.SPECIFIC_STATUS,
            filter_value="status123"
        )
        
        assert subscription.matches({"status_id": "status123"})
        assert not subscription.matches({"status_id": "status456"})
        assert not subscription.matches({"video_id": 123})
    
    def test_subscription_matches_video_status(self):
        """Test subscription matching for video status."""
        mock_websocket = MockWebSocket()
        subscription = Subscription(
            websocket=mock_websocket,
            subscription_type=SubscriptionType.VIDEO_STATUS,
            filter_value="123"
        )
        
        assert subscription.matches({"video_id": 123})
        assert subscription.matches({"video_id": "123"})  # String conversion
        assert not subscription.matches({"video_id": 456})
        assert not subscription.matches({"status_id": "test"})
    
    def test_subscription_matches_worker_status(self):
        """Test subscription matching for worker status."""
        mock_websocket = MockWebSocket()
        subscription = Subscription(
            websocket=mock_websocket,
            subscription_type=SubscriptionType.WORKER_STATUS,
            filter_value="worker-1"
        )
        
        assert subscription.matches({"worker_id": "worker-1"})
        assert not subscription.matches({"worker_id": "worker-2"})
        assert not subscription.matches({"status_id": "test"})


class TestConnectionManager:
    """Test cases for ConnectionManager."""
    
    @pytest.fixture
    def connection_manager(self):
        """ConnectionManager fixture."""
        return ConnectionManager()
    
    def test_connection_manager_init(self, connection_manager):
        """Test ConnectionManager initialization."""
        assert len(connection_manager.active_connections) == 0
        assert len(connection_manager.subscriptions) == 0
        assert len(connection_manager.subscription_index) == 0
    
    @pytest.mark.asyncio
    async def test_connect_websocket(self, connection_manager):
        """Test WebSocket connection."""
        mock_websocket = MockWebSocket()
        
        await connection_manager.connect(mock_websocket, "client123")
        
        assert mock_websocket in connection_manager.active_connections
        assert len(mock_websocket.messages_sent) == 1
        
        # Check welcome message
        welcome_msg = json.loads(mock_websocket.messages_sent[0])
        assert welcome_msg["type"] == "system"
        assert welcome_msg["event"] == "connected"
        assert welcome_msg["data"]["client_id"] == "client123"
    
    def test_disconnect_websocket(self, connection_manager):
        """Test WebSocket disconnection."""
        mock_websocket = MockWebSocket()
        connection_manager.active_connections.add(mock_websocket)
        
        # Add a subscription
        subscription = Subscription(
            websocket=mock_websocket,
            subscription_type=SubscriptionType.ALL_STATUSES
        )
        connection_manager.subscriptions[mock_websocket].append(subscription)
        connection_manager.subscription_index["all_statuses:all"].append(subscription)
        
        # Disconnect
        connection_manager.disconnect(mock_websocket)
        
        assert mock_websocket not in connection_manager.active_connections
        assert mock_websocket not in connection_manager.subscriptions
        assert len(connection_manager.subscription_index["all_statuses:all"]) == 0
    
    @pytest.mark.asyncio
    async def test_send_personal_message(self, connection_manager):
        """Test sending personal message."""
        mock_websocket = MockWebSocket()
        connection_manager.active_connections.add(mock_websocket)
        
        message = WebSocketMessage(
            type="test",
            event="test_event",
            data={"test": "data"},
            timestamp=datetime.utcnow()
        )
        
        await connection_manager.send_personal_message(mock_websocket, message)
        
        assert len(mock_websocket.messages_sent) == 1
        sent_data = json.loads(mock_websocket.messages_sent[0])
        assert sent_data["type"] == "test"
        assert sent_data["event"] == "test_event"
        assert sent_data["data"]["test"] == "data"
    
    @pytest.mark.asyncio
    async def test_send_personal_message_disconnect_on_error(self, connection_manager):
        """Test disconnection on send error."""
        mock_websocket = MockWebSocket()
        mock_websocket.disconnect()  # Simulate disconnection
        connection_manager.active_connections.add(mock_websocket)
        
        message = WebSocketMessage(
            type="test",
            event="test_event",
            data={"test": "data"},
            timestamp=datetime.utcnow()
        )
        
        await connection_manager.send_personal_message(mock_websocket, message)
        
        # Should be removed from active connections
        assert mock_websocket not in connection_manager.active_connections
    
    @pytest.mark.asyncio
    async def test_broadcast_message(self, connection_manager):
        """Test broadcasting message."""
        mock_websocket1 = MockWebSocket()
        mock_websocket2 = MockWebSocket()
        
        connection_manager.active_connections.add(mock_websocket1)
        connection_manager.active_connections.add(mock_websocket2)
        
        message = WebSocketMessage(
            type="broadcast",
            event="system_message",
            data={"message": "Hello all"},
            timestamp=datetime.utcnow()
        )
        
        await connection_manager.broadcast_message(message)
        
        assert len(mock_websocket1.messages_sent) == 1
        assert len(mock_websocket2.messages_sent) == 1
        
        # Check both received the same message
        msg1 = json.loads(mock_websocket1.messages_sent[0])
        msg2 = json.loads(mock_websocket2.messages_sent[0])
        assert msg1 == msg2
        assert msg1["data"]["message"] == "Hello all"
    
    @pytest.mark.asyncio
    async def test_send_to_subscribers(self, connection_manager):
        """Test sending to subscribers."""
        mock_websocket1 = MockWebSocket()
        mock_websocket2 = MockWebSocket()
        
        connection_manager.active_connections.add(mock_websocket1)
        connection_manager.active_connections.add(mock_websocket2)
        
        # Subscribe websocket1 to specific status
        connection_manager.subscribe(
            mock_websocket1,
            SubscriptionType.SPECIFIC_STATUS,
            "status123"
        )
        
        # Subscribe websocket2 to all statuses
        connection_manager.subscribe(
            mock_websocket2,
            SubscriptionType.ALL_STATUSES
        )
        
        # Send event that matches specific status
        await connection_manager.send_to_subscribers(
            "status_changed",
            {"status_id": "status123", "new_status": "COMPLETED"}
        )
        
        # Both should receive the message
        assert len(mock_websocket1.messages_sent) == 1
        assert len(mock_websocket2.messages_sent) == 1
        
        # Send event that doesn't match specific status
        mock_websocket1.messages_sent.clear()
        mock_websocket2.messages_sent.clear()
        
        await connection_manager.send_to_subscribers(
            "status_changed",
            {"status_id": "status456", "new_status": "COMPLETED"}
        )
        
        # Only websocket2 (all statuses) should receive the message
        assert len(mock_websocket1.messages_sent) == 0
        assert len(mock_websocket2.messages_sent) == 1
    
    def test_subscribe_and_unsubscribe(self, connection_manager):
        """Test subscription and unsubscription."""
        mock_websocket = MockWebSocket()
        connection_manager.active_connections.add(mock_websocket)
        
        # Subscribe
        sub_id = connection_manager.subscribe(
            mock_websocket,
            SubscriptionType.SPECIFIC_STATUS,
            "status123",
            "sub123"
        )
        
        assert sub_id == "sub123"
        assert len(connection_manager.subscriptions[mock_websocket]) == 1
        assert len(connection_manager.subscription_index["specific_status:status123"]) == 1
        
        # Unsubscribe
        success = connection_manager.unsubscribe(mock_websocket, "sub123")
        
        assert success
        assert len(connection_manager.subscriptions[mock_websocket]) == 0
        assert len(connection_manager.subscription_index["specific_status:status123"]) == 0
        
        # Try to unsubscribe non-existent subscription
        success = connection_manager.unsubscribe(mock_websocket, "nonexistent")
        assert not success
    
    def test_get_counts(self, connection_manager):
        """Test connection and subscription counts."""
        mock_websocket1 = MockWebSocket()
        mock_websocket2 = MockWebSocket()
        
        connection_manager.active_connections.add(mock_websocket1)
        connection_manager.active_connections.add(mock_websocket2)
        
        assert connection_manager.get_connection_count() == 2
        assert connection_manager.get_subscription_count() == 0
        
        connection_manager.subscribe(mock_websocket1, SubscriptionType.ALL_STATUSES)
        connection_manager.subscribe(mock_websocket2, SubscriptionType.SPECIFIC_STATUS, "status123")
        
        assert connection_manager.get_subscription_count() == 2


class TestRealTimeStatusService:
    """Test cases for RealTimeStatusService."""
    
    @pytest.fixture
    def mock_db_session(self):
        """Mock database session."""
        return Mock()
    
    @pytest.fixture
    def mock_status_service(self):
        """Mock StatusService."""
        return Mock()
    
    @pytest.fixture
    def mock_status_updater(self):
        """Mock StatusUpdater."""
        mock_updater = Mock()
        mock_updater.event_handler = Mock()
        mock_updater.event_handler.subscribe = Mock()
        return mock_updater
    
    @pytest.fixture
    def realtime_service(self, mock_db_session):
        """RealTimeStatusService fixture."""
        with patch('src.services.realtime_status_service.StatusService'), \
             patch('src.services.realtime_status_service.StatusUpdater'):
            service = RealTimeStatusService(db_session=mock_db_session)
            return service
    
    def test_realtime_service_init(self, realtime_service):
        """Test RealTimeStatusService initialization."""
        assert realtime_service.connection_manager is not None
        assert realtime_service.status_service is not None
        assert realtime_service.status_updater is not None
        assert len(realtime_service._background_tasks) == 0
        assert not realtime_service._shutdown_flag
    
    @pytest.mark.asyncio
    async def test_connect_websocket(self, realtime_service):
        """Test WebSocket connection."""
        mock_websocket = MockWebSocket()
        
        await realtime_service.connect_websocket(mock_websocket, "client123")
        
        assert mock_websocket in realtime_service.connection_manager.active_connections
        assert len(mock_websocket.messages_sent) == 1
    
    def test_disconnect_websocket(self, realtime_service):
        """Test WebSocket disconnection."""
        mock_websocket = MockWebSocket()
        realtime_service.connection_manager.active_connections.add(mock_websocket)
        
        realtime_service.disconnect_websocket(mock_websocket)
        
        assert mock_websocket not in realtime_service.connection_manager.active_connections
    
    @pytest.mark.asyncio
    async def test_handle_subscribe_message(self, realtime_service):
        """Test handling subscribe message."""
        mock_websocket = MockWebSocket()
        realtime_service.connection_manager.active_connections.add(mock_websocket)
        
        message = {
            "type": "subscribe",
            "subscription_type": "all_statuses",
            "id": "msg123"
        }
        
        await realtime_service.handle_websocket_message(mock_websocket, message)
        
        # Should have sent a response
        assert len(mock_websocket.messages_sent) == 1
        response = json.loads(mock_websocket.messages_sent[0])
        assert response["type"] == "system"
        assert response["event"] == "subscribed"
        assert "subscription_id" in response["data"]
    
    @pytest.mark.asyncio
    async def test_handle_ping_message(self, realtime_service):
        """Test handling ping message."""
        mock_websocket = MockWebSocket()
        
        message = {
            "type": "ping",
            "id": "ping123"
        }
        
        await realtime_service.handle_websocket_message(mock_websocket, message)
        
        # Should have sent pong response
        assert len(mock_websocket.messages_sent) == 1
        response = json.loads(mock_websocket.messages_sent[0])
        assert response["type"] == "system"
        assert response["event"] == "pong"
        assert "server_time" in response["data"]
        assert "connections" in response["data"]
    
    @pytest.mark.asyncio
    async def test_handle_get_status_message(self, realtime_service):
        """Test handling get status message."""
        mock_websocket = MockWebSocket()
        
        # Mock status service response
        mock_status = Mock()
        mock_status.status_id = "status123"
        mock_status.status = ProcessingStatusType.COMPLETED
        mock_status.progress_percentage = 100.0
        mock_status.current_step = "Completed"
        mock_status.worker_id = "worker-1"
        mock_status.updated_at = datetime.utcnow()
        
        realtime_service.status_service.get_processing_status.return_value = mock_status
        
        message = {
            "type": "get_status",
            "status_id": "status123",
            "id": "status_req123"
        }
        
        await realtime_service.handle_websocket_message(mock_websocket, message)
        
        # Should have sent status response
        assert len(mock_websocket.messages_sent) == 1
        response = json.loads(mock_websocket.messages_sent[0])
        assert response["type"] == "data"
        assert response["event"] == "status_response"
        assert response["data"]["status_id"] == "status123"
        assert response["data"]["found"] is True
        assert response["data"]["status"]["status"] == "COMPLETED"
    
    @pytest.mark.asyncio
    async def test_handle_get_status_not_found(self, realtime_service):
        """Test handling get status message when status not found."""
        mock_websocket = MockWebSocket()
        
        # Mock status service response - not found
        realtime_service.status_service.get_processing_status.return_value = None
        
        message = {
            "type": "get_status",
            "status_id": "nonexistent",
            "id": "status_req123"
        }
        
        await realtime_service.handle_websocket_message(mock_websocket, message)
        
        # Should have sent status response with found = False
        assert len(mock_websocket.messages_sent) == 1
        response = json.loads(mock_websocket.messages_sent[0])
        assert response["data"]["found"] is False
        assert response["data"]["status"] is None
    
    @pytest.mark.asyncio
    async def test_handle_unknown_message_type(self, realtime_service):
        """Test handling unknown message type."""
        mock_websocket = MockWebSocket()
        
        message = {
            "type": "unknown_type",
            "id": "unknown123"
        }
        
        await realtime_service.handle_websocket_message(mock_websocket, message)
        
        # Should have sent error response
        assert len(mock_websocket.messages_sent) == 1
        response = json.loads(mock_websocket.messages_sent[0])
        assert response["type"] == "error"
        assert "Unknown message type" in response["data"]["error"]
    
    @pytest.mark.asyncio
    async def test_broadcast_system_message(self, realtime_service):
        """Test broadcasting system message."""
        mock_websocket1 = MockWebSocket()
        mock_websocket2 = MockWebSocket()
        
        realtime_service.connection_manager.active_connections.add(mock_websocket1)
        realtime_service.connection_manager.active_connections.add(mock_websocket2)
        
        await realtime_service.broadcast_system_message("Test message", "test_event")
        
        # Both should receive the message
        assert len(mock_websocket1.messages_sent) == 1
        assert len(mock_websocket2.messages_sent) == 1
        
        response = json.loads(mock_websocket1.messages_sent[0])
        assert response["type"] == "system"
        assert response["event"] == "test_event"
        assert response["data"]["message"] == "Test message"
    
    def test_get_connection_stats(self, realtime_service):
        """Test getting connection statistics."""
        mock_websocket = MockWebSocket()
        realtime_service.connection_manager.active_connections.add(mock_websocket)
        realtime_service.connection_manager.subscribe(
            mock_websocket,
            SubscriptionType.ALL_STATUSES
        )
        
        stats = realtime_service.get_connection_stats()
        
        assert stats["active_connections"] == 1
        assert stats["total_subscriptions"] == 1
        assert stats["subscription_types"]["all_statuses"] == 1
        assert "background_tasks" in stats
        assert "metrics_cache_expires" in stats
    
    @pytest.mark.asyncio
    async def test_start_stop_background_tasks(self, realtime_service):
        """Test starting and stopping background tasks."""
        # Start background tasks
        await realtime_service.start_background_tasks()
        
        assert not realtime_service._shutdown_flag
        assert len(realtime_service._background_tasks) == 2  # metrics and cleanup tasks
        
        # Stop background tasks
        await realtime_service.stop_background_tasks()
        
        assert realtime_service._shutdown_flag
        assert len(realtime_service._background_tasks) == 0


if __name__ == "__main__":
    pytest.main([__file__])