"""
Comprehensive tests for the notification system.

This module provides extensive test coverage for all notification system components
including models, services, API endpoints, webhook client, and integration layers.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ..database.notification_models import (
    NotificationConfig, Notification, NotificationLog, WebhookEndpoint,
    NotificationType, NotificationEvent, NotificationStatus, NotificationPriority
)
from ..database.models import Base
from .notification_service import (
    NotificationService, NotificationCreateRequest, NotificationTriggerRequest,
    NotificationDeliveryResult, NotificationStats, NotificationServiceError
)
from .notification_retry_service import (
    NotificationRetryService, RetryConfig, RetryStrategy, RetryStats
)
from .notification_monitoring_service import (
    NotificationMonitoringService, NotificationMetrics, ConfigurationMetrics
)
from .notification_event_integration import (
    NotificationEventListener, NotificationEventIntegration
)
from ..utils.webhook_client import (
    WebhookClient, WebhookConfig, WebhookRequest, WebhookResponse,
    WebhookAuthType, WebhookStatus
)


# Test fixtures
@pytest.fixture(scope="function")
def test_db_session():
    """Create a test database session."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    
    # Create notification tables
    from ..database.notification_models import create_notification_tables
    create_notification_tables(engine)
    
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    
    yield session
    
    session.close()


@pytest.fixture
def sample_notification_config(test_db_session):
    """Create a sample notification configuration."""
    config = NotificationConfig(
        config_id="test_config_1",
        name="Test Configuration",
        description="Test configuration for unit tests",
        notification_type=NotificationType.WEBHOOK,
        event_triggers=[NotificationEvent.PROCESSING_COMPLETED.value],
        target_address="https://example.com/webhook",
        priority=NotificationPriority.NORMAL,
        rate_limit_per_hour=100,
        rate_limit_per_day=1000,
        template_config={
            "event_templates": {
                "processing_completed": {
                    "subject": "Processing Complete",
                    "message": "Video {video_id} processing completed successfully"
                }
            }
        },
        retry_config={
            "max_retries": 3,
            "initial_delay_seconds": 60,
            "backoff_multiplier": 2.0
        }
    )
    test_db_session.add(config)
    test_db_session.commit()
    return config


@pytest.fixture
def sample_notification(test_db_session, sample_notification_config):
    """Create a sample notification."""
    notification = Notification(
        notification_id="test_notif_1",
        config_id=sample_notification_config.id,
        event_type=NotificationEvent.PROCESSING_COMPLETED,
        event_source="video_123",
        event_metadata={"video_id": 123, "title": "Test Video"},
        status=NotificationStatus.PENDING,
        priority=NotificationPriority.NORMAL,
        target_address="https://example.com/webhook",
        subject="Processing Complete",
        message="Video 123 processing completed successfully",
        payload={
            "event_type": "processing_completed",
            "video_id": 123,
            "timestamp": datetime.utcnow().isoformat()
        }
    )
    test_db_session.add(notification)
    test_db_session.commit()
    return notification


class TestNotificationModels:
    """Test notification database models."""

    def test_notification_config_creation(self, test_db_session):
        """Test creating a notification configuration."""
        config = NotificationConfig(
            config_id="test_config",
            name="Test Config",
            notification_type=NotificationType.EMAIL,
            event_triggers=[NotificationEvent.PROCESSING_COMPLETED.value],
            target_address="test@example.com"
        )
        
        test_db_session.add(config)
        test_db_session.commit()
        
        assert config.id is not None
        assert config.config_id == "test_config"
        assert config.notification_type == NotificationType.EMAIL
        assert config.is_active is True
        assert config.trigger_count_today == 0
        assert config.trigger_count_total == 0

    def test_notification_config_validation(self):
        """Test notification configuration validation."""
        # Test empty config ID
        with pytest.raises(ValueError):
            config = NotificationConfig(
                config_id="",
                name="Test",
                notification_type=NotificationType.EMAIL,
                event_triggers=[NotificationEvent.PROCESSING_COMPLETED.value],
                target_address="test@example.com"
            )

    def test_notification_creation(self, test_db_session, sample_notification_config):
        """Test creating a notification."""
        notification = Notification(
            notification_id="test_notification",
            config_id=sample_notification_config.id,
            event_type=NotificationEvent.PROCESSING_COMPLETED,
            status=NotificationStatus.PENDING,
            priority=NotificationPriority.NORMAL,
            target_address="https://example.com/webhook",
            message="Test message"
        )
        
        test_db_session.add(notification)
        test_db_session.commit()
        
        assert notification.id is not None
        assert notification.notification_id == "test_notification"
        assert notification.is_pending is True
        assert notification.can_retry is False  # No failures yet

    def test_notification_properties(self, test_db_session, sample_notification):
        """Test notification properties."""
        # Test pending status
        assert sample_notification.is_pending is True
        assert sample_notification.is_sent is False
        assert sample_notification.is_failed is False
        
        # Test ready to send
        assert sample_notification.is_ready_to_send is True
        
        # Update to sent status
        sample_notification.status = NotificationStatus.SENT
        sample_notification.sent_at = datetime.utcnow()
        test_db_session.commit()
        
        assert sample_notification.is_sent is True
        assert sample_notification.is_pending is False


class TestNotificationService:
    """Test the notification service."""

    def test_service_initialization(self, test_db_session):
        """Test notification service initialization."""
        service = NotificationService(db_session=test_db_session)
        assert service.db_session == test_db_session
        assert service._should_close_session is False

    def test_create_notification_config(self, test_db_session):
        """Test creating a notification configuration."""
        service = NotificationService(db_session=test_db_session)
        
        request = NotificationCreateRequest(
            name="Test Config",
            notification_type=NotificationType.WEBHOOK,
            event_triggers=[NotificationEvent.PROCESSING_COMPLETED],
            target_address="https://example.com/webhook",
            description="Test configuration"
        )
        
        config = service.create_notification_config(request)
        
        assert config is not None
        assert config.name == "Test Config"
        assert config.notification_type == NotificationType.WEBHOOK
        assert NotificationEvent.PROCESSING_COMPLETED.value in config.event_triggers
        assert config.target_address == "https://example.com/webhook"

    def test_get_notification_config(self, test_db_session, sample_notification_config):
        """Test retrieving a notification configuration."""
        service = NotificationService(db_session=test_db_session)
        
        config = service.get_notification_config(sample_notification_config.config_id)
        
        assert config is not None
        assert config.config_id == sample_notification_config.config_id
        assert config.name == sample_notification_config.name

    def test_trigger_notification(self, test_db_session, sample_notification_config):
        """Test triggering notifications."""
        service = NotificationService(db_session=test_db_session)
        
        request = NotificationTriggerRequest(
            event_type=NotificationEvent.PROCESSING_COMPLETED,
            event_source="video_123",
            event_metadata={"video_id": 123, "title": "Test Video"}
        )
        
        notifications = service.trigger_notification(request)
        
        assert len(notifications) == 1
        assert notifications[0].event_type == NotificationEvent.PROCESSING_COMPLETED
        assert notifications[0].event_source == "video_123"

    @patch('src.services.notification_service.WebhookClient')
    def test_send_webhook_notification(self, mock_webhook_client, test_db_session, sample_notification):
        """Test sending a webhook notification."""
        # Mock webhook client response
        mock_response = WebhookResponse(
            status=WebhookStatus.SUCCESS,
            status_code=200,
            response_text="OK",
            response_time_ms=150
        )
        mock_client_instance = Mock()
        mock_client_instance.send.return_value = mock_response
        mock_client_instance.__enter__.return_value = mock_client_instance
        mock_client_instance.__exit__.return_value = None
        mock_webhook_client.return_value = mock_client_instance
        
        service = NotificationService(db_session=test_db_session)
        
        result = service.send_notification(sample_notification.notification_id)
        
        assert result.is_success is True
        assert result.status == NotificationStatus.DELIVERED
        assert result.delivery_time_ms == 150

    def test_get_notification_stats(self, test_db_session, sample_notification):
        """Test getting notification statistics."""
        service = NotificationService(db_session=test_db_session)
        
        # Update notification to delivered status
        sample_notification.status = NotificationStatus.DELIVERED
        sample_notification.sent_at = datetime.utcnow()
        sample_notification.delivered_at = datetime.utcnow()
        test_db_session.commit()
        
        stats = service.get_notification_stats()
        
        assert stats.total_notifications == 1
        assert stats.sent_notifications == 1
        assert stats.failed_notifications == 0
        assert stats.success_rate == 100.0


class TestWebhookClient:
    """Test the webhook client."""

    def test_webhook_config_creation(self):
        """Test creating webhook configuration."""
        config = WebhookConfig(
            url="https://example.com/webhook",
            auth_type=WebhookAuthType.BEARER,
            auth_config={"token": "test_token"},
            timeout_seconds=30
        )
        
        assert config.url == "https://example.com/webhook"
        assert config.auth_type == WebhookAuthType.BEARER
        assert config.timeout_seconds == 30

    def test_webhook_config_validation(self):
        """Test webhook configuration validation."""
        # Test invalid URL
        with pytest.raises(ValueError):
            config = WebhookConfig(url="invalid-url")
        
        # Test invalid timeout
        with pytest.raises(ValueError):
            config = WebhookConfig(url="https://example.com", timeout_seconds=0)

    @patch('src.utils.webhook_client.httpx')
    def test_webhook_request_success(self, mock_httpx):
        """Test successful webhook request."""
        # Mock httpx response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = "OK"
        mock_response.headers = {"Content-Type": "application/json"}
        mock_response.url = "https://example.com/webhook"
        
        mock_client = Mock()
        mock_client.request.return_value = mock_response
        mock_httpx.Client.return_value.__enter__.return_value = mock_client
        
        config = WebhookConfig(url="https://example.com/webhook")
        request = WebhookRequest(payload={"test": "data"})
        
        client = WebhookClient(config)
        result = client.send(request)
        
        assert result.is_success is True
        assert result.status_code == 200

    def test_webhook_authentication_headers(self):
        """Test webhook authentication header generation."""
        config = WebhookConfig(
            url="https://example.com/webhook",
            auth_type=WebhookAuthType.BEARER,
            auth_config={"token": "test_token"}
        )
        
        client = WebhookClient(config)
        request = WebhookRequest(payload={"test": "data"})
        
        headers = client._prepare_headers(request)
        
        assert "Authorization" in headers
        assert headers["Authorization"] == "Bearer test_token"


class TestNotificationRetryService:
    """Test the notification retry service."""

    def test_retry_service_initialization(self, test_db_session):
        """Test retry service initialization."""
        service = NotificationRetryService(db_session=test_db_session)
        assert service.db_session == test_db_session
        assert service.default_retry_config is not None

    def test_schedule_retry(self, test_db_session, sample_notification):
        """Test scheduling a notification for retry."""
        # Set notification to failed status
        sample_notification.status = NotificationStatus.FAILED
        sample_notification.retry_count = 1
        test_db_session.commit()
        
        service = NotificationRetryService(db_session=test_db_session)
        
        success = service.schedule_retry(sample_notification.notification_id)
        
        assert success is True
        
        # Refresh from database
        test_db_session.refresh(sample_notification)
        assert sample_notification.status == NotificationStatus.RETRY_PENDING
        assert sample_notification.retry_schedule is not None

    def test_move_to_dead_letter_queue(self, test_db_session, sample_notification):
        """Test moving notification to dead letter queue."""
        service = NotificationRetryService(db_session=test_db_session)
        
        success = service.move_to_dead_letter_queue(
            sample_notification.notification_id,
            "Exceeded max retries"
        )
        
        assert success is True
        
        # Refresh from database
        test_db_session.refresh(sample_notification)
        assert sample_notification.status == NotificationStatus.CANCELLED
        assert "dead_letter_queue" in sample_notification.delivery_metadata

    def test_retry_statistics(self, test_db_session, sample_notification):
        """Test getting retry statistics."""
        # Create some retry data
        sample_notification.retry_count = 2
        sample_notification.status = NotificationStatus.DELIVERED
        test_db_session.commit()
        
        service = NotificationRetryService(db_session=test_db_session)
        
        stats = service.get_retry_statistics()
        
        assert stats.total_retry_attempts == 2
        assert stats.successful_retries == 1
        assert stats.failed_retries == 0


class TestNotificationMonitoring:
    """Test the notification monitoring service."""

    def test_monitoring_service_initialization(self, test_db_session):
        """Test monitoring service initialization."""
        service = NotificationMonitoringService(db_session=test_db_session)
        assert service.db_session == test_db_session

    def test_system_metrics(self, test_db_session, sample_notification):
        """Test getting system metrics."""
        # Set up some test data
        sample_notification.status = NotificationStatus.DELIVERED
        sample_notification.sent_at = datetime.utcnow()
        sample_notification.delivered_at = datetime.utcnow()
        test_db_session.commit()
        
        service = NotificationMonitoringService(db_session=test_db_session)
        
        metrics = service.get_system_metrics()
        
        assert metrics.total_notifications == 1
        assert metrics.successful_notifications == 1
        assert metrics.success_rate == 100.0

    def test_system_health_check(self, test_db_session):
        """Test system health check."""
        service = NotificationMonitoringService(db_session=test_db_session)
        
        health = service.check_system_health()
        
        assert "status" in health
        assert "timestamp" in health
        assert "metrics" in health
        assert health["status"] in ["healthy", "degraded", "unhealthy", "unknown"]


class TestNotificationEventIntegration:
    """Test the notification event integration."""

    @patch('src.services.notification_event_integration.get_event_manager')
    def test_event_listener_initialization(self, mock_get_event_manager):
        """Test event listener initialization."""
        mock_event_manager = Mock()
        mock_get_event_manager.return_value = mock_event_manager
        
        listener = NotificationEventListener()
        
        assert listener.listener_id == "notification_event_listener"
        assert len(listener.event_mapping) > 0

    @patch('src.services.notification_event_integration.get_event_manager')
    def test_integration_start_stop(self, mock_get_event_manager):
        """Test starting and stopping integration."""
        mock_event_manager = Mock()
        mock_get_event_manager.return_value = mock_event_manager
        
        integration = NotificationEventIntegration()
        
        # Test start
        success = integration.start()
        assert success is True
        assert integration.is_active is True
        
        # Test stop
        success = integration.stop()
        assert success is True
        assert integration.is_active is False

    @patch('src.services.notification_event_integration.get_event_manager')
    def test_integration_status(self, mock_get_event_manager):
        """Test getting integration status."""
        mock_event_manager = Mock()
        mock_get_event_manager.return_value = mock_event_manager
        
        integration = NotificationEventIntegration()
        integration.start()
        
        status = integration.get_status()
        
        assert "is_active" in status
        assert "listener_registered" in status
        assert "event_mappings" in status
        assert status["is_active"] is True


class TestNotificationAPI:
    """Test notification API endpoints."""

    def test_create_config_request_validation(self):
        """Test API request validation."""
        from ..api.notifications import CreateNotificationConfigRequest
        
        # Valid request
        request = CreateNotificationConfigRequest(
            name="Test Config",
            notification_type="webhook",
            event_triggers=["processing_completed"],
            target_address="https://example.com/webhook"
        )
        
        assert request.name == "Test Config"
        assert request.notification_type == "webhook"

    def test_invalid_email_validation(self):
        """Test email validation in API request."""
        from ..api.notifications import CreateNotificationConfigRequest
        
        with pytest.raises(ValueError):
            request = CreateNotificationConfigRequest(
                name="Test Config",
                notification_type="email",
                event_triggers=["processing_completed"],
                target_address="invalid-email"
            )


class TestIntegrationScenarios:
    """Test end-to-end integration scenarios."""

    @patch('src.services.notification_service.WebhookClient')
    def test_full_notification_flow(self, mock_webhook_client, test_db_session):
        """Test complete notification flow from trigger to delivery."""
        # Mock webhook client
        mock_response = WebhookResponse(
            status=WebhookStatus.SUCCESS,
            status_code=200,
            response_text="OK"
        )
        mock_client_instance = Mock()
        mock_client_instance.send.return_value = mock_response
        mock_client_instance.__enter__.return_value = mock_client_instance
        mock_client_instance.__exit__.return_value = None
        mock_webhook_client.return_value = mock_client_instance
        
        # Create configuration
        service = NotificationService(db_session=test_db_session)
        
        config_request = NotificationCreateRequest(
            name="Integration Test Config",
            notification_type=NotificationType.WEBHOOK,
            event_triggers=[NotificationEvent.PROCESSING_COMPLETED],
            target_address="https://example.com/webhook"
        )
        
        config = service.create_notification_config(config_request)
        assert config is not None
        
        # Trigger notification
        trigger_request = NotificationTriggerRequest(
            event_type=NotificationEvent.PROCESSING_COMPLETED,
            event_source="video_123",
            event_metadata={"video_id": 123}
        )
        
        notifications = service.trigger_notification(trigger_request)
        assert len(notifications) == 1
        
        notification = notifications[0]
        assert notification.status == NotificationStatus.PENDING
        
        # Send notification
        result = service.send_notification(notification.notification_id)
        assert result.is_success is True
        
        # Verify notification updated
        test_db_session.refresh(notification)
        assert notification.status == NotificationStatus.DELIVERED

    def test_retry_scenario(self, test_db_session, sample_notification_config):
        """Test notification retry scenario."""
        # Create failed notification
        notification = Notification(
            notification_id="retry_test",
            config_id=sample_notification_config.id,
            event_type=NotificationEvent.PROCESSING_COMPLETED,
            status=NotificationStatus.FAILED,
            priority=NotificationPriority.NORMAL,
            target_address="https://example.com/webhook",
            message="Test message",
            retry_count=1,
            max_retries=3,
            error_info="Connection timeout"
        )
        test_db_session.add(notification)
        test_db_session.commit()
        
        # Test retry service
        retry_service = NotificationRetryService(db_session=test_db_session)
        
        # Schedule retry
        success = retry_service.schedule_retry(notification.notification_id)
        assert success is True
        
        # Verify status updated
        test_db_session.refresh(notification)
        assert notification.status == NotificationStatus.RETRY_PENDING

    def test_monitoring_scenario(self, test_db_session):
        """Test monitoring scenario with multiple notifications."""
        # Create test data
        configs = []
        notifications = []
        
        for i in range(3):
            config = NotificationConfig(
                config_id=f"test_config_{i}",
                name=f"Test Config {i}",
                notification_type=NotificationType.WEBHOOK,
                event_triggers=[NotificationEvent.PROCESSING_COMPLETED.value],
                target_address=f"https://example{i}.com/webhook"
            )
            test_db_session.add(config)
            test_db_session.flush()  # Get ID
            configs.append(config)
            
            for j in range(2):
                notification = Notification(
                    notification_id=f"test_notif_{i}_{j}",
                    config_id=config.id,
                    event_type=NotificationEvent.PROCESSING_COMPLETED,
                    status=NotificationStatus.DELIVERED if j == 0 else NotificationStatus.FAILED,
                    priority=NotificationPriority.NORMAL,
                    target_address=config.target_address,
                    message=f"Test message {i}_{j}"
                )
                test_db_session.add(notification)
                notifications.append(notification)
        
        test_db_session.commit()
        
        # Test monitoring
        monitoring_service = NotificationMonitoringService(db_session=test_db_session)
        
        # System metrics
        metrics = monitoring_service.get_system_metrics()
        assert metrics.total_notifications == 6
        assert metrics.successful_notifications == 3
        assert metrics.failed_notifications == 3
        assert metrics.success_rate == 50.0
        
        # Configuration metrics
        config_metrics = monitoring_service.get_configuration_metrics()
        assert len(config_metrics) == 3
        
        for metric in config_metrics:
            assert metric.total_notifications == 2
            assert metric.successful_notifications == 1
            assert metric.failed_notifications == 1
            assert metric.success_rate == 50.0
        
        # Health check
        health = monitoring_service.check_system_health()
        assert health["status"] in ["healthy", "degraded", "unhealthy"]


# Convenience test runners

def run_all_tests():
    """Run all notification system tests."""
    pytest.main([__file__, "-v"])


def run_specific_test(test_class_name: str):
    """Run tests for a specific class."""
    pytest.main([f"{__file__}::{test_class_name}", "-v"])


if __name__ == "__main__":
    run_all_tests()