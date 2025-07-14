"""
Integration between notification system and event system.

This module provides event listeners that automatically trigger notifications
when specific events occur in the YouTube Summarizer processing workflow.
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional, Set
from datetime import datetime

from .notification_service import (
    NotificationService, NotificationTriggerRequest,
    NotificationServiceError
)
from .status_events import (
    StatusEventManager, StatusEventListener, StatusEvent, EventType,
    get_event_manager
)
from ..database.notification_models import NotificationEvent, NotificationPriority
from ..database.status_models import ProcessingStatusType
from ..database.connection import get_db_session

logger = logging.getLogger(__name__)


class NotificationEventListener(StatusEventListener):
    """
    Event listener that triggers notifications based on status events.
    
    This listener connects the status event system with the notification system,
    automatically triggering notifications when specific events occur.
    """

    def __init__(self, notification_service: Optional[NotificationService] = None):
        """
        Initialize the notification event listener.
        
        Args:
            notification_service: Optional notification service instance
        """
        self.notification_service = notification_service
        self.listener_id = "notification_event_listener"
        self.logger = logging.getLogger(f"{__name__}.NotificationEventListener")
        
        # Map status events to notification events
        self.event_mapping = {
            EventType.STATUS_CREATED: NotificationEvent.PROCESSING_STARTED,
            EventType.STATUS_COMPLETED: NotificationEvent.PROCESSING_COMPLETED,
            EventType.STATUS_FAILED: NotificationEvent.PROCESSING_FAILED,
            EventType.STATUS_CANCELLED: NotificationEvent.PROCESSING_CANCELLED,
            EventType.WORKFLOW_COMPLETED: NotificationEvent.VIDEO_PROCESSED,
            EventType.ERROR_OCCURRED: NotificationEvent.ERROR_OCCURRED,
            EventType.BATCH_CREATED: NotificationEvent.BATCH_STARTED,
            EventType.BATCH_COMPLETED: NotificationEvent.BATCH_COMPLETED,
            EventType.BATCH_ITEM_PROCESSED: NotificationEvent.BATCH_PROGRESS_UPDATE,
            EventType.STALE_STATUS_DETECTED: NotificationEvent.RETRY_EXHAUSTED,
        }
        
        # Events that should be mapped to high priority notifications
        self.high_priority_events = {
            EventType.ERROR_OCCURRED,
            EventType.STATUS_FAILED,
            EventType.STALE_STATUS_DETECTED
        }
        
        # Events that should be mapped to urgent priority notifications
        self.urgent_priority_events = {
            EventType.STATUS_CANCELLED
        }

    def get_listener_id(self) -> str:
        """Get the unique identifier for this listener."""
        return self.listener_id

    def can_handle_event(self, event: StatusEvent) -> bool:
        """
        Check if this listener can handle the given event.
        
        Args:
            event: Status event to check
            
        Returns:
            bool: True if the event can be handled
        """
        return event.event_type in self.event_mapping

    async def handle_event(self, event: StatusEvent) -> None:
        """
        Handle a status event by triggering appropriate notifications.
        
        Args:
            event: Status event to handle
        """
        try:
            # Get notification service
            notification_service = self._get_notification_service()
            if not notification_service:
                self.logger.warning("No notification service available, skipping notification")
                return

            # Map the event to notification event type
            notification_event_type = self.event_mapping.get(event.event_type)
            if not notification_event_type:
                self.logger.debug(f"No notification mapping for event type: {event.event_type}")
                return

            # Determine priority
            priority = self._determine_priority(event)

            # Build event metadata
            event_metadata = self._build_event_metadata(event)

            # Build event source identifier
            event_source = self._build_event_source(event)

            # Create trigger request
            trigger_request = NotificationTriggerRequest(
                event_type=notification_event_type,
                event_source=event_source,
                event_metadata=event_metadata,
                priority=priority
            )

            # Trigger notifications
            notifications = notification_service.trigger_notification(trigger_request)
            
            self.logger.info(
                f"Triggered {len(notifications)} notifications for event {event.event_type.value} "
                f"(status: {event.status_id})"
            )

        except Exception as e:
            self.logger.error(f"Failed to handle event {event.event_type.value}: {str(e)}")

    def _get_notification_service(self) -> Optional[NotificationService]:
        """Get notification service instance."""
        if self.notification_service:
            return self.notification_service
        
        try:
            # Create a new notification service with its own database session
            return NotificationService()
        except Exception as e:
            self.logger.error(f"Failed to create notification service: {str(e)}")
            return None

    def _determine_priority(self, event: StatusEvent) -> NotificationPriority:
        """
        Determine notification priority based on event type and properties.
        
        Args:
            event: Status event
            
        Returns:
            NotificationPriority: Determined priority
        """
        # Check for urgent events
        if event.event_type in self.urgent_priority_events:
            return NotificationPriority.URGENT
        
        # Check for high priority events
        if event.event_type in self.high_priority_events:
            return NotificationPriority.HIGH
        
        # Use event priority if available
        if hasattr(event, 'priority') and event.priority:
            priority_mapping = {
                'critical': NotificationPriority.URGENT,
                'high': NotificationPriority.HIGH,
                'normal': NotificationPriority.NORMAL,
                'low': NotificationPriority.LOW
            }
            return priority_mapping.get(event.priority.value, NotificationPriority.NORMAL)
        
        # Default to normal priority
        return NotificationPriority.NORMAL

    def _build_event_metadata(self, event: StatusEvent) -> Dict[str, Any]:
        """
        Build event metadata from status event.
        
        Args:
            event: Status event
            
        Returns:
            Dict with event metadata
        """
        metadata = {
            'event_id': event.event_id,
            'event_type': event.event_type.value,
            'timestamp': event.timestamp.isoformat(),
            'status_id': event.status_id
        }

        # Add status information
        if event.previous_status:
            metadata['previous_status'] = event.previous_status.value
        
        if event.new_status:
            metadata['new_status'] = event.new_status.value
        
        if event.progress_percentage is not None:
            metadata['progress_percentage'] = event.progress_percentage
        
        if event.current_step:
            metadata['current_step'] = event.current_step

        # Add context information
        if event.video_id:
            metadata['video_id'] = event.video_id
        
        if event.batch_item_id:
            metadata['batch_item_id'] = event.batch_item_id
        
        if event.processing_session_id:
            metadata['processing_session_id'] = event.processing_session_id
        
        if event.worker_id:
            metadata['worker_id'] = event.worker_id
        
        if event.node_name:
            metadata['node_name'] = event.node_name
        
        if event.workflow_name:
            metadata['workflow_name'] = event.workflow_name

        # Add error information if available
        if hasattr(event, 'error_message') and event.error_message:
            metadata['error_message'] = event.error_message
        
        if hasattr(event, 'error_code') and event.error_code:
            metadata['error_code'] = event.error_code

        # Add execution information if available
        if hasattr(event, 'execution_time_seconds') and event.execution_time_seconds:
            metadata['execution_time_seconds'] = event.execution_time_seconds

        return metadata

    def _build_event_source(self, event: StatusEvent) -> str:
        """
        Build event source identifier from status event.
        
        Args:
            event: Status event
            
        Returns:
            str: Event source identifier
        """
        # Prioritize video ID if available
        if event.video_id:
            return f"video_{event.video_id}"
        
        # Use batch item ID if available
        if event.batch_item_id:
            return f"batch_item_{event.batch_item_id}"
        
        # Use processing session ID if available
        if event.processing_session_id:
            return f"session_{event.processing_session_id}"
        
        # Fall back to status ID
        return f"status_{event.status_id}"


class NotificationEventIntegration:
    """
    Integration service that connects notification system with event system.
    
    This service manages the registration and lifecycle of notification event listeners.
    """

    def __init__(self, 
                 event_manager: Optional[StatusEventManager] = None,
                 notification_service: Optional[NotificationService] = None):
        """
        Initialize the notification event integration.
        
        Args:
            event_manager: Optional event manager instance
            notification_service: Optional notification service instance
        """
        self.event_manager = event_manager or get_event_manager()
        self.notification_service = notification_service
        self.listener = None
        self.is_active = False
        self.logger = logging.getLogger(f"{__name__}.NotificationEventIntegration")

    def start(self) -> bool:
        """
        Start the notification event integration.
        
        Returns:
            bool: True if started successfully
        """
        try:
            if self.is_active:
                self.logger.warning("Notification event integration is already active")
                return True

            # Create and register the listener
            self.listener = NotificationEventListener(self.notification_service)
            
            # Register with event manager
            self.event_manager.register_listener(self.listener)
            
            self.is_active = True
            self.logger.info("Notification event integration started successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to start notification event integration: {str(e)}")
            return False

    def stop(self) -> bool:
        """
        Stop the notification event integration.
        
        Returns:
            bool: True if stopped successfully
        """
        try:
            if not self.is_active:
                self.logger.warning("Notification event integration is not active")
                return True

            if self.listener:
                # Unregister the listener
                self.event_manager.unregister_listener(self.listener.get_listener_id())
                self.listener = None

            self.is_active = False
            self.logger.info("Notification event integration stopped successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to stop notification event integration: {str(e)}")
            return False

    def restart(self) -> bool:
        """
        Restart the notification event integration.
        
        Returns:
            bool: True if restarted successfully
        """
        self.stop()
        return self.start()

    def get_status(self) -> Dict[str, Any]:
        """
        Get the status of the notification event integration.
        
        Returns:
            Dict with integration status
        """
        return {
            'is_active': self.is_active,
            'listener_registered': self.listener is not None,
            'listener_id': self.listener.get_listener_id() if self.listener else None,
            'event_mappings': len(self.listener.event_mapping) if self.listener else 0,
            'high_priority_events': len(self.listener.high_priority_events) if self.listener else 0,
            'urgent_priority_events': len(self.listener.urgent_priority_events) if self.listener else 0
        }


# Global integration instance
_global_integration = None


def get_notification_event_integration() -> NotificationEventIntegration:
    """
    Get the global notification event integration instance.
    
    Returns:
        NotificationEventIntegration: Global integration instance
    """
    global _global_integration
    if _global_integration is None:
        _global_integration = NotificationEventIntegration()
    return _global_integration


def start_notification_integration() -> bool:
    """
    Start the global notification event integration.
    
    Returns:
        bool: True if started successfully
    """
    integration = get_notification_event_integration()
    return integration.start()


def stop_notification_integration() -> bool:
    """
    Stop the global notification event integration.
    
    Returns:
        bool: True if stopped successfully
    """
    integration = get_notification_event_integration()
    return integration.stop()


def restart_notification_integration() -> bool:
    """
    Restart the global notification event integration.
    
    Returns:
        bool: True if restarted successfully
    """
    integration = get_notification_event_integration()
    return integration.restart()


def get_notification_integration_status() -> Dict[str, Any]:
    """
    Get the status of the global notification event integration.
    
    Returns:
        Dict with integration status
    """
    integration = get_notification_event_integration()
    return integration.get_status()


# Context manager for temporary integration
class TemporaryNotificationIntegration:
    """Context manager for temporary notification integration."""

    def __init__(self, 
                 event_manager: Optional[StatusEventManager] = None,
                 notification_service: Optional[NotificationService] = None):
        """
        Initialize temporary integration.
        
        Args:
            event_manager: Optional event manager instance
            notification_service: Optional notification service instance
        """
        self.integration = NotificationEventIntegration(event_manager, notification_service)

    def __enter__(self) -> NotificationEventIntegration:
        """Start the integration."""
        self.integration.start()
        return self.integration

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop the integration."""
        self.integration.stop()


# Convenience function for testing
def trigger_test_notifications(
    video_id: Optional[int] = None,
    batch_item_id: Optional[int] = None
) -> Dict[str, Any]:
    """
    Trigger test notifications for development and testing.
    
    Args:
        video_id: Optional video ID for test
        batch_item_id: Optional batch item ID for test
        
    Returns:
        Dict with test results
    """
    try:
        # Create test events
        test_events = [
            StatusEvent(
                event_id=f"test_event_started_{datetime.utcnow().isoformat()}",
                event_type=EventType.WORKFLOW_STARTED,
                timestamp=datetime.utcnow(),
                status_id="test_status_1",
                video_id=video_id,
                batch_item_id=batch_item_id,
                new_status=ProcessingStatusType.STARTING
            ),
            StatusEvent(
                event_id=f"test_event_completed_{datetime.utcnow().isoformat()}",
                event_type=EventType.WORKFLOW_COMPLETED,
                timestamp=datetime.utcnow(),
                status_id="test_status_2",
                video_id=video_id,
                batch_item_id=batch_item_id,
                new_status=ProcessingStatusType.COMPLETED,
                progress_percentage=100.0
            )
        ]

        # Get integration and trigger events
        integration = get_notification_event_integration()
        if not integration.is_active:
            integration.start()

        results = []
        for event in test_events:
            try:
                asyncio.run(integration.listener.handle_event(event))
                results.append({
                    'event_type': event.event_type.value,
                    'status': 'success'
                })
            except Exception as e:
                results.append({
                    'event_type': event.event_type.value,
                    'status': 'failed',
                    'error': str(e)
                })

        return {
            'success': True,
            'test_events_processed': len(test_events),
            'results': results
        }

    except Exception as e:
        logger.error(f"Failed to trigger test notifications: {str(e)}")
        return {
            'success': False,
            'error': str(e),
            'test_events_processed': 0,
            'results': []
        }