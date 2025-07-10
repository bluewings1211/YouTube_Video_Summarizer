"""
Notification service for YouTube Summarizer application.
Provides comprehensive notification management functionality.
"""

import asyncio
import logging
import uuid
import json
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Union, Tuple
from dataclasses import dataclass
from sqlalchemy.orm import Session, selectinload, joinedload
from sqlalchemy import select, update, delete, func, and_, or_, desc
from sqlalchemy.exc import SQLAlchemyError, IntegrityError

from ..database.connection import get_db_session
from ..database.notification_models import (
    NotificationConfig, Notification, NotificationLog, WebhookEndpoint,
    NotificationType, NotificationEvent, NotificationStatus, NotificationPriority
)
from ..database.models import Video
from ..database.batch_models import Batch, BatchItem
from ..database.status_models import ProcessingStatus
from ..database.exceptions import (
    DatabaseError, DatabaseConnectionError, DatabaseQueryError,
    classify_database_error, is_recoverable_error, should_retry_operation
)
from ..database.transaction_manager import TransactionManager, managed_transaction
from ..utils.error_messages import ErrorMessages

logger = logging.getLogger(__name__)


@dataclass
class NotificationCreateRequest:
    """Request data for creating a notification configuration."""
    name: str
    notification_type: NotificationType
    event_triggers: List[NotificationEvent]
    target_address: str
    description: Optional[str] = None
    user_id: Optional[str] = None
    priority: NotificationPriority = NotificationPriority.NORMAL
    rate_limit_per_hour: Optional[int] = None
    rate_limit_per_day: Optional[int] = None
    template_config: Optional[Dict[str, Any]] = None
    filter_conditions: Optional[Dict[str, Any]] = None
    retry_config: Optional[Dict[str, Any]] = None
    auth_config: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class NotificationTriggerRequest:
    """Request data for triggering a notification."""
    event_type: NotificationEvent
    event_source: Optional[str] = None
    event_metadata: Optional[Dict[str, Any]] = None
    priority: Optional[NotificationPriority] = None
    scheduled_at: Optional[datetime] = None
    custom_message: Optional[str] = None
    custom_subject: Optional[str] = None


@dataclass
class NotificationDeliveryResult:
    """Result of notification delivery attempt."""
    notification_id: str
    status: NotificationStatus
    delivered_at: Optional[datetime] = None
    error_message: Optional[str] = None
    response_data: Optional[Dict[str, Any]] = None
    delivery_time_ms: Optional[int] = None


@dataclass
class NotificationStats:
    """Statistics for notification performance."""
    total_notifications: int
    pending_notifications: int
    sent_notifications: int
    failed_notifications: int
    success_rate: float
    average_delivery_time_ms: Optional[float]
    rate_limited_configs: int


class NotificationServiceError(Exception):
    """Custom exception for notification service operations."""
    pass


class NotificationService:
    """
    Service class for managing notification operations.
    
    This service provides comprehensive notification functionality including:
    - Notification configuration management
    - Event-triggered notification delivery
    - Retry logic and error handling
    - Performance tracking and monitoring
    - Rate limiting and filtering
    """

    def __init__(self, db_session: Optional[Session] = None):
        """
        Initialize the NotificationService.
        
        Args:
            db_session: Optional database session. If not provided, a new session will be created.
        """
        self.db_session = db_session
        self._should_close_session = db_session is None
        self.transaction_manager = TransactionManager()

    def __enter__(self):
        """Context manager entry."""
        if self.db_session is None:
            self.db_session = get_db_session()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self._should_close_session and self.db_session:
            self.db_session.close()

    # Configuration Management

    def create_notification_config(
        self,
        request: NotificationCreateRequest
    ) -> NotificationConfig:
        """
        Create a new notification configuration.
        
        Args:
            request: Notification configuration request data
            
        Returns:
            NotificationConfig: The created configuration
            
        Raises:
            NotificationServiceError: If configuration creation fails
            ValueError: If invalid parameters are provided
        """
        try:
            # Generate unique config ID
            config_id = f"config_{uuid.uuid4().hex[:12]}"
            
            # Validate event triggers
            event_trigger_values = [event.value for event in request.event_triggers]
            
            # Create configuration
            config = NotificationConfig(
                config_id=config_id,
                name=request.name,
                description=request.description,
                user_id=request.user_id,
                notification_type=request.notification_type,
                event_triggers=event_trigger_values,
                target_address=request.target_address,
                priority=request.priority,
                rate_limit_per_hour=request.rate_limit_per_hour,
                rate_limit_per_day=request.rate_limit_per_day,
                template_config=request.template_config or {},
                filter_conditions=request.filter_conditions or {},
                retry_config=request.retry_config or {
                    "max_retries": 3,
                    "initial_delay_seconds": 60,
                    "max_delay_seconds": 3600,
                    "backoff_multiplier": 2.0
                },
                auth_config=request.auth_config,
                metadata=request.metadata or {}
            )
            
            self.db_session.add(config)
            self.db_session.commit()
            
            logger.info(f"Created notification config: {config_id}")
            return config
            
        except SQLAlchemyError as e:
            self.db_session.rollback()
            error_msg = f"Failed to create notification config: {str(e)}"
            logger.error(error_msg)
            raise NotificationServiceError(error_msg)

    def get_notification_config(self, config_id: str) -> Optional[NotificationConfig]:
        """
        Get a notification configuration by ID.
        
        Args:
            config_id: Configuration ID
            
        Returns:
            NotificationConfig: The configuration or None if not found
        """
        try:
            return self.db_session.query(NotificationConfig).filter(
                NotificationConfig.config_id == config_id
            ).first()
        except SQLAlchemyError as e:
            logger.error(f"Failed to get notification config {config_id}: {str(e)}")
            return None

    def update_notification_config(
        self,
        config_id: str,
        updates: Dict[str, Any]
    ) -> Optional[NotificationConfig]:
        """
        Update a notification configuration.
        
        Args:
            config_id: Configuration ID
            updates: Dictionary of fields to update
            
        Returns:
            NotificationConfig: The updated configuration or None if not found
        """
        try:
            config = self.get_notification_config(config_id)
            if not config:
                return None
            
            # Update allowed fields
            allowed_fields = {
                'name', 'description', 'event_triggers', 'target_address',
                'is_active', 'priority', 'rate_limit_per_hour', 'rate_limit_per_day',
                'template_config', 'filter_conditions', 'retry_config', 'auth_config',
                'metadata'
            }
            
            for field, value in updates.items():
                if field in allowed_fields and hasattr(config, field):
                    setattr(config, field, value)
            
            self.db_session.commit()
            logger.info(f"Updated notification config: {config_id}")
            return config
            
        except SQLAlchemyError as e:
            self.db_session.rollback()
            logger.error(f"Failed to update notification config {config_id}: {str(e)}")
            return None

    def delete_notification_config(self, config_id: str) -> bool:
        """
        Delete a notification configuration.
        
        Args:
            config_id: Configuration ID
            
        Returns:
            bool: True if deleted, False if not found
        """
        try:
            config = self.get_notification_config(config_id)
            if not config:
                return False
            
            self.db_session.delete(config)
            self.db_session.commit()
            logger.info(f"Deleted notification config: {config_id}")
            return True
            
        except SQLAlchemyError as e:
            self.db_session.rollback()
            logger.error(f"Failed to delete notification config {config_id}: {str(e)}")
            return False

    def list_notification_configs(
        self,
        user_id: Optional[str] = None,
        notification_type: Optional[NotificationType] = None,
        is_active: Optional[bool] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[NotificationConfig]:
        """
        List notification configurations with filtering.
        
        Args:
            user_id: Filter by user ID
            notification_type: Filter by notification type
            is_active: Filter by active status
            limit: Maximum number of results
            offset: Number of results to skip
            
        Returns:
            List[NotificationConfig]: List of configurations
        """
        try:
            query = self.db_session.query(NotificationConfig)
            
            if user_id is not None:
                query = query.filter(NotificationConfig.user_id == user_id)
            
            if notification_type is not None:
                query = query.filter(NotificationConfig.notification_type == notification_type)
            
            if is_active is not None:
                query = query.filter(NotificationConfig.is_active == is_active)
            
            return query.order_by(
                NotificationConfig.created_at.desc()
            ).limit(limit).offset(offset).all()
            
        except SQLAlchemyError as e:
            logger.error(f"Failed to list notification configs: {str(e)}")
            return []

    # Notification Triggering and Delivery

    def trigger_notification(
        self,
        request: NotificationTriggerRequest
    ) -> List[Notification]:
        """
        Trigger notifications for a specific event.
        
        Args:
            request: Notification trigger request
            
        Returns:
            List[Notification]: List of created notifications
        """
        try:
            # Find matching configurations
            configs = self._find_matching_configs(request.event_type, request.event_source)
            
            notifications = []
            for config in configs:
                # Check rate limits
                if self._is_rate_limited(config):
                    logger.warning(f"Config {config.config_id} is rate limited")
                    continue
                
                # Apply filters
                if not self._passes_filters(config, request):
                    logger.debug(f"Event filtered out for config {config.config_id}")
                    continue
                
                # Create notification
                notification = self._create_notification(config, request)
                if notification:
                    notifications.append(notification)
                    
                    # Update config trigger counts
                    config.trigger_count_today += 1
                    config.trigger_count_total += 1
                    config.last_triggered_at = datetime.utcnow()
            
            self.db_session.commit()
            logger.info(f"Triggered {len(notifications)} notifications for event {request.event_type.value}")
            return notifications
            
        except SQLAlchemyError as e:
            self.db_session.rollback()
            logger.error(f"Failed to trigger notifications: {str(e)}")
            return []

    def send_notification(self, notification_id: str) -> NotificationDeliveryResult:
        """
        Send a specific notification.
        
        Args:
            notification_id: Notification ID
            
        Returns:
            NotificationDeliveryResult: Delivery result
        """
        try:
            notification = self.db_session.query(Notification).filter(
                Notification.notification_id == notification_id
            ).first()
            
            if not notification:
                return NotificationDeliveryResult(
                    notification_id=notification_id,
                    status=NotificationStatus.FAILED,
                    error_message="Notification not found"
                )
            
            if not notification.is_ready_to_send:
                return NotificationDeliveryResult(
                    notification_id=notification_id,
                    status=notification.status,
                    error_message="Notification not ready to send"
                )
            
            # Update status to sending
            notification.status = NotificationStatus.SENDING
            self.db_session.commit()
            
            # Send notification based on type
            config = notification.config
            result = self._deliver_notification(notification, config)
            
            # Update notification with result
            notification.status = result.status
            notification.sent_at = datetime.utcnow()
            if result.status == NotificationStatus.DELIVERED:
                notification.delivered_at = result.delivered_at
            if result.error_message:
                notification.error_info = result.error_message
            if result.response_data:
                notification.delivery_metadata = result.response_data
            
            self.db_session.commit()
            
            # Log result
            self._log_notification_result(notification, result)
            
            return result
            
        except SQLAlchemyError as e:
            self.db_session.rollback()
            logger.error(f"Failed to send notification {notification_id}: {str(e)}")
            return NotificationDeliveryResult(
                notification_id=notification_id,
                status=NotificationStatus.FAILED,
                error_message=str(e)
            )

    def send_pending_notifications(self, limit: int = 50) -> List[NotificationDeliveryResult]:
        """
        Send pending notifications.
        
        Args:
            limit: Maximum number of notifications to process
            
        Returns:
            List[NotificationDeliveryResult]: List of delivery results
        """
        try:
            # Get pending notifications ready to send
            notifications = self.db_session.query(Notification).filter(
                and_(
                    Notification.status == NotificationStatus.PENDING,
                    Notification.scheduled_at <= datetime.utcnow()
                )
            ).order_by(
                Notification.priority.desc(),
                Notification.scheduled_at.asc()
            ).limit(limit).all()
            
            results = []
            for notification in notifications:
                result = self.send_notification(notification.notification_id)
                results.append(result)
            
            return results
            
        except SQLAlchemyError as e:
            logger.error(f"Failed to send pending notifications: {str(e)}")
            return []

    def retry_failed_notifications(self, limit: int = 20) -> List[NotificationDeliveryResult]:
        """
        Retry failed notifications that are eligible for retry.
        
        Args:
            limit: Maximum number of notifications to retry
            
        Returns:
            List[NotificationDeliveryResult]: List of retry results
        """
        try:
            # Get failed notifications that can be retried
            notifications = self.db_session.query(Notification).filter(
                and_(
                    Notification.status.in_([NotificationStatus.FAILED, NotificationStatus.RETRY_PENDING]),
                    Notification.retry_count < Notification.max_retries
                )
            ).order_by(
                Notification.priority.desc(),
                Notification.updated_at.asc()
            ).limit(limit).all()
            
            results = []
            for notification in notifications:
                # Calculate retry delay
                retry_delay = self._calculate_retry_delay(notification)
                if datetime.utcnow() < notification.updated_at + timedelta(seconds=retry_delay):
                    continue  # Not ready for retry yet
                
                # Increment retry count
                notification.retry_count += 1
                notification.status = NotificationStatus.PENDING
                
                # Send notification
                result = self.send_notification(notification.notification_id)
                results.append(result)
            
            return results
            
        except SQLAlchemyError as e:
            logger.error(f"Failed to retry notifications: {str(e)}")
            return []

    # Statistics and Monitoring

    def get_notification_stats(
        self,
        user_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> NotificationStats:
        """
        Get notification statistics.
        
        Args:
            user_id: Filter by user ID
            start_date: Start date for statistics
            end_date: End date for statistics
            
        Returns:
            NotificationStats: Statistics object
        """
        try:
            query = self.db_session.query(Notification)
            
            if user_id:
                query = query.join(NotificationConfig).filter(
                    NotificationConfig.user_id == user_id
                )
            
            if start_date:
                query = query.filter(Notification.created_at >= start_date)
            
            if end_date:
                query = query.filter(Notification.created_at <= end_date)
            
            # Get counts by status
            status_counts = query.with_entities(
                Notification.status,
                func.count(Notification.id)
            ).group_by(Notification.status).all()
            
            total_notifications = sum(count for _, count in status_counts)
            status_dict = dict(status_counts)
            
            pending = status_dict.get(NotificationStatus.PENDING, 0)
            sent = status_dict.get(NotificationStatus.SENT, 0) + status_dict.get(NotificationStatus.DELIVERED, 0)
            failed = status_dict.get(NotificationStatus.FAILED, 0)
            
            success_rate = (sent / total_notifications * 100) if total_notifications > 0 else 0.0
            
            # Get average delivery time
            avg_delivery_time = self.db_session.query(
                func.avg(
                    func.extract('epoch', Notification.delivered_at - Notification.sent_at) * 1000
                )
            ).filter(
                and_(
                    Notification.sent_at.isnot(None),
                    Notification.delivered_at.isnot(None)
                )
            ).scalar()
            
            # Count rate limited configs
            rate_limited_configs = self.db_session.query(NotificationConfig).filter(
                or_(
                    and_(
                        NotificationConfig.rate_limit_per_day.isnot(None),
                        NotificationConfig.trigger_count_today >= NotificationConfig.rate_limit_per_day
                    )
                )
            ).count()
            
            return NotificationStats(
                total_notifications=total_notifications,
                pending_notifications=pending,
                sent_notifications=sent,
                failed_notifications=failed,
                success_rate=success_rate,
                average_delivery_time_ms=avg_delivery_time,
                rate_limited_configs=rate_limited_configs
            )
            
        except SQLAlchemyError as e:
            logger.error(f"Failed to get notification stats: {str(e)}")
            return NotificationStats(
                total_notifications=0,
                pending_notifications=0,
                sent_notifications=0,
                failed_notifications=0,
                success_rate=0.0,
                average_delivery_time_ms=None,
                rate_limited_configs=0
            )

    # Private Helper Methods

    def _find_matching_configs(
        self,
        event_type: NotificationEvent,
        event_source: Optional[str] = None
    ) -> List[NotificationConfig]:
        """Find notification configurations that match the event."""
        try:
            # Query active configs that have the event in their triggers
            configs = self.db_session.query(NotificationConfig).filter(
                and_(
                    NotificationConfig.is_active == True,
                    func.json_array_length(NotificationConfig.event_triggers) > 0
                )
            ).all()
            
            # Filter configs that contain the event type
            matching_configs = []
            for config in configs:
                if event_type.value in config.event_triggers:
                    matching_configs.append(config)
            
            return matching_configs
            
        except SQLAlchemyError as e:
            logger.error(f"Failed to find matching configs: {str(e)}")
            return []

    def _is_rate_limited(self, config: NotificationConfig) -> bool:
        """Check if a configuration is rate limited."""
        return config.is_rate_limited_daily

    def _passes_filters(
        self,
        config: NotificationConfig,
        request: NotificationTriggerRequest
    ) -> bool:
        """Check if the event passes the configuration filters."""
        if not config.filter_conditions:
            return True
        
        # Implement filter logic based on config.filter_conditions
        # This is a basic implementation - can be extended for complex filtering
        filters = config.filter_conditions
        
        if 'event_source_patterns' in filters:
            patterns = filters['event_source_patterns']
            if request.event_source and patterns:
                import re
                for pattern in patterns:
                    if re.match(pattern, request.event_source):
                        return True
                return False
        
        return True

    def _create_notification(
        self,
        config: NotificationConfig,
        request: NotificationTriggerRequest
    ) -> Optional[Notification]:
        """Create a notification from config and trigger request."""
        try:
            notification_id = f"notif_{uuid.uuid4().hex[:12]}"
            
            # Build message from template or use custom message
            message = request.custom_message or self._build_message_from_template(config, request)
            subject = request.custom_subject or self._build_subject_from_template(config, request)
            
            # Build payload
            payload = {
                'event_type': request.event_type.value,
                'event_source': request.event_source,
                'event_metadata': request.event_metadata or {},
                'config_id': config.config_id,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            notification = Notification(
                notification_id=notification_id,
                config_id=config.id,
                event_type=request.event_type,
                event_source=request.event_source,
                event_metadata=request.event_metadata,
                status=NotificationStatus.PENDING,
                priority=request.priority or config.priority,
                target_address=config.target_address,
                subject=subject,
                message=message,
                payload=payload,
                scheduled_at=request.scheduled_at or datetime.utcnow(),
                max_retries=config.retry_config.get('max_retries', 3) if config.retry_config else 3
            )
            
            self.db_session.add(notification)
            return notification
            
        except Exception as e:
            logger.error(f"Failed to create notification: {str(e)}")
            return None

    def _build_message_from_template(
        self,
        config: NotificationConfig,
        request: NotificationTriggerRequest
    ) -> str:
        """Build notification message from template."""
        template_config = config.template_config or {}
        
        # Get template for the event type
        event_templates = template_config.get('event_templates', {})
        event_template = event_templates.get(request.event_type.value)
        
        if event_template and 'message' in event_template:
            template = event_template['message']
        else:
            # Default template
            template = f"Event {request.event_type.value} occurred"
            if request.event_source:
                template += f" for {request.event_source}"
        
        # Simple template substitution
        # In a production system, you might use Jinja2 or similar
        if request.event_metadata:
            for key, value in request.event_metadata.items():
                template = template.replace(f"{{{key}}}", str(value))
        
        return template

    def _build_subject_from_template(
        self,
        config: NotificationConfig,
        request: NotificationTriggerRequest
    ) -> Optional[str]:
        """Build notification subject from template."""
        template_config = config.template_config or {}
        
        # Get template for the event type
        event_templates = template_config.get('event_templates', {})
        event_template = event_templates.get(request.event_type.value)
        
        if event_template and 'subject' in event_template:
            template = event_template['subject']
        else:
            # Default subject
            template = f"YouTube Summarizer: {request.event_type.value.replace('_', ' ').title()}"
        
        # Simple template substitution
        if request.event_metadata:
            for key, value in request.event_metadata.items():
                template = template.replace(f"{{{key}}}", str(value))
        
        return template

    def _deliver_notification(
        self,
        notification: Notification,
        config: NotificationConfig
    ) -> NotificationDeliveryResult:
        """Deliver a notification based on its type."""
        try:
            start_time = datetime.utcnow()
            
            if config.notification_type == NotificationType.WEBHOOK:
                result = self._send_webhook(notification, config)
            elif config.notification_type == NotificationType.EMAIL:
                result = self._send_email(notification, config)
            else:
                result = NotificationDeliveryResult(
                    notification_id=notification.notification_id,
                    status=NotificationStatus.FAILED,
                    error_message=f"Notification type {config.notification_type.value} not implemented"
                )
            
            end_time = datetime.utcnow()
            result.delivery_time_ms = int((end_time - start_time).total_seconds() * 1000)
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to deliver notification {notification.notification_id}: {str(e)}")
            return NotificationDeliveryResult(
                notification_id=notification.notification_id,
                status=NotificationStatus.FAILED,
                error_message=str(e)
            )

    def _send_webhook(
        self,
        notification: Notification,
        config: NotificationConfig
    ) -> NotificationDeliveryResult:
        """Send webhook notification."""
        # This is a placeholder implementation
        # In a real implementation, you would use an HTTP client to send the webhook
        # For now, we'll simulate success
        return NotificationDeliveryResult(
            notification_id=notification.notification_id,
            status=NotificationStatus.DELIVERED,
            delivered_at=datetime.utcnow(),
            response_data={'status': 'success', 'message': 'Webhook delivered (simulated)'}
        )

    def _send_email(
        self,
        notification: Notification,
        config: NotificationConfig
    ) -> NotificationDeliveryResult:
        """Send email notification."""
        # This is a placeholder implementation
        # In a real implementation, you would use an email service
        # For now, we'll simulate success
        return NotificationDeliveryResult(
            notification_id=notification.notification_id,
            status=NotificationStatus.DELIVERED,
            delivered_at=datetime.utcnow(),
            response_data={'status': 'success', 'message': 'Email delivered (simulated)'}
        )

    def _calculate_retry_delay(self, notification: Notification) -> int:
        """Calculate retry delay in seconds."""
        config = notification.config
        retry_config = config.retry_config if config.retry_config else {}
        
        initial_delay = retry_config.get('initial_delay_seconds', 60)
        max_delay = retry_config.get('max_delay_seconds', 3600)
        backoff_multiplier = retry_config.get('backoff_multiplier', 2.0)
        
        delay = initial_delay * (backoff_multiplier ** (notification.retry_count - 1))
        return min(delay, max_delay)

    def _log_notification_result(
        self,
        notification: Notification,
        result: NotificationDeliveryResult
    ):
        """Log notification delivery result."""
        try:
            log_level = 'INFO' if result.status == NotificationStatus.DELIVERED else 'ERROR'
            message = f"Notification delivery {result.status.value}"
            if result.error_message:
                message += f": {result.error_message}"
            
            log_entry = NotificationLog(
                notification_id=notification.id,
                log_level=log_level,
                message=message,
                log_data={
                    'delivery_time_ms': result.delivery_time_ms,
                    'response_data': result.response_data,
                    'retry_count': notification.retry_count
                }
            )
            
            self.db_session.add(log_entry)
            self.db_session.commit()
            
        except SQLAlchemyError as e:
            logger.error(f"Failed to log notification result: {str(e)}")


# Convenience functions for common operations

def create_notification_config(
    name: str,
    notification_type: NotificationType,
    event_triggers: List[NotificationEvent],
    target_address: str,
    **kwargs
) -> Optional[NotificationConfig]:
    """
    Convenience function to create a notification configuration.
    
    Args:
        name: Configuration name
        notification_type: Type of notification
        event_triggers: List of events that trigger notifications
        target_address: Target address for notifications
        **kwargs: Additional configuration options
        
    Returns:
        NotificationConfig: Created configuration or None if failed
    """
    with NotificationService() as service:
        request = NotificationCreateRequest(
            name=name,
            notification_type=notification_type,
            event_triggers=event_triggers,
            target_address=target_address,
            **kwargs
        )
        try:
            return service.create_notification_config(request)
        except NotificationServiceError:
            return None


def trigger_event_notification(
    event_type: NotificationEvent,
    event_source: Optional[str] = None,
    event_metadata: Optional[Dict[str, Any]] = None,
    **kwargs
) -> int:
    """
    Convenience function to trigger notifications for an event.
    
    Args:
        event_type: Type of event
        event_source: Source of the event
        event_metadata: Additional event data
        **kwargs: Additional trigger options
        
    Returns:
        int: Number of notifications triggered
    """
    with NotificationService() as service:
        request = NotificationTriggerRequest(
            event_type=event_type,
            event_source=event_source,
            event_metadata=event_metadata,
            **kwargs
        )
        notifications = service.trigger_notification(request)
        return len(notifications)


def process_notification_queue(max_notifications: int = 50) -> int:
    """
    Convenience function to process pending notifications.
    
    Args:
        max_notifications: Maximum number of notifications to process
        
    Returns:
        int: Number of notifications processed
    """
    with NotificationService() as service:
        results = service.send_pending_notifications(limit=max_notifications)
        return len(results)