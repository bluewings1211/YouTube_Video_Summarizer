"""
Notification retry service for managing failed notification retry logic.

This module provides advanced retry mechanisms for failed notifications including
exponential backoff, dead letter queue management, and retry statistics.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func

from .notification_service import NotificationService, NotificationServiceError
from ..database.connection import get_db_session
from ..database.notification_models import (
    Notification, NotificationConfig, NotificationStatus, NotificationPriority
)

logger = logging.getLogger(__name__)


class RetryStrategy(Enum):
    """Retry strategy options."""
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIXED_INTERVAL = "fixed_interval"
    IMMEDIATE = "immediate"


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    initial_delay_seconds: int = 60
    max_delay_seconds: int = 3600
    backoff_multiplier: float = 2.0
    max_retries: int = 3
    retry_on_rate_limit: bool = True
    retry_on_timeout: bool = True
    retry_on_connection_error: bool = True
    dead_letter_queue_enabled: bool = True
    retry_priority_boost: bool = True


@dataclass
class RetryStats:
    """Statistics for retry operations."""
    total_retry_attempts: int
    successful_retries: int
    failed_retries: int
    dead_letter_notifications: int
    average_retry_delay_seconds: float
    retry_success_rate: float
    most_retried_config_id: Optional[str]
    most_common_error: Optional[str]


@dataclass
class RetryResult:
    """Result of a retry attempt."""
    notification_id: str
    retry_attempt: int
    success: bool
    error_message: Optional[str] = None
    next_retry_at: Optional[datetime] = None
    moved_to_dead_letter: bool = False


class NotificationRetryService:
    """
    Service for managing notification retry logic and dead letter queue.
    
    This service provides advanced retry mechanisms including:
    - Exponential backoff retry logic
    - Dead letter queue for permanently failed notifications
    - Retry statistics and monitoring
    - Configurable retry strategies
    """

    def __init__(self, 
                 db_session: Optional[Session] = None,
                 notification_service: Optional[NotificationService] = None,
                 default_retry_config: Optional[RetryConfig] = None):
        """
        Initialize the notification retry service.
        
        Args:
            db_session: Optional database session
            notification_service: Optional notification service instance
            default_retry_config: Default retry configuration
        """
        self.db_session = db_session
        self._should_close_session = db_session is None
        self.notification_service = notification_service
        self.default_retry_config = default_retry_config or RetryConfig()
        self.logger = logging.getLogger(f"{__name__}.NotificationRetryService")

    def __enter__(self):
        """Context manager entry."""
        if self.db_session is None:
            self.db_session = get_db_session()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self._should_close_session and self.db_session:
            self.db_session.close()

    def process_retry_queue(self, 
                          limit: int = 50,
                          priority_filter: Optional[NotificationPriority] = None) -> List[RetryResult]:
        """
        Process notifications in the retry queue.
        
        Args:
            limit: Maximum number of notifications to process
            priority_filter: Optional priority filter
            
        Returns:
            List[RetryResult]: Results of retry attempts
        """
        try:
            # Get notifications ready for retry
            notifications = self._get_notifications_for_retry(limit, priority_filter)
            
            if not notifications:
                self.logger.debug("No notifications ready for retry")
                return []

            self.logger.info(f"Processing {len(notifications)} notifications for retry")
            
            results = []
            notification_service = self._get_notification_service()
            
            for notification in notifications:
                try:
                    result = self._retry_notification(notification, notification_service)
                    results.append(result)
                    
                    # Add small delay between retries to avoid overwhelming the system
                    time.sleep(0.1)
                    
                except Exception as e:
                    self.logger.error(f"Failed to retry notification {notification.notification_id}: {str(e)}")
                    results.append(RetryResult(
                        notification_id=notification.notification_id,
                        retry_attempt=notification.retry_count,
                        success=False,
                        error_message=f"Retry processing failed: {str(e)}"
                    ))

            return results

        except Exception as e:
            self.logger.error(f"Failed to process retry queue: {str(e)}")
            return []

    def schedule_retry(self, 
                      notification_id: str,
                      retry_config: Optional[RetryConfig] = None) -> bool:
        """
        Schedule a notification for retry.
        
        Args:
            notification_id: Notification ID to schedule
            retry_config: Optional retry configuration override
            
        Returns:
            bool: True if scheduled successfully
        """
        try:
            notification = self.db_session.query(Notification).filter(
                Notification.notification_id == notification_id
            ).first()
            
            if not notification:
                self.logger.warning(f"Notification {notification_id} not found for retry scheduling")
                return False

            if not notification.can_retry:
                self.logger.warning(f"Notification {notification_id} cannot be retried")
                return False

            config = retry_config or self._get_retry_config_for_notification(notification)
            next_retry_at = self._calculate_next_retry_time(notification, config)
            
            # Update notification for retry
            notification.status = NotificationStatus.RETRY_PENDING
            notification.scheduled_at = next_retry_at
            
            # Update retry schedule in notification
            if not notification.retry_schedule:
                notification.retry_schedule = {}
            
            notification.retry_schedule.update({
                'strategy': config.strategy.value,
                'next_retry_at': next_retry_at.isoformat(),
                'retry_count': notification.retry_count,
                'last_scheduled_at': datetime.utcnow().isoformat()
            })
            
            self.db_session.commit()
            
            self.logger.info(f"Scheduled notification {notification_id} for retry at {next_retry_at}")
            return True

        except Exception as e:
            self.db_session.rollback()
            self.logger.error(f"Failed to schedule retry for notification {notification_id}: {str(e)}")
            return False

    def move_to_dead_letter_queue(self, notification_id: str, reason: str) -> bool:
        """
        Move a notification to the dead letter queue.
        
        Args:
            notification_id: Notification ID to move
            reason: Reason for moving to dead letter queue
            
        Returns:
            bool: True if moved successfully
        """
        try:
            notification = self.db_session.query(Notification).filter(
                Notification.notification_id == notification_id
            ).first()
            
            if not notification:
                self.logger.warning(f"Notification {notification_id} not found for dead letter queue")
                return False

            # Update notification status
            notification.status = NotificationStatus.CANCELLED
            notification.error_info = f"Moved to dead letter queue: {reason}"
            
            # Update delivery metadata
            if not notification.delivery_metadata:
                notification.delivery_metadata = {}
            
            notification.delivery_metadata.update({
                'dead_letter_queue': True,
                'dead_letter_reason': reason,
                'dead_letter_at': datetime.utcnow().isoformat(),
                'final_retry_count': notification.retry_count
            })
            
            self.db_session.commit()
            
            self.logger.warning(f"Moved notification {notification_id} to dead letter queue: {reason}")
            return True

        except Exception as e:
            self.db_session.rollback()
            self.logger.error(f"Failed to move notification {notification_id} to dead letter queue: {str(e)}")
            return False

    def get_retry_statistics(self, 
                           start_date: Optional[datetime] = None,
                           end_date: Optional[datetime] = None) -> RetryStats:
        """
        Get retry statistics for monitoring and analysis.
        
        Args:
            start_date: Optional start date for statistics
            end_date: Optional end date for statistics
            
        Returns:
            RetryStats: Retry statistics
        """
        try:
            query = self.db_session.query(Notification)
            
            if start_date:
                query = query.filter(Notification.created_at >= start_date)
            
            if end_date:
                query = query.filter(Notification.created_at <= end_date)
            
            # Get all notifications with retry attempts
            notifications_with_retries = query.filter(
                Notification.retry_count > 0
            ).all()
            
            if not notifications_with_retries:
                return RetryStats(
                    total_retry_attempts=0,
                    successful_retries=0,
                    failed_retries=0,
                    dead_letter_notifications=0,
                    average_retry_delay_seconds=0.0,
                    retry_success_rate=0.0,
                    most_retried_config_id=None,
                    most_common_error=None
                )
            
            # Calculate statistics
            total_retry_attempts = sum(n.retry_count for n in notifications_with_retries)
            successful_retries = len([n for n in notifications_with_retries if n.status in [
                NotificationStatus.SENT, NotificationStatus.DELIVERED
            ]])
            failed_retries = len([n for n in notifications_with_retries if n.status == NotificationStatus.FAILED])
            dead_letter_notifications = len([n for n in notifications_with_retries if n.status == NotificationStatus.CANCELLED])
            
            # Calculate average retry delay
            retry_delays = []
            for notification in notifications_with_retries:
                if notification.retry_schedule and 'retry_delays' in notification.retry_schedule:
                    retry_delays.extend(notification.retry_schedule['retry_delays'])
            
            average_retry_delay = sum(retry_delays) / len(retry_delays) if retry_delays else 0.0
            
            # Calculate success rate
            total_retried = successful_retries + failed_retries + dead_letter_notifications
            retry_success_rate = (successful_retries / total_retried * 100) if total_retried > 0 else 0.0
            
            # Find most retried config
            config_retry_counts = {}
            for notification in notifications_with_retries:
                config_id = str(notification.config_id)
                config_retry_counts[config_id] = config_retry_counts.get(config_id, 0) + notification.retry_count
            
            most_retried_config_id = max(config_retry_counts, key=config_retry_counts.get) if config_retry_counts else None
            
            # Find most common error
            error_counts = {}
            for notification in notifications_with_retries:
                if notification.error_info:
                    # Extract main error message (first line)
                    error_key = notification.error_info.split('\n')[0][:100]
                    error_counts[error_key] = error_counts.get(error_key, 0) + 1
            
            most_common_error = max(error_counts, key=error_counts.get) if error_counts else None
            
            return RetryStats(
                total_retry_attempts=total_retry_attempts,
                successful_retries=successful_retries,
                failed_retries=failed_retries,
                dead_letter_notifications=dead_letter_notifications,
                average_retry_delay_seconds=average_retry_delay,
                retry_success_rate=retry_success_rate,
                most_retried_config_id=most_retried_config_id,
                most_common_error=most_common_error
            )

        except Exception as e:
            self.logger.error(f"Failed to get retry statistics: {str(e)}")
            return RetryStats(
                total_retry_attempts=0,
                successful_retries=0,
                failed_retries=0,
                dead_letter_notifications=0,
                average_retry_delay_seconds=0.0,
                retry_success_rate=0.0,
                most_retried_config_id=None,
                most_common_error=None
            )

    def cleanup_old_notifications(self, 
                                 older_than_days: int = 30,
                                 keep_dead_letter: bool = True) -> int:
        """
        Clean up old notifications from the database.
        
        Args:
            older_than_days: Delete notifications older than this many days
            keep_dead_letter: Whether to keep dead letter queue notifications
            
        Returns:
            int: Number of notifications deleted
        """
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=older_than_days)
            
            query = self.db_session.query(Notification).filter(
                Notification.created_at < cutoff_date
            )
            
            # Optionally keep dead letter queue notifications
            if keep_dead_letter:
                query = query.filter(
                    Notification.status != NotificationStatus.CANCELLED
                )
            
            # Only delete completed or failed notifications
            query = query.filter(
                Notification.status.in_([
                    NotificationStatus.DELIVERED,
                    NotificationStatus.FAILED,
                    NotificationStatus.CANCELLED
                ])
            )
            
            notifications_to_delete = query.all()
            count = len(notifications_to_delete)
            
            if count > 0:
                for notification in notifications_to_delete:
                    self.db_session.delete(notification)
                
                self.db_session.commit()
                self.logger.info(f"Cleaned up {count} old notifications")
            
            return count

        except Exception as e:
            self.db_session.rollback()
            self.logger.error(f"Failed to cleanup old notifications: {str(e)}")
            return 0

    # Private helper methods

    def _get_notification_service(self) -> NotificationService:
        """Get notification service instance."""
        if self.notification_service:
            return self.notification_service
        
        return NotificationService(db_session=self.db_session)

    def _get_notifications_for_retry(self, 
                                   limit: int,
                                   priority_filter: Optional[NotificationPriority] = None) -> List[Notification]:
        """Get notifications ready for retry."""
        query = self.db_session.query(Notification).filter(
            and_(
                Notification.status.in_([NotificationStatus.FAILED, NotificationStatus.RETRY_PENDING]),
                Notification.retry_count < Notification.max_retries,
                Notification.scheduled_at <= datetime.utcnow()
            )
        )
        
        if priority_filter:
            query = query.filter(Notification.priority == priority_filter)
        
        return query.order_by(
            Notification.priority.desc(),
            Notification.scheduled_at.asc()
        ).limit(limit).all()

    def _retry_notification(self, 
                          notification: Notification,
                          notification_service: NotificationService) -> RetryResult:
        """Retry a single notification."""
        try:
            retry_config = self._get_retry_config_for_notification(notification)
            
            # Check if notification should be moved to dead letter queue
            if not self._should_retry_notification(notification, retry_config):
                reason = f"Exceeded max retries ({notification.max_retries})"
                self.move_to_dead_letter_queue(notification.notification_id, reason)
                return RetryResult(
                    notification_id=notification.notification_id,
                    retry_attempt=notification.retry_count,
                    success=False,
                    error_message=reason,
                    moved_to_dead_letter=True
                )
            
            # Increment retry count
            notification.retry_count += 1
            
            # Boost priority if configured
            if retry_config.retry_priority_boost and notification.priority != NotificationPriority.URGENT:
                if notification.priority == NotificationPriority.LOW:
                    notification.priority = NotificationPriority.NORMAL
                elif notification.priority == NotificationPriority.NORMAL:
                    notification.priority = NotificationPriority.HIGH
                elif notification.priority == NotificationPriority.HIGH:
                    notification.priority = NotificationPriority.URGENT
            
            # Attempt to send the notification
            result = notification_service.send_notification(notification.notification_id)
            
            if result.is_success:
                # Successful retry
                self.logger.info(f"Successfully retried notification {notification.notification_id} (attempt {notification.retry_count})")
                return RetryResult(
                    notification_id=notification.notification_id,
                    retry_attempt=notification.retry_count,
                    success=True
                )
            else:
                # Failed retry - schedule next retry or move to dead letter queue
                if notification.retry_count >= notification.max_retries:
                    reason = f"Failed after {notification.retry_count} retry attempts: {result.error_message}"
                    self.move_to_dead_letter_queue(notification.notification_id, reason)
                    return RetryResult(
                        notification_id=notification.notification_id,
                        retry_attempt=notification.retry_count,
                        success=False,
                        error_message=reason,
                        moved_to_dead_letter=True
                    )
                else:
                    # Schedule next retry
                    next_retry_at = self._calculate_next_retry_time(notification, retry_config)
                    notification.scheduled_at = next_retry_at
                    notification.status = NotificationStatus.RETRY_PENDING
                    self.db_session.commit()
                    
                    return RetryResult(
                        notification_id=notification.notification_id,
                        retry_attempt=notification.retry_count,
                        success=False,
                        error_message=result.error_message,
                        next_retry_at=next_retry_at
                    )

        except Exception as e:
            self.logger.error(f"Failed to retry notification {notification.notification_id}: {str(e)}")
            return RetryResult(
                notification_id=notification.notification_id,
                retry_attempt=notification.retry_count,
                success=False,
                error_message=str(e)
            )

    def _get_retry_config_for_notification(self, notification: Notification) -> RetryConfig:
        """Get retry configuration for a notification."""
        # Try to get retry config from notification config
        if notification.config and notification.config.retry_config:
            retry_config_dict = notification.config.retry_config
            return RetryConfig(
                strategy=RetryStrategy(retry_config_dict.get('strategy', 'exponential_backoff')),
                initial_delay_seconds=retry_config_dict.get('initial_delay_seconds', 60),
                max_delay_seconds=retry_config_dict.get('max_delay_seconds', 3600),
                backoff_multiplier=retry_config_dict.get('backoff_multiplier', 2.0),
                max_retries=retry_config_dict.get('max_retries', 3),
                retry_on_rate_limit=retry_config_dict.get('retry_on_rate_limit', True),
                retry_on_timeout=retry_config_dict.get('retry_on_timeout', True),
                retry_on_connection_error=retry_config_dict.get('retry_on_connection_error', True),
                dead_letter_queue_enabled=retry_config_dict.get('dead_letter_queue_enabled', True),
                retry_priority_boost=retry_config_dict.get('retry_priority_boost', True)
            )
        
        return self.default_retry_config

    def _should_retry_notification(self, notification: Notification, retry_config: RetryConfig) -> bool:
        """Check if a notification should be retried."""
        if notification.retry_count >= notification.max_retries:
            return False
        
        # Check if the error type is retryable based on configuration
        if notification.error_info:
            error_info = notification.error_info.lower()
            
            if 'rate limit' in error_info and not retry_config.retry_on_rate_limit:
                return False
            
            if 'timeout' in error_info and not retry_config.retry_on_timeout:
                return False
            
            if 'connection' in error_info and not retry_config.retry_on_connection_error:
                return False
        
        return True

    def _calculate_next_retry_time(self, notification: Notification, retry_config: RetryConfig) -> datetime:
        """Calculate the next retry time for a notification."""
        if retry_config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = retry_config.initial_delay_seconds * (retry_config.backoff_multiplier ** (notification.retry_count - 1))
            delay = min(delay, retry_config.max_delay_seconds)
        elif retry_config.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = retry_config.initial_delay_seconds * notification.retry_count
            delay = min(delay, retry_config.max_delay_seconds)
        elif retry_config.strategy == RetryStrategy.FIXED_INTERVAL:
            delay = retry_config.initial_delay_seconds
        else:  # IMMEDIATE
            delay = 0
        
        return datetime.utcnow() + timedelta(seconds=delay)


# Convenience functions

def process_notification_retries(limit: int = 50) -> List[RetryResult]:
    """
    Convenience function to process notification retries.
    
    Args:
        limit: Maximum number of notifications to process
        
    Returns:
        List[RetryResult]: Results of retry attempts
    """
    with NotificationRetryService() as service:
        return service.process_retry_queue(limit=limit)


def get_retry_statistics() -> RetryStats:
    """
    Convenience function to get retry statistics.
    
    Returns:
        RetryStats: Current retry statistics
    """
    with NotificationRetryService() as service:
        return service.get_retry_statistics()


def cleanup_old_notifications(older_than_days: int = 30) -> int:
    """
    Convenience function to cleanup old notifications.
    
    Args:
        older_than_days: Delete notifications older than this many days
        
    Returns:
        int: Number of notifications deleted
    """
    with NotificationRetryService() as service:
        return service.cleanup_old_notifications(older_than_days=older_than_days)