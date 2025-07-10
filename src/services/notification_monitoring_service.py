"""
Notification monitoring service for tracking and analyzing notification performance.

This module provides comprehensive monitoring capabilities for the notification system
including metrics collection, alerting, and performance analysis.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_, desc

from ..database.connection import get_db_session
from ..database.notification_models import (
    Notification, NotificationConfig, NotificationLog,
    NotificationStatus, NotificationPriority, NotificationEvent
)

logger = logging.getLogger(__name__)


@dataclass
class NotificationMetrics:
    """Comprehensive notification metrics."""
    total_notifications: int
    successful_notifications: int
    failed_notifications: int
    pending_notifications: int
    cancelled_notifications: int
    success_rate: float
    failure_rate: float
    average_delivery_time_ms: Optional[float]
    median_delivery_time_ms: Optional[float]
    p95_delivery_time_ms: Optional[float]
    total_retries: int
    retry_success_rate: float
    active_configurations: int
    rate_limited_configurations: int
    top_error_types: List[Dict[str, Any]]
    hourly_volume: List[Dict[str, Any]]


@dataclass
class ConfigurationMetrics:
    """Metrics for individual notification configurations."""
    config_id: str
    config_name: str
    total_notifications: int
    successful_notifications: int
    failed_notifications: int
    success_rate: float
    average_delivery_time_ms: Optional[float]
    total_retries: int
    last_triggered_at: Optional[datetime]
    trigger_count_today: int
    trigger_count_total: int
    is_rate_limited: bool
    most_common_event_type: Optional[str]
    error_rate: float


class NotificationMonitoringService:
    """
    Service for monitoring notification system performance and health.
    
    Provides comprehensive metrics, alerting, and performance analysis for
    the notification system.
    """

    def __init__(self, db_session: Optional[Session] = None):
        """
        Initialize the notification monitoring service.
        
        Args:
            db_session: Optional database session
        """
        self.db_session = db_session
        self._should_close_session = db_session is None
        self.logger = logging.getLogger(f"{__name__}.NotificationMonitoringService")

    def __enter__(self):
        """Context manager entry."""
        if self.db_session is None:
            self.db_session = get_db_session()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self._should_close_session and self.db_session:
            self.db_session.close()

    def get_system_metrics(self, 
                          start_date: Optional[datetime] = None,
                          end_date: Optional[datetime] = None) -> NotificationMetrics:
        """
        Get comprehensive system-wide notification metrics.
        
        Args:
            start_date: Optional start date for metrics
            end_date: Optional end date for metrics
            
        Returns:
            NotificationMetrics: System metrics
        """
        try:
            # Default to last 24 hours if no dates provided
            if not start_date:
                start_date = datetime.utcnow() - timedelta(hours=24)
            if not end_date:
                end_date = datetime.utcnow()

            # Base query
            query = self.db_session.query(Notification).filter(
                and_(
                    Notification.created_at >= start_date,
                    Notification.created_at <= end_date
                )
            )

            notifications = query.all()

            if not notifications:
                return self._empty_metrics()

            # Calculate basic metrics
            total_notifications = len(notifications)
            successful_notifications = len([n for n in notifications if n.status in [
                NotificationStatus.SENT, NotificationStatus.DELIVERED
            ]])
            failed_notifications = len([n for n in notifications if n.status == NotificationStatus.FAILED])
            pending_notifications = len([n for n in notifications if n.status in [
                NotificationStatus.PENDING, NotificationStatus.SENDING, NotificationStatus.RETRY_PENDING
            ]])
            cancelled_notifications = len([n for n in notifications if n.status == NotificationStatus.CANCELLED])

            success_rate = (successful_notifications / total_notifications * 100) if total_notifications > 0 else 0.0
            failure_rate = (failed_notifications / total_notifications * 100) if total_notifications > 0 else 0.0

            # Calculate delivery time metrics
            delivery_times = []
            for notification in notifications:
                if notification.delivery_time_seconds is not None:
                    delivery_times.append(notification.delivery_time_seconds * 1000)  # Convert to ms

            avg_delivery_time = sum(delivery_times) / len(delivery_times) if delivery_times else None
            
            # Calculate percentiles
            sorted_times = sorted(delivery_times) if delivery_times else []
            median_delivery_time = None
            p95_delivery_time = None
            
            if sorted_times:
                median_idx = len(sorted_times) // 2
                median_delivery_time = sorted_times[median_idx]
                
                p95_idx = int(len(sorted_times) * 0.95)
                p95_delivery_time = sorted_times[min(p95_idx, len(sorted_times) - 1)]

            # Calculate retry metrics
            total_retries = sum(n.retry_count for n in notifications)
            retried_notifications = [n for n in notifications if n.retry_count > 0]
            retry_success = len([n for n in retried_notifications if n.status in [
                NotificationStatus.SENT, NotificationStatus.DELIVERED
            ]])
            retry_success_rate = (retry_success / len(retried_notifications) * 100) if retried_notifications else 0.0

            # Get configuration metrics
            active_configs = self.db_session.query(NotificationConfig).filter(
                NotificationConfig.is_active == True
            ).count()

            rate_limited_configs = self.db_session.query(NotificationConfig).filter(
                and_(
                    NotificationConfig.is_active == True,
                    NotificationConfig.rate_limit_per_day <= NotificationConfig.trigger_count_today
                )
            ).count()

            # Get top error types
            top_error_types = self._get_top_error_types(start_date, end_date)

            # Get hourly volume
            hourly_volume = self._get_hourly_volume(start_date, end_date)

            return NotificationMetrics(
                total_notifications=total_notifications,
                successful_notifications=successful_notifications,
                failed_notifications=failed_notifications,
                pending_notifications=pending_notifications,
                cancelled_notifications=cancelled_notifications,
                success_rate=success_rate,
                failure_rate=failure_rate,
                average_delivery_time_ms=avg_delivery_time,
                median_delivery_time_ms=median_delivery_time,
                p95_delivery_time_ms=p95_delivery_time,
                total_retries=total_retries,
                retry_success_rate=retry_success_rate,
                active_configurations=active_configs,
                rate_limited_configurations=rate_limited_configs,
                top_error_types=top_error_types,
                hourly_volume=hourly_volume
            )

        except Exception as e:
            self.logger.error(f"Failed to get system metrics: {str(e)}")
            return self._empty_metrics()

    def get_configuration_metrics(self, 
                                config_id: Optional[str] = None,
                                start_date: Optional[datetime] = None,
                                end_date: Optional[datetime] = None) -> List[ConfigurationMetrics]:
        """
        Get metrics for notification configurations.
        
        Args:
            config_id: Optional specific configuration ID
            start_date: Optional start date for metrics
            end_date: Optional end date for metrics
            
        Returns:
            List[ConfigurationMetrics]: Configuration metrics
        """
        try:
            # Default to last 24 hours if no dates provided
            if not start_date:
                start_date = datetime.utcnow() - timedelta(hours=24)
            if not end_date:
                end_date = datetime.utcnow()

            # Get configurations
            config_query = self.db_session.query(NotificationConfig)
            if config_id:
                config_query = config_query.filter(NotificationConfig.config_id == config_id)

            configurations = config_query.all()
            metrics_list = []

            for config in configurations:
                # Get notifications for this config in the time period
                notifications = self.db_session.query(Notification).filter(
                    and_(
                        Notification.config_id == config.id,
                        Notification.created_at >= start_date,
                        Notification.created_at <= end_date
                    )
                ).all()

                if not notifications and not config_id:
                    continue  # Skip configs with no notifications unless specifically requested

                # Calculate metrics
                total_notifications = len(notifications)
                successful_notifications = len([n for n in notifications if n.status in [
                    NotificationStatus.SENT, NotificationStatus.DELIVERED
                ]])
                failed_notifications = len([n for n in notifications if n.status == NotificationStatus.FAILED])

                success_rate = (successful_notifications / total_notifications * 100) if total_notifications > 0 else 0.0
                error_rate = (failed_notifications / total_notifications * 100) if total_notifications > 0 else 0.0

                # Calculate delivery time
                delivery_times = [
                    n.delivery_time_seconds * 1000 for n in notifications 
                    if n.delivery_time_seconds is not None
                ]
                avg_delivery_time = sum(delivery_times) / len(delivery_times) if delivery_times else None

                # Calculate total retries
                total_retries = sum(n.retry_count for n in notifications)

                # Find most common event type
                event_counts = {}
                for notification in notifications:
                    event_type = notification.event_type.value
                    event_counts[event_type] = event_counts.get(event_type, 0) + 1

                most_common_event = max(event_counts, key=event_counts.get) if event_counts else None

                # Check if rate limited
                is_rate_limited = (
                    config.rate_limit_per_day is not None and
                    config.trigger_count_today >= config.rate_limit_per_day
                )

                metrics_list.append(ConfigurationMetrics(
                    config_id=config.config_id,
                    config_name=config.name,
                    total_notifications=total_notifications,
                    successful_notifications=successful_notifications,
                    failed_notifications=failed_notifications,
                    success_rate=success_rate,
                    average_delivery_time_ms=avg_delivery_time,
                    total_retries=total_retries,
                    last_triggered_at=config.last_triggered_at,
                    trigger_count_today=config.trigger_count_today,
                    trigger_count_total=config.trigger_count_total,
                    is_rate_limited=is_rate_limited,
                    most_common_event_type=most_common_event,
                    error_rate=error_rate
                ))

            return metrics_list

        except Exception as e:
            self.logger.error(f"Failed to get configuration metrics: {str(e)}")
            return []

    def check_system_health(self) -> Dict[str, Any]:
        """
        Check the overall health of the notification system.
        
        Returns:
            Dict with health check results
        """
        try:
            now = datetime.utcnow()
            last_hour = now - timedelta(hours=1)
            last_24_hours = now - timedelta(hours=24)

            # Check recent notification volume
            recent_notifications = self.db_session.query(Notification).filter(
                Notification.created_at >= last_hour
            ).count()

            daily_notifications = self.db_session.query(Notification).filter(
                Notification.created_at >= last_24_hours
            ).count()

            # Check error rate in last hour
            recent_failed = self.db_session.query(Notification).filter(
                and_(
                    Notification.created_at >= last_hour,
                    Notification.status == NotificationStatus.FAILED
                )
            ).count()

            error_rate = (recent_failed / recent_notifications * 100) if recent_notifications > 0 else 0.0

            # Check pending notifications (potential backlog)
            pending_notifications = self.db_session.query(Notification).filter(
                Notification.status.in_([
                    NotificationStatus.PENDING,
                    NotificationStatus.RETRY_PENDING
                ])
            ).count()

            # Check stale notifications (pending for > 1 hour)
            stale_notifications = self.db_session.query(Notification).filter(
                and_(
                    Notification.status.in_([
                        NotificationStatus.PENDING,
                        NotificationStatus.RETRY_PENDING
                    ]),
                    Notification.scheduled_at <= last_hour
                )
            ).count()

            # Check active configurations
            active_configs = self.db_session.query(NotificationConfig).filter(
                NotificationConfig.is_active == True
            ).count()

            # Check rate limited configurations
            rate_limited_configs = self.db_session.query(NotificationConfig).filter(
                and_(
                    NotificationConfig.is_active == True,
                    NotificationConfig.rate_limit_per_day <= NotificationConfig.trigger_count_today
                )
            ).count()

            # Determine overall health status
            health_status = "healthy"
            issues = []

            if error_rate > 10:
                health_status = "degraded"
                issues.append(f"High error rate: {error_rate:.1f}%")

            if stale_notifications > 100:
                health_status = "degraded"
                issues.append(f"High number of stale notifications: {stale_notifications}")

            if pending_notifications > 1000:
                health_status = "degraded"
                issues.append(f"High notification backlog: {pending_notifications}")

            if error_rate > 25 or stale_notifications > 500:
                health_status = "unhealthy"

            return {
                "status": health_status,
                "timestamp": now.isoformat(),
                "issues": issues,
                "metrics": {
                    "recent_notifications_1h": recent_notifications,
                    "daily_notifications_24h": daily_notifications,
                    "error_rate_1h": error_rate,
                    "pending_notifications": pending_notifications,
                    "stale_notifications": stale_notifications,
                    "active_configurations": active_configs,
                    "rate_limited_configurations": rate_limited_configs
                }
            }

        except Exception as e:
            self.logger.error(f"Failed to check system health: {str(e)}")
            return {
                "status": "unknown",
                "timestamp": datetime.utcnow().isoformat(),
                "issues": [f"Health check failed: {str(e)}"],
                "metrics": {}
            }

    def _empty_metrics(self) -> NotificationMetrics:
        """Return empty metrics object."""
        return NotificationMetrics(
            total_notifications=0,
            successful_notifications=0,
            failed_notifications=0,
            pending_notifications=0,
            cancelled_notifications=0,
            success_rate=0.0,
            failure_rate=0.0,
            average_delivery_time_ms=None,
            median_delivery_time_ms=None,
            p95_delivery_time_ms=None,
            total_retries=0,
            retry_success_rate=0.0,
            active_configurations=0,
            rate_limited_configurations=0,
            top_error_types=[],
            hourly_volume=[]
        )

    def _get_top_error_types(self, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        """Get top error types in the time period."""
        try:
            failed_notifications = self.db_session.query(Notification).filter(
                and_(
                    Notification.created_at >= start_date,
                    Notification.created_at <= end_date,
                    Notification.status == NotificationStatus.FAILED,
                    Notification.error_info.isnot(None)
                )
            ).all()

            error_counts = {}
            for notification in failed_notifications:
                # Extract main error message (first line)
                error_key = notification.error_info.split('\n')[0][:100]
                error_counts[error_key] = error_counts.get(error_key, 0) + 1

            # Sort by count and return top 10
            sorted_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            
            return [
                {"error_message": error, "count": count}
                for error, count in sorted_errors
            ]

        except Exception as e:
            self.logger.error(f"Failed to get top error types: {str(e)}")
            return []

    def _get_hourly_volume(self, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        """Get hourly notification volume."""
        try:
            # Group notifications by hour
            hourly_data = self.db_session.query(
                func.date_trunc('hour', Notification.created_at).label('hour'),
                func.count(Notification.id).label('total'),
                func.sum(
                    func.case(
                        [(Notification.status.in_([NotificationStatus.SENT, NotificationStatus.DELIVERED]), 1)],
                        else_=0
                    )
                ).label('successful'),
                func.sum(
                    func.case(
                        [(Notification.status == NotificationStatus.FAILED, 1)],
                        else_=0
                    )
                ).label('failed')
            ).filter(
                and_(
                    Notification.created_at >= start_date,
                    Notification.created_at <= end_date
                )
            ).group_by(
                func.date_trunc('hour', Notification.created_at)
            ).order_by('hour').all()

            return [
                {
                    "hour": hour.isoformat(),
                    "total": total,
                    "successful": successful,
                    "failed": failed,
                    "success_rate": (successful / total * 100) if total > 0 else 0.0
                }
                for hour, total, successful, failed in hourly_data
            ]

        except Exception as e:
            self.logger.error(f"Failed to get hourly volume: {str(e)}")
            return []


# Convenience functions

def get_notification_system_health() -> Dict[str, Any]:
    """
    Convenience function to get notification system health.
    
    Returns:
        Dict with health check results
    """
    with NotificationMonitoringService() as service:
        return service.check_system_health()


def get_notification_metrics(hours: int = 24) -> NotificationMetrics:
    """
    Convenience function to get notification metrics.
    
    Args:
        hours: Number of hours to look back
        
    Returns:
        NotificationMetrics: System metrics
    """
    start_date = datetime.utcnow() - timedelta(hours=hours)
    with NotificationMonitoringService() as service:
        return service.get_system_metrics(start_date=start_date)