"""
Notification API endpoints for YouTube Summarizer application.
Provides REST API endpoints for notification configuration and management.
"""

import logging
import uuid
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Union
from fastapi import APIRouter, Depends, HTTPException, Query, Path, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from sqlalchemy.orm import Session
from enum import Enum

from ..database.connection import get_db_session
from ..database.notification_models import (
    NotificationType, NotificationEvent, NotificationStatus, NotificationPriority
)
from ..services.notification_service import (
    NotificationService, NotificationServiceError,
    NotificationCreateRequest, NotificationTriggerRequest,
    NotificationDeliveryResult, NotificationStats
)
from ..utils.error_messages import ErrorMessages

logger = logging.getLogger(__name__)

# Create FastAPI router
router = APIRouter(prefix="/api/v1/notifications", tags=["notifications"])


# Pydantic enum models for API responses
class NotificationTypeEnum(str, Enum):
    """Notification type enumeration for API responses."""
    EMAIL = "email"
    WEBHOOK = "webhook"
    SMS = "sms"
    SLACK = "slack"
    DISCORD = "discord"
    TEAMS = "teams"
    CUSTOM = "custom"


class NotificationEventEnum(str, Enum):
    """Notification event enumeration for API responses."""
    PROCESSING_STARTED = "processing_started"
    PROCESSING_COMPLETED = "processing_completed"
    PROCESSING_FAILED = "processing_failed"
    PROCESSING_CANCELLED = "processing_cancelled"
    BATCH_STARTED = "batch_started"
    BATCH_COMPLETED = "batch_completed"
    BATCH_FAILED = "batch_failed"
    BATCH_PROGRESS_UPDATE = "batch_progress_update"
    VIDEO_PROCESSED = "video_processed"
    ERROR_OCCURRED = "error_occurred"
    RETRY_EXHAUSTED = "retry_exhausted"
    SYSTEM_MAINTENANCE = "system_maintenance"
    CUSTOM_EVENT = "custom_event"


class NotificationStatusEnum(str, Enum):
    """Notification status enumeration for API responses."""
    PENDING = "pending"
    SENDING = "sending"
    SENT = "sent"
    DELIVERED = "delivered"
    FAILED = "failed"
    RETRY_PENDING = "retry_pending"
    CANCELLED = "cancelled"


class NotificationPriorityEnum(str, Enum):
    """Notification priority enumeration for API responses."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


# Request models
class CreateNotificationConfigRequest(BaseModel):
    """Request model for creating a notification configuration."""
    name: str = Field(..., description="Configuration name", min_length=1, max_length=255)
    description: Optional[str] = Field(None, description="Optional description", max_length=1000)
    user_id: Optional[str] = Field(None, description="User identifier", max_length=255)
    notification_type: NotificationTypeEnum = Field(..., description="Type of notification")
    event_triggers: List[NotificationEventEnum] = Field(..., description="Events that trigger notifications", min_items=1)
    target_address: str = Field(..., description="Target address for notifications", min_length=1, max_length=2000)
    priority: NotificationPriorityEnum = Field(NotificationPriorityEnum.NORMAL, description="Notification priority")
    rate_limit_per_hour: Optional[int] = Field(None, description="Maximum notifications per hour", ge=1, le=1000)
    rate_limit_per_day: Optional[int] = Field(None, description="Maximum notifications per day", ge=1, le=10000)
    template_config: Optional[Dict[str, Any]] = Field(None, description="Template configuration")
    filter_conditions: Optional[Dict[str, Any]] = Field(None, description="Filter conditions")
    retry_config: Optional[Dict[str, Any]] = Field(None, description="Retry configuration")
    auth_config: Optional[Dict[str, Any]] = Field(None, description="Authentication configuration")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

    @validator('target_address')
    def validate_target_address(cls, v, values):
        """Validate target address based on notification type."""
        notification_type = values.get('notification_type')
        
        if notification_type == NotificationTypeEnum.EMAIL:
            # Basic email validation
            if '@' not in v or '.' not in v.split('@')[-1]:
                raise ValueError("Invalid email address format")
        elif notification_type == NotificationTypeEnum.WEBHOOK:
            # Basic URL validation
            if not v.startswith(('http://', 'https://')):
                raise ValueError("Webhook URL must start with http:// or https://")
        
        return v

    @validator('rate_limit_per_day')
    def validate_daily_limit_vs_hourly(cls, v, values):
        """Ensure daily limit is reasonable compared to hourly limit."""
        hourly_limit = values.get('rate_limit_per_hour')
        if v is not None and hourly_limit is not None:
            if v < hourly_limit:
                raise ValueError("Daily limit cannot be less than hourly limit")
        return v


class UpdateNotificationConfigRequest(BaseModel):
    """Request model for updating a notification configuration."""
    name: Optional[str] = Field(None, description="Configuration name", min_length=1, max_length=255)
    description: Optional[str] = Field(None, description="Description", max_length=1000)
    event_triggers: Optional[List[NotificationEventEnum]] = Field(None, description="Events that trigger notifications", min_items=1)
    target_address: Optional[str] = Field(None, description="Target address", min_length=1, max_length=2000)
    is_active: Optional[bool] = Field(None, description="Whether configuration is active")
    priority: Optional[NotificationPriorityEnum] = Field(None, description="Notification priority")
    rate_limit_per_hour: Optional[int] = Field(None, description="Maximum notifications per hour", ge=1, le=1000)
    rate_limit_per_day: Optional[int] = Field(None, description="Maximum notifications per day", ge=1, le=10000)
    template_config: Optional[Dict[str, Any]] = Field(None, description="Template configuration")
    filter_conditions: Optional[Dict[str, Any]] = Field(None, description="Filter conditions")
    retry_config: Optional[Dict[str, Any]] = Field(None, description="Retry configuration")
    auth_config: Optional[Dict[str, Any]] = Field(None, description="Authentication configuration")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class TriggerNotificationRequest(BaseModel):
    """Request model for triggering notifications."""
    event_type: NotificationEventEnum = Field(..., description="Type of event")
    event_source: Optional[str] = Field(None, description="Source of the event", max_length=255)
    event_metadata: Optional[Dict[str, Any]] = Field(None, description="Additional event data")
    priority: Optional[NotificationPriorityEnum] = Field(None, description="Override priority")
    scheduled_at: Optional[datetime] = Field(None, description="When to send notification")
    custom_message: Optional[str] = Field(None, description="Custom message override", max_length=5000)
    custom_subject: Optional[str] = Field(None, description="Custom subject override", max_length=1000)


# Response models
class NotificationConfigResponse(BaseModel):
    """Response model for notification configuration."""
    id: int
    config_id: str
    name: str
    description: Optional[str]
    user_id: Optional[str]
    notification_type: str
    event_triggers: List[str]
    target_address: str
    is_active: bool
    priority: str
    rate_limit_per_hour: Optional[int]
    rate_limit_per_day: Optional[int]
    template_config: Optional[Dict[str, Any]]
    filter_conditions: Optional[Dict[str, Any]]
    retry_config: Optional[Dict[str, Any]]
    auth_config: Optional[Dict[str, Any]]
    created_at: datetime
    updated_at: datetime
    last_triggered_at: Optional[datetime]
    trigger_count_today: int
    trigger_count_total: int
    metadata: Optional[Dict[str, Any]]
    remaining_daily_quota: Optional[int]

    class Config:
        from_attributes = True

    @validator('remaining_daily_quota', pre=True, always=True)
    def calculate_remaining_quota(cls, v, values):
        """Calculate remaining daily quota."""
        rate_limit_per_day = values.get('rate_limit_per_day')
        trigger_count_today = values.get('trigger_count_today', 0)
        
        if rate_limit_per_day is not None:
            return max(0, rate_limit_per_day - trigger_count_today)
        return None


class NotificationResponse(BaseModel):
    """Response model for individual notifications."""
    id: int
    notification_id: str
    config_id: int
    event_type: str
    event_source: Optional[str]
    event_metadata: Optional[Dict[str, Any]]
    status: str
    priority: str
    target_address: str
    subject: Optional[str]
    message: str
    payload: Optional[Dict[str, Any]]
    created_at: datetime
    updated_at: datetime
    scheduled_at: datetime
    sent_at: Optional[datetime]
    delivered_at: Optional[datetime]
    retry_count: int
    max_retries: int
    retry_schedule: Optional[Dict[str, Any]]
    error_info: Optional[str]
    delivery_metadata: Optional[Dict[str, Any]]
    webhook_response: Optional[Dict[str, Any]]
    external_id: Optional[str]
    delivery_time_seconds: Optional[int]
    queue_time_seconds: Optional[int]

    class Config:
        from_attributes = True


class NotificationStatsResponse(BaseModel):
    """Response model for notification statistics."""
    total_notifications: int
    pending_notifications: int
    sent_notifications: int
    failed_notifications: int
    success_rate: float
    average_delivery_time_ms: Optional[float]
    rate_limited_configs: int


class NotificationLogResponse(BaseModel):
    """Response model for notification logs."""
    id: int
    notification_id: int
    log_level: str
    message: str
    log_data: Optional[Dict[str, Any]]
    created_at: datetime
    worker_id: Optional[str]
    execution_context: Optional[Dict[str, Any]]
    stack_trace: Optional[str]
    external_references: Optional[Dict[str, Any]]

    class Config:
        from_attributes = True


class WebhookEndpointResponse(BaseModel):
    """Response model for webhook endpoints."""
    id: int
    endpoint_id: str
    name: str
    description: Optional[str]
    url: str
    http_method: str
    auth_type: str
    auth_config: Optional[Dict[str, Any]]
    headers: Optional[Dict[str, Any]]
    timeout_seconds: int
    is_active: bool
    health_check_enabled: bool
    health_check_url: Optional[str]
    health_check_interval_minutes: Optional[int]
    last_health_check_at: Optional[datetime]
    health_status: Optional[str]
    retry_config: Optional[Dict[str, Any]]
    rate_limit_config: Optional[Dict[str, Any]]
    created_at: datetime
    updated_at: datetime
    last_used_at: Optional[datetime]
    success_count: int
    failure_count: int
    total_requests: int
    success_rate_percentage: float
    average_response_time_ms: Optional[float]
    metadata: Optional[Dict[str, Any]]
    is_healthy: bool

    class Config:
        from_attributes = True


# Dependency injection
def get_notification_service(db: Session = Depends(get_db_session)) -> NotificationService:
    """Dependency to get notification service."""
    return NotificationService(db_session=db)


# Configuration endpoints
@router.post("/configs", response_model=NotificationConfigResponse, status_code=status.HTTP_201_CREATED)
async def create_notification_config(
    request: CreateNotificationConfigRequest,
    service: NotificationService = Depends(get_notification_service)
):
    """
    Create a new notification configuration.
    
    Args:
        request: Configuration creation request
        service: Notification service dependency
        
    Returns:
        Created notification configuration
        
    Raises:
        HTTPException: If configuration creation fails
    """
    try:
        # Convert enum values to proper enum types
        event_triggers = [NotificationEvent(event.value) for event in request.event_triggers]
        notification_type = NotificationType(request.notification_type.value)
        priority = NotificationPriority(request.priority.value)
        
        # Create service request
        service_request = NotificationCreateRequest(
            name=request.name,
            description=request.description,
            user_id=request.user_id,
            notification_type=notification_type,
            event_triggers=event_triggers,
            target_address=request.target_address,
            priority=priority,
            rate_limit_per_hour=request.rate_limit_per_hour,
            rate_limit_per_day=request.rate_limit_per_day,
            template_config=request.template_config,
            filter_conditions=request.filter_conditions,
            retry_config=request.retry_config,
            auth_config=request.auth_config,
            metadata=request.metadata
        )
        
        # Create configuration
        config = service.create_notification_config(service_request)
        
        logger.info(f"Created notification config: {config.config_id}")
        return NotificationConfigResponse.from_orm(config)
        
    except NotificationServiceError as e:
        logger.error(f"Failed to create notification config: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error creating notification config: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ErrorMessages.INTERNAL_SERVER_ERROR
        )


@router.get("/configs/{config_id}", response_model=NotificationConfigResponse)
async def get_notification_config(
    config_id: str = Path(..., description="Configuration ID"),
    service: NotificationService = Depends(get_notification_service)
):
    """
    Get a notification configuration by ID.
    
    Args:
        config_id: Configuration ID
        service: Notification service dependency
        
    Returns:
        Notification configuration
        
    Raises:
        HTTPException: If configuration not found
    """
    try:
        config = service.get_notification_config(config_id)
        if not config:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Notification configuration {config_id} not found"
            )
        
        return NotificationConfigResponse.from_orm(config)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error getting notification config {config_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ErrorMessages.INTERNAL_SERVER_ERROR
        )


@router.put("/configs/{config_id}", response_model=NotificationConfigResponse)
async def update_notification_config(
    config_id: str = Path(..., description="Configuration ID"),
    request: UpdateNotificationConfigRequest = ...,
    service: NotificationService = Depends(get_notification_service)
):
    """
    Update a notification configuration.
    
    Args:
        config_id: Configuration ID
        request: Update request
        service: Notification service dependency
        
    Returns:
        Updated notification configuration
        
    Raises:
        HTTPException: If configuration not found or update fails
    """
    try:
        # Build updates dictionary
        updates = {}
        
        # Only include fields that are provided
        if request.name is not None:
            updates['name'] = request.name
        if request.description is not None:
            updates['description'] = request.description
        if request.event_triggers is not None:
            updates['event_triggers'] = [event.value for event in request.event_triggers]
        if request.target_address is not None:
            updates['target_address'] = request.target_address
        if request.is_active is not None:
            updates['is_active'] = request.is_active
        if request.priority is not None:
            updates['priority'] = NotificationPriority(request.priority.value)
        if request.rate_limit_per_hour is not None:
            updates['rate_limit_per_hour'] = request.rate_limit_per_hour
        if request.rate_limit_per_day is not None:
            updates['rate_limit_per_day'] = request.rate_limit_per_day
        if request.template_config is not None:
            updates['template_config'] = request.template_config
        if request.filter_conditions is not None:
            updates['filter_conditions'] = request.filter_conditions
        if request.retry_config is not None:
            updates['retry_config'] = request.retry_config
        if request.auth_config is not None:
            updates['auth_config'] = request.auth_config
        if request.metadata is not None:
            updates['metadata'] = request.metadata
        
        # Update configuration
        config = service.update_notification_config(config_id, updates)
        if not config:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Notification configuration {config_id} not found"
            )
        
        logger.info(f"Updated notification config: {config_id}")
        return NotificationConfigResponse.from_orm(config)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error updating notification config {config_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ErrorMessages.INTERNAL_SERVER_ERROR
        )


@router.delete("/configs/{config_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_notification_config(
    config_id: str = Path(..., description="Configuration ID"),
    service: NotificationService = Depends(get_notification_service)
):
    """
    Delete a notification configuration.
    
    Args:
        config_id: Configuration ID
        service: Notification service dependency
        
    Raises:
        HTTPException: If configuration not found
    """
    try:
        success = service.delete_notification_config(config_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Notification configuration {config_id} not found"
            )
        
        logger.info(f"Deleted notification config: {config_id}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error deleting notification config {config_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ErrorMessages.INTERNAL_SERVER_ERROR
        )


@router.get("/configs", response_model=List[NotificationConfigResponse])
async def list_notification_configs(
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    notification_type: Optional[NotificationTypeEnum] = Query(None, description="Filter by notification type"),
    is_active: Optional[bool] = Query(None, description="Filter by active status"),
    limit: int = Query(100, description="Maximum number of results", ge=1, le=1000),
    offset: int = Query(0, description="Number of results to skip", ge=0),
    service: NotificationService = Depends(get_notification_service)
):
    """
    List notification configurations with filtering.
    
    Args:
        user_id: Filter by user ID
        notification_type: Filter by notification type
        is_active: Filter by active status
        limit: Maximum number of results
        offset: Number of results to skip
        service: Notification service dependency
        
    Returns:
        List of notification configurations
    """
    try:
        # Convert enum to proper type
        type_filter = NotificationType(notification_type.value) if notification_type else None
        
        configs = service.list_notification_configs(
            user_id=user_id,
            notification_type=type_filter,
            is_active=is_active,
            limit=limit,
            offset=offset
        )
        
        return [NotificationConfigResponse.from_orm(config) for config in configs]
        
    except Exception as e:
        logger.error(f"Unexpected error listing notification configs: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ErrorMessages.INTERNAL_SERVER_ERROR
        )


# Notification triggering endpoints
@router.post("/trigger", response_model=Dict[str, Any])
async def trigger_notification(
    request: TriggerNotificationRequest,
    service: NotificationService = Depends(get_notification_service)
):
    """
    Trigger notifications for an event.
    
    Args:
        request: Trigger request
        service: Notification service dependency
        
    Returns:
        Trigger result with notification count
    """
    try:
        # Convert enum values
        event_type = NotificationEvent(request.event_type.value)
        priority = NotificationPriority(request.priority.value) if request.priority else None
        
        # Create service request
        service_request = NotificationTriggerRequest(
            event_type=event_type,
            event_source=request.event_source,
            event_metadata=request.event_metadata,
            priority=priority,
            scheduled_at=request.scheduled_at,
            custom_message=request.custom_message,
            custom_subject=request.custom_subject
        )
        
        # Trigger notifications
        notifications = service.trigger_notification(service_request)
        
        logger.info(f"Triggered {len(notifications)} notifications for event {event_type.value}")
        
        return {
            "success": True,
            "message": f"Triggered {len(notifications)} notifications",
            "notification_count": len(notifications),
            "event_type": event_type.value,
            "event_source": request.event_source,
            "triggered_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Unexpected error triggering notifications: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ErrorMessages.INTERNAL_SERVER_ERROR
        )


@router.post("/send-pending", response_model=Dict[str, Any])
async def send_pending_notifications(
    limit: int = Query(50, description="Maximum notifications to process", ge=1, le=200),
    service: NotificationService = Depends(get_notification_service)
):
    """
    Send pending notifications.
    
    Args:
        limit: Maximum number of notifications to process
        service: Notification service dependency
        
    Returns:
        Processing result
    """
    try:
        results = service.send_pending_notifications(limit=limit)
        
        success_count = len([r for r in results if r.status in ['sent', 'delivered']])
        failed_count = len([r for r in results if r.status == 'failed'])
        
        logger.info(f"Processed {len(results)} pending notifications: {success_count} success, {failed_count} failed")
        
        return {
            "success": True,
            "message": f"Processed {len(results)} notifications",
            "total_processed": len(results),
            "successful": success_count,
            "failed": failed_count,
            "processed_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Unexpected error sending pending notifications: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ErrorMessages.INTERNAL_SERVER_ERROR
        )


@router.post("/retry-failed", response_model=Dict[str, Any])
async def retry_failed_notifications(
    limit: int = Query(20, description="Maximum notifications to retry", ge=1, le=100),
    service: NotificationService = Depends(get_notification_service)
):
    """
    Retry failed notifications.
    
    Args:
        limit: Maximum number of notifications to retry
        service: Notification service dependency
        
    Returns:
        Retry result
    """
    try:
        results = service.retry_failed_notifications(limit=limit)
        
        success_count = len([r for r in results if r.status in ['sent', 'delivered']])
        failed_count = len([r for r in results if r.status == 'failed'])
        
        logger.info(f"Retried {len(results)} failed notifications: {success_count} success, {failed_count} still failed")
        
        return {
            "success": True,
            "message": f"Retried {len(results)} notifications",
            "total_retried": len(results),
            "successful": success_count,
            "still_failed": failed_count,
            "retried_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Unexpected error retrying failed notifications: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ErrorMessages.INTERNAL_SERVER_ERROR
        )


# Statistics and monitoring endpoints
@router.get("/stats", response_model=NotificationStatsResponse)
async def get_notification_stats(
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    start_date: Optional[datetime] = Query(None, description="Start date for statistics"),
    end_date: Optional[datetime] = Query(None, description="End date for statistics"),
    service: NotificationService = Depends(get_notification_service)
):
    """
    Get notification statistics.
    
    Args:
        user_id: Filter by user ID
        start_date: Start date for statistics
        end_date: End date for statistics
        service: Notification service dependency
        
    Returns:
        Notification statistics
    """
    try:
        stats = service.get_notification_stats(
            user_id=user_id,
            start_date=start_date,
            end_date=end_date
        )
        
        return NotificationStatsResponse(
            total_notifications=stats.total_notifications,
            pending_notifications=stats.pending_notifications,
            sent_notifications=stats.sent_notifications,
            failed_notifications=stats.failed_notifications,
            success_rate=stats.success_rate,
            average_delivery_time_ms=stats.average_delivery_time_ms,
            rate_limited_configs=stats.rate_limited_configs
        )
        
    except Exception as e:
        logger.error(f"Unexpected error getting notification stats: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ErrorMessages.INTERNAL_SERVER_ERROR
        )