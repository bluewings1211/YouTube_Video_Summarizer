"""
Notification and webhook database models for YouTube Summarizer application.
Defines SQLAlchemy models for notification settings and webhook functionality.
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
from sqlalchemy import (
    Column, Integer, String, Text, DateTime, JSON, Float, Boolean,
    ForeignKey, UniqueConstraint, Index, Enum
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, validates
from sqlalchemy.sql import func
import enum

from .models import Base


class NotificationType(enum.Enum):
    """Notification type enumeration."""
    EMAIL = "email"
    WEBHOOK = "webhook"
    SMS = "sms"
    SLACK = "slack"
    DISCORD = "discord"
    TEAMS = "teams"
    CUSTOM = "custom"


class NotificationEvent(enum.Enum):
    """Notification event trigger enumeration."""
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


class NotificationStatus(enum.Enum):
    """Notification delivery status enumeration."""
    PENDING = "pending"
    SENDING = "sending"
    SENT = "sent"
    DELIVERED = "delivered"
    FAILED = "failed"
    RETRY_PENDING = "retry_pending"
    CANCELLED = "cancelled"


class NotificationPriority(enum.Enum):
    """Notification priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class NotificationConfig(Base):
    """
    Notification configuration model for storing user notification preferences.
    
    This model stores notification settings that define when, how, and where
    notifications should be sent for various events in the system.
    
    Attributes:
        id: Primary key
        config_id: Unique configuration identifier
        name: Human-readable configuration name
        description: Optional description of the configuration
        user_id: User identifier (external to this system)
        notification_type: Type of notification (email, webhook, etc.)
        event_triggers: JSON array of events that trigger this notification
        target_address: Destination address (email, webhook URL, etc.)
        is_active: Whether the notification configuration is active
        priority: Notification priority level
        rate_limit_per_hour: Maximum notifications per hour (null = no limit)
        rate_limit_per_day: Maximum notifications per day (null = no limit)
        template_config: JSON configuration for notification templates
        filter_conditions: JSON configuration for filtering notifications
        retry_config: JSON configuration for retry behavior
        auth_config: JSON configuration for authentication (encrypted)
        created_at: Configuration creation timestamp
        updated_at: Configuration last update timestamp
        last_triggered_at: Last time this configuration was triggered
        trigger_count_today: Number of triggers today
        trigger_count_total: Total trigger count
        metadata: Additional configuration metadata
    """
    __tablename__ = 'notification_configs'
    
    id = Column(Integer, primary_key=True, index=True)
    config_id = Column(String(255), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    user_id = Column(String(255), nullable=True, index=True)  # External user ID
    notification_type = Column(Enum(NotificationType), nullable=False)
    event_triggers = Column(JSON, nullable=False)  # Array of NotificationEvent values
    target_address = Column(String(2000), nullable=False)
    is_active = Column(Boolean, nullable=False, default=True)
    priority = Column(Enum(NotificationPriority), nullable=False, default=NotificationPriority.NORMAL)
    rate_limit_per_hour = Column(Integer, nullable=True)
    rate_limit_per_day = Column(Integer, nullable=True)
    template_config = Column(JSON, nullable=True)
    filter_conditions = Column(JSON, nullable=True)
    retry_config = Column(JSON, nullable=True)
    auth_config = Column(JSON, nullable=True)  # Encrypted authentication details
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    last_triggered_at = Column(DateTime(timezone=True), nullable=True)
    trigger_count_today = Column(Integer, nullable=False, default=0)
    trigger_count_total = Column(Integer, nullable=False, default=0)
    config_metadata = Column(JSON, nullable=True)
    
    # Relationships
    notifications = relationship("Notification", back_populates="config", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_notification_config_config_id', 'config_id'),
        Index('idx_notification_config_user_id', 'user_id'),
        Index('idx_notification_config_notification_type', 'notification_type'),
        Index('idx_notification_config_is_active', 'is_active'),
        Index('idx_notification_config_priority', 'priority'),
        Index('idx_notification_config_created_at', 'created_at'),
        Index('idx_notification_config_last_triggered_at', 'last_triggered_at'),
        Index('idx_notification_config_user_active', 'user_id', 'is_active'),
        Index('idx_notification_config_type_active', 'notification_type', 'is_active'),
        UniqueConstraint('config_id', name='uq_notification_config_config_id'),
    )
    
    @validates('config_id')
    def validate_config_id(self, key, config_id):
        """Validate configuration ID format."""
        if not config_id or not config_id.strip():
            raise ValueError("Configuration ID cannot be empty")
        return config_id.strip()
    
    @validates('name')
    def validate_name(self, key, name):
        """Validate configuration name."""
        if not name or not name.strip():
            raise ValueError("Configuration name cannot be empty")
        return name.strip()
    
    @validates('target_address')
    def validate_target_address(self, key, target_address):
        """Validate target address."""
        if not target_address or not target_address.strip():
            raise ValueError("Target address cannot be empty")
        return target_address.strip()
    
    @validates('event_triggers')
    def validate_event_triggers(self, key, event_triggers):
        """Validate event triggers."""
        if not isinstance(event_triggers, list) or len(event_triggers) == 0:
            raise ValueError("Event triggers must be a non-empty list")
        
        # Validate that all triggers are valid NotificationEvent values
        valid_events = [e.value for e in NotificationEvent]
        for trigger in event_triggers:
            if trigger not in valid_events:
                raise ValueError(f"Invalid event trigger: {trigger}")
        
        return event_triggers
    
    @validates('rate_limit_per_hour')
    def validate_rate_limit_per_hour(self, key, rate_limit_per_hour):
        """Validate hourly rate limit."""
        if rate_limit_per_hour is not None and rate_limit_per_hour <= 0:
            raise ValueError("Rate limit per hour must be positive")
        return rate_limit_per_hour
    
    @validates('rate_limit_per_day')
    def validate_rate_limit_per_day(self, key, rate_limit_per_day):
        """Validate daily rate limit."""
        if rate_limit_per_day is not None and rate_limit_per_day <= 0:
            raise ValueError("Rate limit per day must be positive")
        return rate_limit_per_day
    
    @validates('trigger_count_today')
    def validate_trigger_count_today(self, key, trigger_count_today):
        """Validate today's trigger count."""
        if trigger_count_today < 0:
            raise ValueError("Trigger count today cannot be negative")
        return trigger_count_today
    
    @validates('trigger_count_total')
    def validate_trigger_count_total(self, key, trigger_count_total):
        """Validate total trigger count."""
        if trigger_count_total < 0:
            raise ValueError("Total trigger count cannot be negative")
        return trigger_count_total
    
    @property
    def is_rate_limited_hourly(self) -> bool:
        """Check if configuration is rate limited hourly."""
        if not self.rate_limit_per_hour:
            return False
        
        # This would need to be implemented with actual hourly tracking
        # For now, return False - implementation would require additional tracking
        return False
    
    @property
    def is_rate_limited_daily(self) -> bool:
        """Check if configuration is rate limited daily."""
        if not self.rate_limit_per_day:
            return False
        
        return self.trigger_count_today >= self.rate_limit_per_day
    
    @property
    def remaining_daily_quota(self) -> Optional[int]:
        """Calculate remaining daily notification quota."""
        if not self.rate_limit_per_day:
            return None
        
        return max(0, self.rate_limit_per_day - self.trigger_count_today)
    
    def can_trigger_for_event(self, event: str) -> bool:
        """Check if configuration can trigger for a specific event."""
        return (
            self.is_active and
            event in self.event_triggers and
            not self.is_rate_limited_daily
        )
    
    def __repr__(self):
        return f"<NotificationConfig(id={self.id}, config_id='{self.config_id}', name='{self.name}', type='{self.notification_type.value}')>"


class Notification(Base):
    """
    Notification model for storing individual notification delivery records.
    
    This model tracks individual notification delivery attempts, providing
    a complete audit trail of all notifications sent through the system.
    
    Attributes:
        id: Primary key
        notification_id: Unique notification identifier
        config_id: Foreign key to NotificationConfig
        event_type: The event that triggered this notification
        event_source: Source of the event (video_id, batch_id, etc.)
        event_metadata: Additional event-specific metadata
        status: Current notification delivery status
        priority: Notification priority level
        target_address: Destination address for this notification
        subject: Notification subject/title
        message: Notification message content
        payload: Full notification payload (JSON)
        created_at: Notification creation timestamp
        updated_at: Notification last update timestamp
        scheduled_at: When notification should be sent
        sent_at: When notification was actually sent
        delivered_at: When notification was confirmed delivered
        retry_count: Number of retry attempts
        max_retries: Maximum number of retries allowed
        retry_schedule: JSON configuration for retry scheduling
        error_info: Error information if delivery failed
        delivery_metadata: Additional delivery metadata
        webhook_response: Response from webhook delivery (if applicable)
        external_id: External system identifier
    """
    __tablename__ = 'notifications'
    
    id = Column(Integer, primary_key=True, index=True)
    notification_id = Column(String(255), unique=True, nullable=False, index=True)
    config_id = Column(Integer, ForeignKey('notification_configs.id', ondelete='CASCADE'), nullable=False)
    event_type = Column(Enum(NotificationEvent), nullable=False)
    event_source = Column(String(255), nullable=True, index=True)  # video_id, batch_id, etc.
    event_config_metadata = Column(JSON, nullable=True)
    status = Column(Enum(NotificationStatus), nullable=False, default=NotificationStatus.PENDING)
    priority = Column(Enum(NotificationPriority), nullable=False, default=NotificationPriority.NORMAL)
    target_address = Column(String(2000), nullable=False)
    subject = Column(String(1000), nullable=True)
    message = Column(Text, nullable=False)
    payload = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    scheduled_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    sent_at = Column(DateTime(timezone=True), nullable=True)
    delivered_at = Column(DateTime(timezone=True), nullable=True)
    retry_count = Column(Integer, nullable=False, default=0)
    max_retries = Column(Integer, nullable=False, default=3)
    retry_schedule = Column(JSON, nullable=True)
    error_info = Column(Text, nullable=True)
    delivery_config_metadata = Column(JSON, nullable=True)
    webhook_response = Column(JSON, nullable=True)
    external_id = Column(String(255), nullable=True, index=True)
    
    # Relationships
    config = relationship("NotificationConfig", back_populates="notifications")
    
    # Indexes
    __table_args__ = (
        Index('idx_notification_notification_id', 'notification_id'),
        Index('idx_notification_config_id', 'config_id'),
        Index('idx_notification_event_type', 'event_type'),
        Index('idx_notification_event_source', 'event_source'),
        Index('idx_notification_status', 'status'),
        Index('idx_notification_priority', 'priority'),
        Index('idx_notification_created_at', 'created_at'),
        Index('idx_notification_scheduled_at', 'scheduled_at'),
        Index('idx_notification_sent_at', 'sent_at'),
        Index('idx_notification_external_id', 'external_id'),
        Index('idx_notification_status_scheduled', 'status', 'scheduled_at'),
        Index('idx_notification_config_event', 'config_id', 'event_type'),
        Index('idx_notification_source_event', 'event_source', 'event_type'),
        UniqueConstraint('notification_id', name='uq_notification_notification_id'),
    )
    
    @validates('notification_id')
    def validate_notification_id(self, key, notification_id):
        """Validate notification ID format."""
        if not notification_id or not notification_id.strip():
            raise ValueError("Notification ID cannot be empty")
        return notification_id.strip()
    
    @validates('target_address')
    def validate_target_address(self, key, target_address):
        """Validate target address."""
        if not target_address or not target_address.strip():
            raise ValueError("Target address cannot be empty")
        return target_address.strip()
    
    @validates('message')
    def validate_message(self, key, message):
        """Validate notification message."""
        if not message or not message.strip():
            raise ValueError("Notification message cannot be empty")
        return message.strip()
    
    @validates('retry_count')
    def validate_retry_count(self, key, retry_count):
        """Validate retry count."""
        if retry_count < 0:
            raise ValueError("Retry count cannot be negative")
        return retry_count
    
    @validates('max_retries')
    def validate_max_retries(self, key, max_retries):
        """Validate max retries."""
        if max_retries < 0:
            raise ValueError("Max retries cannot be negative")
        return max_retries
    
    @property
    def is_pending(self) -> bool:
        """Check if notification is pending delivery."""
        return self.status == NotificationStatus.PENDING
    
    @property
    def is_sent(self) -> bool:
        """Check if notification has been sent."""
        return self.status in [NotificationStatus.SENT, NotificationStatus.DELIVERED]
    
    @property
    def is_failed(self) -> bool:
        """Check if notification delivery failed."""
        return self.status == NotificationStatus.FAILED
    
    @property
    def can_retry(self) -> bool:
        """Check if notification can be retried."""
        return (
            self.retry_count < self.max_retries and
            self.status in [NotificationStatus.FAILED, NotificationStatus.RETRY_PENDING]
        )
    
    @property
    def is_ready_to_send(self) -> bool:
        """Check if notification is ready to be sent."""
        return (
            self.status == NotificationStatus.PENDING and
            self.scheduled_at <= datetime.utcnow()
        )
    
    @property
    def delivery_time_seconds(self) -> Optional[int]:
        """Calculate delivery time in seconds."""
        if not self.sent_at or not self.delivered_at:
            return None
        return int((self.delivered_at - self.sent_at).total_seconds())
    
    @property
    def queue_time_seconds(self) -> Optional[int]:
        """Calculate queue time in seconds."""
        if not self.sent_at:
            return None
        return int((self.sent_at - self.created_at).total_seconds())
    
    def __repr__(self):
        return f"<Notification(id={self.id}, notification_id='{self.notification_id}', event='{self.event_type.value}', status='{self.status.value}')>"


class NotificationLog(Base):
    """
    Notification log model for storing detailed notification delivery logs.
    
    This model provides detailed logging of notification delivery attempts,
    including success/failure reasons, timing information, and debug data.
    
    Attributes:
        id: Primary key
        notification_id: Foreign key to Notification
        log_level: Log level (DEBUG, INFO, WARNING, ERROR)
        message: Log message
        log_data: Additional log data (JSON)
        created_at: Log entry timestamp
        worker_id: Identifier of the worker that created this log
        execution_context: Execution context information
        stack_trace: Stack trace if error occurred
        external_references: External system references
    """
    __tablename__ = 'notification_logs'
    
    id = Column(Integer, primary_key=True, index=True)
    notification_id = Column(Integer, ForeignKey('notifications.id', ondelete='CASCADE'), nullable=False)
    log_level = Column(String(20), nullable=False, default='INFO')
    message = Column(Text, nullable=False)
    log_data = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    worker_id = Column(String(255), nullable=True)
    execution_context = Column(JSON, nullable=True)
    stack_trace = Column(Text, nullable=True)
    external_references = Column(JSON, nullable=True)
    
    # Relationships
    notification = relationship("Notification", foreign_keys=[notification_id])
    
    # Indexes
    __table_args__ = (
        Index('idx_notification_log_notification_id', 'notification_id'),
        Index('idx_notification_log_log_level', 'log_level'),
        Index('idx_notification_log_created_at', 'created_at'),
        Index('idx_notification_log_worker_id', 'worker_id'),
        Index('idx_notification_log_notification_created', 'notification_id', 'created_at'),
        Index('idx_notification_log_level_created', 'log_level', 'created_at'),
    )
    
    @validates('log_level')
    def validate_log_level(self, key, log_level):
        """Validate log level."""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if log_level not in valid_levels:
            raise ValueError(f"Log level must be one of: {valid_levels}")
        return log_level
    
    @validates('message')
    def validate_message(self, key, message):
        """Validate log message."""
        if not message or not message.strip():
            raise ValueError("Log message cannot be empty")
        return message.strip()
    
    def __repr__(self):
        return f"<NotificationLog(id={self.id}, notification_id={self.notification_id}, level='{self.log_level}', message='{self.message[:50]}...')>"


class WebhookEndpoint(Base):
    """
    Webhook endpoint model for storing webhook endpoint configurations.
    
    This model manages webhook endpoints that can be used for notifications,
    including authentication, rate limiting, and health monitoring.
    
    Attributes:
        id: Primary key
        endpoint_id: Unique endpoint identifier
        name: Human-readable endpoint name
        description: Optional endpoint description
        url: Webhook endpoint URL
        http_method: HTTP method to use (POST, PUT, PATCH)
        auth_type: Authentication type (none, basic, bearer, api_key, custom)
        auth_config: Authentication configuration (encrypted)
        headers: Default headers to include (JSON)
        timeout_seconds: Request timeout in seconds
        is_active: Whether the endpoint is active
        health_check_enabled: Whether health checks are enabled
        health_check_url: Optional health check URL
        health_check_interval_minutes: Health check interval
        last_health_check_at: Last health check timestamp
        health_status: Current health status
        retry_config: Retry configuration (JSON)
        rate_limit_config: Rate limiting configuration (JSON)
        created_at: Endpoint creation timestamp
        updated_at: Endpoint last update timestamp
        last_used_at: Last time endpoint was used
        success_count: Total successful deliveries
        failure_count: Total failed deliveries
        average_response_time_ms: Average response time in milliseconds
        metadata: Additional endpoint metadata
    """
    __tablename__ = 'webhook_endpoints'
    
    id = Column(Integer, primary_key=True, index=True)
    endpoint_id = Column(String(255), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    url = Column(String(2000), nullable=False)
    http_method = Column(String(10), nullable=False, default='POST')
    auth_type = Column(String(50), nullable=False, default='none')
    auth_config = Column(JSON, nullable=True)  # Encrypted
    headers = Column(JSON, nullable=True)
    timeout_seconds = Column(Integer, nullable=False, default=30)
    is_active = Column(Boolean, nullable=False, default=True)
    health_check_enabled = Column(Boolean, nullable=False, default=False)
    health_check_url = Column(String(2000), nullable=True)
    health_check_interval_minutes = Column(Integer, nullable=True, default=60)
    last_health_check_at = Column(DateTime(timezone=True), nullable=True)
    health_status = Column(String(50), nullable=True, default='unknown')
    retry_config = Column(JSON, nullable=True)
    rate_limit_config = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    last_used_at = Column(DateTime(timezone=True), nullable=True)
    success_count = Column(Integer, nullable=False, default=0)
    failure_count = Column(Integer, nullable=False, default=0)
    average_response_time_ms = Column(Float, nullable=True)
    config_metadata = Column(JSON, nullable=True)
    
    # Indexes
    __table_args__ = (
        Index('idx_webhook_endpoint_endpoint_id', 'endpoint_id'),
        Index('idx_webhook_endpoint_is_active', 'is_active'),
        Index('idx_webhook_endpoint_health_status', 'health_status'),
        Index('idx_webhook_endpoint_created_at', 'created_at'),
        Index('idx_webhook_endpoint_last_used_at', 'last_used_at'),
        Index('idx_webhook_endpoint_last_health_check_at', 'last_health_check_at'),
        Index('idx_webhook_endpoint_active_health', 'is_active', 'health_status'),
        UniqueConstraint('endpoint_id', name='uq_webhook_endpoint_endpoint_id'),
    )
    
    @validates('endpoint_id')
    def validate_endpoint_id(self, key, endpoint_id):
        """Validate endpoint ID format."""
        if not endpoint_id or not endpoint_id.strip():
            raise ValueError("Endpoint ID cannot be empty")
        return endpoint_id.strip()
    
    @validates('name')
    def validate_name(self, key, name):
        """Validate endpoint name."""
        if not name or not name.strip():
            raise ValueError("Endpoint name cannot be empty")
        return name.strip()
    
    @validates('url')
    def validate_url(self, key, url):
        """Validate endpoint URL."""
        if not url or not url.strip():
            raise ValueError("Endpoint URL cannot be empty")
        url = url.strip()
        if not url.startswith(('http://', 'https://')):
            raise ValueError("Endpoint URL must start with http:// or https://")
        return url
    
    @validates('http_method')
    def validate_http_method(self, key, http_method):
        """Validate HTTP method."""
        valid_methods = ['GET', 'POST', 'PUT', 'PATCH', 'DELETE']
        if http_method.upper() not in valid_methods:
            raise ValueError(f"HTTP method must be one of: {valid_methods}")
        return http_method.upper()
    
    @validates('timeout_seconds')
    def validate_timeout_seconds(self, key, timeout_seconds):
        """Validate timeout seconds."""
        if timeout_seconds <= 0 or timeout_seconds > 300:
            raise ValueError("Timeout must be between 1 and 300 seconds")
        return timeout_seconds
    
    @validates('success_count')
    def validate_success_count(self, key, success_count):
        """Validate success count."""
        if success_count < 0:
            raise ValueError("Success count cannot be negative")
        return success_count
    
    @validates('failure_count')
    def validate_failure_count(self, key, failure_count):
        """Validate failure count."""
        if failure_count < 0:
            raise ValueError("Failure count cannot be negative")
        return failure_count
    
    @property
    def total_requests(self) -> int:
        """Calculate total requests."""
        return self.success_count + self.failure_count
    
    @property
    def success_rate_percentage(self) -> float:
        """Calculate success rate percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.success_count / self.total_requests) * 100
    
    @property
    def is_healthy(self) -> bool:
        """Check if endpoint is healthy."""
        return self.health_status in ['healthy', 'unknown'] and self.is_active
    
    def __repr__(self):
        return f"<WebhookEndpoint(id={self.id}, endpoint_id='{self.endpoint_id}', name='{self.name}', url='{self.url}')>"


# Model utilities for notification system
def get_notification_model_by_name(model_name: str):
    """Get notification model class by name."""
    models = {
        'NotificationConfig': NotificationConfig,
        'Notification': Notification,
        'NotificationLog': NotificationLog,
        'WebhookEndpoint': WebhookEndpoint,
    }
    return models.get(model_name)


def get_all_notification_models():
    """Get all notification model classes."""
    return [NotificationConfig, Notification, NotificationLog, WebhookEndpoint]


def create_notification_tables(engine):
    """Create all notification tables in the database."""
    # Create notification tables
    for model in get_all_notification_models():
        model.__table__.create(bind=engine, checkfirst=True)


def drop_notification_tables(engine):
    """Drop all notification tables from the database."""
    for model in reversed(get_all_notification_models()):
        model.__table__.drop(bind=engine, checkfirst=True)