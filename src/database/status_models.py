"""
Status tracking database models for YouTube Summarizer application.
Defines SQLAlchemy models for comprehensive status tracking functionality.
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


class ProcessingStatusType(enum.Enum):
    """Processing status type enumeration."""
    QUEUED = "queued"
    STARTING = "starting"
    YOUTUBE_METADATA = "youtube_metadata"
    TRANSCRIPT_EXTRACTION = "transcript_extraction"
    LANGUAGE_DETECTION = "language_detection"
    SUMMARY_GENERATION = "summary_generation"
    KEYWORD_EXTRACTION = "keyword_extraction"
    TIMESTAMPED_SEGMENTS = "timestamped_segments"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRY_PENDING = "retry_pending"


class ProcessingPriority(enum.Enum):
    """Processing priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class StatusChangeType(enum.Enum):
    """Status change event types."""
    STATUS_UPDATE = "status_update"
    PROGRESS_UPDATE = "progress_update"
    ERROR_OCCURRED = "error_occurred"
    RETRY_SCHEDULED = "retry_scheduled"
    MANUAL_INTERVENTION = "manual_intervention"
    SYSTEM_EVENT = "system_event"


class ProcessingStatus(Base):
    """
    Processing status model for tracking detailed processing status.
    
    This model provides comprehensive status tracking for video processing,
    extending beyond the basic ProcessingSession model to include detailed
    step-by-step tracking, progress monitoring, and status history.
    
    Attributes:
        id: Primary key
        status_id: Unique status identifier
        video_id: Foreign key to Video (nullable if video not yet created)
        batch_item_id: Foreign key to BatchItem (nullable for standalone processing)
        processing_session_id: Foreign key to ProcessingSession (nullable)
        status: Current processing status
        substatus: Detailed substatus information
        priority: Processing priority
        progress_percentage: Current progress percentage (0-100)
        current_step: Current processing step description
        total_steps: Total number of steps in the process
        completed_steps: Number of completed steps
        estimated_completion_time: Estimated completion timestamp
        created_at: Status creation timestamp
        updated_at: Status last update timestamp
        started_at: Processing start timestamp
        completed_at: Processing completion timestamp
        worker_id: Identifier of the worker processing this item
        heartbeat_at: Last heartbeat timestamp
        retry_count: Number of retry attempts
        max_retries: Maximum number of retries allowed
        processing_metadata: Additional processing metadata
        result_metadata: Processing result metadata
        error_info: Error information if processing failed
        tags: JSON array of tags for categorization
        external_id: External identifier for integration
    """
    __tablename__ = 'processing_status'
    
    id = Column(Integer, primary_key=True, index=True)
    status_id = Column(String(255), unique=True, nullable=False, index=True)
    video_id = Column(Integer, ForeignKey('videos.id', ondelete='CASCADE'), nullable=True)
    batch_item_id = Column(Integer, ForeignKey('batch_items.id', ondelete='SET NULL'), nullable=True)
    processing_session_id = Column(Integer, ForeignKey('processing_sessions.id', ondelete='SET NULL'), nullable=True)
    status = Column(Enum(ProcessingStatusType), nullable=False, default=ProcessingStatusType.QUEUED)
    substatus = Column(String(255), nullable=True)
    priority = Column(Enum(ProcessingPriority), nullable=False, default=ProcessingPriority.NORMAL)
    progress_percentage = Column(Float, nullable=False, default=0.0)
    current_step = Column(String(255), nullable=True)
    total_steps = Column(Integer, nullable=True)
    completed_steps = Column(Integer, nullable=False, default=0)
    estimated_completion_time = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    worker_id = Column(String(255), nullable=True)
    heartbeat_at = Column(DateTime(timezone=True), nullable=True)
    retry_count = Column(Integer, nullable=False, default=0)
    max_retries = Column(Integer, nullable=False, default=3)
    processing_metadata = Column(JSON, nullable=True)
    result_metadata = Column(JSON, nullable=True)
    error_info = Column(Text, nullable=True)
    tags = Column(JSON, nullable=True)
    external_id = Column(String(255), nullable=True, index=True)
    
    # Relationships
    video = relationship("Video", foreign_keys=[video_id])
    batch_item = relationship("BatchItem", foreign_keys=[batch_item_id])
    processing_session = relationship("ProcessingSession", foreign_keys=[processing_session_id])
    status_history = relationship("StatusHistory", back_populates="processing_status", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_processing_status_status_id', 'status_id'),
        Index('idx_processing_status_video_id', 'video_id'),
        Index('idx_processing_status_batch_item_id', 'batch_item_id'),
        Index('idx_processing_status_processing_session_id', 'processing_session_id'),
        Index('idx_processing_status_status', 'status'),
        Index('idx_processing_status_priority', 'priority'),
        Index('idx_processing_status_created_at', 'created_at'),
        Index('idx_processing_status_updated_at', 'updated_at'),
        Index('idx_processing_status_heartbeat_at', 'heartbeat_at'),
        Index('idx_processing_status_worker_id', 'worker_id'),
        Index('idx_processing_status_external_id', 'external_id'),
        Index('idx_processing_status_status_priority', 'status', 'priority'),
        Index('idx_processing_status_status_updated', 'status', 'updated_at'),
        Index('idx_processing_status_progress', 'progress_percentage'),
        UniqueConstraint('status_id', name='uq_processing_status_status_id'),
    )
    
    @validates('status_id')
    def validate_status_id(self, key, status_id):
        """Validate status ID format."""
        if not status_id or not status_id.strip():
            raise ValueError("Status ID cannot be empty")
        return status_id.strip()
    
    @validates('progress_percentage')
    def validate_progress_percentage(self, key, progress_percentage):
        """Validate progress percentage."""
        if progress_percentage < 0 or progress_percentage > 100:
            raise ValueError("Progress percentage must be between 0 and 100")
        return progress_percentage
    
    @validates('completed_steps')
    def validate_completed_steps(self, key, completed_steps):
        """Validate completed steps."""
        if completed_steps < 0:
            raise ValueError("Completed steps cannot be negative")
        return completed_steps
    
    @validates('total_steps')
    def validate_total_steps(self, key, total_steps):
        """Validate total steps."""
        if total_steps is not None and total_steps <= 0:
            raise ValueError("Total steps must be positive")
        return total_steps
    
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
    def is_active(self) -> bool:
        """Check if processing is currently active."""
        return self.status in [
            ProcessingStatusType.STARTING,
            ProcessingStatusType.YOUTUBE_METADATA,
            ProcessingStatusType.TRANSCRIPT_EXTRACTION,
            ProcessingStatusType.LANGUAGE_DETECTION,
            ProcessingStatusType.SUMMARY_GENERATION,
            ProcessingStatusType.KEYWORD_EXTRACTION,
            ProcessingStatusType.TIMESTAMPED_SEGMENTS,
            ProcessingStatusType.FINALIZING
        ]
    
    @property
    def is_completed(self) -> bool:
        """Check if processing is completed."""
        return self.status == ProcessingStatusType.COMPLETED
    
    @property
    def is_failed(self) -> bool:
        """Check if processing has failed."""
        return self.status == ProcessingStatusType.FAILED
    
    @property
    def is_cancelled(self) -> bool:
        """Check if processing was cancelled."""
        return self.status == ProcessingStatusType.CANCELLED
    
    @property
    def can_retry(self) -> bool:
        """Check if processing can be retried."""
        return (
            self.retry_count < self.max_retries and
            self.status in [ProcessingStatusType.FAILED, ProcessingStatusType.RETRY_PENDING]
        )
    
    @property
    def is_stale(self, timeout_seconds: int = 300) -> bool:
        """Check if status is stale (no heartbeat for timeout period)."""
        if not self.heartbeat_at or not self.is_active:
            return False
        return (datetime.utcnow() - self.heartbeat_at).total_seconds() > timeout_seconds
    
    @property
    def step_progress_percentage(self) -> float:
        """Calculate step-based progress percentage."""
        if not self.total_steps or self.total_steps == 0:
            return self.progress_percentage
        return (self.completed_steps / self.total_steps) * 100
    
    @property
    def estimated_remaining_time_seconds(self) -> Optional[int]:
        """Calculate estimated remaining time in seconds."""
        if not self.estimated_completion_time:
            return None
        remaining = self.estimated_completion_time - datetime.utcnow()
        return max(0, int(remaining.total_seconds()))
    
    def __repr__(self):
        return f"<ProcessingStatus(id={self.id}, status_id='{self.status_id}', status='{self.status.value}', progress={self.progress_percentage:.1f}%)>"


class StatusHistory(Base):
    """
    Status history model for tracking all status changes and events.
    
    This model provides a complete audit trail of all status changes,
    allowing for detailed analysis of processing patterns and debugging.
    
    Attributes:
        id: Primary key
        processing_status_id: Foreign key to ProcessingStatus
        change_type: Type of status change event
        previous_status: Previous status value
        new_status: New status value
        previous_progress: Previous progress percentage
        new_progress: New progress percentage
        change_reason: Reason for the status change
        change_metadata: Additional metadata about the change
        worker_id: Identifier of the worker that made the change
        created_at: Change timestamp
        duration_seconds: Duration in this status (calculated)
        error_info: Error information if change was due to error
        external_trigger: External system that triggered the change
    """
    __tablename__ = 'status_history'
    
    id = Column(Integer, primary_key=True, index=True)
    processing_status_id = Column(Integer, ForeignKey('processing_status.id', ondelete='CASCADE'), nullable=False)
    change_type = Column(Enum(StatusChangeType), nullable=False)
    previous_status = Column(Enum(ProcessingStatusType), nullable=True)
    new_status = Column(Enum(ProcessingStatusType), nullable=False)
    previous_progress = Column(Float, nullable=True)
    new_progress = Column(Float, nullable=False)
    change_reason = Column(String(500), nullable=True)
    change_metadata = Column(JSON, nullable=True)
    worker_id = Column(String(255), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    duration_seconds = Column(Integer, nullable=True)
    error_info = Column(Text, nullable=True)
    external_trigger = Column(String(255), nullable=True)
    
    # Relationships
    processing_status = relationship("ProcessingStatus", back_populates="status_history")
    
    # Indexes
    __table_args__ = (
        Index('idx_status_history_processing_status_id', 'processing_status_id'),
        Index('idx_status_history_change_type', 'change_type'),
        Index('idx_status_history_new_status', 'new_status'),
        Index('idx_status_history_created_at', 'created_at'),
        Index('idx_status_history_worker_id', 'worker_id'),
        Index('idx_status_history_external_trigger', 'external_trigger'),
        Index('idx_status_history_status_created', 'processing_status_id', 'created_at'),
        Index('idx_status_history_status_type', 'new_status', 'change_type'),
    )
    
    @validates('new_progress')
    def validate_new_progress(self, key, new_progress):
        """Validate new progress percentage."""
        if new_progress < 0 or new_progress > 100:
            raise ValueError("Progress percentage must be between 0 and 100")
        return new_progress
    
    @validates('previous_progress')
    def validate_previous_progress(self, key, previous_progress):
        """Validate previous progress percentage."""
        if previous_progress is not None and (previous_progress < 0 or previous_progress > 100):
            raise ValueError("Progress percentage must be between 0 and 100")
        return previous_progress
    
    @validates('duration_seconds')
    def validate_duration_seconds(self, key, duration_seconds):
        """Validate duration seconds."""
        if duration_seconds is not None and duration_seconds < 0:
            raise ValueError("Duration cannot be negative")
        return duration_seconds
    
    def __repr__(self):
        return f"<StatusHistory(id={self.id}, change_type='{self.change_type.value}', {self.previous_status} -> {self.new_status})>"


class StatusMetrics(Base):
    """
    Status metrics model for storing aggregated processing metrics.
    
    This model provides pre-calculated metrics for reporting and dashboard
    functionality, improving query performance for status analytics.
    
    Attributes:
        id: Primary key
        metric_date: Date for which metrics are calculated
        metric_hour: Hour for which metrics are calculated (0-23)
        total_items: Total number of items processed in this period
        completed_items: Number of completed items
        failed_items: Number of failed items
        cancelled_items: Number of cancelled items
        average_processing_time_seconds: Average processing time
        median_processing_time_seconds: Median processing time
        max_processing_time_seconds: Maximum processing time
        min_processing_time_seconds: Minimum processing time
        retry_rate_percentage: Percentage of items that required retries
        success_rate_percentage: Percentage of successful completions
        queue_wait_time_seconds: Average queue wait time
        worker_utilization_percentage: Worker utilization percentage
        created_at: Metrics creation timestamp
        updated_at: Metrics last update timestamp
        metrics_metadata: Additional metrics metadata
    """
    __tablename__ = 'status_metrics'
    
    id = Column(Integer, primary_key=True, index=True)
    metric_date = Column(DateTime(timezone=True), nullable=False)
    metric_hour = Column(Integer, nullable=True)  # 0-23, null for daily metrics
    total_items = Column(Integer, nullable=False, default=0)
    completed_items = Column(Integer, nullable=False, default=0)
    failed_items = Column(Integer, nullable=False, default=0)
    cancelled_items = Column(Integer, nullable=False, default=0)
    average_processing_time_seconds = Column(Float, nullable=True)
    median_processing_time_seconds = Column(Float, nullable=True)
    max_processing_time_seconds = Column(Float, nullable=True)
    min_processing_time_seconds = Column(Float, nullable=True)
    retry_rate_percentage = Column(Float, nullable=True)
    success_rate_percentage = Column(Float, nullable=True)
    queue_wait_time_seconds = Column(Float, nullable=True)
    worker_utilization_percentage = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    metrics_metadata = Column(JSON, nullable=True)
    
    # Indexes
    __table_args__ = (
        Index('idx_status_metrics_metric_date', 'metric_date'),
        Index('idx_status_metrics_metric_hour', 'metric_hour'),
        Index('idx_status_metrics_date_hour', 'metric_date', 'metric_hour'),
        Index('idx_status_metrics_success_rate', 'success_rate_percentage'),
        Index('idx_status_metrics_total_items', 'total_items'),
        UniqueConstraint('metric_date', 'metric_hour', name='uq_status_metrics_date_hour'),
    )
    
    @validates('metric_hour')
    def validate_metric_hour(self, key, metric_hour):
        """Validate metric hour."""
        if metric_hour is not None and (metric_hour < 0 or metric_hour > 23):
            raise ValueError("Metric hour must be between 0 and 23")
        return metric_hour
    
    @validates('total_items')
    def validate_total_items(self, key, total_items):
        """Validate total items."""
        if total_items < 0:
            raise ValueError("Total items cannot be negative")
        return total_items
    
    @validates('completed_items')
    def validate_completed_items(self, key, completed_items):
        """Validate completed items."""
        if completed_items < 0:
            raise ValueError("Completed items cannot be negative")
        return completed_items
    
    @validates('failed_items')
    def validate_failed_items(self, key, failed_items):
        """Validate failed items."""
        if failed_items < 0:
            raise ValueError("Failed items cannot be negative")
        return failed_items
    
    @validates('cancelled_items')
    def validate_cancelled_items(self, key, cancelled_items):
        """Validate cancelled items."""
        if cancelled_items < 0:
            raise ValueError("Cancelled items cannot be negative")
        return cancelled_items
    
    @property
    def pending_items(self) -> int:
        """Calculate number of pending items."""
        return self.total_items - self.completed_items - self.failed_items - self.cancelled_items
    
    def __repr__(self):
        return f"<StatusMetrics(id={self.id}, date={self.metric_date.date()}, total={self.total_items}, success_rate={self.success_rate_percentage:.1f}%)>"


# Model utilities for status tracking
def get_status_model_by_name(model_name: str):
    """Get status model class by name."""
    models = {
        'ProcessingStatus': ProcessingStatus,
        'StatusHistory': StatusHistory,
        'StatusMetrics': StatusMetrics,
    }
    return models.get(model_name)


def get_all_status_models():
    """Get all status model classes."""
    return [ProcessingStatus, StatusHistory, StatusMetrics]


def create_status_tables(engine):
    """Create all status tracking tables in the database."""
    # Import required models to ensure they're available for foreign key references
    from .models import Video
    from .batch_models import BatchItem, ProcessingSession
    
    # Create status tables
    for model in get_all_status_models():
        model.__table__.create(bind=engine, checkfirst=True)


def drop_status_tables(engine):
    """Drop all status tracking tables from the database."""
    for model in reversed(get_all_status_models()):
        model.__table__.drop(bind=engine, checkfirst=True)