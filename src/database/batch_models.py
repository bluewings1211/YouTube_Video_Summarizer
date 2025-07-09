"""
Batch processing database models for YouTube Summarizer application.
Defines SQLAlchemy models for batch processing functionality.
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


class BatchStatus(enum.Enum):
    """Batch processing status enumeration."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class BatchItemStatus(enum.Enum):
    """Individual batch item status enumeration."""
    QUEUED = "queued"
    PROCESSING = "processing"
    METADATA_PROCESSING = "metadata_processing"
    SUMMARIZING = "summarizing"
    KEYWORD_EXTRACTION = "keyword_extraction"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class BatchPriority(enum.Enum):
    """Batch processing priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class Batch(Base):
    """
    Batch model for storing batch processing information.
    
    Attributes:
        id: Primary key
        batch_id: Unique batch identifier
        name: Optional batch name
        description: Optional batch description
        status: Current batch status
        priority: Batch processing priority
        total_items: Total number of items in batch
        completed_items: Number of completed items
        failed_items: Number of failed items
        created_at: Batch creation timestamp
        updated_at: Batch last update timestamp
        started_at: Batch processing start timestamp
        completed_at: Batch completion timestamp
        webhook_url: Optional webhook URL for notifications
        batch_metadata: Additional batch metadata
        error_info: Error information if batch failed
    """
    __tablename__ = 'batches'
    
    id = Column(Integer, primary_key=True, index=True)
    batch_id = Column(String(255), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=True)
    description = Column(Text, nullable=True)
    status = Column(Enum(BatchStatus), nullable=False, default=BatchStatus.PENDING)
    priority = Column(Enum(BatchPriority), nullable=False, default=BatchPriority.NORMAL)
    total_items = Column(Integer, nullable=False, default=0)
    completed_items = Column(Integer, nullable=False, default=0)
    failed_items = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    webhook_url = Column(String(2000), nullable=True)
    batch_metadata = Column(JSON, nullable=True)
    error_info = Column(Text, nullable=True)
    
    # Relationships
    batch_items = relationship("BatchItem", back_populates="batch", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_batch_id', 'batch_id'),
        Index('idx_batch_status', 'status'),
        Index('idx_batch_priority', 'priority'),
        Index('idx_batch_created_at', 'created_at'),
        Index('idx_batch_status_priority', 'status', 'priority'),
    )
    
    @validates('batch_id')
    def validate_batch_id(self, key, batch_id):
        """Validate batch ID format."""
        if not batch_id or len(batch_id) < 1:
            raise ValueError("Batch ID cannot be empty")
        return batch_id
    
    @validates('total_items')
    def validate_total_items(self, key, total_items):
        """Validate total items count."""
        if total_items < 0:
            raise ValueError("Total items cannot be negative")
        return total_items
    
    @validates('completed_items')
    def validate_completed_items(self, key, completed_items):
        """Validate completed items count."""
        if completed_items < 0:
            raise ValueError("Completed items cannot be negative")
        return completed_items
    
    @validates('failed_items')
    def validate_failed_items(self, key, failed_items):
        """Validate failed items count."""
        if failed_items < 0:
            raise ValueError("Failed items cannot be negative")
        return failed_items
    
    @property
    def pending_items(self) -> int:
        """Calculate number of pending items."""
        total = self.total_items or 0
        completed = self.completed_items or 0
        failed = self.failed_items or 0
        return total - completed - failed
    
    @property
    def progress_percentage(self) -> float:
        """Calculate completion percentage."""
        if not self.total_items or self.total_items == 0:
            return 0.0
        completed = self.completed_items or 0
        return (completed / self.total_items) * 100
    
    @property
    def is_completed(self) -> bool:
        """Check if batch is completed."""
        return self.status == BatchStatus.COMPLETED
    
    @property
    def is_failed(self) -> bool:
        """Check if batch has failed."""
        return self.status == BatchStatus.FAILED
    
    def __repr__(self):
        return f"<Batch(id={self.id}, batch_id='{self.batch_id}', status='{self.status.value}', progress={self.progress_percentage:.1f}%)>"


class BatchItem(Base):
    """
    Batch item model for storing individual items within a batch.
    
    Attributes:
        id: Primary key
        batch_id: Foreign key to Batch
        video_id: Foreign key to Video (nullable if video not yet created)
        url: Video URL to process
        status: Current item status
        priority: Item processing priority
        processing_order: Order in which item should be processed
        created_at: Item creation timestamp
        updated_at: Item last update timestamp
        started_at: Item processing start timestamp
        completed_at: Item completion timestamp
        retry_count: Number of retry attempts
        max_retries: Maximum number of retries allowed
        error_info: Error information if item failed
        processing_data: Additional processing metadata
        result_data: Processing result data
    """
    __tablename__ = 'batch_items'
    
    id = Column(Integer, primary_key=True, index=True)
    batch_id = Column(Integer, ForeignKey('batches.id', ondelete='CASCADE'), nullable=False)
    video_id = Column(Integer, ForeignKey('videos.id', ondelete='SET NULL'), nullable=True)
    url = Column(String(2000), nullable=False)
    status = Column(Enum(BatchItemStatus), nullable=False, default=BatchItemStatus.QUEUED)
    priority = Column(Enum(BatchPriority), nullable=False, default=BatchPriority.NORMAL)
    processing_order = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    retry_count = Column(Integer, nullable=False, default=0)
    max_retries = Column(Integer, nullable=False, default=3)
    error_info = Column(Text, nullable=True)
    processing_data = Column(JSON, nullable=True)
    result_data = Column(JSON, nullable=True)
    
    # Relationships
    batch = relationship("Batch", back_populates="batch_items")
    video = relationship("Video", foreign_keys=[video_id])
    
    # Indexes
    __table_args__ = (
        Index('idx_batch_item_batch_id', 'batch_id'),
        Index('idx_batch_item_video_id', 'video_id'),
        Index('idx_batch_item_status', 'status'),
        Index('idx_batch_item_priority', 'priority'),
        Index('idx_batch_item_processing_order', 'processing_order'),
        Index('idx_batch_item_created_at', 'created_at'),
        Index('idx_batch_item_status_priority', 'status', 'priority'),
        Index('idx_batch_item_batch_status', 'batch_id', 'status'),
    )
    
    @validates('url')
    def validate_url(self, key, url):
        """Validate video URL."""
        if not url or not url.strip():
            raise ValueError("URL cannot be empty")
        return url.strip()
    
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
    
    @validates('processing_order')
    def validate_processing_order(self, key, processing_order):
        """Validate processing order."""
        if processing_order < 0:
            raise ValueError("Processing order cannot be negative")
        return processing_order
    
    @property
    def can_retry(self) -> bool:
        """Check if item can be retried."""
        return self.retry_count < self.max_retries and self.status == BatchItemStatus.FAILED
    
    @property
    def is_completed(self) -> bool:
        """Check if item is completed."""
        return self.status == BatchItemStatus.COMPLETED
    
    @property
    def is_failed(self) -> bool:
        """Check if item has failed."""
        return self.status == BatchItemStatus.FAILED
    
    @property
    def is_processing(self) -> bool:
        """Check if item is currently processing."""
        return self.status in [
            BatchItemStatus.PROCESSING,
            BatchItemStatus.METADATA_PROCESSING,
            BatchItemStatus.SUMMARIZING,
            BatchItemStatus.KEYWORD_EXTRACTION
        ]
    
    def __repr__(self):
        return f"<BatchItem(id={self.id}, batch_id={self.batch_id}, status='{self.status.value}', url='{self.url[:50]}...')>"


class QueueItem(Base):
    """
    Queue item model for managing processing queue.
    
    Attributes:
        id: Primary key
        batch_item_id: Foreign key to BatchItem
        queue_name: Name of the queue
        priority: Processing priority
        scheduled_at: When the item should be processed
        created_at: Queue item creation timestamp
        updated_at: Queue item last update timestamp
        locked_at: When the item was locked for processing
        locked_by: Identifier of the worker processing this item
        lock_expires_at: When the lock expires
        retry_count: Number of retry attempts
        max_retries: Maximum number of retries allowed
        error_info: Error information if processing failed
        queue_metadata: Additional queue metadata
    """
    __tablename__ = 'queue_items'
    
    id = Column(Integer, primary_key=True, index=True)
    batch_item_id = Column(Integer, ForeignKey('batch_items.id', ondelete='CASCADE'), nullable=False)
    queue_name = Column(String(255), nullable=False, default='default')
    priority = Column(Enum(BatchPriority), nullable=False, default=BatchPriority.NORMAL)
    scheduled_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    locked_at = Column(DateTime(timezone=True), nullable=True)
    locked_by = Column(String(255), nullable=True)
    lock_expires_at = Column(DateTime(timezone=True), nullable=True)
    retry_count = Column(Integer, nullable=False, default=0)
    max_retries = Column(Integer, nullable=False, default=3)
    error_info = Column(Text, nullable=True)
    queue_metadata = Column(JSON, nullable=True)
    
    # Relationships
    batch_item = relationship("BatchItem", foreign_keys=[batch_item_id])
    
    # Indexes
    __table_args__ = (
        Index('idx_queue_item_batch_item_id', 'batch_item_id'),
        Index('idx_queue_item_queue_name', 'queue_name'),
        Index('idx_queue_item_priority', 'priority'),
        Index('idx_queue_item_scheduled_at', 'scheduled_at'),
        Index('idx_queue_item_locked_at', 'locked_at'),
        Index('idx_queue_item_locked_by', 'locked_by'),
        Index('idx_queue_item_lock_expires_at', 'lock_expires_at'),
        Index('idx_queue_item_queue_priority', 'queue_name', 'priority'),
        Index('idx_queue_item_queue_scheduled', 'queue_name', 'scheduled_at'),
        Index('idx_queue_item_available', 'queue_name', 'scheduled_at', 'locked_at'),
    )
    
    @validates('queue_name')
    def validate_queue_name(self, key, queue_name):
        """Validate queue name."""
        if not queue_name or not queue_name.strip():
            raise ValueError("Queue name cannot be empty")
        return queue_name.strip()
    
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
    def is_locked(self) -> bool:
        """Check if item is currently locked."""
        if not self.locked_at or not self.lock_expires_at:
            return False
        return datetime.utcnow() < self.lock_expires_at
    
    @property
    def is_available(self) -> bool:
        """Check if item is available for processing."""
        now = datetime.utcnow()
        return (
            self.scheduled_at <= now and
            not self.is_locked and
            self.retry_count < self.max_retries
        )
    
    @property
    def can_retry(self) -> bool:
        """Check if item can be retried."""
        return self.retry_count < self.max_retries
    
    def __repr__(self):
        return f"<QueueItem(id={self.id}, batch_item_id={self.batch_item_id}, queue='{self.queue_name}', priority='{self.priority.value}')>"


class ProcessingSession(Base):
    """
    Processing session model for tracking active processing sessions.
    
    Attributes:
        id: Primary key
        session_id: Unique session identifier
        batch_item_id: Foreign key to BatchItem
        worker_id: Identifier of the worker processing this session
        started_at: Session start timestamp
        updated_at: Session last update timestamp
        heartbeat_at: Last heartbeat timestamp
        status: Current session status
        progress_percentage: Current progress percentage
        current_step: Current processing step
        session_metadata: Additional session metadata
        error_info: Error information if session failed
    """
    __tablename__ = 'processing_sessions'
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(255), unique=True, nullable=False, index=True)
    batch_item_id = Column(Integer, ForeignKey('batch_items.id', ondelete='CASCADE'), nullable=False)
    worker_id = Column(String(255), nullable=False)
    started_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    heartbeat_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    status = Column(Enum(BatchItemStatus), nullable=False, default=BatchItemStatus.PROCESSING)
    progress_percentage = Column(Float, nullable=False, default=0.0)
    current_step = Column(String(255), nullable=True)
    session_metadata = Column(JSON, nullable=True)
    error_info = Column(Text, nullable=True)
    
    # Relationships
    batch_item = relationship("BatchItem", foreign_keys=[batch_item_id])
    
    # Indexes
    __table_args__ = (
        Index('idx_processing_session_session_id', 'session_id'),
        Index('idx_processing_session_batch_item_id', 'batch_item_id'),
        Index('idx_processing_session_worker_id', 'worker_id'),
        Index('idx_processing_session_heartbeat_at', 'heartbeat_at'),
        Index('idx_processing_session_status', 'status'),
        Index('idx_processing_session_started_at', 'started_at'),
    )
    
    @validates('session_id')
    def validate_session_id(self, key, session_id):
        """Validate session ID."""
        if not session_id or not session_id.strip():
            raise ValueError("Session ID cannot be empty")
        return session_id.strip()
    
    @validates('worker_id')
    def validate_worker_id(self, key, worker_id):
        """Validate worker ID."""
        if not worker_id or not worker_id.strip():
            raise ValueError("Worker ID cannot be empty")
        return worker_id.strip()
    
    @validates('progress_percentage')
    def validate_progress_percentage(self, key, progress_percentage):
        """Validate progress percentage."""
        if progress_percentage < 0 or progress_percentage > 100:
            raise ValueError("Progress percentage must be between 0 and 100")
        return progress_percentage
    
    @property
    def is_stale(self, timeout_seconds: int = 300) -> bool:
        """Check if session is stale (no heartbeat for timeout period)."""
        if not self.heartbeat_at:
            return True
        return (datetime.utcnow() - self.heartbeat_at).total_seconds() > timeout_seconds
    
    def __repr__(self):
        return f"<ProcessingSession(id={self.id}, session_id='{self.session_id}', worker='{self.worker_id}', progress={self.progress_percentage:.1f}%)>"


# Model utilities for batch processing
def get_batch_model_by_name(model_name: str):
    """Get batch model class by name."""
    models = {
        'Batch': Batch,
        'BatchItem': BatchItem,
        'QueueItem': QueueItem,
        'ProcessingSession': ProcessingSession,
    }
    return models.get(model_name)


def get_all_batch_models():
    """Get all batch model classes."""
    return [Batch, BatchItem, QueueItem, ProcessingSession]


def create_batch_tables(engine):
    """Create all batch processing tables in the database."""
    # Import Video model to ensure it's available for foreign key references
    from .models import Video
    
    # Create batch tables
    for model in get_all_batch_models():
        model.__table__.create(bind=engine, checkfirst=True)


def drop_batch_tables(engine):
    """Drop all batch processing tables from the database."""
    for model in reversed(get_all_batch_models()):
        model.__table__.drop(bind=engine, checkfirst=True)