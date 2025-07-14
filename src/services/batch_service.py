"""
Batch processing service for YouTube video summarization.

This service provides comprehensive batch processing capabilities including:
- Batch creation and management
- Batch lifecycle management (create, process, complete, cancel)
- Queue management for processing items
- Processing session tracking
- Error handling and validation
- Integration with existing video processing workflow
"""

import logging
import uuid
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from sqlalchemy.orm import Session, selectinload, joinedload
from sqlalchemy import select, update, delete, func, and_, or_
from sqlalchemy.exc import SQLAlchemyError, IntegrityError

from ..database.batch_models import (
    Batch, BatchItem, QueueItem, ProcessingSession,
    BatchStatus, BatchItemStatus, BatchPriority
)
from ..database.models import Video
from ..database.connection import get_database_session
from ..database.exceptions import (
    DatabaseError, DatabaseConnectionError, DatabaseQueryError,
    classify_database_error, is_recoverable_error, should_retry_operation
)
from ..database.transaction_manager import (
    TransactionManager, managed_transaction, OperationType, TransactionResult
)
from ..flow import YouTubeSummarizerFlow
from ..utils.validators import extract_youtube_video_id as extract_video_id_from_url
from ..utils.batch_monitor import get_batch_monitor, BatchMonitor
from ..utils.batch_logger import get_batch_logger, BatchLogger, EventType, LogLevel, BatchOperationLogger

logger = logging.getLogger(__name__)


@dataclass
class BatchCreateRequest:
    """Request data for creating a new batch."""
    name: Optional[str] = None
    description: Optional[str] = None
    urls: List[str] = None
    priority: BatchPriority = BatchPriority.NORMAL
    webhook_url: Optional[str] = None
    batch_metadata: Optional[Dict[str, Any]] = None


@dataclass
class BatchProgressInfo:
    """Progress information for a batch."""
    batch_id: str
    status: BatchStatus
    total_items: int
    completed_items: int
    failed_items: int
    pending_items: int
    progress_percentage: float
    started_at: Optional[datetime]
    estimated_completion: Optional[datetime]


@dataclass
class BatchItemResult:
    """Result of processing a batch item."""
    batch_item_id: int
    status: BatchItemStatus
    video_id: Optional[int] = None
    error_message: Optional[str] = None
    processing_time_seconds: Optional[float] = None
    result_data: Optional[Dict[str, Any]] = None


class BatchServiceError(Exception):
    """Custom exception for batch service operations."""
    pass


class BatchService:
    """
    Service class for managing batch processing operations.
    
    This service provides comprehensive batch processing capabilities including:
    - Creating and managing batches
    - Processing queue management
    - Lifecycle management (create, process, complete, cancel)
    - Error handling and retry logic
    - Integration with existing video processing workflow
    """

    def __init__(self, session: Optional[Session] = None, max_retries: int = 3, retry_delay: float = 1.0):
        """
        Initialize the batch service.
        
        Args:
            session: Optional database session. If not provided, will use dependency injection.
            max_retries: Maximum number of retries for database operations
            retry_delay: Delay between retries in seconds
        """
        self._session = session
        self._logger = logging.getLogger(f"{__name__}.BatchService")
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Initialize monitoring and logging
        self._monitor = get_batch_monitor()
        self._batch_logger = get_batch_logger()

    def _get_session(self) -> Session:
        """Get database session (internal method)."""
        if self._session:
            return self._session
        else:
            raise BatchServiceError("No database session provided")

    def _generate_batch_id(self) -> str:
        """Generate unique batch ID."""
        return f"batch_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

    def _validate_urls(self, urls: List[str]) -> List[str]:
        """
        Validate and normalize YouTube URLs.
        
        Args:
            urls: List of YouTube URLs to validate
            
        Returns:
            List of validated URLs
            
        Raises:
            BatchServiceError: If validation fails
        """
        if not urls:
            raise BatchServiceError("URLs list cannot be empty")
        
        validated_urls = []
        for url in urls:
            if not url or not url.strip():
                continue
                
            url = url.strip()
            try:
                video_id = extract_video_id_from_url(url)
                if not video_id:
                    raise BatchServiceError(f"Invalid YouTube URL: {url}")
                validated_urls.append(url)
            except Exception as e:
                raise BatchServiceError(f"URL validation failed for {url}: {e}")
        
        if not validated_urls:
            raise BatchServiceError("No valid URLs provided")
        
        return validated_urls

    def create_batch(self, request: BatchCreateRequest) -> Batch:
        """
        Create a new batch for processing.
        
        Args:
            request: Batch creation request
            
        Returns:
            Created Batch object
            
        Raises:
            BatchServiceError: If creation fails
        """
        try:
            session = self._get_session()
            
            # Validate URLs
            validated_urls = self._validate_urls(request.urls)
            
            with managed_transaction(session, description="Create new batch") as txn:
                # Generate batch ID
                batch_id = self._generate_batch_id()
                
                # Start monitoring for batch creation
                with BatchOperationLogger(
                    self._batch_logger,
                    batch_id,
                    "batch_creation",
                    context={
                        "total_items": len(validated_urls),
                        "priority": request.priority.value,
                        "name": request.name
                    }
                ) as op_logger:
                    
                    # Create batch record
                    batch = Batch(
                        batch_id=batch_id,
                        name=request.name,
                        description=request.description,
                        status=BatchStatus.PENDING,
                        priority=request.priority,
                        total_items=len(validated_urls),
                        webhook_url=request.webhook_url,
                        batch_metadata=request.batch_metadata or {}
                    )
                    
                    session.add(batch)
                    session.flush()  # Get the batch.id
                    
                    # Start monitoring
                    self._monitor.start_batch_monitoring(batch_id, len(validated_urls))
                    
                    # Log batch creation
                    self._batch_logger.log_batch_event(
                        batch_id=batch_id,
                        event_type=EventType.BATCH_CREATED,
                        message=f"Created batch with {len(validated_urls)} items",
                        context={
                            "name": request.name,
                            "priority": request.priority.value,
                            "total_items": len(validated_urls)
                        }
                    )
                    
                    # Create batch items
                    batch_items = []
                    for order, url in enumerate(validated_urls):
                        batch_item = BatchItem(
                            batch_id=batch.id,
                            url=url,
                            status=BatchItemStatus.QUEUED,
                            priority=request.priority,
                            processing_order=order,
                            max_retries=3
                        )
                        batch_items.append(batch_item)
                    
                    session.add_all(batch_items)
                    session.flush()
                    
                    # Create queue items for processing
                    queue_items = []
                    for batch_item in batch_items:
                        queue_item = QueueItem(
                            batch_item_id=batch_item.id,
                            queue_name='video_processing',
                            priority=request.priority,
                            max_retries=3
                        )
                        queue_items.append(queue_item)
                        
                        # Log item queued
                        self._batch_logger.log_item_event(
                            batch_id=batch_id,
                            batch_item_id=batch_item.id,
                            event_type=EventType.ITEM_QUEUED,
                            message=f"Queued item for processing: {url}",
                            context={"url": url, "processing_order": order}
                        )
                    
                    session.add_all(queue_items)
                    
                    # Log operation
                    txn.execute_operation(
                        OperationType.CREATE,
                        f"Created batch {batch_id} with {len(validated_urls)} items",
                        "batches",
                        lambda: None,
                        target_id=batch.id,
                        parameters={
                            "batch_id": batch_id,
                            "total_items": len(validated_urls),
                            "priority": request.priority.value
                        }
                    )
                    
                    result = txn.commit_transaction()
                    
                    self._logger.info(f"Created batch {batch_id} with {len(validated_urls)} items")
                    return batch
                
        except SQLAlchemyError as e:
            db_error = classify_database_error(e)
            self._batch_logger.log_error(
                message="Database error creating batch",
                error=e,
                component="batch_service",
                context={"urls_count": len(request.urls) if request.urls else 0}
            )
            raise BatchServiceError(f"Failed to create batch: {db_error.message}")
        except Exception as e:
            self._batch_logger.log_error(
                message="Unexpected error creating batch",
                error=e,
                component="batch_service",
                context={"urls_count": len(request.urls) if request.urls else 0}
            )
            raise BatchServiceError(f"Unexpected error: {e}")

    def get_batch(self, batch_id: str) -> Optional[Batch]:
        """
        Get batch by ID with related data.
        
        Args:
            batch_id: Batch identifier
            
        Returns:
            Batch object if found, None otherwise
            
        Raises:
            BatchServiceError: If query fails
        """
        try:
            session = self._get_session()
            
            query = select(Batch).options(
                selectinload(Batch.batch_items)
            ).where(Batch.batch_id == batch_id)
            
            result = session.execute(query)
            batch = result.scalar_one_or_none()
            
            if batch:
                self._logger.info(f"Retrieved batch {batch_id}")
            else:
                self._logger.info(f"Batch {batch_id} not found")
            
            return batch
            
        except SQLAlchemyError as e:
            db_error = classify_database_error(e)
            self._logger.error(f"Database error getting batch: {db_error}")
            raise BatchServiceError(f"Failed to get batch: {db_error.message}")
        except Exception as e:
            self._logger.error(f"Unexpected error getting batch: {e}")
            raise BatchServiceError(f"Unexpected error: {e}")

    def get_batch_progress(self, batch_id: str) -> Optional[BatchProgressInfo]:
        """
        Get batch progress information.
        
        Args:
            batch_id: Batch identifier
            
        Returns:
            BatchProgressInfo if found, None otherwise
            
        Raises:
            BatchServiceError: If query fails
        """
        try:
            session = self._get_session()
            
            batch = self.get_batch(batch_id)
            if not batch:
                return None
            
            # Calculate estimated completion time
            estimated_completion = None
            if batch.started_at and batch.status == BatchStatus.PROCESSING:
                if batch.completed_items > 0:
                    avg_time_per_item = (datetime.utcnow() - batch.started_at).total_seconds() / batch.completed_items
                    remaining_items = batch.total_items - batch.completed_items
                    estimated_completion = datetime.utcnow() + timedelta(seconds=avg_time_per_item * remaining_items)
            
            progress_info = BatchProgressInfo(
                batch_id=batch_id,
                status=batch.status,
                total_items=batch.total_items,
                completed_items=batch.completed_items,
                failed_items=batch.failed_items,
                pending_items=batch.pending_items,
                progress_percentage=batch.progress_percentage,
                started_at=batch.started_at,
                estimated_completion=estimated_completion
            )
            
            return progress_info
            
        except Exception as e:
            self._logger.error(f"Error getting batch progress: {e}")
            raise BatchServiceError(f"Failed to get batch progress: {e}")

    def start_batch_processing(self, batch_id: str) -> bool:
        """
        Start processing a batch.
        
        Args:
            batch_id: Batch identifier
            
        Returns:
            True if started successfully, False otherwise
            
        Raises:
            BatchServiceError: If operation fails
        """
        try:
            session = self._get_session()
            
            with managed_transaction(session, description=f"Start batch processing {batch_id}") as txn:
                batch = self.get_batch(batch_id)
                if not batch:
                    raise BatchServiceError(f"Batch {batch_id} not found")
                
                if batch.status != BatchStatus.PENDING:
                    raise BatchServiceError(f"Batch {batch_id} is not in PENDING status")
                
                # Update batch status
                batch.status = BatchStatus.PROCESSING
                batch.started_at = datetime.utcnow()
                
                # Log batch started
                self._batch_logger.log_batch_event(
                    batch_id=batch_id,
                    event_type=EventType.BATCH_STARTED,
                    message=f"Started batch processing with {batch.total_items} items",
                    context={
                        "total_items": batch.total_items,
                        "priority": batch.priority.value,
                        "name": batch.name
                    }
                )
                
                # Log operation
                txn.execute_operation(
                    OperationType.UPDATE,
                    f"Started batch processing for {batch_id}",
                    "batches",
                    lambda: None,
                    target_id=batch.id,
                    parameters={"status": BatchStatus.PROCESSING.value}
                )
                
                result = txn.commit_transaction()
                
                self._logger.info(f"Started batch processing for {batch_id}")
                return True
                
        except SQLAlchemyError as e:
            db_error = classify_database_error(e)
            self._logger.error(f"Database error starting batch: {db_error}")
            raise BatchServiceError(f"Failed to start batch: {db_error.message}")
        except Exception as e:
            self._logger.error(f"Unexpected error starting batch: {e}")
            raise BatchServiceError(f"Unexpected error: {e}")

    def cancel_batch(self, batch_id: str, reason: Optional[str] = None) -> bool:
        """
        Cancel a batch and all its items.
        
        Args:
            batch_id: Batch identifier
            reason: Cancellation reason
            
        Returns:
            True if cancelled successfully, False otherwise
            
        Raises:
            BatchServiceError: If operation fails
        """
        try:
            session = self._get_session()
            
            with managed_transaction(session, description=f"Cancel batch {batch_id}") as txn:
                batch = self.get_batch(batch_id)
                if not batch:
                    raise BatchServiceError(f"Batch {batch_id} not found")
                
                if batch.status in [BatchStatus.COMPLETED, BatchStatus.CANCELLED]:
                    raise BatchServiceError(f"Batch {batch_id} is already {batch.status.value}")
                
                # Update batch status
                batch.status = BatchStatus.CANCELLED
                batch.completed_at = datetime.utcnow()
                batch.error_info = reason or "Cancelled by user"
                
                # Cancel all pending batch items
                cancelled_items = session.execute(
                    update(BatchItem)
                    .where(
                        and_(
                            BatchItem.batch_id == batch.id,
                            BatchItem.status.in_([
                                BatchItemStatus.QUEUED,
                                BatchItemStatus.PROCESSING
                            ])
                        )
                    )
                    .values(status=BatchItemStatus.CANCELLED)
                ).rowcount
                
                # Log operation
                txn.execute_operation(
                    OperationType.UPDATE,
                    f"Cancelled batch {batch_id} and {cancelled_items} items",
                    "batches",
                    lambda: None,
                    target_id=batch.id,
                    parameters={
                        "status": BatchStatus.CANCELLED.value,
                        "cancelled_items": cancelled_items,
                        "reason": reason
                    }
                )
                
                result = txn.commit_transaction()
                
                self._logger.info(f"Cancelled batch {batch_id} with {cancelled_items} items")
                return True
                
        except SQLAlchemyError as e:
            db_error = classify_database_error(e)
            self._logger.error(f"Database error cancelling batch: {db_error}")
            raise BatchServiceError(f"Failed to cancel batch: {db_error.message}")
        except Exception as e:
            self._logger.error(f"Unexpected error cancelling batch: {e}")
            raise BatchServiceError(f"Unexpected error: {e}")

    def get_next_queue_item(self, queue_name: str = 'video_processing', worker_id: str = None) -> Optional[QueueItem]:
        """
        Get the next available queue item for processing.
        
        Args:
            queue_name: Name of the queue
            worker_id: Worker identifier for locking
            
        Returns:
            Next available QueueItem or None if none available
            
        Raises:
            BatchServiceError: If query fails
        """
        try:
            session = self._get_session()
            
            with managed_transaction(session, description="Get next queue item") as txn:
                # Find available queue items
                query = select(QueueItem).options(
                    selectinload(QueueItem.batch_item)
                ).where(
                    and_(
                        QueueItem.queue_name == queue_name,
                        QueueItem.scheduled_at <= datetime.utcnow(),
                        or_(
                            QueueItem.locked_at.is_(None),
                            QueueItem.lock_expires_at < datetime.utcnow()
                        ),
                        QueueItem.retry_count < QueueItem.max_retries
                    )
                ).order_by(
                    QueueItem.priority.desc(),
                    QueueItem.scheduled_at.asc()
                ).limit(1)
                
                result = session.execute(query)
                queue_item = result.scalar_one_or_none()
                
                if queue_item and worker_id:
                    # Lock the item
                    queue_item.locked_at = datetime.utcnow()
                    queue_item.locked_by = worker_id
                    queue_item.lock_expires_at = datetime.utcnow() + timedelta(minutes=30)
                    
                    # Update batch item status
                    if queue_item.batch_item:
                        queue_item.batch_item.status = BatchItemStatus.PROCESSING
                        queue_item.batch_item.started_at = datetime.utcnow()
                    
                    # Log operation
                    txn.execute_operation(
                        OperationType.UPDATE,
                        f"Locked queue item {queue_item.id} for worker {worker_id}",
                        "queue_items",
                        lambda: None,
                        target_id=queue_item.id,
                        parameters={"worker_id": worker_id}
                    )
                
                result = txn.commit_transaction()
                
                if queue_item:
                    self._logger.info(f"Retrieved queue item {queue_item.id} for worker {worker_id}")
                
                return queue_item
                
        except SQLAlchemyError as e:
            db_error = classify_database_error(e)
            self._logger.error(f"Database error getting queue item: {db_error}")
            raise BatchServiceError(f"Failed to get queue item: {db_error.message}")
        except Exception as e:
            self._logger.error(f"Unexpected error getting queue item: {e}")
            raise BatchServiceError(f"Unexpected error: {e}")

    def complete_batch_item(self, batch_item_id: int, result: BatchItemResult) -> bool:
        """
        Complete processing of a batch item.
        
        Args:
            batch_item_id: Batch item ID
            result: Processing result
            
        Returns:
            True if completed successfully, False otherwise
            
        Raises:
            BatchServiceError: If operation fails
        """
        try:
            session = self._get_session()
            
            with managed_transaction(session, description=f"Complete batch item {batch_item_id}") as txn:
                # Get batch item
                batch_item = session.get(BatchItem, batch_item_id)
                if not batch_item:
                    raise BatchServiceError(f"Batch item {batch_item_id} not found")
                
                # Calculate processing duration
                processing_duration = None
                if batch_item.started_at:
                    processing_duration = (datetime.utcnow() - batch_item.started_at).total_seconds()
                
                # Update batch item
                batch_item.status = result.status
                batch_item.completed_at = datetime.utcnow()
                batch_item.video_id = result.video_id
                batch_item.error_info = result.error_message
                batch_item.result_data = result.result_data
                
                # Get batch information
                batch = session.get(Batch, batch_item.batch_id)
                batch_id = batch.batch_id if batch else "unknown"
                
                # Log item completion
                event_type = EventType.ITEM_COMPLETED if result.status == BatchItemStatus.COMPLETED else EventType.ITEM_FAILED
                self._batch_logger.log_item_event(
                    batch_id=batch_id,
                    batch_item_id=batch_item_id,
                    event_type=event_type,
                    message=f"Completed item with status {result.status.value}",
                    context={
                        "url": batch_item.url,
                        "video_id": result.video_id,
                        "error_message": result.error_message,
                        "retry_count": batch_item.retry_count
                    },
                    duration=processing_duration
                )
                
                # Update monitoring
                self._monitor.finish_item_monitoring(batch_item_id, result.status)
                
                # Record error if failed
                if result.status == BatchItemStatus.FAILED:
                    self._monitor.record_error(batch_id, batch_item_id, result.error_message or "Unknown error")
                
                # Remove from queue
                session.execute(
                    delete(QueueItem).where(QueueItem.batch_item_id == batch_item_id)
                )
                
                # Update batch counters
                if batch:
                    if result.status == BatchItemStatus.COMPLETED:
                        batch.completed_items += 1
                    elif result.status == BatchItemStatus.FAILED:
                        batch.failed_items += 1
                    
                    # Check if batch is complete
                    if batch.completed_items + batch.failed_items >= batch.total_items:
                        batch.status = BatchStatus.COMPLETED
                        batch.completed_at = datetime.utcnow()
                        
                        # Log batch completion
                        self._batch_logger.log_batch_event(
                            batch_id=batch_id,
                            event_type=EventType.BATCH_COMPLETED,
                            message=f"Batch completed with {batch.completed_items} successful and {batch.failed_items} failed items",
                            context={
                                "total_items": batch.total_items,
                                "completed_items": batch.completed_items,
                                "failed_items": batch.failed_items,
                                "success_rate": (batch.completed_items / batch.total_items) * 100 if batch.total_items > 0 else 0
                            }
                        )
                        
                        # Finish monitoring
                        self._monitor.finish_batch_monitoring(batch_id)
                
                # Log operation
                txn.execute_operation(
                    OperationType.UPDATE,
                    f"Completed batch item {batch_item_id} with status {result.status.value}",
                    "batch_items",
                    lambda: None,
                    target_id=batch_item_id,
                    parameters={"status": result.status.value, "video_id": result.video_id}
                )
                
                result = txn.commit_transaction()
                
                self._logger.info(f"Completed batch item {batch_item_id} with status {result.status.value}")
                return True
                
        except SQLAlchemyError as e:
            db_error = classify_database_error(e)
            self._logger.error(f"Database error completing batch item: {db_error}")
            raise BatchServiceError(f"Failed to complete batch item: {db_error.message}")
        except Exception as e:
            self._logger.error(f"Unexpected error completing batch item: {e}")
            raise BatchServiceError(f"Unexpected error: {e}")

    def retry_failed_batch_item(self, batch_item_id: int) -> bool:
        """
        Retry a failed batch item.
        
        Args:
            batch_item_id: Batch item ID
            
        Returns:
            True if retry was scheduled, False otherwise
            
        Raises:
            BatchServiceError: If operation fails
        """
        try:
            session = self._get_session()
            
            with managed_transaction(session, description=f"Retry batch item {batch_item_id}") as txn:
                # Get batch item
                batch_item = session.get(BatchItem, batch_item_id)
                if not batch_item:
                    raise BatchServiceError(f"Batch item {batch_item_id} not found")
                
                if not batch_item.can_retry:
                    raise BatchServiceError(f"Batch item {batch_item_id} cannot be retried")
                
                # Get batch information
                batch = session.get(Batch, batch_item.batch_id)
                batch_id = batch.batch_id if batch else "unknown"
                
                # Update batch item
                batch_item.status = BatchItemStatus.QUEUED
                batch_item.retry_count += 1
                batch_item.error_info = None
                batch_item.started_at = None
                batch_item.completed_at = None
                
                # Log retry event
                self._batch_logger.log_item_event(
                    batch_id=batch_id,
                    batch_item_id=batch_item_id,
                    event_type=EventType.ITEM_RETRIED,
                    message=f"Retrying item (attempt {batch_item.retry_count})",
                    context={
                        "url": batch_item.url,
                        "retry_count": batch_item.retry_count,
                        "max_retries": batch_item.max_retries
                    }
                )
                
                # Record retry in monitoring
                self._monitor.record_retry(batch_id, batch_item_id)
                
                # Create new queue item
                queue_item = QueueItem(
                    batch_item_id=batch_item_id,
                    queue_name='video_processing',
                    priority=batch_item.priority,
                    retry_count=batch_item.retry_count,
                    max_retries=batch_item.max_retries
                )
                session.add(queue_item)
                
                # Update batch counters
                if batch:
                    batch.failed_items -= 1
                    if batch.status == BatchStatus.COMPLETED:
                        batch.status = BatchStatus.PROCESSING
                        batch.completed_at = None
                
                # Log operation
                txn.execute_operation(
                    OperationType.UPDATE,
                    f"Retried batch item {batch_item_id} (attempt {batch_item.retry_count})",
                    "batch_items",
                    lambda: None,
                    target_id=batch_item_id,
                    parameters={"retry_count": batch_item.retry_count}
                )
                
                result = txn.commit_transaction()
                
                self._logger.info(f"Retried batch item {batch_item_id} (attempt {batch_item.retry_count})")
                return True
                
        except SQLAlchemyError as e:
            db_error = classify_database_error(e)
            self._logger.error(f"Database error retrying batch item: {db_error}")
            raise BatchServiceError(f"Failed to retry batch item: {db_error.message}")
        except Exception as e:
            self._logger.error(f"Unexpected error retrying batch item: {e}")
            raise BatchServiceError(f"Unexpected error: {e}")

    def create_processing_session(self, batch_item_id: int, worker_id: str) -> ProcessingSession:
        """
        Create a processing session for tracking progress.
        
        Args:
            batch_item_id: Batch item ID
            worker_id: Worker identifier
            
        Returns:
            Created ProcessingSession
            
        Raises:
            BatchServiceError: If creation fails
        """
        try:
            session = self._get_session()
            
            session_id = f"session_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            
            processing_session = ProcessingSession(
                session_id=session_id,
                batch_item_id=batch_item_id,
                worker_id=worker_id,
                status=BatchItemStatus.PROCESSING,
                progress_percentage=0.0
            )
            
            session.add(processing_session)
            session.commit()
            
            self._logger.info(f"Created processing session {session_id} for batch item {batch_item_id}")
            return processing_session
            
        except SQLAlchemyError as e:
            db_error = classify_database_error(e)
            self._logger.error(f"Database error creating processing session: {db_error}")
            raise BatchServiceError(f"Failed to create processing session: {db_error.message}")
        except Exception as e:
            self._logger.error(f"Unexpected error creating processing session: {e}")
            raise BatchServiceError(f"Unexpected error: {e}")

    def update_processing_session(self, session_id: str, progress: float, current_step: str = None) -> bool:
        """
        Update processing session progress.
        
        Args:
            session_id: Processing session ID
            progress: Progress percentage (0-100)
            current_step: Current processing step
            
        Returns:
            True if updated successfully, False otherwise
            
        Raises:
            BatchServiceError: If update fails
        """
        try:
            session = self._get_session()
            
            processing_session = session.execute(
                select(ProcessingSession).where(ProcessingSession.session_id == session_id)
            ).scalar_one_or_none()
            
            if not processing_session:
                raise BatchServiceError(f"Processing session {session_id} not found")
            
            processing_session.progress_percentage = progress
            processing_session.current_step = current_step
            processing_session.heartbeat_at = datetime.utcnow()
            
            session.commit()
            
            self._logger.debug(f"Updated processing session {session_id}: {progress}% - {current_step}")
            return True
            
        except SQLAlchemyError as e:
            db_error = classify_database_error(e)
            self._logger.error(f"Database error updating processing session: {db_error}")
            raise BatchServiceError(f"Failed to update processing session: {db_error.message}")
        except Exception as e:
            self._logger.error(f"Unexpected error updating processing session: {e}")
            raise BatchServiceError(f"Unexpected error: {e}")

    def get_batch_statistics(self) -> Dict[str, Any]:
        """
        Get batch processing statistics.
        
        Returns:
            Dictionary containing batch statistics
            
        Raises:
            BatchServiceError: If query fails
        """
        try:
            session = self._get_session()
            
            # Total batches
            total_batches = session.execute(
                select(func.count(Batch.id))
            ).scalar()
            
            # Batch status counts
            status_counts = {}
            for status in BatchStatus:
                count = session.execute(
                    select(func.count(Batch.id)).where(Batch.status == status)
                ).scalar()
                status_counts[status.value] = count
            
            # Total batch items
            total_items = session.execute(
                select(func.count(BatchItem.id))
            ).scalar()
            
            # Batch item status counts
            item_status_counts = {}
            for status in BatchItemStatus:
                count = session.execute(
                    select(func.count(BatchItem.id)).where(BatchItem.status == status)
                ).scalar()
                item_status_counts[status.value] = count
            
            # Active processing sessions
            active_sessions = session.execute(
                select(func.count(ProcessingSession.id)).where(
                    ProcessingSession.heartbeat_at > datetime.utcnow() - timedelta(minutes=5)
                )
            ).scalar()
            
            statistics = {
                "total_batches": total_batches,
                "batch_status_counts": status_counts,
                "total_batch_items": total_items,
                "item_status_counts": item_status_counts,
                "active_processing_sessions": active_sessions
            }
            
            self._logger.info("Retrieved batch statistics")
            return statistics
            
        except SQLAlchemyError as e:
            db_error = classify_database_error(e)
            self._logger.error(f"Database error getting batch statistics: {db_error}")
            raise BatchServiceError(f"Failed to get batch statistics: {db_error.message}")
        except Exception as e:
            self._logger.error(f"Unexpected error getting batch statistics: {e}")
            raise BatchServiceError(f"Unexpected error: {e}")

    def cleanup_stale_sessions(self, timeout_minutes: int = 30) -> int:
        """
        Clean up stale processing sessions.
        
        Args:
            timeout_minutes: Session timeout in minutes
            
        Returns:
            Number of sessions cleaned up
            
        Raises:
            BatchServiceError: If cleanup fails
        """
        try:
            session = self._get_session()
            
            cutoff_time = datetime.utcnow() - timedelta(minutes=timeout_minutes)
            
            # Find stale sessions
            stale_sessions = session.execute(
                select(ProcessingSession).where(
                    ProcessingSession.heartbeat_at < cutoff_time
                )
            ).scalars().all()
            
            cleaned_count = 0
            for ps in stale_sessions:
                # Update related batch item status
                batch_item = session.get(BatchItem, ps.batch_item_id)
                if batch_item and batch_item.status == BatchItemStatus.PROCESSING:
                    batch_item.status = BatchItemStatus.FAILED
                    batch_item.error_info = "Processing session timed out"
                    batch_item.completed_at = datetime.utcnow()
                
                # Remove session
                session.delete(ps)
                cleaned_count += 1
            
            session.commit()
            
            self._logger.info(f"Cleaned up {cleaned_count} stale processing sessions")
            return cleaned_count
            
        except SQLAlchemyError as e:
            db_error = classify_database_error(e)
            self._logger.error(f"Database error cleaning up stale sessions: {db_error}")
            raise BatchServiceError(f"Failed to cleanup stale sessions: {db_error.message}")
        except Exception as e:
            self._logger.error(f"Unexpected error cleaning up stale sessions: {e}")
            raise BatchServiceError(f"Unexpected error: {e}")

    def list_batches(self, 
                     status: Optional[BatchStatus] = None,
                     limit: int = 50,
                     offset: int = 0) -> List[Batch]:
        """
        List batches with optional filtering.
        
        Args:
            status: Optional status filter
            limit: Maximum number of batches to return
            offset: Number of batches to skip
            
        Returns:
            List of Batch objects
            
        Raises:
            BatchServiceError: If query fails
        """
        try:
            session = self._get_session()
            
            query = select(Batch).options(
                selectinload(Batch.batch_items)
            )
            
            if status:
                query = query.where(Batch.status == status)
            
            query = query.order_by(Batch.created_at.desc()).limit(limit).offset(offset)
            
            result = session.execute(query)
            batches = result.scalars().all()
            
            self._logger.info(f"Listed {len(batches)} batches")
            return batches
            
        except SQLAlchemyError as e:
            db_error = classify_database_error(e)
            self._logger.error(f"Database error listing batches: {db_error}")
            raise BatchServiceError(f"Failed to list batches: {db_error.message}")
        except Exception as e:
            self._logger.error(f"Unexpected error listing batches: {e}")
            raise BatchServiceError(f"Unexpected error: {e}")


# Dependency injection function for FastAPI
def get_batch_service() -> BatchService:
    """
    Get BatchService instance for dependency injection.
    
    Returns:
        BatchService instance
    """
    return BatchService()