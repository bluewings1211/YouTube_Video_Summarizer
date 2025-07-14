"""
Queue management service for YouTube video summarization batch processing.

This service provides comprehensive queue management capabilities including:
- Priority-based queue processing
- Worker management and coordination
- Locking mechanisms for concurrent processing
- Queue monitoring and statistics
- Integration with existing batch processing system
- Automatic cleanup of stale locks and sessions
"""

import logging
import uuid
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from sqlalchemy.orm import Session, selectinload, joinedload
from sqlalchemy import select, update, delete, func, and_, or_, text
from sqlalchemy.exc import SQLAlchemyError, IntegrityError
import threading
import time
import json

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
from ..utils.batch_monitor import get_batch_monitor, BatchMonitor
from ..utils.batch_logger import get_batch_logger, BatchLogger, EventType, LogLevel

logger = logging.getLogger(__name__)


class QueueWorkerStatus(Enum):
    """Queue worker status enumeration."""
    IDLE = "idle"
    PROCESSING = "processing"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


class QueueHealthStatus(Enum):
    """Queue health status enumeration."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    OFFLINE = "offline"


@dataclass
class WorkerInfo:
    """Information about a queue worker."""
    worker_id: str
    queue_name: str
    status: QueueWorkerStatus
    last_heartbeat: datetime
    current_batch_item_id: Optional[int] = None
    started_at: datetime = field(default_factory=datetime.utcnow)
    processed_items: int = 0
    failed_items: int = 0
    worker_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QueueStatistics:
    """Queue processing statistics."""
    queue_name: str
    total_items: int
    pending_items: int
    processing_items: int
    completed_items: int
    failed_items: int
    locked_items: int
    stale_locks: int
    active_workers: int
    average_processing_time: float
    last_processed_at: Optional[datetime]
    health_status: QueueHealthStatus
    priority_distribution: Dict[str, int]


@dataclass
class QueueProcessingOptions:
    """Options for queue processing."""
    max_workers: int = 5
    worker_timeout_minutes: int = 30
    lock_timeout_minutes: int = 15
    heartbeat_interval_seconds: int = 30
    stale_lock_cleanup_interval_minutes: int = 5
    max_retries: int = 3
    retry_delay_minutes: int = 5
    enable_priority_processing: bool = True
    enable_worker_monitoring: bool = True
    enable_automatic_cleanup: bool = True


class QueueServiceError(Exception):
    """Custom exception for queue service operations."""
    pass


class QueueService:
    """
    Service class for managing queue processing operations.
    
    This service provides comprehensive queue management capabilities including:
    - Priority-based queue processing
    - Worker management and coordination
    - Locking mechanisms for concurrent processing
    - Queue monitoring and statistics
    - Integration with existing batch processing system
    - Automatic cleanup of stale locks and sessions
    """

    def __init__(self, 
                 session: Optional[Session] = None,
                 options: Optional[QueueProcessingOptions] = None):
        """
        Initialize the queue service.
        
        Args:
            session: Optional database session. If not provided, will use dependency injection.
            options: Queue processing options
        """
        self._session = session
        self._logger = logging.getLogger(f"{__name__}.QueueService")
        self.options = options or QueueProcessingOptions()
        
        # Initialize monitoring and logging
        self._monitor = get_batch_monitor()
        self._batch_logger = get_batch_logger()
        
        # Internal state
        self._worker_registry: Dict[str, WorkerInfo] = {}
        self._queue_locks: Dict[str, threading.Lock] = {}
        self._cleanup_thread: Optional[threading.Thread] = None
        self._cleanup_running = False
        self._start_cleanup_thread()

    def _get_session(self) -> Session:
        """Get database session (internal method)."""
        if self._session:
            return self._session
        else:
            raise QueueServiceError("No database session provided")

    def _get_queue_lock(self, queue_name: str) -> threading.Lock:
        """Get or create a lock for a specific queue."""
        if queue_name not in self._queue_locks:
            self._queue_locks[queue_name] = threading.Lock()
        return self._queue_locks[queue_name]

    def _generate_worker_id(self, queue_name: str) -> str:
        """Generate unique worker ID."""
        return f"worker_{queue_name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

    def _start_cleanup_thread(self):
        """Start the cleanup thread for stale locks and sessions."""
        if self.options.enable_automatic_cleanup and not self._cleanup_running:
            self._cleanup_running = True
            self._cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
            self._cleanup_thread.start()
            self._logger.info("Started automatic cleanup thread")

    def _cleanup_worker(self):
        """Background worker for cleaning up stale locks and sessions."""
        while self._cleanup_running:
            try:
                # Clean up stale locks
                self._cleanup_stale_locks()
                
                # Clean up stale processing sessions
                self._cleanup_stale_sessions()
                
                # Clean up inactive workers
                self._cleanup_inactive_workers()
                
                # Sleep for the configured interval
                time.sleep(self.options.stale_lock_cleanup_interval_minutes * 60)
                
            except Exception as e:
                self._logger.error(f"Error in cleanup worker: {e}")
                time.sleep(60)  # Sleep for 1 minute on error

    def register_worker(self, queue_name: str, worker_id: Optional[str] = None) -> WorkerInfo:
        """
        Register a new worker for queue processing.
        
        Args:
            queue_name: Name of the queue
            worker_id: Optional worker ID. If not provided, will generate one.
            
        Returns:
            WorkerInfo object for the registered worker
            
        Raises:
            QueueServiceError: If registration fails
        """
        try:
            if not worker_id:
                worker_id = self._generate_worker_id(queue_name)
            
            worker_info = WorkerInfo(
                worker_id=worker_id,
                queue_name=queue_name,
                status=QueueWorkerStatus.IDLE,
                last_heartbeat=datetime.utcnow()
            )
            
            self._worker_registry[worker_id] = worker_info
            
            # Log worker registration
            self._batch_logger.log_worker_event(
                worker_id=worker_id,
                event_type=EventType.WORKER_STARTED,
                message=f"Registered worker for queue {queue_name}",
                context={"queue_name": queue_name}
            )
            
            self._logger.info(f"Registered worker {worker_id} for queue {queue_name}")
            return worker_info
            
        except Exception as e:
            self._batch_logger.log_error(
                message=f"Error registering worker for queue {queue_name}",
                error=e,
                component="queue_service"
            )
            raise QueueServiceError(f"Failed to register worker: {e}")

    def unregister_worker(self, worker_id: str) -> bool:
        """
        Unregister a worker from queue processing.
        
        Args:
            worker_id: Worker identifier
            
        Returns:
            True if unregistered successfully, False otherwise
        """
        try:
            if worker_id in self._worker_registry:
                # Mark worker as stopped
                self._worker_registry[worker_id].status = QueueWorkerStatus.STOPPED
                
                # Release any locks held by this worker
                self._release_worker_locks(worker_id)
                
                # Log worker unregistration
                self._batch_logger.log_worker_event(
                    worker_id=worker_id,
                    event_type=EventType.WORKER_STOPPED,
                    message=f"Unregistered worker {worker_id}",
                    context={"queue_name": self._worker_registry[worker_id].queue_name}
                )
                
                # Remove from registry
                del self._worker_registry[worker_id]
                
                self._logger.info(f"Unregistered worker {worker_id}")
                return True
            else:
                self._logger.warning(f"Worker {worker_id} not found in registry")
                return False
                
        except Exception as e:
            self._batch_logger.log_error(
                message=f"Error unregistering worker {worker_id}",
                error=e,
                worker_id=worker_id,
                component="queue_service"
            )
            return False

    def update_worker_heartbeat(self, worker_id: str, status: Optional[QueueWorkerStatus] = None) -> bool:
        """
        Update worker heartbeat and status.
        
        Args:
            worker_id: Worker identifier
            status: Optional new status
            
        Returns:
            True if updated successfully, False otherwise
        """
        try:
            if worker_id in self._worker_registry:
                worker_info = self._worker_registry[worker_id]
                worker_info.last_heartbeat = datetime.utcnow()
                
                if status:
                    worker_info.status = status
                
                self._logger.debug(f"Updated heartbeat for worker {worker_id}")
                return True
            else:
                self._logger.warning(f"Worker {worker_id} not found in registry")
                return False
                
        except Exception as e:
            self._logger.error(f"Error updating worker heartbeat: {e}")
            return False

    def get_next_queue_item(self, 
                           queue_name: str, 
                           worker_id: str,
                           priority_filter: Optional[List[BatchPriority]] = None) -> Optional[QueueItem]:
        """
        Get the next available queue item for processing with locking.
        
        Args:
            queue_name: Name of the queue
            worker_id: Worker identifier for locking
            priority_filter: Optional list of priorities to filter by
            
        Returns:
            Next available QueueItem or None if none available
            
        Raises:
            QueueServiceError: If operation fails
        """
        try:
            session = self._get_session()
            queue_lock = self._get_queue_lock(queue_name)
            
            with queue_lock:
                with managed_transaction(session, description=f"Get next queue item for worker {worker_id}") as txn:
                    # Build query for available items
                    query = select(QueueItem).options(
                        selectinload(QueueItem.batch_item).selectinload(BatchItem.batch)
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
                    )
                    
                    # Apply priority filter if provided
                    if priority_filter:
                        query = query.where(QueueItem.priority.in_(priority_filter))
                    
                    # Order by priority and scheduled time
                    if self.options.enable_priority_processing:
                        query = query.order_by(
                            QueueItem.priority.desc(),
                            QueueItem.scheduled_at.asc()
                        )
                    else:
                        query = query.order_by(QueueItem.scheduled_at.asc())
                    
                    query = query.limit(1)
                    
                    result = session.execute(query)
                    queue_item = result.scalar_one_or_none()
                    
                    if queue_item:
                        # Lock the item
                        lock_expires_at = datetime.utcnow() + timedelta(minutes=self.options.lock_timeout_minutes)
                        queue_item.locked_at = datetime.utcnow()
                        queue_item.locked_by = worker_id
                        queue_item.lock_expires_at = lock_expires_at
                        
                        # Update batch item status
                        if queue_item.batch_item:
                            queue_item.batch_item.status = BatchItemStatus.PROCESSING
                            queue_item.batch_item.started_at = datetime.utcnow()
                            
                            # Start monitoring for this item
                            batch = queue_item.batch_item.batch
                            if batch:
                                self._monitor.start_item_monitoring(
                                    queue_item.batch_item_id,
                                    batch.batch_id,
                                    queue_item.batch_item.url
                                )
                                
                                # Log item started
                                self._batch_logger.log_item_event(
                                    batch_id=batch.batch_id,
                                    batch_item_id=queue_item.batch_item_id,
                                    event_type=EventType.ITEM_STARTED,
                                    message=f"Started processing item",
                                    worker_id=worker_id,
                                    context={
                                        "url": queue_item.batch_item.url,
                                        "queue_name": queue_name,
                                        "priority": queue_item.priority.value
                                    }
                                )
                        
                        # Update worker info
                        if worker_id in self._worker_registry:
                            worker_info = self._worker_registry[worker_id]
                            worker_info.status = QueueWorkerStatus.PROCESSING
                            worker_info.current_batch_item_id = queue_item.batch_item_id
                            worker_info.last_heartbeat = datetime.utcnow()
                        
                        # Log operation
                        txn.execute_operation(
                            OperationType.UPDATE,
                            f"Locked queue item {queue_item.id} for worker {worker_id}",
                            "queue_items",
                            lambda: None,
                            target_id=queue_item.id,
                            parameters={
                                "worker_id": worker_id,
                                "lock_expires_at": lock_expires_at.isoformat()
                            }
                        )
                    
                    result = txn.commit_transaction()
                    
                    if queue_item:
                        self._logger.info(f"Retrieved and locked queue item {queue_item.id} for worker {worker_id}")
                    
                    return queue_item
                    
        except SQLAlchemyError as e:
            db_error = classify_database_error(e)
            self._logger.error(f"Database error getting queue item: {db_error}")
            raise QueueServiceError(f"Failed to get queue item: {db_error.message}")
        except Exception as e:
            self._logger.error(f"Unexpected error getting queue item: {e}")
            raise QueueServiceError(f"Unexpected error: {e}")

    def release_queue_item(self, queue_item_id: int, worker_id: str) -> bool:
        """
        Release a locked queue item without completing it.
        
        Args:
            queue_item_id: Queue item ID
            worker_id: Worker identifier
            
        Returns:
            True if released successfully, False otherwise
            
        Raises:
            QueueServiceError: If operation fails
        """
        try:
            session = self._get_session()
            
            with managed_transaction(session, description=f"Release queue item {queue_item_id}") as txn:
                # Get and validate queue item
                queue_item = session.get(QueueItem, queue_item_id)
                if not queue_item:
                    raise QueueServiceError(f"Queue item {queue_item_id} not found")
                
                if queue_item.locked_by != worker_id:
                    raise QueueServiceError(f"Queue item {queue_item_id} is not locked by worker {worker_id}")
                
                # Release the lock
                queue_item.locked_at = None
                queue_item.locked_by = None
                queue_item.lock_expires_at = None
                
                # Update batch item status
                if queue_item.batch_item:
                    queue_item.batch_item.status = BatchItemStatus.QUEUED
                    queue_item.batch_item.started_at = None
                
                # Update worker info
                if worker_id in self._worker_registry:
                    worker_info = self._worker_registry[worker_id]
                    worker_info.status = QueueWorkerStatus.IDLE
                    worker_info.current_batch_item_id = None
                    worker_info.last_heartbeat = datetime.utcnow()
                
                # Log operation
                txn.execute_operation(
                    OperationType.UPDATE,
                    f"Released queue item {queue_item_id} from worker {worker_id}",
                    "queue_items",
                    lambda: None,
                    target_id=queue_item_id,
                    parameters={"worker_id": worker_id}
                )
                
                result = txn.commit_transaction()
                
                self._logger.info(f"Released queue item {queue_item_id} from worker {worker_id}")
                return True
                
        except SQLAlchemyError as e:
            db_error = classify_database_error(e)
            self._logger.error(f"Database error releasing queue item: {db_error}")
            raise QueueServiceError(f"Failed to release queue item: {db_error.message}")
        except Exception as e:
            self._logger.error(f"Unexpected error releasing queue item: {e}")
            raise QueueServiceError(f"Unexpected error: {e}")

    def complete_queue_item(self, 
                           queue_item_id: int, 
                           worker_id: str,
                           status: BatchItemStatus,
                           result_data: Optional[Dict[str, Any]] = None,
                           error_message: Optional[str] = None) -> bool:
        """
        Complete processing of a queue item.
        
        Args:
            queue_item_id: Queue item ID
            worker_id: Worker identifier
            status: Final processing status
            result_data: Optional result data
            error_message: Optional error message
            
        Returns:
            True if completed successfully, False otherwise
            
        Raises:
            QueueServiceError: If operation fails
        """
        try:
            session = self._get_session()
            
            with managed_transaction(session, description=f"Complete queue item {queue_item_id}") as txn:
                # Get and validate queue item
                queue_item = session.get(QueueItem, queue_item_id)
                if not queue_item:
                    raise QueueServiceError(f"Queue item {queue_item_id} not found")
                
                if queue_item.locked_by != worker_id:
                    raise QueueServiceError(f"Queue item {queue_item_id} is not locked by worker {worker_id}")
                
                # Update batch item
                batch_item = queue_item.batch_item
                if batch_item:
                    batch_item.status = status
                    batch_item.completed_at = datetime.utcnow()
                    batch_item.result_data = result_data
                    batch_item.error_info = error_message
                    
                    # Update batch counters
                    batch = batch_item.batch
                    if batch:
                        if status == BatchItemStatus.COMPLETED:
                            batch.completed_items += 1
                        elif status == BatchItemStatus.FAILED:
                            batch.failed_items += 1
                        
                        # Check if batch is complete
                        total_processed = batch.completed_items + batch.failed_items
                        if total_processed >= batch.total_items:
                            batch.status = BatchStatus.COMPLETED
                            batch.completed_at = datetime.utcnow()
                
                # Remove from queue
                session.delete(queue_item)
                
                # Update worker info
                if worker_id in self._worker_registry:
                    worker_info = self._worker_registry[worker_id]
                    worker_info.status = QueueWorkerStatus.IDLE
                    worker_info.current_batch_item_id = None
                    worker_info.last_heartbeat = datetime.utcnow()
                    
                    if status == BatchItemStatus.COMPLETED:
                        worker_info.processed_items += 1
                    elif status == BatchItemStatus.FAILED:
                        worker_info.failed_items += 1
                
                # Log operation
                txn.execute_operation(
                    OperationType.UPDATE,
                    f"Completed queue item {queue_item_id} with status {status.value}",
                    "queue_items",
                    lambda: None,
                    target_id=queue_item_id,
                    parameters={
                        "worker_id": worker_id,
                        "status": status.value
                    }
                )
                
                result = txn.commit_transaction()
                
                self._logger.info(f"Completed queue item {queue_item_id} with status {status.value}")
                return True
                
        except SQLAlchemyError as e:
            db_error = classify_database_error(e)
            self._logger.error(f"Database error completing queue item: {db_error}")
            raise QueueServiceError(f"Failed to complete queue item: {db_error.message}")
        except Exception as e:
            self._logger.error(f"Unexpected error completing queue item: {e}")
            raise QueueServiceError(f"Unexpected error: {e}")

    def retry_queue_item(self, queue_item_id: int, delay_minutes: Optional[int] = None) -> bool:
        """
        Retry a failed queue item.
        
        Args:
            queue_item_id: Queue item ID
            delay_minutes: Optional delay before retry
            
        Returns:
            True if retry was scheduled, False otherwise
            
        Raises:
            QueueServiceError: If operation fails
        """
        try:
            session = self._get_session()
            
            with managed_transaction(session, description=f"Retry queue item {queue_item_id}") as txn:
                # Get queue item
                queue_item = session.get(QueueItem, queue_item_id)
                if not queue_item:
                    raise QueueServiceError(f"Queue item {queue_item_id} not found")
                
                if not queue_item.can_retry:
                    raise QueueServiceError(f"Queue item {queue_item_id} cannot be retried")
                
                # Update queue item
                queue_item.retry_count += 1
                queue_item.locked_at = None
                queue_item.locked_by = None
                queue_item.lock_expires_at = None
                queue_item.error_info = None
                
                # Schedule retry
                retry_delay = delay_minutes or self.options.retry_delay_minutes
                queue_item.scheduled_at = datetime.utcnow() + timedelta(minutes=retry_delay)
                
                # Update batch item
                if queue_item.batch_item:
                    queue_item.batch_item.status = BatchItemStatus.QUEUED
                    queue_item.batch_item.started_at = None
                    queue_item.batch_item.completed_at = None
                    queue_item.batch_item.error_info = None
                    queue_item.batch_item.retry_count = queue_item.retry_count
                
                # Log operation
                txn.execute_operation(
                    OperationType.UPDATE,
                    f"Retried queue item {queue_item_id} (attempt {queue_item.retry_count})",
                    "queue_items",
                    lambda: None,
                    target_id=queue_item_id,
                    parameters={
                        "retry_count": queue_item.retry_count,
                        "scheduled_at": queue_item.scheduled_at.isoformat()
                    }
                )
                
                result = txn.commit_transaction()
                
                self._logger.info(f"Retried queue item {queue_item_id} (attempt {queue_item.retry_count})")
                return True
                
        except SQLAlchemyError as e:
            db_error = classify_database_error(e)
            self._logger.error(f"Database error retrying queue item: {db_error}")
            raise QueueServiceError(f"Failed to retry queue item: {db_error.message}")
        except Exception as e:
            self._logger.error(f"Unexpected error retrying queue item: {e}")
            raise QueueServiceError(f"Unexpected error: {e}")

    def get_queue_statistics(self, queue_name: str) -> QueueStatistics:
        """
        Get comprehensive queue statistics.
        
        Args:
            queue_name: Name of the queue
            
        Returns:
            QueueStatistics object
            
        Raises:
            QueueServiceError: If operation fails
        """
        try:
            session = self._get_session()
            
            # Basic queue item counts
            total_items = session.execute(
                select(func.count(QueueItem.id)).where(QueueItem.queue_name == queue_name)
            ).scalar() or 0
            
            pending_items = session.execute(
                select(func.count(QueueItem.id)).where(
                    and_(
                        QueueItem.queue_name == queue_name,
                        QueueItem.locked_at.is_(None),
                        QueueItem.scheduled_at <= datetime.utcnow()
                    )
                )
            ).scalar() or 0
            
            processing_items = session.execute(
                select(func.count(QueueItem.id)).where(
                    and_(
                        QueueItem.queue_name == queue_name,
                        QueueItem.locked_at.is_not(None),
                        QueueItem.lock_expires_at > datetime.utcnow()
                    )
                )
            ).scalar() or 0
            
            locked_items = session.execute(
                select(func.count(QueueItem.id)).where(
                    and_(
                        QueueItem.queue_name == queue_name,
                        QueueItem.locked_at.is_not(None)
                    )
                )
            ).scalar() or 0
            
            stale_locks = session.execute(
                select(func.count(QueueItem.id)).where(
                    and_(
                        QueueItem.queue_name == queue_name,
                        QueueItem.locked_at.is_not(None),
                        QueueItem.lock_expires_at < datetime.utcnow()
                    )
                )
            ).scalar() or 0
            
            # Priority distribution
            priority_result = session.execute(
                select(QueueItem.priority, func.count(QueueItem.id)).where(
                    QueueItem.queue_name == queue_name
                ).group_by(QueueItem.priority)
            ).all()
            
            priority_distribution = {priority.value: count for priority, count in priority_result}
            
            # Active workers for this queue
            active_workers = sum(1 for worker in self._worker_registry.values() 
                               if worker.queue_name == queue_name and 
                               worker.status in [QueueWorkerStatus.IDLE, QueueWorkerStatus.PROCESSING])
            
            # Batch item statistics
            completed_items = session.execute(
                select(func.count(BatchItem.id)).where(
                    and_(
                        BatchItem.status == BatchItemStatus.COMPLETED,
                        BatchItem.id.in_(
                            select(QueueItem.batch_item_id).where(QueueItem.queue_name == queue_name)
                        )
                    )
                )
            ).scalar() or 0
            
            failed_items = session.execute(
                select(func.count(BatchItem.id)).where(
                    and_(
                        BatchItem.status == BatchItemStatus.FAILED,
                        BatchItem.id.in_(
                            select(QueueItem.batch_item_id).where(QueueItem.queue_name == queue_name)
                        )
                    )
                )
            ).scalar() or 0
            
            # Average processing time
            avg_processing_time = session.execute(
                select(func.avg(
                    func.extract('epoch', BatchItem.completed_at - BatchItem.started_at)
                )).where(
                    and_(
                        BatchItem.status == BatchItemStatus.COMPLETED,
                        BatchItem.started_at.is_not(None),
                        BatchItem.completed_at.is_not(None)
                    )
                )
            ).scalar() or 0.0
            
            # Last processed timestamp
            last_processed_at = session.execute(
                select(func.max(BatchItem.completed_at)).where(
                    BatchItem.status == BatchItemStatus.COMPLETED
                )
            ).scalar()
            
            # Determine health status
            health_status = self._calculate_queue_health(
                total_items, pending_items, processing_items, stale_locks, active_workers
            )
            
            return QueueStatistics(
                queue_name=queue_name,
                total_items=total_items,
                pending_items=pending_items,
                processing_items=processing_items,
                completed_items=completed_items,
                failed_items=failed_items,
                locked_items=locked_items,
                stale_locks=stale_locks,
                active_workers=active_workers,
                average_processing_time=avg_processing_time,
                last_processed_at=last_processed_at,
                health_status=health_status,
                priority_distribution=priority_distribution
            )
            
        except SQLAlchemyError as e:
            db_error = classify_database_error(e)
            self._logger.error(f"Database error getting queue statistics: {db_error}")
            raise QueueServiceError(f"Failed to get queue statistics: {db_error.message}")
        except Exception as e:
            self._logger.error(f"Unexpected error getting queue statistics: {e}")
            raise QueueServiceError(f"Unexpected error: {e}")

    def _calculate_queue_health(self, 
                               total_items: int, 
                               pending_items: int, 
                               processing_items: int,
                               stale_locks: int, 
                               active_workers: int) -> QueueHealthStatus:
        """Calculate queue health status based on metrics."""
        if active_workers == 0:
            return QueueHealthStatus.OFFLINE
        
        if stale_locks > 0:
            stale_ratio = stale_locks / max(total_items, 1)
            if stale_ratio > 0.1:  # More than 10% stale locks
                return QueueHealthStatus.CRITICAL
            elif stale_ratio > 0.05:  # More than 5% stale locks
                return QueueHealthStatus.WARNING
        
        if pending_items > 0:
            pending_ratio = pending_items / max(total_items, 1)
            if pending_ratio > 0.8:  # More than 80% pending
                return QueueHealthStatus.WARNING
        
        return QueueHealthStatus.HEALTHY

    def get_worker_statistics(self, worker_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get worker statistics.
        
        Args:
            worker_id: Optional specific worker ID. If not provided, returns all workers.
            
        Returns:
            Dictionary containing worker statistics
        """
        try:
            if worker_id:
                if worker_id in self._worker_registry:
                    worker = self._worker_registry[worker_id]
                    return {
                        "worker_id": worker.worker_id,
                        "queue_name": worker.queue_name,
                        "status": worker.status.value,
                        "last_heartbeat": worker.last_heartbeat.isoformat(),
                        "started_at": worker.started_at.isoformat(),
                        "processed_items": worker.processed_items,
                        "failed_items": worker.failed_items,
                        "current_batch_item_id": worker.current_batch_item_id,
                        "uptime_seconds": (datetime.utcnow() - worker.started_at).total_seconds()
                    }
                else:
                    return {}
            else:
                return {
                    "total_workers": len(self._worker_registry),
                    "active_workers": sum(1 for w in self._worker_registry.values() 
                                        if w.status in [QueueWorkerStatus.IDLE, QueueWorkerStatus.PROCESSING]),
                    "workers": [
                        {
                            "worker_id": w.worker_id,
                            "queue_name": w.queue_name,
                            "status": w.status.value,
                            "last_heartbeat": w.last_heartbeat.isoformat(),
                            "processed_items": w.processed_items,
                            "failed_items": w.failed_items,
                            "current_batch_item_id": w.current_batch_item_id
                        } for w in self._worker_registry.values()
                    ]
                }
                
        except Exception as e:
            self._logger.error(f"Error getting worker statistics: {e}")
            raise QueueServiceError(f"Failed to get worker statistics: {e}")

    def pause_queue(self, queue_name: str) -> bool:
        """
        Pause processing for a specific queue.
        
        Args:
            queue_name: Name of the queue to pause
            
        Returns:
            True if paused successfully, False otherwise
        """
        try:
            paused_workers = 0
            for worker in self._worker_registry.values():
                if worker.queue_name == queue_name and worker.status != QueueWorkerStatus.STOPPED:
                    worker.status = QueueWorkerStatus.PAUSED
                    paused_workers += 1
            
            self._logger.info(f"Paused queue {queue_name} with {paused_workers} workers")
            return True
            
        except Exception as e:
            self._logger.error(f"Error pausing queue {queue_name}: {e}")
            return False

    def resume_queue(self, queue_name: str) -> bool:
        """
        Resume processing for a specific queue.
        
        Args:
            queue_name: Name of the queue to resume
            
        Returns:
            True if resumed successfully, False otherwise
        """
        try:
            resumed_workers = 0
            for worker in self._worker_registry.values():
                if worker.queue_name == queue_name and worker.status == QueueWorkerStatus.PAUSED:
                    worker.status = QueueWorkerStatus.IDLE
                    resumed_workers += 1
            
            self._logger.info(f"Resumed queue {queue_name} with {resumed_workers} workers")
            return True
            
        except Exception as e:
            self._logger.error(f"Error resuming queue {queue_name}: {e}")
            return False

    def _cleanup_stale_locks(self) -> int:
        """Clean up stale locks from the database."""
        try:
            session = self._get_session()
            
            cutoff_time = datetime.utcnow()
            
            # Find and release stale locks
            stale_items = session.execute(
                select(QueueItem).where(
                    and_(
                        QueueItem.locked_at.is_not(None),
                        QueueItem.lock_expires_at < cutoff_time
                    )
                )
            ).scalars().all()
            
            cleaned_count = 0
            for item in stale_items:
                # Release the lock
                item.locked_at = None
                item.locked_by = None
                item.lock_expires_at = None
                
                # Update batch item status
                if item.batch_item:
                    item.batch_item.status = BatchItemStatus.QUEUED
                    item.batch_item.started_at = None
                
                cleaned_count += 1
            
            if cleaned_count > 0:
                session.commit()
                self._logger.info(f"Cleaned up {cleaned_count} stale locks")
            
            return cleaned_count
            
        except Exception as e:
            self._logger.error(f"Error cleaning up stale locks: {e}")
            return 0

    def _cleanup_stale_sessions(self) -> int:
        """Clean up stale processing sessions."""
        try:
            session = self._get_session()
            
            cutoff_time = datetime.utcnow() - timedelta(minutes=self.options.worker_timeout_minutes)
            
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
            
            if cleaned_count > 0:
                session.commit()
                self._logger.info(f"Cleaned up {cleaned_count} stale processing sessions")
            
            return cleaned_count
            
        except Exception as e:
            self._logger.error(f"Error cleaning up stale sessions: {e}")
            return 0

    def _cleanup_inactive_workers(self) -> int:
        """Clean up inactive workers from the registry."""
        try:
            cutoff_time = datetime.utcnow() - timedelta(minutes=self.options.worker_timeout_minutes)
            
            inactive_workers = []
            for worker_id, worker in self._worker_registry.items():
                if worker.last_heartbeat < cutoff_time:
                    inactive_workers.append(worker_id)
            
            for worker_id in inactive_workers:
                self._logger.warning(f"Removing inactive worker {worker_id}")
                self._release_worker_locks(worker_id)
                del self._worker_registry[worker_id]
            
            return len(inactive_workers)
            
        except Exception as e:
            self._logger.error(f"Error cleaning up inactive workers: {e}")
            return 0

    def _release_worker_locks(self, worker_id: str):
        """Release all locks held by a specific worker."""
        try:
            session = self._get_session()
            
            # Release all locks held by this worker
            session.execute(
                update(QueueItem).where(
                    QueueItem.locked_by == worker_id
                ).values(
                    locked_at=None,
                    locked_by=None,
                    lock_expires_at=None
                )
            )
            
            session.commit()
            self._logger.info(f"Released all locks for worker {worker_id}")
            
        except Exception as e:
            self._logger.error(f"Error releasing locks for worker {worker_id}: {e}")

    def shutdown(self):
        """Shutdown the queue service and cleanup resources."""
        try:
            self._logger.info("Shutting down queue service...")
            
            # Stop cleanup thread
            self._cleanup_running = False
            if self._cleanup_thread:
                self._cleanup_thread.join(timeout=5)
            
            # Stop all workers
            for worker_id in list(self._worker_registry.keys()):
                self.unregister_worker(worker_id)
            
            # Clean up any remaining locks
            self._cleanup_stale_locks()
            
            self._logger.info("Queue service shutdown complete")
            
        except Exception as e:
            self._logger.error(f"Error during shutdown: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()


# Dependency injection function for FastAPI
def get_queue_service(session: Session = None, 
                     options: Optional[QueueProcessingOptions] = None) -> QueueService:
    """
    Get QueueService instance for dependency injection.
    
    Args:
        session: Database session
        options: Queue processing options
        
    Returns:
        QueueService instance
    """
    return QueueService(session, options)