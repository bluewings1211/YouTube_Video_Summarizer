"""
Concurrent batch processing service for YouTube video summarization.

This service provides advanced concurrency control for batch processing operations including:
- Thread-safe batch operations with proper locking
- Concurrent worker management with resource allocation
- Rate limiting and throttling for API calls
- Data consistency across concurrent operations
- Deadlock prevention and recovery
- Performance monitoring and optimization
"""

import asyncio
import logging
import threading
import time
from typing import Optional, Dict, Any, List, Callable, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from contextlib import contextmanager, asynccontextmanager
import uuid
import weakref
from enum import Enum

from sqlalchemy.orm import Session
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
from ..services.batch_service import BatchService, BatchServiceError, BatchCreateRequest
from ..services.batch_processor import BatchProcessor, BatchProcessorError
from ..services.queue_service import QueueService, QueueServiceError, QueueProcessingOptions
from ..utils.concurrency_manager import (
    ConcurrencyManager, ResourceType, ResourceQuota, 
    acquire_shared_lock, acquire_exclusive_lock, allocate_resource,
    get_global_concurrency_manager, ConcurrencyError, ResourceExhaustedError,
    TimeoutError as ConcurrencyTimeoutError, RateLimitedError
)
from ..flow import YouTubeSummarizerFlow, WorkflowConfig


logger = logging.getLogger(__name__)


class ConcurrentBatchMode(Enum):
    """Concurrency modes for batch processing."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    ADAPTIVE = "adaptive"
    BOUNDED_PARALLEL = "bounded_parallel"


class WorkerState(Enum):
    """States for concurrent workers."""
    INITIALIZING = "initializing"
    IDLE = "idle"
    PROCESSING = "processing"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class ConcurrentBatchConfig:
    """Configuration for concurrent batch processing."""
    max_concurrent_batches: int = 5
    max_concurrent_items_per_batch: int = 10
    max_total_concurrent_items: int = 50
    max_workers_per_batch: int = 3
    max_api_calls_per_second: float = 2.0
    max_database_connections: int = 10
    worker_timeout_seconds: float = 300.0
    batch_timeout_seconds: float = 3600.0
    enable_rate_limiting: bool = True
    enable_resource_throttling: bool = True
    enable_deadlock_detection: bool = True
    enable_performance_monitoring: bool = True
    cleanup_interval_seconds: int = 60
    heartbeat_interval_seconds: int = 30
    retry_failed_items: bool = True
    max_retries_per_item: int = 3
    backoff_multiplier: float = 2.0
    min_backoff_seconds: float = 1.0
    max_backoff_seconds: float = 60.0


@dataclass
class WorkerInfo:
    """Information about a concurrent worker."""
    worker_id: str
    batch_id: str
    worker_state: WorkerState
    current_item_id: Optional[int] = None
    started_at: datetime = field(default_factory=datetime.utcnow)
    last_heartbeat: datetime = field(default_factory=datetime.utcnow)
    processed_items: int = 0
    failed_items: int = 0
    total_processing_time: float = 0.0
    current_operation: Optional[str] = None
    error_count: int = 0
    last_error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConcurrentBatchStatistics:
    """Statistics for concurrent batch processing."""
    total_batches: int = 0
    active_batches: int = 0
    completed_batches: int = 0
    failed_batches: int = 0
    total_items: int = 0
    processing_items: int = 0
    completed_items: int = 0
    failed_items: int = 0
    active_workers: int = 0
    total_workers: int = 0
    average_processing_time: float = 0.0
    throughput_items_per_second: float = 0.0
    resource_utilization: Dict[str, float] = field(default_factory=dict)
    error_rate: float = 0.0
    retry_rate: float = 0.0
    last_updated: datetime = field(default_factory=datetime.utcnow)


class ConcurrentBatchError(Exception):
    """Base exception for concurrent batch processing errors."""
    pass


class ConcurrentWorkerError(ConcurrentBatchError):
    """Exception for worker-related errors."""
    pass


class ConcurrentBatchTimeoutError(ConcurrentBatchError):
    """Exception for batch timeout errors."""
    pass


class ConcurrentBatchService:
    """
    Advanced concurrent batch processing service.
    
    This service provides:
    - Thread-safe batch operations with proper locking
    - Concurrent worker management with resource allocation
    - Rate limiting and throttling for external API calls
    - Data consistency across concurrent operations
    - Deadlock prevention and recovery mechanisms
    - Performance monitoring and optimization
    """
    
    def __init__(self, 
                 config: Optional[ConcurrentBatchConfig] = None,
                 session: Optional[Session] = None,
                 concurrency_manager: Optional[ConcurrencyManager] = None):
        """
        Initialize concurrent batch service.
        
        Args:
            config: Configuration for concurrent batch processing
            session: Database session (optional)
            concurrency_manager: Concurrency manager instance (optional)
        """
        self.config = config or ConcurrentBatchConfig()
        self._session = session
        self._concurrency_manager = concurrency_manager or get_global_concurrency_manager()
        
        # Core services
        self._batch_service: Optional[BatchService] = None
        self._queue_service: Optional[QueueService] = None
        self._batch_processor: Optional[BatchProcessor] = None
        
        # Worker management
        self._worker_registry: Dict[str, WorkerInfo] = {}
        self._worker_futures: Dict[str, Future] = {}
        self._worker_executor = ThreadPoolExecutor(
            max_workers=self.config.max_total_concurrent_items,
            thread_name_prefix="concurrent_batch_worker"
        )
        
        # Batch tracking
        self._active_batches: Set[str] = set()
        self._batch_workers: Dict[str, Set[str]] = {}  # batch_id -> set of worker_ids
        self._batch_locks: Dict[str, threading.RLock] = {}
        
        # Performance monitoring
        self._statistics = ConcurrentBatchStatistics()
        self._performance_metrics = {}
        
        # Synchronization
        self._global_lock = threading.RLock()
        self._shutdown_event = threading.Event()
        
        # Background tasks
        self._monitoring_thread: Optional[threading.Thread] = None
        self._cleanup_thread: Optional[threading.Thread] = None
        
        # Logging
        self._logger = logging.getLogger(f"{__name__}.ConcurrentBatchService")
        
        # Configure resource quotas
        self._configure_resource_quotas()
        
        # Start background tasks
        self._start_background_tasks()
    
    def _configure_resource_quotas(self):
        """Configure resource quotas for concurrency management."""
        # Database connection quota
        db_quota = ResourceQuota(
            max_concurrent=self.config.max_database_connections,
            max_per_second=20.0,
            burst_capacity=self.config.max_database_connections * 2,
            timeout_seconds=30.0
        )
        self._concurrency_manager.configure_resource_quota(ResourceType.DATABASE_CONNECTION, db_quota)
        
        # API request quota
        api_quota = ResourceQuota(
            max_concurrent=self.config.max_total_concurrent_items,
            max_per_second=self.config.max_api_calls_per_second,
            burst_capacity=int(self.config.max_api_calls_per_second * 5),
            timeout_seconds=60.0
        )
        self._concurrency_manager.configure_resource_quota(ResourceType.API_REQUEST, api_quota)
        
        # Worker thread quota
        worker_quota = ResourceQuota(
            max_concurrent=self.config.max_total_concurrent_items,
            max_per_second=10.0,
            burst_capacity=self.config.max_total_concurrent_items * 2,
            timeout_seconds=self.config.worker_timeout_seconds
        )
        self._concurrency_manager.configure_resource_quota(ResourceType.WORKER_THREAD, worker_quota)
    
    def _start_background_tasks(self):
        """Start background monitoring and cleanup tasks."""
        if self.config.enable_performance_monitoring:
            self._monitoring_thread = threading.Thread(
                target=self._monitoring_worker,
                daemon=True,
                name="concurrent_batch_monitor"
            )
            self._monitoring_thread.start()
        
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_worker,
            daemon=True,
            name="concurrent_batch_cleanup"
        )
        self._cleanup_thread.start()
        
        self._logger.info("Started background tasks for concurrent batch service")
    
    def _get_session(self) -> Session:
        """Get database session."""
        if self._session:
            return self._session
        else:
            return get_database_session()
    
    def _get_batch_service(self, session: Session) -> BatchService:
        """Get batch service instance."""
        if self._batch_service is None:
            self._batch_service = BatchService(session)
        return self._batch_service
    
    def _get_queue_service(self, session: Session) -> QueueService:
        """Get queue service instance."""
        if self._queue_service is None:
            options = QueueProcessingOptions(
                max_workers=self.config.max_total_concurrent_items,
                worker_timeout_minutes=int(self.config.worker_timeout_seconds / 60),
                lock_timeout_minutes=int(self.config.worker_timeout_seconds / 60),
                heartbeat_interval_seconds=self.config.heartbeat_interval_seconds,
                enable_priority_processing=True,
                enable_worker_monitoring=True
            )
            self._queue_service = QueueService(session, options)
        return self._queue_service
    
    def _get_batch_processor(self) -> BatchProcessor:
        """Get batch processor instance."""
        if self._batch_processor is None:
            workflow_config = WorkflowConfig()
            self._batch_processor = BatchProcessor(
                workflow_config=workflow_config
            )
        return self._batch_processor
    
    def _get_batch_lock(self, batch_id: str) -> threading.RLock:
        """Get or create a lock for a specific batch."""
        with self._global_lock:
            if batch_id not in self._batch_locks:
                self._batch_locks[batch_id] = threading.RLock()
            return self._batch_locks[batch_id]
    
    async def create_concurrent_batch(self, 
                                    request: BatchCreateRequest,
                                    concurrency_mode: ConcurrentBatchMode = ConcurrentBatchMode.ADAPTIVE,
                                    max_concurrent_items: Optional[int] = None) -> Batch:
        """
        Create a new batch for concurrent processing.
        
        Args:
            request: Batch creation request
            concurrency_mode: Concurrency mode for processing
            max_concurrent_items: Maximum concurrent items for this batch
            
        Returns:
            Created Batch object
            
        Raises:
            ConcurrentBatchError: If creation fails
        """
        try:
            # Check if we can create a new batch
            with self._global_lock:
                if len(self._active_batches) >= self.config.max_concurrent_batches:
                    raise ConcurrentBatchError(
                        f"Maximum concurrent batches ({self.config.max_concurrent_batches}) reached"
                    )
            
            # Allocate database resource
            with allocate_resource(
                ResourceType.DATABASE_CONNECTION,
                f"create_batch_{uuid.uuid4().hex[:8]}",
                priority="high",
                timeout=30.0
            ):
                session = self._get_session()
                batch_service = self._get_batch_service(session)
                
                # Create the batch
                batch = batch_service.create_batch(request)
                
                # Register the batch for concurrent processing
                with self._global_lock:
                    self._active_batches.add(batch.batch_id)
                    self._batch_workers[batch.batch_id] = set()
                    
                    # Update statistics
                    self._statistics.total_batches += 1
                    self._statistics.active_batches += 1
                    self._statistics.total_items += batch.total_items
                
                # Store batch configuration
                batch_metadata = batch.batch_metadata or {}
                batch_metadata.update({
                    'concurrency_mode': concurrency_mode.value,
                    'max_concurrent_items': max_concurrent_items or self.config.max_concurrent_items_per_batch,
                    'created_by_concurrent_service': True
                })
                batch.batch_metadata = batch_metadata
                
                session.commit()
                
                self._logger.info(f"Created concurrent batch {batch.batch_id} with {batch.total_items} items")
                return batch
                
        except Exception as e:
            self._logger.error(f"Error creating concurrent batch: {e}")
            raise ConcurrentBatchError(f"Failed to create concurrent batch: {e}")
    
    async def process_batch_concurrently(self, 
                                       batch_id: str,
                                       max_workers: Optional[int] = None) -> Dict[str, Any]:
        """
        Process a batch using concurrent workers.
        
        Args:
            batch_id: Batch identifier
            max_workers: Maximum number of workers to use
            
        Returns:
            Processing results dictionary
            
        Raises:
            ConcurrentBatchError: If processing fails
        """
        try:
            # Acquire exclusive lock for this batch
            lock_id = f"batch_processing_{batch_id}"
            with acquire_exclusive_lock(lock_id, f"process_batch_{batch_id}", timeout=60.0):
                
                # Get batch information
                session = self._get_session()
                batch_service = self._get_batch_service(session)
                batch = batch_service.get_batch(batch_id)
                
                if not batch:
                    raise ConcurrentBatchError(f"Batch {batch_id} not found")
                
                if batch.status != BatchStatus.PENDING:
                    raise ConcurrentBatchError(f"Batch {batch_id} is not in PENDING status")
                
                # Start batch processing
                batch_service.start_batch_processing(batch_id)
                
                # Determine worker count
                max_workers = max_workers or min(
                    self.config.max_workers_per_batch,
                    batch.total_items,
                    self.config.max_concurrent_items_per_batch
                )
                
                # Create and start workers
                workers = []
                for i in range(max_workers):
                    worker_id = f"worker_{batch_id}_{i}_{uuid.uuid4().hex[:8]}"
                    worker = self._create_batch_worker(batch_id, worker_id)
                    workers.append(worker)
                
                # Start all workers
                worker_tasks = []
                for worker in workers:
                    future = self._worker_executor.submit(self._run_batch_worker, worker)
                    worker_tasks.append(future)
                    self._worker_futures[worker.worker_id] = future
                
                self._logger.info(f"Started {len(workers)} workers for batch {batch_id}")
                
                # Monitor batch progress
                start_time = time.time()
                timeout = self.config.batch_timeout_seconds
                
                while True:
                    # Check if batch is complete
                    session = self._get_session()
                    batch_service = self._get_batch_service(session)
                    progress = batch_service.get_batch_progress(batch_id)
                    
                    if not progress:
                        break
                    
                    if progress.status in [BatchStatus.COMPLETED, BatchStatus.CANCELLED, BatchStatus.FAILED]:
                        break
                    
                    # Check timeout
                    if time.time() - start_time > timeout:
                        self._logger.warning(f"Batch {batch_id} processing timeout")
                        await self._cancel_batch_workers(batch_id)
                        raise ConcurrentBatchTimeoutError(f"Batch {batch_id} processing timed out")
                    
                    # Brief wait before next check
                    await asyncio.sleep(5)
                
                # Wait for all workers to complete
                await self._wait_for_workers_completion(worker_tasks, timeout=60.0)
                
                # Get final results
                final_progress = batch_service.get_batch_progress(batch_id)
                
                # Clean up batch tracking
                with self._global_lock:
                    self._active_batches.discard(batch_id)
                    self._batch_workers.pop(batch_id, None)
                    
                    # Update statistics
                    self._statistics.active_batches -= 1
                    if final_progress:
                        if final_progress.status == BatchStatus.COMPLETED:
                            self._statistics.completed_batches += 1
                        elif final_progress.status == BatchStatus.FAILED:
                            self._statistics.failed_batches += 1
                
                results = {
                    'batch_id': batch_id,
                    'status': final_progress.status.value if final_progress else 'unknown',
                    'total_items': final_progress.total_items if final_progress else 0,
                    'completed_items': final_progress.completed_items if final_progress else 0,
                    'failed_items': final_progress.failed_items if final_progress else 0,
                    'progress_percentage': final_progress.progress_percentage if final_progress else 0,
                    'workers_used': len(workers),
                    'processing_time_seconds': time.time() - start_time
                }
                
                self._logger.info(f"Completed batch {batch_id} processing: {results}")
                return results
                
        except Exception as e:
            self._logger.error(f"Error processing batch {batch_id}: {e}")
            
            # Clean up on error
            await self._cancel_batch_workers(batch_id)
            raise ConcurrentBatchError(f"Failed to process batch {batch_id}: {e}")
    
    def _create_batch_worker(self, batch_id: str, worker_id: str) -> WorkerInfo:
        """Create a new batch worker."""
        worker = WorkerInfo(
            worker_id=worker_id,
            batch_id=batch_id,
            worker_state=WorkerState.INITIALIZING
        )
        
        with self._global_lock:
            self._worker_registry[worker_id] = worker
            self._batch_workers[batch_id].add(worker_id)
            self._statistics.total_workers += 1
            self._statistics.active_workers += 1
        
        self._logger.debug(f"Created worker {worker_id} for batch {batch_id}")
        return worker
    
    def _run_batch_worker(self, worker: WorkerInfo):
        """Run a batch worker in a separate thread."""
        worker_id = worker.worker_id
        batch_id = worker.batch_id
        
        try:
            worker.worker_state = WorkerState.IDLE
            self._logger.info(f"Started worker {worker_id} for batch {batch_id}")
            
            while not self._shutdown_event.is_set():
                try:
                    # Allocate resources for processing
                    with allocate_resource(
                        ResourceType.WORKER_THREAD,
                        worker_id,
                        priority="normal",
                        timeout=30.0
                    ):
                        # Get next item to process
                        queue_item = self._get_next_queue_item(batch_id, worker_id)
                        
                        if not queue_item:
                            # No more items, worker can exit
                            break
                        
                        # Process the item
                        self._process_queue_item(worker, queue_item)
                        
                        # Update heartbeat
                        worker.last_heartbeat = datetime.utcnow()
                    
                except ConcurrencyError as e:
                    self._logger.warning(f"Concurrency error in worker {worker_id}: {e}")
                    worker.error_count += 1
                    time.sleep(min(worker.error_count * 2, 30))  # Exponential backoff
                    
                except Exception as e:
                    self._logger.error(f"Error in worker {worker_id}: {e}")
                    worker.error_count += 1
                    worker.last_error = str(e)
                    worker.worker_state = WorkerState.ERROR
                    
                    if worker.error_count >= 5:
                        self._logger.error(f"Worker {worker_id} exceeded error limit, stopping")
                        break
                    
                    time.sleep(min(worker.error_count * 5, 60))  # Exponential backoff
                    worker.worker_state = WorkerState.IDLE
            
        except Exception as e:
            self._logger.error(f"Fatal error in worker {worker_id}: {e}")
            worker.worker_state = WorkerState.ERROR
        
        finally:
            # Clean up worker
            self._cleanup_worker(worker_id)
            self._logger.info(f"Worker {worker_id} stopped")
    
    def _get_next_queue_item(self, batch_id: str, worker_id: str) -> Optional[QueueItem]:
        """Get next queue item for processing."""
        try:
            with allocate_resource(
                ResourceType.DATABASE_CONNECTION,
                f"get_queue_item_{worker_id}",
                priority="normal",
                timeout=30.0
            ):
                session = self._get_session()
                queue_service = self._get_queue_service(session)
                
                # Get next item
                queue_item = queue_service.get_next_queue_item(
                    queue_name='video_processing',
                    worker_id=worker_id
                )
                
                # Verify item belongs to our batch
                if queue_item and queue_item.batch_item:
                    if queue_item.batch_item.batch.batch_id != batch_id:
                        # Release item if it's not from our batch
                        queue_service.release_queue_item(queue_item.id, worker_id)
                        return None
                
                return queue_item
                
        except Exception as e:
            self._logger.error(f"Error getting next queue item for worker {worker_id}: {e}")
            return None
    
    def _process_queue_item(self, worker: WorkerInfo, queue_item: QueueItem):
        """Process a queue item."""
        worker_id = worker.worker_id
        batch_item_id = queue_item.batch_item_id
        
        try:
            worker.worker_state = WorkerState.PROCESSING
            worker.current_item_id = batch_item_id
            worker.current_operation = "processing_item"
            
            start_time = time.time()
            
            # Allocate API resources
            with allocate_resource(
                ResourceType.API_REQUEST,
                f"process_item_{worker_id}",
                priority="normal",
                timeout=120.0
            ):
                # Get batch processor
                batch_processor = self._get_batch_processor()
                
                # Process the item
                result = asyncio.run(
                    batch_processor.process_batch_item(batch_item_id, worker_id)
                )
                
                # Complete the queue item
                with allocate_resource(
                    ResourceType.DATABASE_CONNECTION,
                    f"complete_item_{worker_id}",
                    priority="normal",
                    timeout=30.0
                ):
                    session = self._get_session()
                    queue_service = self._get_queue_service(session)
                    
                    queue_service.complete_queue_item(
                        queue_item.id,
                        worker_id,
                        result.status,
                        result.result_data,
                        result.error_message
                    )
                
                # Update worker statistics
                processing_time = time.time() - start_time
                worker.total_processing_time += processing_time
                
                if result.status == BatchItemStatus.COMPLETED:
                    worker.processed_items += 1
                    with self._global_lock:
                        self._statistics.completed_items += 1
                else:
                    worker.failed_items += 1
                    with self._global_lock:
                        self._statistics.failed_items += 1
                
                self._logger.debug(f"Worker {worker_id} processed item {batch_item_id} in {processing_time:.2f}s")
                
        except Exception as e:
            self._logger.error(f"Error processing item {batch_item_id} in worker {worker_id}: {e}")
            worker.failed_items += 1
            worker.last_error = str(e)
            
            # Try to fail the queue item
            try:
                with allocate_resource(
                    ResourceType.DATABASE_CONNECTION,
                    f"fail_item_{worker_id}",
                    priority="normal",
                    timeout=30.0
                ):
                    session = self._get_session()
                    queue_service = self._get_queue_service(session)
                    
                    queue_service.complete_queue_item(
                        queue_item.id,
                        worker_id,
                        BatchItemStatus.FAILED,
                        None,
                        str(e)
                    )
            except Exception as cleanup_error:
                self._logger.error(f"Error failing queue item {queue_item.id}: {cleanup_error}")
        
        finally:
            worker.worker_state = WorkerState.IDLE
            worker.current_item_id = None
            worker.current_operation = None
    
    async def _wait_for_workers_completion(self, worker_tasks: List[Future], timeout: float = 60.0):
        """Wait for all workers to complete."""
        try:
            # Wait for all tasks to complete
            for future in as_completed(worker_tasks, timeout=timeout):
                try:
                    future.result()
                except Exception as e:
                    self._logger.error(f"Worker task failed: {e}")
                    
        except Exception as e:
            self._logger.error(f"Error waiting for workers: {e}")
    
    async def _cancel_batch_workers(self, batch_id: str):
        """Cancel all workers for a batch."""
        try:
            with self._global_lock:
                worker_ids = self._batch_workers.get(batch_id, set()).copy()
            
            for worker_id in worker_ids:
                try:
                    # Cancel the worker future
                    if worker_id in self._worker_futures:
                        future = self._worker_futures[worker_id]
                        future.cancel()
                    
                    # Update worker state
                    if worker_id in self._worker_registry:
                        self._worker_registry[worker_id].worker_state = WorkerState.STOPPING
                    
                except Exception as e:
                    self._logger.error(f"Error cancelling worker {worker_id}: {e}")
            
            self._logger.info(f"Cancelled {len(worker_ids)} workers for batch {batch_id}")
            
        except Exception as e:
            self._logger.error(f"Error cancelling batch workers: {e}")
    
    def _cleanup_worker(self, worker_id: str):
        """Clean up a worker."""
        try:
            with self._global_lock:
                # Remove from registry
                if worker_id in self._worker_registry:
                    worker = self._worker_registry.pop(worker_id)
                    worker.worker_state = WorkerState.STOPPED
                    self._statistics.active_workers -= 1
                
                # Remove from batch workers
                for batch_id, workers in self._batch_workers.items():
                    workers.discard(worker_id)
                
                # Remove future
                if worker_id in self._worker_futures:
                    del self._worker_futures[worker_id]
            
            self._logger.debug(f"Cleaned up worker {worker_id}")
            
        except Exception as e:
            self._logger.error(f"Error cleaning up worker {worker_id}: {e}")
    
    def _monitoring_worker(self):
        """Background monitoring worker."""
        while not self._shutdown_event.is_set():
            try:
                self._update_statistics()
                time.sleep(self.config.heartbeat_interval_seconds)
            except Exception as e:
                self._logger.error(f"Error in monitoring worker: {e}")
                time.sleep(60)
    
    def _cleanup_worker(self):
        """Background cleanup worker."""
        while not self._shutdown_event.is_set():
            try:
                self._cleanup_stale_workers()
                self._cleanup_unused_locks()
                time.sleep(self.config.cleanup_interval_seconds)
            except Exception as e:
                self._logger.error(f"Error in cleanup worker: {e}")
                time.sleep(60)
    
    def _update_statistics(self):
        """Update performance statistics."""
        try:
            with self._global_lock:
                # Update worker statistics
                active_workers = sum(1 for w in self._worker_registry.values() 
                                   if w.worker_state in [WorkerState.PROCESSING, WorkerState.IDLE])
                self._statistics.active_workers = active_workers
                
                # Calculate averages
                if self._statistics.completed_items > 0:
                    total_time = sum(w.total_processing_time for w in self._worker_registry.values())
                    self._statistics.average_processing_time = total_time / self._statistics.completed_items
                
                # Calculate error rate
                total_operations = self._statistics.completed_items + self._statistics.failed_items
                if total_operations > 0:
                    self._statistics.error_rate = self._statistics.failed_items / total_operations
                
                # Update timestamp
                self._statistics.last_updated = datetime.utcnow()
                
        except Exception as e:
            self._logger.error(f"Error updating statistics: {e}")
    
    def _cleanup_stale_workers(self):
        """Clean up stale workers."""
        try:
            cutoff_time = datetime.utcnow() - timedelta(seconds=self.config.worker_timeout_seconds)
            stale_workers = []
            
            with self._global_lock:
                for worker_id, worker in self._worker_registry.items():
                    if worker.last_heartbeat < cutoff_time:
                        stale_workers.append(worker_id)
            
            for worker_id in stale_workers:
                self._logger.warning(f"Cleaning up stale worker {worker_id}")
                self._cleanup_worker(worker_id)
                
        except Exception as e:
            self._logger.error(f"Error cleaning up stale workers: {e}")
    
    def _cleanup_unused_locks(self):
        """Clean up unused batch locks."""
        try:
            with self._global_lock:
                unused_locks = []
                for batch_id in list(self._batch_locks.keys()):
                    if batch_id not in self._active_batches:
                        unused_locks.append(batch_id)
                
                for batch_id in unused_locks:
                    del self._batch_locks[batch_id]
                    
        except Exception as e:
            self._logger.error(f"Error cleaning up unused locks: {e}")
    
    def get_concurrent_statistics(self) -> Dict[str, Any]:
        """Get comprehensive concurrent batch statistics."""
        with self._global_lock:
            stats = {
                'service_statistics': {
                    'total_batches': self._statistics.total_batches,
                    'active_batches': self._statistics.active_batches,
                    'completed_batches': self._statistics.completed_batches,
                    'failed_batches': self._statistics.failed_batches,
                    'total_items': self._statistics.total_items,
                    'processing_items': self._statistics.processing_items,
                    'completed_items': self._statistics.completed_items,
                    'failed_items': self._statistics.failed_items,
                    'active_workers': self._statistics.active_workers,
                    'total_workers': self._statistics.total_workers,
                    'average_processing_time': self._statistics.average_processing_time,
                    'error_rate': self._statistics.error_rate,
                    'last_updated': self._statistics.last_updated.isoformat()
                },
                'active_batches': list(self._active_batches),
                'worker_details': {
                    worker_id: {
                        'batch_id': worker.batch_id,
                        'state': worker.worker_state.value,
                        'current_item_id': worker.current_item_id,
                        'processed_items': worker.processed_items,
                        'failed_items': worker.failed_items,
                        'uptime_seconds': (datetime.utcnow() - worker.started_at).total_seconds(),
                        'last_heartbeat': worker.last_heartbeat.isoformat(),
                        'error_count': worker.error_count,
                        'last_error': worker.last_error
                    }
                    for worker_id, worker in self._worker_registry.items()
                },
                'concurrency_statistics': self._concurrency_manager.get_comprehensive_statistics()
            }
            
            return stats
    
    def pause_batch_processing(self, batch_id: str) -> bool:
        """Pause processing for a specific batch."""
        try:
            with self._global_lock:
                worker_ids = self._batch_workers.get(batch_id, set()).copy()
            
            paused_workers = 0
            for worker_id in worker_ids:
                if worker_id in self._worker_registry:
                    worker = self._worker_registry[worker_id]
                    if worker.worker_state in [WorkerState.IDLE, WorkerState.PROCESSING]:
                        worker.worker_state = WorkerState.PAUSED
                        paused_workers += 1
            
            self._logger.info(f"Paused {paused_workers} workers for batch {batch_id}")
            return True
            
        except Exception as e:
            self._logger.error(f"Error pausing batch {batch_id}: {e}")
            return False
    
    def resume_batch_processing(self, batch_id: str) -> bool:
        """Resume processing for a specific batch."""
        try:
            with self._global_lock:
                worker_ids = self._batch_workers.get(batch_id, set()).copy()
            
            resumed_workers = 0
            for worker_id in worker_ids:
                if worker_id in self._worker_registry:
                    worker = self._worker_registry[worker_id]
                    if worker.worker_state == WorkerState.PAUSED:
                        worker.worker_state = WorkerState.IDLE
                        resumed_workers += 1
            
            self._logger.info(f"Resumed {resumed_workers} workers for batch {batch_id}")
            return True
            
        except Exception as e:
            self._logger.error(f"Error resuming batch {batch_id}: {e}")
            return False
    
    def cancel_batch_processing(self, batch_id: str) -> bool:
        """Cancel processing for a specific batch."""
        try:
            asyncio.run(self._cancel_batch_workers(batch_id))
            
            with self._global_lock:
                self._active_batches.discard(batch_id)
                self._batch_workers.pop(batch_id, None)
            
            self._logger.info(f"Cancelled processing for batch {batch_id}")
            return True
            
        except Exception as e:
            self._logger.error(f"Error cancelling batch {batch_id}: {e}")
            return False
    
    def shutdown(self):
        """Shutdown the concurrent batch service."""
        self._logger.info("Shutting down concurrent batch service...")
        
        try:
            # Signal shutdown
            self._shutdown_event.set()
            
            # Cancel all active batches
            with self._global_lock:
                active_batches = self._active_batches.copy()
            
            for batch_id in active_batches:
                self.cancel_batch_processing(batch_id)
            
            # Shutdown worker executor
            self._worker_executor.shutdown(wait=True, timeout=30)
            
            # Wait for background threads
            if self._monitoring_thread:
                self._monitoring_thread.join(timeout=5)
            
            if self._cleanup_thread:
                self._cleanup_thread.join(timeout=5)
            
            self._logger.info("Concurrent batch service shutdown complete")
            
        except Exception as e:
            self._logger.error(f"Error during shutdown: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()


# Factory functions
def create_concurrent_batch_service(config: Optional[ConcurrentBatchConfig] = None) -> ConcurrentBatchService:
    """
    Create a ConcurrentBatchService instance.
    
    Args:
        config: Optional configuration for concurrent batch processing
        
    Returns:
        ConcurrentBatchService instance
    """
    return ConcurrentBatchService(config)


def get_concurrent_batch_service(session: Optional[Session] = None,
                                config: Optional[ConcurrentBatchConfig] = None) -> ConcurrentBatchService:
    """
    Get ConcurrentBatchService instance for dependency injection.
    
    Args:
        session: Optional database session
        config: Optional configuration
        
    Returns:
        ConcurrentBatchService instance
    """
    return ConcurrentBatchService(config, session)