"""
Status updater service for real-time status updates.
Provides automated status update mechanisms and event handling.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
from sqlalchemy.orm import Session

from .status_service import StatusService
from ..database.connection import get_db_session
from ..database.status_models import ProcessingStatusType, StatusChangeType, ProcessingStatus
from ..utils.error_messages import ErrorMessageProvider


class UpdateSourceType(Enum):
    """Status update source types."""
    WORKER = "worker"
    SYSTEM = "system"
    SCHEDULER = "scheduler"
    API = "api"
    WEBHOOK = "webhook"
    MONITOR = "monitor"


@dataclass
class StatusUpdate:
    """Status update data structure."""
    status_id: str
    new_status: ProcessingStatusType
    progress_percentage: Optional[float] = None
    current_step: Optional[str] = None
    completed_steps: Optional[int] = None
    worker_id: Optional[str] = None
    error_info: Optional[str] = None
    change_reason: Optional[str] = None
    change_metadata: Optional[Dict[str, Any]] = None
    estimated_completion_time: Optional[datetime] = None
    source_type: UpdateSourceType = UpdateSourceType.SYSTEM
    source_id: Optional[str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        # Convert datetime objects to ISO format strings
        if data['timestamp']:
            data['timestamp'] = data['timestamp'].isoformat()
        if data['estimated_completion_time']:
            data['estimated_completion_time'] = data['estimated_completion_time'].isoformat()
        # Convert enum values to strings
        data['new_status'] = data['new_status'].value
        data['source_type'] = data['source_type'].value
        return data


@dataclass
class ProgressUpdate:
    """Progress update data structure."""
    status_id: str
    progress_percentage: float
    current_step: Optional[str] = None
    completed_steps: Optional[int] = None
    worker_id: Optional[str] = None
    processing_metadata: Optional[Dict[str, Any]] = None
    source_type: UpdateSourceType = UpdateSourceType.WORKER
    source_id: Optional[str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        # Convert datetime objects to ISO format strings
        if data['timestamp']:
            data['timestamp'] = data['timestamp'].isoformat()
        # Convert enum values to strings
        data['source_type'] = data['source_type'].value
        return data


@dataclass
class ErrorUpdate:
    """Error update data structure."""
    status_id: str
    error_info: str
    error_metadata: Optional[Dict[str, Any]] = None
    worker_id: Optional[str] = None
    should_retry: bool = True
    source_type: UpdateSourceType = UpdateSourceType.WORKER
    source_id: Optional[str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        # Convert datetime objects to ISO format strings
        if data['timestamp']:
            data['timestamp'] = data['timestamp'].isoformat()
        # Convert enum values to strings
        data['source_type'] = data['source_type'].value
        return data


class StatusUpdateEventHandler:
    """Event handler for status update events."""
    
    def __init__(self):
        self.event_listeners: Dict[str, List[Callable]] = {
            'status_changed': [],
            'progress_updated': [],
            'error_occurred': [],
            'heartbeat_missed': [],
            'processing_completed': [],
            'processing_failed': [],
            'retry_scheduled': []
        }
        self.logger = logging.getLogger(__name__)
    
    def subscribe(self, event_type: str, callback: Callable):
        """
        Subscribe to status update events.
        
        Args:
            event_type: Type of event to subscribe to
            callback: Callback function to execute
        """
        if event_type not in self.event_listeners:
            raise ValueError(f"Unknown event type: {event_type}")
        
        self.event_listeners[event_type].append(callback)
    
    def unsubscribe(self, event_type: str, callback: Callable):
        """
        Unsubscribe from status update events.
        
        Args:
            event_type: Type of event to unsubscribe from
            callback: Callback function to remove
        """
        if event_type in self.event_listeners:
            try:
                self.event_listeners[event_type].remove(callback)
            except ValueError:
                pass
    
    async def emit_event(self, event_type: str, event_data: Dict[str, Any]):
        """
        Emit a status update event.
        
        Args:
            event_type: Type of event
            event_data: Event data
        """
        if event_type not in self.event_listeners:
            return
        
        for callback in self.event_listeners[event_type]:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event_data)
                else:
                    callback(event_data)
            except Exception as e:
                self.logger.error(f"Error in event callback for {event_type}: {e}")


class StatusUpdater:
    """
    Status updater service for managing real-time status updates.
    
    This service provides automated status update mechanisms, event handling,
    and batch update capabilities for efficient status management.
    """
    
    def __init__(self, db_session: Optional[Session] = None):
        """
        Initialize the StatusUpdater.
        
        Args:
            db_session: Optional database session
        """
        self.db_session = db_session
        self._should_close_session = db_session is None
        self.status_service = StatusService(db_session)
        self.event_handler = StatusUpdateEventHandler()
        self.logger = logging.getLogger(__name__)
        
        # Update queues for batch processing
        self.status_update_queue: List[StatusUpdate] = []
        self.progress_update_queue: List[ProgressUpdate] = []
        self.error_update_queue: List[ErrorUpdate] = []
        
        # Processing flags
        self._processing_updates = False
        self._batch_size = 100
        self._batch_interval = 5  # seconds
    
    def __enter__(self):
        """Context manager entry."""
        if self.db_session is None:
            self.db_session = get_db_session()
        self.status_service.db_session = self.db_session
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self._should_close_session and self.db_session:
            self.db_session.close()
    
    def queue_status_update(self, update: StatusUpdate):
        """
        Queue a status update for batch processing.
        
        Args:
            update: Status update to queue
        """
        self.status_update_queue.append(update)
        self.logger.debug(f"Queued status update for {update.status_id}: {update.new_status.value}")
    
    def queue_progress_update(self, update: ProgressUpdate):
        """
        Queue a progress update for batch processing.
        
        Args:
            update: Progress update to queue
        """
        self.progress_update_queue.append(update)
        self.logger.debug(f"Queued progress update for {update.status_id}: {update.progress_percentage}%")
    
    def queue_error_update(self, update: ErrorUpdate):
        """
        Queue an error update for batch processing.
        
        Args:
            update: Error update to queue
        """
        self.error_update_queue.append(update)
        self.logger.debug(f"Queued error update for {update.status_id}: {update.error_info}")
    
    async def process_update_queues(self):
        """Process all queued updates in batches."""
        if self._processing_updates:
            return
        
        try:
            self._processing_updates = True
            
            # Process status updates
            if self.status_update_queue:
                await self._process_status_updates()
            
            # Process progress updates
            if self.progress_update_queue:
                await self._process_progress_updates()
            
            # Process error updates
            if self.error_update_queue:
                await self._process_error_updates()
        
        finally:
            self._processing_updates = False
    
    async def _process_status_updates(self):
        """Process queued status updates."""
        while self.status_update_queue:
            batch = self.status_update_queue[:self._batch_size]
            self.status_update_queue = self.status_update_queue[self._batch_size:]
            
            for update in batch:
                try:
                    status = self.status_service.update_status(
                        status_id=update.status_id,
                        new_status=update.new_status,
                        progress_percentage=update.progress_percentage,
                        current_step=update.current_step,
                        completed_steps=update.completed_steps,
                        worker_id=update.worker_id,
                        error_info=update.error_info,
                        change_reason=update.change_reason,
                        change_metadata=update.change_metadata,
                        estimated_completion_time=update.estimated_completion_time
                    )
                    
                    # Emit status changed event
                    await self.event_handler.emit_event('status_changed', {
                        'status_id': update.status_id,
                        'new_status': update.new_status.value,
                        'update': update.to_dict(),
                        'processing_status': status
                    })
                    
                    # Emit completion events
                    if update.new_status == ProcessingStatusType.COMPLETED:
                        await self.event_handler.emit_event('processing_completed', {
                            'status_id': update.status_id,
                            'processing_status': status
                        })
                    elif update.new_status == ProcessingStatusType.FAILED:
                        await self.event_handler.emit_event('processing_failed', {
                            'status_id': update.status_id,
                            'processing_status': status
                        })
                    elif update.new_status == ProcessingStatusType.RETRY_PENDING:
                        await self.event_handler.emit_event('retry_scheduled', {
                            'status_id': update.status_id,
                            'processing_status': status
                        })
                
                except Exception as e:
                    self.logger.error(f"Error processing status update for {update.status_id}: {e}")
    
    async def _process_progress_updates(self):
        """Process queued progress updates."""
        while self.progress_update_queue:
            batch = self.progress_update_queue[:self._batch_size]
            self.progress_update_queue = self.progress_update_queue[self._batch_size:]
            
            for update in batch:
                try:
                    status = self.status_service.update_progress(
                        status_id=update.status_id,
                        progress_percentage=update.progress_percentage,
                        current_step=update.current_step,
                        completed_steps=update.completed_steps,
                        worker_id=update.worker_id,
                        processing_metadata=update.processing_metadata
                    )
                    
                    # Emit progress updated event
                    await self.event_handler.emit_event('progress_updated', {
                        'status_id': update.status_id,
                        'progress_percentage': update.progress_percentage,
                        'update': update.to_dict(),
                        'processing_status': status
                    })
                
                except Exception as e:
                    self.logger.error(f"Error processing progress update for {update.status_id}: {e}")
    
    async def _process_error_updates(self):
        """Process queued error updates."""
        while self.error_update_queue:
            batch = self.error_update_queue[:self._batch_size]
            self.error_update_queue = self.error_update_queue[self._batch_size:]
            
            for update in batch:
                try:
                    status = self.status_service.record_error(
                        status_id=update.status_id,
                        error_info=update.error_info,
                        error_metadata=update.error_metadata,
                        worker_id=update.worker_id,
                        should_retry=update.should_retry
                    )
                    
                    # Emit error occurred event
                    await self.event_handler.emit_event('error_occurred', {
                        'status_id': update.status_id,
                        'error_info': update.error_info,
                        'update': update.to_dict(),
                        'processing_status': status
                    })
                
                except Exception as e:
                    self.logger.error(f"Error processing error update for {update.status_id}: {e}")
    
    async def start_batch_processor(self):
        """Start the batch update processor."""
        self.logger.info("Starting batch update processor")
        
        while True:
            try:
                await self.process_update_queues()
                await asyncio.sleep(self._batch_interval)
            except Exception as e:
                self.logger.error(f"Error in batch processor: {e}")
                await asyncio.sleep(self._batch_interval)
    
    async def monitor_stale_statuses(self, check_interval: int = 60, timeout_seconds: int = 300):
        """
        Monitor for stale statuses and handle them.
        
        Args:
            check_interval: Check interval in seconds
            timeout_seconds: Timeout threshold in seconds
        """
        self.logger.info("Starting stale status monitor")
        
        while True:
            try:
                stale_statuses = self.status_service.get_stale_statuses(
                    timeout_seconds=timeout_seconds
                )
                
                for status in stale_statuses:
                    await self.event_handler.emit_event('heartbeat_missed', {
                        'status_id': status.status_id,
                        'worker_id': status.worker_id,
                        'last_heartbeat': status.heartbeat_at,
                        'processing_status': status
                    })
                    
                    # Mark as failed if no heartbeat for too long
                    if status.heartbeat_at and (datetime.utcnow() - status.heartbeat_at).total_seconds() > timeout_seconds * 2:
                        self.queue_status_update(StatusUpdate(
                            status_id=status.status_id,
                            new_status=ProcessingStatusType.FAILED,
                            change_reason=f"Stale status: no heartbeat for {timeout_seconds * 2} seconds",
                            source_type=UpdateSourceType.MONITOR
                        ))
                
                await asyncio.sleep(check_interval)
            except Exception as e:
                self.logger.error(f"Error in stale status monitor: {e}")
                await asyncio.sleep(check_interval)
    
    def create_status_update(
        self,
        status_id: str,
        new_status: ProcessingStatusType,
        **kwargs
    ) -> StatusUpdate:
        """
        Create a status update object.
        
        Args:
            status_id: Status identifier
            new_status: New status value
            **kwargs: Additional update parameters
            
        Returns:
            StatusUpdate object
        """
        return StatusUpdate(
            status_id=status_id,
            new_status=new_status,
            **kwargs
        )
    
    def create_progress_update(
        self,
        status_id: str,
        progress_percentage: float,
        **kwargs
    ) -> ProgressUpdate:
        """
        Create a progress update object.
        
        Args:
            status_id: Status identifier
            progress_percentage: Progress percentage
            **kwargs: Additional update parameters
            
        Returns:
            ProgressUpdate object
        """
        return ProgressUpdate(
            status_id=status_id,
            progress_percentage=progress_percentage,
            **kwargs
        )
    
    def create_error_update(
        self,
        status_id: str,
        error_info: str,
        **kwargs
    ) -> ErrorUpdate:
        """
        Create an error update object.
        
        Args:
            status_id: Status identifier
            error_info: Error information
            **kwargs: Additional update parameters
            
        Returns:
            ErrorUpdate object
        """
        return ErrorUpdate(
            status_id=status_id,
            error_info=error_info,
            **kwargs
        )
    
    # Convenience methods for immediate updates
    async def update_status_immediately(self, update: StatusUpdate) -> ProcessingStatus:
        """
        Update status immediately without queuing.
        
        Args:
            update: Status update
            
        Returns:
            Updated ProcessingStatus
        """
        status = self.status_service.update_status(
            status_id=update.status_id,
            new_status=update.new_status,
            progress_percentage=update.progress_percentage,
            current_step=update.current_step,
            completed_steps=update.completed_steps,
            worker_id=update.worker_id,
            error_info=update.error_info,
            change_reason=update.change_reason,
            change_metadata=update.change_metadata,
            estimated_completion_time=update.estimated_completion_time
        )
        
        # Emit event
        await self.event_handler.emit_event('status_changed', {
            'status_id': update.status_id,
            'new_status': update.new_status.value,
            'update': update.to_dict(),
            'processing_status': status
        })
        
        return status
    
    async def update_progress_immediately(self, update: ProgressUpdate) -> ProcessingStatus:
        """
        Update progress immediately without queuing.
        
        Args:
            update: Progress update
            
        Returns:
            Updated ProcessingStatus
        """
        status = self.status_service.update_progress(
            status_id=update.status_id,
            progress_percentage=update.progress_percentage,
            current_step=update.current_step,
            completed_steps=update.completed_steps,
            worker_id=update.worker_id,
            processing_metadata=update.processing_metadata
        )
        
        # Emit event
        await self.event_handler.emit_event('progress_updated', {
            'status_id': update.status_id,
            'progress_percentage': update.progress_percentage,
            'update': update.to_dict(),
            'processing_status': status
        })
        
        return status
    
    async def record_error_immediately(self, update: ErrorUpdate) -> ProcessingStatus:
        """
        Record error immediately without queuing.
        
        Args:
            update: Error update
            
        Returns:
            Updated ProcessingStatus
        """
        status = self.status_service.record_error(
            status_id=update.status_id,
            error_info=update.error_info,
            error_metadata=update.error_metadata,
            worker_id=update.worker_id,
            should_retry=update.should_retry
        )
        
        # Emit event
        await self.event_handler.emit_event('error_occurred', {
            'status_id': update.status_id,
            'error_info': update.error_info,
            'update': update.to_dict(),
            'processing_status': status
        })
        
        return status