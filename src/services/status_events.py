"""
Event system for status changes in the YouTube Summarizer application.

This module provides a comprehensive event system that allows components
to subscribe to and receive notifications about status changes throughout
the processing workflow.
"""

import asyncio
import json
import logging
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable, Union, Set
from dataclasses import dataclass, asdict
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..database.status_models import ProcessingStatusType, StatusChangeType
from ..database.connection import get_db_session


logger = logging.getLogger(__name__)


class EventType(Enum):
    """Types of status events that can be emitted."""
    
    # Status change events
    STATUS_CREATED = "status_created"
    STATUS_UPDATED = "status_updated"
    STATUS_COMPLETED = "status_completed"
    STATUS_FAILED = "status_failed"
    STATUS_CANCELLED = "status_cancelled"
    STATUS_RETRY_SCHEDULED = "status_retry_scheduled"
    
    # Progress events
    PROGRESS_UPDATED = "progress_updated"
    MILESTONE_REACHED = "milestone_reached"
    
    # Error events
    ERROR_OCCURRED = "error_occurred"
    ERROR_RESOLVED = "error_resolved"
    
    # Workflow events
    WORKFLOW_STARTED = "workflow_started"
    WORKFLOW_COMPLETED = "workflow_completed"
    WORKFLOW_FAILED = "workflow_failed"
    
    # Node events
    NODE_STARTED = "node_started"
    NODE_COMPLETED = "node_completed"
    NODE_FAILED = "node_failed"
    
    # Batch events
    BATCH_CREATED = "batch_created"
    BATCH_STARTED = "batch_started"
    BATCH_COMPLETED = "batch_completed"
    BATCH_ITEM_PROCESSED = "batch_item_processed"
    
    # Heartbeat events
    HEARTBEAT_RECEIVED = "heartbeat_received"
    HEARTBEAT_MISSED = "heartbeat_missed"
    STALE_STATUS_DETECTED = "stale_status_detected"


class EventPriority(Enum):
    """Event priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class StatusEvent:
    """Data structure for status change events."""
    
    event_id: str
    event_type: EventType
    timestamp: datetime
    status_id: str
    priority: EventPriority = EventPriority.NORMAL
    
    # Status information
    previous_status: Optional[ProcessingStatusType] = None
    new_status: Optional[ProcessingStatusType] = None
    progress_percentage: Optional[float] = None
    current_step: Optional[str] = None
    
    # Context information
    video_id: Optional[int] = None
    batch_item_id: Optional[int] = None
    processing_session_id: Optional[int] = None
    worker_id: Optional[str] = None
    node_name: Optional[str] = None
    workflow_name: Optional[str] = None
    
    # Error information
    error_info: Optional[str] = None
    error_type: Optional[str] = None
    retry_count: Optional[int] = None
    
    # Additional metadata
    metadata: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None
    
    def __post_init__(self):
        """Post-initialization processing."""
        if not self.event_id:
            self.event_id = f"event_{uuid.uuid4().hex[:12]}"
        if not self.timestamp:
            self.timestamp = datetime.utcnow()
        if not self.tags:
            self.tags = []
        if not self.metadata:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization."""
        data = asdict(self)
        
        # Convert datetime to ISO format
        if data['timestamp']:
            data['timestamp'] = data['timestamp'].isoformat()
        
        # Convert enums to string values
        data['event_type'] = data['event_type'].value
        data['priority'] = data['priority'].value
        
        if data['previous_status']:
            data['previous_status'] = data['previous_status'].value
        if data['new_status']:
            data['new_status'] = data['new_status'].value
        
        return data
    
    def to_json(self) -> str:
        """Convert event to JSON string."""
        return json.dumps(self.to_dict(), default=str)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StatusEvent':
        """Create event from dictionary."""
        # Convert string values back to enums
        if 'event_type' in data:
            data['event_type'] = EventType(data['event_type'])
        if 'priority' in data:
            data['priority'] = EventPriority(data['priority'])
        if 'previous_status' in data and data['previous_status']:
            data['previous_status'] = ProcessingStatusType(data['previous_status'])
        if 'new_status' in data and data['new_status']:
            data['new_status'] = ProcessingStatusType(data['new_status'])
        
        # Convert timestamp string back to datetime
        if 'timestamp' in data and isinstance(data['timestamp'], str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        
        return cls(**data)


class EventHandler(ABC):
    """Abstract base class for event handlers."""
    
    @abstractmethod
    async def handle_event(self, event: StatusEvent) -> bool:
        """
        Handle a status event.
        
        Args:
            event: The status event to handle
            
        Returns:
            True if the event was handled successfully, False otherwise
        """
        pass
    
    @abstractmethod
    def get_handled_event_types(self) -> Set[EventType]:
        """
        Get the set of event types this handler can process.
        
        Returns:
            Set of EventType values this handler supports
        """
        pass
    
    def get_priority_filter(self) -> Optional[Set[EventPriority]]:
        """
        Get the priority filter for this handler.
        
        Returns:
            Set of EventPriority values to filter on, or None for all priorities
        """
        return None
    
    def should_handle_event(self, event: StatusEvent) -> bool:
        """
        Determine if this handler should process the given event.
        
        Args:
            event: The event to check
            
        Returns:
            True if this handler should process the event
        """
        # Check event type
        if event.event_type not in self.get_handled_event_types():
            return False
        
        # Check priority filter
        priority_filter = self.get_priority_filter()
        if priority_filter and event.priority not in priority_filter:
            return False
        
        return True


class LoggingEventHandler(EventHandler):
    """Event handler that logs status events."""
    
    def __init__(self, log_level: int = logging.INFO):
        self.logger = logging.getLogger(f"{__name__}.LoggingEventHandler")
        self.log_level = log_level
    
    async def handle_event(self, event: StatusEvent) -> bool:
        """Log the status event."""
        try:
            # Format log message based on event type
            if event.event_type in [EventType.STATUS_UPDATED, EventType.PROGRESS_UPDATED]:
                message = (
                    f"Status {event.event_type.value}: {event.status_id} -> "
                    f"{event.new_status.value if event.new_status else 'N/A'} "
                    f"({event.progress_percentage:.1f}% - {event.current_step or 'No step'})"
                )
            elif event.event_type in [EventType.ERROR_OCCURRED, EventType.STATUS_FAILED]:
                message = (
                    f"Error {event.event_type.value}: {event.status_id} - "
                    f"{event.error_info or 'Unknown error'}"
                )
            else:
                message = (
                    f"Event {event.event_type.value}: {event.status_id} - "
                    f"{event.current_step or event.new_status.value if event.new_status else 'Status change'}"
                )
            
            # Add context information
            context_parts = []
            if event.video_id:
                context_parts.append(f"video_id={event.video_id}")
            if event.node_name:
                context_parts.append(f"node={event.node_name}")
            if event.worker_id:
                context_parts.append(f"worker={event.worker_id}")
            
            if context_parts:
                message += f" [{', '.join(context_parts)}]"
            
            self.logger.log(self.log_level, message)
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to log status event: {e}")
            return False
    
    def get_handled_event_types(self) -> Set[EventType]:
        """Handle all event types."""
        return set(EventType)


class DatabaseEventHandler(EventHandler):
    """Event handler that stores events in the database."""
    
    def __init__(self, table_name: str = "status_events"):
        self.table_name = table_name
        self.logger = logging.getLogger(f"{__name__}.DatabaseEventHandler")
    
    async def handle_event(self, event: StatusEvent) -> bool:
        """Store the event in the database."""
        try:
            # For now, we'll use a simple logging approach
            # In a full implementation, this would use SQLAlchemy to store events
            with get_db_session() as session:
                # Here you would create and insert an event record
                # For demonstration, we'll just log
                self.logger.debug(f"Would store event {event.event_id} in database table {self.table_name}")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to store event in database: {e}")
            return False
    
    def get_handled_event_types(self) -> Set[EventType]:
        """Handle all event types for database storage."""
        return set(EventType)


class WebhookEventHandler(EventHandler):
    """Event handler that sends events to external webhooks."""
    
    def __init__(self, webhook_urls: List[str], event_types: Optional[Set[EventType]] = None):
        self.webhook_urls = webhook_urls
        self.handled_event_types = event_types or set(EventType)
        self.logger = logging.getLogger(f"{__name__}.WebhookEventHandler")
    
    async def handle_event(self, event: StatusEvent) -> bool:
        """Send event to configured webhooks."""
        try:
            import aiohttp
            
            event_data = event.to_dict()
            
            async with aiohttp.ClientSession() as session:
                tasks = []
                for url in self.webhook_urls:
                    task = self._send_webhook(session, url, event_data)
                    tasks.append(task)
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Check if at least one webhook succeeded
                success_count = sum(1 for result in results if result is True)
                if success_count > 0:
                    self.logger.info(f"Sent event {event.event_id} to {success_count}/{len(self.webhook_urls)} webhooks")
                    return True
                else:
                    self.logger.error(f"Failed to send event {event.event_id} to any webhooks")
                    return False
                    
        except Exception as e:
            self.logger.error(f"Failed to send webhook for event {event.event_id}: {e}")
            return False
    
    async def _send_webhook(self, session, url: str, event_data: Dict[str, Any]) -> bool:
        """Send event data to a single webhook URL."""
        try:
            async with session.post(
                url,
                json=event_data,
                headers={'Content-Type': 'application/json'},
                timeout=30
            ) as response:
                if response.status < 400:
                    return True
                else:
                    self.logger.warning(f"Webhook {url} responded with status {response.status}")
                    return False
                    
        except Exception as e:
            self.logger.error(f"Error sending webhook to {url}: {e}")
            return False
    
    def get_handled_event_types(self) -> Set[EventType]:
        """Return configured event types."""
        return self.handled_event_types


class StatusEventManager:
    """
    Central manager for status change events.
    
    This class coordinates the event system, managing event handlers
    and providing methods to emit events throughout the application.
    """
    
    def __init__(self, max_workers: int = 5):
        self.handlers: List[EventHandler] = []
        self.event_queue: asyncio.Queue = asyncio.Queue()
        self.max_workers = max_workers
        self.running = False
        self.worker_tasks: List[asyncio.Task] = []
        self.logger = logging.getLogger(__name__)
        
        # Event statistics
        self.events_processed = 0
        self.events_failed = 0
        self.handlers_failed = 0
        
        # Add default logging handler
        self.add_handler(LoggingEventHandler())
    
    def add_handler(self, handler: EventHandler):
        """
        Add an event handler to the manager.
        
        Args:
            handler: Event handler to add
        """
        self.handlers.append(handler)
        self.logger.info(f"Added event handler: {handler.__class__.__name__}")
    
    def remove_handler(self, handler: EventHandler):
        """
        Remove an event handler from the manager.
        
        Args:
            handler: Event handler to remove
        """
        if handler in self.handlers:
            self.handlers.remove(handler)
            self.logger.info(f"Removed event handler: {handler.__class__.__name__}")
    
    async def emit_event(self, event: StatusEvent):
        """
        Emit a status event to all registered handlers.
        
        Args:
            event: Status event to emit
        """
        try:
            await self.event_queue.put(event)
            self.logger.debug(f"Queued event: {event.event_id} ({event.event_type.value})")
        except Exception as e:
            self.logger.error(f"Failed to queue event: {e}")
    
    async def emit_status_created(
        self,
        status_id: str,
        initial_status: ProcessingStatusType,
        **kwargs
    ) -> StatusEvent:
        """Emit a status created event."""
        event = StatusEvent(
            event_id=f"created_{uuid.uuid4().hex[:8]}",
            event_type=EventType.STATUS_CREATED,
            timestamp=datetime.utcnow(),
            status_id=status_id,
            new_status=initial_status,
            **kwargs
        )
        await self.emit_event(event)
        return event
    
    async def emit_status_updated(
        self,
        status_id: str,
        previous_status: ProcessingStatusType,
        new_status: ProcessingStatusType,
        **kwargs
    ) -> StatusEvent:
        """Emit a status updated event."""
        event = StatusEvent(
            event_id=f"updated_{uuid.uuid4().hex[:8]}",
            event_type=EventType.STATUS_UPDATED,
            timestamp=datetime.utcnow(),
            status_id=status_id,
            previous_status=previous_status,
            new_status=new_status,
            **kwargs
        )
        await self.emit_event(event)
        return event
    
    async def emit_progress_updated(
        self,
        status_id: str,
        progress_percentage: float,
        current_step: Optional[str] = None,
        **kwargs
    ) -> StatusEvent:
        """Emit a progress updated event."""
        event = StatusEvent(
            event_id=f"progress_{uuid.uuid4().hex[:8]}",
            event_type=EventType.PROGRESS_UPDATED,
            timestamp=datetime.utcnow(),
            status_id=status_id,
            progress_percentage=progress_percentage,
            current_step=current_step,
            **kwargs
        )
        await self.emit_event(event)
        return event
    
    async def emit_error_occurred(
        self,
        status_id: str,
        error_info: str,
        error_type: Optional[str] = None,
        **kwargs
    ) -> StatusEvent:
        """Emit an error occurred event."""
        event = StatusEvent(
            event_id=f"error_{uuid.uuid4().hex[:8]}",
            event_type=EventType.ERROR_OCCURRED,
            timestamp=datetime.utcnow(),
            status_id=status_id,
            error_info=error_info,
            error_type=error_type,
            priority=EventPriority.HIGH,
            **kwargs
        )
        await self.emit_event(event)
        return event
    
    async def emit_workflow_started(
        self,
        status_id: str,
        workflow_name: str,
        **kwargs
    ) -> StatusEvent:
        """Emit a workflow started event."""
        event = StatusEvent(
            event_id=f"workflow_start_{uuid.uuid4().hex[:8]}",
            event_type=EventType.WORKFLOW_STARTED,
            timestamp=datetime.utcnow(),
            status_id=status_id,
            workflow_name=workflow_name,
            **kwargs
        )
        await self.emit_event(event)
        return event
    
    async def emit_workflow_completed(
        self,
        status_id: str,
        workflow_name: str,
        **kwargs
    ) -> StatusEvent:
        """Emit a workflow completed event."""
        event = StatusEvent(
            event_id=f"workflow_complete_{uuid.uuid4().hex[:8]}",
            event_type=EventType.WORKFLOW_COMPLETED,
            timestamp=datetime.utcnow(),
            status_id=status_id,
            workflow_name=workflow_name,
            **kwargs
        )
        await self.emit_event(event)
        return event
    
    async def emit_node_started(
        self,
        status_id: str,
        node_name: str,
        **kwargs
    ) -> StatusEvent:
        """Emit a node started event."""
        event = StatusEvent(
            event_id=f"node_start_{uuid.uuid4().hex[:8]}",
            event_type=EventType.NODE_STARTED,
            timestamp=datetime.utcnow(),
            status_id=status_id,
            node_name=node_name,
            **kwargs
        )
        await self.emit_event(event)
        return event
    
    async def emit_node_completed(
        self,
        status_id: str,
        node_name: str,
        **kwargs
    ) -> StatusEvent:
        """Emit a node completed event."""
        event = StatusEvent(
            event_id=f"node_complete_{uuid.uuid4().hex[:8]}",
            event_type=EventType.NODE_COMPLETED,
            timestamp=datetime.utcnow(),
            status_id=status_id,
            node_name=node_name,
            **kwargs
        )
        await self.emit_event(event)
        return event
    
    async def start(self):
        """Start the event processing workers."""
        if self.running:
            return
        
        self.running = True
        self.logger.info(f"Starting event manager with {self.max_workers} workers")
        
        # Start worker tasks
        for i in range(self.max_workers):
            task = asyncio.create_task(self._worker(f"worker_{i}"))
            self.worker_tasks.append(task)
    
    async def stop(self):
        """Stop the event processing workers."""
        if not self.running:
            return
        
        self.running = False
        self.logger.info("Stopping event manager")
        
        # Cancel all worker tasks
        for task in self.worker_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self.worker_tasks:
            await asyncio.gather(*self.worker_tasks, return_exceptions=True)
        
        self.worker_tasks.clear()
    
    async def _worker(self, worker_name: str):
        """Event processing worker."""
        self.logger.debug(f"Started event worker: {worker_name}")
        
        try:
            while self.running:
                try:
                    # Get event from queue with timeout
                    event = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)
                    
                    # Process event with all handlers
                    await self._process_event(event)
                    
                    # Mark task as done
                    self.event_queue.task_done()
                    
                except asyncio.TimeoutError:
                    # Timeout is normal, continue
                    continue
                except Exception as e:
                    self.logger.error(f"Error in worker {worker_name}: {e}")
                    
        except asyncio.CancelledError:
            self.logger.debug(f"Worker {worker_name} was cancelled")
        except Exception as e:
            self.logger.error(f"Worker {worker_name} failed: {e}")
    
    async def _process_event(self, event: StatusEvent):
        """Process an event with all applicable handlers."""
        try:
            self.events_processed += 1
            
            # Get handlers that should process this event
            applicable_handlers = [
                handler for handler in self.handlers
                if handler.should_handle_event(event)
            ]
            
            if not applicable_handlers:
                self.logger.debug(f"No handlers for event {event.event_id}")
                return
            
            # Process event with all applicable handlers
            tasks = []
            for handler in applicable_handlers:
                task = asyncio.create_task(handler.handle_event(event))
                tasks.append(task)
            
            # Wait for all handlers to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Check results
            success_count = 0
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.handlers_failed += 1
                    self.logger.error(f"Handler {applicable_handlers[i].__class__.__name__} failed for event {event.event_id}: {result}")
                elif result is True:
                    success_count += 1
            
            if success_count == 0:
                self.events_failed += 1
                self.logger.warning(f"All handlers failed for event {event.event_id}")
            else:
                self.logger.debug(f"Event {event.event_id} processed by {success_count}/{len(applicable_handlers)} handlers")
                
        except Exception as e:
            self.events_failed += 1
            self.logger.error(f"Failed to process event {event.event_id}: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get event processing statistics."""
        return {
            'events_processed': self.events_processed,
            'events_failed': self.events_failed,
            'handlers_failed': self.handlers_failed,
            'handlers_count': len(self.handlers),
            'queue_size': self.event_queue.qsize(),
            'workers_running': len(self.worker_tasks),
            'is_running': self.running
        }


# Global event manager instance
_event_manager: Optional[StatusEventManager] = None


def get_event_manager() -> StatusEventManager:
    """Get the global event manager instance."""
    global _event_manager
    if _event_manager is None:
        _event_manager = StatusEventManager()
    return _event_manager


async def initialize_event_system():
    """Initialize the global event system."""
    manager = get_event_manager()
    if not manager.running:
        await manager.start()


async def shutdown_event_system():
    """Shutdown the global event system."""
    global _event_manager
    if _event_manager:
        await _event_manager.stop()
        _event_manager = None