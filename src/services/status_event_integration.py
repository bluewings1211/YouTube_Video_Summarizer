"""
Integration between status tracking and event system.

This module provides enhanced versions of status services that automatically
emit events when status changes occur, creating a comprehensive event-driven
status tracking system.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from .status_service import StatusService
from .status_updater import StatusUpdater, StatusUpdate, ProgressUpdate, ErrorUpdate
from .status_events import (
    StatusEventManager, StatusEvent, EventType, EventPriority,
    get_event_manager
)
from ..database.status_models import ProcessingStatusType, ProcessingPriority, StatusChangeType
from ..database.connection import get_db_session


logger = logging.getLogger(__name__)


class EventAwareStatusService(StatusService):
    """
    Status service that automatically emits events when status changes occur.
    
    This extends the base StatusService to integrate with the event system,
    providing automatic event emission for all status operations.
    """
    
    def __init__(self, db_session=None, event_manager: Optional[StatusEventManager] = None):
        super().__init__(db_session)
        self.event_manager = event_manager or get_event_manager()
        self.logger = logging.getLogger(f"{__name__}.EventAwareStatusService")
    
    def create_processing_status(self, *args, **kwargs):
        """Create processing status and emit creation event."""
        try:
            # Create status using parent implementation
            status = super().create_processing_status(*args, **kwargs)
            
            # Emit status created event
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Schedule the event emission
                    asyncio.create_task(self._emit_status_created_event(status))
                else:
                    # Run the event emission
                    loop.run_until_complete(self._emit_status_created_event(status))
            except RuntimeError:
                # No event loop, skip event emission
                self.logger.warning(f"No event loop available, skipping event emission for status creation: {status.status_id}")
            
            return status
            
        except Exception as e:
            self.logger.error(f"Failed to create status with event emission: {e}")
            raise
    
    def update_status(self, *args, **kwargs):
        """Update status and emit update event."""
        try:
            # Get current status for comparison
            status_id = args[0] if args else kwargs.get('status_id')
            previous_status_record = self.get_processing_status(status_id) if status_id else None
            previous_status = previous_status_record.status if previous_status_record else None
            
            # Update status using parent implementation
            status = super().update_status(*args, **kwargs)
            
            # Emit status updated event
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Schedule the event emission
                    asyncio.create_task(self._emit_status_updated_event(status, previous_status))
                else:
                    # Run the event emission
                    loop.run_until_complete(self._emit_status_updated_event(status, previous_status))
            except RuntimeError:
                # No event loop, skip event emission
                self.logger.warning(f"No event loop available, skipping event emission for status update: {status.status_id}")
            
            return status
            
        except Exception as e:
            self.logger.error(f"Failed to update status with event emission: {e}")
            raise
    
    def update_progress(self, *args, **kwargs):
        """Update progress and emit progress event."""
        try:
            # Update progress using parent implementation
            status = super().update_progress(*args, **kwargs)
            
            # Emit progress updated event
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Schedule the event emission
                    asyncio.create_task(self._emit_progress_updated_event(status))
                else:
                    # Run the event emission
                    loop.run_until_complete(self._emit_progress_updated_event(status))
            except RuntimeError:
                # No event loop, skip event emission
                self.logger.warning(f"No event loop available, skipping event emission for progress update: {status.status_id}")
            
            return status
            
        except Exception as e:
            self.logger.error(f"Failed to update progress with event emission: {e}")
            raise
    
    def record_error(self, *args, **kwargs):
        """Record error and emit error event."""
        try:
            # Record error using parent implementation
            status = super().record_error(*args, **kwargs)
            
            # Emit error occurred event
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Schedule the event emission
                    asyncio.create_task(self._emit_error_occurred_event(status, *args, **kwargs))
                else:
                    # Run the event emission
                    loop.run_until_complete(self._emit_error_occurred_event(status, *args, **kwargs))
            except RuntimeError:
                # No event loop, skip event emission
                self.logger.warning(f"No event loop available, skipping event emission for error: {status.status_id}")
            
            return status
            
        except Exception as e:
            self.logger.error(f"Failed to record error with event emission: {e}")
            raise
    
    async def _emit_status_created_event(self, status):
        """Emit status created event."""
        try:
            await self.event_manager.emit_status_created(
                status_id=status.status_id,
                initial_status=status.status,
                video_id=status.video_id,
                batch_item_id=status.batch_item_id,
                processing_session_id=status.processing_session_id,
                worker_id=status.worker_id,
                priority=EventPriority.NORMAL,
                metadata={
                    'total_steps': status.total_steps,
                    'max_retries': status.max_retries,
                    'external_id': status.external_id,
                    'tags': status.tags,
                    'created_at': status.created_at.isoformat() if status.created_at else None
                }
            )
            self.logger.debug(f"Emitted status created event for {status.status_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to emit status created event: {e}")
    
    async def _emit_status_updated_event(self, status, previous_status):
        """Emit status updated event."""
        try:
            # Determine event type based on new status
            if status.status == ProcessingStatusType.COMPLETED:
                event_type = EventType.STATUS_COMPLETED
                priority = EventPriority.NORMAL
            elif status.status == ProcessingStatusType.FAILED:
                event_type = EventType.STATUS_FAILED
                priority = EventPriority.HIGH
            elif status.status == ProcessingStatusType.CANCELLED:
                event_type = EventType.STATUS_CANCELLED
                priority = EventPriority.NORMAL
            elif status.status == ProcessingStatusType.RETRY_PENDING:
                event_type = EventType.STATUS_RETRY_SCHEDULED
                priority = EventPriority.HIGH
            else:
                event_type = EventType.STATUS_UPDATED
                priority = EventPriority.NORMAL
            
            event = StatusEvent(
                event_type=event_type,
                status_id=status.status_id,
                previous_status=previous_status,
                new_status=status.status,
                progress_percentage=status.progress_percentage,
                current_step=status.current_step,
                video_id=status.video_id,
                batch_item_id=status.batch_item_id,
                processing_session_id=status.processing_session_id,
                worker_id=status.worker_id,
                priority=priority,
                metadata={
                    'completed_steps': status.completed_steps,
                    'retry_count': status.retry_count,
                    'updated_at': status.updated_at.isoformat() if status.updated_at else None,
                    'estimated_completion_time': status.estimated_completion_time.isoformat() if status.estimated_completion_time else None
                }
            )
            
            await self.event_manager.emit_event(event)
            self.logger.debug(f"Emitted status updated event for {status.status_id}: {previous_status} -> {status.status}")
            
        except Exception as e:
            self.logger.error(f"Failed to emit status updated event: {e}")
    
    async def _emit_progress_updated_event(self, status):
        """Emit progress updated event."""
        try:
            await self.event_manager.emit_progress_updated(
                status_id=status.status_id,
                progress_percentage=status.progress_percentage,
                current_step=status.current_step,
                video_id=status.video_id,
                batch_item_id=status.batch_item_id,
                processing_session_id=status.processing_session_id,
                worker_id=status.worker_id,
                metadata={
                    'completed_steps': status.completed_steps,
                    'total_steps': status.total_steps,
                    'updated_at': status.updated_at.isoformat() if status.updated_at else None
                }
            )
            self.logger.debug(f"Emitted progress updated event for {status.status_id}: {status.progress_percentage}%")
            
        except Exception as e:
            self.logger.error(f"Failed to emit progress updated event: {e}")
    
    async def _emit_error_occurred_event(self, status, *args, **kwargs):
        """Emit error occurred event."""
        try:
            # Extract error information from arguments
            error_info = args[1] if len(args) > 1 else kwargs.get('error_info', 'Unknown error')
            error_metadata = kwargs.get('error_metadata', {})
            
            await self.event_manager.emit_error_occurred(
                status_id=status.status_id,
                error_info=error_info,
                error_type=error_metadata.get('error_type'),
                video_id=status.video_id,
                batch_item_id=status.batch_item_id,
                processing_session_id=status.processing_session_id,
                worker_id=status.worker_id,
                retry_count=status.retry_count,
                metadata={
                    'should_retry': status.can_retry,
                    'max_retries': status.max_retries,
                    'error_metadata': error_metadata,
                    'updated_at': status.updated_at.isoformat() if status.updated_at else None
                }
            )
            self.logger.debug(f"Emitted error occurred event for {status.status_id}: {error_info}")
            
        except Exception as e:
            self.logger.error(f"Failed to emit error occurred event: {e}")


class EventAwareStatusUpdater(StatusUpdater):
    """
    Status updater that integrates with the event system.
    
    This extends the base StatusUpdater to emit events during
    batch status update processing.
    """
    
    def __init__(self, db_session=None, event_manager: Optional[StatusEventManager] = None):
        # Create event-aware status service
        event_aware_status_service = EventAwareStatusService(db_session, event_manager)
        
        # Initialize parent with event-aware service
        super().__init__(db_session)
        self.status_service = event_aware_status_service
        self.event_manager = event_manager or get_event_manager()
        self.logger = logging.getLogger(f"{__name__}.EventAwareStatusUpdater")
    
    async def _process_status_updates(self):
        """Process status updates with event emission."""
        # The parent implementation will use our event-aware status service
        # which will automatically emit events
        await super()._process_status_updates()
    
    async def _process_progress_updates(self):
        """Process progress updates with event emission."""
        # The parent implementation will use our event-aware status service
        # which will automatically emit events
        await super()._process_progress_updates()
    
    async def _process_error_updates(self):
        """Process error updates with event emission."""
        # The parent implementation will use our event-aware status service
        # which will automatically emit events
        await super()._process_error_updates()


class StatusEventIntegrator:
    """
    Central integration point for status tracking and event system.
    
    This class provides a unified interface for status operations
    that automatically emit appropriate events.
    """
    
    def __init__(self, event_manager: Optional[StatusEventManager] = None):
        self.event_manager = event_manager or get_event_manager()
        self.logger = logging.getLogger(f"{__name__}.StatusEventIntegrator")
        
        # Create event-aware services
        self._status_service = None
        self._status_updater = None
    
    @property
    def status_service(self) -> EventAwareStatusService:
        """Get event-aware status service."""
        if self._status_service is None:
            self._status_service = EventAwareStatusService(event_manager=self.event_manager)
        return self._status_service
    
    @property
    def status_updater(self) -> EventAwareStatusUpdater:
        """Get event-aware status updater."""
        if self._status_updater is None:
            self._status_updater = EventAwareStatusUpdater(event_manager=self.event_manager)
        return self._status_updater
    
    async def emit_workflow_started(
        self,
        workflow_status_id: str,
        workflow_name: str,
        **kwargs
    ):
        """Emit workflow started event."""
        try:
            await self.event_manager.emit_workflow_started(
                status_id=workflow_status_id,
                workflow_name=workflow_name,
                **kwargs
            )
            self.logger.debug(f"Emitted workflow started event: {workflow_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to emit workflow started event: {e}")
    
    async def emit_workflow_completed(
        self,
        workflow_status_id: str,
        workflow_name: str,
        **kwargs
    ):
        """Emit workflow completed event."""
        try:
            await self.event_manager.emit_workflow_completed(
                status_id=workflow_status_id,
                workflow_name=workflow_name,
                **kwargs
            )
            self.logger.debug(f"Emitted workflow completed event: {workflow_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to emit workflow completed event: {e}")
    
    async def emit_workflow_failed(
        self,
        workflow_status_id: str,
        workflow_name: str,
        error_info: str,
        **kwargs
    ):
        """Emit workflow failed event."""
        try:
            event = StatusEvent(
                event_type=EventType.WORKFLOW_FAILED,
                status_id=workflow_status_id,
                workflow_name=workflow_name,
                error_info=error_info,
                priority=EventPriority.HIGH,
                **kwargs
            )
            await self.event_manager.emit_event(event)
            self.logger.debug(f"Emitted workflow failed event: {workflow_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to emit workflow failed event: {e}")
    
    async def emit_node_started(
        self,
        node_status_id: str,
        node_name: str,
        **kwargs
    ):
        """Emit node started event."""
        try:
            await self.event_manager.emit_node_started(
                status_id=node_status_id,
                node_name=node_name,
                **kwargs
            )
            self.logger.debug(f"Emitted node started event: {node_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to emit node started event: {e}")
    
    async def emit_node_completed(
        self,
        node_status_id: str,
        node_name: str,
        **kwargs
    ):
        """Emit node completed event."""
        try:
            await self.event_manager.emit_node_completed(
                status_id=node_status_id,
                node_name=node_name,
                **kwargs
            )
            self.logger.debug(f"Emitted node completed event: {node_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to emit node completed event: {e}")
    
    async def emit_node_failed(
        self,
        node_status_id: str,
        node_name: str,
        error_info: str,
        **kwargs
    ):
        """Emit node failed event."""
        try:
            event = StatusEvent(
                event_type=EventType.NODE_FAILED,
                status_id=node_status_id,
                node_name=node_name,
                error_info=error_info,
                priority=EventPriority.HIGH,
                **kwargs
            )
            await self.event_manager.emit_event(event)
            self.logger.debug(f"Emitted node failed event: {node_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to emit node failed event: {e}")
    
    async def emit_batch_created(
        self,
        batch_status_id: str,
        batch_size: int,
        **kwargs
    ):
        """Emit batch created event."""
        try:
            event = StatusEvent(
                event_type=EventType.BATCH_CREATED,
                status_id=batch_status_id,
                metadata={'batch_size': batch_size},
                **kwargs
            )
            await self.event_manager.emit_event(event)
            self.logger.debug(f"Emitted batch created event: {batch_size} items")
            
        except Exception as e:
            self.logger.error(f"Failed to emit batch created event: {e}")
    
    async def emit_batch_item_processed(
        self,
        batch_item_status_id: str,
        batch_item_id: int,
        success: bool,
        **kwargs
    ):
        """Emit batch item processed event."""
        try:
            event = StatusEvent(
                event_type=EventType.BATCH_ITEM_PROCESSED,
                status_id=batch_item_status_id,
                batch_item_id=batch_item_id,
                metadata={'success': success},
                **kwargs
            )
            await self.event_manager.emit_event(event)
            self.logger.debug(f"Emitted batch item processed event: {batch_item_id} (success={success})")
            
        except Exception as e:
            self.logger.error(f"Failed to emit batch item processed event: {e}")
    
    async def emit_heartbeat_missed(
        self,
        status_id: str,
        worker_id: str,
        last_heartbeat: Optional[datetime] = None,
        **kwargs
    ):
        """Emit heartbeat missed event."""
        try:
            event = StatusEvent(
                event_type=EventType.HEARTBEAT_MISSED,
                status_id=status_id,
                worker_id=worker_id,
                priority=EventPriority.HIGH,
                metadata={
                    'last_heartbeat': last_heartbeat.isoformat() if last_heartbeat else None
                },
                **kwargs
            )
            await self.event_manager.emit_event(event)
            self.logger.debug(f"Emitted heartbeat missed event: {worker_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to emit heartbeat missed event: {e}")
    
    def get_event_statistics(self) -> Dict[str, Any]:
        """Get event system statistics."""
        return self.event_manager.get_statistics()


# Global integrator instance
_integrator: Optional[StatusEventIntegrator] = None


def get_status_event_integrator() -> StatusEventIntegrator:
    """Get the global status event integrator instance."""
    global _integrator
    if _integrator is None:
        _integrator = StatusEventIntegrator()
    return _integrator