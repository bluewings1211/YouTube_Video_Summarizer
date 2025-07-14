"""
Status tracking integration for workflow and node processing.

This module provides mixins and utilities to integrate status tracking
into existing processing workflows and nodes without major refactoring.
"""

import logging
import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime
from contextlib import contextmanager

from .status_service import StatusService
from .status_updater import StatusUpdater, StatusUpdate, ProgressUpdate, ErrorUpdate, UpdateSourceType
from ..database.status_models import ProcessingStatusType, ProcessingPriority
from ..database.connection import get_db_session


logger = logging.getLogger(__name__)


class StatusTrackingMixin:
    """
    Mixin class to add status tracking capabilities to workflow and node classes.
    
    This mixin can be added to any class that needs status tracking without
    modifying the existing inheritance hierarchy.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._status_service: Optional[StatusService] = None
        self._status_updater: Optional[StatusUpdater] = None
        self._current_status_id: Optional[str] = None
        self._status_tracking_enabled = True
        self._worker_id = f"{self.__class__.__name__}_{uuid.uuid4().hex[:8]}"
        
        # Initialize status tracking if enabled
        if self._status_tracking_enabled:
            self._initialize_status_tracking()
    
    def _initialize_status_tracking(self):
        """Initialize status tracking services."""
        try:
            db_session = get_db_session()
            self._status_service = StatusService(db_session)
            self._status_updater = StatusUpdater(db_session)
            logger.debug(f"Status tracking initialized for {self.__class__.__name__}")
        except Exception as e:
            logger.warning(f"Failed to initialize status tracking: {e}")
            self._status_tracking_enabled = False
    
    def _create_processing_status(
        self,
        video_id: Optional[int] = None,
        batch_item_id: Optional[int] = None,
        processing_session_id: Optional[int] = None,
        priority: ProcessingPriority = ProcessingPriority.NORMAL,
        total_steps: Optional[int] = None,
        external_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        processing_metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Create a new processing status record.
        
        Returns:
            Status ID if successful, None otherwise
        """
        if not self._status_tracking_enabled or not self._status_service:
            return None
        
        try:
            status = self._status_service.create_processing_status(
                video_id=video_id,
                batch_item_id=batch_item_id,
                processing_session_id=processing_session_id,
                priority=priority,
                total_steps=total_steps,
                external_id=external_id,
                tags=tags or [self.__class__.__name__],
                processing_metadata=processing_metadata
            )
            self._current_status_id = status.status_id
            return status.status_id
        except Exception as e:
            logger.error(f"Failed to create processing status: {e}")
            return None
    
    def _update_status(
        self,
        new_status: ProcessingStatusType,
        progress_percentage: Optional[float] = None,
        current_step: Optional[str] = None,
        completed_steps: Optional[int] = None,
        error_info: Optional[str] = None,
        change_reason: Optional[str] = None,
        change_metadata: Optional[Dict[str, Any]] = None,
        immediate: bool = False
    ) -> bool:
        """
        Update processing status.
        
        Args:
            new_status: New status value
            progress_percentage: Progress percentage (0-100)
            current_step: Current processing step description
            completed_steps: Number of completed steps
            error_info: Error information if applicable
            change_reason: Reason for status change
            change_metadata: Additional metadata
            immediate: Whether to update immediately or queue for batch processing
            
        Returns:
            True if update was successful/queued, False otherwise
        """
        if not self._status_tracking_enabled or not self._current_status_id:
            return False
        
        try:
            update = StatusUpdate(
                status_id=self._current_status_id,
                new_status=new_status,
                progress_percentage=progress_percentage,
                current_step=current_step,
                completed_steps=completed_steps,
                worker_id=self._worker_id,
                error_info=error_info,
                change_reason=change_reason,
                change_metadata=change_metadata,
                source_type=UpdateSourceType.WORKER
            )
            
            if immediate:
                # Immediate update using status service directly
                self._status_service.update_status(
                    status_id=self._current_status_id,
                    new_status=new_status,
                    progress_percentage=progress_percentage,
                    current_step=current_step,
                    completed_steps=completed_steps,
                    worker_id=self._worker_id,
                    error_info=error_info,
                    change_reason=change_reason,
                    change_metadata=change_metadata
                )
            else:
                # Queue for batch processing
                self._status_updater.queue_status_update(update)
            
            return True
        except Exception as e:
            logger.error(f"Failed to update status: {e}")
            return False
    
    def _update_progress(
        self,
        progress_percentage: float,
        current_step: Optional[str] = None,
        completed_steps: Optional[int] = None,
        processing_metadata: Optional[Dict[str, Any]] = None,
        immediate: bool = False
    ) -> bool:
        """
        Update processing progress.
        
        Args:
            progress_percentage: Progress percentage (0-100)
            current_step: Current processing step description
            completed_steps: Number of completed steps
            processing_metadata: Additional processing metadata
            immediate: Whether to update immediately or queue for batch processing
            
        Returns:
            True if update was successful/queued, False otherwise
        """
        if not self._status_tracking_enabled or not self._current_status_id:
            return False
        
        try:
            update = ProgressUpdate(
                status_id=self._current_status_id,
                progress_percentage=progress_percentage,
                current_step=current_step,
                completed_steps=completed_steps,
                worker_id=self._worker_id,
                processing_metadata=processing_metadata,
                source_type=UpdateSourceType.WORKER
            )
            
            if immediate:
                # Immediate update using status service directly
                self._status_service.update_progress(
                    status_id=self._current_status_id,
                    progress_percentage=progress_percentage,
                    current_step=current_step,
                    completed_steps=completed_steps,
                    worker_id=self._worker_id,
                    processing_metadata=processing_metadata
                )
            else:
                # Queue for batch processing
                self._status_updater.queue_progress_update(update)
            
            return True
        except Exception as e:
            logger.error(f"Failed to update progress: {e}")
            return False
    
    def _record_error(
        self,
        error_info: str,
        error_metadata: Optional[Dict[str, Any]] = None,
        should_retry: bool = True,
        immediate: bool = True
    ) -> bool:
        """
        Record an error for processing status.
        
        Args:
            error_info: Error information
            error_metadata: Additional error metadata
            should_retry: Whether to schedule retry
            immediate: Whether to update immediately or queue for batch processing
            
        Returns:
            True if error was recorded successfully/queued, False otherwise
        """
        if not self._status_tracking_enabled or not self._current_status_id:
            return False
        
        try:
            update = ErrorUpdate(
                status_id=self._current_status_id,
                error_info=error_info,
                error_metadata=error_metadata,
                worker_id=self._worker_id,
                should_retry=should_retry,
                source_type=UpdateSourceType.WORKER
            )
            
            if immediate:
                # Immediate update using status service directly
                self._status_service.record_error(
                    status_id=self._current_status_id,
                    error_info=error_info,
                    error_metadata=error_metadata,
                    worker_id=self._worker_id,
                    should_retry=should_retry
                )
            else:
                # Queue for batch processing
                self._status_updater.queue_error_update(update)
            
            return True
        except Exception as e:
            logger.error(f"Failed to record error: {e}")
            return False
    
    def _send_heartbeat(self, progress_percentage: Optional[float] = None, current_step: Optional[str] = None) -> bool:
        """
        Send a heartbeat to indicate the process is still active.
        
        Args:
            progress_percentage: Optional progress update
            current_step: Optional current step update
            
        Returns:
            True if heartbeat was sent successfully, False otherwise
        """
        if not self._status_tracking_enabled or not self._current_status_id:
            return False
        
        try:
            self._status_service.heartbeat(
                status_id=self._current_status_id,
                worker_id=self._worker_id,
                progress_percentage=progress_percentage,
                current_step=current_step
            )
            return True
        except Exception as e:
            logger.error(f"Failed to send heartbeat: {e}")
            return False
    
    @contextmanager
    def _status_tracking_context(
        self,
        operation_name: str,
        total_steps: Optional[int] = None,
        video_id: Optional[int] = None,
        batch_item_id: Optional[int] = None,
        processing_session_id: Optional[int] = None,
        auto_complete: bool = True
    ):
        """
        Context manager for automatic status tracking during operations.
        
        Args:
            operation_name: Name of the operation being tracked
            total_steps: Total number of steps in the operation
            video_id: Optional video ID
            batch_item_id: Optional batch item ID
            processing_session_id: Optional processing session ID
            auto_complete: Whether to automatically mark as completed on success
        """
        if not self._status_tracking_enabled:
            yield None
            return
        
        # Create status if not already exists
        if not self._current_status_id:
            status_id = self._create_processing_status(
                video_id=video_id,
                batch_item_id=batch_item_id,
                processing_session_id=processing_session_id,
                total_steps=total_steps,
                tags=[self.__class__.__name__, operation_name],
                processing_metadata={'operation': operation_name}
            )
        else:
            status_id = self._current_status_id
        
        # Mark as starting
        self._update_status(
            ProcessingStatusType.STARTING,
            current_step=f"Starting {operation_name}",
            change_reason=f"Started {operation_name}",
            immediate=True
        )
        
        try:
            yield status_id
            
            # Mark as completed if requested
            if auto_complete:
                self._update_status(
                    ProcessingStatusType.COMPLETED,
                    progress_percentage=100.0,
                    current_step=f"Completed {operation_name}",
                    change_reason=f"Successfully completed {operation_name}",
                    immediate=True
                )
        
        except Exception as e:
            # Mark as failed
            self._record_error(
                error_info=f"Error in {operation_name}: {str(e)}",
                error_metadata={'operation': operation_name, 'error_type': type(e).__name__},
                should_retry=True,
                immediate=True
            )
            raise
    
    def _get_current_status_id(self) -> Optional[str]:
        """Get the current status ID."""
        return self._current_status_id
    
    def _set_current_status_id(self, status_id: str):
        """Set the current status ID (for external status management)."""
        self._current_status_id = status_id
    
    def _disable_status_tracking(self):
        """Disable status tracking for this instance."""
        self._status_tracking_enabled = False
    
    def _enable_status_tracking(self):
        """Enable status tracking for this instance."""
        if not self._status_tracking_enabled:
            self._status_tracking_enabled = True
            self._initialize_status_tracking()


class WorkflowStatusManager:
    """
    Manager class for coordinating status tracking across workflow execution.
    
    This class manages the overall workflow status and coordinates status
    updates from individual nodes and components.
    """
    
    def __init__(self, workflow_name: str, db_session=None):
        self.workflow_name = workflow_name
        self.db_session = db_session or get_db_session()
        self.status_service = StatusService(self.db_session)
        self.status_updater = StatusUpdater(self.db_session)
        
        # Workflow state
        self.workflow_status_id: Optional[str] = None
        self.node_status_mapping: Dict[str, str] = {}  # node_name -> status_id
        self.current_node_index = 0
        self.total_nodes = 0
        
        # Step mapping for detailed progress tracking
        self.step_mapping = {
            'YouTubeDataNode': ProcessingStatusType.YOUTUBE_METADATA,
            'YouTubeTranscriptNode': ProcessingStatusType.TRANSCRIPT_EXTRACTION,
            'SummarizationNode': ProcessingStatusType.SUMMARY_GENERATION,
            'KeywordExtractionNode': ProcessingStatusType.KEYWORD_EXTRACTION,
            'TimestampNode': ProcessingStatusType.TIMESTAMPED_SEGMENTS
        }
        
        logger.debug(f"WorkflowStatusManager initialized for {workflow_name}")
    
    def start_workflow(
        self,
        video_id: Optional[int] = None,
        batch_item_id: Optional[int] = None,
        processing_session_id: Optional[int] = None,
        node_names: Optional[List[str]] = None,
        priority: ProcessingPriority = ProcessingPriority.NORMAL
    ) -> str:
        """
        Start workflow-level status tracking.
        
        Args:
            video_id: Video ID being processed
            batch_item_id: Batch item ID if part of batch processing
            processing_session_id: Processing session ID
            node_names: List of node names that will be executed
            priority: Processing priority
            
        Returns:
            Workflow status ID
        """
        try:
            # Set total nodes for progress calculation
            self.total_nodes = len(node_names) if node_names else 4  # Default estimate
            
            # Create workflow status
            status = self.status_service.create_processing_status(
                video_id=video_id,
                batch_item_id=batch_item_id,
                processing_session_id=processing_session_id,
                priority=priority,
                total_steps=self.total_nodes,
                external_id=f"workflow_{uuid.uuid4().hex[:12]}",
                tags=[self.workflow_name, 'workflow'],
                processing_metadata={
                    'workflow_name': self.workflow_name,
                    'node_names': node_names or [],
                    'workflow_type': 'single_video'
                }
            )
            
            self.workflow_status_id = status.status_id
            
            # Update to starting status
            self.status_service.update_status(
                status_id=self.workflow_status_id,
                new_status=ProcessingStatusType.STARTING,
                current_step="Initializing workflow",
                change_reason="Workflow started"
            )
            
            logger.info(f"Started workflow status tracking: {self.workflow_status_id}")
            return self.workflow_status_id
            
        except Exception as e:
            logger.error(f"Failed to start workflow status tracking: {e}")
            raise
    
    def start_node(self, node_name: str, video_id: Optional[int] = None) -> str:
        """
        Start node-level status tracking.
        
        Args:
            node_name: Name of the node being started
            video_id: Video ID being processed
            
        Returns:
            Node status ID
        """
        try:
            # Create node status
            status = self.status_service.create_processing_status(
                video_id=video_id,
                priority=ProcessingPriority.NORMAL,
                total_steps=3,  # prep, exec, post
                external_id=f"node_{node_name}_{uuid.uuid4().hex[:8]}",
                tags=[node_name, 'node'],
                processing_metadata={
                    'node_name': node_name,
                    'workflow_status_id': self.workflow_status_id,
                    'node_type': 'processing_node'
                }
            )
            
            node_status_id = status.status_id
            self.node_status_mapping[node_name] = node_status_id
            
            # Update node status to appropriate processing stage
            processing_status = self.step_mapping.get(node_name, ProcessingStatusType.STARTING)
            self.status_service.update_status(
                status_id=node_status_id,
                new_status=processing_status,
                current_step=f"Starting {node_name}",
                change_reason=f"Node {node_name} started"
            )
            
            # Update workflow progress
            self._update_workflow_progress(node_name, 'started')
            
            logger.debug(f"Started node status tracking: {node_name} -> {node_status_id}")
            return node_status_id
            
        except Exception as e:
            logger.error(f"Failed to start node status tracking for {node_name}: {e}")
            raise
    
    def finish_node(self, node_name: str, success: bool = True, error_info: Optional[str] = None):
        """
        Finish node-level status tracking.
        
        Args:
            node_name: Name of the node being finished
            success: Whether the node completed successfully
            error_info: Error information if failed
        """
        try:
            node_status_id = self.node_status_mapping.get(node_name)
            if not node_status_id:
                logger.warning(f"No status ID found for node {node_name}")
                return
            
            if success:
                # Mark node as completed
                self.status_service.update_status(
                    status_id=node_status_id,
                    new_status=ProcessingStatusType.COMPLETED,
                    progress_percentage=100.0,
                    current_step=f"Completed {node_name}",
                    completed_steps=3,
                    change_reason=f"Node {node_name} completed successfully"
                )
                
                # Update workflow progress
                self.current_node_index += 1
                self._update_workflow_progress(node_name, 'completed')
            else:
                # Mark node as failed
                self.status_service.record_error(
                    status_id=node_status_id,
                    error_info=error_info or f"Node {node_name} failed",
                    should_retry=True
                )
                
                # Update workflow status to failed
                if self.workflow_status_id:
                    self.status_service.update_status(
                        status_id=self.workflow_status_id,
                        new_status=ProcessingStatusType.FAILED,
                        error_info=error_info or f"Workflow failed at node {node_name}",
                        change_reason=f"Node {node_name} failed"
                    )
            
            logger.debug(f"Finished node status tracking: {node_name} (success={success})")
            
        except Exception as e:
            logger.error(f"Failed to finish node status tracking for {node_name}: {e}")
    
    def finish_workflow(self, success: bool = True, error_info: Optional[str] = None):
        """
        Finish workflow-level status tracking.
        
        Args:
            success: Whether the workflow completed successfully
            error_info: Error information if failed
        """
        try:
            if not self.workflow_status_id:
                return
            
            if success:
                # Mark workflow as completed
                self.status_service.update_status(
                    status_id=self.workflow_status_id,
                    new_status=ProcessingStatusType.COMPLETED,
                    progress_percentage=100.0,
                    current_step="Workflow completed",
                    completed_steps=self.total_nodes,
                    change_reason="Workflow completed successfully"
                )
            else:
                # Mark workflow as failed
                self.status_service.record_error(
                    status_id=self.workflow_status_id,
                    error_info=error_info or "Workflow failed",
                    should_retry=False  # Workflow-level failures typically don't retry automatically
                )
            
            logger.info(f"Finished workflow status tracking: {self.workflow_status_id} (success={success})")
            
        except Exception as e:
            logger.error(f"Failed to finish workflow status tracking: {e}")
    
    def _update_workflow_progress(self, node_name: str, phase: str):
        """Update workflow-level progress based on node progress."""
        try:
            if not self.workflow_status_id:
                return
            
            # Calculate progress percentage
            if phase == 'started':
                progress = (self.current_node_index / self.total_nodes) * 100
                current_step = f"Processing {node_name}"
            elif phase == 'completed':
                progress = (self.current_node_index / self.total_nodes) * 100
                current_step = f"Completed {node_name}"
            else:
                progress = None
                current_step = f"{phase} {node_name}"
            
            # Update workflow status
            if progress is not None:
                self.status_service.update_progress(
                    status_id=self.workflow_status_id,
                    progress_percentage=progress,
                    current_step=current_step,
                    completed_steps=self.current_node_index
                )
            
        except Exception as e:
            logger.error(f"Failed to update workflow progress: {e}")
    
    def get_workflow_status_id(self) -> Optional[str]:
        """Get the workflow status ID."""
        return self.workflow_status_id
    
    def get_node_status_id(self, node_name: str) -> Optional[str]:
        """Get the status ID for a specific node."""
        return self.node_status_mapping.get(node_name)
    
    def cleanup(self):
        """Cleanup resources."""
        try:
            if self.db_session:
                self.db_session.close()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")