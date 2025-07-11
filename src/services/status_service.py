"""
Status tracking service for YouTube Summarizer application.
Provides comprehensive status management functionality.
"""

import uuid
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Union
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, asc, func
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from ..database.connection import get_db_session
from ..database.status_models import (
    ProcessingStatus, StatusHistory, StatusMetrics,
    ProcessingStatusType, ProcessingPriority, StatusChangeType
)
from ..database.models import Video
from ..database.batch_models import BatchItem, ProcessingSession
from ..utils.error_messages import ErrorMessageProvider


class StatusService:
    """
    Service class for managing processing status and history.
    
    This service provides comprehensive status tracking functionality including
    status creation, updates, history tracking, and metrics aggregation.
    """
    
    def __init__(self, db_session: Optional[Session] = None):
        """
        Initialize the StatusService.
        
        Args:
            db_session: Optional database session. If not provided, a new session will be created.
        """
        self.db_session = db_session
        self._should_close_session = db_session is None
    
    def __enter__(self):
        """Context manager entry."""
        if self.db_session is None:
            self.db_session = get_db_session()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self._should_close_session and self.db_session:
            self.db_session.close()
    
    def create_processing_status(
        self,
        video_id: Optional[int] = None,
        batch_item_id: Optional[int] = None,
        processing_session_id: Optional[int] = None,
        priority: ProcessingPriority = ProcessingPriority.NORMAL,
        total_steps: Optional[int] = None,
        max_retries: int = 3,
        external_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        processing_metadata: Optional[Dict[str, Any]] = None
    ) -> ProcessingStatus:
        """
        Create a new processing status record.
        
        Args:
            video_id: Optional video ID
            batch_item_id: Optional batch item ID
            processing_session_id: Optional processing session ID
            priority: Processing priority
            total_steps: Total number of processing steps
            max_retries: Maximum number of retry attempts
            external_id: External system identifier
            tags: List of tags for categorization
            processing_metadata: Additional processing metadata
            
        Returns:
            ProcessingStatus: The created status record
            
        Raises:
            ValueError: If invalid parameters are provided
            SQLAlchemyError: If database operation fails
        """
        try:
            # Generate unique status ID
            status_id = f"status_{uuid.uuid4().hex[:12]}"
            
            # Create processing status
            status = ProcessingStatus(
                status_id=status_id,
                video_id=video_id,
                batch_item_id=batch_item_id,
                processing_session_id=processing_session_id,
                status=ProcessingStatusType.QUEUED,
                priority=priority,
                progress_percentage=0.0,
                total_steps=total_steps,
                completed_steps=0,
                max_retries=max_retries,
                external_id=external_id,
                tags=tags,
                processing_metadata=processing_metadata or {}
            )
            
            self.db_session.add(status)
            self.db_session.commit()
            
            # Create initial status history entry
            self._create_status_history(
                status.id,
                StatusChangeType.STATUS_UPDATE,
                None,
                ProcessingStatusType.QUEUED,
                0.0,
                0.0,
                "Initial status creation"
            )
            
            return status
            
        except IntegrityError as e:
            self.db_session.rollback()
            raise ValueError(f"Failed to create processing status: {str(e)}")
        except SQLAlchemyError as e:
            self.db_session.rollback()
            raise e
    
    def update_status(
        self,
        status_id: str,
        new_status: ProcessingStatusType,
        progress_percentage: Optional[float] = None,
        current_step: Optional[str] = None,
        completed_steps: Optional[int] = None,
        worker_id: Optional[str] = None,
        error_info: Optional[str] = None,
        change_reason: Optional[str] = None,
        change_metadata: Optional[Dict[str, Any]] = None,
        estimated_completion_time: Optional[datetime] = None
    ) -> ProcessingStatus:
        """
        Update processing status.
        
        Args:
            status_id: Status identifier
            new_status: New status value
            progress_percentage: Updated progress percentage
            current_step: Current processing step description
            completed_steps: Number of completed steps
            worker_id: Worker identifier
            error_info: Error information if applicable
            change_reason: Reason for status change
            change_metadata: Additional change metadata
            estimated_completion_time: Estimated completion time
            
        Returns:
            ProcessingStatus: The updated status record
            
        Raises:
            ValueError: If status not found or invalid parameters
            SQLAlchemyError: If database operation fails
        """
        try:
            status = self.get_processing_status(status_id)
            if not status:
                raise ValueError(f"Processing status not found: {status_id}")
            
            # Store previous values for history
            previous_status = status.status
            previous_progress = status.progress_percentage
            
            # Update status fields
            status.status = new_status
            if progress_percentage is not None:
                status.progress_percentage = progress_percentage
            if current_step is not None:
                status.current_step = current_step
            if completed_steps is not None:
                status.completed_steps = completed_steps
            if worker_id is not None:
                status.worker_id = worker_id
            if error_info is not None:
                status.error_info = error_info
            if estimated_completion_time is not None:
                status.estimated_completion_time = estimated_completion_time
            
            # Update timestamps
            status.updated_at = datetime.utcnow()
            status.heartbeat_at = datetime.utcnow()
            
            # Set started_at if transitioning from QUEUED
            if previous_status == ProcessingStatusType.QUEUED and new_status != ProcessingStatusType.QUEUED:
                status.started_at = datetime.utcnow()
            
            # Set completed_at if transitioning to final state
            if new_status in [ProcessingStatusType.COMPLETED, ProcessingStatusType.FAILED, ProcessingStatusType.CANCELLED]:
                status.completed_at = datetime.utcnow()
            
            self.db_session.commit()
            
            # Create status history entry
            self._create_status_history(
                status.id,
                StatusChangeType.STATUS_UPDATE,
                previous_status,
                new_status,
                previous_progress,
                status.progress_percentage,
                change_reason or f"Status updated to {new_status.value}",
                change_metadata,
                worker_id
            )
            
            return status
            
        except ValueError:
            raise
        except SQLAlchemyError as e:
            self.db_session.rollback()
            raise e
    
    def update_progress(
        self,
        status_id: str,
        progress_percentage: float,
        current_step: Optional[str] = None,
        completed_steps: Optional[int] = None,
        worker_id: Optional[str] = None,
        processing_metadata: Optional[Dict[str, Any]] = None
    ) -> ProcessingStatus:
        """
        Update processing progress.
        
        Args:
            status_id: Status identifier
            progress_percentage: Updated progress percentage
            current_step: Current processing step description
            completed_steps: Number of completed steps
            worker_id: Worker identifier
            processing_metadata: Additional processing metadata
            
        Returns:
            ProcessingStatus: The updated status record
        """
        try:
            status = self.get_processing_status(status_id)
            if not status:
                raise ValueError(f"Processing status not found: {status_id}")
            
            previous_progress = status.progress_percentage
            
            # Update progress fields
            status.progress_percentage = progress_percentage
            if current_step is not None:
                status.current_step = current_step
            if completed_steps is not None:
                status.completed_steps = completed_steps
            if worker_id is not None:
                status.worker_id = worker_id
            if processing_metadata is not None:
                status.processing_metadata = {
                    **(status.processing_metadata or {}),
                    **processing_metadata
                }
            
            # Update timestamps
            status.updated_at = datetime.utcnow()
            status.heartbeat_at = datetime.utcnow()
            
            self.db_session.commit()
            
            # Create progress history entry
            self._create_status_history(
                status.id,
                StatusChangeType.PROGRESS_UPDATE,
                status.status,
                status.status,
                previous_progress,
                progress_percentage,
                f"Progress updated to {progress_percentage:.1f}%",
                processing_metadata,
                worker_id
            )
            
            return status
            
        except ValueError:
            raise
        except SQLAlchemyError as e:
            self.db_session.rollback()
            raise e
    
    def record_error(
        self,
        status_id: str,
        error_info: str,
        error_metadata: Optional[Dict[str, Any]] = None,
        worker_id: Optional[str] = None,
        should_retry: bool = True
    ) -> ProcessingStatus:
        """
        Record an error for processing status.
        
        Args:
            status_id: Status identifier
            error_info: Error information
            error_metadata: Additional error metadata
            worker_id: Worker identifier
            should_retry: Whether to schedule retry
            
        Returns:
            ProcessingStatus: The updated status record
        """
        try:
            status = self.get_processing_status(status_id)
            if not status:
                raise ValueError(f"Processing status not found: {status_id}")
            
            # Update error information
            status.error_info = error_info
            status.updated_at = datetime.utcnow()
            
            # Increment retry count
            status.retry_count += 1
            
            # Determine new status
            if should_retry and status.can_retry:
                new_status = ProcessingStatusType.RETRY_PENDING
            else:
                new_status = ProcessingStatusType.FAILED
                status.completed_at = datetime.utcnow()
            
            previous_status = status.status
            status.status = new_status
            
            self.db_session.commit()
            
            # Create error history entry
            self._create_status_history(
                status.id,
                StatusChangeType.ERROR_OCCURRED,
                previous_status,
                new_status,
                status.progress_percentage,
                status.progress_percentage,
                f"Error occurred: {error_info}",
                error_metadata,
                worker_id,
                error_info=error_info
            )
            
            return status
            
        except ValueError:
            raise
        except SQLAlchemyError as e:
            self.db_session.rollback()
            raise e
    
    def heartbeat(
        self,
        status_id: str,
        worker_id: str,
        progress_percentage: Optional[float] = None,
        current_step: Optional[str] = None
    ) -> ProcessingStatus:
        """
        Update processing heartbeat.
        
        Args:
            status_id: Status identifier
            worker_id: Worker identifier
            progress_percentage: Optional progress update
            current_step: Optional current step update
            
        Returns:
            ProcessingStatus: The updated status record
        """
        try:
            status = self.get_processing_status(status_id)
            if not status:
                raise ValueError(f"Processing status not found: {status_id}")
            
            # Update heartbeat
            status.heartbeat_at = datetime.utcnow()
            status.worker_id = worker_id
            
            if progress_percentage is not None:
                status.progress_percentage = progress_percentage
            if current_step is not None:
                status.current_step = current_step
            
            self.db_session.commit()
            return status
            
        except ValueError:
            raise
        except SQLAlchemyError as e:
            self.db_session.rollback()
            raise e
    
    def get_processing_status(self, status_id: str) -> Optional[ProcessingStatus]:
        """
        Get processing status by ID.
        
        Args:
            status_id: Status identifier
            
        Returns:
            ProcessingStatus or None if not found
        """
        try:
            return self.db_session.query(ProcessingStatus).filter(
                ProcessingStatus.status_id == status_id
            ).first()
        except SQLAlchemyError as e:
            raise e
    
    def get_status_by_video_id(self, video_id: int) -> Optional[ProcessingStatus]:
        """
        Get processing status by video ID.
        
        Args:
            video_id: Video ID
            
        Returns:
            ProcessingStatus or None if not found
        """
        try:
            return self.db_session.query(ProcessingStatus).filter(
                ProcessingStatus.video_id == video_id
            ).order_by(desc(ProcessingStatus.created_at)).first()
        except SQLAlchemyError as e:
            raise e
    
    def get_status_by_batch_item_id(self, batch_item_id: int) -> Optional[ProcessingStatus]:
        """
        Get processing status by batch item ID.
        
        Args:
            batch_item_id: Batch item ID
            
        Returns:
            ProcessingStatus or None if not found
        """
        try:
            return self.db_session.query(ProcessingStatus).filter(
                ProcessingStatus.batch_item_id == batch_item_id
            ).order_by(desc(ProcessingStatus.created_at)).first()
        except SQLAlchemyError as e:
            raise e
    
    def get_active_statuses(
        self,
        worker_id: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[ProcessingStatus]:
        """
        Get currently active processing statuses.
        
        Args:
            worker_id: Optional worker ID filter
            limit: Optional limit on results
            
        Returns:
            List of active ProcessingStatus records
        """
        try:
            query = self.db_session.query(ProcessingStatus).filter(
                ProcessingStatus.status.in_([
                    ProcessingStatusType.STARTING,
                    ProcessingStatusType.YOUTUBE_METADATA,
                    ProcessingStatusType.TRANSCRIPT_EXTRACTION,
                    ProcessingStatusType.LANGUAGE_DETECTION,
                    ProcessingStatusType.SUMMARY_GENERATION,
                    ProcessingStatusType.KEYWORD_EXTRACTION,
                    ProcessingStatusType.TIMESTAMPED_SEGMENTS,
                    ProcessingStatusType.FINALIZING
                ])
            )
            
            if worker_id:
                query = query.filter(ProcessingStatus.worker_id == worker_id)
            
            query = query.order_by(desc(ProcessingStatus.updated_at))
            
            if limit:
                query = query.limit(limit)
            
            return query.all()
        except SQLAlchemyError as e:
            raise e
    
    def get_stale_statuses(
        self,
        timeout_seconds: int = 300,
        limit: Optional[int] = None
    ) -> List[ProcessingStatus]:
        """
        Get stale processing statuses (no heartbeat for timeout period).
        
        Args:
            timeout_seconds: Timeout period in seconds
            limit: Optional limit on results
            
        Returns:
            List of stale ProcessingStatus records
        """
        try:
            cutoff_time = datetime.utcnow() - timedelta(seconds=timeout_seconds)
            
            query = self.db_session.query(ProcessingStatus).filter(
                and_(
                    ProcessingStatus.status.in_([
                        ProcessingStatusType.STARTING,
                        ProcessingStatusType.YOUTUBE_METADATA,
                        ProcessingStatusType.TRANSCRIPT_EXTRACTION,
                        ProcessingStatusType.LANGUAGE_DETECTION,
                        ProcessingStatusType.SUMMARY_GENERATION,
                        ProcessingStatusType.KEYWORD_EXTRACTION,
                        ProcessingStatusType.TIMESTAMPED_SEGMENTS,
                        ProcessingStatusType.FINALIZING
                    ]),
                    or_(
                        ProcessingStatus.heartbeat_at.is_(None),
                        ProcessingStatus.heartbeat_at < cutoff_time
                    )
                )
            )
            
            if limit:
                query = query.limit(limit)
            
            return query.all()
        except SQLAlchemyError as e:
            raise e
    
    def get_status_history(
        self,
        status_id: str,
        limit: Optional[int] = None,
        change_type: Optional[StatusChangeType] = None
    ) -> List[StatusHistory]:
        """
        Get status history for a processing status.
        
        Args:
            status_id: Status identifier
            limit: Optional limit on results
            change_type: Optional change type filter
            
        Returns:
            List of StatusHistory records
        """
        try:
            status = self.get_processing_status(status_id)
            if not status:
                raise ValueError(f"Processing status not found: {status_id}")
            
            query = self.db_session.query(StatusHistory).filter(
                StatusHistory.processing_status_id == status.id
            )
            
            if change_type:
                query = query.filter(StatusHistory.change_type == change_type)
            
            query = query.order_by(desc(StatusHistory.created_at))
            
            if limit:
                query = query.limit(limit)
            
            return query.all()
        except ValueError:
            raise
        except SQLAlchemyError as e:
            raise e
    
    def _create_status_history(
        self,
        processing_status_id: int,
        change_type: StatusChangeType,
        previous_status: Optional[ProcessingStatusType],
        new_status: ProcessingStatusType,
        previous_progress: float,
        new_progress: float,
        change_reason: Optional[str] = None,
        change_metadata: Optional[Dict[str, Any]] = None,
        worker_id: Optional[str] = None,
        error_info: Optional[str] = None,
        external_trigger: Optional[str] = None
    ) -> StatusHistory:
        """
        Create a status history entry.
        
        Args:
            processing_status_id: Processing status ID
            change_type: Type of change
            previous_status: Previous status value
            new_status: New status value
            previous_progress: Previous progress percentage
            new_progress: New progress percentage
            change_reason: Reason for change
            change_metadata: Additional change metadata
            worker_id: Worker identifier
            error_info: Error information
            external_trigger: External trigger source
            
        Returns:
            StatusHistory: The created history record
        """
        try:
            # Calculate duration if there's a previous status
            duration_seconds = None
            if previous_status:
                last_history = self.db_session.query(StatusHistory).filter(
                    StatusHistory.processing_status_id == processing_status_id
                ).order_by(desc(StatusHistory.created_at)).first()
                
                if last_history:
                    duration = datetime.utcnow() - last_history.created_at
                    duration_seconds = int(duration.total_seconds())
            
            history = StatusHistory(
                processing_status_id=processing_status_id,
                change_type=change_type,
                previous_status=previous_status,
                new_status=new_status,
                previous_progress=previous_progress,
                new_progress=new_progress,
                change_reason=change_reason,
                change_metadata=change_metadata,
                worker_id=worker_id,
                duration_seconds=duration_seconds,
                error_info=error_info,
                external_trigger=external_trigger
            )
            
            self.db_session.add(history)
            self.db_session.commit()
            
            return history
            
        except SQLAlchemyError as e:
            self.db_session.rollback()
            raise e
    
    def cleanup_old_statuses(
        self,
        days_old: int = 30,
        keep_failed: bool = True
    ) -> int:
        """
        Clean up old processing statuses.
        
        Args:
            days_old: Age threshold in days
            keep_failed: Whether to keep failed statuses
            
        Returns:
            Number of records cleaned up
        """
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_old)
            
            query = self.db_session.query(ProcessingStatus).filter(
                ProcessingStatus.created_at < cutoff_date,
                ProcessingStatus.status.in_([
                    ProcessingStatusType.COMPLETED,
                    ProcessingStatusType.CANCELLED
                ])
            )
            
            if not keep_failed:
                query = query.filter(
                    ProcessingStatus.status != ProcessingStatusType.FAILED
                )
            
            count = query.count()
            query.delete(synchronize_session=False)
            self.db_session.commit()
            
            return count
            
        except SQLAlchemyError as e:
            self.db_session.rollback()
            raise e