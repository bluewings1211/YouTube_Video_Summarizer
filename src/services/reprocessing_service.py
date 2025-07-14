"""
Video reprocessing service for handling video reprocessing operations.

This service provides functionality to reprocess existing videos,
including cache clearing, workflow triggering, and progress tracking.
"""

import logging
from datetime import datetime
from typing import Optional, Dict, Any, List, Set
from dataclasses import dataclass
from enum import Enum
from sqlalchemy.orm import Session
from sqlalchemy import select, update, delete
from sqlalchemy.exc import SQLAlchemyError
from fastapi import Depends

from ..database.models import (
    Video, Transcript, Summary, Keyword, 
    TimestampedSegment, ProcessingMetadata
)
from ..database.exceptions import (
    DatabaseError, DatabaseConnectionError, DatabaseQueryError,
    classify_database_error
)
from ..database.connection import get_database_session_dependency

logger = logging.getLogger(__name__)


class ReprocessingMode(Enum):
    """Reprocessing mode options."""
    FULL = "full"                      # Reprocess everything
    TRANSCRIPT_ONLY = "transcript_only"  # Only reprocess transcript
    SUMMARY_ONLY = "summary_only"      # Only reprocess summary
    KEYWORDS_ONLY = "keywords_only"    # Only reprocess keywords
    SEGMENTS_ONLY = "segments_only"    # Only reprocess timestamped segments
    INCREMENTAL = "incremental"        # Only reprocess missing components


class ReprocessingStatus(Enum):
    """Status of reprocessing operations."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ReprocessingRequest:
    """Request for video reprocessing."""
    video_id: int
    mode: ReprocessingMode
    force: bool = False
    clear_cache: bool = True
    preserve_metadata: bool = True
    requested_by: Optional[str] = None
    workflow_params: Optional[Dict[str, Any]] = None


@dataclass
class ReprocessingResult:
    """Result of video reprocessing operation."""
    success: bool
    video_id: int
    mode: ReprocessingMode
    status: ReprocessingStatus
    message: str
    cleared_components: List[str]
    processing_metadata_id: Optional[int] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    execution_time_seconds: Optional[float] = None
    error_details: Optional[Dict[str, Any]] = None


@dataclass
class ReprocessingValidation:
    """Validation result for reprocessing request."""
    can_reprocess: bool
    video_exists: bool
    current_status: Optional[str]
    existing_components: Dict[str, int]
    potential_issues: List[str]
    recommendations: List[str]


class ReprocessingServiceError(Exception):
    """Custom exception for reprocessing service operations."""
    pass


class ReprocessingService:
    """
    Service class for managing video reprocessing operations.
    
    This class provides methods to validate, initiate, and monitor
    video reprocessing operations with cache clearing capabilities.
    """

    def __init__(self, session: Optional[Session] = None):
        """
        Initialize the reprocessing service.
        
        Args:
            session: Optional database session. If not provided, will use dependency injection.
        """
        self._session = session
        self._logger = logging.getLogger(f"{__name__}.ReprocessingService")

    def _get_session(self) -> Session:
        """Get database session (internal method)."""
        if self._session:
            return self._session
        else:
            # This should be used with dependency injection
            raise ReprocessingServiceError("No database session provided")

    def validate_reprocessing_request(self, request: ReprocessingRequest) -> ReprocessingValidation:
        """
        Validate a reprocessing request.
        
        Args:
            request: Reprocessing request to validate
            
        Returns:
            ReprocessingValidation with validation results
            
        Raises:
            ReprocessingServiceError: If validation fails
        """
        try:
            session = self._get_session()
            
            # Check if video exists
            video = session.get(Video, request.video_id)
            if not video:
                return ReprocessingValidation(
                    can_reprocess=False,
                    video_exists=False,
                    current_status=None,
                    existing_components={},
                    potential_issues=["Video not found"],
                    recommendations=["Verify video ID exists in database"]
                )
            
            # Get current processing status
            current_status = None
            latest_metadata = session.execute(
                select(ProcessingMetadata).where(
                    ProcessingMetadata.video_id == request.video_id
                ).order_by(ProcessingMetadata.created_at.desc())
            ).scalar_one_or_none()
            
            if latest_metadata:
                current_status = latest_metadata.status
            
            # Count existing components
            existing_components = {
                "transcripts": session.execute(
                    select(Transcript).where(Transcript.video_id == request.video_id)
                ).rowcount,
                "summaries": session.execute(
                    select(Summary).where(Summary.video_id == request.video_id)
                ).rowcount,
                "keywords": session.execute(
                    select(Keyword).where(Keyword.video_id == request.video_id)
                ).rowcount,
                "timestamped_segments": session.execute(
                    select(TimestampedSegment).where(TimestampedSegment.video_id == request.video_id)
                ).rowcount,
                "processing_metadata": session.execute(
                    select(ProcessingMetadata).where(ProcessingMetadata.video_id == request.video_id)
                ).rowcount
            }
            
            # Check for potential issues
            potential_issues = []
            recommendations = []
            
            # Check if currently processing
            if current_status in ['pending', 'processing']:
                potential_issues.append(f"Video is currently being processed (status: {current_status})")
                recommendations.append("Wait for current processing to complete or use force=true to override")
            
            # Check for incremental mode validity
            if request.mode == ReprocessingMode.INCREMENTAL:
                missing_components = [comp for comp, count in existing_components.items() 
                                    if count == 0 and comp != 'processing_metadata']
                if not missing_components:
                    potential_issues.append("Incremental mode requested but no components are missing")
                    recommendations.append("Use a specific mode or FULL mode instead")
            
            # Check for component-specific modes
            if request.mode == ReprocessingMode.TRANSCRIPT_ONLY and existing_components['transcripts'] == 0:
                recommendations.append("No existing transcript to reprocess - will create new one")
            elif request.mode == ReprocessingMode.SUMMARY_ONLY and existing_components['summaries'] == 0:
                recommendations.append("No existing summary to reprocess - will create new one")
            elif request.mode == ReprocessingMode.KEYWORDS_ONLY and existing_components['keywords'] == 0:
                recommendations.append("No existing keywords to reprocess - will create new ones")
            elif request.mode == ReprocessingMode.SEGMENTS_ONLY and existing_components['timestamped_segments'] == 0:
                recommendations.append("No existing segments to reprocess - will create new ones")
            
            # Determine if can reprocess
            can_reprocess = len(potential_issues) == 0 or request.force
            
            return ReprocessingValidation(
                can_reprocess=can_reprocess,
                video_exists=True,
                current_status=current_status,
                existing_components=existing_components,
                potential_issues=potential_issues,
                recommendations=recommendations
            )
            
        except SQLAlchemyError as e:
            db_error = classify_database_error(e)
            self._logger.error(f"Database error validating reprocessing request: {db_error}")
            raise ReprocessingServiceError(f"Failed to validate request: {db_error.message}")
        except Exception as e:
            self._logger.error(f"Unexpected error validating reprocessing request: {e}")
            raise ReprocessingServiceError(f"Unexpected error: {e}")

    def clear_video_cache(self, video_id: int, mode: ReprocessingMode, preserve_metadata: bool = True) -> List[str]:
        """
        Clear cached data for a video based on reprocessing mode.
        
        Args:
            video_id: Database video ID
            mode: Reprocessing mode determining what to clear
            preserve_metadata: Whether to preserve processing metadata
            
        Returns:
            List of cleared component names
            
        Raises:
            ReprocessingServiceError: If cache clearing fails
        """
        try:
            session = self._get_session()
            cleared_components = []
            
            # Define what to clear based on mode
            clear_mapping = {
                ReprocessingMode.FULL: ['transcripts', 'summaries', 'keywords', 'timestamped_segments'],
                ReprocessingMode.TRANSCRIPT_ONLY: ['transcripts'],
                ReprocessingMode.SUMMARY_ONLY: ['summaries'],
                ReprocessingMode.KEYWORDS_ONLY: ['keywords'],
                ReprocessingMode.SEGMENTS_ONLY: ['timestamped_segments'],
                ReprocessingMode.INCREMENTAL: []  # Will be determined dynamically
            }
            
            components_to_clear = clear_mapping.get(mode, [])
            
            # For incremental mode, only clear components that exist
            if mode == ReprocessingMode.INCREMENTAL:
                existing_components = {
                    'transcripts': session.execute(select(Transcript.id).where(Transcript.video_id == video_id)).rowcount,
                    'summaries': session.execute(select(Summary.id).where(Summary.video_id == video_id)).rowcount,
                    'keywords': session.execute(select(Keyword.id).where(Keyword.video_id == video_id)).rowcount,
                    'timestamped_segments': session.execute(select(TimestampedSegment.id).where(TimestampedSegment.video_id == video_id)).rowcount
                }
                components_to_clear = [comp for comp, count in existing_components.items() if count == 0]
            
            # Clear components
            for component in components_to_clear:
                if component == 'transcripts':
                    result = session.execute(delete(Transcript).where(Transcript.video_id == video_id))
                    if result.rowcount > 0:
                        cleared_components.append('transcripts')
                        self._logger.info(f"Cleared {result.rowcount} transcript(s) for video {video_id}")
                
                elif component == 'summaries':
                    result = session.execute(delete(Summary).where(Summary.video_id == video_id))
                    if result.rowcount > 0:
                        cleared_components.append('summaries')
                        self._logger.info(f"Cleared {result.rowcount} summary(s) for video {video_id}")
                
                elif component == 'keywords':
                    result = session.execute(delete(Keyword).where(Keyword.video_id == video_id))
                    if result.rowcount > 0:
                        cleared_components.append('keywords')
                        self._logger.info(f"Cleared {result.rowcount} keyword(s) for video {video_id}")
                
                elif component == 'timestamped_segments':
                    result = session.execute(delete(TimestampedSegment).where(TimestampedSegment.video_id == video_id))
                    if result.rowcount > 0:
                        cleared_components.append('timestamped_segments')
                        self._logger.info(f"Cleared {result.rowcount} segment(s) for video {video_id}")
            
            # Clear processing metadata if not preserving
            if not preserve_metadata:
                result = session.execute(delete(ProcessingMetadata).where(ProcessingMetadata.video_id == video_id))
                if result.rowcount > 0:
                    cleared_components.append('processing_metadata')
                    self._logger.info(f"Cleared {result.rowcount} processing metadata for video {video_id}")
            
            session.commit()
            
            self._logger.info(f"Cache cleared for video {video_id}: {cleared_components}")
            return cleared_components
            
        except SQLAlchemyError as e:
            db_error = classify_database_error(e)
            self._logger.error(f"Database error clearing cache: {db_error}")
            raise ReprocessingServiceError(f"Failed to clear cache: {db_error.message}")
        except Exception as e:
            self._logger.error(f"Unexpected error clearing cache: {e}")
            raise ReprocessingServiceError(f"Unexpected error: {e}")

    def initiate_reprocessing(self, request: ReprocessingRequest) -> ReprocessingResult:
        """
        Initiate video reprocessing operation.
        
        Args:
            request: Reprocessing request details
            
        Returns:
            ReprocessingResult with operation details
            
        Raises:
            ReprocessingServiceError: If reprocessing initiation fails
        """
        try:
            session = self._get_session()
            start_time = datetime.now()
            
            # Validate request if not forced
            if not request.force:
                validation = self.validate_reprocessing_request(request)
                if not validation.can_reprocess:
                    return ReprocessingResult(
                        success=False,
                        video_id=request.video_id,
                        mode=request.mode,
                        status=ReprocessingStatus.FAILED,
                        message=f"Validation failed: {'; '.join(validation.potential_issues)}",
                        cleared_components=[],
                        start_time=start_time,
                        end_time=datetime.now(),
                        execution_time_seconds=0
                    )
            
            # Clear cache if requested
            cleared_components = []
            if request.clear_cache:
                cleared_components = self.clear_video_cache(
                    request.video_id, 
                    request.mode, 
                    request.preserve_metadata
                )
            
            # Create processing metadata record
            processing_metadata = ProcessingMetadata(
                video_id=request.video_id,
                workflow_params=request.workflow_params or {
                    "reprocessing_mode": request.mode.value,
                    "requested_by": request.requested_by,
                    "clear_cache": request.clear_cache,
                    "preserve_metadata": request.preserve_metadata
                },
                status=ReprocessingStatus.PENDING.value
            )
            
            session.add(processing_metadata)
            session.commit()
            session.refresh(processing_metadata)
            
            # Here we would normally trigger the actual processing workflow
            # For now, we'll simulate the workflow trigger
            self._logger.info(f"Reprocessing initiated for video {request.video_id} with mode {request.mode.value}")
            
            # TODO: Integrate with actual workflow system
            # This is where we would call the workflow orchestrator
            # workflow_result = self._trigger_workflow(request.video_id, request.mode, processing_metadata.id)
            
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            return ReprocessingResult(
                success=True,
                video_id=request.video_id,
                mode=request.mode,
                status=ReprocessingStatus.PENDING,
                message=f"Reprocessing initiated successfully with mode: {request.mode.value}",
                cleared_components=cleared_components,
                processing_metadata_id=processing_metadata.id,
                start_time=start_time,
                end_time=end_time,
                execution_time_seconds=execution_time
            )
            
        except SQLAlchemyError as e:
            db_error = classify_database_error(e)
            self._logger.error(f"Database error initiating reprocessing: {db_error}")
            raise ReprocessingServiceError(f"Failed to initiate reprocessing: {db_error.message}")
        except Exception as e:
            self._logger.error(f"Unexpected error initiating reprocessing: {e}")
            raise ReprocessingServiceError(f"Unexpected error: {e}")

    def get_reprocessing_status(self, video_id: int) -> Optional[Dict[str, Any]]:
        """
        Get the current reprocessing status for a video.
        
        Args:
            video_id: Database video ID
            
        Returns:
            Dictionary with reprocessing status information or None if not found
            
        Raises:
            ReprocessingServiceError: If status retrieval fails
        """
        try:
            session = self._get_session()
            
            # Get latest processing metadata
            latest_metadata = session.execute(
                select(ProcessingMetadata).where(
                    ProcessingMetadata.video_id == video_id
                ).order_by(ProcessingMetadata.created_at.desc())
            ).scalar_one_or_none()
            
            if not latest_metadata:
                return None
            
            # Get video info
            video = session.get(Video, video_id)
            if not video:
                return None
            
            return {
                "video_id": video_id,
                "video_title": video.title,
                "video_youtube_id": video.video_id,
                "processing_metadata_id": latest_metadata.id,
                "status": latest_metadata.status,
                "workflow_params": latest_metadata.workflow_params,
                "error_info": latest_metadata.error_info,
                "created_at": latest_metadata.created_at,
                "is_reprocessing": latest_metadata.workflow_params and 
                                 latest_metadata.workflow_params.get("reprocessing_mode") is not None
            }
            
        except SQLAlchemyError as e:
            db_error = classify_database_error(e)
            self._logger.error(f"Database error getting reprocessing status: {db_error}")
            raise ReprocessingServiceError(f"Failed to get status: {db_error.message}")
        except Exception as e:
            self._logger.error(f"Unexpected error getting reprocessing status: {e}")
            raise ReprocessingServiceError(f"Unexpected error: {e}")

    def cancel_reprocessing(self, video_id: int, reason: str = "User requested") -> bool:
        """
        Cancel an active reprocessing operation.
        
        Args:
            video_id: Database video ID
            reason: Reason for cancellation
            
        Returns:
            True if cancellation was successful, False otherwise
            
        Raises:
            ReprocessingServiceError: If cancellation fails
        """
        try:
            session = self._get_session()
            
            # Find active processing metadata
            active_metadata = session.execute(
                select(ProcessingMetadata).where(
                    ProcessingMetadata.video_id == video_id,
                    ProcessingMetadata.status.in_(['pending', 'processing'])
                ).order_by(ProcessingMetadata.created_at.desc())
            ).scalar_one_or_none()
            
            if not active_metadata:
                self._logger.warning(f"No active reprocessing found for video {video_id}")
                return False
            
            # Update status to cancelled
            active_metadata.status = ReprocessingStatus.CANCELLED.value
            active_metadata.error_info = f"Cancelled: {reason}"
            
            session.commit()
            
            self._logger.info(f"Reprocessing cancelled for video {video_id}: {reason}")
            return True
            
        except SQLAlchemyError as e:
            db_error = classify_database_error(e)
            self._logger.error(f"Database error cancelling reprocessing: {db_error}")
            raise ReprocessingServiceError(f"Failed to cancel reprocessing: {db_error.message}")
        except Exception as e:
            self._logger.error(f"Unexpected error cancelling reprocessing: {e}")
            raise ReprocessingServiceError(f"Unexpected error: {e}")

    def get_reprocessing_history(self, video_id: int, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get reprocessing history for a video.
        
        Args:
            video_id: Database video ID
            limit: Maximum number of records to return
            
        Returns:
            List of reprocessing history records
            
        Raises:
            ReprocessingServiceError: If history retrieval fails
        """
        try:
            session = self._get_session()
            
            # Get processing metadata records
            metadata_records = session.execute(
                select(ProcessingMetadata).where(
                    ProcessingMetadata.video_id == video_id
                ).order_by(ProcessingMetadata.created_at.desc()).limit(limit)
            ).scalars().all()
            
            history = []
            for record in metadata_records:
                history.append({
                    "id": record.id,
                    "status": record.status,
                    "workflow_params": record.workflow_params,
                    "error_info": record.error_info,
                    "created_at": record.created_at,
                    "is_reprocessing": record.workflow_params and 
                                     record.workflow_params.get("reprocessing_mode") is not None,
                    "reprocessing_mode": record.workflow_params.get("reprocessing_mode") if record.workflow_params else None
                })
            
            return history
            
        except SQLAlchemyError as e:
            db_error = classify_database_error(e)
            self._logger.error(f"Database error getting reprocessing history: {db_error}")
            raise ReprocessingServiceError(f"Failed to get history: {db_error.message}")
        except Exception as e:
            self._logger.error(f"Unexpected error getting reprocessing history: {e}")
            raise ReprocessingServiceError(f"Unexpected error: {e}")


# Dependency injection function for FastAPI
def get_reprocessing_service(session: Session = Depends(get_database_session_dependency)) -> ReprocessingService:
    """
    Get ReprocessingService instance for dependency injection.
    
    Args:
        session: Database session from FastAPI dependency
        
    Returns:
        ReprocessingService instance
    """
    return ReprocessingService(session)