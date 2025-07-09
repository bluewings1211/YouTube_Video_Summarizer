"""
Video processing service with database persistence.

This service handles all database operations for video processing,
including saving transcripts, summaries, keywords, and metadata.
"""

import logging
from datetime import datetime
from typing import Optional, Dict, Any, List
from sqlalchemy.orm import Session, selectinload
from sqlalchemy import select, update, delete
from sqlalchemy.exc import SQLAlchemyError, IntegrityError, TimeoutError

from ..database.models import (
    Video, Transcript, Summary, Keyword, 
    TimestampedSegment, ProcessingMetadata
)
from ..database.connection import get_database_session
from ..database.exceptions import (
    DatabaseError, DatabaseConnectionError, DatabaseQueryError,
    DatabaseConstraintError, DatabaseTimeoutError, DatabaseUnavailableError,
    classify_database_error, is_recoverable_error, should_retry_operation
)
from ..database.monitor import MonitoredOperation, db_monitor

logger = logging.getLogger(__name__)


class VideoServiceError(Exception):
    """Custom exception for video service operations."""
    pass


class VideoService:
    """
    Service class for managing video processing and database persistence.
    
    This class provides methods to create video records, save processing results,
    and manage video data in the database.
    """

    def __init__(self, session: Optional[Session] = None, max_retries: int = 3, retry_delay: float = 1.0):
        """
        Initialize the video service.
        
        Args:
            session: Optional database session. If not provided, will use dependency injection.
            max_retries: Maximum number of retries for database operations
            retry_delay: Delay between retries in seconds
        """
        self._session = session
        self._logger = logging.getLogger(f"{__name__}.VideoService")
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def _get_session(self) -> Session:
        """Get database session (internal method)."""
        if self._session:
            return self._session
        else:
            # This method is deprecated - use context manager instead
            raise VideoServiceError("No database session provided. Use context manager for database operations.")

    def _retry_database_operation(self, operation, *args, **kwargs):
        """
        Retry a database operation with exponential backoff.
        
        Args:
            operation: Async function to retry
            *args: Arguments for the operation
            **kwargs: Keyword arguments for the operation
            
        Returns:
            Result of the operation
            
        Raises:
            DatabaseError: If all retries fail
        """
        import time
        
        last_error = None
        retry_count = 0
        
        while retry_count <= self.max_retries:
            try:
                return operation(*args, **kwargs)
                
            except Exception as e:
                # Classify the error
                if isinstance(e, DatabaseError):
                    db_error = e
                else:
                    db_error = classify_database_error(e)
                
                last_error = db_error
                
                # Check if we should retry
                if not should_retry_operation(db_error, retry_count, self.max_retries):
                    self._logger.error(f"Database operation failed, not retrying: {db_error}")
                    raise
                
                retry_count += 1
                delay = self.retry_delay * (2 ** (retry_count - 1))  # Exponential backoff
                
                self._logger.warning(
                    f"Database operation failed (attempt {retry_count}/{self.max_retries + 1}), "
                    f"retrying in {delay:.2f}s: {db_error}"
                )
                
                time.sleep(delay)
        
        # All retries exhausted
        self._logger.error(f"Database operation failed after {retry_count} retries: {last_error}")
        raise last_error

    def create_video_record(
        self,
        video_id: str,
        title: str,
        duration: Optional[int] = None,
        url: Optional[str] = None,
        channel_name: Optional[str] = None,
        description: Optional[str] = None,
        published_date: Optional[str] = None,
        view_count: Optional[int] = None,
        transcript_content: Optional[str] = None,
        transcript_language: Optional[str] = None
    ) -> Optional[Video]:
        """
        Create a new video record in the database.
        
        Args:
            video_id: YouTube video ID
            title: Video title
            duration: Video duration in seconds
            url: Video URL
            channel_name: Channel name
            description: Video description
            published_date: Published date
            view_count: View count
            transcript_content: Transcript content
            transcript_language: Transcript language
            
        Returns:
            Created Video object or None if failed
            
        Raises:
            VideoServiceError: If creation fails
        """
        with MonitoredOperation("create_video_record"):
            try:
                # Use context manager for session
                with get_database_session() as session:
                    # Check if video already exists
                    existing_video_query = select(Video).where(Video.video_id == video_id)
                    result = session.execute(existing_video_query)
                    existing_video = result.scalar_one_or_none()
                    
                    if existing_video:
                        self._logger.info(f"Video {video_id} already exists, returning existing record")
                        return existing_video
                    
                    # Create new video record
                    video = Video(
                        video_id=video_id,
                        title=title,
                        duration=duration,
                        url=url or f"https://www.youtube.com/watch?v={video_id}"
                    )
                    
                    session.add(video)
                    session.commit()
                    session.refresh(video)
                    
                    # Create transcript record if content provided
                    if transcript_content:
                        from ..database.models import Transcript
                        transcript = Transcript(
                            video_id=video.id,
                            content=transcript_content,
                            language=transcript_language or 'unknown'
                        )
                        session.add(transcript)
                        session.commit()
                    
                    self._logger.info(f"Created video record for {video_id}")
                    return video
                
            except IntegrityError as e:
                # Handle constraint violations (e.g., duplicate video_id)
                self._logger.warning(f"Video {video_id} already exists (constraint violation): {e}")
                # Try to return existing video
                try:
                    with get_database_session() as session:
                        existing_video_query = select(Video).where(Video.video_id == video_id)
                        result = session.execute(existing_video_query)
                        existing_video = result.scalar_one_or_none()
                        if existing_video:
                            return existing_video
                except Exception:
                    pass
                return None
            
            except Exception as e:
                self._logger.error(f"Unexpected error creating video record: {e}")
                raise VideoServiceError(f"Unexpected error: {e}")

    def get_video_by_video_id(self, video_id: str) -> Optional[Video]:
        """
        Get video by YouTube video ID.
        
        Args:
            video_id: YouTube video ID
            
        Returns:
            Video object if found, None otherwise
        """
        try:
            session = self._get_session()
            
            stmt = select(Video).where(Video.video_id == video_id)
            result = session.execute(stmt)
            video = result.scalar_one_or_none()
            
            return video
            
        except SQLAlchemyError as e:
            self._logger.error(f"Database error getting video: {e}")
            raise VideoServiceError(f"Failed to get video: {e}")

    def get_video_by_id(self, video_id: int) -> Optional[Video]:
        """
        Get video by database ID with all related data.
        
        Args:
            video_id: Database video ID
            
        Returns:
            Video object with all related data if found, None otherwise
        """
        try:
            session = self._get_session()
            
            stmt = select(Video).options(
                selectinload(Video.transcripts),
                selectinload(Video.summaries),
                selectinload(Video.keywords),
                selectinload(Video.timestamped_segments),
                selectinload(Video.processing_metadata)
            ).where(Video.id == video_id)
            
            result = session.execute(stmt)
            video = result.scalar_one_or_none()
            
            return video
            
        except SQLAlchemyError as e:
            self._logger.error(f"Database error getting video by ID: {e}")
            raise VideoServiceError(f"Failed to get video by ID: {e}")

    def save_transcript(
        self,
        video_id: str,
        content: str,
        language: Optional[str] = None
    ) -> Transcript:
        """
        Save transcript for a video.
        
        Args:
            video_id: YouTube video ID
            content: Transcript content
            language: Language code
            
        Returns:
            Created Transcript object
            
        Raises:
            VideoServiceError: If saving fails
        """
        try:
            session = self._get_session()
            
            # Get video record
            video = self.get_video_by_video_id(video_id)
            if not video:
                raise VideoServiceError(f"Video {video_id} not found")
            
            # Create transcript record
            transcript = Transcript(
                video_id=video.id,
                content=content,
                language=language
            )
            
            session.add(transcript)
            session.commit()
            session.refresh(transcript)
            
            self._logger.info(f"Saved transcript for video {video_id}")
            return transcript
            
        except SQLAlchemyError as e:
            self._logger.error(f"Database error saving transcript: {e}")
            raise VideoServiceError(f"Failed to save transcript: {e}")
        except Exception as e:
            self._logger.error(f"Unexpected error saving transcript: {e}")
            raise VideoServiceError(f"Unexpected error: {e}")

    def save_summary(
        self,
        video_id: str,
        content: str,
        processing_time: Optional[float] = None
    ) -> Summary:
        """
        Save summary for a video.
        
        Args:
            video_id: YouTube video ID
            content: Summary content
            processing_time: Time taken to generate summary
            
        Returns:
            Created Summary object
            
        Raises:
            VideoServiceError: If saving fails
        """
        try:
            # Use session context manager properly
            if self._session:
                # Use provided session directly (not available in current setup)
                stmt = select(Video).where(Video.video_id == video_id)
                result = self._session.execute(stmt)
                video = result.scalar_one_or_none()
                
                if not video:
                    raise VideoServiceError(f"Video {video_id} not found")
                
                summary = Summary(
                    video_id=video.id,
                    content=content,
                    processing_time=processing_time
                )
                
                self._session.add(summary)
                self._session.flush()  # Don't commit here, let caller handle transaction
                self._session.refresh(summary)
                
            else:
                # Create new session using context manager
                from ..database.connection import get_database_session
                with get_database_session() as session:
                    # Get video record using the session
                    stmt = select(Video).where(Video.video_id == video_id)
                    result = session.execute(stmt)
                    video = result.scalar_one_or_none()
                    
                    if not video:
                        # Create a basic video record if it doesn't exist
                        video = Video(
                            video_id=video_id,
                            title=f"Video {video_id}",  # Will be updated with actual title later
                            duration=0,  # Will be updated later
                            url=f"https://www.youtube.com/watch?v={video_id}"
                        )
                        session.add(video)
                        session.flush()  # Get the ID without committing yet
                    
                    # Create summary record
                    summary = Summary(
                        video_id=video.id,
                        content=content,
                        processing_time=processing_time
                    )
                    
                    session.add(summary)
                    session.commit()
                    session.refresh(summary)
            
            self._logger.info(f"Saved summary for video {video_id}")
            return summary
            
        except SQLAlchemyError as e:
            self._logger.error(f"Database error saving summary: {e}")
            raise VideoServiceError(f"Failed to save summary: {e}")
        except Exception as e:
            self._logger.error(f"Unexpected error saving summary: {e}")
            raise VideoServiceError(f"Unexpected error: {e}")

    def save_keywords(
        self,
        video_id: str,
        keywords_data: Dict[str, Any]
    ) -> Keyword:
        """
        Save keywords for a video.
        
        Args:
            video_id: YouTube video ID
            keywords_data: Keywords data as JSON
            
        Returns:
            Created Keyword object
            
        Raises:
            VideoServiceError: If saving fails
        """
        try:
            # Use session context manager properly
            if self._session:
                # Use provided session directly
                stmt = select(Video).where(Video.video_id == video_id)
                result = self._session.execute(stmt)
                video = result.scalar_one_or_none()
                
                if not video:
                    # Create a basic video record if it doesn't exist
                    video = Video(
                        video_id=video_id,
                        title=f"Video {video_id}",
                        duration=0,
                        url=f"https://www.youtube.com/watch?v={video_id}"
                    )
                    self._session.add(video)
                    self._session.flush()
                
                keyword = Keyword(
                    video_id=video.id,
                    keywords_json=keywords_data
                )
                
                self._session.add(keyword)
                self._session.flush()  # Don't commit here, let caller handle transaction
                self._session.refresh(keyword)
                
            else:
                # Create new session using context manager
                from ..database.connection import get_database_session
                with get_database_session() as session:
                    # Get video record
                    stmt = select(Video).where(Video.video_id == video_id)
                    result = session.execute(stmt)
                    video = result.scalar_one_or_none()
                    
                    if not video:
                        # Create a basic video record if it doesn't exist
                        video = Video(
                            video_id=video_id,
                            title=f"Video {video_id}",
                            duration_seconds=0,
                            description="",
                            channel_name="Unknown",
                            upload_date=datetime.utcnow()
                        )
                        session.add(video)
                        session.flush()
                    
                    # Create keyword record
                    keyword = Keyword(
                        video_id=video.id,
                        keywords_json=keywords_data
                    )
                    
                    session.add(keyword)
                    # Don't call commit here - let the context manager handle it
                    session.flush()  # Flush to get the ID
                    session.refresh(keyword)
            
            self._logger.info(f"Saved keywords for video {video_id}")
            return keyword
            
        except SQLAlchemyError as e:
            self._logger.error(f"Database error saving keywords: {e}")
            raise VideoServiceError(f"Failed to save keywords: {e}")
        except Exception as e:
            self._logger.error(f"Unexpected error saving keywords: {e}")
            raise VideoServiceError(f"Unexpected error: {e}")

    def save_timestamped_segments(
        self,
        video_id: str,
        segments_data: Dict[str, Any]
    ) -> TimestampedSegment:
        """
        Save timestamped segments for a video.
        
        Args:
            video_id: YouTube video ID
            segments_data: Segments data as JSON
            
        Returns:
            Created TimestampedSegment object
            
        Raises:
            VideoServiceError: If saving fails
        """
        try:
            # Use session context manager properly
            if self._session:
                # Use provided session directly
                stmt = select(Video).where(Video.video_id == video_id)
                result = self._session.execute(stmt)
                video = result.scalar_one_or_none()
                
                if not video:
                    # Create a basic video record if it doesn't exist
                    video = Video(
                        video_id=video_id,
                        title=f"Video {video_id}",
                        duration=0,
                        url=f"https://www.youtube.com/watch?v={video_id}"
                    )
                    self._session.add(video)
                    self._session.flush()
                
                segment = TimestampedSegment(
                    video_id=video.id,
                    segments_json=segments_data
                )
                
                self._session.add(segment)
                self._session.flush()  # Don't commit here, let caller handle transaction
                self._session.refresh(segment)
                
            else:
                # Create new session using context manager
                from ..database.connection import get_database_session
                with get_database_session() as session:
                    # Get video record
                    stmt = select(Video).where(Video.video_id == video_id)
                    result = session.execute(stmt)
                    video = result.scalar_one_or_none()
                    
                    if not video:
                        # Create a basic video record if it doesn't exist
                        video = Video(
                            video_id=video_id,
                            title=f"Video {video_id}",
                            duration_seconds=0,
                            description="",
                            channel_name="Unknown",
                            upload_date=datetime.utcnow()
                        )
                        session.add(video)
                        session.flush()
                    
                    # Create timestamped segment record
                    segment = TimestampedSegment(
                        video_id=video.id,
                        segments_json=segments_data
                    )
                    
                    session.add(segment)
                    # Don't call commit here - let the context manager handle it
                    session.flush()  # Flush to get the ID
                    session.refresh(segment)
            
            self._logger.info(f"Saved timestamped segments for video {video_id}")
            return segment
            
        except SQLAlchemyError as e:
            self._logger.error(f"Database error saving segments: {e}")
            raise VideoServiceError(f"Failed to save segments: {e}")
        except Exception as e:
            self._logger.error(f"Unexpected error saving segments: {e}")
            raise VideoServiceError(f"Unexpected error: {e}")

    def save_processing_metadata(
        self,
        video_id: str,
        workflow_params: Optional[Dict[str, Any]] = None,
        status: str = "pending",
        error_info: Optional[str] = None
    ) -> ProcessingMetadata:
        """
        Save processing metadata for a video.
        
        Args:
            video_id: YouTube video ID
            workflow_params: Workflow parameters
            status: Processing status
            error_info: Error information if any
            
        Returns:
            Created ProcessingMetadata object
            
        Raises:
            VideoServiceError: If saving fails
        """
        try:
            session = self._get_session()
            
            # Get video record
            video = self.get_video_by_video_id(video_id)
            if not video:
                raise VideoServiceError(f"Video {video_id} not found")
            
            # Create processing metadata record
            metadata = ProcessingMetadata(
                video_id=video.id,
                workflow_params=workflow_params,
                status=status,
                error_info=error_info
            )
            
            session.add(metadata)
            session.commit()
            session.refresh(metadata)
            
            self._logger.info(f"Saved processing metadata for video {video_id}")
            return metadata
            
        except SQLAlchemyError as e:
            self._logger.error(f"Database error saving metadata: {e}")
            raise VideoServiceError(f"Failed to save metadata: {e}")
        except Exception as e:
            self._logger.error(f"Unexpected error saving metadata: {e}")
            raise VideoServiceError(f"Unexpected error: {e}")

    def update_processing_status(
        self,
        video_id: str,
        status: str,
        error_info: Optional[str] = None
    ) -> Optional[ProcessingMetadata]:
        """
        Update processing status for a video.
        
        Args:
            video_id: YouTube video ID
            status: New processing status
            error_info: Error information if any
            
        Returns:
            Updated ProcessingMetadata object or None if not found
            
        Raises:
            VideoServiceError: If update fails
        """
        try:
            session = self._get_session()
            
            # Get video record
            video = self.get_video_by_video_id(video_id)
            if not video:
                raise VideoServiceError(f"Video {video_id} not found")
            
            # Update processing metadata
            stmt = update(ProcessingMetadata).where(
                ProcessingMetadata.video_id == video.id
            ).values(
                status=status,
                error_info=error_info
            )
            
            session.execute(stmt)
            session.commit()
            
            # Get updated metadata
            metadata_stmt = select(ProcessingMetadata).where(
                ProcessingMetadata.video_id == video.id
            ).order_by(ProcessingMetadata.created_at.desc())
            
            result = session.execute(metadata_stmt)
            metadata = result.scalar_one_or_none()
            
            self._logger.info(f"Updated processing status for video {video_id} to {status}")
            return metadata
            
        except SQLAlchemyError as e:
            self._logger.error(f"Database error updating status: {e}")
            raise VideoServiceError(f"Failed to update status: {e}")
        except Exception as e:
            self._logger.error(f"Unexpected error updating status: {e}")
            raise VideoServiceError(f"Unexpected error: {e}")

    def video_exists(self, video_id: str) -> bool:
        """
        Check if a video exists in the database.
        
        Args:
            video_id: YouTube video ID
            
        Returns:
            True if video exists, False otherwise
        """
        try:
            video = self.get_video_by_video_id(video_id)
            return video is not None
            
        except VideoServiceError:
            return False

    def get_processing_status(self, video_id: str) -> Optional[str]:
        """
        Get current processing status for a video.
        
        Args:
            video_id: YouTube video ID
            
        Returns:
            Processing status or None if not found
        """
        try:
            session = self._get_session()
            
            # Get video record
            video = self.get_video_by_video_id(video_id)
            if not video:
                return None
            
            # Get latest processing metadata
            stmt = select(ProcessingMetadata).where(
                ProcessingMetadata.video_id == video.id
            ).order_by(ProcessingMetadata.created_at.desc())
            
            result = session.execute(stmt)
            metadata = result.scalar_one_or_none()
            
            return metadata.status if metadata else None
            
        except SQLAlchemyError as e:
            self._logger.error(f"Database error getting status: {e}")
            return None

    def delete_video_data(self, video_id: str) -> bool:
        """
        Delete all data for a video (cascade delete).
        
        Args:
            video_id: YouTube video ID
            
        Returns:
            True if deletion succeeded, False otherwise
        """
        try:
            session = self._get_session()
            
            # Get video record
            video = self.get_video_by_video_id(video_id)
            if not video:
                self._logger.warning(f"Video {video_id} not found for deletion")
                return False
            
            # Delete video (cascade will handle related records)
            session.delete(video)
            session.commit()
            
            self._logger.info(f"Deleted video data for {video_id}")
            return True
            
        except SQLAlchemyError as e:
            self._logger.error(f"Database error deleting video: {e}")
            raise VideoServiceError(f"Failed to delete video: {e}")
        except Exception as e:
            self._logger.error(f"Unexpected error deleting video: {e}")
            raise VideoServiceError(f"Unexpected error: {e}")


# Convenience function for creating video service with session
def create_video_service_with_session() -> VideoService:
    """
    Create a VideoService instance with a new database session.
    
    Returns:
        VideoService instance
    """
    # This would typically be used in contexts where you need a standalone service
    # In FastAPI, you'd use dependency injection instead
    return VideoService()


# Dependency injection function for FastAPI
def get_video_service(session: Session) -> VideoService:
    """
    Get VideoService instance for dependency injection.
    
    Args:
        session: Database session from FastAPI dependency
        
    Returns:
        VideoService instance
    """
    return VideoService(session)