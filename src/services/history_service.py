"""
History service for querying processed videos.

This service provides methods to query video processing history,
including pagination, filtering, and search capabilities.
"""

import logging
from datetime import datetime, date
from typing import Optional, Dict, Any, List, Tuple
from sqlalchemy.orm import Session
from sqlalchemy.orm import selectinload, joinedload
from sqlalchemy import select, func, and_, or_, text, delete
from sqlalchemy.exc import SQLAlchemyError
from dataclasses import dataclass
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
from ..database.cascade_delete import (
    CascadeDeleteManager, CascadeDeleteResult, CascadeDeleteValidation,
    create_cascade_delete_manager, validate_video_deletion, execute_enhanced_cascade_delete
)

logger = logging.getLogger(__name__)


@dataclass
class PaginationInfo:
    """Pagination information for query results."""
    page: int
    page_size: int
    total_items: int
    total_pages: int
    has_next: bool
    has_previous: bool


@dataclass
class VideoHistoryItem:
    """Data class for video history items."""
    id: int
    video_id: str
    title: str
    duration: Optional[int]
    url: str
    created_at: datetime
    updated_at: datetime
    processing_status: Optional[str]
    has_transcript: bool
    has_summary: bool
    has_keywords: bool
    has_segments: bool


class HistoryServiceError(Exception):
    """Custom exception for history service operations."""
    pass


class HistoryService:
    """
    Service class for querying video processing history.
    
    This class provides methods to retrieve video processing history
    with pagination, filtering, and search capabilities.
    """

    def __init__(self, session: Optional[Session] = None):
        """
        Initialize the history service.
        
        Args:
            session: Optional database session. If not provided, will use dependency injection.
        """
        self._session = session
        self._logger = logging.getLogger(f"{__name__}.HistoryService")

    def _get_session(self) -> Session:
        """Get database session (internal method)."""
        if self._session:
            return self._session
        else:
            # This should be used with dependency injection
            raise HistoryServiceError("No database session provided")

    def get_videos_paginated(
        self,
        page: int = 1,
        page_size: int = 20,
        sort_by: str = "created_at",
        sort_order: str = "desc"
    ) -> Tuple[List[VideoHistoryItem], PaginationInfo]:
        """
        Get paginated list of videos with processing information.
        
        Args:
            page: Page number (1-based)
            page_size: Number of items per page
            sort_by: Field to sort by (created_at, updated_at, title)
            sort_order: Sort order (asc, desc)
            
        Returns:
            Tuple of (video_items, pagination_info)
            
        Raises:
            HistoryServiceError: If query fails
        """
        try:
            session = self._get_session()
            
            # Validate parameters
            if page < 1:
                page = 1
            if page_size < 1 or page_size > 100:
                page_size = 20
            if sort_by not in ["created_at", "updated_at", "title"]:
                sort_by = "created_at"
            if sort_order not in ["asc", "desc"]:
                sort_order = "desc"
            
            # Build base query with eager loading
            query = select(Video).options(
                selectinload(Video.processing_metadata),
                selectinload(Video.transcripts),
                selectinload(Video.summaries),
                selectinload(Video.keywords),
                selectinload(Video.timestamped_segments)
            )
            
            # Apply sorting
            sort_column = getattr(Video, sort_by)
            if sort_order == "desc":
                query = query.order_by(sort_column.desc())
            else:
                query = query.order_by(sort_column.asc())
            
            # Get total count
            count_query = select(func.count(Video.id))
            count_result = session.execute(count_query)
            total_items = count_result.scalar()
            
            # Calculate pagination
            total_pages = (total_items + page_size - 1) // page_size
            offset = (page - 1) * page_size
            
            # Apply pagination
            query = query.offset(offset).limit(page_size)
            
            # Execute query
            result = session.execute(query)
            videos = result.scalars().all()
            
            # Convert to VideoHistoryItem objects
            video_items = []
            for video in videos:
                # Get latest processing status
                processing_status = None
                if video.processing_metadata:
                    latest_metadata = max(video.processing_metadata, key=lambda m: m.created_at)
                    processing_status = latest_metadata.status
                
                video_item = VideoHistoryItem(
                    id=video.id,
                    video_id=video.video_id,
                    title=video.title,
                    duration=video.duration,
                    url=video.url,
                    created_at=video.created_at,
                    updated_at=video.updated_at,
                    processing_status=processing_status,
                    has_transcript=len(video.transcripts) > 0,
                    has_summary=len(video.summaries) > 0,
                    has_keywords=len(video.keywords) > 0,
                    has_segments=len(video.timestamped_segments) > 0
                )
                video_items.append(video_item)
            
            # Create pagination info
            pagination_info = PaginationInfo(
                page=page,
                page_size=page_size,
                total_items=total_items,
                total_pages=total_pages,
                has_next=page < total_pages,
                has_previous=page > 1
            )
            
            self._logger.info(f"Retrieved {len(video_items)} videos (page {page}/{total_pages})")
            return video_items, pagination_info
            
        except SQLAlchemyError as e:
            db_error = classify_database_error(e)
            self._logger.error(f"Database error getting paginated videos: {db_error}")
            raise HistoryServiceError(f"Failed to get videos: {db_error.message}")
        except Exception as e:
            self._logger.error(f"Unexpected error getting paginated videos: {e}")
            raise HistoryServiceError(f"Unexpected error: {e}")

    def get_video_by_id(self, video_id: int) -> Optional[Video]:
        """
        Get detailed video information by database ID.
        
        Args:
            video_id: Database video ID
            
        Returns:
            Video object with all related data if found, None otherwise
            
        Raises:
            HistoryServiceError: If query fails
        """
        try:
            session = self._get_session()
            
            # Query with all related data
            query = select(Video).options(
                selectinload(Video.transcripts),
                selectinload(Video.summaries),
                selectinload(Video.keywords),
                selectinload(Video.timestamped_segments),
                selectinload(Video.processing_metadata)
            ).where(Video.id == video_id)
            
            result = session.execute(query)
            video = result.scalar_one_or_none()
            
            if video:
                self._logger.info(f"Retrieved video details for ID {video_id}")
            else:
                self._logger.info(f"Video with ID {video_id} not found")
            
            return video
            
        except SQLAlchemyError as e:
            db_error = classify_database_error(e)
            self._logger.error(f"Database error getting video by ID: {db_error}")
            raise HistoryServiceError(f"Failed to get video: {db_error.message}")
        except Exception as e:
            self._logger.error(f"Unexpected error getting video by ID: {e}")
            raise HistoryServiceError(f"Unexpected error: {e}")

    def search_videos(
        self,
        query: str,
        page: int = 1,
        page_size: int = 20
    ) -> Tuple[List[VideoHistoryItem], PaginationInfo]:
        """
        Search videos by title or content.
        
        Args:
            query: Search query string
            page: Page number (1-based)
            page_size: Number of items per page
            
        Returns:
            Tuple of (video_items, pagination_info)
            
        Raises:
            HistoryServiceError: If search fails
        """
        try:
            session = self._get_session()
            
            # Validate parameters
            if page < 1:
                page = 1
            if page_size < 1 or page_size > 100:
                page_size = 20
            
            # Sanitize search query
            search_term = f"%{query.strip()}%"
            
            # Build search query
            search_query = select(Video).options(
                selectinload(Video.processing_metadata),
                selectinload(Video.transcripts),
                selectinload(Video.summaries),
                selectinload(Video.keywords),
                selectinload(Video.timestamped_segments)
            ).where(
                or_(
                    Video.title.ilike(search_term),
                    Video.transcripts.any(Transcript.content.ilike(search_term)),
                    Video.summaries.any(Summary.content.ilike(search_term))
                )
            ).order_by(Video.created_at.desc())
            
            # Get total count for search
            count_query = select(func.count(Video.id)).where(
                or_(
                    Video.title.ilike(search_term),
                    Video.transcripts.any(Transcript.content.ilike(search_term)),
                    Video.summaries.any(Summary.content.ilike(search_term))
                )
            )
            count_result = session.execute(count_query)
            total_items = count_result.scalar()
            
            # Calculate pagination
            total_pages = (total_items + page_size - 1) // page_size
            offset = (page - 1) * page_size
            
            # Apply pagination
            search_query = search_query.offset(offset).limit(page_size)
            
            # Execute search
            result = session.execute(search_query)
            videos = result.scalars().all()
            
            # Convert to VideoHistoryItem objects
            video_items = []
            for video in videos:
                # Get latest processing status
                processing_status = None
                if video.processing_metadata:
                    latest_metadata = max(video.processing_metadata, key=lambda m: m.created_at)
                    processing_status = latest_metadata.status
                
                video_item = VideoHistoryItem(
                    id=video.id,
                    video_id=video.video_id,
                    title=video.title,
                    duration=video.duration,
                    url=video.url,
                    created_at=video.created_at,
                    updated_at=video.updated_at,
                    processing_status=processing_status,
                    has_transcript=len(video.transcripts) > 0,
                    has_summary=len(video.summaries) > 0,
                    has_keywords=len(video.keywords) > 0,
                    has_segments=len(video.timestamped_segments) > 0
                )
                video_items.append(video_item)
            
            # Create pagination info
            pagination_info = PaginationInfo(
                page=page,
                page_size=page_size,
                total_items=total_items,
                total_pages=total_pages,
                has_next=page < total_pages,
                has_previous=page > 1
            )
            
            self._logger.info(f"Search for '{query}' returned {len(video_items)} videos")
            return video_items, pagination_info
            
        except SQLAlchemyError as e:
            db_error = classify_database_error(e)
            self._logger.error(f"Database error searching videos: {db_error}")
            raise HistoryServiceError(f"Failed to search videos: {db_error.message}")
        except Exception as e:
            self._logger.error(f"Unexpected error searching videos: {e}")
            raise HistoryServiceError(f"Unexpected error: {e}")

    def filter_videos_by_date(
        self,
        date_from: Optional[date] = None,
        date_to: Optional[date] = None,
        page: int = 1,
        page_size: int = 20
    ) -> Tuple[List[VideoHistoryItem], PaginationInfo]:
        """
        Filter videos by date range.
        
        Args:
            date_from: Start date (inclusive)
            date_to: End date (inclusive)
            page: Page number (1-based)
            page_size: Number of items per page
            
        Returns:
            Tuple of (video_items, pagination_info)
            
        Raises:
            HistoryServiceError: If filtering fails
        """
        try:
            session = self._get_session()
            
            # Validate parameters
            if page < 1:
                page = 1
            if page_size < 1 or page_size > 100:
                page_size = 20
            
            # Build date filter query
            query = select(Video).options(
                selectinload(Video.processing_metadata),
                selectinload(Video.transcripts),
                selectinload(Video.summaries),
                selectinload(Video.keywords),
                selectinload(Video.timestamped_segments)
            )
            
            # Apply date filters
            date_conditions = []
            if date_from:
                date_conditions.append(Video.created_at >= datetime.combine(date_from, datetime.min.time()))
            if date_to:
                date_conditions.append(Video.created_at <= datetime.combine(date_to, datetime.max.time()))
            
            if date_conditions:
                query = query.where(and_(*date_conditions))
            
            # Order by creation date
            query = query.order_by(Video.created_at.desc())
            
            # Get total count
            count_query = select(func.count(Video.id))
            if date_conditions:
                count_query = count_query.where(and_(*date_conditions))
            
            count_result = session.execute(count_query)
            total_items = count_result.scalar()
            
            # Calculate pagination
            total_pages = (total_items + page_size - 1) // page_size
            offset = (page - 1) * page_size
            
            # Apply pagination
            query = query.offset(offset).limit(page_size)
            
            # Execute query
            result = session.execute(query)
            videos = result.scalars().all()
            
            # Convert to VideoHistoryItem objects
            video_items = []
            for video in videos:
                # Get latest processing status
                processing_status = None
                if video.processing_metadata:
                    latest_metadata = max(video.processing_metadata, key=lambda m: m.created_at)
                    processing_status = latest_metadata.status
                
                video_item = VideoHistoryItem(
                    id=video.id,
                    video_id=video.video_id,
                    title=video.title,
                    duration=video.duration,
                    url=video.url,
                    created_at=video.created_at,
                    updated_at=video.updated_at,
                    processing_status=processing_status,
                    has_transcript=len(video.transcripts) > 0,
                    has_summary=len(video.summaries) > 0,
                    has_keywords=len(video.keywords) > 0,
                    has_segments=len(video.timestamped_segments) > 0
                )
                video_items.append(video_item)
            
            # Create pagination info
            pagination_info = PaginationInfo(
                page=page,
                page_size=page_size,
                total_items=total_items,
                total_pages=total_pages,
                has_next=page < total_pages,
                has_previous=page > 1
            )
            
            self._logger.info(f"Date filter returned {len(video_items)} videos")
            return video_items, pagination_info
            
        except SQLAlchemyError as e:
            db_error = classify_database_error(e)
            self._logger.error(f"Database error filtering videos by date: {db_error}")
            raise HistoryServiceError(f"Failed to filter videos: {db_error.message}")
        except Exception as e:
            self._logger.error(f"Unexpected error filtering videos by date: {e}")
            raise HistoryServiceError(f"Unexpected error: {e}")

    def _calculate_pagination_info(self, page: int, page_size: int, total_items: int) -> PaginationInfo:
        """
        Calculate pagination information.
        
        Args:
            page: Current page number
            page_size: Items per page
            total_items: Total number of items
            
        Returns:
            PaginationInfo object
        """
        total_pages = (total_items + page_size - 1) // page_size
        
        return PaginationInfo(
            page=page,
            page_size=page_size,
            total_items=total_items,
            total_pages=total_pages,
            has_next=page < total_pages,
            has_previous=page > 1
        )

    def get_video_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about processed videos.
        
        Returns:
            Dictionary containing video statistics
            
        Raises:
            HistoryServiceError: If query fails
        """
        try:
            session = self._get_session()
            
            # Total videos
            total_videos_query = select(func.count(Video.id))
            total_videos_result = session.execute(total_videos_query)
            total_videos = total_videos_result.scalar()
            
            # Videos with transcripts
            videos_with_transcripts_query = select(func.count(Video.id.distinct())).where(
                Video.transcripts.any()
            )
            videos_with_transcripts_result = session.execute(videos_with_transcripts_query)
            videos_with_transcripts = videos_with_transcripts_result.scalar()
            
            # Videos with summaries
            videos_with_summaries_query = select(func.count(Video.id.distinct())).where(
                Video.summaries.any()
            )
            videos_with_summaries_result = session.execute(videos_with_summaries_query)
            videos_with_summaries = videos_with_summaries_result.scalar()
            
            # Processing status counts
            status_counts_query = select(
                ProcessingMetadata.status,
                func.count(ProcessingMetadata.status)
            ).group_by(ProcessingMetadata.status)
            status_counts_result = session.execute(status_counts_query)
            status_counts = dict(status_counts_result.fetchall())
            
            statistics = {
                "total_videos": total_videos,
                "videos_with_transcripts": videos_with_transcripts,
                "videos_with_summaries": videos_with_summaries,
                "videos_with_keywords": 0,  # Will be calculated if needed
                "videos_with_segments": 0,  # Will be calculated if needed
                "processing_status_counts": status_counts,
                "completion_rate": (videos_with_summaries / total_videos * 100) if total_videos > 0 else 0
            }
            
            self._logger.info("Retrieved video statistics")
            return statistics
            
        except SQLAlchemyError as e:
            db_error = classify_database_error(e)
            self._logger.error(f"Database error getting video statistics: {db_error}")
            raise HistoryServiceError(f"Failed to get statistics: {db_error.message}")
        except Exception as e:
            self._logger.error(f"Unexpected error getting video statistics: {e}")
            raise HistoryServiceError(f"Unexpected error: {e}")

    def delete_video_by_id(self, video_id: int) -> bool:
        """
        Delete a specific video record and all its associated data.
        
        This method performs a cascading delete that removes:
        - Video record from the videos table
        - All associated transcripts
        - All associated summaries
        - All associated keywords
        - All associated timestamped segments
        - All associated processing metadata
        
        Args:
            video_id: Database video ID to delete
            
        Returns:
            True if video was deleted successfully, False if video was not found
            
        Raises:
            HistoryServiceError: If deletion fails
        """
        try:
            session = self._get_session()
            
            # First, verify the video exists
            video = session.get(Video, video_id)
            if not video:
                self._logger.warning(f"Video with ID {video_id} not found for deletion")
                return False
            
            # Store video info for logging
            video_info = {
                'id': video.id,
                'video_id': video.video_id,
                'title': video.title,
                'url': video.url
            }
            
            # Delete the video (cascading delete will handle related records)
            session.delete(video)
            session.commit()
            
            self._logger.info(f"Successfully deleted video: {video_info}")
            return True
            
        except SQLAlchemyError as e:
            db_error = classify_database_error(e)
            self._logger.error(f"Database error deleting video {video_id}: {db_error}")
            raise HistoryServiceError(f"Failed to delete video: {db_error.message}")
        except Exception as e:
            self._logger.error(f"Unexpected error deleting video {video_id}: {e}")
            raise HistoryServiceError(f"Unexpected error: {e}")

    def delete_video_by_video_id(self, video_id: str) -> bool:
        """
        Delete a specific video record by YouTube video ID.
        
        Args:
            video_id: YouTube video ID (11 characters)
            
        Returns:
            True if video was deleted successfully, False if video was not found
            
        Raises:
            HistoryServiceError: If deletion fails
        """
        try:
            session = self._get_session()
            
            # Find the video by YouTube video ID
            query = select(Video).where(Video.video_id == video_id)
            result = session.execute(query)
            video = result.scalar_one_or_none()
            
            if not video:
                self._logger.warning(f"Video with video_id '{video_id}' not found for deletion")
                return False
            
            # Use the database ID deletion method
            return self.delete_video_by_id(video.id)
            
        except SQLAlchemyError as e:
            db_error = classify_database_error(e)
            self._logger.error(f"Database error deleting video by video_id '{video_id}': {db_error}")
            raise HistoryServiceError(f"Failed to delete video: {db_error.message}")
        except Exception as e:
            self._logger.error(f"Unexpected error deleting video by video_id '{video_id}': {e}")
            raise HistoryServiceError(f"Unexpected error: {e}")

    def delete_multiple_videos(self, video_ids: List[int]) -> Dict[str, Any]:
        """
        Delete multiple videos by their database IDs.
        
        Args:
            video_ids: List of database video IDs to delete
            
        Returns:
            Dictionary containing deletion results:
            {
                "deleted_count": int,
                "failed_count": int,
                "not_found_count": int,
                "deleted_videos": List[Dict],
                "failed_videos": List[Dict]
            }
            
        Raises:
            HistoryServiceError: If batch deletion fails
        """
        try:
            session = self._get_session()
            
            results = {
                "deleted_count": 0,
                "failed_count": 0,
                "not_found_count": 0,
                "deleted_videos": [],
                "failed_videos": []
            }
            
            for video_id in video_ids:
                try:
                    # Verify the video exists
                    video = session.get(Video, video_id)
                    if not video:
                        results["not_found_count"] += 1
                        results["failed_videos"].append({
                            "video_id": video_id,
                            "error": "Video not found"
                        })
                        continue
                    
                    # Store video info for results
                    video_info = {
                        "id": video.id,
                        "video_id": video.video_id,
                        "title": video.title,
                        "url": video.url
                    }
                    
                    # Delete the video
                    session.delete(video)
                    session.commit()
                    
                    results["deleted_count"] += 1
                    results["deleted_videos"].append(video_info)
                    
                except Exception as e:
                    results["failed_count"] += 1
                    results["failed_videos"].append({
                        "video_id": video_id,
                        "error": str(e)
                    })
                    self._logger.error(f"Failed to delete video {video_id}: {e}")
                    # Continue with next video
                    continue
            
            self._logger.info(f"Batch deletion completed: {results['deleted_count']} deleted, "
                            f"{results['failed_count']} failed, {results['not_found_count']} not found")
            
            return results
            
        except SQLAlchemyError as e:
            db_error = classify_database_error(e)
            self._logger.error(f"Database error in batch deletion: {db_error}")
            raise HistoryServiceError(f"Failed to delete videos: {db_error.message}")
        except Exception as e:
            self._logger.error(f"Unexpected error in batch deletion: {e}")
            raise HistoryServiceError(f"Unexpected error: {e}")

    def get_video_deletion_info(self, video_id: int) -> Optional[Dict[str, Any]]:
        """
        Get information about what will be deleted when deleting a video.
        
        Args:
            video_id: Database video ID
            
        Returns:
            Dictionary containing deletion impact information or None if video not found
            
        Raises:
            HistoryServiceError: If query fails
        """
        try:
            session = self._get_session()
            
            # Get video with related data counts
            query = select(Video).options(
                selectinload(Video.transcripts),
                selectinload(Video.summaries),
                selectinload(Video.keywords),
                selectinload(Video.timestamped_segments),
                selectinload(Video.processing_metadata)
            ).where(Video.id == video_id)
            
            result = session.execute(query)
            video = result.scalar_one_or_none()
            
            if not video:
                return None
            
            deletion_info = {
                "video": {
                    "id": video.id,
                    "video_id": video.video_id,
                    "title": video.title,
                    "url": video.url,
                    "created_at": video.created_at,
                    "updated_at": video.updated_at
                },
                "related_data_counts": {
                    "transcripts": len(video.transcripts),
                    "summaries": len(video.summaries),
                    "keywords": len(video.keywords),
                    "timestamped_segments": len(video.timestamped_segments),
                    "processing_metadata": len(video.processing_metadata)
                },
                "total_related_records": (
                    len(video.transcripts) + len(video.summaries) + 
                    len(video.keywords) + len(video.timestamped_segments) + 
                    len(video.processing_metadata)
                )
            }
            
            return deletion_info
            
        except SQLAlchemyError as e:
            db_error = classify_database_error(e)
            self._logger.error(f"Database error getting deletion info for video {video_id}: {db_error}")
            raise HistoryServiceError(f"Failed to get deletion info: {db_error.message}")
        except Exception as e:
            self._logger.error(f"Unexpected error getting deletion info for video {video_id}: {e}")
            raise HistoryServiceError(f"Unexpected error: {e}")

    def validate_video_deletion(self, video_id: int) -> CascadeDeleteValidation:
        """
        Validate that a video can be safely deleted using enhanced cascade delete logic.
        
        Args:
            video_id: Database video ID
            
        Returns:
            CascadeDeleteValidation with detailed validation results
            
        Raises:
            HistoryServiceError: If validation fails
        """
        try:
            session = self._get_session()
            validation = validate_video_deletion(session, video_id)
            
            self._logger.info(f"Validation for video {video_id}: can_delete={validation.can_delete}, "
                            f"related_records={validation.total_related_records}")
            
            return validation
            
        except Exception as e:
            self._logger.error(f"Error validating video deletion for {video_id}: {e}")
            raise HistoryServiceError(f"Failed to validate deletion: {e}")

    def enhanced_delete_video_by_id(self, video_id: int, force: bool = False) -> CascadeDeleteResult:
        """
        Delete a video using enhanced cascade delete with validation and monitoring.
        
        Args:
            video_id: Database video ID to delete
            force: Skip validation if True
            
        Returns:
            CascadeDeleteResult with detailed operation results
            
        Raises:
            HistoryServiceError: If deletion fails
        """
        try:
            session = self._get_session()
            result = execute_enhanced_cascade_delete(session, video_id, force=force)
            
            if result.success:
                self._logger.info(f"Enhanced cascade delete successful for video {video_id}: "
                                f"deleted {result.total_deleted} records in {result.execution_time_ms:.2f}ms")
            else:
                self._logger.error(f"Enhanced cascade delete failed for video {video_id}: {result.error_message}")
            
            return result
            
        except Exception as e:
            self._logger.error(f"Error in enhanced cascade delete for video {video_id}: {e}")
            raise HistoryServiceError(f"Failed to delete video: {e}")

    def enhanced_batch_delete_videos(self, video_ids: List[int], force: bool = False) -> List[CascadeDeleteResult]:
        """
        Delete multiple videos using enhanced cascade delete with validation and monitoring.
        
        Args:
            video_ids: List of database video IDs to delete
            force: Skip validation if True
            
        Returns:
            List of CascadeDeleteResult for each video
            
        Raises:
            HistoryServiceError: If batch deletion fails
        """
        try:
            session = self._get_session()
            manager = create_cascade_delete_manager(session)
            results = manager.batch_cascade_delete(video_ids, force=force)
            
            # Log batch summary
            successful = sum(1 for r in results if r.success)
            failed = len(results) - successful
            total_deleted = sum(r.total_deleted for r in results)
            
            self._logger.info(f"Enhanced batch delete completed: {successful} successful, "
                            f"{failed} failed, {total_deleted} total records deleted")
            
            return results
            
        except Exception as e:
            self._logger.error(f"Error in enhanced batch delete: {e}")
            raise HistoryServiceError(f"Failed to delete videos: {e}")

    def verify_cascade_delete_integrity(self, video_id: int) -> Dict[str, Any]:
        """
        Verify that a cascade delete operation completed successfully.
        
        Args:
            video_id: Video ID that was deleted
            
        Returns:
            Dictionary with integrity verification results
            
        Raises:
            HistoryServiceError: If verification fails
        """
        try:
            session = self._get_session()
            manager = create_cascade_delete_manager(session)
            
            integrity_result = manager.verify_cascade_integrity(video_id)
            
            if integrity_result.get('integrity_check_passed'):
                self._logger.info(f"Cascade delete integrity verified for video {video_id}")
            else:
                self._logger.warning(f"Cascade delete integrity issues for video {video_id}: {integrity_result}")
            
            return integrity_result
            
        except Exception as e:
            self._logger.error(f"Error verifying cascade delete integrity for video {video_id}: {e}")
            raise HistoryServiceError(f"Failed to verify integrity: {e}")

    def cleanup_orphaned_records(self, video_id: int) -> Dict[str, int]:
        """
        Clean up any orphaned records left after deletion.
        
        Args:
            video_id: Video ID to clean up orphans for
            
        Returns:
            Dictionary with counts of cleaned up records
            
        Raises:
            HistoryServiceError: If cleanup fails
        """
        try:
            session = self._get_session()
            manager = create_cascade_delete_manager(session)
            
            cleaned_counts = manager.cleanup_orphaned_records(video_id)
            
            if cleaned_counts:
                self._logger.info(f"Cleaned up orphaned records for video {video_id}: {cleaned_counts}")
            else:
                self._logger.info(f"No orphaned records found for video {video_id}")
            
            return cleaned_counts
            
        except Exception as e:
            self._logger.error(f"Error cleaning up orphaned records for video {video_id}: {e}")
            raise HistoryServiceError(f"Failed to cleanup orphaned records: {e}")

    def get_cascade_delete_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about cascade delete operations.
        
        Returns:
            Dictionary with cascade delete statistics
            
        Raises:
            HistoryServiceError: If statistics retrieval fails
        """
        try:
            session = self._get_session()
            manager = create_cascade_delete_manager(session)
            
            stats = manager.get_cascade_delete_statistics()
            
            self._logger.info("Retrieved cascade delete statistics")
            return stats
            
        except Exception as e:
            self._logger.error(f"Error getting cascade delete statistics: {e}")
            raise HistoryServiceError(f"Failed to get statistics: {e}")


# Dependency injection function for FastAPI
def get_history_service(session: Session = Depends(get_database_session_dependency)) -> HistoryService:
    """
    Get HistoryService instance for dependency injection.
    
    Args:
        session: Database session from FastAPI dependency
        
    Returns:
        HistoryService instance
    """
    return HistoryService(session)