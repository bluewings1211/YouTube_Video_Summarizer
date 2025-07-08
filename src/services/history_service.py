"""
History service for querying processed videos.

This service provides methods to query video processing history,
including pagination, filtering, and search capabilities.
"""

import logging
from datetime import datetime, date
from typing import Optional, Dict, Any, List, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload, joinedload
from sqlalchemy import select, func, and_, or_, text
from sqlalchemy.exc import SQLAlchemyError
from dataclasses import dataclass

from ..database.models import (
    Video, Transcript, Summary, Keyword, 
    TimestampedSegment, ProcessingMetadata
)
from ..database.exceptions import (
    DatabaseError, DatabaseConnectionError, DatabaseQueryError,
    classify_database_error
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

    def __init__(self, session: Optional[AsyncSession] = None):
        """
        Initialize the history service.
        
        Args:
            session: Optional database session. If not provided, will use dependency injection.
        """
        self._session = session
        self._logger = logging.getLogger(f"{__name__}.HistoryService")

    async def _get_session(self) -> AsyncSession:
        """Get database session (internal method)."""
        if self._session:
            return self._session
        else:
            # This should be used with dependency injection
            raise HistoryServiceError("No database session provided")

    async def get_videos_paginated(
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
            session = await self._get_session()
            
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
            count_result = await session.execute(count_query)
            total_items = count_result.scalar()
            
            # Calculate pagination
            total_pages = (total_items + page_size - 1) // page_size
            offset = (page - 1) * page_size
            
            # Apply pagination
            query = query.offset(offset).limit(page_size)
            
            # Execute query
            result = await session.execute(query)
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

    async def get_video_by_id(self, video_id: int) -> Optional[Video]:
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
            session = await self._get_session()
            
            # Query with all related data
            query = select(Video).options(
                selectinload(Video.transcripts),
                selectinload(Video.summaries),
                selectinload(Video.keywords),
                selectinload(Video.timestamped_segments),
                selectinload(Video.processing_metadata)
            ).where(Video.id == video_id)
            
            result = await session.execute(query)
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

    async def search_videos(
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
            session = await self._get_session()
            
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
            count_result = await session.execute(count_query)
            total_items = count_result.scalar()
            
            # Calculate pagination
            total_pages = (total_items + page_size - 1) // page_size
            offset = (page - 1) * page_size
            
            # Apply pagination
            search_query = search_query.offset(offset).limit(page_size)
            
            # Execute search
            result = await session.execute(search_query)
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

    async def filter_videos_by_date(
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
            session = await self._get_session()
            
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
            
            count_result = await session.execute(count_query)
            total_items = count_result.scalar()
            
            # Calculate pagination
            total_pages = (total_items + page_size - 1) // page_size
            offset = (page - 1) * page_size
            
            # Apply pagination
            query = query.offset(offset).limit(page_size)
            
            # Execute query
            result = await session.execute(query)
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

    async def get_video_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about processed videos.
        
        Returns:
            Dictionary containing video statistics
            
        Raises:
            HistoryServiceError: If query fails
        """
        try:
            session = await self._get_session()
            
            # Total videos
            total_videos_query = select(func.count(Video.id))
            total_videos_result = await session.execute(total_videos_query)
            total_videos = total_videos_result.scalar()
            
            # Videos with transcripts
            videos_with_transcripts_query = select(func.count(Video.id.distinct())).where(
                Video.transcripts.any()
            )
            videos_with_transcripts_result = await session.execute(videos_with_transcripts_query)
            videos_with_transcripts = videos_with_transcripts_result.scalar()
            
            # Videos with summaries
            videos_with_summaries_query = select(func.count(Video.id.distinct())).where(
                Video.summaries.any()
            )
            videos_with_summaries_result = await session.execute(videos_with_summaries_query)
            videos_with_summaries = videos_with_summaries_result.scalar()
            
            # Processing status counts
            status_counts_query = select(
                ProcessingMetadata.status,
                func.count(ProcessingMetadata.status)
            ).group_by(ProcessingMetadata.status)
            status_counts_result = await session.execute(status_counts_query)
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


# Dependency injection function for FastAPI
def get_history_service(session: AsyncSession) -> HistoryService:
    """
    Get HistoryService instance for dependency injection.
    
    Args:
        session: Database session from FastAPI dependency
        
    Returns:
        HistoryService instance
    """
    return HistoryService(session)