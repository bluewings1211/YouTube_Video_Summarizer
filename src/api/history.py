"""
History API endpoints for querying processed videos.

Provides REST endpoints for retrieving video processing history
with pagination, filtering, and search capabilities.
"""

import logging
from datetime import datetime, date
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field, validator
from sqlalchemy.orm import Session

from ..database.connection import get_database_session_dependency
from ..services.history_service import HistoryService, get_history_service, HistoryServiceError
from ..database.models import Video, Transcript, Summary, Keyword, TimestampedSegment, ProcessingMetadata
from ..database.cascade_delete import CascadeDeleteResult, CascadeDeleteValidation

logger = logging.getLogger(__name__)

# Create FastAPI router
router = APIRouter(prefix="/api/v1/history", tags=["history"])


# Pydantic models for API responses
class VideoHistoryResponse(BaseModel):
    """Response model for video history items."""
    id: int
    video_id: str
    title: str
    duration: Optional[int] = None
    url: str
    created_at: datetime
    updated_at: datetime
    processing_status: Optional[str] = None
    has_transcript: bool = False
    has_summary: bool = False
    has_keywords: bool = False
    has_segments: bool = False

    class Config:
        from_attributes = True


class PaginationResponse(BaseModel):
    """Response model for pagination metadata."""
    page: int
    page_size: int
    total_items: int
    total_pages: int
    has_next: bool
    has_previous: bool

    class Config:
        from_attributes = True


class VideoListResponse(BaseModel):
    """Response model for paginated video list."""
    videos: List[VideoHistoryResponse]
    pagination: PaginationResponse


class TranscriptResponse(BaseModel):
    """Response model for video transcript."""
    id: int
    content: str
    language: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


class SummaryResponse(BaseModel):
    """Response model for video summary."""
    id: int
    content: str
    processing_time: Optional[float] = None
    created_at: datetime

    class Config:
        from_attributes = True


class KeywordResponse(BaseModel):
    """Response model for video keywords."""
    id: int
    keywords_json: Dict[str, Any]
    created_at: datetime

    class Config:
        from_attributes = True


class TimestampedSegmentResponse(BaseModel):
    """Response model for timestamped segments."""
    id: int
    segments_json: Dict[str, Any]
    created_at: datetime

    class Config:
        from_attributes = True


class ProcessingMetadataResponse(BaseModel):
    """Response model for processing metadata."""
    id: int
    workflow_params: Optional[Dict[str, Any]] = None
    status: str
    error_info: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


class VideoDetailResponse(BaseModel):
    """Response model for detailed video information."""
    id: int
    video_id: str
    title: str
    duration: Optional[int] = None
    url: str
    created_at: datetime
    updated_at: datetime
    transcripts: List[TranscriptResponse] = []
    summaries: List[SummaryResponse] = []
    keywords: List[KeywordResponse] = []
    timestamped_segments: List[TimestampedSegmentResponse] = []
    processing_metadata: List[ProcessingMetadataResponse] = []

    class Config:
        from_attributes = True


class VideoStatisticsResponse(BaseModel):
    """Response model for video statistics."""
    total_videos: int
    videos_with_transcripts: int
    videos_with_summaries: int
    videos_with_keywords: int
    videos_with_segments: int
    processing_status_counts: Dict[str, int]
    completion_rate: float


class ErrorResponse(BaseModel):
    """Response model for API errors."""
    error: str
    message: str
    details: Optional[Dict[str, Any]] = None


class VideoDeletionInfoResponse(BaseModel):
    """Response model for video deletion information."""
    video: Dict[str, Any]
    related_data_counts: Dict[str, int]
    total_related_records: int

    class Config:
        from_attributes = True


class CascadeDeleteValidationResponse(BaseModel):
    """Response model for cascade delete validation."""
    can_delete: bool
    video_exists: bool
    related_counts: Dict[str, int]
    potential_issues: List[str]
    total_related_records: int

    class Config:
        from_attributes = True


class CascadeDeleteResultResponse(BaseModel):
    """Response model for cascade delete result."""
    success: bool
    video_id: int
    deleted_counts: Dict[str, int]
    total_deleted: int
    error_message: Optional[str] = None
    execution_time_ms: Optional[float] = None

    class Config:
        from_attributes = True


class BatchDeleteResponse(BaseModel):
    """Response model for batch delete operations."""
    results: List[CascadeDeleteResultResponse]
    summary: Dict[str, Any]


class DeleteVideoRequest(BaseModel):
    """Request model for video deletion."""
    force: bool = Field(False, description="Skip validation and force deletion")
    audit_user: Optional[str] = Field(None, description="User performing the deletion")


class BatchDeleteVideoRequest(BaseModel):
    """Request model for batch video deletion."""
    video_ids: List[int] = Field(..., description="List of video IDs to delete")
    force: bool = Field(False, description="Skip validation and force deletion")
    audit_user: Optional[str] = Field(None, description="User performing the deletion")


class IntegrityCheckResponse(BaseModel):
    """Response model for deletion integrity check."""
    video_exists: bool
    has_orphaned_records: bool
    orphaned_records: Dict[str, int]
    integrity_check_passed: bool
    error: Optional[str] = None


class CascadeDeleteStatisticsResponse(BaseModel):
    """Response model for cascade delete statistics."""
    total_videos: int
    average_related_records: Dict[str, Dict[str, Any]]
    videos_with_most_related: List[Dict[str, Any]]


# Query parameter models
class VideoListParams(BaseModel):
    """Query parameters for video list endpoint."""
    page: int = Field(1, ge=1, description="Page number (1-based)")
    page_size: int = Field(20, ge=1, le=100, description="Number of items per page")
    sort_by: str = Field("created_at", pattern="^(created_at|updated_at|title)$", description="Sort field")
    sort_order: str = Field("desc", pattern="^(asc|desc)$", description="Sort order")
    
    @validator('sort_by')
    def validate_sort_by(cls, v):
        allowed_fields = ['created_at', 'updated_at', 'title']
        if v not in allowed_fields:
            raise ValueError(f"sort_by must be one of: {', '.join(allowed_fields)}")
        return v


class VideoFilterParams(BaseModel):
    """Query parameters for video filtering."""
    page: int = Field(1, ge=1, description="Page number (1-based)")
    page_size: int = Field(20, ge=1, le=100, description="Number of items per page")
    date_from: Optional[date] = Field(None, description="Start date (YYYY-MM-DD)")
    date_to: Optional[date] = Field(None, description="End date (YYYY-MM-DD)")
    keywords: Optional[str] = Field(None, description="Keywords to search for")
    title_search: Optional[str] = Field(None, description="Search in video titles")
    
    @validator('date_to')
    def validate_date_range(cls, v, values):
        if v and 'date_from' in values and values['date_from'] and v < values['date_from']:
            raise ValueError("date_to must be after date_from")
        return v


# API endpoints
@router.get("/videos", response_model=VideoListResponse)
async def get_videos(
    page: int = Query(1, ge=1, description="Page number (1-based)"),
    page_size: int = Query(20, ge=1, le=100, description="Number of items per page"),
    sort_by: str = Query("created_at", pattern="^(created_at|updated_at|title)$", description="Sort field"),
    sort_order: str = Query("desc", pattern="^(asc|desc)$", description="Sort order"),
    date_from: Optional[date] = Query(None, description="Start date (YYYY-MM-DD)"),
    date_to: Optional[date] = Query(None, description="End date (YYYY-MM-DD)"),
    keywords: Optional[str] = Query(None, description="Keywords to search for"),
    title_search: Optional[str] = Query(None, description="Search in video titles"),
    history_service: HistoryService = Depends(get_history_service)
):
    """
    Get paginated list of processed videos with optional filtering.
    
    This endpoint supports:
    - Pagination with configurable page size
    - Sorting by creation date, update date, or title
    - Date range filtering
    - Keyword search across video content
    - Title search
    """
    try:
        # Validate date range
        if date_from and date_to and date_to < date_from:
            raise HTTPException(
                status_code=400,
                detail="date_to must be after date_from"
            )
        
        # Determine which service method to use based on filters
        if keywords or title_search:
            # Use search functionality
            search_query = ""
            if keywords:
                search_query += keywords
            if title_search:
                if search_query:
                    search_query += " " + title_search
                else:
                    search_query = title_search
            
            video_items, pagination_info = history_service.search_videos(
                query=search_query,
                page=page,
                page_size=page_size
            )
        elif date_from or date_to:
            # Use date filtering
            video_items, pagination_info = history_service.filter_videos_by_date(
                date_from=date_from,
                date_to=date_to,
                page=page,
                page_size=page_size
            )
        else:
            # Use regular pagination
            video_items, pagination_info = history_service.get_videos_paginated(
                page=page,
                page_size=page_size,
                sort_by=sort_by,
                sort_order=sort_order
            )
        
        # Convert to response models
        video_responses = [
            VideoHistoryResponse(
                id=item.id,
                video_id=item.video_id,
                title=item.title,
                duration=item.duration,
                url=item.url,
                created_at=item.created_at,
                updated_at=item.updated_at,
                processing_status=item.processing_status,
                has_transcript=item.has_transcript,
                has_summary=item.has_summary,
                has_keywords=item.has_keywords,
                has_segments=item.has_segments
            )
            for item in video_items
        ]
        
        pagination_response = PaginationResponse(
            page=pagination_info.page,
            page_size=pagination_info.page_size,
            total_items=pagination_info.total_items,
            total_pages=pagination_info.total_pages,
            has_next=pagination_info.has_next,
            has_previous=pagination_info.has_previous
        )
        
        return VideoListResponse(
            videos=video_responses,
            pagination=pagination_response
        )
        
    except HistoryServiceError as e:
        logger.error(f"History service error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in get_videos: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/videos/{video_id}", response_model=VideoDetailResponse)
async def get_video_detail(
    video_id: int,
    history_service: HistoryService = Depends(get_history_service)
):
    """
    Get detailed information for a specific video.
    
    Returns complete video information including:
    - Basic video metadata
    - All transcripts
    - All summaries
    - All keywords
    - All timestamped segments
    - All processing metadata
    """
    try:
        # Get video with all related data
        video = history_service.get_video_by_id(video_id)
        
        if not video:
            raise HTTPException(
                status_code=404,
                detail=f"Video with ID {video_id} not found"
            )
        
        # Convert to response model
        video_response = VideoDetailResponse(
            id=video.id,
            video_id=video.video_id,
            title=video.title,
            duration=video.duration,
            url=video.url,
            created_at=video.created_at,
            updated_at=video.updated_at,
            transcripts=[
                TranscriptResponse(
                    id=transcript.id,
                    content=transcript.content,
                    language=transcript.language,
                    created_at=transcript.created_at
                )
                for transcript in video.transcripts
            ],
            summaries=[
                SummaryResponse(
                    id=summary.id,
                    content=summary.content,
                    processing_time=summary.processing_time,
                    created_at=summary.created_at
                )
                for summary in video.summaries
            ],
            keywords=[
                KeywordResponse(
                    id=keyword.id,
                    keywords_json=keyword.keywords_json,
                    created_at=keyword.created_at
                )
                for keyword in video.keywords
            ],
            timestamped_segments=[
                TimestampedSegmentResponse(
                    id=segment.id,
                    segments_json=segment.segments_json,
                    created_at=segment.created_at
                )
                for segment in video.timestamped_segments
            ],
            processing_metadata=[
                ProcessingMetadataResponse(
                    id=metadata.id,
                    workflow_params=metadata.workflow_params,
                    status=metadata.status,
                    error_info=metadata.error_info,
                    created_at=metadata.created_at
                )
                for metadata in video.processing_metadata
            ]
        )
        
        return video_response
        
    except HistoryServiceError as e:
        logger.error(f"History service error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in get_video_detail: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/statistics", response_model=VideoStatisticsResponse)
async def get_video_statistics(
    history_service: HistoryService = Depends(get_history_service)
):
    """
    Get statistics about processed videos.
    
    Returns:
    - Total number of videos
    - Number of videos with transcripts
    - Number of videos with summaries
    - Processing status counts
    - Completion rate
    """
    try:
        statistics = history_service.get_video_statistics()
        
        return VideoStatisticsResponse(
            total_videos=statistics["total_videos"],
            videos_with_transcripts=statistics["videos_with_transcripts"],
            videos_with_summaries=statistics["videos_with_summaries"],
            videos_with_keywords=statistics.get("videos_with_keywords", 0),
            videos_with_segments=statistics.get("videos_with_segments", 0),
            processing_status_counts=statistics["processing_status_counts"],
            completion_rate=statistics["completion_rate"]
        )
        
    except HistoryServiceError as e:
        logger.error(f"History service error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in get_video_statistics: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Health check endpoint
@router.get("/health")
async def health_check():
    """
    Health check endpoint for the history API.
    """
    return {"status": "healthy", "service": "history_api"}


# Video deletion endpoints
@router.get("/videos/{video_id}/deletion-info", response_model=VideoDeletionInfoResponse)
async def get_video_deletion_info(
    video_id: int,
    history_service: HistoryService = Depends(get_history_service)
):
    """
    Get information about what will be deleted when deleting a video.
    
    Returns detailed information about the video and all related records
    that will be affected by the deletion operation.
    """
    try:
        deletion_info = history_service.get_video_deletion_info(video_id)
        
        if not deletion_info:
            raise HTTPException(
                status_code=404,
                detail=f"Video with ID {video_id} not found"
            )
        
        return VideoDeletionInfoResponse(**deletion_info)
        
    except HistoryServiceError as e:
        logger.error(f"History service error getting deletion info: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error getting deletion info: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/videos/{video_id}/validate-deletion", response_model=CascadeDeleteValidationResponse)
async def validate_video_deletion(
    video_id: int,
    history_service: HistoryService = Depends(get_history_service)
):
    """
    Validate that a video can be safely deleted.
    
    Checks for potential issues like active processing tasks,
    large numbers of related records, or other constraints.
    """
    try:
        validation = history_service.validate_video_deletion(video_id)
        
        return CascadeDeleteValidationResponse(
            can_delete=validation.can_delete,
            video_exists=validation.video_exists,
            related_counts=validation.related_counts,
            potential_issues=validation.potential_issues,
            total_related_records=validation.total_related_records
        )
        
    except HistoryServiceError as e:
        logger.error(f"History service error validating deletion: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error validating deletion: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/videos/{video_id}", response_model=CascadeDeleteResultResponse)
async def delete_video(
    video_id: int,
    delete_request: DeleteVideoRequest,
    history_service: HistoryService = Depends(get_history_service)
):
    """
    Delete a video and all its related data.
    
    Performs a cascade delete operation that removes the video
    and all associated transcripts, summaries, keywords, segments,
    and processing metadata.
    """
    try:
        # Use enhanced cascade delete
        result = history_service.enhanced_delete_video_by_id(
            video_id=video_id,
            force=delete_request.force
        )
        
        if not result.success:
            # If deletion failed, return 400 with error details
            raise HTTPException(
                status_code=400,
                detail=f"Failed to delete video: {result.error_message}"
            )
        
        # Log the deletion with audit information
        logger.info(f"Video {video_id} deleted successfully by {delete_request.audit_user or 'unknown'}")
        
        return CascadeDeleteResultResponse(
            success=result.success,
            video_id=result.video_id,
            deleted_counts=result.deleted_counts,
            total_deleted=result.total_deleted,
            error_message=result.error_message,
            execution_time_ms=result.execution_time_ms
        )
        
    except HTTPException:
        raise
    except HistoryServiceError as e:
        logger.error(f"History service error deleting video: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error deleting video: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/videos/{video_id}/by-youtube-id", response_model=CascadeDeleteResultResponse)
async def delete_video_by_youtube_id(
    video_id: str,
    delete_request: DeleteVideoRequest,
    history_service: HistoryService = Depends(get_history_service)
):
    """
    Delete a video by its YouTube video ID.
    
    Finds the video by YouTube ID and performs cascade deletion.
    """
    try:
        # First find the video by YouTube ID
        if not history_service.delete_video_by_video_id(video_id):
            raise HTTPException(
                status_code=404,
                detail=f"Video with YouTube ID '{video_id}' not found"
            )
        
        logger.info(f"Video with YouTube ID '{video_id}' deleted successfully by {delete_request.audit_user or 'unknown'}")
        
        # Since delete_video_by_video_id returns only boolean, we create a simplified response
        return CascadeDeleteResultResponse(
            success=True,
            video_id=0,  # We don't have the database ID in this context
            deleted_counts={},
            total_deleted=1,
            error_message=None,
            execution_time_ms=None
        )
        
    except HTTPException:
        raise
    except HistoryServiceError as e:
        logger.error(f"History service error deleting video by YouTube ID: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error deleting video by YouTube ID: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/videos/batch-delete", response_model=BatchDeleteResponse)
async def batch_delete_videos(
    delete_request: BatchDeleteVideoRequest,
    history_service: HistoryService = Depends(get_history_service)
):
    """
    Delete multiple videos in a batch operation.
    
    Performs cascade deletion for multiple videos with detailed
    results for each video.
    """
    try:
        if not delete_request.video_ids:
            raise HTTPException(
                status_code=400,
                detail="No video IDs provided for batch deletion"
            )
        
        if len(delete_request.video_ids) > 100:
            raise HTTPException(
                status_code=400,
                detail="Batch deletion limited to 100 videos at a time"
            )
        
        # Perform batch deletion
        results = history_service.enhanced_batch_delete_videos(
            video_ids=delete_request.video_ids,
            force=delete_request.force
        )
        
        # Convert results to response format
        response_results = [
            CascadeDeleteResultResponse(
                success=result.success,
                video_id=result.video_id,
                deleted_counts=result.deleted_counts,
                total_deleted=result.total_deleted,
                error_message=result.error_message,
                execution_time_ms=result.execution_time_ms
            )
            for result in results
        ]
        
        # Calculate summary
        successful_count = sum(1 for r in results if r.success)
        failed_count = len(results) - successful_count
        total_deleted = sum(r.total_deleted for r in results)
        total_time = sum(r.execution_time_ms or 0 for r in results)
        
        summary = {
            "total_videos": len(delete_request.video_ids),
            "successful_deletions": successful_count,
            "failed_deletions": failed_count,
            "total_records_deleted": total_deleted,
            "total_execution_time_ms": total_time,
            "success_rate": (successful_count / len(results)) * 100 if results else 0
        }
        
        logger.info(f"Batch deletion completed: {successful_count} successful, {failed_count} failed by {delete_request.audit_user or 'unknown'}")
        
        return BatchDeleteResponse(
            results=response_results,
            summary=summary
        )
        
    except HTTPException:
        raise
    except HistoryServiceError as e:
        logger.error(f"History service error in batch deletion: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in batch deletion: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/videos/{video_id}/integrity-check", response_model=IntegrityCheckResponse)
async def check_deletion_integrity(
    video_id: int,
    history_service: HistoryService = Depends(get_history_service)
):
    """
    Check the integrity of a video deletion operation.
    
    Verifies that a video and all related records have been
    properly deleted with no orphaned records remaining.
    """
    try:
        integrity_result = history_service.verify_cascade_delete_integrity(video_id)
        
        return IntegrityCheckResponse(
            video_exists=integrity_result.get('video_exists', False),
            has_orphaned_records=integrity_result.get('has_orphaned_records', False),
            orphaned_records=integrity_result.get('orphaned_records', {}),
            integrity_check_passed=integrity_result.get('integrity_check_passed', False),
            error=integrity_result.get('error')
        )
        
    except HistoryServiceError as e:
        logger.error(f"History service error checking integrity: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error checking integrity: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/videos/{video_id}/cleanup-orphans")
async def cleanup_orphaned_records(
    video_id: int,
    history_service: HistoryService = Depends(get_history_service)
):
    """
    Clean up any orphaned records for a deleted video.
    
    Removes any related records that may have been left behind
    during a deletion operation.
    """
    try:
        cleaned_counts = history_service.cleanup_orphaned_records(video_id)
        
        return {
            "video_id": video_id,
            "cleaned_records": cleaned_counts,
            "total_cleaned": sum(cleaned_counts.values()),
            "message": "Orphaned records cleaned successfully" if cleaned_counts else "No orphaned records found"
        }
        
    except HistoryServiceError as e:
        logger.error(f"History service error cleaning orphans: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error cleaning orphans: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/deletion-statistics", response_model=CascadeDeleteStatisticsResponse)
async def get_cascade_delete_statistics(
    history_service: HistoryService = Depends(get_history_service)
):
    """
    Get statistics about cascade delete operations.
    
    Returns information about video deletion patterns,
    average related record counts, and videos with most related data.
    """
    try:
        stats = history_service.get_cascade_delete_statistics()
        
        return CascadeDeleteStatisticsResponse(
            total_videos=stats.get('total_videos', 0),
            average_related_records=stats.get('average_related_records', {}),
            videos_with_most_related=stats.get('videos_with_most_related', [])
        )
        
    except HistoryServiceError as e:
        logger.error(f"History service error getting statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error getting statistics: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Error handlers would be added to the main app, not the router
# These are handled in the endpoints with try/catch blocks instead