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
from ..services.reprocessing_service import (
    ReprocessingService, get_reprocessing_service, ReprocessingServiceError,
    ReprocessingRequest, ReprocessingResult, ReprocessingValidation, 
    ReprocessingMode, ReprocessingStatus
)
from ..database.models import Video, Transcript, Summary, Keyword, TimestampedSegment, ProcessingMetadata
from ..database.cascade_delete import CascadeDeleteResult, CascadeDeleteValidation
from ..database.transaction_manager import TransactionResult

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


# Reprocessing models
class ReprocessingModeEnum(str, Enum):
    """Reprocessing mode enumeration."""
    FULL = "full"
    TRANSCRIPT_ONLY = "transcript_only"
    SUMMARY_ONLY = "summary_only"
    KEYWORDS_ONLY = "keywords_only"
    SEGMENTS_ONLY = "segments_only"
    INCREMENTAL = "incremental"


class ReprocessingStatusEnum(str, Enum):
    """Reprocessing status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ReprocessingRequestModel(BaseModel):
    """Request model for video reprocessing."""
    mode: ReprocessingModeEnum = Field(ReprocessingModeEnum.FULL, description="Reprocessing mode")
    force: bool = Field(False, description="Force reprocessing even if validation fails")
    clear_cache: bool = Field(True, description="Clear cached data before reprocessing")
    preserve_metadata: bool = Field(True, description="Preserve existing processing metadata")
    requested_by: Optional[str] = Field(None, description="User requesting reprocessing")
    workflow_params: Optional[Dict[str, Any]] = Field(None, description="Additional workflow parameters")


class ReprocessingValidationResponse(BaseModel):
    """Response model for reprocessing validation."""
    can_reprocess: bool
    video_exists: bool
    current_status: Optional[str] = None
    existing_components: Dict[str, int]
    potential_issues: List[str]
    recommendations: List[str]

    class Config:
        from_attributes = True


class ReprocessingResultResponse(BaseModel):
    """Response model for reprocessing result."""
    success: bool
    video_id: int
    mode: ReprocessingModeEnum
    status: ReprocessingStatusEnum
    message: str
    cleared_components: List[str]
    processing_metadata_id: Optional[int] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    execution_time_seconds: Optional[float] = None
    error_details: Optional[Dict[str, Any]] = None

    class Config:
        from_attributes = True


class ReprocessingStatusResponse(BaseModel):
    """Response model for reprocessing status."""
    video_id: int
    video_title: str
    video_youtube_id: str
    processing_metadata_id: int
    status: str
    workflow_params: Optional[Dict[str, Any]] = None
    error_info: Optional[str] = None
    created_at: datetime
    is_reprocessing: bool

    class Config:
        from_attributes = True


class ReprocessingHistoryResponse(BaseModel):
    """Response model for reprocessing history."""
    history: List[Dict[str, Any]]
    total_count: int


class CancelReprocessingRequest(BaseModel):
    """Request model for cancelling reprocessing."""
    reason: str = Field("User requested", description="Reason for cancellation")


class TransactionResultResponse(BaseModel):
    """Response model for transaction results."""
    success: bool
    transaction_id: str
    status: str
    operations: List[Dict[str, Any]]
    savepoints: List[Dict[str, Any]]
    start_time: datetime
    end_time: Optional[datetime] = None
    execution_time_seconds: Optional[float] = None
    error_message: Optional[str] = None
    rollback_reason: Optional[str] = None

    class Config:
        from_attributes = True


class TransactionalDeleteRequest(BaseModel):
    """Request model for transactional deletion."""
    create_savepoints: bool = Field(True, description="Create savepoints before critical operations")
    audit_user: Optional[str] = Field(None, description="User performing the deletion")


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


# Video reprocessing endpoints
@router.get("/videos/{video_id}/validate-reprocessing", response_model=ReprocessingValidationResponse)
async def validate_video_reprocessing(
    video_id: int,
    mode: ReprocessingModeEnum = Query(ReprocessingModeEnum.FULL, description="Reprocessing mode"),
    force: bool = Query(False, description="Force reprocessing even if validation fails"),
    reprocessing_service: ReprocessingService = Depends(get_reprocessing_service)
):
    """
    Validate that a video can be reprocessed.
    
    Checks for potential issues like active processing tasks,
    existing components, and provides recommendations.
    """
    try:
        request = ReprocessingRequest(
            video_id=video_id,
            mode=ReprocessingMode(mode.value),
            force=force
        )
        
        validation = reprocessing_service.validate_reprocessing_request(request)
        
        return ReprocessingValidationResponse(
            can_reprocess=validation.can_reprocess,
            video_exists=validation.video_exists,
            current_status=validation.current_status,
            existing_components=validation.existing_components,
            potential_issues=validation.potential_issues,
            recommendations=validation.recommendations
        )
        
    except ReprocessingServiceError as e:
        logger.error(f"Reprocessing service error validating: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error validating reprocessing: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/videos/{video_id}/reprocess", response_model=ReprocessingResultResponse)
async def reprocess_video(
    video_id: int,
    reprocessing_request: ReprocessingRequestModel,
    reprocessing_service: ReprocessingService = Depends(get_reprocessing_service)
):
    """
    Initiate reprocessing for a video.
    
    Triggers the reprocessing workflow with the specified mode,
    optionally clearing cached data and preserving metadata.
    """
    try:
        request = ReprocessingRequest(
            video_id=video_id,
            mode=ReprocessingMode(reprocessing_request.mode.value),
            force=reprocessing_request.force,
            clear_cache=reprocessing_request.clear_cache,
            preserve_metadata=reprocessing_request.preserve_metadata,
            requested_by=reprocessing_request.requested_by,
            workflow_params=reprocessing_request.workflow_params
        )
        
        result = reprocessing_service.initiate_reprocessing(request)
        
        if not result.success:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to initiate reprocessing: {result.message}"
            )
        
        # Log the reprocessing initiation
        logger.info(f"Video {video_id} reprocessing initiated with mode {reprocessing_request.mode.value} by {reprocessing_request.requested_by or 'unknown'}")
        
        return ReprocessingResultResponse(
            success=result.success,
            video_id=result.video_id,
            mode=ReprocessingModeEnum(result.mode.value),
            status=ReprocessingStatusEnum(result.status.value),
            message=result.message,
            cleared_components=result.cleared_components,
            processing_metadata_id=result.processing_metadata_id,
            start_time=result.start_time,
            end_time=result.end_time,
            execution_time_seconds=result.execution_time_seconds,
            error_details=result.error_details
        )
        
    except HTTPException:
        raise
    except ReprocessingServiceError as e:
        logger.error(f"Reprocessing service error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error reprocessing video: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/videos/{video_id}/reprocessing-status", response_model=ReprocessingStatusResponse)
async def get_video_reprocessing_status(
    video_id: int,
    reprocessing_service: ReprocessingService = Depends(get_reprocessing_service)
):
    """
    Get the current reprocessing status for a video.
    
    Returns the current processing status, including whether
    it's actively being reprocessed and any error information.
    """
    try:
        status = reprocessing_service.get_reprocessing_status(video_id)
        
        if not status:
            raise HTTPException(
                status_code=404,
                detail=f"No processing status found for video {video_id}"
            )
        
        return ReprocessingStatusResponse(
            video_id=status['video_id'],
            video_title=status['video_title'],
            video_youtube_id=status['video_youtube_id'],
            processing_metadata_id=status['processing_metadata_id'],
            status=status['status'],
            workflow_params=status['workflow_params'],
            error_info=status['error_info'],
            created_at=status['created_at'],
            is_reprocessing=status['is_reprocessing']
        )
        
    except HTTPException:
        raise
    except ReprocessingServiceError as e:
        logger.error(f"Reprocessing service error getting status: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error getting reprocessing status: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/videos/{video_id}/cancel-reprocessing")
async def cancel_video_reprocessing(
    video_id: int,
    cancel_request: CancelReprocessingRequest,
    reprocessing_service: ReprocessingService = Depends(get_reprocessing_service)
):
    """
    Cancel an active reprocessing operation for a video.
    
    Stops the reprocessing workflow and marks the operation
    as cancelled with the provided reason.
    """
    try:
        success = reprocessing_service.cancel_reprocessing(video_id, cancel_request.reason)
        
        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"No active reprocessing found for video {video_id}"
            )
        
        logger.info(f"Video {video_id} reprocessing cancelled: {cancel_request.reason}")
        
        return {
            "success": True,
            "video_id": video_id,
            "message": f"Reprocessing cancelled: {cancel_request.reason}"
        }
        
    except HTTPException:
        raise
    except ReprocessingServiceError as e:
        logger.error(f"Reprocessing service error cancelling: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error cancelling reprocessing: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/videos/{video_id}/reprocessing-history", response_model=ReprocessingHistoryResponse)
async def get_video_reprocessing_history(
    video_id: int,
    limit: int = Query(10, ge=1, le=100, description="Maximum number of records to return"),
    reprocessing_service: ReprocessingService = Depends(get_reprocessing_service)
):
    """
    Get reprocessing history for a video.
    
    Returns a list of all reprocessing operations performed
    on the video, including their status and details.
    """
    try:
        history = reprocessing_service.get_reprocessing_history(video_id, limit)
        
        return ReprocessingHistoryResponse(
            history=history,
            total_count=len(history)
        )
        
    except ReprocessingServiceError as e:
        logger.error(f"Reprocessing service error getting history: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error getting reprocessing history: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/videos/{video_id}/clear-cache")
async def clear_video_cache(
    video_id: int,
    mode: ReprocessingModeEnum = Query(ReprocessingModeEnum.FULL, description="Cache clearing mode"),
    preserve_metadata: bool = Query(True, description="Preserve processing metadata"),
    reprocessing_service: ReprocessingService = Depends(get_reprocessing_service)
):
    """
    Clear cached data for a video.
    
    Removes cached processing results based on the specified mode
    without triggering reprocessing.
    """
    try:
        cleared_components = reprocessing_service.clear_video_cache(
            video_id=video_id,
            mode=ReprocessingMode(mode.value),
            preserve_metadata=preserve_metadata
        )
        
        logger.info(f"Cache cleared for video {video_id} with mode {mode.value}")
        
        return {
            "success": True,
            "video_id": video_id,
            "mode": mode.value,
            "cleared_components": cleared_components,
            "message": f"Cache cleared successfully. Components cleared: {', '.join(cleared_components) if cleared_components else 'none'}"
        }
        
    except ReprocessingServiceError as e:
        logger.error(f"Reprocessing service error clearing cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error clearing cache: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Enhanced transaction-based deletion endpoints
@router.delete("/videos/{video_id}/transactional", response_model=TransactionResultResponse)
async def transactional_delete_video(
    video_id: int,
    delete_request: TransactionalDeleteRequest,
    history_service: HistoryService = Depends(get_history_service)
):
    """
    Delete a video using enhanced transaction management with rollback capabilities.
    
    This endpoint provides detailed transaction tracking, savepoint management,
    and comprehensive rollback capabilities for video deletion operations.
    """
    try:
        result = history_service.transactional_delete_video_by_id(
            video_id=video_id,
            create_savepoints=delete_request.create_savepoints
        )
        
        # Log the transactional deletion
        logger.info(f"Transactional deletion of video {video_id} completed by {delete_request.audit_user or 'unknown'}")
        
        return TransactionResultResponse(
            success=result.success,
            transaction_id=result.transaction_id,
            status=result.status.value,
            operations=[{
                "id": op.id,
                "type": op.operation_type.value,
                "description": op.description,
                "target_table": op.target_table,
                "target_id": op.target_id,
                "success": op.success,
                "affected_rows": op.affected_rows,
                "executed_at": op.executed_at,
                "error_message": op.error_message
            } for op in result.operations],
            savepoints=[{
                "name": sp.name,
                "created_at": sp.created_at,
                "operations_count": sp.operations_count,
                "description": sp.description
            } for sp in result.savepoints],
            start_time=result.start_time,
            end_time=result.end_time,
            execution_time_seconds=result.execution_time_seconds,
            error_message=result.error_message,
            rollback_reason=result.rollback_reason
        )
        
    except HistoryServiceError as e:
        logger.error(f"History service error in transactional deletion: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in transactional deletion: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/videos/batch-delete-transactional", response_model=TransactionResultResponse)
async def transactional_batch_delete_videos(
    delete_request: Dict[str, Any],
    history_service: HistoryService = Depends(get_history_service)
):
    """
    Delete multiple videos using enhanced transaction management with rollback capabilities.
    
    This endpoint provides batch deletion with individual video savepoints,
    allowing partial rollback if individual deletions fail.
    """
    try:
        # Validate request
        if "video_ids" not in delete_request:
            raise HTTPException(
                status_code=400,
                detail="video_ids field is required"
            )
        
        video_ids = delete_request["video_ids"]
        create_savepoints = delete_request.get("create_savepoints", True)
        audit_user = delete_request.get("audit_user")
        
        if not video_ids:
            raise HTTPException(
                status_code=400,
                detail="No video IDs provided for batch deletion"
            )
        
        if len(video_ids) > 50:  # Lower limit for transactional batch to avoid timeout
            raise HTTPException(
                status_code=400,
                detail="Transactional batch deletion limited to 50 videos at a time"
            )
        
        result = history_service.transactional_batch_delete_videos(
            video_ids=video_ids,
            create_savepoints=create_savepoints
        )
        
        # Log the transactional batch deletion
        logger.info(f"Transactional batch deletion of {len(video_ids)} videos completed by {audit_user or 'unknown'}")
        
        return TransactionResultResponse(
            success=result.success,
            transaction_id=result.transaction_id,
            status=result.status.value,
            operations=[{
                "id": op.id,
                "type": op.operation_type.value,
                "description": op.description,
                "target_table": op.target_table,
                "target_id": op.target_id,
                "target_ids": op.target_ids,
                "success": op.success,
                "affected_rows": op.affected_rows,
                "executed_at": op.executed_at,
                "error_message": op.error_message,
                "parameters": op.parameters
            } for op in result.operations],
            savepoints=[{
                "name": sp.name,
                "created_at": sp.created_at,
                "operations_count": sp.operations_count,
                "description": sp.description
            } for sp in result.savepoints],
            start_time=result.start_time,
            end_time=result.end_time,
            execution_time_seconds=result.execution_time_seconds,
            error_message=result.error_message,
            rollback_reason=result.rollback_reason
        )
        
    except HTTPException:
        raise
    except HistoryServiceError as e:
        logger.error(f"History service error in transactional batch deletion: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in transactional batch deletion: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/videos/{video_id}/test-rollback", response_model=TransactionResultResponse)
async def test_transaction_rollback(
    video_id: int,
    history_service: HistoryService = Depends(get_history_service)
):
    """
    Test transaction rollback functionality (for testing purposes).
    
    This endpoint demonstrates the rollback capabilities by making a change
    and then rolling it back to verify the rollback mechanism works correctly.
    """
    try:
        result = history_service.test_transaction_rollback(video_id)
        
        logger.info(f"Transaction rollback test completed for video {video_id}")
        
        return TransactionResultResponse(
            success=result.success,
            transaction_id=result.transaction_id,
            status=result.status.value,
            operations=[{
                "id": op.id,
                "type": op.operation_type.value,
                "description": op.description,
                "target_table": op.target_table,
                "target_id": op.target_id,
                "success": op.success,
                "executed_at": op.executed_at,
                "error_message": op.error_message,
                "parameters": op.parameters
            } for op in result.operations],
            savepoints=[{
                "name": sp.name,
                "created_at": sp.created_at,
                "operations_count": sp.operations_count,
                "description": sp.description
            } for sp in result.savepoints],
            start_time=result.start_time,
            end_time=result.end_time,
            execution_time_seconds=result.execution_time_seconds,
            error_message=result.error_message,
            rollback_reason=result.rollback_reason
        )
        
    except HTTPException:
        raise
    except HistoryServiceError as e:
        logger.error(f"History service error in rollback test: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in rollback test: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Error handlers would be added to the main app, not the router
# These are handled in the endpoints with try/catch blocks instead