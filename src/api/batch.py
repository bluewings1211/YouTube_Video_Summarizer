"""
Batch processing API endpoints for YouTube video summarization.

This module provides comprehensive REST endpoints for batch processing operations,
including batch creation, management, monitoring, and queue operations.
"""

import logging
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any, Union
from fastapi import APIRouter, Depends, HTTPException, Query, Path, status
from pydantic import BaseModel, Field, validator
from sqlalchemy.orm import Session
from enum import Enum

from ..database.connection import get_database_session_dependency
from ..database.batch_models import BatchStatus, BatchItemStatus, BatchPriority
from ..services.batch_service import (
    BatchService, get_batch_service, BatchServiceError,
    BatchCreateRequest, BatchProgressInfo, BatchItemResult
)
from ..utils.validators import validate_youtube_url_detailed

logger = logging.getLogger(__name__)

# Create FastAPI router
router = APIRouter(prefix="/api/v1/batch", tags=["batch"])


# Pydantic enum models for API responses
class BatchStatusEnum(str, Enum):
    """Batch status enumeration for API responses."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class BatchItemStatusEnum(str, Enum):
    """Batch item status enumeration for API responses."""
    QUEUED = "queued"
    PROCESSING = "processing"
    METADATA_PROCESSING = "metadata_processing"
    SUMMARIZING = "summarizing"
    KEYWORD_EXTRACTION = "keyword_extraction"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class BatchPriorityEnum(str, Enum):
    """Batch priority enumeration for API responses."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


# Request models
class CreateBatchRequest(BaseModel):
    """Request model for creating a new batch."""
    name: Optional[str] = Field(None, description="Optional batch name", max_length=255)
    description: Optional[str] = Field(None, description="Optional batch description", max_length=1000)
    urls: List[str] = Field(..., description="List of YouTube URLs to process", min_items=1, max_items=100)
    priority: BatchPriorityEnum = Field(BatchPriorityEnum.NORMAL, description="Batch processing priority")
    webhook_url: Optional[str] = Field(None, description="Optional webhook URL for notifications", max_length=2000)
    batch_metadata: Optional[Dict[str, Any]] = Field(None, description="Additional batch metadata")

    @validator('urls')
    def validate_urls(cls, v):
        """Validate YouTube URLs."""
        if not v:
            raise ValueError("At least one URL is required")
        
        validated_urls = []
        for i, url in enumerate(v):
            if not url or not url.strip():
                raise ValueError(f"URL at index {i} is empty")
            
            url = url.strip()
            validation_result = validate_youtube_url_detailed(url)
            if not validation_result.is_valid:
                raise ValueError(f"Invalid YouTube URL at index {i}: {validation_result.error_message}")
            
            validated_urls.append(url)
        
        # Check for duplicates
        if len(validated_urls) != len(set(validated_urls)):
            raise ValueError("Duplicate URLs are not allowed")
        
        return validated_urls

    @validator('webhook_url')
    def validate_webhook_url(cls, v):
        """Validate webhook URL if provided."""
        if v and not v.startswith(('http://', 'https://')):
            raise ValueError("Webhook URL must start with http:// or https://")
        return v


class StartBatchRequest(BaseModel):
    """Request model for starting a batch."""
    force: bool = Field(False, description="Force start even if batch is not in pending state")


class CancelBatchRequest(BaseModel):
    """Request model for cancelling a batch."""
    reason: Optional[str] = Field(None, description="Reason for cancellation", max_length=500)


class RetryBatchItemRequest(BaseModel):
    """Request model for retrying a failed batch item."""
    force: bool = Field(False, description="Force retry even if max retries exceeded")


class UpdateBatchRequest(BaseModel):
    """Request model for updating batch properties."""
    name: Optional[str] = Field(None, description="Update batch name", max_length=255)
    description: Optional[str] = Field(None, description="Update batch description", max_length=1000)
    priority: Optional[BatchPriorityEnum] = Field(None, description="Update batch priority")
    webhook_url: Optional[str] = Field(None, description="Update webhook URL", max_length=2000)
    batch_metadata: Optional[Dict[str, Any]] = Field(None, description="Update batch metadata")


# Response models
class BatchItemResponse(BaseModel):
    """Response model for batch item information."""
    id: int
    batch_id: int
    video_id: Optional[int] = None
    url: str
    status: BatchItemStatusEnum
    priority: BatchPriorityEnum
    processing_order: int
    created_at: datetime
    updated_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    retry_count: int
    max_retries: int
    error_info: Optional[str] = None
    processing_data: Optional[Dict[str, Any]] = None
    result_data: Optional[Dict[str, Any]] = None
    can_retry: bool

    class Config:
        from_attributes = True


class BatchResponse(BaseModel):
    """Response model for batch information."""
    id: int
    batch_id: str
    name: Optional[str] = None
    description: Optional[str] = None
    status: BatchStatusEnum
    priority: BatchPriorityEnum
    total_items: int
    completed_items: int
    failed_items: int
    pending_items: int
    progress_percentage: float
    created_at: datetime
    updated_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    webhook_url: Optional[str] = None
    batch_metadata: Optional[Dict[str, Any]] = None
    error_info: Optional[str] = None
    is_completed: bool
    is_failed: bool
    batch_items: List[BatchItemResponse] = []

    class Config:
        from_attributes = True


class BatchSummaryResponse(BaseModel):
    """Response model for batch summary (without items)."""
    id: int
    batch_id: str
    name: Optional[str] = None
    description: Optional[str] = None
    status: BatchStatusEnum
    priority: BatchPriorityEnum
    total_items: int
    completed_items: int
    failed_items: int
    pending_items: int
    progress_percentage: float
    created_at: datetime
    updated_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    is_completed: bool
    is_failed: bool

    class Config:
        from_attributes = True


class BatchProgressResponse(BaseModel):
    """Response model for batch progress information."""
    batch_id: str
    status: BatchStatusEnum
    total_items: int
    completed_items: int
    failed_items: int
    pending_items: int
    progress_percentage: float
    started_at: Optional[datetime] = None
    estimated_completion: Optional[datetime] = None

    class Config:
        from_attributes = True


class BatchStatisticsResponse(BaseModel):
    """Response model for batch statistics."""
    total_batches: int
    batch_status_counts: Dict[str, int]
    total_batch_items: int
    item_status_counts: Dict[str, int]
    active_processing_sessions: int

    class Config:
        from_attributes = True


class QueueItemResponse(BaseModel):
    """Response model for queue item information."""
    id: int
    batch_item_id: int
    queue_name: str
    priority: BatchPriorityEnum
    scheduled_at: datetime
    created_at: datetime
    updated_at: datetime
    locked_at: Optional[datetime] = None
    locked_by: Optional[str] = None
    lock_expires_at: Optional[datetime] = None
    retry_count: int
    max_retries: int
    error_info: Optional[str] = None
    is_locked: bool
    is_available: bool
    can_retry: bool

    class Config:
        from_attributes = True


class ProcessingSessionResponse(BaseModel):
    """Response model for processing session information."""
    id: int
    session_id: str
    batch_item_id: int
    worker_id: str
    started_at: datetime
    updated_at: datetime
    heartbeat_at: datetime
    status: BatchItemStatusEnum
    progress_percentage: float
    current_step: Optional[str] = None
    session_metadata: Optional[Dict[str, Any]] = None
    error_info: Optional[str] = None
    is_stale: bool

    class Config:
        from_attributes = True


class BatchListResponse(BaseModel):
    """Response model for batch list with pagination."""
    batches: List[BatchSummaryResponse]
    total_count: int
    page: int
    page_size: int
    has_next: bool
    has_previous: bool


class ErrorResponse(BaseModel):
    """Response model for API errors."""
    error: str
    message: str
    details: Optional[Dict[str, Any]] = None


# API Endpoints

@router.post("/batches", response_model=BatchResponse, status_code=status.HTTP_201_CREATED)
async def create_batch(
    request: CreateBatchRequest,
    batch_service: BatchService = Depends(get_batch_service),
    db_session: Session = Depends(get_database_session_dependency)
):
    """
    Create a new batch for processing multiple YouTube videos.
    
    This endpoint creates a new batch with the specified URLs and configuration.
    The batch will be in PENDING status and can be started using the start endpoint.
    
    **Features:**
    - Validates all YouTube URLs before creating the batch
    - Supports batch priorities for processing order
    - Optional webhook notifications
    - Comprehensive error handling and validation
    """
    try:
        # Update batch service with database session
        batch_service._session = db_session
        
        # Create batch request
        batch_request = BatchCreateRequest(
            name=request.name,
            description=request.description,
            urls=request.urls,
            priority=BatchPriority(request.priority.value),
            webhook_url=request.webhook_url,
            batch_metadata=request.batch_metadata
        )
        
        # Create batch
        batch = batch_service.create_batch(batch_request)
        
        # Convert to response model
        batch_response = BatchResponse(
            id=batch.id,
            batch_id=batch.batch_id,
            name=batch.name,
            description=batch.description,
            status=BatchStatusEnum(batch.status.value),
            priority=BatchPriorityEnum(batch.priority.value),
            total_items=batch.total_items,
            completed_items=batch.completed_items,
            failed_items=batch.failed_items,
            pending_items=batch.pending_items,
            progress_percentage=batch.progress_percentage,
            created_at=batch.created_at,
            updated_at=batch.updated_at,
            started_at=batch.started_at,
            completed_at=batch.completed_at,
            webhook_url=batch.webhook_url,
            batch_metadata=batch.batch_metadata,
            error_info=batch.error_info,
            is_completed=batch.is_completed,
            is_failed=batch.is_failed,
            batch_items=[
                BatchItemResponse(
                    id=item.id,
                    batch_id=item.batch_id,
                    video_id=item.video_id,
                    url=item.url,
                    status=BatchItemStatusEnum(item.status.value),
                    priority=BatchPriorityEnum(item.priority.value),
                    processing_order=item.processing_order,
                    created_at=item.created_at,
                    updated_at=item.updated_at,
                    started_at=item.started_at,
                    completed_at=item.completed_at,
                    retry_count=item.retry_count,
                    max_retries=item.max_retries,
                    error_info=item.error_info,
                    processing_data=item.processing_data,
                    result_data=item.result_data,
                    can_retry=item.can_retry
                )
                for item in batch.batch_items
            ]
        )
        
        logger.info(f"Created batch {batch.batch_id} with {len(request.urls)} items")
        return batch_response
        
    except BatchServiceError as e:
        logger.error(f"Batch service error creating batch: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to create batch: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error creating batch: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.get("/batches", response_model=BatchListResponse)
async def list_batches(
    batch_status: Optional[BatchStatusEnum] = Query(None, description="Filter by batch status"),
    page: int = Query(1, ge=1, description="Page number (1-based)"),
    page_size: int = Query(20, ge=1, le=100, description="Number of items per page"),
    batch_service: BatchService = Depends(get_batch_service),
    db_session: Session = Depends(get_database_session_dependency)
):
    """
    List batches with optional filtering and pagination.
    
    **Features:**
    - Filter by batch status
    - Pagination support
    - Ordered by creation date (newest first)
    - Includes batch summary information
    """
    try:
        # Update batch service with database session
        batch_service._session = db_session
        
        # Convert status filter
        status_filter = None
        if batch_status:
            status_filter = BatchStatus(batch_status.value)
        
        # Calculate offset
        offset = (page - 1) * page_size
        
        # Get batches
        batches = batch_service.list_batches(
            status=status_filter,
            limit=page_size,
            offset=offset
        )
        
        # Get total count for pagination
        all_batches = batch_service.list_batches(
            status=status_filter,
            limit=1000,  # Large limit to count all
            offset=0
        )
        total_count = len(all_batches)
        
        # Convert to response models
        batch_responses = [
            BatchSummaryResponse(
                id=batch.id,
                batch_id=batch.batch_id,
                name=batch.name,
                description=batch.description,
                status=BatchStatusEnum(batch.status.value),
                priority=BatchPriorityEnum(batch.priority.value),
                total_items=batch.total_items,
                completed_items=batch.completed_items,
                failed_items=batch.failed_items,
                pending_items=batch.pending_items,
                progress_percentage=batch.progress_percentage,
                created_at=batch.created_at,
                updated_at=batch.updated_at,
                started_at=batch.started_at,
                completed_at=batch.completed_at,
                is_completed=batch.is_completed,
                is_failed=batch.is_failed
            )
            for batch in batches
        ]
        
        # Calculate pagination info
        has_next = offset + page_size < total_count
        has_previous = page > 1
        
        return BatchListResponse(
            batches=batch_responses,
            total_count=total_count,
            page=page,
            page_size=page_size,
            has_next=has_next,
            has_previous=has_previous
        )
        
    except BatchServiceError as e:
        logger.error(f"Batch service error listing batches: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list batches: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error listing batches: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.get("/batches/{batch_id}", response_model=BatchResponse)
async def get_batch(
    batch_id: str = Path(..., description="Batch identifier"),
    batch_service: BatchService = Depends(get_batch_service),
    db_session: Session = Depends(get_database_session_dependency)
):
    """
    Get detailed information about a specific batch.
    
    **Returns:**
    - Complete batch information
    - All batch items with their current status
    - Processing progress and statistics
    """
    try:
        # Update batch service with database session
        batch_service._session = db_session
        
        # Get batch
        batch = batch_service.get_batch(batch_id)
        
        if not batch:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Batch {batch_id} not found"
            )
        
        # Convert to response model
        batch_response = BatchResponse(
            id=batch.id,
            batch_id=batch.batch_id,
            name=batch.name,
            description=batch.description,
            status=BatchStatusEnum(batch.status.value),
            priority=BatchPriorityEnum(batch.priority.value),
            total_items=batch.total_items,
            completed_items=batch.completed_items,
            failed_items=batch.failed_items,
            pending_items=batch.pending_items,
            progress_percentage=batch.progress_percentage,
            created_at=batch.created_at,
            updated_at=batch.updated_at,
            started_at=batch.started_at,
            completed_at=batch.completed_at,
            webhook_url=batch.webhook_url,
            batch_metadata=batch.batch_metadata,
            error_info=batch.error_info,
            is_completed=batch.is_completed,
            is_failed=batch.is_failed,
            batch_items=[
                BatchItemResponse(
                    id=item.id,
                    batch_id=item.batch_id,
                    video_id=item.video_id,
                    url=item.url,
                    status=BatchItemStatusEnum(item.status.value),
                    priority=BatchPriorityEnum(item.priority.value),
                    processing_order=item.processing_order,
                    created_at=item.created_at,
                    updated_at=item.updated_at,
                    started_at=item.started_at,
                    completed_at=item.completed_at,
                    retry_count=item.retry_count,
                    max_retries=item.max_retries,
                    error_info=item.error_info,
                    processing_data=item.processing_data,
                    result_data=item.result_data,
                    can_retry=item.can_retry
                )
                for item in batch.batch_items
            ]
        )
        
        return batch_response
        
    except BatchServiceError as e:
        logger.error(f"Batch service error getting batch: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get batch: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error getting batch: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.get("/batches/{batch_id}/progress", response_model=BatchProgressResponse)
async def get_batch_progress(
    batch_id: str = Path(..., description="Batch identifier"),
    batch_service: BatchService = Depends(get_batch_service),
    db_session: Session = Depends(get_database_session_dependency)
):
    """
    Get batch progress information with completion estimates.
    
    **Returns:**
    - Current processing progress
    - Estimated completion time
    - Item counts by status
    """
    try:
        # Update batch service with database session
        batch_service._session = db_session
        
        # Get batch progress
        progress = batch_service.get_batch_progress(batch_id)
        
        if not progress:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Batch {batch_id} not found"
            )
        
        # Convert to response model
        return BatchProgressResponse(
            batch_id=progress.batch_id,
            status=BatchStatusEnum(progress.status.value),
            total_items=progress.total_items,
            completed_items=progress.completed_items,
            failed_items=progress.failed_items,
            pending_items=progress.pending_items,
            progress_percentage=progress.progress_percentage,
            started_at=progress.started_at,
            estimated_completion=progress.estimated_completion
        )
        
    except BatchServiceError as e:
        logger.error(f"Batch service error getting batch progress: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get batch progress: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error getting batch progress: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.post("/batches/{batch_id}/start")
async def start_batch(
    batch_id: str = Path(..., description="Batch identifier"),
    request: StartBatchRequest = StartBatchRequest(),
    batch_service: BatchService = Depends(get_batch_service),
    db_session: Session = Depends(get_database_session_dependency)
):
    """
    Start processing a batch.
    
    **Features:**
    - Validates batch can be started
    - Sets batch status to PROCESSING
    - Initializes processing timestamps
    - Returns success confirmation
    """
    try:
        # Update batch service with database session
        batch_service._session = db_session
        
        # Start batch processing
        success = batch_service.start_batch_processing(batch_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to start batch {batch_id}"
            )
        
        logger.info(f"Started batch processing for {batch_id}")
        return {
            "success": True,
            "message": f"Batch {batch_id} started successfully",
            "batch_id": batch_id
        }
        
    except BatchServiceError as e:
        logger.error(f"Batch service error starting batch: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to start batch: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error starting batch: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.post("/batches/{batch_id}/cancel")
async def cancel_batch(
    batch_id: str = Path(..., description="Batch identifier"),
    request: CancelBatchRequest = CancelBatchRequest(),
    batch_service: BatchService = Depends(get_batch_service),
    db_session: Session = Depends(get_database_session_dependency)
):
    """
    Cancel a batch and all its pending items.
    
    **Features:**
    - Cancels batch and all pending items
    - Records cancellation reason
    - Updates batch status to CANCELLED
    - Returns cancellation confirmation
    """
    try:
        # Update batch service with database session
        batch_service._session = db_session
        
        # Cancel batch
        success = batch_service.cancel_batch(batch_id, request.reason)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to cancel batch {batch_id}"
            )
        
        logger.info(f"Cancelled batch {batch_id}: {request.reason}")
        return {
            "success": True,
            "message": f"Batch {batch_id} cancelled successfully",
            "batch_id": batch_id,
            "reason": request.reason
        }
        
    except BatchServiceError as e:
        logger.error(f"Batch service error cancelling batch: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to cancel batch: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error cancelling batch: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.post("/batches/{batch_id}/items/{item_id}/retry")
async def retry_batch_item(
    batch_id: str = Path(..., description="Batch identifier"),
    item_id: int = Path(..., description="Batch item ID"),
    request: RetryBatchItemRequest = RetryBatchItemRequest(),
    batch_service: BatchService = Depends(get_batch_service),
    db_session: Session = Depends(get_database_session_dependency)
):
    """
    Retry a failed batch item.
    
    **Features:**
    - Validates item can be retried
    - Resets item status to QUEUED
    - Increments retry count
    - Re-queues item for processing
    """
    try:
        # Update batch service with database session
        batch_service._session = db_session
        
        # Retry batch item
        success = batch_service.retry_failed_batch_item(item_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to retry batch item {item_id}"
            )
        
        logger.info(f"Retried batch item {item_id} in batch {batch_id}")
        return {
            "success": True,
            "message": f"Batch item {item_id} retried successfully",
            "batch_id": batch_id,
            "item_id": item_id
        }
        
    except BatchServiceError as e:
        logger.error(f"Batch service error retrying batch item: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to retry batch item: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error retrying batch item: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.get("/queue/next")
async def get_next_queue_item(
    queue_name: str = Query("video_processing", description="Queue name"),
    worker_id: str = Query(..., description="Worker identifier"),
    batch_service: BatchService = Depends(get_batch_service),
    db_session: Session = Depends(get_database_session_dependency)
):
    """
    Get the next available queue item for processing.
    
    **Features:**
    - Retrieves highest priority available item
    - Locks item for the requesting worker
    - Sets lock expiration time
    - Returns item with batch context
    """
    try:
        # Update batch service with database session
        batch_service._session = db_session
        
        # Get next queue item
        queue_item = batch_service.get_next_queue_item(queue_name, worker_id)
        
        if not queue_item:
            return {
                "available": False,
                "message": "No items available in queue"
            }
        
        # Convert to response
        return {
            "available": True,
            "queue_item": {
                "id": queue_item.id,
                "batch_item_id": queue_item.batch_item_id,
                "queue_name": queue_item.queue_name,
                "priority": queue_item.priority.value,
                "locked_by": queue_item.locked_by,
                "lock_expires_at": queue_item.lock_expires_at,
                "batch_item": {
                    "id": queue_item.batch_item.id,
                    "batch_id": queue_item.batch_item.batch_id,
                    "url": queue_item.batch_item.url,
                    "status": queue_item.batch_item.status.value,
                    "retry_count": queue_item.batch_item.retry_count,
                    "max_retries": queue_item.batch_item.max_retries
                } if queue_item.batch_item else None
            }
        }
        
    except BatchServiceError as e:
        logger.error(f"Batch service error getting queue item: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get queue item: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error getting queue item: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.post("/queue/complete/{batch_item_id}")
async def complete_batch_item(
    batch_item_id: int = Path(..., description="Batch item ID"),
    result: Dict[str, Any] = ...,
    batch_service: BatchService = Depends(get_batch_service),
    db_session: Session = Depends(get_database_session_dependency)
):
    """
    Complete processing of a batch item.
    
    **Features:**
    - Updates item status to COMPLETED or FAILED
    - Records processing results
    - Updates batch progress counters
    - Removes item from processing queue
    """
    try:
        # Update batch service with database session
        batch_service._session = db_session
        
        # Create result object
        batch_result = BatchItemResult(
            batch_item_id=batch_item_id,
            status=BatchItemStatus(result.get('status', 'failed')),
            video_id=result.get('video_id'),
            error_message=result.get('error_message'),
            processing_time_seconds=result.get('processing_time_seconds'),
            result_data=result.get('result_data')
        )
        
        # Complete batch item
        success = batch_service.complete_batch_item(batch_item_id, batch_result)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to complete batch item {batch_item_id}"
            )
        
        logger.info(f"Completed batch item {batch_item_id} with status {batch_result.status.value}")
        return {
            "success": True,
            "message": f"Batch item {batch_item_id} completed successfully",
            "batch_item_id": batch_item_id,
            "status": batch_result.status.value
        }
        
    except BatchServiceError as e:
        logger.error(f"Batch service error completing batch item: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to complete batch item: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error completing batch item: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.get("/statistics", response_model=BatchStatisticsResponse)
async def get_batch_statistics(
    batch_service: BatchService = Depends(get_batch_service),
    db_session: Session = Depends(get_database_session_dependency)
):
    """
    Get batch processing statistics.
    
    **Returns:**
    - Total batches and items counts
    - Status distribution
    - Active processing sessions
    - Performance metrics
    """
    try:
        # Update batch service with database session
        batch_service._session = db_session
        
        # Get statistics
        stats = batch_service.get_batch_statistics()
        
        return BatchStatisticsResponse(
            total_batches=stats["total_batches"],
            batch_status_counts=stats["batch_status_counts"],
            total_batch_items=stats["total_batch_items"],
            item_status_counts=stats["item_status_counts"],
            active_processing_sessions=stats["active_processing_sessions"]
        )
        
    except BatchServiceError as e:
        logger.error(f"Batch service error getting statistics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get statistics: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error getting statistics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.post("/cleanup/stale-sessions")
async def cleanup_stale_sessions(
    timeout_minutes: int = Query(30, ge=1, le=1440, description="Session timeout in minutes"),
    batch_service: BatchService = Depends(get_batch_service),
    db_session: Session = Depends(get_database_session_dependency)
):
    """
    Clean up stale processing sessions.
    
    **Features:**
    - Removes sessions that haven't sent heartbeat
    - Updates related batch items to FAILED status
    - Configurable timeout period
    - Returns cleanup count
    """
    try:
        # Update batch service with database session
        batch_service._session = db_session
        
        # Clean up stale sessions
        cleaned_count = batch_service.cleanup_stale_sessions(timeout_minutes)
        
        logger.info(f"Cleaned up {cleaned_count} stale processing sessions")
        return {
            "success": True,
            "message": f"Cleaned up {cleaned_count} stale sessions",
            "cleaned_count": cleaned_count,
            "timeout_minutes": timeout_minutes
        }
        
    except BatchServiceError as e:
        logger.error(f"Batch service error cleaning up sessions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cleanup sessions: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error cleaning up sessions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.get("/health")
async def batch_health_check():
    """
    Health check endpoint for batch processing system.
    
    **Returns:**
    - Service status
    - Component health
    - Timestamp
    """
    return {
        "status": "healthy",
        "service": "batch_processing",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "components": {
            "batch_service": "ready",
            "queue_system": "ready",
            "database": "ready"
        }
    }


# Batch processing session management endpoints
@router.post("/sessions/{batch_item_id}")
async def create_processing_session(
    batch_item_id: int = Path(..., description="Batch item ID"),
    worker_id: str = Query(..., description="Worker identifier"),
    batch_service: BatchService = Depends(get_batch_service),
    db_session: Session = Depends(get_database_session_dependency)
):
    """
    Create a processing session for progress tracking.
    
    **Features:**
    - Creates session for tracking progress
    - Associates with worker and batch item
    - Initializes progress tracking
    - Returns session identifier
    """
    try:
        # Update batch service with database session
        batch_service._session = db_session
        
        # Create processing session
        session = batch_service.create_processing_session(batch_item_id, worker_id)
        
        return {
            "success": True,
            "session_id": session.session_id,
            "batch_item_id": batch_item_id,
            "worker_id": worker_id,
            "created_at": session.started_at
        }
        
    except BatchServiceError as e:
        logger.error(f"Batch service error creating session: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to create session: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error creating session: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.put("/sessions/{session_id}/progress")
async def update_session_progress(
    session_id: str = Path(..., description="Session identifier"),
    progress: float = Query(..., ge=0, le=100, description="Progress percentage"),
    current_step: Optional[str] = Query(None, description="Current processing step"),
    batch_service: BatchService = Depends(get_batch_service),
    db_session: Session = Depends(get_database_session_dependency)
):
    """
    Update processing session progress.
    
    **Features:**
    - Updates progress percentage
    - Records current processing step
    - Updates heartbeat timestamp
    - Returns update confirmation
    """
    try:
        # Update batch service with database session
        batch_service._session = db_session
        
        # Update session progress
        success = batch_service.update_processing_session(session_id, progress, current_step)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session {session_id} not found"
            )
        
        return {
            "success": True,
            "session_id": session_id,
            "progress": progress,
            "current_step": current_step,
            "updated_at": datetime.utcnow()
        }
        
    except BatchServiceError as e:
        logger.error(f"Batch service error updating session: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to update session: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error updating session: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


# Error handlers would be added to the main app, not the router
# These are handled in the endpoints with try/catch blocks instead