"""
Status tracking API endpoints for YouTube Summarizer application.
Provides REST API endpoints for status monitoring and management.
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Union
from fastapi import APIRouter, Depends, HTTPException, Query, Path
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field, validator
from enum import Enum

from ..database.connection import get_db_session
from ..database.status_models import ProcessingStatusType, ProcessingPriority, StatusChangeType
from ..services.status_service import StatusService
from ..services.status_metrics_service import StatusMetricsService
from ..utils.error_messages import ErrorMessageProvider


# Request/Response Models
class StatusResponse(BaseModel):
    """Processing status response model."""
    status_id: str
    video_id: Optional[int] = None
    batch_item_id: Optional[int] = None
    processing_session_id: Optional[int] = None
    status: str
    substatus: Optional[str] = None
    priority: str
    progress_percentage: float
    current_step: Optional[str] = None
    total_steps: Optional[int] = None
    completed_steps: int
    estimated_completion_time: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    worker_id: Optional[str] = None
    heartbeat_at: Optional[datetime] = None
    retry_count: int
    max_retries: int
    processing_metadata: Optional[Dict[str, Any]] = None
    result_metadata: Optional[Dict[str, Any]] = None
    error_info: Optional[str] = None
    tags: Optional[List[str]] = None
    external_id: Optional[str] = None
    
    class Config:
        from_attributes = True


class StatusHistoryResponse(BaseModel):
    """Status history response model."""
    id: int
    change_type: str
    previous_status: Optional[str] = None
    new_status: str
    previous_progress: Optional[float] = None
    new_progress: float
    change_reason: Optional[str] = None
    change_metadata: Optional[Dict[str, Any]] = None
    worker_id: Optional[str] = None
    created_at: datetime
    duration_seconds: Optional[int] = None
    error_info: Optional[str] = None
    external_trigger: Optional[str] = None
    
    class Config:
        from_attributes = True


class StatusMetricsResponse(BaseModel):
    """Status metrics response model."""
    id: int
    metric_date: datetime
    metric_hour: Optional[int] = None
    total_items: int
    completed_items: int
    failed_items: int
    cancelled_items: int
    pending_items: int
    average_processing_time_seconds: Optional[float] = None
    median_processing_time_seconds: Optional[float] = None
    max_processing_time_seconds: Optional[float] = None
    min_processing_time_seconds: Optional[float] = None
    retry_rate_percentage: Optional[float] = None
    success_rate_percentage: Optional[float] = None
    queue_wait_time_seconds: Optional[float] = None
    worker_utilization_percentage: Optional[float] = None
    created_at: datetime
    updated_at: datetime
    metrics_metadata: Optional[Dict[str, Any]] = None
    
    class Config:
        from_attributes = True


class StatusUpdateRequest(BaseModel):
    """Status update request model."""
    new_status: str
    progress_percentage: Optional[float] = Field(None, ge=0, le=100)
    current_step: Optional[str] = None
    completed_steps: Optional[int] = Field(None, ge=0)
    worker_id: Optional[str] = None
    error_info: Optional[str] = None
    change_reason: Optional[str] = None
    change_metadata: Optional[Dict[str, Any]] = None
    estimated_completion_time: Optional[datetime] = None
    
    @validator('new_status')
    def validate_status(cls, v):
        try:
            ProcessingStatusType(v)
            return v
        except ValueError:
            raise ValueError(f"Invalid status: {v}")


class ProgressUpdateRequest(BaseModel):
    """Progress update request model."""
    progress_percentage: float = Field(..., ge=0, le=100)
    current_step: Optional[str] = None
    completed_steps: Optional[int] = Field(None, ge=0)
    worker_id: Optional[str] = None
    processing_metadata: Optional[Dict[str, Any]] = None


class ErrorReportRequest(BaseModel):
    """Error report request model."""
    error_info: str
    error_metadata: Optional[Dict[str, Any]] = None
    worker_id: Optional[str] = None
    should_retry: bool = True


class HeartbeatRequest(BaseModel):
    """Heartbeat request model."""
    worker_id: str
    progress_percentage: Optional[float] = Field(None, ge=0, le=100)
    current_step: Optional[str] = None


class StatusListResponse(BaseModel):
    """Status list response model."""
    statuses: List[StatusResponse]
    total_count: int
    page: int
    page_size: int
    has_next: bool
    has_previous: bool


class PerformanceSummaryResponse(BaseModel):
    """Performance summary response model."""
    active_processing_count: int
    active_workers: List[str]
    worker_count: int
    average_progress: float
    today_total: int
    today_completed: int
    today_failed: int
    today_success_rate: float
    success_rate_trend: float
    recent_metrics_count: int
    timestamp: str


class WorkerPerformanceResponse(BaseModel):
    """Worker performance response model."""
    worker_id: str
    total_processed: int
    completed_count: int
    failed_count: int
    success_rate: float
    error_rate: float
    average_processing_time: Optional[float] = None
    processing_time_samples: int
    days_analyzed: int
    timestamp: str


# API Router
router = APIRouter(prefix="/api/status", tags=["Status Tracking"])


def get_status_service(db: Session = Depends(get_db_session)) -> StatusService:
    """Dependency to get StatusService instance."""
    return StatusService(db_session=db)


def get_metrics_service(db: Session = Depends(get_db_session)) -> StatusMetricsService:
    """Dependency to get StatusMetricsService instance."""
    return StatusMetricsService(db_session=db)


# Status endpoints
@router.get("/{status_id}", response_model=StatusResponse)
def get_status(
    status_id: str = Path(..., description="Status ID"),
    service: StatusService = Depends(get_status_service)
):
    """Get processing status by ID."""
    try:
        status = service.get_processing_status(status_id)
        if not status:
            raise HTTPException(
                status_code=404,
                detail=f"Processing status not found: {status_id}"
            )
        
        return StatusResponse.from_orm(status)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving status: {str(e)}"
        )


@router.get("/video/{video_id}", response_model=StatusResponse)
def get_status_by_video(
    video_id: int = Path(..., description="Video ID"),
    service: StatusService = Depends(get_status_service)
):
    """Get processing status by video ID."""
    try:
        status = service.get_status_by_video_id(video_id)
        if not status:
            raise HTTPException(
                status_code=404,
                detail=f"Processing status not found for video: {video_id}"
            )
        
        return StatusResponse.from_orm(status)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving status: {str(e)}"
        )


@router.get("/batch/{batch_item_id}", response_model=StatusResponse)
def get_status_by_batch_item(
    batch_item_id: int = Path(..., description="Batch item ID"),
    service: StatusService = Depends(get_status_service)
):
    """Get processing status by batch item ID."""
    try:
        status = service.get_status_by_batch_item_id(batch_item_id)
        if not status:
            raise HTTPException(
                status_code=404,
                detail=f"Processing status not found for batch item: {batch_item_id}"
            )
        
        return StatusResponse.from_orm(status)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving status: {str(e)}"
        )


@router.get("/", response_model=StatusListResponse)
def list_statuses(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Page size"),
    status_filter: Optional[str] = Query(None, description="Status filter"),
    worker_id: Optional[str] = Query(None, description="Worker ID filter"),
    active_only: bool = Query(False, description="Show only active statuses"),
    service: StatusService = Depends(get_status_service)
):
    """List processing statuses with pagination and filtering."""
    try:
        # Calculate offset
        offset = (page - 1) * page_size
        
        # Get statuses based on filters
        if active_only:
            statuses = service.get_active_statuses(worker_id=worker_id, limit=page_size + 1)
            # For active statuses, we don't have a total count readily available
            total_count = len(statuses)
        else:
            # This would need to be implemented in the service
            # For now, just get active statuses
            statuses = service.get_active_statuses(worker_id=worker_id, limit=page_size + 1)
            total_count = len(statuses)
        
        # Check if there are more results
        has_next = len(statuses) > page_size
        if has_next:
            statuses = statuses[:-1]  # Remove the extra item
        
        # Convert to response models
        status_responses = [StatusResponse.from_orm(status) for status in statuses]
        
        return StatusListResponse(
            statuses=status_responses,
            total_count=total_count,
            page=page,
            page_size=page_size,
            has_next=has_next,
            has_previous=page > 1
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error listing statuses: {str(e)}"
        )


@router.get("/active/count")
def get_active_count(
    worker_id: Optional[str] = Query(None, description="Worker ID filter"),
    service: StatusService = Depends(get_status_service)
):
    """Get count of active processing statuses."""
    try:
        statuses = service.get_active_statuses(worker_id=worker_id)
        return {"active_count": len(statuses)}
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting active count: {str(e)}"
        )


@router.get("/stale/list", response_model=List[StatusResponse])
def get_stale_statuses(
    timeout_seconds: int = Query(300, ge=60, description="Timeout in seconds"),
    limit: Optional[int] = Query(50, ge=1, le=500, description="Limit results"),
    service: StatusService = Depends(get_status_service)
):
    """Get stale processing statuses."""
    try:
        statuses = service.get_stale_statuses(
            timeout_seconds=timeout_seconds,
            limit=limit
        )
        return [StatusResponse.from_orm(status) for status in statuses]
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting stale statuses: {str(e)}"
        )


# Status update endpoints
@router.put("/{status_id}", response_model=StatusResponse)
def update_status(
    status_id: str = Path(..., description="Status ID"),
    update_request: StatusUpdateRequest = ...,
    service: StatusService = Depends(get_status_service)
):
    """Update processing status."""
    try:
        # Validate status type
        new_status_type = ProcessingStatusType(update_request.new_status)
        
        updated_status = service.update_status(
            status_id=status_id,
            new_status=new_status_type,
            progress_percentage=update_request.progress_percentage,
            current_step=update_request.current_step,
            completed_steps=update_request.completed_steps,
            worker_id=update_request.worker_id,
            error_info=update_request.error_info,
            change_reason=update_request.change_reason,
            change_metadata=update_request.change_metadata,
            estimated_completion_time=update_request.estimated_completion_time
        )
        
        return StatusResponse.from_orm(updated_status)
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error updating status: {str(e)}"
        )


@router.patch("/{status_id}/progress", response_model=StatusResponse)
def update_progress(
    status_id: str = Path(..., description="Status ID"),
    progress_request: ProgressUpdateRequest = ...,
    service: StatusService = Depends(get_status_service)
):
    """Update processing progress."""
    try:
        updated_status = service.update_progress(
            status_id=status_id,
            progress_percentage=progress_request.progress_percentage,
            current_step=progress_request.current_step,
            completed_steps=progress_request.completed_steps,
            worker_id=progress_request.worker_id,
            processing_metadata=progress_request.processing_metadata
        )
        
        return StatusResponse.from_orm(updated_status)
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error updating progress: {str(e)}"
        )


@router.post("/{status_id}/error", response_model=StatusResponse)
def report_error(
    status_id: str = Path(..., description="Status ID"),
    error_request: ErrorReportRequest = ...,
    service: StatusService = Depends(get_status_service)
):
    """Report an error for processing status."""
    try:
        updated_status = service.record_error(
            status_id=status_id,
            error_info=error_request.error_info,
            error_metadata=error_request.error_metadata,
            worker_id=error_request.worker_id,
            should_retry=error_request.should_retry
        )
        
        return StatusResponse.from_orm(updated_status)
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error reporting error: {str(e)}"
        )


@router.post("/{status_id}/heartbeat", response_model=StatusResponse)
def update_heartbeat(
    status_id: str = Path(..., description="Status ID"),
    heartbeat_request: HeartbeatRequest = ...,
    service: StatusService = Depends(get_status_service)
):
    """Update processing heartbeat."""
    try:
        updated_status = service.heartbeat(
            status_id=status_id,
            worker_id=heartbeat_request.worker_id,
            progress_percentage=heartbeat_request.progress_percentage,
            current_step=heartbeat_request.current_step
        )
        
        return StatusResponse.from_orm(updated_status)
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error updating heartbeat: {str(e)}"
        )


# Status history endpoints
@router.get("/{status_id}/history", response_model=List[StatusHistoryResponse])
def get_status_history(
    status_id: str = Path(..., description="Status ID"),
    limit: Optional[int] = Query(50, ge=1, le=500, description="Limit results"),
    change_type: Optional[str] = Query(None, description="Change type filter"),
    service: StatusService = Depends(get_status_service)
):
    """Get status history for a processing status."""
    try:
        # Validate change type if provided
        change_type_enum = None
        if change_type:
            try:
                change_type_enum = StatusChangeType(change_type)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid change type: {change_type}"
                )
        
        history = service.get_status_history(
            status_id=status_id,
            limit=limit,
            change_type=change_type_enum
        )
        
        return [StatusHistoryResponse.from_orm(entry) for entry in history]
    
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting status history: {str(e)}"
        )


# Metrics endpoints
@router.get("/metrics/summary", response_model=PerformanceSummaryResponse)
def get_performance_summary(
    service: StatusMetricsService = Depends(get_metrics_service)
):
    """Get current performance summary."""
    try:
        summary = service.get_current_performance_summary()
        return PerformanceSummaryResponse(**summary)
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting performance summary: {str(e)}"
        )


@router.get("/metrics/worker/{worker_id}", response_model=WorkerPerformanceResponse)
def get_worker_performance(
    worker_id: str = Path(..., description="Worker ID"),
    days: int = Query(7, ge=1, le=90, description="Days to analyze"),
    service: StatusMetricsService = Depends(get_metrics_service)
):
    """Get performance metrics for a specific worker."""
    try:
        performance = service.get_worker_performance(worker_id, days)
        return WorkerPerformanceResponse(**performance)
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting worker performance: {str(e)}"
        )


@router.get("/metrics/distribution")
def get_status_distribution(
    days: int = Query(7, ge=1, le=90, description="Days to analyze"),
    service: StatusMetricsService = Depends(get_metrics_service)
):
    """Get distribution of processing statuses."""
    try:
        distribution = service.get_status_distribution(days)
        return {"distribution": distribution, "days_analyzed": days}
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting status distribution: {str(e)}"
        )


@router.get("/metrics/hourly", response_model=List[StatusMetricsResponse])
def get_hourly_metrics(
    start_date: datetime = Query(..., description="Start date"),
    end_date: datetime = Query(..., description="End date"),
    service: StatusMetricsService = Depends(get_metrics_service)
):
    """Get hourly metrics for a date range."""
    try:
        if (end_date - start_date).days > 7:
            raise HTTPException(
                status_code=400,
                detail="Date range cannot exceed 7 days for hourly metrics"
            )
        
        metrics = service.get_metrics_for_period(start_date, end_date, hourly=True)
        return [StatusMetricsResponse.from_orm(metric) for metric in metrics]
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting hourly metrics: {str(e)}"
        )


@router.get("/metrics/daily", response_model=List[StatusMetricsResponse])
def get_daily_metrics(
    start_date: datetime = Query(..., description="Start date"),
    end_date: datetime = Query(..., description="End date"),
    service: StatusMetricsService = Depends(get_metrics_service)
):
    """Get daily metrics for a date range."""
    try:
        if (end_date - start_date).days > 90:
            raise HTTPException(
                status_code=400,
                detail="Date range cannot exceed 90 days for daily metrics"
            )
        
        metrics = service.get_metrics_for_period(start_date, end_date, hourly=False)
        return [StatusMetricsResponse.from_orm(metric) for metric in metrics]
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting daily metrics: {str(e)}"
        )


# Utility endpoints
@router.delete("/cleanup")
def cleanup_old_statuses(
    days_old: int = Query(30, ge=1, le=365, description="Days old threshold"),
    keep_failed: bool = Query(True, description="Keep failed statuses"),
    service: StatusService = Depends(get_status_service)
):
    """Clean up old processing statuses."""
    try:
        count = service.cleanup_old_statuses(days_old, keep_failed)
        return {"cleaned_up_count": count}
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error cleaning up statuses: {str(e)}"
        )


@router.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


# Include additional utility endpoints
@router.get("/enums/status-types")
def get_status_types():
    """Get available status types."""
    return {
        "status_types": [status.value for status in ProcessingStatusType]
    }


@router.get("/enums/priorities")
def get_priorities():
    """Get available priority levels."""
    return {
        "priorities": [priority.value for priority in ProcessingPriority]
    }


@router.get("/enums/change-types")
def get_change_types():
    """Get available change types."""
    return {
        "change_types": [change_type.value for change_type in StatusChangeType]
    }