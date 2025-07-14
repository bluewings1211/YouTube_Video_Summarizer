"""
Enhanced status tracking API endpoints with advanced filtering and pagination.

This module extends the existing status API with comprehensive filtering,
sorting, search, and pagination capabilities.
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Union
from fastapi import APIRouter, Depends, HTTPException, Query, Path, Body
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field, validator
from enum import Enum

from ..database.connection import get_db_session
from ..database.status_models import ProcessingStatusType, ProcessingPriority, StatusChangeType
from ..services.status_service import StatusService
from ..services.status_filtering import (
    StatusFilterService, FilterQuery, FilterCondition, SortCondition,
    PaginationParams, SearchParams, FilterOperator, SortOrder,
    QuickFilterPresets
)
from .status import (
    StatusResponse, StatusHistoryResponse, StatusListResponse,
    get_status_service
)


# Enhanced Request/Response Models
class FilterConditionModel(BaseModel):
    """Filter condition model for API requests."""
    field: str
    operator: str
    value: Optional[Any] = None
    value2: Optional[Any] = None
    
    @validator('operator')
    def validate_operator(cls, v):
        try:
            FilterOperator(v)
            return v
        except ValueError:
            raise ValueError(f"Invalid filter operator: {v}")


class SortConditionModel(BaseModel):
    """Sort condition model for API requests."""
    field: str
    order: str = "desc"
    
    @validator('order')
    def validate_order(cls, v):
        if v.lower() not in ['asc', 'desc']:
            raise ValueError("Sort order must be 'asc' or 'desc'")
        return v.lower()


class SearchParamsModel(BaseModel):
    """Search parameters model for API requests."""
    query: str
    fields: List[str] = Field(default=['current_step', 'error_info', 'tags'])
    exact_match: bool = False
    case_sensitive: bool = False


class AdvancedFilterRequest(BaseModel):
    """Advanced filter request model."""
    filters: List[FilterConditionModel] = Field(default=[])
    sorts: List[SortConditionModel] = Field(default=[])
    search: Optional[SearchParamsModel] = None
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=20, ge=1, le=100)
    include_counts: bool = True
    include_related: bool = False


class EnhancedStatusListResponse(BaseModel):
    """Enhanced status list response with filtering metadata."""
    statuses: List[StatusResponse]
    total_count: int
    page: int
    page_size: int
    total_pages: int
    has_next: bool
    has_previous: bool
    filters_applied: int
    query_time_ms: float
    metadata: Optional[Dict[str, Any]] = None


class StatusAggregatesResponse(BaseModel):
    """Status aggregates response model."""
    total_count: int
    status_distribution: Dict[str, int]
    priority_distribution: Dict[str, int]
    progress_statistics: Dict[str, float]
    worker_statistics: Dict[str, int]
    time_statistics: Dict[str, int]
    timestamp: str


class PresetFilterType(str, Enum):
    """Predefined filter types."""
    ACTIVE = "active"
    FAILED = "failed"
    COMPLETED_TODAY = "completed_today"
    HIGH_PRIORITY = "high_priority"
    STALE = "stale"


# Enhanced API Router
router = APIRouter(prefix="/api/status/enhanced", tags=["Enhanced Status Tracking"])


def get_filter_service(db: Session = Depends(get_db_session)) -> StatusFilterService:
    """Dependency to get StatusFilterService instance."""
    return StatusFilterService(db_session=db)


# Enhanced status listing with advanced filtering
@router.post("/filter", response_model=EnhancedStatusListResponse)
def filter_statuses(
    filter_request: AdvancedFilterRequest,
    filter_service: StatusFilterService = Depends(get_filter_service)
):
    """
    Filter processing statuses with advanced filtering, sorting, and pagination.
    
    This endpoint provides comprehensive filtering capabilities including:
    - Multiple filter conditions with various operators
    - Multi-field sorting
    - Full-text search across specified fields
    - Pagination with metadata
    - Performance metrics
    """
    try:
        # Convert request models to service models
        filters = []
        for filter_model in filter_request.filters:
            filters.append(FilterCondition(
                field=filter_model.field,
                operator=FilterOperator(filter_model.operator),
                value=filter_model.value,
                value2=filter_model.value2
            ))
        
        sorts = []
        for sort_model in filter_request.sorts:
            sorts.append(SortCondition(
                field=sort_model.field,
                order=SortOrder(sort_model.order)
            ))
        
        search = None
        if filter_request.search:
            search = SearchParams(
                query=filter_request.search.query,
                fields=filter_request.search.fields,
                exact_match=filter_request.search.exact_match,
                case_sensitive=filter_request.search.case_sensitive
            )
        
        pagination = PaginationParams(
            page=filter_request.page,
            page_size=filter_request.page_size
        )
        
        # Create filter query
        filter_query = FilterQuery(
            filters=filters,
            sorts=sorts,
            pagination=pagination,
            search=search,
            include_counts=filter_request.include_counts,
            include_related=filter_request.include_related
        )
        
        # Execute filter
        result = filter_service.filter_statuses(filter_query)
        
        # Convert to response models
        status_responses = [StatusResponse.from_orm(status) for status in result.items]
        
        return EnhancedStatusListResponse(
            statuses=status_responses,
            total_count=result.total_count,
            page=result.page,
            page_size=result.page_size,
            total_pages=result.total_pages,
            has_next=result.has_next,
            has_previous=result.has_previous,
            filters_applied=result.filters_applied,
            query_time_ms=result.query_time_ms,
            metadata=result.metadata
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error filtering statuses: {str(e)}"
        )


# Preset filters for common use cases
@router.get("/preset/{preset_type}", response_model=EnhancedStatusListResponse)
def get_preset_filter(
    preset_type: PresetFilterType,
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Page size"),
    filter_service: StatusFilterService = Depends(get_filter_service)
):
    """
    Get statuses using predefined filter presets.
    
    Available presets:
    - active: Currently active processing statuses
    - failed: Failed statuses from last 24 hours
    - completed_today: Statuses completed today
    - high_priority: High priority statuses
    - stale: Stale statuses (no heartbeat for 30+ minutes)
    """
    try:
        # Get preset filter query
        if preset_type == PresetFilterType.ACTIVE:
            filter_query = QuickFilterPresets.active_statuses()
        elif preset_type == PresetFilterType.FAILED:
            filter_query = QuickFilterPresets.failed_statuses()
        elif preset_type == PresetFilterType.COMPLETED_TODAY:
            filter_query = QuickFilterPresets.completed_today()
        elif preset_type == PresetFilterType.HIGH_PRIORITY:
            filter_query = QuickFilterPresets.high_priority()
        elif preset_type == PresetFilterType.STALE:
            filter_query = QuickFilterPresets.stale_statuses()
        else:
            raise HTTPException(status_code=400, detail=f"Unknown preset type: {preset_type}")
        
        # Override pagination
        filter_query.pagination = PaginationParams(page=page, page_size=page_size)
        
        # Execute filter
        result = filter_service.filter_statuses(filter_query)
        
        # Convert to response models
        status_responses = [StatusResponse.from_orm(status) for status in result.items]
        
        return EnhancedStatusListResponse(
            statuses=status_responses,
            total_count=result.total_count,
            page=result.page,
            page_size=result.page_size,
            total_pages=result.total_pages,
            has_next=result.has_next,
            has_previous=result.has_previous,
            filters_applied=result.filters_applied,
            query_time_ms=result.query_time_ms,
            metadata={**result.metadata, 'preset_type': preset_type.value}
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting preset filter: {str(e)}"
        )


# Search statuses with full-text search
@router.get("/search", response_model=EnhancedStatusListResponse)
def search_statuses(
    q: str = Query(..., description="Search query"),
    fields: List[str] = Query(
        default=['current_step', 'error_info', 'tags'],
        description="Fields to search in"
    ),
    exact_match: bool = Query(False, description="Exact match search"),
    case_sensitive: bool = Query(False, description="Case sensitive search"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Page size"),
    filter_service: StatusFilterService = Depends(get_filter_service)
):
    """
    Search processing statuses using full-text search.
    
    Searches across specified fields with support for:
    - Exact or partial matching
    - Case sensitive/insensitive search
    - Multiple field search
    """
    try:
        # Create search parameters
        search = SearchParams(
            query=q,
            fields=fields,
            exact_match=exact_match,
            case_sensitive=case_sensitive
        )
        
        pagination = PaginationParams(page=page, page_size=page_size)
        
        # Create filter query with search
        filter_query = FilterQuery(
            search=search,
            pagination=pagination,
            sorts=[SortCondition('updated_at', SortOrder.DESC)]
        )
        
        # Execute search
        result = filter_service.filter_statuses(filter_query)
        
        # Convert to response models
        status_responses = [StatusResponse.from_orm(status) for status in result.items]
        
        return EnhancedStatusListResponse(
            statuses=status_responses,
            total_count=result.total_count,
            page=result.page,
            page_size=result.page_size,
            total_pages=result.total_pages,
            has_next=result.has_next,
            has_previous=result.has_previous,
            filters_applied=result.filters_applied,
            query_time_ms=result.query_time_ms,
            metadata={
                **result.metadata,
                'search_query': q,
                'search_fields': fields
            }
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error searching statuses: {str(e)}"
        )


# Get aggregated statistics
@router.get("/aggregates", response_model=StatusAggregatesResponse)
def get_status_aggregates(
    start_date: Optional[datetime] = Query(None, description="Start date for filtering"),
    end_date: Optional[datetime] = Query(None, description="End date for filtering"),
    filter_service: StatusFilterService = Depends(get_filter_service)
):
    """
    Get aggregated statistics for processing statuses.
    
    Provides comprehensive statistics including:
    - Status distribution
    - Priority distribution
    - Progress statistics
    - Worker statistics
    - Time-based statistics
    """
    try:
        # Create date range if provided
        date_range = None
        if start_date and end_date:
            date_range = (start_date, end_date)
        elif start_date:
            date_range = (start_date, datetime.utcnow())
        elif end_date:
            # Default to last 30 days if only end_date provided
            date_range = (end_date - timedelta(days=30), end_date)
        
        # Get aggregates
        aggregates = filter_service.get_status_aggregates(date_range=date_range)
        
        return StatusAggregatesResponse(**aggregates)
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting status aggregates: {str(e)}"
        )


# Enhanced status history with filtering
@router.post("/{status_id}/history/filter", response_model=List[StatusHistoryResponse])
def filter_status_history(
    status_id: str = Path(..., description="Status ID"),
    filter_request: Optional[AdvancedFilterRequest] = Body(None),
    filter_service: StatusFilterService = Depends(get_filter_service)
):
    """
    Get status history with advanced filtering and pagination.
    
    Provides filtered access to status change history with support for:
    - Time-based filtering
    - Change type filtering
    - Worker filtering
    - Full-text search in change reasons and metadata
    """
    try:
        filter_query = None
        if filter_request:
            # Convert request models to service models
            filters = []
            for filter_model in filter_request.filters:
                filters.append(FilterCondition(
                    field=filter_model.field,
                    operator=FilterOperator(filter_model.operator),
                    value=filter_model.value,
                    value2=filter_model.value2
                ))
            
            sorts = []
            for sort_model in filter_request.sorts:
                sorts.append(SortCondition(
                    field=sort_model.field,
                    order=SortOrder(sort_model.order)
                ))
            
            search = None
            if filter_request.search:
                search = SearchParams(
                    query=filter_request.search.query,
                    fields=filter_request.search.fields,
                    exact_match=filter_request.search.exact_match,
                    case_sensitive=filter_request.search.case_sensitive
                )
            
            pagination = PaginationParams(
                page=filter_request.page,
                page_size=filter_request.page_size
            )
            
            filter_query = FilterQuery(
                filters=filters,
                sorts=sorts,
                pagination=pagination,
                search=search,
                include_counts=filter_request.include_counts
            )
        
        # Get filtered history
        result = filter_service.filter_status_history(
            status_id=status_id,
            filter_query=filter_query
        )
        
        # Convert to response models
        return [StatusHistoryResponse.from_orm(history) for history in result.items]
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error filtering status history: {str(e)}"
        )


# Export available filter fields and operators
@router.get("/schema/fields")
def get_available_fields():
    """Get available fields for filtering and sorting."""
    return {
        "filterable_fields": [
            "status_id", "video_id", "batch_item_id", "processing_session_id",
            "status", "priority", "progress_percentage", "current_step",
            "total_steps", "completed_steps", "worker_id", "retry_count",
            "max_retries", "error_info", "created_at", "updated_at",
            "started_at", "completed_at", "heartbeat_at", "external_id", "tags"
        ],
        "sortable_fields": [
            "created_at", "updated_at", "started_at", "completed_at",
            "heartbeat_at", "progress_percentage", "retry_count", "priority"
        ],
        "searchable_fields": [
            "current_step", "error_info", "tags", "external_id"
        ]
    }


@router.get("/schema/operators")
def get_available_operators():
    """Get available filter operators."""
    return {
        "operators": [
            {"name": "eq", "description": "Equal"},
            {"name": "ne", "description": "Not equal"},
            {"name": "gt", "description": "Greater than"},
            {"name": "gte", "description": "Greater than or equal"},
            {"name": "lt", "description": "Less than"},
            {"name": "lte", "description": "Less than or equal"},
            {"name": "in", "description": "In list"},
            {"name": "not_in", "description": "Not in list"},
            {"name": "like", "description": "SQL LIKE pattern"},
            {"name": "ilike", "description": "Case-insensitive LIKE"},
            {"name": "is_null", "description": "Is NULL"},
            {"name": "is_not_null", "description": "Is not NULL"},
            {"name": "between", "description": "Between two values"},
            {"name": "contains", "description": "Array/JSON contains"}
        ],
        "sort_orders": ["asc", "desc"]
    }


# Bulk operations with filtering
@router.post("/bulk/cancel")
def bulk_cancel_statuses(
    filter_request: AdvancedFilterRequest,
    reason: str = Body(..., description="Cancellation reason"),
    filter_service: StatusFilterService = Depends(get_filter_service),
    status_service: StatusService = Depends(get_status_service)
):
    """
    Bulk cancel statuses matching filter criteria.
    
    This endpoint allows canceling multiple statuses that match
    the specified filter criteria with a single operation.
    """
    try:
        # Convert and execute filter
        # ... (similar conversion as in filter_statuses)
        
        # For safety, limit bulk operations
        if filter_request.page_size > 50:
            raise HTTPException(
                status_code=400,
                detail="Bulk operations limited to 50 items per request"
            )
        
        # This would be implemented with proper bulk operations
        # For now, return a placeholder response
        return {
            "message": "Bulk operation not yet implemented",
            "filters_would_apply_to": "TBD statuses",
            "reason": reason
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error in bulk cancel operation: {str(e)}"
        )