"""
Advanced filtering and pagination services for status tracking.

This module provides comprehensive filtering, sorting, and pagination
capabilities for the status tracking system with support for complex
queries and performance optimization.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from sqlalchemy.orm import Session, Query
from sqlalchemy import and_, or_, not_, desc, asc, func, text
from sqlalchemy.sql import operators

from ..database.connection import get_db_session
from ..database.status_models import (
    ProcessingStatus, StatusHistory, StatusMetrics,
    ProcessingStatusType, ProcessingPriority, StatusChangeType
)
from ..database.models import Video
from ..database.batch_models import BatchItem, ProcessingSession


logger = logging.getLogger(__name__)


class SortOrder(Enum):
    """Sort order enumeration."""
    ASC = "asc"
    DESC = "desc"


class FilterOperator(Enum):
    """Filter operator enumeration."""
    EQ = "eq"          # Equal
    NE = "ne"          # Not equal
    GT = "gt"          # Greater than
    GTE = "gte"        # Greater than or equal
    LT = "lt"          # Less than
    LTE = "lte"        # Less than or equal
    IN = "in"          # In list
    NOT_IN = "not_in"  # Not in list
    LIKE = "like"      # SQL LIKE pattern
    ILIKE = "ilike"    # Case-insensitive LIKE
    IS_NULL = "is_null"        # Is NULL
    IS_NOT_NULL = "is_not_null" # Is not NULL
    BETWEEN = "between"        # Between two values
    CONTAINS = "contains"      # Array/JSON contains
    REGEX = "regex"           # Regular expression match


@dataclass
class FilterCondition:
    """Represents a single filter condition."""
    field: str
    operator: FilterOperator
    value: Any = None
    value2: Any = None  # For BETWEEN operator
    
    def __post_init__(self):
        """Validate filter condition."""
        if self.operator == FilterOperator.BETWEEN and self.value2 is None:
            raise ValueError("BETWEEN operator requires value2")
        
        if self.operator in [FilterOperator.IS_NULL, FilterOperator.IS_NOT_NULL] and self.value is not None:
            raise ValueError(f"{self.operator.value} operator should not have a value")


@dataclass
class SortCondition:
    """Represents a sorting condition."""
    field: str
    order: SortOrder = SortOrder.DESC
    
    def __post_init__(self):
        """Validate sort condition."""
        if not isinstance(self.order, SortOrder):
            if isinstance(self.order, str):
                self.order = SortOrder(self.order.lower())
            else:
                raise ValueError(f"Invalid sort order: {self.order}")


@dataclass
class PaginationParams:
    """Pagination parameters."""
    page: int = 1
    page_size: int = 20
    max_page_size: int = 100
    
    def __post_init__(self):
        """Validate pagination parameters."""
        if self.page < 1:
            self.page = 1
        if self.page_size < 1:
            self.page_size = 20
        if self.page_size > self.max_page_size:
            self.page_size = self.max_page_size
    
    @property
    def offset(self) -> int:
        """Calculate offset for database query."""
        return (self.page - 1) * self.page_size
    
    @property
    def limit(self) -> int:
        """Get limit for database query."""
        return self.page_size


@dataclass
class SearchParams:
    """Search parameters for full-text search."""
    query: str
    fields: List[str] = field(default_factory=lambda: ['current_step', 'error_info', 'tags'])
    exact_match: bool = False
    case_sensitive: bool = False


@dataclass
class FilterQuery:
    """Complete filter query specification."""
    filters: List[FilterCondition] = field(default_factory=list)
    sorts: List[SortCondition] = field(default_factory=list)
    pagination: PaginationParams = field(default_factory=PaginationParams)
    search: Optional[SearchParams] = None
    include_counts: bool = True
    include_related: bool = False


@dataclass
class FilterResult:
    """Result of a filtered query."""
    items: List[Any]
    total_count: int
    page: int
    page_size: int
    total_pages: int
    has_next: bool
    has_previous: bool
    filters_applied: int
    query_time_ms: float
    metadata: Optional[Dict[str, Any]] = None


class StatusFilterService:
    """
    Advanced filtering and pagination service for processing statuses.
    
    Provides comprehensive filtering, sorting, and pagination capabilities
    with performance optimization and complex query support.
    """
    
    def __init__(self, db_session: Optional[Session] = None):
        """Initialize the filter service."""
        self.db_session = db_session
        self._should_close_session = db_session is None
        self.logger = logging.getLogger(f"{__name__}.StatusFilterService")
        
        # Field mapping for status table
        self.status_fields = {
            'status_id': ProcessingStatus.status_id,
            'video_id': ProcessingStatus.video_id,
            'batch_item_id': ProcessingStatus.batch_item_id,
            'processing_session_id': ProcessingStatus.processing_session_id,
            'status': ProcessingStatus.status,
            'priority': ProcessingStatus.priority,
            'progress_percentage': ProcessingStatus.progress_percentage,
            'current_step': ProcessingStatus.current_step,
            'total_steps': ProcessingStatus.total_steps,
            'completed_steps': ProcessingStatus.completed_steps,
            'worker_id': ProcessingStatus.worker_id,
            'retry_count': ProcessingStatus.retry_count,
            'max_retries': ProcessingStatus.max_retries,
            'error_info': ProcessingStatus.error_info,
            'created_at': ProcessingStatus.created_at,
            'updated_at': ProcessingStatus.updated_at,
            'started_at': ProcessingStatus.started_at,
            'completed_at': ProcessingStatus.completed_at,
            'heartbeat_at': ProcessingStatus.heartbeat_at,
            'external_id': ProcessingStatus.external_id,
            'tags': ProcessingStatus.tags,
            'processing_metadata': ProcessingStatus.processing_metadata,
            'result_metadata': ProcessingStatus.result_metadata
        }
    
    def __enter__(self):
        """Context manager entry."""
        if self.db_session is None:
            self.db_session = get_db_session()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self._should_close_session and self.db_session:
            self.db_session.close()
    
    def filter_statuses(self, filter_query: FilterQuery) -> FilterResult:
        """
        Filter processing statuses with comprehensive filtering and pagination.
        
        Args:
            filter_query: Complete filter specification
            
        Returns:
            FilterResult with items and metadata
        """
        start_time = datetime.utcnow()
        
        try:
            # Build base query
            query = self.db_session.query(ProcessingStatus)
            
            # Apply filters
            query = self._apply_filters(query, filter_query.filters)
            
            # Apply search
            if filter_query.search:
                query = self._apply_search(query, filter_query.search)
            
            # Get total count before pagination
            total_count = query.count() if filter_query.include_counts else 0
            
            # Apply sorting
            query = self._apply_sorting(query, filter_query.sorts)
            
            # Apply pagination
            paginated_query = self._apply_pagination(query, filter_query.pagination)
            
            # Execute query
            items = paginated_query.all()
            
            # Calculate pagination metadata
            total_pages = (total_count + filter_query.pagination.page_size - 1) // filter_query.pagination.page_size
            has_next = filter_query.pagination.page < total_pages
            has_previous = filter_query.pagination.page > 1
            
            # Calculate query time
            query_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return FilterResult(
                items=items,
                total_count=total_count,
                page=filter_query.pagination.page,
                page_size=filter_query.pagination.page_size,
                total_pages=total_pages,
                has_next=has_next,
                has_previous=has_previous,
                filters_applied=len(filter_query.filters),
                query_time_ms=query_time_ms,
                metadata={
                    'search_applied': filter_query.search is not None,
                    'sorts_applied': len(filter_query.sorts),
                    'include_counts': filter_query.include_counts
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error filtering statuses: {e}")
            raise
    
    def filter_status_history(
        self, 
        status_id: Optional[str] = None,
        filter_query: Optional[FilterQuery] = None
    ) -> FilterResult:
        """
        Filter status history with advanced filtering.
        
        Args:
            status_id: Optional status ID to filter by
            filter_query: Advanced filter specification
            
        Returns:
            FilterResult with history items
        """
        start_time = datetime.utcnow()
        
        try:
            # Build base query
            if status_id:
                # Get status first
                status = self.db_session.query(ProcessingStatus).filter(
                    ProcessingStatus.status_id == status_id
                ).first()
                
                if not status:
                    raise ValueError(f"Status not found: {status_id}")
                
                query = self.db_session.query(StatusHistory).filter(
                    StatusHistory.processing_status_id == status.id
                )
            else:
                query = self.db_session.query(StatusHistory)
            
            # Apply advanced filtering if provided
            if filter_query:
                # Map filters to history fields
                history_filters = self._map_history_filters(filter_query.filters)
                query = self._apply_history_filters(query, history_filters)
                
                # Apply search
                if filter_query.search:
                    query = self._apply_history_search(query, filter_query.search)
                
                # Get total count
                total_count = query.count() if filter_query.include_counts else 0
                
                # Apply sorting
                query = self._apply_history_sorting(query, filter_query.sorts)
                
                # Apply pagination
                query = self._apply_pagination(query, filter_query.pagination)
                
                pagination = filter_query.pagination
            else:
                # Default behavior
                total_count = query.count()
                query = query.order_by(desc(StatusHistory.created_at))
                pagination = PaginationParams()
            
            # Execute query
            items = query.all()
            
            # Calculate pagination metadata
            total_pages = (total_count + pagination.page_size - 1) // pagination.page_size
            has_next = pagination.page < total_pages
            has_previous = pagination.page > 1
            
            # Calculate query time
            query_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return FilterResult(
                items=items,
                total_count=total_count,
                page=pagination.page,
                page_size=pagination.page_size,
                total_pages=total_pages,
                has_next=has_next,
                has_previous=has_previous,
                filters_applied=len(filter_query.filters) if filter_query else 0,
                query_time_ms=query_time_ms
            )
            
        except Exception as e:
            self.logger.error(f"Error filtering status history: {e}")
            raise
    
    def get_status_aggregates(
        self,
        filters: Optional[List[FilterCondition]] = None,
        group_by: Optional[List[str]] = None,
        date_range: Optional[Tuple[datetime, datetime]] = None
    ) -> Dict[str, Any]:
        """
        Get aggregated statistics for statuses with filtering.
        
        Args:
            filters: Optional filters to apply
            group_by: Optional fields to group by
            date_range: Optional date range filter
            
        Returns:
            Dictionary with aggregated statistics
        """
        try:
            # Build base query
            query = self.db_session.query(ProcessingStatus)
            
            # Apply filters
            if filters:
                query = self._apply_filters(query, filters)
            
            # Apply date range filter
            if date_range:
                start_date, end_date = date_range
                query = query.filter(
                    ProcessingStatus.created_at.between(start_date, end_date)
                )
            
            # Calculate aggregates
            total_count = query.count()
            
            # Status distribution
            status_counts = (
                query.with_entities(
                    ProcessingStatus.status,
                    func.count(ProcessingStatus.id).label('count')
                )
                .group_by(ProcessingStatus.status)
                .all()
            )
            
            # Priority distribution
            priority_counts = (
                query.with_entities(
                    ProcessingStatus.priority,
                    func.count(ProcessingStatus.id).label('count')
                )
                .group_by(ProcessingStatus.priority)
                .all()
            )
            
            # Progress statistics
            progress_stats = query.with_entities(
                func.avg(ProcessingStatus.progress_percentage).label('avg_progress'),
                func.min(ProcessingStatus.progress_percentage).label('min_progress'),
                func.max(ProcessingStatus.progress_percentage).label('max_progress')
            ).first()
            
            # Worker statistics
            worker_stats = (
                query.filter(ProcessingStatus.worker_id.isnot(None))
                .with_entities(
                    func.count(func.distinct(ProcessingStatus.worker_id)).label('unique_workers'),
                    func.count(ProcessingStatus.id).label('total_with_workers')
                )
                .first()
            )
            
            # Time-based statistics
            now = datetime.utcnow()
            time_stats = {
                'last_hour': query.filter(
                    ProcessingStatus.updated_at >= now - timedelta(hours=1)
                ).count(),
                'last_24_hours': query.filter(
                    ProcessingStatus.updated_at >= now - timedelta(days=1)
                ).count(),
                'last_7_days': query.filter(
                    ProcessingStatus.updated_at >= now - timedelta(days=7)
                ).count()
            }
            
            return {
                'total_count': total_count,
                'status_distribution': {
                    status.value: count for status, count in status_counts
                },
                'priority_distribution': {
                    priority.value: count for priority, count in priority_counts
                },
                'progress_statistics': {
                    'average': float(progress_stats.avg_progress or 0),
                    'minimum': float(progress_stats.min_progress or 0),
                    'maximum': float(progress_stats.max_progress or 0)
                },
                'worker_statistics': {
                    'unique_workers': worker_stats.unique_workers or 0,
                    'total_with_workers': worker_stats.total_with_workers or 0
                },
                'time_statistics': time_stats,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting status aggregates: {e}")
            raise
    
    def _apply_filters(self, query: Query, filters: List[FilterCondition]) -> Query:
        """Apply filter conditions to query."""
        for filter_condition in filters:
            query = self._apply_single_filter(query, filter_condition)
        return query
    
    def _apply_single_filter(self, query: Query, filter_condition: FilterCondition) -> Query:
        """Apply a single filter condition to query."""
        try:
            # Get field reference
            field = self.status_fields.get(filter_condition.field)
            if field is None:
                self.logger.warning(f"Unknown filter field: {filter_condition.field}")
                return query
            
            # Apply operator
            if filter_condition.operator == FilterOperator.EQ:
                return query.filter(field == filter_condition.value)
            elif filter_condition.operator == FilterOperator.NE:
                return query.filter(field != filter_condition.value)
            elif filter_condition.operator == FilterOperator.GT:
                return query.filter(field > filter_condition.value)
            elif filter_condition.operator == FilterOperator.GTE:
                return query.filter(field >= filter_condition.value)
            elif filter_condition.operator == FilterOperator.LT:
                return query.filter(field < filter_condition.value)
            elif filter_condition.operator == FilterOperator.LTE:
                return query.filter(field <= filter_condition.value)
            elif filter_condition.operator == FilterOperator.IN:
                return query.filter(field.in_(filter_condition.value))
            elif filter_condition.operator == FilterOperator.NOT_IN:
                return query.filter(~field.in_(filter_condition.value))
            elif filter_condition.operator == FilterOperator.LIKE:
                return query.filter(field.like(filter_condition.value))
            elif filter_condition.operator == FilterOperator.ILIKE:
                return query.filter(field.ilike(filter_condition.value))
            elif filter_condition.operator == FilterOperator.IS_NULL:
                return query.filter(field.is_(None))
            elif filter_condition.operator == FilterOperator.IS_NOT_NULL:
                return query.filter(field.isnot(None))
            elif filter_condition.operator == FilterOperator.BETWEEN:
                return query.filter(field.between(filter_condition.value, filter_condition.value2))
            elif filter_condition.operator == FilterOperator.CONTAINS:
                # For JSON/Array fields
                if hasattr(field.type, 'python_type'):
                    return query.filter(field.contains(filter_condition.value))
                else:
                    self.logger.warning(f"CONTAINS operator not supported for field: {filter_condition.field}")
                    return query
            else:
                self.logger.warning(f"Unsupported filter operator: {filter_condition.operator}")
                return query
                
        except Exception as e:
            self.logger.error(f"Error applying filter {filter_condition}: {e}")
            return query
    
    def _apply_search(self, query: Query, search: SearchParams) -> Query:
        """Apply search conditions to query."""
        if not search.query:
            return query
        
        try:
            search_conditions = []
            
            for field_name in search.fields:
                field = self.status_fields.get(field_name)
                if field is None:
                    continue
                
                if search.exact_match:
                    if search.case_sensitive:
                        search_conditions.append(field == search.query)
                    else:
                        search_conditions.append(func.lower(field) == search.query.lower())
                else:
                    search_pattern = f"%{search.query}%"
                    if search.case_sensitive:
                        search_conditions.append(field.like(search_pattern))
                    else:
                        search_conditions.append(field.ilike(search_pattern))
            
            if search_conditions:
                return query.filter(or_(*search_conditions))
            
            return query
            
        except Exception as e:
            self.logger.error(f"Error applying search: {e}")
            return query
    
    def _apply_sorting(self, query: Query, sorts: List[SortCondition]) -> Query:
        """Apply sorting conditions to query."""
        if not sorts:
            # Default sort by updated_at desc
            return query.order_by(desc(ProcessingStatus.updated_at))
        
        for sort_condition in sorts:
            field = self.status_fields.get(sort_condition.field)
            if field is None:
                self.logger.warning(f"Unknown sort field: {sort_condition.field}")
                continue
            
            if sort_condition.order == SortOrder.ASC:
                query = query.order_by(asc(field))
            else:
                query = query.order_by(desc(field))
        
        return query
    
    def _apply_pagination(self, query: Query, pagination: PaginationParams) -> Query:
        """Apply pagination to query."""
        return query.offset(pagination.offset).limit(pagination.limit)
    
    def _map_history_filters(self, filters: List[FilterCondition]) -> List[FilterCondition]:
        """Map status filters to history table fields."""
        # This would map field names from status to history table
        # For now, return as-is assuming correct field names
        return filters
    
    def _apply_history_filters(self, query: Query, filters: List[FilterCondition]) -> Query:
        """Apply filters to status history query."""
        # Similar to _apply_filters but for StatusHistory table
        # Implementation would be similar to _apply_filters
        return query
    
    def _apply_history_search(self, query: Query, search: SearchParams) -> Query:
        """Apply search to status history query."""
        # Similar to _apply_search but for StatusHistory table
        # Implementation would be similar to _apply_search
        return query
    
    def _apply_history_sorting(self, query: Query, sorts: List[SortCondition]) -> Query:
        """Apply sorting to status history query."""
        # Similar to _apply_sorting but for StatusHistory table
        # Implementation would be similar to _apply_sorting
        return query


class QuickFilterPresets:
    """Predefined filter presets for common use cases."""
    
    @staticmethod
    def active_statuses() -> FilterQuery:
        """Filter for active processing statuses."""
        return FilterQuery(
            filters=[
                FilterCondition(
                    field='status',
                    operator=FilterOperator.IN,
                    value=[
                        ProcessingStatusType.STARTING,
                        ProcessingStatusType.YOUTUBE_METADATA,
                        ProcessingStatusType.TRANSCRIPT_EXTRACTION,
                        ProcessingStatusType.LANGUAGE_DETECTION,
                        ProcessingStatusType.SUMMARY_GENERATION,
                        ProcessingStatusType.KEYWORD_EXTRACTION,
                        ProcessingStatusType.TIMESTAMPED_SEGMENTS,
                        ProcessingStatusType.FINALIZING
                    ]
                )
            ],
            sorts=[SortCondition('updated_at', SortOrder.DESC)]
        )
    
    @staticmethod
    def failed_statuses(last_hours: int = 24) -> FilterQuery:
        """Filter for failed statuses in the last N hours."""
        cutoff_time = datetime.utcnow() - timedelta(hours=last_hours)
        
        return FilterQuery(
            filters=[
                FilterCondition(
                    field='status',
                    operator=FilterOperator.EQ,
                    value=ProcessingStatusType.FAILED
                ),
                FilterCondition(
                    field='updated_at',
                    operator=FilterOperator.GTE,
                    value=cutoff_time
                )
            ],
            sorts=[SortCondition('updated_at', SortOrder.DESC)]
        )
    
    @staticmethod
    def completed_today() -> FilterQuery:
        """Filter for statuses completed today."""
        today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        
        return FilterQuery(
            filters=[
                FilterCondition(
                    field='status',
                    operator=FilterOperator.EQ,
                    value=ProcessingStatusType.COMPLETED
                ),
                FilterCondition(
                    field='completed_at',
                    operator=FilterOperator.GTE,
                    value=today
                )
            ],
            sorts=[SortCondition('completed_at', SortOrder.DESC)]
        )
    
    @staticmethod
    def high_priority() -> FilterQuery:
        """Filter for high priority statuses."""
        return FilterQuery(
            filters=[
                FilterCondition(
                    field='priority',
                    operator=FilterOperator.EQ,
                    value=ProcessingPriority.HIGH
                )
            ],
            sorts=[SortCondition('created_at', SortOrder.DESC)]
        )
    
    @staticmethod
    def stale_statuses(timeout_minutes: int = 30) -> FilterQuery:
        """Filter for stale statuses."""
        cutoff_time = datetime.utcnow() - timedelta(minutes=timeout_minutes)
        
        return FilterQuery(
            filters=[
                FilterCondition(
                    field='status',
                    operator=FilterOperator.IN,
                    value=[
                        ProcessingStatusType.STARTING,
                        ProcessingStatusType.YOUTUBE_METADATA,
                        ProcessingStatusType.TRANSCRIPT_EXTRACTION,
                        ProcessingStatusType.LANGUAGE_DETECTION,
                        ProcessingStatusType.SUMMARY_GENERATION,
                        ProcessingStatusType.KEYWORD_EXTRACTION,
                        ProcessingStatusType.TIMESTAMPED_SEGMENTS,
                        ProcessingStatusType.FINALIZING
                    ]
                ),
                FilterCondition(
                    field='heartbeat_at',
                    operator=FilterOperator.LT,
                    value=cutoff_time
                )
            ],
            sorts=[SortCondition('heartbeat_at', SortOrder.ASC)]
        )
    
    @staticmethod
    def by_worker(worker_id: str) -> FilterQuery:
        """Filter for statuses by specific worker."""
        return FilterQuery(
            filters=[
                FilterCondition(
                    field='worker_id',
                    operator=FilterOperator.EQ,
                    value=worker_id
                )
            ],
            sorts=[SortCondition('updated_at', SortOrder.DESC)]
        )
    
    @staticmethod
    def date_range(start_date: datetime, end_date: datetime) -> FilterQuery:
        """Filter for statuses within date range."""
        return FilterQuery(
            filters=[
                FilterCondition(
                    field='created_at',
                    operator=FilterOperator.BETWEEN,
                    value=start_date,
                    value2=end_date
                )
            ],
            sorts=[SortCondition('created_at', SortOrder.DESC)]
        )