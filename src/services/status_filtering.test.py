"""
Tests for status filtering and pagination services.

This module tests the advanced filtering, sorting, and pagination
capabilities of the status tracking system.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from .status_filtering import (
    StatusFilterService, FilterQuery, FilterCondition, SortCondition,
    PaginationParams, SearchParams, FilterOperator, SortOrder,
    QuickFilterPresets, FilterResult
)
from ..database.status_models import ProcessingStatusType, ProcessingPriority


class TestFilterCondition:
    """Test cases for FilterCondition."""
    
    def test_filter_condition_creation(self):
        """Test creating filter conditions."""
        condition = FilterCondition(
            field="status",
            operator=FilterOperator.EQ,
            value=ProcessingStatusType.COMPLETED
        )
        
        assert condition.field == "status"
        assert condition.operator == FilterOperator.EQ
        assert condition.value == ProcessingStatusType.COMPLETED
        assert condition.value2 is None
    
    def test_between_operator_validation(self):
        """Test BETWEEN operator requires value2."""
        with pytest.raises(ValueError, match="BETWEEN operator requires value2"):
            FilterCondition(
                field="created_at",
                operator=FilterOperator.BETWEEN,
                value=datetime.utcnow()
                # Missing value2
            )
    
    def test_null_operators_validation(self):
        """Test IS_NULL operators should not have values."""
        with pytest.raises(ValueError, match="should not have a value"):
            FilterCondition(
                field="error_info",
                operator=FilterOperator.IS_NULL,
                value="should_not_be_here"
            )


class TestSortCondition:
    """Test cases for SortCondition."""
    
    def test_sort_condition_creation(self):
        """Test creating sort conditions."""
        condition = SortCondition(
            field="created_at",
            order=SortOrder.DESC
        )
        
        assert condition.field == "created_at"
        assert condition.order == SortOrder.DESC
    
    def test_string_sort_order_conversion(self):
        """Test converting string to SortOrder enum."""
        condition = SortCondition(field="updated_at", order="asc")
        assert condition.order == SortOrder.ASC
        
        condition = SortCondition(field="updated_at", order="DESC")
        assert condition.order == SortOrder.DESC
    
    def test_invalid_sort_order(self):
        """Test invalid sort order raises error."""
        with pytest.raises(ValueError, match="Invalid sort order"):
            SortCondition(field="created_at", order="invalid")


class TestPaginationParams:
    """Test cases for PaginationParams."""
    
    def test_pagination_params_defaults(self):
        """Test default pagination parameters."""
        params = PaginationParams()
        
        assert params.page == 1
        assert params.page_size == 20
        assert params.max_page_size == 100
        assert params.offset == 0
        assert params.limit == 20
    
    def test_pagination_params_validation(self):
        """Test pagination parameter validation."""
        # Test negative page
        params = PaginationParams(page=-1)
        assert params.page == 1
        
        # Test zero page_size
        params = PaginationParams(page_size=0)
        assert params.page_size == 20
        
        # Test page_size exceeding max
        params = PaginationParams(page_size=200, max_page_size=100)
        assert params.page_size == 100
    
    def test_offset_calculation(self):
        """Test offset calculation."""
        params = PaginationParams(page=3, page_size=25)
        assert params.offset == 50  # (3-1) * 25


class TestSearchParams:
    """Test cases for SearchParams."""
    
    def test_search_params_defaults(self):
        """Test default search parameters."""
        params = SearchParams(query="test")
        
        assert params.query == "test"
        assert params.fields == ['current_step', 'error_info', 'tags']
        assert params.exact_match is False
        assert params.case_sensitive is False
    
    def test_search_params_custom(self):
        """Test custom search parameters."""
        params = SearchParams(
            query="error",
            fields=["error_info"],
            exact_match=True,
            case_sensitive=True
        )
        
        assert params.query == "error"
        assert params.fields == ["error_info"]
        assert params.exact_match is True
        assert params.case_sensitive is True


class TestFilterQuery:
    """Test cases for FilterQuery."""
    
    def test_filter_query_defaults(self):
        """Test default filter query."""
        query = FilterQuery()
        
        assert query.filters == []
        assert query.sorts == []
        assert isinstance(query.pagination, PaginationParams)
        assert query.search is None
        assert query.include_counts is True
        assert query.include_related is False
    
    def test_filter_query_complete(self):
        """Test complete filter query."""
        filters = [FilterCondition("status", FilterOperator.EQ, ProcessingStatusType.COMPLETED)]
        sorts = [SortCondition("created_at", SortOrder.DESC)]
        pagination = PaginationParams(page=2, page_size=50)
        search = SearchParams("test query")
        
        query = FilterQuery(
            filters=filters,
            sorts=sorts,
            pagination=pagination,
            search=search,
            include_counts=False,
            include_related=True
        )
        
        assert query.filters == filters
        assert query.sorts == sorts
        assert query.pagination == pagination
        assert query.search == search
        assert query.include_counts is False
        assert query.include_related is True


class TestStatusFilterService:
    """Test cases for StatusFilterService."""
    
    @pytest.fixture
    def mock_db_session(self):
        """Create mock database session."""
        session = Mock()
        return session
    
    @pytest.fixture
    def filter_service(self, mock_db_session):
        """Create filter service with mock session."""
        return StatusFilterService(db_session=mock_db_session)
    
    def test_filter_service_initialization(self, filter_service):
        """Test filter service initialization."""
        assert filter_service.db_session is not None
        assert filter_service._should_close_session is False
        assert hasattr(filter_service, 'status_fields')
        assert 'status_id' in filter_service.status_fields
    
    def test_context_manager(self):
        """Test filter service as context manager."""
        with patch('src.services.status_filtering.get_db_session') as mock_get_session:
            mock_session = Mock()
            mock_get_session.return_value = mock_session
            
            with StatusFilterService() as service:
                assert service.db_session == mock_session
    
    @patch('src.services.status_filtering.get_db_session')
    def test_filter_statuses(self, mock_get_session, filter_service):
        """Test filtering statuses."""
        # Mock query results
        mock_query = Mock()
        mock_query.count.return_value = 5
        mock_query.all.return_value = []
        mock_query.offset.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.filter.return_value = mock_query
        
        filter_service.db_session.query.return_value = mock_query
        
        # Create filter query
        filter_query = FilterQuery(
            filters=[FilterCondition("status", FilterOperator.EQ, ProcessingStatusType.COMPLETED)],
            pagination=PaginationParams(page=1, page_size=20)
        )
        
        # Execute filter
        result = filter_service.filter_statuses(filter_query)
        
        # Verify result
        assert isinstance(result, FilterResult)
        assert result.total_count == 5
        assert result.page == 1
        assert result.page_size == 20
        assert result.items == []
        assert result.filters_applied == 1
        assert result.query_time_ms >= 0
    
    def test_apply_single_filter_eq(self, filter_service):
        """Test applying EQ filter."""
        mock_query = Mock()
        mock_field = Mock()
        filter_service.status_fields['test_field'] = mock_field
        
        condition = FilterCondition("test_field", FilterOperator.EQ, "test_value")
        result = filter_service._apply_single_filter(mock_query, condition)
        
        mock_query.filter.assert_called_once()
    
    def test_apply_single_filter_between(self, filter_service):
        """Test applying BETWEEN filter."""
        mock_query = Mock()
        mock_field = Mock()
        filter_service.status_fields['test_field'] = mock_field
        
        condition = FilterCondition(
            "test_field", 
            FilterOperator.BETWEEN, 
            "value1", 
            "value2"
        )
        result = filter_service._apply_single_filter(mock_query, condition)
        
        mock_query.filter.assert_called_once()
    
    def test_apply_single_filter_unknown_field(self, filter_service):
        """Test applying filter with unknown field."""
        mock_query = Mock()
        
        condition = FilterCondition("unknown_field", FilterOperator.EQ, "test_value")
        result = filter_service._apply_single_filter(mock_query, condition)
        
        # Should return original query unchanged
        assert result == mock_query
        mock_query.filter.assert_not_called()
    
    def test_apply_search(self, filter_service):
        """Test applying search conditions."""
        mock_query = Mock()
        mock_field = Mock()
        filter_service.status_fields['current_step'] = mock_field
        
        search = SearchParams(
            query="test query",
            fields=['current_step'],
            exact_match=False,
            case_sensitive=False
        )
        
        result = filter_service._apply_search(mock_query, search)
        
        mock_query.filter.assert_called_once()
    
    def test_apply_sorting_default(self, filter_service):
        """Test applying default sorting."""
        mock_query = Mock()
        
        result = filter_service._apply_sorting(mock_query, [])
        
        # Should apply default sort by updated_at desc
        mock_query.order_by.assert_called_once()
    
    def test_apply_sorting_custom(self, filter_service):
        """Test applying custom sorting."""
        mock_query = Mock()
        mock_field = Mock()
        filter_service.status_fields['created_at'] = mock_field
        
        sorts = [SortCondition("created_at", SortOrder.ASC)]
        result = filter_service._apply_sorting(mock_query, sorts)
        
        mock_query.order_by.assert_called_once()
    
    def test_apply_pagination(self, filter_service):
        """Test applying pagination."""
        mock_query = Mock()
        mock_query.offset.return_value = mock_query
        
        pagination = PaginationParams(page=3, page_size=25)
        result = filter_service._apply_pagination(mock_query, pagination)
        
        mock_query.offset.assert_called_with(50)  # (3-1) * 25
        mock_query.limit.assert_called_with(25)
    
    @patch('src.services.status_filtering.get_db_session')
    def test_get_status_aggregates(self, mock_get_session, filter_service):
        """Test getting status aggregates."""
        # Mock query results
        mock_query = Mock()
        mock_query.count.return_value = 100
        mock_query.filter.return_value = mock_query
        mock_query.with_entities.return_value = mock_query
        mock_query.group_by.return_value = mock_query
        mock_query.all.return_value = [
            (ProcessingStatusType.COMPLETED, 80),
            (ProcessingStatusType.FAILED, 20)
        ]
        mock_query.first.return_value = Mock(
            avg_progress=75.5,
            min_progress=0.0,
            max_progress=100.0,
            unique_workers=5,
            total_with_workers=95
        )
        
        filter_service.db_session.query.return_value = mock_query
        
        # Get aggregates
        result = filter_service.get_status_aggregates()
        
        # Verify result structure
        assert 'total_count' in result
        assert 'status_distribution' in result
        assert 'priority_distribution' in result
        assert 'progress_statistics' in result
        assert 'worker_statistics' in result
        assert 'time_statistics' in result
        assert 'timestamp' in result


class TestQuickFilterPresets:
    """Test cases for QuickFilterPresets."""
    
    def test_active_statuses_preset(self):
        """Test active statuses preset."""
        query = QuickFilterPresets.active_statuses()
        
        assert len(query.filters) == 1
        assert query.filters[0].field == 'status'
        assert query.filters[0].operator == FilterOperator.IN
        assert ProcessingStatusType.STARTING in query.filters[0].value
        assert ProcessingStatusType.COMPLETED not in query.filters[0].value
        
        assert len(query.sorts) == 1
        assert query.sorts[0].field == 'updated_at'
        assert query.sorts[0].order == SortOrder.DESC
    
    def test_failed_statuses_preset(self):
        """Test failed statuses preset."""
        query = QuickFilterPresets.failed_statuses(last_hours=12)
        
        assert len(query.filters) == 2
        
        # Check status filter
        status_filter = query.filters[0]
        assert status_filter.field == 'status'
        assert status_filter.operator == FilterOperator.EQ
        assert status_filter.value == ProcessingStatusType.FAILED
        
        # Check time filter
        time_filter = query.filters[1]
        assert time_filter.field == 'updated_at'
        assert time_filter.operator == FilterOperator.GTE
        assert isinstance(time_filter.value, datetime)
    
    def test_completed_today_preset(self):
        """Test completed today preset."""
        query = QuickFilterPresets.completed_today()
        
        assert len(query.filters) == 2
        
        # Check status filter
        status_filter = query.filters[0]
        assert status_filter.field == 'status'
        assert status_filter.value == ProcessingStatusType.COMPLETED
        
        # Check date filter
        date_filter = query.filters[1]
        assert date_filter.field == 'completed_at'
        assert date_filter.operator == FilterOperator.GTE
    
    def test_high_priority_preset(self):
        """Test high priority preset."""
        query = QuickFilterPresets.high_priority()
        
        assert len(query.filters) == 1
        assert query.filters[0].field == 'priority'
        assert query.filters[0].value == ProcessingPriority.HIGH
    
    def test_stale_statuses_preset(self):
        """Test stale statuses preset."""
        query = QuickFilterPresets.stale_statuses(timeout_minutes=45)
        
        assert len(query.filters) == 2
        
        # Check status filter (active statuses)
        status_filter = query.filters[0]
        assert status_filter.field == 'status'
        assert status_filter.operator == FilterOperator.IN
        
        # Check heartbeat filter
        heartbeat_filter = query.filters[1]
        assert heartbeat_filter.field == 'heartbeat_at'
        assert heartbeat_filter.operator == FilterOperator.LT
        assert isinstance(heartbeat_filter.value, datetime)
    
    def test_by_worker_preset(self):
        """Test by worker preset."""
        worker_id = "worker_123"
        query = QuickFilterPresets.by_worker(worker_id)
        
        assert len(query.filters) == 1
        assert query.filters[0].field == 'worker_id'
        assert query.filters[0].operator == FilterOperator.EQ
        assert query.filters[0].value == worker_id
    
    def test_date_range_preset(self):
        """Test date range preset."""
        start_date = datetime.utcnow() - timedelta(days=7)
        end_date = datetime.utcnow()
        
        query = QuickFilterPresets.date_range(start_date, end_date)
        
        assert len(query.filters) == 1
        assert query.filters[0].field == 'created_at'
        assert query.filters[0].operator == FilterOperator.BETWEEN
        assert query.filters[0].value == start_date
        assert query.filters[0].value2 == end_date


if __name__ == "__main__":
    pytest.main([__file__])