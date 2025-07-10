# Status Tracking Filtering and Pagination Guide

This guide demonstrates how to use the advanced filtering, sorting, and pagination capabilities of the status tracking system.

## Overview

The enhanced status tracking API provides comprehensive filtering capabilities through the `/api/status/enhanced` endpoints. These endpoints support:

- **Complex Filtering**: Multiple filter conditions with various operators
- **Multi-field Sorting**: Sort by multiple fields with different orders
- **Full-text Search**: Search across specified fields
- **Pagination**: Efficient pagination with metadata
- **Preset Filters**: Common filter presets for quick access
- **Aggregations**: Statistical summaries and distributions

## Basic Usage

### Simple Status List with Pagination

```http
GET /api/status/enhanced/preset/active?page=1&page_size=20
```

### Advanced Filtering

```http
POST /api/status/enhanced/filter
Content-Type: application/json

{
  "filters": [
    {
      "field": "status",
      "operator": "in",
      "value": ["starting", "youtube_metadata", "transcript_extraction"]
    },
    {
      "field": "created_at",
      "operator": "gte",
      "value": "2023-01-01T00:00:00Z"
    }
  ],
  "sorts": [
    {
      "field": "updated_at",
      "order": "desc"
    }
  ],
  "page": 1,
  "page_size": 50
}
```

## Filter Operators

### Comparison Operators
- `eq`: Equal to
- `ne`: Not equal to
- `gt`: Greater than
- `gte`: Greater than or equal to
- `lt`: Less than
- `lte`: Less than or equal to

### List Operators
- `in`: Value in list
- `not_in`: Value not in list

### Text Operators
- `like`: SQL LIKE pattern matching
- `ilike`: Case-insensitive LIKE pattern matching

### Null Operators
- `is_null`: Field is NULL
- `is_not_null`: Field is not NULL

### Range Operators
- `between`: Value between two values (requires `value2`)

### JSON/Array Operators
- `contains`: JSON field contains value

## Available Fields

### Filterable Fields
- `status_id`: Unique status identifier
- `video_id`: Associated video ID
- `batch_item_id`: Associated batch item ID
- `processing_session_id`: Processing session ID
- `status`: Processing status value
- `priority`: Processing priority
- `progress_percentage`: Progress percentage (0-100)
- `current_step`: Current processing step
- `total_steps`: Total number of steps
- `completed_steps`: Number of completed steps
- `worker_id`: Worker identifier
- `retry_count`: Number of retries
- `max_retries`: Maximum retry limit
- `error_info`: Error information
- `created_at`: Creation timestamp
- `updated_at`: Last update timestamp
- `started_at`: Processing start timestamp
- `completed_at`: Processing completion timestamp
- `heartbeat_at`: Last heartbeat timestamp
- `external_id`: External identifier
- `tags`: Associated tags

### Searchable Fields
- `current_step`: Current processing step description
- `error_info`: Error information and messages
- `tags`: Associated tags
- `external_id`: External identifiers

## Example Queries

### 1. Find Failed Statuses from Last 24 Hours

```json
{
  "filters": [
    {
      "field": "status",
      "operator": "eq",
      "value": "failed"
    },
    {
      "field": "updated_at",
      "operator": "gte",
      "value": "2023-12-01T00:00:00Z"
    }
  ],
  "sorts": [
    {
      "field": "updated_at",
      "order": "desc"
    }
  ]
}
```

### 2. Find High Priority Statuses in Progress

```json
{
  "filters": [
    {
      "field": "priority",
      "operator": "eq",
      "value": "high"
    },
    {
      "field": "status",
      "operator": "in",
      "value": ["starting", "youtube_metadata", "transcript_extraction", "summary_generation"]
    }
  ],
  "sorts": [
    {
      "field": "created_at",
      "order": "asc"
    }
  ]
}
```

### 3. Find Statuses by Specific Worker

```json
{
  "filters": [
    {
      "field": "worker_id",
      "operator": "eq",
      "value": "worker_abc123"
    }
  ],
  "sorts": [
    {
      "field": "updated_at",
      "order": "desc"
    }
  ]
}
```

### 4. Find Stale Statuses (No Heartbeat for 30+ Minutes)

```json
{
  "filters": [
    {
      "field": "status",
      "operator": "in",
      "value": ["starting", "youtube_metadata", "transcript_extraction", "summary_generation"]
    },
    {
      "field": "heartbeat_at",
      "operator": "lt",
      "value": "2023-12-01T12:00:00Z"
    }
  ]
}
```

### 5. Find Statuses with Progress Between 50-90%

```json
{
  "filters": [
    {
      "field": "progress_percentage",
      "operator": "between",
      "value": 50.0,
      "value2": 90.0
    }
  ]
}
```

### 6. Search for Error Messages

```json
{
  "search": {
    "query": "timeout",
    "fields": ["error_info", "current_step"],
    "exact_match": false,
    "case_sensitive": false
  }
}
```

## Preset Filters

The API provides several preset filters for common use cases:

### Active Statuses
```http
GET /api/status/enhanced/preset/active
```
Returns statuses that are currently being processed.

### Failed Statuses
```http
GET /api/status/enhanced/preset/failed
```
Returns statuses that failed in the last 24 hours.

### Completed Today
```http
GET /api/status/enhanced/preset/completed_today
```
Returns statuses completed today.

### High Priority
```http
GET /api/status/enhanced/preset/high_priority
```
Returns high priority statuses.

### Stale Statuses
```http
GET /api/status/enhanced/preset/stale
```
Returns statuses without heartbeat for 30+ minutes.

## Search Functionality

### Basic Search
```http
GET /api/status/enhanced/search?q=error&fields=error_info&fields=current_step
```

### Advanced Search with Filters
```json
{
  "search": {
    "query": "youtube",
    "fields": ["current_step", "error_info"],
    "exact_match": false,
    "case_sensitive": false
  },
  "filters": [
    {
      "field": "status",
      "operator": "eq",
      "value": "failed"
    }
  ]
}
```

## Aggregations and Statistics

### Get Status Aggregates
```http
GET /api/status/enhanced/aggregates?start_date=2023-12-01T00:00:00Z&end_date=2023-12-31T23:59:59Z
```

Returns:
- Total count
- Status distribution
- Priority distribution  
- Progress statistics
- Worker statistics
- Time-based statistics

## Response Format

### Standard List Response
```json
{
  "statuses": [...],
  "total_count": 150,
  "page": 1,
  "page_size": 20,
  "total_pages": 8,
  "has_next": true,
  "has_previous": false,
  "filters_applied": 2,
  "query_time_ms": 45.2,
  "metadata": {
    "search_applied": true,
    "sorts_applied": 1,
    "include_counts": true
  }
}
```

### Aggregates Response
```json
{
  "total_count": 1000,
  "status_distribution": {
    "completed": 800,
    "failed": 150,
    "starting": 50
  },
  "priority_distribution": {
    "normal": 900,
    "high": 80,
    "low": 20
  },
  "progress_statistics": {
    "average": 75.5,
    "minimum": 0.0,
    "maximum": 100.0
  },
  "worker_statistics": {
    "unique_workers": 10,
    "total_with_workers": 950
  },
  "time_statistics": {
    "last_hour": 45,
    "last_24_hours": 200,
    "last_7_days": 800
  },
  "timestamp": "2023-12-01T12:00:00Z"
}
```

## Performance Considerations

### Efficient Filtering
- Use indexed fields when possible (`created_at`, `updated_at`, `status`)
- Limit page size to reasonable values (≤100)
- Use preset filters for common queries
- Consider date ranges for large datasets

### Query Optimization
- Use specific filters rather than broad searches
- Combine multiple conditions to reduce result sets
- Use `include_counts: false` when counts aren't needed
- Cache frequently used filter combinations

### Monitoring
- Monitor `query_time_ms` in responses
- Use aggregates endpoints for dashboard statistics
- Implement client-side caching for preset filters

## Error Handling

### Validation Errors (400)
```json
{
  "detail": "Invalid filter operator: invalid_op"
}
```

### Server Errors (500)
```json
{
  "detail": "Error filtering statuses: Database connection failed"
}
```

## Rate Limiting

Be mindful of API rate limits:
- Avoid frequent large queries
- Use pagination appropriately
- Cache results when possible
- Use preset filters for common queries

## Examples in Different Languages

### Python
```python
import requests

# Advanced filtering
filter_data = {
    "filters": [
        {"field": "status", "operator": "eq", "value": "failed"},
        {"field": "created_at", "operator": "gte", "value": "2023-12-01T00:00:00Z"}
    ],
    "page": 1,
    "page_size": 50
}

response = requests.post(
    "http://api.example.com/api/status/enhanced/filter",
    json=filter_data
)
```

### JavaScript
```javascript
// Search statuses
const searchParams = new URLSearchParams({
    q: 'error',
    fields: 'error_info',
    fields: 'current_step',
    page: 1,
    page_size: 20
});

fetch(`/api/status/enhanced/search?${searchParams}`)
    .then(response => response.json())
    .then(data => console.log(data));
```

### curl
```bash
# Get active statuses
curl -X GET "http://api.example.com/api/status/enhanced/preset/active?page=1&page_size=20"

# Advanced filtering
curl -X POST "http://api.example.com/api/status/enhanced/filter" \
  -H "Content-Type: application/json" \
  -d '{
    "filters": [
      {"field": "status", "operator": "eq", "value": "failed"}
    ],
    "page": 1,
    "page_size": 20
  }'
```