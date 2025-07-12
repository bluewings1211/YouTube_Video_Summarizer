# YouTube Video Summarizer API Documentation

## Overview

The YouTube Video Summarizer is a comprehensive AI-powered service for processing YouTube videos with transcript extraction, content summarization, keyword extraction, and timestamped segment analysis. This API provides a complete workflow management system with batch processing, real-time monitoring, and notification capabilities.

### Base URL
- **Development**: `http://localhost:8000`
- **Production**: `https://your-domain.com`

### API Version
- **Current Version**: v1.0.0
- **API Prefix**: `/api/v1/`

## Table of Contents

1. [Core Endpoints](#core-endpoints)
2. [History Management API](#history-management-api)
3. [Batch Processing API](#batch-processing-api)
4. [Status Tracking API](#status-tracking-api)
5. [Enhanced Status API](#enhanced-status-api)
6. [Notifications API](#notifications-api)
7. [Real-time Status API](#real-time-status-api)
8. [Data Models](#data-models)
9. [Error Handling](#error-handling)
10. [Authentication](#authentication)
11. [Rate Limiting](#rate-limiting)

---

## Core Endpoints

### Video Summarization

The primary endpoint for processing YouTube videos.

#### POST `/api/v1/summarize`

Summarize a YouTube video with AI-powered analysis.

**Request Body:**
```json
{
  "youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
  "reprocess_policy": "never"
}
```

**Parameters:**
- `youtube_url` (string, required): Valid YouTube video URL
- `reprocess_policy` (string, optional): Policy for handling duplicates
  - `never`: Skip if already processed
  - `always`: Always reprocess
  - `if_failed`: Reprocess only if previous attempt failed

**Response (200 OK):**
```json
{
  "video_id": "dQw4w9WgXcQ",
  "title": "Rick Astley - Never Gonna Give You Up",
  "duration": 213,
  "summary": "This music video features Rick Astley performing his iconic hit song...",
  "timestamped_segments": [
    {
      "timestamp": "00:45",
      "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=45s",
      "description": "Main chorus begins with the iconic hook",
      "importance_rating": 10
    }
  ],
  "keywords": ["Rick Astley", "Never Gonna Give You Up", "80s music"],
  "processing_time": 3.24
}
```

**Error Responses:**
- `400`: Invalid YouTube URL or request parameters
- `404`: Video not found or not accessible
- `422`: Video cannot be processed (no transcript, too long, etc.)
- `500`: Internal server error or AI service unavailable

### Health Check

#### GET `/health`

Check the overall health status of the API service.

**Response (200 OK):**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "version": "1.0.0",
  "components": {
    "workflow": {
      "status": "healthy",
      "ready": true
    },
    "database": {
      "status": "healthy",
      "response_time_ms": 5.2,
      "pool_status": "available"
    }
  }
}
```

#### GET `/health/database`

Detailed database health information.

### Metrics

#### GET `/metrics`

Application performance metrics for monitoring.

**Response (200 OK):**
```json
{
  "uptime_seconds": 3600,
  "timestamp": "2024-01-15T10:30:00Z",
  "version": "1.0.0",
  "workflow_status": {
    "initialized": true,
    "ready": true
  }
}
```

### Root Information

#### GET `/`

API information and endpoint discovery.

**Response (200 OK):**
```json
{
  "service": "YouTube Summarizer",
  "version": "1.0.0",
  "description": "AI-powered YouTube video summarization service",
  "endpoints": {
    "summarize": "/api/v1/summarize",
    "history": "/api/v1/history/videos",
    "batch": "/api/v1/batch/batches",
    "health": "/health",
    "docs": "/api/docs"
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

---

## History Management API

Base path: `/api/v1/history`

The History API provides comprehensive video processing history management with advanced deletion, reprocessing, and transaction capabilities.

### Video Listing and Search

#### GET `/videos`

Retrieve paginated list of processed videos with advanced filtering.

**Query Parameters:**
- `page` (integer): Page number (default: 1)
- `page_size` (integer): Items per page (default: 10, max: 100)
- `sort_by` (string): Sort field (created_at, title, duration, status)
- `sort_order` (string): Sort direction (asc, desc)
- `date_from` (string): Filter by date range start (ISO format)
- `date_to` (string): Filter by date range end (ISO format)
- `keywords` (string): Filter by keywords (comma-separated)
- `title_search` (string): Search in video titles

**Response (200 OK):**
```json
{
  "videos": [
    {
      "id": "uuid",
      "youtube_id": "dQw4w9WgXcQ",
      "title": "Video Title",
      "duration": 213,
      "created_at": "2024-01-15T10:30:00Z",
      "status": "completed",
      "summary_word_count": 245
    }
  ],
  "pagination": {
    "page": 1,
    "page_size": 10,
    "total_count": 150,
    "total_pages": 15,
    "has_next": true,
    "has_previous": false
  }
}
```

#### GET `/videos/{video_id}`

Get detailed information for a specific video.

**Response (200 OK):**
```json
{
  "video": {
    "id": "uuid",
    "youtube_id": "dQw4w9WgXcQ",
    "title": "Video Title",
    "duration": 213,
    "transcript": "Full transcript text...",
    "summary": "AI-generated summary...",
    "keywords": ["keyword1", "keyword2"],
    "segments": [
      {
        "timestamp": "01:30",
        "description": "Key moment",
        "importance_rating": 8
      }
    ],
    "processing_metadata": {
      "processing_time": 3.24,
      "llm_model": "gpt-4",
      "created_at": "2024-01-15T10:30:00Z",
      "updated_at": "2024-01-15T10:30:00Z"
    }
  }
}
```

### Video Statistics

#### GET `/statistics`

Get comprehensive statistics about video processing.

**Response (200 OK):**
```json
{
  "total_videos": 1250,
  "completed_videos": 1180,
  "failed_videos": 45,
  "processing_videos": 25,
  "completion_rate": 94.4,
  "average_processing_time": 4.2,
  "total_duration_hours": 2840.5,
  "status_distribution": {
    "completed": 1180,
    "failed": 45,
    "processing": 25
  },
  "daily_counts": [
    {"date": "2024-01-15", "count": 48},
    {"date": "2024-01-14", "count": 52}
  ]
}
```

### Video Deletion Operations

#### GET `/videos/{video_id}/deletion-info`

Preview the impact of deleting a video.

**Response (200 OK):**
```json
{
  "video_id": "uuid",
  "related_records": {
    "transcripts": 1,
    "summaries": 1,
    "keywords": 5,
    "segments": 8,
    "batch_items": 0,
    "status_records": 3
  },
  "total_records": 18,
  "estimated_impact": "low",
  "warnings": []
}
```

#### DELETE `/videos/{video_id}`

Delete a video with cascading deletion of related records.

**Request Body:**
```json
{
  "force": false,
  "audit_user": "user@example.com"
}
```

**Response (200 OK):**
```json
{
  "success": true,
  "deleted_records": {
    "videos": 1,
    "transcripts": 1,
    "summaries": 1,
    "keywords": 5,
    "segments": 8,
    "total": 16
  },
  "processing_time": 0.245,
  "audit_log": "deletion_audit_uuid"
}
```

#### POST `/videos/batch-delete`

Delete multiple videos in a single operation.

**Request Body:**
```json
{
  "video_ids": ["uuid1", "uuid2", "uuid3"],
  "force": false,
  "audit_user": "user@example.com"
}
```

### Video Reprocessing

#### POST `/videos/{video_id}/reprocess`

Initiate reprocessing of a video with configurable options.

**Request Body:**
```json
{
  "mode": "full",
  "force": false,
  "clear_cache": true,
  "preserve_metadata": false
}
```

**Parameters:**
- `mode`: Reprocessing mode
  - `full`: Complete reprocessing
  - `summary_only`: Regenerate only summary
  - `keywords_only`: Regenerate only keywords
  - `segments_only`: Regenerate only timestamped segments
- `force`: Override safety checks
- `clear_cache`: Clear cached data before reprocessing
- `preserve_metadata`: Keep original processing metadata

**Response (200 OK):**
```json
{
  "reprocessing_id": "uuid",
  "status": "initiated",
  "estimated_time": 180,
  "mode": "full",
  "message": "Reprocessing initiated successfully"
}
```

#### GET `/videos/{video_id}/reprocessing-status`

Check the status of an ongoing reprocessing operation.

**Response (200 OK):**
```json
{
  "reprocessing_id": "uuid",
  "status": "processing",
  "progress_percentage": 65,
  "current_step": "generating_summary",
  "estimated_remaining_time": 45,
  "steps_completed": ["extract_transcript", "analyze_content"],
  "steps_remaining": ["generate_keywords", "create_segments"]
}
```

---

## Batch Processing API

Base path: `/api/v1/batch`

The Batch Processing API enables efficient processing of multiple YouTube videos simultaneously with advanced queue management and progress tracking.

### Batch Management

#### POST `/batches`

Create a new batch processing job.

**Request Body:**
```json
{
  "name": "Marketing Videos Batch",
  "description": "Process all marketing videos for Q1 campaign",
  "urls": [
    "https://www.youtube.com/watch?v=video1",
    "https://www.youtube.com/watch?v=video2"
  ],
  "priority": "high",
  "webhook_url": "https://your-app.com/webhook/batch-complete",
  "processing_options": {
    "reprocess_policy": "never",
    "enable_notifications": true
  }
}
```

**Response (201 Created):**
```json
{
  "batch_id": "batch_uuid",
  "name": "Marketing Videos Batch",
  "status": "created",
  "total_items": 2,
  "created_at": "2024-01-15T10:30:00Z",
  "estimated_processing_time": 600,
  "items": [
    {
      "item_id": "item_uuid_1",
      "youtube_url": "https://www.youtube.com/watch?v=video1",
      "status": "pending",
      "position": 1
    },
    {
      "item_id": "item_uuid_2",
      "youtube_url": "https://www.youtube.com/watch?v=video2",
      "status": "pending",
      "position": 2
    }
  ]
}
```

#### GET `/batches`

List batches with filtering and pagination.

**Query Parameters:**
- `batch_status` (string): Filter by status (created, processing, completed, failed, cancelled)
- `page` (integer): Page number
- `page_size` (integer): Items per page

**Response (200 OK):**
```json
{
  "batches": [
    {
      "batch_id": "batch_uuid",
      "name": "Marketing Videos Batch",
      "status": "processing",
      "total_items": 2,
      "completed_items": 1,
      "failed_items": 0,
      "progress_percentage": 50,
      "created_at": "2024-01-15T10:30:00Z",
      "started_at": "2024-01-15T10:32:00Z"
    }
  ],
  "pagination": {
    "page": 1,
    "page_size": 10,
    "total_count": 5,
    "has_next": false
  }
}
```

#### GET `/batches/{batch_id}`

Get detailed information about a specific batch.

**Response (200 OK):**
```json
{
  "batch_id": "batch_uuid",
  "name": "Marketing Videos Batch",
  "description": "Process all marketing videos for Q1 campaign",
  "status": "processing",
  "priority": "high",
  "total_items": 2,
  "completed_items": 1,
  "failed_items": 0,
  "progress_percentage": 50,
  "created_at": "2024-01-15T10:30:00Z",
  "started_at": "2024-01-15T10:32:00Z",
  "estimated_completion": "2024-01-15T10:42:00Z",
  "items": [
    {
      "item_id": "item_uuid_1",
      "youtube_url": "https://www.youtube.com/watch?v=video1",
      "status": "completed",
      "video_id": "dQw4w9WgXcQ",
      "processing_time": 3.24,
      "completed_at": "2024-01-15T10:35:00Z"
    },
    {
      "item_id": "item_uuid_2",
      "youtube_url": "https://www.youtube.com/watch?v=video2",
      "status": "processing",
      "progress_percentage": 65,
      "current_step": "generating_summary",
      "started_at": "2024-01-15T10:35:30Z"
    }
  ]
}
```

### Batch Operations

#### POST `/batches/{batch_id}/start`

Start processing a batch.

**Request Body:**
```json
{
  "force": false
}
```

**Response (200 OK):**
```json
{
  "batch_id": "batch_uuid",
  "status": "processing",
  "started_at": "2024-01-15T10:32:00Z",
  "estimated_completion": "2024-01-15T10:42:00Z",
  "message": "Batch processing started successfully"
}
```

#### POST `/batches/{batch_id}/cancel`

Cancel a batch that is queued or currently processing.

**Request Body:**
```json
{
  "reason": "User requested cancellation"
}
```

**Response (200 OK):**
```json
{
  "batch_id": "batch_uuid",
  "status": "cancelled",
  "cancelled_at": "2024-01-15T10:35:00Z",
  "items_affected": 1,
  "message": "Batch cancelled successfully"
}
```

#### GET `/batches/{batch_id}/progress`

Get detailed progress information with time estimates.

**Response (200 OK):**
```json
{
  "batch_id": "batch_uuid",
  "overall_progress": 50,
  "items_completed": 1,
  "items_processing": 1,
  "items_pending": 0,
  "items_failed": 0,
  "estimated_remaining_time": 300,
  "average_processing_time": 180,
  "current_processing_rate": 0.5,
  "detailed_progress": [
    {
      "item_id": "item_uuid_1",
      "status": "completed",
      "progress": 100
    },
    {
      "item_id": "item_uuid_2",
      "status": "processing",
      "progress": 65,
      "current_step": "generating_summary",
      "estimated_remaining": 60
    }
  ]
}
```

### Queue Management

#### GET `/queue/next`

Get the next item from the processing queue (for workers).

**Query Parameters:**
- `queue_name` (string): Specific queue name
- `worker_id` (string): Worker identifier

**Response (200 OK):**
```json
{
  "item_id": "item_uuid",
  "batch_id": "batch_uuid",
  "youtube_url": "https://www.youtube.com/watch?v=video1",
  "priority": "high",
  "processing_options": {
    "reprocess_policy": "never"
  },
  "assigned_at": "2024-01-15T10:32:00Z",
  "worker_id": "worker_001"
}
```

#### POST `/queue/complete/{batch_item_id}`

Mark a batch item as completed and provide results.

**Request Body:**
```json
{
  "status": "completed",
  "video_id": "dQw4w9WgXcQ",
  "processing_time": 3.24,
  "summary": "AI-generated summary...",
  "keywords": ["keyword1", "keyword2"],
  "error_message": null
}
```

### Batch Statistics

#### GET `/statistics`

Get comprehensive batch processing statistics.

**Response (200 OK):**
```json
{
  "total_batches": 150,
  "active_batches": 5,
  "completed_batches": 140,
  "failed_batches": 5,
  "total_items_processed": 3500,
  "average_batch_size": 23,
  "average_processing_time": 180,
  "success_rate": 96.7,
  "queue_statistics": {
    "pending_items": 25,
    "processing_items": 8,
    "average_wait_time": 45
  }
}
```

---

## Status Tracking API

Base path: `/api/status`

Comprehensive real-time status tracking system for monitoring video processing operations.

### Status Retrieval

#### GET `/`

List processing statuses with pagination and filtering.

**Query Parameters:**
- `page` (integer): Page number
- `page_size` (integer): Items per page
- `status_filter` (string): Filter by status type
- `worker_id` (string): Filter by worker
- `active_only` (boolean): Show only active statuses

**Response (200 OK):**
```json
{
  "statuses": [
    {
      "status_id": "status_uuid",
      "video_id": "video_uuid",
      "batch_item_id": "item_uuid",
      "status": "processing",
      "progress_percentage": 65,
      "current_step": "generating_summary",
      "worker_id": "worker_001",
      "started_at": "2024-01-15T10:30:00Z",
      "last_heartbeat": "2024-01-15T10:32:00Z"
    }
  ],
  "pagination": {
    "page": 1,
    "page_size": 20,
    "total_count": 45
  }
}
```

#### GET `/{status_id}`

Get detailed status information by status ID.

#### GET `/video/{video_id}`

Get status information for a specific video.

#### GET `/batch/{batch_item_id}`

Get status information for a specific batch item.

### Status Updates

#### PUT `/{status_id}`

Update processing status with progress information.

**Request Body:**
```json
{
  "new_status": "processing",
  "progress_percentage": 75,
  "current_step": "extracting_keywords",
  "estimated_remaining_time": 30,
  "metadata": {
    "processed_segments": 5,
    "total_segments": 8
  }
}
```

#### PATCH `/{status_id}/progress`

Update only the progress percentage.

**Request Body:**
```json
{
  "progress_percentage": 80,
  "current_step": "finalizing_results"
}
```

#### POST `/{status_id}/error`

Report a processing error.

**Request Body:**
```json
{
  "error_code": "TRANSCRIPT_ERROR",
  "error_message": "Unable to extract transcript from video",
  "error_severity": "high",
  "is_recoverable": true,
  "retry_suggested": true
}
```

#### POST `/{status_id}/heartbeat`

Update heartbeat to indicate worker is still active.

**Response (200 OK):**
```json
{
  "status_id": "status_uuid",
  "heartbeat_updated": true,
  "last_heartbeat": "2024-01-15T10:32:00Z"
}
```

### Status Analytics

#### GET `/metrics/summary`

Get performance summary across all processing operations.

**Response (200 OK):**
```json
{
  "total_processed": 1250,
  "average_processing_time": 180,
  "success_rate": 94.5,
  "current_active": 15,
  "processing_rate_per_hour": 45,
  "peak_concurrent_processing": 25,
  "status_distribution": {
    "completed": 1180,
    "processing": 15,
    "failed": 55
  }
}
```

#### GET `/metrics/worker/{worker_id}`

Get performance metrics for a specific worker.

**Response (200 OK):**
```json
{
  "worker_id": "worker_001",
  "total_processed": 150,
  "success_rate": 96.7,
  "average_processing_time": 175,
  "current_load": 3,
  "last_active": "2024-01-15T10:32:00Z",
  "performance_trend": "improving"
}
```

#### GET `/metrics/hourly`

Get hourly processing metrics.

**Query Parameters:**
- `hours` (integer): Number of hours to include (default: 24)

**Response (200 OK):**
```json
{
  "metrics": [
    {
      "hour": "2024-01-15T10:00:00Z",
      "completed": 45,
      "failed": 2,
      "average_time": 178,
      "peak_concurrent": 12
    }
  ]
}
```

---

## Enhanced Status API

Base path: `/api/status/enhanced`

Advanced status filtering and analytics capabilities.

### Advanced Filtering

#### POST `/filter`

Perform complex filtering across status records.

**Request Body:**
```json
{
  "filters": [
    {
      "field": "status",
      "operator": "in",
      "value": ["processing", "queued"]
    },
    {
      "field": "created_at",
      "operator": "gte",
      "value": "2024-01-15T00:00:00Z"
    }
  ],
  "sorts": [
    {
      "field": "created_at",
      "direction": "desc"
    }
  ],
  "search": {
    "query": "error",
    "fields": ["error_message", "current_step"]
  },
  "pagination": {
    "page": 1,
    "page_size": 50
  }
}
```

#### GET `/preset/{preset_type}`

Use predefined filter presets.

**Available Presets:**
- `active`: Currently processing items
- `failed`: Failed processing attempts
- `completed_today`: Completed in the last 24 hours
- `high_priority`: High-priority items
- `stale`: Items without recent heartbeat

#### GET `/search`

Full-text search across status records.

**Query Parameters:**
- `q` (string): Search query
- `fields` (string): Comma-separated fields to search
- `exact_match` (boolean): Use exact matching
- `case_sensitive` (boolean): Case-sensitive search

### Analytics

#### GET `/aggregates`

Get aggregated statistics with customizable grouping.

**Query Parameters:**
- `start_date` (string): Start date for aggregation
- `end_date` (string): End date for aggregation
- `group_by` (string): Grouping field (status, worker_id, date)

**Response (200 OK):**
```json
{
  "aggregates": {
    "by_status": {
      "completed": 850,
      "processing": 15,
      "failed": 35,
      "queued": 10
    },
    "by_priority": {
      "high": 125,
      "medium": 600,
      "low": 175
    },
    "progress_distribution": {
      "0-25%": 45,
      "26-50%": 38,
      "51-75%": 42,
      "76-100%": 785
    }
  },
  "time_range": {
    "start_date": "2024-01-01T00:00:00Z",
    "end_date": "2024-01-15T23:59:59Z"
  }
}
```

---

## Notifications API

Base path: `/api/v1/notifications`

Comprehensive notification system for event-driven communication.

### Configuration Management

#### POST `/configs`

Create a new notification configuration.

**Request Body:**
```json
{
  "name": "Batch Completion Notifications",
  "type": "webhook",
  "events": ["batch.completed", "batch.failed"],
  "target_address": "https://your-app.com/webhook/notifications",
  "is_active": true,
  "rate_limit": {
    "max_per_minute": 10,
    "max_per_hour": 100
  },
  "retry_config": {
    "max_retries": 3,
    "retry_delay_seconds": 30
  },
  "filter_conditions": {
    "priority": ["high", "critical"]
  }
}
```

**Response (201 Created):**
```json
{
  "config_id": "config_uuid",
  "name": "Batch Completion Notifications",
  "type": "webhook",
  "events": ["batch.completed", "batch.failed"],
  "target_address": "https://your-app.com/webhook/notifications",
  "is_active": true,
  "created_at": "2024-01-15T10:30:00Z"
}
```

#### GET `/configs`

List notification configurations with filtering.

**Query Parameters:**
- `type` (string): Filter by notification type
- `is_active` (boolean): Filter by active status
- `event` (string): Filter by event type

#### PUT `/configs/{config_id}`

Update notification configuration.

#### DELETE `/configs/{config_id}`

Delete notification configuration.

### Notification Operations

#### POST `/trigger`

Manually trigger notifications for specific events.

**Request Body:**
```json
{
  "event_type": "video.completed",
  "metadata": {
    "video_id": "video_uuid",
    "title": "Video Title",
    "processing_time": 180
  },
  "priority": "medium",
  "target_configs": ["config_uuid"]
}
```

#### POST `/send-pending`

Process and send all pending notifications.

**Response (200 OK):**
```json
{
  "processed": 15,
  "sent": 12,
  "failed": 2,
  "skipped": 1,
  "processing_time": 2.5
}
```

#### POST `/retry-failed`

Retry failed notification deliveries.

**Request Body:**
```json
{
  "max_retry_attempts": 3,
  "filter_by_config": "config_uuid"
}
```

### Notification Statistics

#### GET `/stats`

Get comprehensive notification statistics.

**Query Parameters:**
- `user_id` (string): Filter by user
- `start_date` (string): Statistics start date
- `end_date` (string): Statistics end date

**Response (200 OK):**
```json
{
  "total_sent": 1250,
  "total_failed": 45,
  "success_rate": 96.4,
  "average_delivery_time": 1.2,
  "by_type": {
    "webhook": 800,
    "email": 450
  },
  "by_event": {
    "video.completed": 600,
    "batch.completed": 300,
    "processing.failed": 350
  },
  "rate_limiting": {
    "current_rate": 8.5,
    "limit_reached_count": 12
  }
}
```

---

## Real-time Status API

Base path: `/api/realtime`

WebSocket-based real-time monitoring capabilities.

### WebSocket Connection

#### WebSocket `/status`

Establish WebSocket connection for real-time status updates.

**Connection URL:**
```
ws://localhost:8000/api/realtime/status?client_id=unique_client_id
```

**Message Types:**

1. **Subscribe to Updates:**
```json
{
  "type": "subscribe",
  "filters": {
    "status": ["processing", "queued"],
    "worker_id": "worker_001"
  }
}
```

2. **Unsubscribe:**
```json
{
  "type": "unsubscribe"
}
```

3. **Get Current Status:**
```json
{
  "type": "get_status",
  "status_id": "status_uuid"
}
```

4. **Heartbeat:**
```json
{
  "type": "ping"
}
```

**Server Messages:**

1. **Status Update:**
```json
{
  "type": "status_update",
  "data": {
    "status_id": "status_uuid",
    "video_id": "video_uuid",
    "status": "processing",
    "progress_percentage": 75,
    "current_step": "generating_keywords"
  }
}
```

2. **Metrics Update:**
```json
{
  "type": "metrics_update",
  "data": {
    "active_processing": 15,
    "queue_size": 25,
    "average_processing_time": 180
  }
}
```

### Management Endpoints

#### GET `/connections/stats`

Get WebSocket connection statistics.

**Response (200 OK):**
```json
{
  "total_connections": 25,
  "active_subscriptions": 18,
  "average_connection_time": 1800,
  "message_throughput": {
    "sent_per_minute": 45,
    "received_per_minute": 12
  }
}
```

#### POST `/broadcast`

Broadcast system message to all connected clients.

**Request Body:**
```json
{
  "message_type": "system_announcement",
  "content": "System maintenance scheduled for 2:00 AM UTC",
  "priority": "medium"
}
```

#### GET `/demo`

Get demo HTML page for testing WebSocket functionality.

---

## Data Models

### Core Models

#### Video Model
```json
{
  "id": "uuid",
  "youtube_id": "string",
  "title": "string",
  "duration": "integer",
  "transcript": "string",
  "summary": "string", 
  "keywords": ["string"],
  "created_at": "datetime",
  "updated_at": "datetime",
  "status": "enum"
}
```

#### Timestamped Segment Model
```json
{
  "timestamp": "string",
  "url": "string", 
  "description": "string",
  "importance_rating": "integer"
}
```

#### Batch Model
```json
{
  "batch_id": "uuid",
  "name": "string",
  "description": "string",
  "status": "enum",
  "priority": "enum",
  "total_items": "integer",
  "completed_items": "integer",
  "failed_items": "integer",
  "created_at": "datetime",
  "started_at": "datetime",
  "completed_at": "datetime"
}
```

#### Status Model
```json
{
  "status_id": "uuid",
  "video_id": "uuid",
  "batch_item_id": "uuid",
  "status": "enum",
  "progress_percentage": "integer",
  "current_step": "string",
  "worker_id": "string",
  "created_at": "datetime",
  "updated_at": "datetime",
  "last_heartbeat": "datetime"
}
```

### Enumeration Values

#### Processing Status
- `pending`: Waiting to be processed
- `queued`: In processing queue
- `processing`: Currently being processed
- `completed`: Successfully completed
- `failed`: Processing failed
- `cancelled`: Processing cancelled

#### Priority Levels
- `low`: Low priority
- `medium`: Medium priority  
- `high`: High priority
- `critical`: Critical priority

#### Notification Types
- `webhook`: HTTP webhook
- `email`: Email notification
- `slack`: Slack integration
- `discord`: Discord integration

---

## Error Handling

### Error Response Format

All API errors follow a consistent format:

```json
{
  "error": {
    "code": "E1001",
    "category": "validation",
    "severity": "medium",
    "title": "Invalid YouTube URL Format",
    "message": "The provided URL is not a valid YouTube video URL",
    "suggested_actions": [
      "Check URL format",
      "Ensure URL starts with https://www.youtube.com/watch",
      "Try again"
    ],
    "is_recoverable": true,
    "timestamp": "2024-01-15T10:30:00Z",
    "technical_details": "URL validation failed: invalid format"
  }
}
```

### Error Categories

#### Validation Errors (400)
- `E1001`: Invalid YouTube URL Format
- `E1002`: Missing Required Parameters
- `E1003`: Invalid Parameter Values

#### Content Errors (422)
- `E2001`: Video Not Found
- `E2002`: Video Private/Restricted
- `E2003`: No Transcript Available
- `E2004`: Video Live Stream
- `E2005`: Video Too Long
- `E2006`: Unsupported Language

#### System Errors (500)
- `E3001`: LLM Service Error
- `E3002`: Database Connection Error
- `E3003`: Workflow Execution Failed
- `E3004`: Internal Processing Error

#### Network Errors (502/503)
- `E4001`: YouTube API Unavailable
- `E4002`: External Service Timeout
- `E4003`: Rate Limit Exceeded
- `E4004`: Connection Failed

### Error Handling Best Practices

1. **Check Error Category**: Use the `category` field to understand error type
2. **Follow Suggested Actions**: The `suggested_actions` array provides remediation steps
3. **Consider Recoverability**: Use `is_recoverable` to determine retry logic
4. **Monitor Error Codes**: Track specific error codes for system monitoring

---

## Authentication

### API Key Authentication

Include your API key in the request header:

```http
Authorization: Bearer your_api_key_here
```

### Request Headers

#### Required Headers
- `Content-Type: application/json` (for POST/PUT requests)
- `Authorization: Bearer {api_key}` (if authentication enabled)

#### Optional Headers
- `X-Request-ID: {unique_id}` (for request tracking)
- `User-Agent: {your_app_name}/{version}` (for monitoring)

---

## Rate Limiting

### Rate Limit Headers

All responses include rate limiting information:

```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1642234800
X-RateLimit-Window: 3600
```

### Rate Limits by Endpoint

| Endpoint Category | Requests per Hour | Burst Limit |
|-------------------|------------------|-------------|
| Video Summarization | 100 | 10 |
| Batch Operations | 50 | 5 |
| Status Queries | 1000 | 50 |
| History Queries | 500 | 25 |
| Notifications | 200 | 20 |

### Rate Limit Exceeded

When rate limits are exceeded, you'll receive a `429 Too Many Requests` response:

```json
{
  "error": {
    "code": "E4003",
    "category": "rate_limit",
    "severity": "medium",
    "title": "Rate Limit Exceeded",
    "message": "You have exceeded the rate limit for this endpoint",
    "suggested_actions": [
      "Wait before retrying",
      "Implement exponential backoff",
      "Contact support for higher limits"
    ],
    "retry_after": 3600
  }
}
```

---

## SDK and Client Libraries

### Python SDK Example

```python
from youtube_summarizer_client import YouTubeSummarizerClient

client = YouTubeSummarizerClient(api_key="your_api_key")

# Summarize a video
result = client.summarize_video("https://www.youtube.com/watch?v=dQw4w9WgXcQ")

# Create a batch
batch = client.create_batch(
    name="My Batch",
    urls=["https://www.youtube.com/watch?v=video1", "https://www.youtube.com/watch?v=video2"]
)

# Monitor status
status = client.get_status(result.status_id)
```

### JavaScript SDK Example

```javascript
import { YouTubeSummarizerClient } from 'youtube-summarizer-js';

const client = new YouTubeSummarizerClient('your_api_key');

// Summarize a video
const result = await client.summarizeVideo('https://www.youtube.com/watch?v=dQw4w9WgXcQ');

// Create batch
const batch = await client.createBatch({
  name: 'My Batch',
  urls: ['https://www.youtube.com/watch?v=video1', 'https://www.youtube.com/watch?v=video2']
});
```

---

## Support and Resources

### Documentation Links
- **OpenAPI Specification**: `/api/docs` (Swagger UI)
- **ReDoc Documentation**: `/api/redoc`
- **API Schema**: `/openapi.json`

### Support Channels
- **GitHub Issues**: [github.com/your-org/youtube-summarizer/issues](https://github.com/your-org/youtube-summarizer/issues)
- **Discord Community**: [discord.gg/youtube-summarizer](https://discord.gg/youtube-summarizer)
- **Email Support**: support@your-domain.com

### Change Log
- **v1.0.0**: Initial release with core summarization features
- **v1.1.0**: Added batch processing capabilities
- **v1.2.0**: Enhanced status tracking and real-time monitoring
- **v1.3.0**: Notification system and webhook support

---

*Last updated: January 15, 2024*