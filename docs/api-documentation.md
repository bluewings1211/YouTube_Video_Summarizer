# YouTube Summarizer API Documentation

## Overview

The YouTube Summarizer API provides AI-powered video analysis capabilities, extracting transcripts, generating summaries, and identifying key segments from YouTube videos. This REST API is built with FastAPI and includes comprehensive error handling and monitoring.

## Base URL

```
http://localhost:8000  # Development
https://your-domain.com  # Production
```

## Authentication

No authentication required for public endpoints.

## Content Type

All requests and responses use `application/json` content type.

## API Endpoints

### 1. Video Summarization

**POST** `/api/v1/summarize`

The main endpoint for processing YouTube videos and generating comprehensive analysis.

#### Request

```json
{
  "youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
}
```

#### Request Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `youtube_url` | string | Yes | Valid YouTube video URL (youtube.com/watch or youtu.be formats) |

#### Supported URL Formats

- `https://www.youtube.com/watch?v=VIDEO_ID`
- `https://youtu.be/VIDEO_ID`
- `https://www.youtube.com/watch?v=VIDEO_ID&t=123s`
- `https://m.youtube.com/watch?v=VIDEO_ID`

#### Response (Success - 200)

```json
{
  "video_id": "dQw4w9WgXcQ",
  "title": "Rick Astley - Never Gonna Give You Up (Official Music Video)",
  "duration": 213,
  "summary": "This music video features Rick Astley performing his iconic hit song 'Never Gonna Give You Up.' The video showcases classic 80s aesthetics with Rick's distinctive deep voice and dance moves. The song has become a cultural phenomenon, particularly associated with the internet meme known as 'Rickrolling.' The video demonstrates the production values and style typical of 1980s music videos, with its simple set design and focus on the artist's performance. Rick Astley's confident delivery and the song's memorable hook have made it one of the most recognizable songs of the decade.",
  "timestamped_segments": [
    {
      "timestamp": "00:00:10",
      "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=10s",
      "description": "Rick Astley introduces the song with his signature vocal style",
      "importance_rating": 9
    },
    {
      "timestamp": "00:00:45",
      "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=45s",
      "description": "Main chorus begins with the iconic hook",
      "importance_rating": 10
    },
    {
      "timestamp": "00:01:30",
      "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=90s",
      "description": "Dance sequence showcasing 80s choreography",
      "importance_rating": 7
    },
    {
      "timestamp": "00:02:15",
      "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=135s",
      "description": "Bridge section with instrumental break",
      "importance_rating": 6
    },
    {
      "timestamp": "00:03:00",
      "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=180s",
      "description": "Final chorus and outro",
      "importance_rating": 8
    }
  ],
  "keywords": [
    "Rick Astley",
    "Never Gonna Give You Up",
    "80s music",
    "pop music",
    "music video",
    "retro",
    "classic hits"
  ],
  "processing_time": 3.24
}
```

#### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `video_id` | string | YouTube video ID extracted from URL |
| `title` | string | Video title from YouTube metadata |
| `duration` | integer | Video duration in seconds |
| `summary` | string | AI-generated summary (max 500 words) |
| `timestamped_segments` | array | Array of important video segments with timestamps |
| `keywords` | array | Array of 5-8 extracted keywords |
| `processing_time` | float | Processing time in seconds |

#### Timestamped Segment Fields

| Field | Type | Description |
|-------|------|-------------|
| `timestamp` | string | Timestamp in HH:MM:SS or MM:SS format |
| `url` | string | YouTube URL with timestamp parameter |
| `description` | string | Brief description of the segment |
| `importance_rating` | integer | Importance rating from 1-10 |

### 2. Health Check

**GET** `/health`

Returns the current health status of the API service.

#### Response (Success - 200)

```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "version": "1.0.0",
  "workflow_ready": true
}
```

#### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | Service health status ("healthy" or "unhealthy") |
| `timestamp` | string | Current timestamp in ISO format |
| `version` | string | API version |
| `workflow_ready` | boolean | Whether the workflow engine is initialized |

### 3. Metrics

**GET** `/metrics`

Returns application metrics for monitoring and observability.

#### Response (Success - 200)

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

### 4. Root Information

**GET** `/`

Returns basic API information and available endpoints.

#### Response (Success - 200)

```json
{
  "service": "YouTube Summarizer",
  "version": "1.0.0",
  "description": "AI-powered YouTube video summarization service",
  "endpoints": {
    "summarize": "/api/v1/summarize",
    "health": "/health",
    "docs": "/api/docs"
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

## Error Handling

The API provides comprehensive error handling with detailed error responses following a consistent structure.

### Error Response Format

```json
{
  "error": {
    "code": "E1001",
    "category": "validation",
    "severity": "medium",
    "title": "Invalid YouTube URL Format",
    "message": "Please provide a valid YouTube video URL",
    "suggested_actions": [
      "Check URL format",
      "Ensure URL is accessible",
      "Try again"
    ],
    "is_recoverable": true,
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
```

### Common Error Codes

#### 400 Bad Request

**Invalid YouTube URL Format**
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
      "Ensure URL starts with https://www.youtube.com/watch or https://youtu.be/",
      "Try again"
    ],
    "is_recoverable": true,
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
```

**Video Too Long**
```json
{
  "error": {
    "code": "E2005",
    "category": "content",
    "severity": "medium",
    "title": "Video Duration Exceeds Limit",
    "message": "Video duration exceeds the maximum allowed limit of 30 minutes",
    "suggested_actions": [
      "Use shorter videos",
      "Try videos under 30 minutes",
      "Split content into smaller segments"
    ],
    "is_recoverable": true,
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
```

#### 404 Not Found

**Video Not Found**
```json
{
  "error": {
    "code": "E2001",
    "category": "content",
    "severity": "medium",
    "title": "Video Not Found",
    "message": "The requested video could not be found or is not accessible",
    "suggested_actions": [
      "Check video URL",
      "Ensure video is public",
      "Try a different video"
    ],
    "is_recoverable": true,
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
```

#### 422 Unprocessable Entity

**No Transcript Available**
```json
{
  "error": {
    "code": "E2003",
    "category": "content",
    "severity": "medium",
    "title": "Transcript Not Available",
    "message": "No transcript is available for this video",
    "suggested_actions": [
      "Try videos with auto-generated captions",
      "Use videos with manual transcripts",
      "Try different content"
    ],
    "is_recoverable": true,
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
```

**Video is Live Stream**
```json
{
  "error": {
    "code": "E2004",
    "category": "content",
    "severity": "medium",
    "title": "Live Stream Not Supported",
    "message": "Live streams are not supported for summarization",
    "suggested_actions": [
      "Use recorded videos",
      "Wait for stream to end",
      "Try different content"
    ],
    "is_recoverable": true,
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
```

#### 500 Internal Server Error

**LLM Service Error**
```json
{
  "error": {
    "code": "E3001",
    "category": "llm",
    "severity": "high",
    "title": "LLM Service Error",
    "message": "The AI service is temporarily unavailable",
    "suggested_actions": [
      "Try again later",
      "Check service status",
      "Contact support if problem persists"
    ],
    "is_recoverable": true,
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
```

## Request Examples

### cURL Examples

**Basic Summarization Request**
```bash
curl -X POST "http://localhost:8000/api/v1/summarize" \
  -H "Content-Type: application/json" \
  -d '{
    "youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
  }'
```

**With Response Headers**
```bash
curl -X POST "http://localhost:8000/api/v1/summarize" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d '{
    "youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
  }' \
  -w "HTTP Status: %{http_code}\nResponse Time: %{time_total}s\n"
```

**Health Check**
```bash
curl -X GET "http://localhost:8000/health"
```

### Python Examples

**Using requests library**
```python
import requests
import json

# Summarize video
url = "http://localhost:8000/api/v1/summarize"
data = {
    "youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
}

response = requests.post(url, json=data)

if response.status_code == 200:
    result = response.json()
    print(f"Video: {result['title']}")
    print(f"Summary: {result['summary']}")
    print(f"Keywords: {', '.join(result['keywords'])}")
    print(f"Processing time: {result['processing_time']}s")
else:
    error = response.json()
    print(f"Error: {error['error']['message']}")
```

**Async example with aiohttp**
```python
import aiohttp
import asyncio
import json

async def summarize_video(youtube_url):
    async with aiohttp.ClientSession() as session:
        url = "http://localhost:8000/api/v1/summarize"
        data = {"youtube_url": youtube_url}
        
        async with session.post(url, json=data) as response:
            if response.status == 200:
                result = await response.json()
                return result
            else:
                error = await response.json()
                raise Exception(f"API Error: {error['error']['message']}")

# Usage
result = asyncio.run(summarize_video("https://www.youtube.com/watch?v=dQw4w9WgXcQ"))
```

### JavaScript Examples

**Using fetch API**
```javascript
const summarizeVideo = async (youtubeUrl) => {
    const response = await fetch('http://localhost:8000/api/v1/summarize', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            youtube_url: youtubeUrl
        })
    });

    if (!response.ok) {
        const error = await response.json();
        throw new Error(`API Error: ${error.error.message}`);
    }

    return await response.json();
};

// Usage
summarizeVideo('https://www.youtube.com/watch?v=dQw4w9WgXcQ')
    .then(result => {
        console.log('Video:', result.title);
        console.log('Summary:', result.summary);
        console.log('Keywords:', result.keywords.join(', '));
        console.log('Processing time:', result.processing_time + 's');
    })
    .catch(error => console.error('Error:', error.message));
```

**Using axios**
```javascript
const axios = require('axios');

const summarizeVideo = async (youtubeUrl) => {
    try {
        const response = await axios.post('http://localhost:8000/api/v1/summarize', {
            youtube_url: youtubeUrl
        });
        return response.data;
    } catch (error) {
        if (error.response) {
            throw new Error(`API Error: ${error.response.data.error.message}`);
        } else {
            throw new Error(`Network Error: ${error.message}`);
        }
    }
};
```

## Rate Limiting

Currently, no rate limiting is implemented. For production use, consider implementing:

- Per-IP rate limiting
- API key-based quotas
- Concurrent request limits per client

## Performance Considerations

### Response Times

| Operation | Typical Response Time |
|-----------|----------------------|
| Health check | < 100ms |
| Short video (< 5 min) | 2-5 seconds |
| Medium video (5-15 min) | 5-15 seconds |
| Long video (15-30 min) | 15-30 seconds |

### Processing Factors

Response times depend on:
- Video duration
- Transcript complexity
- LLM service response time
- Network latency
- Current server load

### Optimization Tips

1. **Caching**: Results are cached for identical URLs
2. **Concurrent Processing**: Multiple requests can be processed simultaneously
3. **Timeout Configuration**: Configurable timeouts prevent long-running requests
4. **Retry Logic**: Automatic retries for transient failures

## SDK and Client Libraries

### Official SDKs

Currently, no official SDKs are available. The API follows standard REST principles and can be easily integrated with any HTTP client.

### Community Libraries

Consider creating client libraries for popular languages:
- Python: `youtube-summarizer-python`
- JavaScript/Node.js: `youtube-summarizer-js`
- Java: `youtube-summarizer-java`
- Go: `youtube-summarizer-go`

## Interactive Documentation

The API provides interactive documentation through:

- **Swagger UI**: Available at `/api/docs`
- **ReDoc**: Available at `/api/redoc`

These interfaces allow you to:
- Explore all endpoints
- Test requests directly
- View response schemas
- Download OpenAPI specification

## Support and Contact

For API support:
1. Check the troubleshooting section in the main README
2. Review the error response messages
3. Check application logs for detailed error information
4. Open an issue with reproduction steps

## Changelog

### Version 1.0.0
- Initial API release
- Core summarization functionality
- Health check and metrics endpoints
- Comprehensive error handling
- OpenAPI documentation