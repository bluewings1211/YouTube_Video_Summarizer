# YouTube Video Summarizer Web Service

A comprehensive web service that automatically extracts, summarizes, and analyzes YouTube video content using intelligent workflow orchestration. Built with FastAPI, PocketFlow, PostgreSQL, and modern AI/LLM integration.

## Features

- **Intelligent Video Processing**: Automatically extracts transcripts from YouTube videos
- **AI-Powered Summarization**: Generates concise 500-word summaries using OpenAI or Anthropic models
- **Timestamped Navigation**: Creates clickable URLs with timestamps for key video segments
- **Keyword Extraction**: Identifies 5-8 relevant keywords for content categorization
- **Multi-language Support**: Handles Chinese and English video content
- **Database Integration**: PostgreSQL-backed persistent storage for all processed videos
- **History APIs**: RESTful endpoints for querying and filtering processed video history
- **Duplicate Detection**: Intelligent handling of previously processed videos
- **Real-time Processing**: Fast API responses with comprehensive error handling
- **Containerized Deployment**: Full Docker and Docker Compose support with PostgreSQL
- **Comprehensive Testing**: 90%+ test coverage with unit and integration tests

## Quick Start

### Prerequisites

- Python 3.11+ (for local development)
- Docker and Docker Compose (required)
- OpenAI or Anthropic API key

### 🚀 Quick Setup (5 minutes)

For the fastest setup using Docker Compose:

```bash
# 1. Clone the repository
git clone <repository-url>
cd youtube-summarizer-wave

# 2. Create .env file with your API key
echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
echo "DATABASE_URL=postgresql+asyncpg://postgres:password@localhost:5432/youtube_summarizer" >> .env

# 3. Start all services (PostgreSQL + App)
docker-compose up -d

# 4. Run database migrations
docker-compose exec app alembic upgrade head

# 5. Test the service
curl http://localhost:8000/health
curl http://localhost:8000/health/database

# 🎉 Ready to use!
curl -X POST "http://localhost:8000/api/v1/summarize" \
  -H "Content-Type: application/json" \
  -d '{"youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"}'
```

### 1. Clone and Setup

```bash
git clone <repository-url>
cd youtube-summarizer-wave
```

### 2. Environment Configuration

Create a `.env` file in the project root:

```bash
# API Keys (at least one required)
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# LLM Configuration
DEFAULT_LLM_PROVIDER=openai  # or 'anthropic'
OPENAI_MODEL=gpt-4-turbo-preview
ANTHROPIC_MODEL=claude-3-sonnet-20240229

# Database Configuration
DATABASE_URL=postgresql+asyncpg://postgres:password@localhost:5432/youtube_summarizer
POSTGRES_DB=youtube_summarizer
POSTGRES_USER=postgres
POSTGRES_PASSWORD=password
POSTGRES_HOST=localhost
POSTGRES_PORT=5432

# Application Settings
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=info
PORT=8000
HOST=0.0.0.0
```

### 3. Start the Services

All external dependencies (PostgreSQL, Redis) are managed through Docker Compose:

```bash
# Start all services (PostgreSQL + Redis + App)
docker-compose up -d

# Wait for services to be ready (check logs)
docker-compose logs

# Run database migrations
docker-compose exec app alembic upgrade head

# Verify all services are healthy
curl http://localhost:8000/health
curl http://localhost:8000/health/database
```

### 4. Alternative: Development with Python Virtual Environment

For development without Docker (requires Docker Compose for PostgreSQL/Redis):

```bash
# Start only external services
docker-compose up -d postgres redis

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set Python path
export PYTHONPATH=$PWD/src

# Set database URL for local development (optional - alembic.ini has localhost default)
export DATABASE_URL=postgresql+asyncpg://postgres:password@localhost:5432/youtube_summarizer

# Run database migrations
alembic upgrade head

# Run the application
python -m uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload
```

#### Database Configuration Notes

The project uses different database hostnames depending on the environment:

- **Docker environment**: Uses `postgres` as hostname (service name in docker-compose)
- **Local development**: Uses `localhost` as hostname

The migration system automatically handles this:
- `alembic/env.py` checks for `DATABASE_URL` environment variable first
- If not set, uses the default in `alembic.ini` (configured for localhost)
- For Docker containers, the `DATABASE_URL` is automatically set to use `postgres` hostname

### 5. Test the API

```bash
# Health check
curl http://localhost:8000/health

# Summarize a video
curl -X POST "http://localhost:8000/api/v1/summarize" \
  -H "Content-Type: application/json" \
  -d '{"youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"}'
```

## API Documentation

The YouTube Summarizer Web Service provides a comprehensive REST API built with FastAPI. The API includes **11 endpoints** across different categories for video processing, history management, health monitoring, and documentation.

### 📊 API Endpoints Overview

| Category | Endpoint | Method | Description |
|----------|----------|---------|-------------|
| **Core** | `/api/v1/summarize` | POST | Main video summarization |
| **System** | `/health` | GET | Service health check |
| **System** | `/health/database` | GET | Database connectivity check |
| **System** | `/metrics` | GET | Application metrics |
| **History** | `/api/v1/history/videos` | GET | List processed videos |
| **History** | `/api/v1/history/videos/{video_id}` | GET | Get video details |
| **History** | `/api/v1/history/statistics` | GET | Video statistics |
| **History** | `/api/v1/history/health` | GET | History API health |
| **Docs** | `/api/docs` | GET | Interactive API documentation |
| **Docs** | `/api/redoc` | GET | ReDoc documentation |
| **Info** | `/` | GET | Root endpoint with API info |

### 🎯 Core API Endpoints

#### POST /api/v1/summarize
**Main Video Summarization Endpoint**

Processes a YouTube video URL and returns comprehensive analysis including summary, keywords, and timestamped segments.

**Request:**
```json
{
  "youtube_url": "https://www.youtube.com/watch?v=VIDEO_ID",
  "reprocess_policy": "skip_existing"  // optional: "skip_existing" or "force_reprocess"
}
```

**Response:**
```json
{
  "video_id": "VIDEO_ID",
  "title": "Video Title",
  "duration": "PT25M30S",
  "summary": "Generated 500-word summary of the video content...",
  "timestamped_segments": [
    {
      "timestamp": "00:01:30",
      "url": "https://www.youtube.com/watch?v=VIDEO_ID&t=90s",
      "description": "Introduction to key concept",
      "importance_rating": 8
    }
  ],
  "keywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"],
  "processing_time": "2.5s"
}
```

**Error Responses:**
- **400 Bad Request**: Invalid YouTube URL or unsupported video type
- **404 Not Found**: Video not found or unavailable
- **422 Unprocessable Entity**: Video too long (>30 minutes) or no transcript available
- **500 Internal Server Error**: Processing failure or API errors

#### GET /
**Root Information Endpoint**

Returns API information and available endpoints.

**Response:**
```json
{
  "service": "YouTube Video Summarizer",
  "version": "1.0.0",
  "endpoints": {
    "summarize": "/api/v1/summarize",
    "health": "/health",
    "docs": "/api/docs"
  },
  "features": ["AI Summarization", "Keyword Extraction", "Timestamped Segments"]
}
```

### 🏥 Health & Monitoring Endpoints

#### GET /health
**Service Health Check**

Comprehensive health check with database status.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-07-09T10:30:00Z",
  "components": {
    "workflow": "healthy",
    "database": "healthy"
  },
  "uptime": "2h 45m 30s"
}
```

#### GET /health/database
**Database Health Check**

Detailed database connectivity and performance metrics.

**Response:**
```json
{
  "status": "healthy",
  "connection_pool": {
    "active_connections": 5,
    "idle_connections": 10,
    "max_connections": 20
  },
  "response_time_ms": 12.5,
  "last_error": null
}
```

#### GET /metrics
**Application Metrics**

Returns application metrics for monitoring and observability.

**Response:**
```json
{
  "uptime_seconds": 9930,
  "workflow_status": "healthy",
  "database_status": "healthy",
  "processed_videos_count": 150,
  "average_processing_time": 2.8
}
```

### 📚 History API Endpoints

#### GET /api/v1/history/videos
**List Processed Videos**

Lists all processed videos with pagination and filtering capabilities.

**Query Parameters:**
- `page` (int): Page number (1-based, default: 1)
- `page_size` (int): Items per page, 1-100 (default: 20)
- `sort_by` (str): Sort field - "created_at", "updated_at", "title" (default: "created_at")
- `sort_order` (str): Sort order - "asc", "desc" (default: "desc")
- `date_from` (date): Start date filter (YYYY-MM-DD format)
- `date_to` (date): End date filter (YYYY-MM-DD format)
- `keywords` (str): Keywords to search for (comma-separated)
- `title_search` (str): Search in video titles

**Example Response:**
```json
{
  "videos": [
    {
      "id": "uuid-here",
      "video_id": "dQw4w9WgXcQ",
      "title": "Video Title",
      "duration": "PT25M30S",
      "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
      "created_at": "2024-07-08T10:30:00Z",
      "summary_preview": "First 100 characters of summary...",
      "keywords": ["keyword1", "keyword2"]
    }
  ],
  "pagination": {
    "page": 1,
    "page_size": 20,
    "total_items": 150,
    "total_pages": 8,
    "has_next": true,
    "has_prev": false
  }
}
```

#### GET /api/v1/history/videos/{video_id}
**Get Video Details**

Retrieves detailed information for a specific processed video.

**Path Parameters:**
- `video_id` (int): Video ID from the database

**Example Response:**
```json
{
  "id": "uuid-here",
  "video_id": "dQw4w9WgXcQ",
  "title": "Complete Video Title",
  "duration": "PT25M30S",
  "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
  "created_at": "2024-07-08T10:30:00Z",
  "transcript": {
    "content": "Full transcript text...",
    "language": "en"
  },
  "summary": {
    "content": "Complete 500-word summary...",
    "processing_time": 2.5
  },
  "keywords": ["keyword1", "keyword2", "keyword3"],
  "timestamped_segments": [
    {
      "timestamp": "00:01:30",
      "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=90s",
      "description": "Introduction to key concept",
      "importance_rating": 8
    }
  ],
  "processing_metadata": {
    "workflow_params": {},
    "status": "completed",
    "error_info": null
  }
}
```

#### GET /api/v1/history/statistics
**Get Video Statistics**

Returns statistics about processed videos.

**Response:**
```json
{
  "total_videos": 150,
  "completed_videos": 145,
  "failed_videos": 5,
  "completion_rate": 96.7,
  "average_processing_time": 2.8,
  "processing_status_counts": {
    "completed": 145,
    "failed": 5,
    "processing": 0
  }
}
```

#### GET /api/v1/history/health
**History API Health Check**

Health check endpoint specifically for the history API.

**Response:**
```json
{
  "status": "healthy",
  "service": "history_api",
  "database_connection": "healthy"
}
```

### 📖 Documentation Endpoints

#### GET /api/docs
**Interactive API Documentation**

FastAPI's automatically generated Swagger UI for interactive API documentation and testing.

#### GET /api/redoc
**ReDoc API Documentation**

Alternative API documentation interface using ReDoc with a clean, responsive design.

### 🔧 Usage Examples

**Basic video summarization:**
```bash
curl -X POST "http://localhost:8000/api/v1/summarize" \
  -H "Content-Type: application/json" \
  -d '{"youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"}'
```

**Get paginated video history:**
```bash
curl "http://localhost:8000/api/v1/history/videos?page=1&page_size=10&sort_by=created_at&sort_order=desc"
```

**Search videos by keywords:**
```bash
curl "http://localhost:8000/api/v1/history/videos?keywords=python,tutorial&title_search=introduction"
```

**Get video details:**
```bash
curl "http://localhost:8000/api/v1/history/videos/123"
```

**Check system health:**
```bash
curl "http://localhost:8000/health"
curl "http://localhost:8000/health/database"
```

### 🔒 Authentication & Security

Currently, the API does not require authentication. For production deployment, consider implementing:

- API key authentication
- Rate limiting
- Input validation and sanitization
- CORS configuration
- HTTPS enforcement

### 📊 Response Format

All API responses follow a consistent JSON format with appropriate HTTP status codes:

- **200 OK**: Successful request
- **400 Bad Request**: Invalid input or request format
- **404 Not Found**: Resource not found
- **422 Unprocessable Entity**: Validation error
- **500 Internal Server Error**: Server error

Error responses include detailed error information:
```json
{
  "error": "Invalid YouTube URL",
  "message": "The provided URL is not a valid YouTube video URL",
  "details": {
    "url": "invalid-url",
    "code": "INVALID_URL"
  }
}
```

## Development

### Project Structure

```
youtube-summarizer-wave/
├── src/                    # Source code
│   ├── app.py             # FastAPI application
│   ├── config.py          # Configuration management
│   ├── database/          # Database layer
│   │   ├── __init__.py    # Database package init
│   │   ├── models.py      # SQLAlchemy database models
│   │   ├── connection.py  # Database connection management
│   │   ├── exceptions.py  # Database exception classes
│   │   ├── monitor.py     # Database monitoring and health checks
│   │   └── maintenance.py # Database maintenance utilities
│   ├── services/          # Business logic layer
│   │   ├── __init__.py    # Services package init
│   │   ├── video_service.py # Video processing service
│   │   └── history_service.py # History query service
│   ├── api/               # API endpoints
│   │   ├── __init__.py    # API package init
│   │   └── history.py     # History API endpoints
│   ├── refactored_nodes/  # Refactored processing nodes
│   │   ├── __init__.py    # Nodes package init
│   │   ├── transcript_nodes.py # YouTube transcript nodes
│   │   ├── llm_nodes.py   # LLM processing nodes
│   │   ├── validation_nodes.py # Validation nodes
│   │   ├── summary_nodes.py # Summary generation nodes
│   │   └── keyword_nodes.py # Keyword extraction nodes
│   ├── refactored_flow/   # Refactored workflow orchestration
│   │   ├── __init__.py    # Flow package init
│   │   ├── orchestrator.py # Workflow orchestration
│   │   ├── config.py      # Flow configuration
│   │   ├── error_handler.py # Error handling
│   │   └── monitoring.py  # Flow monitoring
│   ├── refactored_utils/  # Refactored utility modules
│   │   ├── __init__.py    # Utils package init
│   │   ├── transcript_fetcher.py # Transcript fetching
│   │   ├── video_metadata.py # Video metadata
│   │   ├── url_validator.py # URL validation
│   │   ├── language_handler.py # Language detection
│   │   └── youtube_errors.py # YouTube error handling
│   └── utils/             # Legacy utility modules (being refactored)
│       ├── youtube_api.py # YouTube API integration
│       ├── call_llm.py    # LLM client
│       ├── validators.py  # Input validation
│       └── error_messages.py # Error handling
├── alembic/               # Database migrations
│   ├── versions/          # Migration versions
│   └── env.py            # Alembic environment
├── scripts/               # Utility scripts
│   ├── manage_migrations.py # Migration management
│   └── database_maintenance.py # Database maintenance CLI
├── tests/                 # Test suites
├── requirements.txt       # Python dependencies
├── alembic.ini           # Database migration configuration
├── Dockerfile            # Container configuration
├── docker-compose.yml    # Multi-service setup with PostgreSQL
└── Makefile             # Development commands
```

### Development Commands

```bash
# Start development environment
make up

# Run tests
make test

# Run linting
make lint

# Format code
make format

# Run database migrations
make migrate

# Create new migration
make migration

# View logs
make logs

# Access container shell
make shell

# Stop services
make down

# Clean up
make clean
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_app.py

# Run integration tests
pytest tests/test_integration_e2e.py

# Run database tests
pytest tests/test_database.py

# Run history API tests
pytest tests/test_history_api.py
```

### Code Quality

The project maintains high code quality standards:

- **Testing**: 90%+ test coverage with pytest
- **Type Checking**: mypy for static type analysis
- **Linting**: flake8 for code style enforcement
- **Formatting**: black for consistent code formatting
- **Documentation**: Comprehensive docstrings and comments

## Architecture

### Workflow Orchestration

The service uses PocketFlow for workflow orchestration with the following processing nodes:

1. **YouTubeTranscriptNode**: Extracts video metadata and transcript
2. **SummarizationNode**: Generates AI-powered summaries
3. **TimestampNode**: Creates timestamped segments with importance ratings
4. **KeywordExtractionNode**: Extracts relevant keywords

### Database Architecture

**PostgreSQL Database Schema:**
- **Videos**: Core video information (ID, title, duration, URL)
- **Transcripts**: Video transcripts with language detection
- **Summaries**: AI-generated summaries with processing metadata
- **Keywords**: Extracted keywords in JSON format
- **TimestampedSegments**: Timestamped video segments with importance ratings
- **ProcessingMetadata**: Workflow execution metadata and error tracking

**Key Features:**
- Async SQLAlchemy 2.0+ models with validation
- Automatic duplicate detection and handling
- Connection pooling and health monitoring
- Database migration system with Alembic
- Comprehensive error handling and recovery

### LLM Integration

Supports multiple LLM providers:

- **OpenAI**: GPT-4 Turbo, GPT-3.5 Turbo
- **Anthropic**: Claude-3 Sonnet, Claude-3 Haiku

### Error Handling

Comprehensive error handling with:

- Input validation for YouTube URLs
- Video accessibility checks (private, live, no transcript)
- Duration validation (30-minute limit)
- Retry mechanisms with exponential backoff
- Detailed error messages and logging

## Production Deployment

### Docker Deployment

```bash
# Build production image
docker build -t youtube-summarizer .

# Run with environment variables
docker run -d \
  -p 8000:8000 \
  -e OPENAI_API_KEY=your_key \
  -e ENVIRONMENT=production \
  --name youtube-summarizer \
  youtube-summarizer
```

### Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `OPENAI_API_KEY` | OpenAI API key | Yes* | - |
| `ANTHROPIC_API_KEY` | Anthropic API key | Yes* | - |
| `DATABASE_URL` | PostgreSQL connection URL | Yes | - |
| `POSTGRES_DB` | Database name | No | `youtube_summarizer` |
| `POSTGRES_USER` | Database user | No | `postgres` |
| `POSTGRES_PASSWORD` | Database password | No | `password` |
| `POSTGRES_HOST` | Database host | No | `localhost` |
| `POSTGRES_PORT` | Database port | No | `5432` |
| `DEFAULT_LLM_PROVIDER` | LLM provider to use | No | `openai` |
| `ENVIRONMENT` | Environment (dev/prod) | No | `development` |
| `PORT` | Server port | No | `8000` |
| `LOG_LEVEL` | Logging level | No | `info` |
| `MAX_VIDEO_DURATION` | Max video length (seconds) - videos longer than this will be rejected | No | `1800` (30 min) |

*At least one API key is required

### Performance Tuning

- **Concurrent Processing**: FastAPI with async/await support
- **Database Connection Pooling**: Optimized PostgreSQL connections
- **Caching**: Redis integration for transcript caching
- **Resource Limits**: Configurable timeouts and retry policies
- **Health Checks**: Built-in health monitoring endpoints for app and database
- **Duplicate Detection**: Avoid reprocessing previously analyzed videos

## Monitoring and Logging

### Application Logs

Structured JSON logging with:

- Request/response logging
- Processing time metrics
- Error tracking and debugging
- Performance monitoring

### Health Checks

- **Application Health**: `/health` endpoint
- **Database Health**: `/health/database` endpoint with connection pool metrics
- **Container Health**: Docker healthcheck integration
- **Dependency Health**: LLM API and YouTube API status

### Database Monitoring

- **Connection Pool Metrics**: Active/idle connection tracking
- **Query Performance**: Response time monitoring
- **Error Tracking**: Database exception logging and alerting
- **Maintenance Tools**: Automated cleanup and statistics

## Troubleshooting

### Common Issues

1. **API Key Not Set**
   ```
   Error: No valid API key found
   Solution: Set OPENAI_API_KEY or ANTHROPIC_API_KEY in .env
   ```

2. **Video Not Accessible**
   ```
   Error: Video is private or unavailable
   Solution: Use public videos with available transcripts
   ```

3. **Video Too Long**
   ```
   Error: Video exceeds maximum duration limit
   Solution: Use videos under 30 minutes, or adjust MAX_VIDEO_DURATION in .env file
   # To allow 60-minute videos, set:
   MAX_VIDEO_DURATION=3600
   ```

4. **No Transcript Available**
   ```
   Error: No transcript found for video
   Solution: Use videos with auto-generated or manual transcripts
   ```

5. **Database Connection Error**
   ```
   Error: Connection to database failed
   Solution: Check PostgreSQL container is running
   
   # Check container status
   docker-compose ps postgres
   
   # Check PostgreSQL logs
   docker-compose logs postgres
   
   # Restart PostgreSQL service
   docker-compose restart postgres
   
   # Test database connection
   docker-compose exec postgres psql -U postgres -d youtube_summarizer -c "SELECT 1;"
   ```

6. **Migration Errors**
   ```
   Error: Migration failed or database schema mismatch
   Solution: Reset and rerun migrations
   
   # Check current migration
   docker-compose exec app alembic current
   
   # Reset to base (WARNING: This will drop all data)
   docker-compose exec app alembic downgrade base
   docker-compose exec app alembic upgrade head
   
   # Or create new migration for schema fixes
   docker-compose exec app alembic revision --autogenerate -m "Fix schema"
   ```

### Debug Mode

Enable debug mode for detailed logging:

```bash
# In .env file
DEBUG=true
LOG_LEVEL=debug
```

### Container Troubleshooting

```bash
# Check all container statuses
docker-compose ps

# Check logs for all services
docker-compose logs

# Check logs for specific service
docker-compose logs app
docker-compose logs postgres
docker-compose logs redis

# Access container shell
docker-compose exec app bash
docker-compose exec postgres psql -U postgres -d youtube_summarizer

# Restart all services
docker-compose restart

# Restart specific service
docker-compose restart postgres

# Check service health
curl http://localhost:8000/health
curl http://localhost:8000/health/database
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

### Code Style

- Follow PEP 8 conventions
- Use black for code formatting
- Include comprehensive docstrings
- Maintain test coverage above 90%

## License

MIT License - see LICENSE file for details.

## Support

For issues and questions:

1. Check the troubleshooting guide
2. Review application logs
3. Open an issue with detailed error information
4. Include environment details and reproduction steps