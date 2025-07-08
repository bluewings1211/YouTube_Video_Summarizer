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

- Python 3.11+
- Docker and Docker Compose
- PostgreSQL 15+ (or use Docker Compose)
- OpenAI or Anthropic API key

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

### 3. Start the Service

#### Using Docker Compose (Recommended)

```bash
# Build and start all services
make build
make up

# View logs
make logs

# Check service health
curl http://localhost:8000/health
```

#### Using Python Virtual Environment

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set Python path
export PYTHONPATH=$PWD/src

# Run database migrations
alembic upgrade head

# Run the application
python -m uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Test the API

```bash
# Health check
curl http://localhost:8000/health

# Summarize a video
curl -X POST "http://localhost:8000/api/v1/summarize" \
  -H "Content-Type: application/json" \
  -d '{"youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"}'
```

## API Documentation

### Video Processing API

#### Endpoint: POST /api/v1/summarize

Processes a YouTube video URL and returns comprehensive analysis including summary, keywords, and timestamped segments.

#### Request

```json
{
  "youtube_url": "https://www.youtube.com/watch?v=VIDEO_ID"
}
```

#### Response

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

#### Error Responses

- **400 Bad Request**: Invalid YouTube URL or unsupported video type
- **404 Not Found**: Video not found or unavailable
- **422 Unprocessable Entity**: Video too long (>30 minutes) or no transcript available
- **500 Internal Server Error**: Processing failure or API errors

### Additional Endpoints

- **GET /health**: Service health check
- **GET /health/database**: Database connectivity check
- **GET /**: API information and documentation
- **GET /api/v1/history/videos**: List processed videos with pagination and filtering
- **GET /api/v1/history/videos/{video_id}**: Get detailed information for a specific video
- **GET /api/v1/history/statistics**: Get processing statistics

### History API

#### Endpoint: GET /api/v1/history/videos

Lists all processed videos with pagination and filtering capabilities.

**Query Parameters:**
- `page` (int): Page number (default: 1)
- `page_size` (int): Items per page, 1-100 (default: 20)
- `sort_by` (str): Sort field - "created_at", "title", "duration" (default: "created_at")
- `sort_order` (str): "asc" or "desc" (default: "desc")
- `date_from` (str): Filter videos from date (ISO format)
- `date_to` (str): Filter videos to date (ISO format)
- `keywords` (str): Filter by keywords (comma-separated)
- `title_search` (str): Search in video titles

**Example Response:**
```json
{
  "videos": [
    {
      "id": "uuid",
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

#### Endpoint: GET /api/v1/history/videos/{video_id}

Retrieves detailed information for a specific processed video.

**Example Response:**
```json
{
  "id": "uuid",
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
| `MAX_VIDEO_DURATION` | Max video length (seconds) | No | `1800` |

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
   Solution: Use videos under 30 minutes
   ```

4. **No Transcript Available**
   ```
   Error: No transcript found for video
   Solution: Use videos with auto-generated or manual transcripts
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
# Check container logs
docker-compose logs app

# Access container shell
docker-compose exec app bash

# Check service health
curl http://localhost:8000/health
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