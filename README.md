# YouTube Video Summarizer Web Service

A comprehensive web service that automatically extracts, summarizes, and analyzes YouTube video content using intelligent workflow orchestration. Built with FastAPI, PocketFlow, and modern AI/LLM integration.

## Features

- **Intelligent Video Processing**: Automatically extracts transcripts from YouTube videos
- **AI-Powered Summarization**: Generates concise 500-word summaries using OpenAI or Anthropic models
- **Timestamped Navigation**: Creates clickable URLs with timestamps for key video segments
- **Keyword Extraction**: Identifies 5-8 relevant keywords for content categorization
- **Multi-language Support**: Handles Chinese and English video content
- **Real-time Processing**: Fast API responses with comprehensive error handling
- **Containerized Deployment**: Full Docker and Docker Compose support
- **Comprehensive Testing**: 90%+ test coverage with unit and integration tests

## Quick Start

### Prerequisites

- Python 3.11+
- Docker and Docker Compose
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

### Endpoint: POST /api/v1/summarize

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
- **GET /**: API information and documentation

## Development

### Project Structure

```
youtube-summarizer-wave/
├── src/                    # Source code
│   ├── app.py             # FastAPI application
│   ├── flow.py            # PocketFlow workflow orchestration
│   ├── nodes.py           # Processing nodes (transcript, summarization, etc.)
│   ├── config.py          # Configuration management
│   └── utils/             # Utility modules
│       ├── youtube_api.py # YouTube API integration
│       ├── call_llm.py    # LLM client
│       ├── validators.py  # Input validation
│       └── error_messages.py # Error handling
├── tests/                 # Test suites
├── requirements.txt       # Python dependencies
├── Dockerfile            # Container configuration
├── docker-compose.yml    # Multi-service setup
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
| `DEFAULT_LLM_PROVIDER` | LLM provider to use | No | `openai` |
| `ENVIRONMENT` | Environment (dev/prod) | No | `development` |
| `PORT` | Server port | No | `8000` |
| `LOG_LEVEL` | Logging level | No | `info` |
| `MAX_VIDEO_DURATION` | Max video length (seconds) | No | `1800` |

*At least one API key is required

### Performance Tuning

- **Concurrent Processing**: FastAPI with async/await support
- **Caching**: Redis integration for transcript caching
- **Resource Limits**: Configurable timeouts and retry policies
- **Health Checks**: Built-in health monitoring endpoints

## Monitoring and Logging

### Application Logs

Structured JSON logging with:

- Request/response logging
- Processing time metrics
- Error tracking and debugging
- Performance monitoring

### Health Checks

- **Application Health**: `/health` endpoint
- **Container Health**: Docker healthcheck integration
- **Dependency Health**: LLM API and YouTube API status

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