"""
FastAPI application for YouTube Video Summarizer Web Service.

This module provides the main web API endpoints for the YouTube summarizer service,
including request validation, response formatting, and integration with the
PocketFlow workflow orchestration system.
"""

import os
import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from uvicorn import run

try:
    from .flow import YouTubeSummarizerFlow, WorkflowError
    from .config import settings
    from .utils.error_messages import (
        ErrorMessageProvider, ErrorCode, ErrorCategory, ErrorSeverity,
        get_youtube_error, get_llm_error, get_network_error
    )
    from .utils.validators import validate_youtube_url_detailed, URLValidationResult
except ImportError:
    # For testing - try absolute imports
    try:
        from flow import YouTubeSummarizerFlow, WorkflowError
        from config import settings
        from utils.error_messages import (
            ErrorMessageProvider, ErrorCode, ErrorCategory, ErrorSeverity,
            get_youtube_error, get_llm_error, get_network_error
        )
        from utils.validators import validate_youtube_url_detailed, URLValidationResult
    except ImportError:
        # Create mock implementations for testing
        class YouTubeSummarizerFlow:
            def __init__(self, **kwargs):
                pass
            def run(self, input_data):
                return {'status': 'mock_success', 'data': {}}
        
        class WorkflowError:
            def __init__(self, **kwargs):
                pass
        
        class MockSettings:
            def __init__(self):
                self.app_name = "YouTube Summarizer"
                self.app_version = "1.0.0"
                self.log_level = "info"
        
        settings = MockSettings()
        
        # Mock error handling classes
        class ErrorMessageProvider:
            @classmethod
            def get_error_details(cls, error_code, **kwargs):
                return None
            @classmethod
            def format_error_response(cls, error_details, **kwargs):
                return {"error": "Mock error"}
        
        class ErrorCode:
            INVALID_URL_FORMAT = "E1001"
        
        def validate_youtube_url_detailed(url):
            class MockResult:
                is_valid = True
                result_type = None
                error_message = None
            return MockResult()
        
        def get_youtube_error(msg, vid=""): return None
        def get_llm_error(msg, prov=""): return None
        def get_network_error(msg, url=""): return None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/app.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

# Global workflow instance for reuse
workflow_instance: Optional[YouTubeSummarizerFlow] = None
app_start_time: float = 0.0

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager for startup and shutdown events.
    """
    global workflow_instance, app_start_time
    
    # Startup
    app_start_time = time.time()
    logger.info("Initializing YouTube Summarizer API")
    try:
        # Create workflow instance with production configuration
        workflow_instance = YouTubeSummarizerFlow(
            enable_monitoring=True,
            enable_fallbacks=True,
            max_retries=2,
            timeout_seconds=300
        )
        logger.info("Workflow instance initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize workflow: {str(e)}")
        workflow_instance = None
    
    # Ensure logs directory exists
    os.makedirs('logs', exist_ok=True)
    
    logger.info("YouTube Summarizer API startup complete")
    yield
    
    # Shutdown
    logger.info("Shutting down YouTube Summarizer API")
    if workflow_instance:
        try:
            workflow_instance.reset_workflow()
            logger.info("Workflow instance cleaned up")
        except Exception as e:
            logger.warning(f"Error during workflow cleanup: {str(e)}")

# Initialize FastAPI app
app = FastAPI(
    title=settings.app_name,
    description="AI-powered YouTube video summarization service with timestamps and keyword extraction",
    version=settings.app_version,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    Middleware to log all HTTP requests and responses for monitoring.
    """
    start_time = time.time()
    request_id = f"req_{int(start_time * 1000)}"
    
    # Log incoming request
    client_ip = request.client.host if request.client else "unknown"
    user_agent = request.headers.get("user-agent", "unknown")
    
    logger.info(f"[{request_id}] Incoming request: {request.method} {request.url.path} "
               f"from {client_ip} - User-Agent: {user_agent}")
    
    # Add request ID to request state for use in endpoints
    request.state.request_id = request_id
    
    try:
        # Process request
        response = await call_next(request)
        
        # Calculate processing time
        process_time = time.time() - start_time
        
        # Log response
        logger.info(f"[{request_id}] Response: {response.status_code} "
                   f"- Processing time: {process_time:.3f}s")
        
        # Add performance headers
        response.headers["X-Process-Time"] = f"{process_time:.3f}"
        response.headers["X-Request-ID"] = request_id
        
        return response
        
    except Exception as e:
        # Log error
        process_time = time.time() - start_time
        logger.error(f"[{request_id}] Request failed: {str(e)} "
                    f"- Processing time: {process_time:.3f}s", exc_info=True)
        
        # Re-raise exception for FastAPI to handle
        raise

# Pydantic models for request/response validation
class SummarizeRequest(BaseModel):
    """Request model for the summarize endpoint."""
    youtube_url: str = Field(
        ...,
        description="YouTube video URL to summarize",
        example="https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    )
    
    @validator('youtube_url')
    def validate_youtube_url(cls, v):
        """Validate YouTube URL format using enhanced validation."""
        # Use the enhanced validation system
        validation_result = validate_youtube_url_detailed(v)
        
        if not validation_result.is_valid:
            # Create detailed error using the error message provider
            error_details = ErrorMessageProvider.create_validation_error(validation_result, str(v))
            # Raise ValueError with user-friendly message
            raise ValueError(error_details.user_message)
        
        return v.strip()

class TimestampedSegment(BaseModel):
    """Model for timestamped video segments."""
    timestamp: str = Field(
        ...,
        description="Timestamp in MM:SS or HH:MM:SS format",
        example="01:30"
    )
    url: str = Field(
        ...,
        description="YouTube URL with timestamp parameter",
        example="https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=90s"
    )
    description: str = Field(
        ...,
        description="Brief description of the segment content",
        example="Introduction to key concept"
    )
    importance_rating: int = Field(
        ...,
        description="Importance rating from 1-10",
        example=8,
        ge=1,
        le=10
    )

class SummarizeResponse(BaseModel):
    """Response model for successful summarization."""
    video_id: str = Field(
        ...,
        description="YouTube video ID",
        example="dQw4w9WgXcQ"
    )
    title: str = Field(
        ...,
        description="Video title",
        example="Example Video Title"
    )
    duration: int = Field(
        ...,
        description="Video duration in seconds",
        example=1530
    )
    summary: str = Field(
        ...,
        description="Generated summary (max 500 words)",
        example="This video discusses..."
    )
    timestamped_segments: List[TimestampedSegment] = Field(
        ...,
        description="List of important timestamped segments",
        example=[]
    )
    keywords: List[str] = Field(
        ...,
        description="Extracted keywords (5-8 items)",
        example=["keyword1", "keyword2", "keyword3"]
    )
    processing_time: float = Field(
        ...,
        description="Processing time in seconds",
        example=2.5
    )

class ErrorResponse(BaseModel):
    """Enhanced error response model with detailed error information."""
    error: Dict[str, Any] = Field(
        ...,
        description="Detailed error information",
        example={
            "code": "E1001",
            "category": "validation",
            "severity": "medium",
            "title": "Invalid YouTube URL Format",
            "message": "Please provide a valid YouTube video URL",
            "suggested_actions": ["Check URL format", "Try again"],
            "is_recoverable": True,
            "timestamp": "2024-01-15T10:30:00Z"
        }
    )

# Enhanced exception handlers using standardized error messages
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Enhanced HTTP exception handler with detailed error information."""
    request_id = getattr(request.state, 'request_id', 'unknown')
    logger.error(f"[{request_id}] HTTP {exc.status_code} error: {exc.detail}")
    
    # Try to map HTTP errors to our error codes
    error_code = ErrorCode.INTERNAL_SERVER_ERROR
    if exc.status_code == 400:
        error_code = ErrorCode.INVALID_URL_FORMAT
    elif exc.status_code == 404:
        error_code = ErrorCode.VIDEO_NOT_FOUND
    elif exc.status_code == 408:
        error_code = ErrorCode.REQUEST_TIMEOUT
    elif exc.status_code == 429:
        error_code = ErrorCode.LLM_RATE_LIMITED
    elif exc.status_code == 503:
        error_code = ErrorCode.SERVICE_UNAVAILABLE
    
    error_details = ErrorMessageProvider.get_error_details(
        error_code=error_code,
        additional_context=str(exc.detail),
        technical_details=f"HTTP {exc.status_code}: {exc.detail}"
    )
    
    response_content = ErrorMessageProvider.format_error_response(
        error_details=error_details,
        include_technical_details=False
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content=response_content
    )

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Enhanced ValueError handler with detailed error information."""
    request_id = getattr(request.state, 'request_id', 'unknown')
    logger.error(f"[{request_id}] Validation error: {str(exc)}")
    
    # Try to determine the specific validation error
    error_message = str(exc)
    error_code = ErrorMessageProvider.get_error_by_pattern(error_message)
    
    if not error_code:
        error_code = ErrorCode.INVALID_URL_FORMAT  # Default for validation errors
    
    error_details = ErrorMessageProvider.get_error_details(
        error_code=error_code,
        additional_context=error_message,
        technical_details=f"ValueError: {error_message}"
    )
    
    response_content = ErrorMessageProvider.format_error_response(
        error_details=error_details,
        include_technical_details=False
    )
    
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=response_content
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Enhanced general exception handler with detailed error information."""
    request_id = getattr(request.state, 'request_id', 'unknown')
    logger.error(f"[{request_id}] Unexpected error: {str(exc)}", exc_info=True)
    
    # Try to categorize the error
    error_message = str(exc)
    error_code = ErrorMessageProvider.get_error_by_pattern(error_message)
    
    if not error_code:
        error_code = ErrorCode.INTERNAL_SERVER_ERROR
    
    error_details = ErrorMessageProvider.get_error_details(
        error_code=error_code,
        additional_context=error_message,
        technical_details=f"{type(exc).__name__}: {error_message}"
    )
    
    response_content = ErrorMessageProvider.format_error_response(
        error_details=error_details,
        include_technical_details=False
    )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=response_content
    )

# Health check endpoint
@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "version": settings.app_version,
        "workflow_ready": workflow_instance is not None
    }

# Monitoring/metrics endpoint
@app.get("/metrics", tags=["Monitoring"])
async def get_metrics():
    """Get application metrics for monitoring."""
    try:
        uptime = time.time() - app_start_time if 'app_start_time' in globals() else 0
        
        # Basic metrics
        metrics = {
            "uptime_seconds": uptime,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "version": settings.app_version,
            "workflow_status": {
                "initialized": workflow_instance is not None,
                "ready": workflow_instance is not None
            }
        }
        
        # Add workflow metrics if available
        if workflow_instance and hasattr(workflow_instance, 'get_workflow_status'):
            try:
                workflow_status = workflow_instance.get_workflow_status()
                metrics["workflow_status"].update(workflow_status)
            except Exception as e:
                logger.warning(f"Failed to get workflow metrics: {str(e)}")
        
        return metrics
        
    except Exception as e:
        logger.error(f"Failed to generate metrics: {str(e)}")
        return {
            "error": "METRICS_ERROR",
            "message": "Failed to generate metrics",
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "description": "AI-powered YouTube video summarization service",
        "endpoints": {
            "summarize": "/api/v1/summarize",
            "health": "/health",
            "docs": "/api/docs"
        },
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }

# Main API endpoint
@app.post(
    "/api/v1/summarize",
    response_model=SummarizeResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        422: {"model": ErrorResponse, "description": "Validation Error"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"}
    },
    tags=["Summarization"],
    summary="Summarize YouTube Video",
    description="Extract transcript, generate summary, timestamps, and keywords from a YouTube video"
)
async def summarize_video(request: SummarizeRequest, http_request: Request):
    """
    Summarize a YouTube video with AI-powered analysis.
    
    This endpoint accepts a YouTube URL and returns:
    - Video metadata (title, duration, video ID)
    - AI-generated summary (max 500 words)
    - Timestamped segments with importance ratings
    - Extracted keywords (5-8 items)
    - Processing performance metrics
    
    The service supports public videos with available transcripts in English and Chinese.
    """
    start_time = time.time()
    request_id = getattr(http_request.state, 'request_id', f"req_{int(start_time * 1000)}")
    
    # Log incoming request details
    logger.info(f"[{request_id}] Received summarization request for URL: {request.youtube_url}")
    
    try:
        # Validate workflow instance is available
        if workflow_instance is None:
            logger.error(f"[{request_id}] Workflow instance not initialized")
            error_details = ErrorMessageProvider.get_error_details(
                error_code=ErrorCode.SERVICE_UNAVAILABLE,
                technical_details="Workflow engine not initialized"
            )
            response_content = ErrorMessageProvider.format_error_response(error_details)
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content=response_content
            )
        
        # Prepare input data for workflow
        input_data = {
            'youtube_url': request.youtube_url,
            'processing_start_time': datetime.utcnow().isoformat(),
            'request_id': request_id
        }
        
        # Log workflow execution start
        logger.info(f"[{request_id}] Starting workflow execution")
        
        # Execute workflow
        try:
            workflow_result = workflow_instance.run(input_data)
        except Exception as workflow_error:
            logger.error(f"[{request_id}] Workflow execution failed: {str(workflow_error)}")
            
            # Handle specific workflow errors
            if isinstance(workflow_error, WorkflowError):
                return await handle_workflow_error(workflow_error, request_id)
            
            # Handle general workflow errors using standardized error messages
            error_message = str(workflow_error)
            
            # Determine appropriate error code based on error message patterns
            error_code = ErrorMessageProvider.get_error_by_pattern(error_message)
            
            # If no pattern match, try specific error classifications
            if not error_code:
                if "private" in error_message.lower():
                    error_code = ErrorCode.VIDEO_PRIVATE
                elif "live" in error_message.lower() or "stream" in error_message.lower():
                    error_code = ErrorCode.VIDEO_LIVE_STREAM
                elif "transcript" in error_message.lower() and "not" in error_message.lower():
                    error_code = ErrorCode.NO_TRANSCRIPT_AVAILABLE
                elif "duration" in error_message.lower() or "long" in error_message.lower():
                    error_code = ErrorCode.VIDEO_TOO_LONG
                elif "invalid" in error_message.lower() or "not found" in error_message.lower():
                    error_code = ErrorCode.VIDEO_NOT_FOUND
                elif "timeout" in error_message.lower():
                    error_code = ErrorCode.PROCESSING_TIMEOUT
                elif "network" in error_message.lower() or "connection" in error_message.lower():
                    error_code = ErrorCode.CONNECTION_FAILED
                else:
                    error_code = ErrorCode.WORKFLOW_EXECUTION_FAILED
            
            # Get detailed error information
            error_details = ErrorMessageProvider.get_error_details(
                error_code=error_code,
                additional_context=error_message,
                technical_details=f"Workflow error: {error_message}"
            )
            
            # Determine appropriate HTTP status code
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
            if error_details.category == ErrorCategory.VALIDATION:
                status_code = status.HTTP_400_BAD_REQUEST
            elif error_details.category == ErrorCategory.YOUTUBE_API:
                status_code = status.HTTP_400_BAD_REQUEST
            elif error_details.category == ErrorCategory.TIMEOUT:
                status_code = status.HTTP_408_REQUEST_TIMEOUT
            elif error_details.category == ErrorCategory.RATE_LIMIT:
                status_code = status.HTTP_429_TOO_MANY_REQUESTS
            elif error_details.category == ErrorCategory.NETWORK:
                status_code = status.HTTP_502_BAD_GATEWAY
            
            response_content = ErrorMessageProvider.format_error_response(error_details)
            return JSONResponse(
                status_code=status_code,
                content=response_content
            )
        
        # Check workflow execution result
        if workflow_result.get('status') == 'failed':
            logger.error(f"[{request_id}] Workflow failed: {workflow_result.get('error', {}).get('message', 'Unknown error')}")
            error_info = workflow_result.get('error', {})
            
            # Extract error information
            error_type = error_info.get('type', 'Unknown')
            error_message = error_info.get('message', 'Workflow execution failed')
            
            # Determine appropriate error code
            error_code = ErrorMessageProvider.get_error_by_pattern(error_message)
            
            if not error_code:
                if 'validation' in error_type.lower() or 'invalid' in error_message.lower():
                    error_code = ErrorCode.INVALID_URL_FORMAT
                elif 'timeout' in error_type.lower() or 'timeout' in error_message.lower():
                    error_code = ErrorCode.PROCESSING_TIMEOUT
                elif 'transcript' in error_message.lower():
                    error_code = ErrorCode.NO_TRANSCRIPT_AVAILABLE
                elif 'youtube' in error_message.lower():
                    error_code = ErrorCode.VIDEO_NOT_FOUND
                else:
                    error_code = ErrorCode.WORKFLOW_EXECUTION_FAILED
            
            # Create detailed error information
            error_details = ErrorMessageProvider.get_error_details(
                error_code=error_code,
                additional_context=error_message,
                technical_details=f"Workflow result error: {error_type} - {error_message}"
            )
            
            # Determine HTTP status code
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
            if error_details.category == ErrorCategory.VALIDATION:
                status_code = status.HTTP_400_BAD_REQUEST
            elif error_details.category == ErrorCategory.YOUTUBE_API:
                status_code = status.HTTP_400_BAD_REQUEST
            elif error_details.category == ErrorCategory.TIMEOUT:
                status_code = status.HTTP_408_REQUEST_TIMEOUT
            
            response_content = ErrorMessageProvider.format_error_response(error_details)
            return JSONResponse(
                status_code=status_code,
                content=response_content
            )
        
        # Extract data from workflow result
        workflow_data = workflow_result.get('data', {})
        processing_time = time.time() - start_time
        
        # Log successful processing
        logger.info(f"[{request_id}] Workflow completed successfully in {processing_time:.2f}s")
        
        # Format timestamped segments
        timestamped_segments = []
        timestamps = workflow_data.get('timestamps', [])
        
        for timestamp_data in timestamps:
            if isinstance(timestamp_data, dict):
                # Extract timestamp info
                timestamp_str = timestamp_data.get('timestamp', '00:00')
                description = timestamp_data.get('description', 'Key moment')
                importance = timestamp_data.get('importance_rating', 5)
                
                # Create timestamped URL
                video_id = workflow_data.get('video_id', '')
                base_url = f"https://www.youtube.com/watch?v={video_id}"
                
                # Convert timestamp to seconds for URL parameter
                timestamp_seconds = convert_timestamp_to_seconds(timestamp_str)
                timestamped_url = f"{base_url}&t={timestamp_seconds}s"
                
                timestamped_segments.append(TimestampedSegment(
                    timestamp=timestamp_str,
                    url=timestamped_url,
                    description=description,
                    importance_rating=importance
                ))
        
        # Prepare response
        response = SummarizeResponse(
            video_id=workflow_data.get('video_id', ''),
            title=workflow_data.get('title', ''),
            duration=workflow_data.get('duration', 0),
            summary=workflow_data.get('summary', ''),
            timestamped_segments=timestamped_segments,
            keywords=workflow_data.get('keywords', []),
            processing_time=processing_time
        )
        
        # Log response summary with structured data
        response_summary = {
            "request_id": request_id,
            "video_id": response.video_id,
            "video_title": response.title[:50] + "..." if len(response.title) > 50 else response.title,
            "video_duration": response.duration,
            "segments_count": len(timestamped_segments),
            "keywords_count": len(response.keywords),
            "summary_word_count": len(response.summary.split()),
            "processing_time": processing_time,
            "status": "success"
        }
        
        logger.info(f"[{request_id}] Request completed successfully", extra=response_summary)
        
        return response
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Handle unexpected errors with detailed error information
        logger.error(f"[{request_id}] Unexpected error: {str(e)}", exc_info=True)
        
        error_details = ErrorMessageProvider.get_error_details(
            error_code=ErrorCode.INTERNAL_SERVER_ERROR,
            additional_context=str(e),
            technical_details=f"Unexpected error in summarize_video: {type(e).__name__}: {str(e)}"
        )
        
        response_content = ErrorMessageProvider.format_error_response(error_details)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=response_content
        )

async def handle_workflow_error(workflow_error: WorkflowError, request_id: str) -> JSONResponse:
    """Enhanced workflow error handler with detailed error information."""
    logger.error(f"[{request_id}] Workflow error: {workflow_error.message}")
    
    # Determine appropriate error code based on workflow error
    error_code = ErrorMessageProvider.get_error_by_pattern(workflow_error.message)
    
    if not error_code:
        error_type = workflow_error.error_type.lower()
        if 'validation' in error_type:
            error_code = ErrorCode.INVALID_URL_FORMAT
        elif 'timeout' in error_type:
            error_code = ErrorCode.PROCESSING_TIMEOUT
        elif 'youtube' in error_type or 'transcript' in error_type:
            error_code = ErrorCode.VIDEO_NOT_FOUND
        elif 'llm' in error_type or 'ai' in error_type:
            error_code = ErrorCode.LLM_SERVICE_UNAVAILABLE
        elif 'network' in error_type:
            error_code = ErrorCode.CONNECTION_FAILED
        else:
            error_code = ErrorCode.WORKFLOW_EXECUTION_FAILED
    
    # Create detailed error information
    error_details = ErrorMessageProvider.get_error_details(
        error_code=error_code,
        additional_context=workflow_error.message,
        technical_details=f"Workflow error in {workflow_error.failed_node}: {workflow_error.message}"
    )
    
    # Determine HTTP status code
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    if error_details.category == ErrorCategory.VALIDATION:
        status_code = status.HTTP_400_BAD_REQUEST
    elif error_details.category == ErrorCategory.YOUTUBE_API:
        status_code = status.HTTP_400_BAD_REQUEST
    elif error_details.category == ErrorCategory.TIMEOUT:
        status_code = status.HTTP_408_REQUEST_TIMEOUT
    elif error_details.category == ErrorCategory.RATE_LIMIT:
        status_code = status.HTTP_429_TOO_MANY_REQUESTS
    elif error_details.category == ErrorCategory.NETWORK:
        status_code = status.HTTP_502_BAD_GATEWAY
    elif error_details.category == ErrorCategory.SERVER:
        status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    
    # Format response with enhanced error details
    response_content = ErrorMessageProvider.format_error_response(
        error_details=error_details,
        include_technical_details=False
    )
    
    # Add workflow-specific details
    response_content["error"]["workflow_details"] = {
        "failed_node": getattr(workflow_error, 'failed_node', 'unknown'),
        "is_recoverable": getattr(workflow_error, 'is_recoverable', True),
        "node_phase": getattr(workflow_error, 'node_phase', 'unknown'),
        "recovery_action": getattr(workflow_error, 'recovery_action', 'retry')
    }
    
    return JSONResponse(
        status_code=status_code,
        content=response_content
    )

def convert_timestamp_to_seconds(timestamp_str: str) -> int:
    """Convert timestamp string (MM:SS or HH:MM:SS) to seconds."""
    try:
        if not timestamp_str:
            return 0
        parts = timestamp_str.split(':')
        if len(parts) == 2:
            # MM:SS format
            minutes, seconds = map(int, parts)
            return minutes * 60 + seconds
        elif len(parts) == 3:
            # HH:MM:SS format
            hours, minutes, seconds = map(int, parts)
            return hours * 3600 + minutes * 60 + seconds
        else:
            return 0
    except (ValueError, TypeError):
        return 0

# Development server runner
if __name__ == "__main__":
    # Development configuration
    run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )