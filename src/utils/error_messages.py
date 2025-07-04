"""
Comprehensive error message definitions for the YouTube Summarizer application.

This module provides standardized error messages, error codes, and error categorization
for consistent error handling across all components of the application.
"""

from enum import Enum
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
import re


class ErrorCategory(Enum):
    """Categories of errors that can occur in the application."""
    VALIDATION = "validation"
    NETWORK = "network"
    YOUTUBE_API = "youtube_api"
    LLM_API = "llm_api"
    PROCESSING = "processing"
    TIMEOUT = "timeout"
    AUTHENTICATION = "authentication"
    RATE_LIMIT = "rate_limit"
    SERVER = "server"
    CONFIGURATION = "configuration"
    WORKFLOW = "workflow"


class ErrorSeverity(Enum):
    """Severity levels for errors."""
    LOW = "low"           # Minor issues, degraded functionality
    MEDIUM = "medium"     # Significant issues, some functionality lost
    HIGH = "high"         # Major issues, core functionality affected
    CRITICAL = "critical" # Critical issues, service unavailable


class ErrorCode(Enum):
    """Standardized error codes for the application."""
    
    # Validation Errors (1000-1999)
    INVALID_URL_FORMAT = "E1001"
    INVALID_VIDEO_ID = "E1002"
    URL_TOO_LONG = "E1003"
    EMPTY_URL = "E1004"
    SUSPICIOUS_CONTENT = "E1005"
    INVALID_DOMAIN = "E1006"
    MALFORMED_URL = "E1007"
    INVALID_INPUT_TYPE = "E1008"
    
    # YouTube API Errors (2000-2999)
    VIDEO_NOT_FOUND = "E2001"
    VIDEO_PRIVATE = "E2002"
    VIDEO_UNAVAILABLE = "E2003"
    VIDEO_LIVE_STREAM = "E2004"
    NO_TRANSCRIPT_AVAILABLE = "E2005"
    TRANSCRIPT_DISABLED = "E2006"
    VIDEO_TOO_LONG = "E2007"
    VIDEO_TOO_SHORT = "E2008"
    UNSUPPORTED_LANGUAGE = "E2009"
    REGION_BLOCKED = "E2010"
    AGE_RESTRICTED = "E2011"
    
    # Network Errors (3000-3999)
    NETWORK_TIMEOUT = "E3001"
    CONNECTION_FAILED = "E3002"
    DNS_RESOLUTION_FAILED = "E3003"
    SSL_ERROR = "E3004"
    PROXY_ERROR = "E3005"
    NETWORK_UNREACHABLE = "E3006"
    
    # LLM API Errors (4000-4999)
    LLM_API_KEY_INVALID = "E4001"
    LLM_API_QUOTA_EXCEEDED = "E4002"
    LLM_REQUEST_TOO_LARGE = "E4003"
    LLM_RESPONSE_INVALID = "E4004"
    LLM_SERVICE_UNAVAILABLE = "E4005"
    LLM_TIMEOUT = "E4006"
    LLM_RATE_LIMITED = "E4007"
    LLM_CONTENT_FILTERED = "E4008"
    
    # Processing Errors (5000-5999)
    TRANSCRIPT_PROCESSING_FAILED = "E5001"
    SUMMARIZATION_FAILED = "E5002"
    TIMESTAMP_EXTRACTION_FAILED = "E5003"
    KEYWORD_EXTRACTION_FAILED = "E5004"
    WORKFLOW_EXECUTION_FAILED = "E5005"
    NODE_EXECUTION_FAILED = "E5006"
    DATA_CORRUPTION = "E5007"
    INSUFFICIENT_CONTENT = "E5008"
    
    # Timeout Errors (6000-6999)
    REQUEST_TIMEOUT = "E6001"
    PROCESSING_TIMEOUT = "E6002"
    LLM_TIMEOUT_EXTENDED = "E6003"
    YOUTUBE_API_TIMEOUT = "E6004"
    WORKFLOW_TIMEOUT = "E6005"
    
    # Server Errors (7000-7999)
    INTERNAL_SERVER_ERROR = "E7001"
    SERVICE_UNAVAILABLE = "E7002"
    CONFIGURATION_ERROR = "E7003"
    DEPENDENCY_UNAVAILABLE = "E7004"
    MEMORY_ERROR = "E7005"
    DISK_SPACE_ERROR = "E7006"
    
    # Authentication & Authorization (8000-8999)
    INVALID_API_KEY = "E8001"
    INSUFFICIENT_PERMISSIONS = "E8002"
    TOKEN_EXPIRED = "E8003"
    AUTHENTICATION_FAILED = "E8004"
    
    # Rate Limiting (9000-9999)
    RATE_LIMIT_EXCEEDED = "E9001"
    DAILY_QUOTA_EXCEEDED = "E9002"
    CONCURRENT_REQUESTS_EXCEEDED = "E9003"


@dataclass
class ErrorDetails:
    """Detailed error information."""
    code: ErrorCode
    category: ErrorCategory
    severity: ErrorSeverity
    title: str
    message: str
    user_message: str
    suggested_actions: List[str] = field(default_factory=list)
    technical_details: Optional[str] = None
    retry_after: Optional[int] = None  # seconds
    is_recoverable: bool = True
    documentation_url: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class ErrorMessageProvider:
    """Provides standardized error messages for the application."""
    
    _ERROR_DEFINITIONS = {
        # Validation Errors
        ErrorCode.INVALID_URL_FORMAT: ErrorDetails(
            code=ErrorCode.INVALID_URL_FORMAT,
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.MEDIUM,
            title="Invalid YouTube URL Format",
            message="The provided URL is not a valid YouTube URL format",
            user_message="Please provide a valid YouTube video URL (e.g., https://www.youtube.com/watch?v=VIDEO_ID)",
            suggested_actions=[
                "Check that the URL starts with 'https://www.youtube.com/' or 'https://youtu.be/'",
                "Ensure the URL contains a valid video ID",
                "Remove any extra characters or spaces from the URL"
            ],
            documentation_url="https://docs.example.com/url-formats"
        ),
        
        ErrorCode.INVALID_VIDEO_ID: ErrorDetails(
            code=ErrorCode.INVALID_VIDEO_ID,
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.MEDIUM,
            title="Invalid Video ID",
            message="The video ID extracted from the URL is not valid",
            user_message="The video ID in your URL appears to be invalid. Please check the URL and try again.",
            suggested_actions=[
                "Verify the video ID is 11 characters long",
                "Ensure the video ID contains only letters, numbers, hyphens, and underscores",
                "Try copying the URL directly from YouTube"
            ]
        ),
        
        ErrorCode.URL_TOO_LONG: ErrorDetails(
            code=ErrorCode.URL_TOO_LONG,
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.LOW,
            title="URL Too Long",
            message="The provided URL exceeds the maximum allowed length",
            user_message="The URL you provided is too long. Please use a shorter, standard YouTube URL.",
            suggested_actions=[
                "Use the standard YouTube URL format",
                "Remove unnecessary query parameters",
                "Use the shortened youtu.be format if available"
            ]
        ),
        
        ErrorCode.EMPTY_URL: ErrorDetails(
            code=ErrorCode.EMPTY_URL,
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.MEDIUM,
            title="Empty URL",
            message="No URL was provided",
            user_message="Please provide a YouTube video URL to summarize.",
            suggested_actions=[
                "Enter a valid YouTube video URL",
                "Make sure the URL field is not empty"
            ]
        ),
        
        ErrorCode.SUSPICIOUS_CONTENT: ErrorDetails(
            code=ErrorCode.SUSPICIOUS_CONTENT,
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.HIGH,
            title="Suspicious Content Detected",
            message="The URL contains potentially harmful content",
            user_message="The URL you provided contains suspicious content and cannot be processed for security reasons.",
            suggested_actions=[
                "Use a clean, standard YouTube URL",
                "Avoid URLs with embedded scripts or special characters",
                "Copy the URL directly from YouTube's address bar"
            ],
            is_recoverable=False
        ),
        
        # YouTube API Errors
        ErrorCode.VIDEO_NOT_FOUND: ErrorDetails(
            code=ErrorCode.VIDEO_NOT_FOUND,
            category=ErrorCategory.YOUTUBE_API,
            severity=ErrorSeverity.MEDIUM,
            title="Video Not Found",
            message="The specified YouTube video could not be found",
            user_message="The video you're trying to summarize was not found. It may have been deleted or made private.",
            suggested_actions=[
                "Check that the video URL is correct",
                "Verify the video exists and is publicly accessible",
                "Try a different video"
            ]
        ),
        
        ErrorCode.VIDEO_PRIVATE: ErrorDetails(
            code=ErrorCode.VIDEO_PRIVATE,
            category=ErrorCategory.YOUTUBE_API,
            severity=ErrorSeverity.MEDIUM,
            title="Private Video",
            message="The video is private and cannot be accessed",
            user_message="This video is private and cannot be summarized. Please try a public video.",
            suggested_actions=[
                "Use a public YouTube video",
                "Ask the video owner to make it public",
                "Try a different video"
            ],
            is_recoverable=False
        ),
        
        ErrorCode.VIDEO_LIVE_STREAM: ErrorDetails(
            code=ErrorCode.VIDEO_LIVE_STREAM,
            category=ErrorCategory.YOUTUBE_API,
            severity=ErrorSeverity.MEDIUM,
            title="Live Stream Not Supported",
            message="Live streams cannot be processed",
            user_message="Live streams and premieres cannot be summarized. Please try a regular uploaded video.",
            suggested_actions=[
                "Wait for the live stream to end and be processed as a regular video",
                "Try a different, non-live video",
                "Check back later when the stream has ended"
            ],
            is_recoverable=False
        ),
        
        ErrorCode.NO_TRANSCRIPT_AVAILABLE: ErrorDetails(
            code=ErrorCode.NO_TRANSCRIPT_AVAILABLE,
            category=ErrorCategory.YOUTUBE_API,
            severity=ErrorSeverity.MEDIUM,
            title="No Transcript Available",
            message="No transcript is available for this video",
            user_message="This video doesn't have transcripts available, which are required for summarization.",
            suggested_actions=[
                "Try a video that has auto-generated or manual captions",
                "Look for videos from channels that typically enable captions",
                "Try a different video with available transcripts"
            ],
            is_recoverable=False
        ),
        
        ErrorCode.VIDEO_TOO_LONG: ErrorDetails(
            code=ErrorCode.VIDEO_TOO_LONG,
            category=ErrorCategory.YOUTUBE_API,
            severity=ErrorSeverity.MEDIUM,
            title="Video Too Long",
            message="The video exceeds the maximum duration limit",
            user_message="This video is too long to process. Please try a video shorter than 30 minutes.",
            suggested_actions=[
                "Choose a video shorter than 30 minutes",
                "Look for highlight reels or shorter versions",
                "Try processing a different video"
            ],
            is_recoverable=False
        ),
        
        # Network Errors
        ErrorCode.NETWORK_TIMEOUT: ErrorDetails(
            code=ErrorCode.NETWORK_TIMEOUT,
            category=ErrorCategory.NETWORK,
            severity=ErrorSeverity.MEDIUM,
            title="Network Timeout",
            message="The request timed out due to network issues",
            user_message="The request timed out. Please check your internet connection and try again.",
            suggested_actions=[
                "Check your internet connection",
                "Try again in a few moments",
                "Contact support if the problem persists"
            ],
            retry_after=30,
            is_recoverable=True
        ),
        
        ErrorCode.CONNECTION_FAILED: ErrorDetails(
            code=ErrorCode.CONNECTION_FAILED,
            category=ErrorCategory.NETWORK,
            severity=ErrorSeverity.HIGH,
            title="Connection Failed",
            message="Failed to establish connection to external services",
            user_message="Unable to connect to required services. Please try again later.",
            suggested_actions=[
                "Check your internet connection",
                "Try again in a few minutes",
                "Contact support if the issue persists"
            ],
            retry_after=60,
            is_recoverable=True
        ),
        
        # LLM API Errors
        ErrorCode.LLM_API_QUOTA_EXCEEDED: ErrorDetails(
            code=ErrorCode.LLM_API_QUOTA_EXCEEDED,
            category=ErrorCategory.LLM_API,
            severity=ErrorSeverity.HIGH,
            title="API Quota Exceeded",
            message="The AI service quota has been exceeded",
            user_message="The AI processing quota has been reached. Please try again later.",
            suggested_actions=[
                "Try again in a few hours",
                "Contact support for increased quota",
                "Use the service during off-peak hours"
            ],
            retry_after=3600,
            is_recoverable=True
        ),
        
        ErrorCode.LLM_RATE_LIMITED: ErrorDetails(
            code=ErrorCode.LLM_RATE_LIMITED,
            category=ErrorCategory.LLM_API,
            severity=ErrorSeverity.MEDIUM,
            title="Rate Limited",
            message="Too many requests to the AI service",
            user_message="You're making requests too quickly. Please wait a moment and try again.",
            suggested_actions=[
                "Wait a few seconds before trying again",
                "Reduce the frequency of your requests",
                "Try again later"
            ],
            retry_after=10,
            is_recoverable=True
        ),
        
        # Processing Errors
        ErrorCode.SUMMARIZATION_FAILED: ErrorDetails(
            code=ErrorCode.SUMMARIZATION_FAILED,
            category=ErrorCategory.PROCESSING,
            severity=ErrorSeverity.HIGH,
            title="Summarization Failed",
            message="Failed to generate summary for the video content",
            user_message="Unable to generate a summary for this video. Please try a different video or try again later.",
            suggested_actions=[
                "Try a different video",
                "Ensure the video has clear, understandable speech",
                "Try again in a few minutes",
                "Contact support if the problem continues"
            ],
            is_recoverable=True
        ),
        
        ErrorCode.WORKFLOW_EXECUTION_FAILED: ErrorDetails(
            code=ErrorCode.WORKFLOW_EXECUTION_FAILED,
            category=ErrorCategory.WORKFLOW,
            severity=ErrorSeverity.HIGH,
            title="Processing Workflow Failed",
            message="The video processing workflow encountered an error",
            user_message="An error occurred while processing your video. Please try again.",
            suggested_actions=[
                "Try the request again",
                "If the problem persists, try a different video",
                "Contact support with the error details"
            ],
            is_recoverable=True
        ),
        
        # Server Errors
        ErrorCode.INTERNAL_SERVER_ERROR: ErrorDetails(
            code=ErrorCode.INTERNAL_SERVER_ERROR,
            category=ErrorCategory.SERVER,
            severity=ErrorSeverity.CRITICAL,
            title="Internal Server Error",
            message="An unexpected server error occurred",
            user_message="Something went wrong on our end. Please try again in a few minutes.",
            suggested_actions=[
                "Try again in a few minutes",
                "Contact support if the problem persists",
                "Check our status page for known issues"
            ],
            retry_after=300,
            is_recoverable=True
        ),
        
        ErrorCode.SERVICE_UNAVAILABLE: ErrorDetails(
            code=ErrorCode.SERVICE_UNAVAILABLE,
            category=ErrorCategory.SERVER,
            severity=ErrorSeverity.CRITICAL,
            title="Service Unavailable",
            message="The service is temporarily unavailable",
            user_message="The service is temporarily unavailable. Please try again later.",
            suggested_actions=[
                "Try again in a few minutes",
                "Check our status page for maintenance updates",
                "Contact support for urgent issues"
            ],
            retry_after=600,
            is_recoverable=True
        )
    }
    
    @classmethod
    def get_error_details(cls, error_code: ErrorCode, 
                         additional_context: Optional[str] = None,
                         technical_details: Optional[str] = None) -> ErrorDetails:
        """
        Get detailed error information for a specific error code.
        
        Args:
            error_code: The error code to get details for
            additional_context: Additional context to include in the message
            technical_details: Technical details for debugging
            
        Returns:
            ErrorDetails object with comprehensive error information
        """
        if error_code not in cls._ERROR_DEFINITIONS:
            return cls._get_unknown_error(error_code, additional_context, technical_details)
        
        error_details = cls._ERROR_DEFINITIONS[error_code]
        
        # Clone the error details to avoid modifying the original
        customized_details = ErrorDetails(
            code=error_details.code,
            category=error_details.category,
            severity=error_details.severity,
            title=error_details.title,
            message=error_details.message,
            user_message=error_details.user_message,
            suggested_actions=error_details.suggested_actions.copy(),
            technical_details=technical_details or error_details.technical_details,
            retry_after=error_details.retry_after,
            is_recoverable=error_details.is_recoverable,
            documentation_url=error_details.documentation_url,
            timestamp=datetime.utcnow().isoformat()
        )
        
        # Add additional context if provided
        if additional_context:
            customized_details.message += f" - {additional_context}"
            customized_details.technical_details = technical_details or additional_context
        
        return customized_details
    
    @classmethod
    def _get_unknown_error(cls, error_code: ErrorCode, 
                          additional_context: Optional[str] = None,
                          technical_details: Optional[str] = None) -> ErrorDetails:
        """Create error details for unknown error codes."""
        return ErrorDetails(
            code=error_code,
            category=ErrorCategory.SERVER,
            severity=ErrorSeverity.HIGH,
            title="Unknown Error",
            message=f"An unknown error occurred: {error_code.value}",
            user_message="An unexpected error occurred. Please try again or contact support.",
            suggested_actions=[
                "Try the request again",
                "Contact support with the error code",
                "Check for any known issues"
            ],
            technical_details=technical_details or additional_context,
            is_recoverable=True
        )
    
    @classmethod
    def format_error_response(cls, error_details: ErrorDetails, 
                             include_technical_details: bool = False) -> Dict[str, Any]:
        """
        Format error details into a standardized API response.
        
        Args:
            error_details: The error details to format
            include_technical_details: Whether to include technical details
            
        Returns:
            Formatted error response dictionary
        """
        response = {
            "error": {
                "code": error_details.code.value,
                "category": error_details.category.value,
                "severity": error_details.severity.value,
                "title": error_details.title,
                "message": error_details.user_message,
                "suggested_actions": error_details.suggested_actions,
                "is_recoverable": error_details.is_recoverable,
                "timestamp": error_details.timestamp
            }
        }
        
        if error_details.retry_after:
            response["error"]["retry_after"] = error_details.retry_after
        
        if error_details.documentation_url:
            response["error"]["documentation_url"] = error_details.documentation_url
        
        if include_technical_details and error_details.technical_details:
            response["error"]["technical_details"] = error_details.technical_details
        
        return response
    
    @classmethod
    def get_error_by_pattern(cls, error_message: str) -> Optional[ErrorCode]:
        """
        Attempt to match an error message to a known error code using patterns.
        
        Args:
            error_message: The error message to analyze
            
        Returns:
            Matching error code if found, None otherwise
        """
        error_message_lower = error_message.lower()
        
        # Define patterns for common error messages
        patterns = {
            ErrorCode.VIDEO_NOT_FOUND: [
                r"video.*not.*found", r"404", r"does.*not.*exist"
            ],
            ErrorCode.VIDEO_PRIVATE: [
                r"private.*video", r"video.*private", r"access.*denied"
            ],
            ErrorCode.VIDEO_LIVE_STREAM: [
                r"live.*stream", r"premiere", r"streaming.*live"
            ],
            ErrorCode.NO_TRANSCRIPT_AVAILABLE: [
                r"no.*transcript", r"transcript.*not.*available", r"captions.*disabled"
            ],
            ErrorCode.VIDEO_TOO_LONG: [
                r"video.*too.*long", r"duration.*limit", r"exceeds.*maximum.*length"
            ],
            ErrorCode.NETWORK_TIMEOUT: [
                r"timeout", r"timed.*out", r"request.*timeout"
            ],
            ErrorCode.CONNECTION_FAILED: [
                r"connection.*failed", r"unable.*to.*connect", r"network.*error"
            ],
            ErrorCode.LLM_API_QUOTA_EXCEEDED: [
                r"quota.*exceeded", r"usage.*limit", r"api.*limit"
            ],
            ErrorCode.LLM_RATE_LIMITED: [
                r"rate.*limit", r"too.*many.*requests", r"throttled"
            ]
        }
        
        for error_code, pattern_list in patterns.items():
            for pattern in pattern_list:
                if re.search(pattern, error_message_lower):
                    return error_code
        
        return None
    
    @classmethod
    def create_validation_error(cls, validation_result, url: str = "") -> ErrorDetails:
        """
        Create error details from a validation result.
        
        Args:
            validation_result: ValidationResult from URL validation
            url: The original URL that failed validation
            
        Returns:
            ErrorDetails object
        """
        from .validators import URLValidationResult
        
        # Map validation results to error codes
        validation_to_error = {
            URLValidationResult.INVALID_FORMAT: ErrorCode.INVALID_URL_FORMAT,
            URLValidationResult.INVALID_DOMAIN: ErrorCode.INVALID_DOMAIN,
            URLValidationResult.INVALID_VIDEO_ID: ErrorCode.INVALID_VIDEO_ID,
            URLValidationResult.TOO_LONG: ErrorCode.URL_TOO_LONG,
            URLValidationResult.EMPTY_URL: ErrorCode.EMPTY_URL,
            URLValidationResult.INVALID_TYPE: ErrorCode.INVALID_INPUT_TYPE,
            URLValidationResult.MALFORMED_URL: ErrorCode.MALFORMED_URL,
            URLValidationResult.SUSPICIOUS_CONTENT: ErrorCode.SUSPICIOUS_CONTENT
        }
        
        error_code = validation_to_error.get(validation_result.result_type, ErrorCode.INVALID_URL_FORMAT)
        technical_details = f"URL: {url}, Validation error: {validation_result.error_message}"
        
        return cls.get_error_details(
            error_code=error_code,
            additional_context=validation_result.error_message,
            technical_details=technical_details
        )


# Convenience functions for common error scenarios
def get_youtube_error(error_message: str, video_id: str = "") -> ErrorDetails:
    """Get error details for YouTube-related errors."""
    provider = ErrorMessageProvider()
    error_code = provider.get_error_by_pattern(error_message)
    
    if not error_code:
        error_code = ErrorCode.VIDEO_NOT_FOUND  # Default for unknown YouTube errors
    
    technical_details = f"Video ID: {video_id}, YouTube error: {error_message}"
    return provider.get_error_details(error_code, technical_details=technical_details)


def get_llm_error(error_message: str, provider: str = "") -> ErrorDetails:
    """Get error details for LLM API errors."""
    error_provider = ErrorMessageProvider()
    error_code = error_provider.get_error_by_pattern(error_message)
    
    if not error_code:
        error_code = ErrorCode.LLM_SERVICE_UNAVAILABLE  # Default for unknown LLM errors
    
    technical_details = f"LLM Provider: {provider}, Error: {error_message}"
    return error_provider.get_error_details(error_code, technical_details=technical_details)


def get_network_error(error_message: str, url: str = "") -> ErrorDetails:
    """Get error details for network-related errors."""
    provider = ErrorMessageProvider()
    error_code = provider.get_error_by_pattern(error_message)
    
    if not error_code:
        error_code = ErrorCode.CONNECTION_FAILED  # Default for unknown network errors
    
    technical_details = f"URL: {url}, Network error: {error_message}"
    return provider.get_error_details(error_code, technical_details=technical_details)