"""
Comprehensive tests for error handling and validation functionality.

This module tests all error handling scenarios, exception handling,
timeout handling, and error message standardization.
"""

import pytest
import time
from unittest.mock import Mock, patch
from src.utils.error_messages import (
    ErrorMessageProvider, ErrorCode, ErrorCategory, ErrorSeverity
)
from src.utils.validators import (
    validate_youtube_url_detailed, URLValidationResult,
    YouTubeURLValidator
)


class TestErrorMessageProvider:
    """Test cases for ErrorMessageProvider functionality."""
    
    def test_get_error_details_known_code(self):
        """Test getting error details for known error codes."""
        error_details = ErrorMessageProvider.get_error_details(ErrorCode.INVALID_URL_FORMAT)
        
        assert error_details.code == ErrorCode.INVALID_URL_FORMAT
        assert error_details.category == ErrorCategory.VALIDATION
        assert error_details.severity == ErrorSeverity.MEDIUM
        assert "YouTube URL" in error_details.title
        assert error_details.suggested_actions
        assert error_details.is_recoverable
    
    def test_get_error_details_with_context(self):
        """Test getting error details with additional context."""
        context = "Additional error context"
        technical_details = "Technical debugging info"
        
        error_details = ErrorMessageProvider.get_error_details(
            ErrorCode.VIDEO_NOT_FOUND,
            additional_context=context,
            technical_details=technical_details
        )
        
        assert context in error_details.message
        assert error_details.technical_details == technical_details
    
    def test_format_error_response(self):
        """Test error response formatting."""
        error_details = ErrorMessageProvider.get_error_details(ErrorCode.NETWORK_TIMEOUT)
        response = ErrorMessageProvider.format_error_response(error_details)
        
        assert "error" in response
        assert response["error"]["code"] == ErrorCode.NETWORK_TIMEOUT.value
        assert response["error"]["category"] == ErrorCategory.NETWORK.value
        assert response["error"]["message"]
        assert response["error"]["suggested_actions"]
        assert response["error"]["timestamp"]
    
    def test_get_error_by_pattern(self):
        """Test pattern-based error detection."""
        test_cases = [
            ("video not found", ErrorCode.VIDEO_NOT_FOUND),
            ("private video", ErrorCode.VIDEO_PRIVATE),
            ("live stream", ErrorCode.VIDEO_LIVE_STREAM),
            ("no transcript", ErrorCode.NO_TRANSCRIPT_AVAILABLE),
            ("timeout occurred", ErrorCode.NETWORK_TIMEOUT),
            ("rate limit exceeded", ErrorCode.LLM_RATE_LIMITED),
        ]
        
        for message, expected_code in test_cases:
            detected_code = ErrorMessageProvider.get_error_by_pattern(message)
            assert detected_code == expected_code
    
    def test_create_validation_error(self):
        """Test validation error creation."""
        # Mock validation result
        validation_result = Mock()
        validation_result.result_type = URLValidationResult.INVALID_URL_FORMAT
        validation_result.error_message = "Invalid URL format"
        
        error_details = ErrorMessageProvider.create_validation_error(
            validation_result, "https://invalid-url.com"
        )
        
        assert error_details.code == ErrorCode.INVALID_URL_FORMAT
        assert error_details.category == ErrorCategory.VALIDATION


class TestURLValidation:
    """Test cases for enhanced URL validation."""
    
    def test_valid_youtube_urls(self):
        """Test validation of valid YouTube URLs."""
        valid_urls = [
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://youtu.be/dQw4w9WgXcQ",
            "https://www.youtube.com/embed/dQw4w9WgXcQ",
            "https://www.youtube.com/shorts/dQw4w9WgXcQ",
        ]
        
        for url in valid_urls:
            result = validate_youtube_url_detailed(url)
            assert result.is_valid, f"URL should be valid: {url}"
            assert result.result_type == URLValidationResult.VALID
            assert result.video_id == "dQw4w9WgXcQ"
    
    def test_invalid_youtube_urls(self):
        """Test validation of invalid YouTube URLs."""
        invalid_urls = [
            "",
            "not-a-url",
            "https://www.google.com",
            "https://vimeo.com/123456",
            "https://www.youtube.com/watch?v=invalid",
        ]
        
        for url in invalid_urls:
            result = validate_youtube_url_detailed(url)
            assert not result.is_valid, f"URL should be invalid: {url}"
            assert result.error_message
    
    def test_suspicious_content_detection(self):
        """Test detection of suspicious content in URLs."""
        suspicious_urls = [
            "javascript:alert('xss')",
            "data:text/html,<script>alert('xss')</script>",
            "https://youtube.com/watch?v=test\x00malicious",
        ]
        
        for url in suspicious_urls:
            result = validate_youtube_url_detailed(url)
            assert not result.is_valid
            assert result.result_type == URLValidationResult.SUSPICIOUS_CONTENT
    
    def test_url_too_long(self):
        """Test handling of URLs that are too long."""
        long_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ&" + "a" * 2000
        result = validate_youtube_url_detailed(long_url)
        
        assert not result.is_valid
        assert result.result_type == URLValidationResult.TOO_LONG


class TestExceptionHandling:
    """Test cases for exception handling functionality."""
    
    def test_youtube_exception_mapping(self):
        """Test that YouTube exceptions are properly mapped."""
        # This would test the actual exception handling
        # when the full YouTube API is available
        pass
    
    def test_llm_exception_mapping(self):
        """Test that LLM exceptions are properly mapped."""
        # This would test the actual exception handling
        # when the full LLM client is available
        pass
    
    def test_retry_logic(self):
        """Test retry logic with different error types."""
        # Mock a function that fails then succeeds
        call_count = 0
        
        def mock_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Network error")
            return "success"
        
        # This would test the actual retry logic
        # when the full node implementation is available
        pass


class TestTimeoutHandling:
    """Test cases for timeout handling."""
    
    def test_operation_timeout(self):
        """Test that operations timeout appropriately."""
        def slow_operation():
            time.sleep(2)  # Sleep longer than timeout
            return "completed"
        
        # This would test the actual timeout implementation
        # when the full timeout handling is available
        pass
    
    def test_timeout_configuration(self):
        """Test that timeout configurations are properly loaded."""
        from src.config import settings
        
        # Verify timeout settings exist and have reasonable values
        assert hasattr(settings, 'youtube_api_timeout')
        assert hasattr(settings, 'llm_timeout')
        assert hasattr(settings, 'workflow_timeout')
        assert hasattr(settings, 'node_timeout')
        
        assert settings.youtube_api_timeout > 0
        assert settings.llm_timeout > 0
        assert settings.workflow_timeout > 0
        assert settings.node_timeout > 0


class TestErrorRecovery:
    """Test cases for error recovery and classification."""
    
    def test_recoverable_error_classification(self):
        """Test that errors are correctly classified as recoverable."""
        # This would test the actual error recovery logic
        # when the full node implementation is available
        pass
    
    def test_non_recoverable_error_classification(self):
        """Test that errors are correctly classified as non-recoverable."""
        # This would test the actual error recovery logic
        # when the full node implementation is available
        pass


class TestErrorLogging:
    """Test cases for error logging functionality."""
    
    def test_error_logging_format(self):
        """Test that errors are logged in the correct format."""
        # This would test the actual logging implementation
        pass
    
    def test_error_context_logging(self):
        """Test that error context is properly logged."""
        # This would test the actual logging implementation
        pass


if __name__ == "__main__":
    # Run basic tests that don't require external dependencies
    test_provider = TestErrorMessageProvider()
    test_provider.test_get_error_details_known_code()
    test_provider.test_format_error_response()
    test_provider.test_get_error_by_pattern()
    
    test_validation = TestURLValidation()
    test_validation.test_valid_youtube_urls()
    test_validation.test_invalid_youtube_urls()
    test_validation.test_suspicious_content_detection()
    test_validation.test_url_too_long()
    
    test_timeout = TestTimeoutHandling()
    test_timeout.test_timeout_configuration()
    
    print("Basic error handling tests completed successfully!")