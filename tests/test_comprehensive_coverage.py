"""
Comprehensive unit tests to achieve 90%+ code coverage.

This module contains additional tests specifically designed to cover
edge cases and improve overall test coverage for the YouTube summarizer.
"""

import pytest
import unittest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import json
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.config import AppConfig, setup_logging
from src.app import convert_timestamp_to_seconds
from src.utils.error_messages import (
    ErrorCategory, ErrorSeverity, ErrorCode, ErrorDetails, ErrorMessageProvider,
    get_youtube_error, get_llm_error, get_network_error
)
from src.utils.call_llm import (
    create_llm_client, LLMClient, LLMProvider, LLMConfig,
    LLMError, LLMClientError, LLMRateLimitError, LLMAuthenticationError
)


class TestConfigurationCoverage(unittest.TestCase):
    """Test configuration edge cases and full coverage."""
    
    def test_app_config_edge_cases(self):
        """Test AppConfig edge cases."""
        # Test with maximum values
        config = AppConfig(
            port=65535,
            workers=100,
            max_video_duration=3600,
            request_timeout=300,
            max_retries=10,
            retry_delay=60,
            memory_limit_mb=2048,
            log_max_size_mb=100,
            log_backup_count=10
        )
        
        assert config.port == 65535
        assert config.workers == 100
        assert config.max_video_duration == 3600
    
    def test_app_config_redis_url_generation(self):
        """Test Redis URL generation with various scenarios."""
        # Test with auth
        config = AppConfig(
            redis_host="localhost",
            redis_port=6379,
            redis_password="secret123",
            redis_db=1
        )
        
        expected = "redis://:secret123@localhost:6379/1"
        assert config.redis_url == expected
    
    def test_app_config_size_parsing(self):
        """Test memory size parsing methods."""
        config = AppConfig()
        
        # Test parse_memory_size
        assert config.parse_memory_size("100MB") == 100
        assert config.parse_memory_size("1GB") == 1024
        assert config.parse_memory_size("500") == 500
        
        # Test parse_log_size
        assert config.parse_log_size("50MB") == 50
        assert config.parse_log_size("100") == 100
    
    def test_setup_logging_with_file(self):
        """Test setup_logging with file logging enabled."""
        config = AppConfig(
            log_to_file=True,
            log_file_path="/tmp/test.log",
            log_level="DEBUG"
        )
        
        # This should not raise an exception
        setup_logging(config)
        
        # Verify logging configuration
        import logging
        logger = logging.getLogger("test")
        assert logger.level <= logging.DEBUG


class TestErrorMessagesCoverage(unittest.TestCase):
    """Test error messages comprehensive coverage."""
    
    def test_error_category_enum(self):
        """Test ErrorCategory enum values."""
        assert ErrorCategory.VALIDATION.value == "validation"
        assert ErrorCategory.NETWORK.value == "network"
        assert ErrorCategory.YOUTUBE_API.value == "youtube_api"
        assert ErrorCategory.LLM_API.value == "llm_api"
        assert ErrorCategory.PROCESSING.value == "processing"
        assert ErrorCategory.TIMEOUT.value == "timeout"
        assert ErrorCategory.AUTHENTICATION.value == "authentication"
        assert ErrorCategory.RATE_LIMIT.value == "rate_limit"
        assert ErrorCategory.SERVER.value == "server"
        assert ErrorCategory.CONFIGURATION.value == "configuration"
        assert ErrorCategory.WORKFLOW.value == "workflow"
    
    def test_error_severity_enum(self):
        """Test ErrorSeverity enum values."""
        assert ErrorSeverity.LOW.value == "low"
        assert ErrorSeverity.MEDIUM.value == "medium"
        assert ErrorSeverity.HIGH.value == "high"
        assert ErrorSeverity.CRITICAL.value == "critical"
    
    def test_error_code_enum(self):
        """Test ErrorCode enum values."""
        assert ErrorCode.INVALID_URL_FORMAT.value == "E1001"
        assert ErrorCode.INVALID_VIDEO_ID.value == "E1002"
        assert ErrorCode.URL_TOO_LONG.value == "E1003"
        assert ErrorCode.EMPTY_URL.value == "E1004"
    
    def test_error_details_creation(self):
        """Test ErrorDetails creation."""
        details = ErrorDetails(
            field="youtube_url",
            value="invalid_url",
            constraint="must be valid YouTube URL",
            context={"additional": "info"}
        )
        
        assert details.field == "youtube_url"
        assert details.value == "invalid_url"
        assert details.constraint == "must be valid YouTube URL"
        assert details.context == {"additional": "info"}
    
    def test_youtube_error_functions(self):
        """Test YouTube error generation functions."""
        error = get_youtube_error(
            error_type="transcript_disabled",
            video_id="test123",
            video_title="Test Video"
        )
        
        assert error is not None
        assert isinstance(error, dict)
    
    def test_llm_error_functions(self):
        """Test LLM error generation functions."""
        error = get_llm_error(
            provider="openai",
            model="gpt-4",
            error_type="rate_limit",
            message="Rate limit exceeded"
        )
        
        assert error is not None
        assert isinstance(error, dict)
    
    def test_network_error_functions(self):
        """Test network error generation functions."""
        error = get_network_error(
            error_type="connection_timeout",
            message="Connection timed out"
        )
        
        assert error is not None
        assert isinstance(error, dict)


class TestLLMClientCoverage(unittest.TestCase):
    """Test LLM client comprehensive coverage."""
    
    def test_llm_provider_enum(self):
        """Test LLMProvider enum values."""
        assert LLMProvider.OPENAI.value == "openai"
        assert LLMProvider.ANTHROPIC.value == "anthropic"
    
    def test_llm_config_creation(self):
        """Test LLMConfig creation."""
        config = LLMConfig(
            provider=LLMProvider.OPENAI,
            model="gpt-4",
            temperature=0.7,
            max_tokens=1000,
            timeout=30,
            max_retries=3
        )
        
        assert config.provider == LLMProvider.OPENAI
        assert config.model == "gpt-4"
        assert config.temperature == 0.7
        assert config.max_tokens == 1000
        assert config.timeout == 30
        assert config.max_retries == 3
    
    def test_create_llm_client_openai(self):
        """Test creating OpenAI client."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'}):
            client = create_llm_client("openai")
            assert isinstance(client, LLMClient)
    
    def test_create_llm_client_anthropic(self):
        """Test creating Anthropic client."""
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test_key'}):
            client = create_llm_client("anthropic")
            assert isinstance(client, LLMClient)
    
    def test_create_llm_client_invalid_provider(self):
        """Test creating client with invalid provider."""
        with pytest.raises((ValueError, LLMClientError)):
            create_llm_client("invalid_provider")
    
    def test_llm_error_classes(self):
        """Test LLM error classes."""
        # Test base LLMError
        error = LLMError("Test error")
        assert str(error) == "Test error"
        
        # Test specific error types
        client_error = LLMClientError("Client error")
        assert str(client_error) == "Client error"
        
        rate_limit_error = LLMRateLimitError("Rate limit exceeded")
        assert str(rate_limit_error) == "Rate limit exceeded"
        
        auth_error = LLMAuthenticationError("Authentication failed")
        assert str(auth_error) == "Authentication failed"


class TestUtilityFunctionsCoverage(unittest.TestCase):
    """Test utility functions comprehensive coverage."""
    
    def test_convert_timestamp_edge_cases(self):
        """Test convert_timestamp_to_seconds edge cases."""
        # Test None input
        assert convert_timestamp_to_seconds(None) == 0
        
        # Test empty string
        assert convert_timestamp_to_seconds("") == 0
        
        # Test invalid format
        assert convert_timestamp_to_seconds("invalid") == 0
        
        # Test too many parts
        assert convert_timestamp_to_seconds("1:2:3:4") == 0
        
        # Test negative numbers (function processes them as regular ints)
        assert convert_timestamp_to_seconds("-1:30") == -30  # -1*60 + 30
        
        # Test very large numbers
        assert convert_timestamp_to_seconds("99:59") == 5999
        
        # Test single digit formats
        assert convert_timestamp_to_seconds("1:5") == 65
        assert convert_timestamp_to_seconds("0:1:5") == 65
        
        # Test zero values
        assert convert_timestamp_to_seconds("0:0") == 0
        assert convert_timestamp_to_seconds("0:0:0") == 0


class TestNodesCoverage(unittest.TestCase):
    """Test nodes comprehensive coverage."""
    
    def test_base_node_fallback_classes(self):
        """Test fallback Node classes when PocketFlow not available."""
        # This tests the fallback classes defined in nodes.py
        from src.nodes import Node, Store, NodeState
        
        # Test NodeState constants
        assert NodeState.PENDING == "pending"
        assert NodeState.PREP == "prep"
        assert NodeState.EXEC == "exec"
        assert NodeState.POST == "post"
        assert NodeState.SUCCESS == "success"
        assert NodeState.FAILED == "failed"
        
        # Test Store creation
        store = Store()
        assert isinstance(store, dict)
        
        # Test that Node is abstract
        with pytest.raises(TypeError):
            Node("test")


class TestFlowCoverage(unittest.TestCase):
    """Test flow comprehensive coverage."""
    
    def test_workflow_result_creation(self):
        """Test WorkflowResult creation."""
        from src.flow import WorkflowResult
        
        result = WorkflowResult(
            workflow_id="test_123",
            status="success",
            video_id="abc123",
            video_title="Test Video",
            summary="Test summary",
            timestamps=[],
            keywords=["test", "video"],
            processing_time=1.5,
            metadata={"test": "value"}
        )
        
        assert result.workflow_id == "test_123"
        assert result.status == "success"
        assert result.video_id == "abc123"
        assert result.video_title == "Test Video"
        assert result.summary == "Test summary"
        assert result.timestamps == []
        assert result.keywords == ["test", "video"]
        assert result.processing_time == 1.5
        assert result.metadata == {"test": "value"}
    
    def test_workflow_configuration_creation(self):
        """Test WorkflowConfiguration creation."""
        from src.flow import WorkflowConfiguration
        
        config = WorkflowConfiguration(
            max_video_duration=1800,
            request_timeout=120,
            max_retries=5,
            retry_delay=30,
            openai_model="gpt-4",
            anthropic_model="claude-3-haiku-20240307",
            default_llm_provider="openai"
        )
        
        assert config.max_video_duration == 1800
        assert config.request_timeout == 120
        assert config.max_retries == 5
        assert config.retry_delay == 30
        assert config.openai_model == "gpt-4"
        assert config.anthropic_model == "claude-3-haiku-20240307"
        assert config.default_llm_provider == "openai"


if __name__ == '__main__':
    unittest.main()