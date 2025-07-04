"""
Coverage-focused tests to achieve 90%+ coverage for Task 7.1.

This test file specifically targets uncovered code paths to maximize coverage.
"""

import pytest
import unittest
from unittest.mock import Mock, patch, MagicMock
import os
import sys
import logging
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Test imports for key modules that need coverage
from src.config import AppConfig, setup_logging
from src.utils.validators import YouTubeURLValidator
from src.utils.error_messages import ErrorMessageProvider


class TestCoverageBoost(unittest.TestCase):
    """Tests specifically designed to increase code coverage."""
    
    def test_config_comprehensive(self):
        """Test comprehensive config functionality."""
        # Test config with various settings
        config = AppConfig(
            debug=True,
            environment="production",
            log_level="DEBUG",
            openai_api_key="test_openai_key",
            anthropic_api_key="test_anthropic_key",
            max_video_duration=3600,
            request_timeout=300,
            max_retries=5,
            retry_delay=60,
            log_to_file=True,
            log_file_path="/tmp/test.log"
        )
        
        # Test property methods
        assert config.is_production is True
        assert config.is_development is False
        
        # Test redis URL generation
        config.redis_password = "secret"
        config.redis_db = 2
        redis_url = config.redis_url
        assert "secret" in redis_url
        assert "/2" in redis_url
        
        # Test logging config
        log_config = config.get_logging_config()
        assert isinstance(log_config, dict)
        assert 'formatters' in log_config
    
    def test_error_message_provider_coverage(self):
        """Test ErrorMessageProvider for coverage."""
        provider = ErrorMessageProvider()
        
        # Test various error message types
        validation_error = provider.create_validation_error(
            field="test_field",
            value="invalid_value",
            constraint="must be valid"
        )
        assert validation_error is not None
        
        # Test YouTube errors
        youtube_error = provider.create_youtube_error(
            error_type="private_video",
            video_id="test123"
        )
        assert youtube_error is not None
        
        # Test LLM errors
        llm_error = provider.create_llm_error(
            provider="openai",
            error_type="rate_limit"
        )
        assert llm_error is not None
        
        # Test workflow errors
        workflow_error = provider.create_workflow_error(
            workflow_id="test_workflow",
            stage="processing",
            error_type="timeout"
        )
        assert workflow_error is not None
    
    def test_validator_edge_cases(self):
        """Test validator edge cases for coverage."""
        validator = YouTubeURLValidator()
        
        # Test various URL formats
        test_urls = [
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://youtu.be/dQw4w9WgXcQ",
            "https://youtube.com/watch?v=dQw4w9WgXcQ",
            "https://m.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://www.youtube.com/embed/dQw4w9WgXcQ",
            "https://www.youtube.com/shorts/dQw4w9WgXcQ",
        ]
        
        for url in test_urls:
            assert validator.validate_youtube_url(url) is True
            video_id = validator.extract_video_id(url)
            assert video_id == "dQw4w9WgXcQ"
        
        # Test invalid URLs
        invalid_urls = [
            "",
            "not_a_url",
            "https://example.com",
            "https://youtube.com/watch",
            "https://youtube.com/watch?v=",
        ]
        
        for url in invalid_urls:
            assert validator.validate_youtube_url(url) is False
        
        # Test video ID validation
        assert validator.is_valid_video_id("dQw4w9WgXcQ") is True
        assert validator.is_valid_video_id("short") is True  # Short IDs are valid
        assert validator.is_valid_video_id("") is False
        assert validator.is_valid_video_id("invalid!@#") is False
    
    def test_setup_logging_coverage(self):
        """Test setup_logging function for coverage."""
        # Test with file logging disabled
        config = AppConfig(log_to_file=False)
        setup_logging(config)  # Should not raise exception
        
        # Test with file logging enabled
        config = AppConfig(
            log_to_file=True,
            log_file_path="/tmp/youtube_summarizer_test.log",
            log_level="INFO"
        )
        setup_logging(config)  # Should not raise exception
        
        # Verify logging is working
        logger = logging.getLogger("test_coverage")
        logger.info("Test message for coverage")
    
    def test_utility_functions_coverage(self):
        """Test utility functions for coverage."""
        from src.app import convert_timestamp_to_seconds
        
        # Test all branches of convert_timestamp_to_seconds
        assert convert_timestamp_to_seconds("1:30") == 90
        assert convert_timestamp_to_seconds("1:30:45") == 5445
        assert convert_timestamp_to_seconds("") == 0
        assert convert_timestamp_to_seconds("invalid") == 0
        assert convert_timestamp_to_seconds("1:2:3:4") == 0
        assert convert_timestamp_to_seconds(None) == 0
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'})
    def test_llm_client_coverage(self):
        """Test LLM client creation for coverage."""
        from src.utils.call_llm import create_llm_client, LLMProvider, LLMConfig
        
        # Test LLM provider enum
        assert LLMProvider.OPENAI.value == "openai"
        assert LLMProvider.ANTHROPIC.value == "anthropic"
        
        # Test LLM config creation
        config = LLMConfig(
            provider=LLMProvider.OPENAI,
            model="gpt-4",
            temperature=0.7,
            max_tokens=1000
        )
        assert config.provider == LLMProvider.OPENAI
        
        # Test client creation
        client = create_llm_client("openai")
        assert client is not None
    
    def test_error_enum_coverage(self):
        """Test error enums for coverage."""
        from src.utils.error_messages import ErrorCategory, ErrorSeverity, ErrorCode
        
        # Test all enum values are accessible
        categories = [item for item in ErrorCategory]
        assert len(categories) > 0
        
        severities = [item for item in ErrorSeverity]
        assert len(severities) == 4
        
        codes = [item for item in ErrorCode]
        assert len(codes) > 0
        
        # Test specific values
        assert ErrorCategory.VALIDATION.value == "validation"
        assert ErrorSeverity.HIGH.value == "high"
        assert ErrorCode.INVALID_URL_FORMAT.value == "E1001"


class TestFlowCoverage(unittest.TestCase):
    """Test flow module for coverage."""
    
    def test_workflow_config_structure(self):
        """Test workflow config creation."""
        from src.flow import WorkflowConfig
        
        config = WorkflowConfig(
            max_retries=3,
            timeout=300,
            circuit_breaker_enabled=True
        )
        assert config.max_retries == 3
        assert config.timeout == 300
        assert config.circuit_breaker_enabled is True
    
    def test_flow_store_operations(self):
        """Test flow store operations."""
        from src.flow import Store
        
        store = Store()
        
        # Test basic store operations
        store['test_key'] = 'test_value'
        assert store['test_key'] == 'test_value'
        assert 'test_key' in store
        
        # Test store methods
        store.update({'another_key': 'another_value'})
        assert store['another_key'] == 'another_value'
    
    def test_flow_creation(self):
        """Test flow creation."""
        from src.flow import create_youtube_summarizer_flow
        
        # Test flow creation
        flow = create_youtube_summarizer_flow()
        assert flow is not None


class TestNodesCoverage(unittest.TestCase):
    """Test nodes module for coverage."""
    
    def test_node_base_classes(self):
        """Test node base classes."""
        from src.nodes import Node, Store, NodeState
        
        # Test NodeState enum
        assert NodeState.PENDING == "pending"
        assert NodeState.SUCCESS == "success"
        assert NodeState.FAILED == "failed"
        
        # Test Store creation
        store = Store()
        store['test'] = 'value'
        assert store['test'] == 'value'
    
    def test_node_error_handling(self):
        """Test node error handling."""
        from src.nodes import NodeError
        
        # Test error creation
        exec_error = NodeError("Execution failed")
        assert str(exec_error) == "Execution failed"
    
    def test_processing_nodes(self):
        """Test processing node classes."""
        from src.nodes import (
            YouTubeTranscriptNode, SummarizationNode, 
            TimestampNode, KeywordExtractionNode, BaseProcessingNode
        )
        
        # Test node creation
        transcript_node = YouTubeTranscriptNode("transcript_node")
        assert transcript_node.name == "transcript_node"
        
        summary_node = SummarizationNode("summary_node")
        assert summary_node.name == "summary_node"
        
        timestamp_node = TimestampNode("timestamp_node")
        assert timestamp_node.name == "timestamp_node"
        
        keyword_node = KeywordExtractionNode("keyword_node")
        assert keyword_node.name == "keyword_node"


if __name__ == '__main__':
    unittest.main()