"""
Unit tests for Ollama integration in call_llm.py.

This module tests the Ollama client functionality including:
- Ollama client initialization
- Model availability checking
- Text generation
- Error handling
- Response format consistency
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from src.utils.call_llm import (
    LLMClient, LLMConfig, LLMProvider, 
    OllamaConnectionError, OllamaModelNotFoundError, OllamaInsufficientResourcesError,
    check_ollama_availability, get_recommended_ollama_model,
    get_default_models, detect_provider_from_model
)


class TestOllamaClientInitialization:
    """Test Ollama client initialization and configuration."""
    
    def test_ollama_config_creation(self):
        """Test creating LLMConfig for Ollama."""
        config = LLMConfig(
            provider=LLMProvider.OLLAMA,
            model="llama3.1:8b",
            ollama_host="http://localhost:11434",
            ollama_keep_alive="5m"
        )
        
        assert config.provider == LLMProvider.OLLAMA
        assert config.model == "llama3.1:8b"
        assert config.ollama_host == "http://localhost:11434"
        assert config.ollama_keep_alive == "5m"
    
    @patch('src.utils.call_llm.OLLAMA_AVAILABLE', True)
    @patch('src.utils.call_llm.ollama')
    def test_ollama_client_init_success(self, mock_ollama):
        """Test successful Ollama client initialization."""
        # Mock the ollama client and its responses
        mock_client = Mock()
        mock_client.list.return_value = {
            'models': [
                {'name': 'llama3.1:8b'},
                {'name': 'mistral:7b'}
            ]
        }
        mock_ollama.Client.return_value = mock_client
        
        config = LLMConfig(
            provider=LLMProvider.OLLAMA,
            model="llama3.1:8b"
        )
        
        client = LLMClient(config)
        
        assert client.client == mock_client
        mock_ollama.Client.assert_called_once_with(host=config.ollama_host)
        mock_client.list.assert_called_once()
    
    @patch('src.utils.call_llm.OLLAMA_AVAILABLE', False)
    def test_ollama_client_init_unavailable(self):
        """Test Ollama client initialization when library is unavailable."""
        config = LLMConfig(
            provider=LLMProvider.OLLAMA,
            model="llama3.1:8b"
        )
        
        with pytest.raises(Exception) as exc_info:
            LLMClient(config)
        
        assert "Ollama library not installed" in str(exc_info.value)
    
    @patch('src.utils.call_llm.OLLAMA_AVAILABLE', True)
    @patch('src.utils.call_llm.ollama')
    def test_ollama_client_init_connection_error(self, mock_ollama):
        """Test Ollama client initialization with connection error."""
        mock_client = Mock()
        mock_client.list.side_effect = ConnectionError("Connection refused")
        mock_ollama.Client.return_value = mock_client
        
        config = LLMConfig(
            provider=LLMProvider.OLLAMA,
            model="llama3.1:8b"
        )
        
        with pytest.raises(OllamaConnectionError):
            LLMClient(config)
    
    @patch('src.utils.call_llm.OLLAMA_AVAILABLE', True)
    @patch('src.utils.call_llm.ollama')
    def test_ollama_client_init_model_not_found(self, mock_ollama):
        """Test Ollama client initialization when model is not found."""
        mock_client = Mock()
        mock_client.list.return_value = {
            'models': [
                {'name': 'mistral:7b'}  # Different model
            ]
        }
        mock_ollama.Client.return_value = mock_client
        
        config = LLMConfig(
            provider=LLMProvider.OLLAMA,
            model="llama3.1:8b"
        )
        
        with pytest.raises(OllamaModelNotFoundError):
            LLMClient(config)


class TestOllamaTextGeneration:
    """Test Ollama text generation functionality."""
    
    @pytest.fixture
    def ollama_client(self):
        """Create a mock Ollama client for testing."""
        with patch('src.utils.call_llm.OLLAMA_AVAILABLE', True):
            with patch('src.utils.call_llm.ollama') as mock_ollama:
                mock_client = Mock()
                mock_client.list.return_value = {
                    'models': [{'name': 'llama3.1:8b'}]
                }
                mock_ollama.Client.return_value = mock_client
                
                config = LLMConfig(
                    provider=LLMProvider.OLLAMA,
                    model="llama3.1:8b"
                )
                
                return LLMClient(config)
    
    def test_generate_text_success(self, ollama_client):
        """Test successful text generation with Ollama."""
        # Mock the chat response
        mock_response = {
            'message': {
                'content': 'This is a generated response from Ollama'
            },
            'model': 'llama3.1:8b',
            'created_at': '2024-01-01T00:00:00Z',
            'done': True,
            'total_duration': 1000000000,
            'load_duration': 100000000,
            'prompt_eval_count': 50,
            'prompt_eval_duration': 200000000,
            'eval_count': 25,
            'eval_duration': 300000000
        }
        
        ollama_client.client.chat.return_value = mock_response
        
        result = ollama_client.generate_text(
            prompt="Test prompt",
            system_prompt="Test system prompt",
            max_tokens=100,
            temperature=0.7
        )
        
        assert result['text'] == 'This is a generated response from Ollama'
        assert 'usage' in result
        assert result['usage']['prompt_tokens'] > 0
        assert result['usage']['completion_tokens'] > 0
        assert result['usage']['total_tokens'] > 0
        assert '_ollama_metadata' in result
        assert result['_ollama_metadata']['model'] == 'llama3.1:8b'
    
    def test_generate_text_with_system_prompt(self, ollama_client):
        """Test text generation with system prompt."""
        mock_response = {
            'message': {'content': 'Response with system prompt'},
            'model': 'llama3.1:8b',
            'done': True
        }
        ollama_client.client.chat.return_value = mock_response
        
        result = ollama_client.generate_text(
            prompt="User prompt",
            system_prompt="You are a helpful assistant"
        )
        
        # Verify the chat was called with correct message structure
        call_args = ollama_client.client.chat.call_args
        messages = call_args[1]['messages']
        
        assert len(messages) == 2
        assert messages[0]['role'] == 'system'
        assert messages[0]['content'] == 'You are a helpful assistant'
        assert messages[1]['role'] == 'user'
        assert messages[1]['content'] == 'User prompt'
    
    def test_generate_text_without_system_prompt(self, ollama_client):
        """Test text generation without system prompt."""
        mock_response = {
            'message': {'content': 'Response without system prompt'},
            'model': 'llama3.1:8b',
            'done': True
        }
        ollama_client.client.chat.return_value = mock_response
        
        result = ollama_client.generate_text(prompt="User prompt only")
        
        # Verify the chat was called with correct message structure
        call_args = ollama_client.client.chat.call_args
        messages = call_args[1]['messages']
        
        assert len(messages) == 1
        assert messages[0]['role'] == 'user'
        assert messages[0]['content'] == 'User prompt only'
    
    def test_generate_text_connection_error(self, ollama_client):
        """Test text generation with connection error."""
        ollama_client.client.chat.side_effect = ConnectionError("Connection refused")
        
        with pytest.raises(OllamaConnectionError):
            ollama_client.generate_text("Test prompt")
    
    def test_generate_text_model_not_found_error(self, ollama_client):
        """Test text generation with model not found error."""
        ollama_client.client.chat.side_effect = Exception("model not found")
        
        with pytest.raises(OllamaModelNotFoundError):
            ollama_client.generate_text("Test prompt")
    
    def test_generate_text_insufficient_resources_error(self, ollama_client):
        """Test text generation with insufficient resources error."""
        ollama_client.client.chat.side_effect = Exception("out of memory")
        
        with pytest.raises(OllamaInsufficientResourcesError):
            ollama_client.generate_text("Test prompt")
    
    def test_generate_text_empty_response(self, ollama_client):
        """Test handling of empty response from Ollama."""
        mock_response = {
            'message': {'content': ''},  # Empty content
            'model': 'llama3.1:8b',
            'done': True
        }
        ollama_client.client.chat.return_value = mock_response
        
        with pytest.raises(ValueError, match="Empty response from Ollama"):
            ollama_client.generate_text("Test prompt")


class TestOllamaErrorHandling:
    """Test Ollama-specific error handling."""
    
    def test_ollama_connection_error(self):
        """Test OllamaConnectionError creation and properties."""
        error = OllamaConnectionError("localhost:11434")
        
        assert "localhost:11434" in error.message
        assert error.provider == "ollama"
        assert error.error_code == "CONNECTION_ERROR"
    
    def test_ollama_model_not_found_error(self):
        """Test OllamaModelNotFoundError creation."""
        error = OllamaModelNotFoundError("llama3.1:8b")
        
        assert "llama3.1:8b" in error.message
        assert "ollama pull" in error.message
        assert error.provider == "ollama"
        assert error.error_code == "MODEL_NOT_FOUND"
    
    def test_ollama_insufficient_resources_error(self):
        """Test OllamaInsufficientResourcesError creation."""
        error = OllamaInsufficientResourcesError("llama3.1:8b")
        
        assert "llama3.1:8b" in error.message
        assert "smaller model" in error.message
        assert error.provider == "ollama"
        assert error.error_code == "INSUFFICIENT_RESOURCES"


class TestOllamaUtilityFunctions:
    """Test Ollama utility functions."""
    
    @patch('src.utils.call_llm.OLLAMA_AVAILABLE', True)
    @patch('src.utils.call_llm.ollama')
    def test_check_ollama_availability_success(self, mock_ollama):
        """Test successful Ollama availability check."""
        mock_client = Mock()
        mock_client.list.return_value = {
            'models': [
                {'name': 'llama3.1:8b'},
                {'name': 'mistral:7b'}
            ]
        }
        mock_ollama.Client.return_value = mock_client
        
        result = check_ollama_availability()
        
        assert result['available'] is True
        assert len(result['models']) == 2
        assert 'llama3.1:8b' in result['models']
        assert 'mistral:7b' in result['models']
        assert result['model_count'] == 2
    
    @patch('src.utils.call_llm.OLLAMA_AVAILABLE', False)
    def test_check_ollama_availability_library_unavailable(self):
        """Test Ollama availability check when library is unavailable."""
        result = check_ollama_availability()
        
        assert result['available'] is False
        assert 'library not installed' in result['reason']
        assert result['models'] == []
    
    @patch('src.utils.call_llm.OLLAMA_AVAILABLE', True)
    @patch('src.utils.call_llm.ollama')
    def test_check_ollama_availability_connection_error(self, mock_ollama):
        """Test Ollama availability check with connection error."""
        mock_client = Mock()
        mock_client.list.side_effect = ConnectionError("Connection refused")
        mock_ollama.Client.return_value = mock_client
        
        result = check_ollama_availability()
        
        assert result['available'] is False
        assert 'Cannot connect' in result['reason']
        assert 'error' in result
    
    def test_get_recommended_ollama_model_summarization(self):
        """Test model recommendation for summarization task."""
        model = get_recommended_ollama_model("summarization", "medium")
        assert model == "llama3.1:8b"
        
        model = get_recommended_ollama_model("summarization", "low")
        assert model == "llama3.2:3b"
        
        model = get_recommended_ollama_model("summarization", "high")
        assert model == "mistral-small:24b"
    
    def test_get_recommended_ollama_model_multilingual(self):
        """Test model recommendation for multilingual tasks."""
        model = get_recommended_ollama_model("multilingual", "medium")
        assert model == "qwen2.5:7b"
        
        model = get_recommended_ollama_model("multilingual", "high")
        assert model == "llama3.3:70b"
    
    def test_get_recommended_ollama_model_general(self):
        """Test model recommendation for general tasks."""
        model = get_recommended_ollama_model("general", "medium")
        assert model == "mistral:7b"
        
        model = get_recommended_ollama_model("unknown_task", "medium")
        assert model == "mistral:7b"  # Should default to general
    
    def test_get_default_models_includes_ollama(self):
        """Test that get_default_models includes Ollama models."""
        models = get_default_models()
        
        assert 'ollama' in models
        assert 'llama3.1:8b' in models['ollama']
        assert 'mistral:7b' in models['ollama']
        assert 'qwen2.5:7b' in models['ollama']
    
    def test_detect_provider_from_model_ollama(self):
        """Test provider detection for Ollama models."""
        assert detect_provider_from_model('llama3.1:8b') == 'ollama'
        assert detect_provider_from_model('mistral:7b') == 'ollama'
        assert detect_provider_from_model('qwen2.5:7b') == 'ollama'
        assert detect_provider_from_model('phi:latest') == 'ollama'
        assert detect_provider_from_model('gemma:2b') == 'ollama'
    
    def test_detect_provider_from_model_non_ollama(self):
        """Test provider detection for non-Ollama models."""
        assert detect_provider_from_model('gpt-4') == 'openai'
        assert detect_provider_from_model('claude-3-sonnet') == 'anthropic'
        assert detect_provider_from_model('unknown-model') == 'openai'  # Default


class TestResponseFormatConsistency:
    """Test that Ollama responses match OpenAI/Anthropic format."""
    
    @patch('src.utils.call_llm.OLLAMA_AVAILABLE', True)
    @patch('src.utils.call_llm.ollama')
    def test_response_format_consistency(self, mock_ollama):
        """Test that Ollama response format matches other providers."""
        # Set up Ollama client
        mock_client = Mock()
        mock_client.list.return_value = {'models': [{'name': 'llama3.1:8b'}]}
        mock_ollama.Client.return_value = mock_client
        
        # Mock chat response
        mock_response = {
            'message': {'content': 'Test response'},
            'model': 'llama3.1:8b',
            'done': True,
            'total_duration': 1000000000,
            'eval_count': 10,
            'prompt_eval_count': 20
        }
        mock_client.chat.return_value = mock_response
        
        config = LLMConfig(provider=LLMProvider.OLLAMA, model="llama3.1:8b")
        client = LLMClient(config)
        
        result = client.generate_text("Test prompt")
        
        # Check that response has the same structure as OpenAI/Anthropic
        assert 'text' in result
        assert 'usage' in result
        assert 'prompt_tokens' in result['usage']
        assert 'completion_tokens' in result['usage']
        assert 'total_tokens' in result['usage']
        
        # Check that Ollama-specific metadata is separate
        assert '_ollama_metadata' in result
        assert 'model' in result['_ollama_metadata']
        assert 'total_duration' in result['_ollama_metadata']


if __name__ == '__main__':
    pytest.main([__file__])