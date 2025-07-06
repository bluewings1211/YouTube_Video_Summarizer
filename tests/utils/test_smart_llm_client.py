"""
Unit tests for SmartLLMClient and Ollama integration.

This module tests the smart LLM client functionality including:
- Model selection logic
- Ollama integration
- Chinese language processing
- Fallback mechanisms
- Error handling
"""

import pytest
import os
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from src.utils.smart_llm_client import (
    SmartLLMClient, TaskRequirements, ModelSelectionStrategy,
    create_smart_client, detect_task_requirements
)
from src.utils.call_llm import (
    LLMConfig, LLMProvider, OllamaConnectionError, 
    OllamaModelNotFoundError, LLMError
)


class TestTaskRequirements:
    """Test TaskRequirements dataclass functionality."""
    
    def test_task_requirements_creation(self):
        """Test creating TaskRequirements with different parameters."""
        requirements = TaskRequirements(
            task_type="summarization",
            language="zh-CN",
            text_length=1000,
            quality_level="high",
            speed_priority=True
        )
        
        assert requirements.task_type == "summarization"
        assert requirements.language == "zh-CN"
        assert requirements.text_length == 1000
        assert requirements.quality_level == "high"
        assert requirements.speed_priority is True
    
    def test_task_requirements_defaults(self):
        """Test TaskRequirements with default values."""
        requirements = TaskRequirements(task_type="keywords")
        
        assert requirements.task_type == "keywords"
        assert requirements.language is None
        assert requirements.text_length is None
        assert requirements.quality_level == "medium"
        assert requirements.speed_priority is False


class TestSmartLLMClient:
    """Test SmartLLMClient functionality."""
    
    @pytest.fixture
    def mock_config(self):
        """Mock configuration for testing."""
        return {
            'default_provider': 'ollama',
            'ollama_model': 'llama3.1:8b',
            'openai_model': 'gpt-3.5-turbo',
            'anthropic_model': 'claude-3-sonnet-20240229',
            'model_selection_strategy': 'auto',
            'chinese_language_model': 'qwen2.5:7b',
            'performance_model': 'mistral:7b',
            'lightweight_model': 'llama3.2:3b',
            'ollama': {
                'host': 'http://localhost:11434',
                'keep_alive': '5m',
                'fallback_enabled': True,
                'fallback_provider': 'openai'
            },
            'max_tokens': 1000,
            'temperature': 0.7
        }
    
    @pytest.fixture
    def smart_client(self, mock_config):
        """Create SmartLLMClient instance for testing."""
        return SmartLLMClient(mock_config)
    
    def test_client_initialization(self, mock_config):
        """Test SmartLLMClient initialization."""
        client = SmartLLMClient(mock_config)
        assert client.config == mock_config
        assert client.strategy == ModelSelectionStrategy.AUTO
    
    def test_client_initialization_without_config(self):
        """Test SmartLLMClient initialization with default config."""
        with patch.dict(os.environ, {
            'DEFAULT_LLM_PROVIDER': 'ollama',
            'OLLAMA_MODEL': 'llama3.1:8b'
        }):
            client = SmartLLMClient()
            assert client.config['default_provider'] == 'ollama'
    
    @patch('src.utils.smart_llm_client.check_ollama_availability')
    def test_get_ollama_status(self, mock_check_ollama, smart_client):
        """Test Ollama status checking."""
        mock_status = {
            'available': True,
            'models': ['llama3.1:8b', 'mistral:7b'],
            'model_count': 2,
            'host': 'http://localhost:11434'
        }
        mock_check_ollama.return_value = mock_status
        
        status = smart_client.get_ollama_status()
        assert status == mock_status
        assert status['available'] is True
        assert len(status['models']) == 2
    
    @patch('src.utils.smart_llm_client.check_ollama_availability')
    def test_get_ollama_status_unavailable(self, mock_check_ollama, smart_client):
        """Test Ollama status when unavailable."""
        mock_status = {
            'available': False,
            'reason': 'Connection refused',
            'models': [],
            'host': 'http://localhost:11434'
        }
        mock_check_ollama.return_value = mock_status
        
        status = smart_client.get_ollama_status()
        assert status['available'] is False
        assert 'Connection refused' in status['reason']
    
    def test_select_optimal_model_cloud_only(self, smart_client):
        """Test model selection with cloud-only strategy."""
        smart_client.strategy = ModelSelectionStrategy.CLOUD_ONLY
        requirements = TaskRequirements(task_type="summarization")
        
        provider, model, config = smart_client.select_optimal_model(requirements)
        
        assert provider in ['openai', 'anthropic']
        assert model in ['gpt-3.5-turbo', 'claude-3-sonnet-20240229']
    
    @patch('src.utils.smart_llm_client.check_ollama_availability')
    def test_select_optimal_model_chinese_content(self, mock_check_ollama, smart_client):
        """Test model selection for Chinese content."""
        mock_check_ollama.return_value = {
            'available': True,
            'models': ['qwen2.5:7b', 'llama3.1:8b'],
            'model_count': 2
        }
        
        requirements = TaskRequirements(
            task_type="summarization",
            language="zh-CN"
        )
        
        provider, model, config = smart_client.select_optimal_model(requirements)
        
        assert provider == 'ollama'
        assert 'qwen' in model.lower()
    
    @patch('src.utils.smart_llm_client.check_ollama_availability')
    def test_select_optimal_model_speed_priority(self, mock_check_ollama, smart_client):
        """Test model selection with speed priority."""
        mock_check_ollama.return_value = {
            'available': True,
            'models': ['llama3.2:3b', 'llama3.1:8b'],
            'model_count': 2
        }
        
        requirements = TaskRequirements(
            task_type="summarization",
            speed_priority=True
        )
        
        provider, model, config = smart_client.select_optimal_model(requirements)
        
        assert provider == 'ollama'
        # Should prefer local models for speed
    
    @patch('src.utils.call_llm.create_llm_client')
    @patch('src.utils.smart_llm_client.check_ollama_availability')
    def test_create_client_success(self, mock_check_ollama, mock_create_client, smart_client):
        """Test successful client creation."""
        mock_check_ollama.return_value = {
            'available': True,
            'models': ['llama3.1:8b'],
            'model_count': 1
        }
        
        mock_llm_client = Mock()
        mock_create_client.return_value = mock_llm_client
        
        requirements = TaskRequirements(task_type="summarization")
        client = smart_client.create_client(requirements)
        
        assert client == mock_llm_client
        mock_create_client.assert_called_once()
    
    @patch('src.utils.call_llm.create_llm_client')
    def test_create_client_with_fallback(self, mock_create_client, smart_client):
        """Test client creation with fallback on Ollama failure."""
        # First call (Ollama) fails, second call (fallback) succeeds
        mock_llm_client = Mock()
        mock_create_client.side_effect = [
            OllamaConnectionError("http://localhost:11434"),
            mock_llm_client
        ]
        
        requirements = TaskRequirements(task_type="summarization")
        client = smart_client.create_client(requirements)
        
        assert client == mock_llm_client
        assert mock_create_client.call_count == 2
    
    @patch('src.utils.smart_llm_client.is_chinese_text')
    @patch('src.utils.smart_llm_client.optimize_chinese_content_for_llm')
    def test_generate_text_with_chinese_optimization(self, mock_optimize, mock_is_chinese, smart_client):
        """Test Chinese text optimization."""
        mock_is_chinese.return_value = True
        mock_optimize.return_value = {
            'system_prompt': '中文总结专家...',
            'user_prompt': '请总结以下中文文本...',
            'optimized_text': '优化后的文本'
        }
        
        # Mock the fallback method to return a test result
        smart_client.generate_text_with_fallback = Mock(return_value={
            'text': '这是一个测试总结',
            'usage': {'prompt_tokens': 100, 'completion_tokens': 50, 'total_tokens': 150}
        })
        
        requirements = TaskRequirements(task_type="summarization")
        result = smart_client.generate_text_with_chinese_optimization(
            "这是一个中文测试文本",
            requirements
        )
        
        assert '这是一个测试总结' in result['text']
        mock_is_chinese.assert_called_once()
        mock_optimize.assert_called_once()
    
    def test_check_provider_health_ollama(self, smart_client):
        """Test provider health check for Ollama."""
        with patch.object(smart_client, 'get_ollama_status') as mock_status:
            mock_status.return_value = {
                'available': True,
                'models': ['llama3.1:8b'],
                'model_count': 1
            }
            
            health = smart_client.check_provider_health('ollama')
            
            assert health['provider'] == 'ollama'
            assert health['available'] is True
            assert len(health['models']) == 1
    
    def test_get_provider_fallback_chain(self, smart_client):
        """Test fallback chain generation."""
        with patch.object(smart_client, 'check_provider_health') as mock_health:
            mock_health.side_effect = [
                {'available': True, 'response_time': 100},  # openai
                {'available': True, 'response_time': 200}   # anthropic
            ]
            
            chain = smart_client.get_provider_fallback_chain('ollama')
            
            assert 'ollama' not in chain  # Excluded because it's the failed provider
            assert len(chain) >= 1
            # Should be sorted by response time
            if len(chain) > 1:
                assert chain == ['openai', 'anthropic']  # openai is faster


class TestChineseLanguageProcessing:
    """Test Chinese language processing functionality."""
    
    @patch('src.utils.smart_llm_client.is_chinese_text')
    @patch('src.utils.smart_llm_client.optimize_chinese_content_for_llm')
    def test_chinese_content_detection_and_optimization(self, mock_optimize, mock_is_chinese):
        """Test Chinese content detection and optimization."""
        mock_is_chinese.return_value = True
        mock_optimize.return_value = {
            'system_prompt': '你是一个专业的中文内容总结专家...',
            'user_prompt': '请总结以下中文文本内容：\n\n这是中文内容\n\n请提供一个专业的摘要：',
            'optimized_text': '这是中文内容'
        }
        
        client = SmartLLMClient()
        client.generate_text_with_fallback = Mock(return_value={
            'text': '这是一个测试摘要',
            'usage': {'prompt_tokens': 50, 'completion_tokens': 25, 'total_tokens': 75}
        })
        
        requirements = TaskRequirements(task_type="summarization")
        result = client.generate_text_with_chinese_optimization("这是中文内容", requirements)
        
        mock_is_chinese.assert_called_once_with("这是中文内容")
        mock_optimize.assert_called_once_with("这是中文内容", "summarization")
        assert result['text'] == '这是一个测试摘要'
    
    @patch('src.utils.smart_llm_client.is_chinese_text')
    def test_english_content_processing(self, mock_is_chinese):
        """Test English content processing (no optimization)."""
        mock_is_chinese.return_value = False
        
        client = SmartLLMClient()
        client.generate_text_with_fallback = Mock(return_value={
            'text': 'This is a test summary',
            'usage': {'prompt_tokens': 50, 'completion_tokens': 25, 'total_tokens': 75}
        })
        
        requirements = TaskRequirements(task_type="summarization")
        result = client.generate_text_with_chinese_optimization("This is English content", requirements)
        
        mock_is_chinese.assert_called_once_with("This is English content")
        assert result['text'] == 'This is a test summary'


class TestFallbackMechanisms:
    """Test fallback mechanisms and error handling."""
    
    def test_fallback_on_ollama_connection_error(self):
        """Test fallback when Ollama connection fails."""
        config = {
            'ollama': {
                'fallback_enabled': True,
                'fallback_provider': 'openai'
            },
            'openai_model': 'gpt-3.5-turbo',
            'max_tokens': 1000,
            'temperature': 0.7
        }
        
        client = SmartLLMClient(config)
        
        with patch.object(client, 'create_client') as mock_create_client:
            # First call fails with Ollama error
            mock_create_client.side_effect = OllamaConnectionError("localhost:11434")
            
            with patch.object(client, 'get_provider_fallback_chain') as mock_fallback_chain:
                mock_fallback_chain.return_value = ['openai']
                
                with patch.object(client, '_create_fallback_client') as mock_fallback_client:
                    mock_fallback_llm = Mock()
                    mock_fallback_llm.generate_text.return_value = {
                        'text': 'Fallback response',
                        'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}
                    }
                    mock_fallback_client.return_value = mock_fallback_llm
                    
                    requirements = TaskRequirements(task_type="summarization")
                    result = client.generate_text_with_fallback(
                        "Test prompt",
                        requirements
                    )
                    
                    assert result['text'] == 'Fallback response'
                    assert '_fallback_used' in result
                    assert result['_fallback_used']['fallback_provider'] == 'openai'
    
    def test_fallback_disabled(self):
        """Test behavior when fallback is disabled."""
        config = {
            'ollama': {
                'fallback_enabled': False
            }
        }
        
        client = SmartLLMClient(config)
        
        with patch.object(client, 'create_client') as mock_create_client:
            mock_create_client.side_effect = OllamaConnectionError("localhost:11434")
            
            requirements = TaskRequirements(task_type="summarization")
            
            with pytest.raises(OllamaConnectionError):
                client.generate_text_with_fallback("Test prompt", requirements)
    
    def test_all_providers_fail(self):
        """Test when all providers fail."""
        client = SmartLLMClient()
        
        with patch.object(client, 'create_client') as mock_create_client:
            mock_create_client.side_effect = LLMError("Initial error")
            
            with patch.object(client, 'get_provider_fallback_chain') as mock_fallback_chain:
                mock_fallback_chain.return_value = ['openai', 'anthropic']
                
                with patch.object(client, '_create_fallback_client') as mock_fallback_client:
                    mock_fallback_llm = Mock()
                    mock_fallback_llm.generate_text.side_effect = [
                        LLMError("OpenAI error"),
                        LLMError("Anthropic error")
                    ]
                    mock_fallback_client.return_value = mock_fallback_llm
                    
                    requirements = TaskRequirements(task_type="summarization")
                    
                    with pytest.raises(LLMError):
                        client.generate_text_with_fallback("Test prompt", requirements)


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_create_smart_client(self):
        """Test create_smart_client convenience function."""
        config = {'test': 'config'}
        client = create_smart_client(config)
        
        assert isinstance(client, SmartLLMClient)
        assert client.config == config
    
    def test_create_smart_client_no_config(self):
        """Test create_smart_client without config."""
        client = create_smart_client()
        
        assert isinstance(client, SmartLLMClient)
        assert client.config is not None
    
    @patch('src.utils.smart_llm_client.detect_language')
    def test_detect_task_requirements(self, mock_detect_language):
        """Test detect_task_requirements function."""
        mock_detect_language.return_value = 'zh-CN'
        
        requirements = detect_task_requirements(
            text="这是中文文本",
            task_type="summarization",
            quality_level="high"
        )
        
        assert requirements.task_type == "summarization"
        assert requirements.language == "zh-CN"
        assert requirements.quality_level == "high"
        assert requirements.text_length == 4  # Number of words
    
    def test_detect_task_requirements_no_text(self):
        """Test detect_task_requirements without text."""
        requirements = detect_task_requirements(
            task_type="keywords",
            speed_priority=True
        )
        
        assert requirements.task_type == "keywords"
        assert requirements.language is None
        assert requirements.text_length == 0
        assert requirements.speed_priority is True


if __name__ == '__main__':
    pytest.main([__file__])