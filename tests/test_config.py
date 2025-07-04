"""
Unit tests for configuration module.
"""

import pytest
import os
from unittest.mock import patch, MagicMock
from pydantic import ValidationError

from src.config import AppConfig, settings, setup_logging


class TestAppConfig:
    """Test cases for AppConfig class."""
    
    def test_default_config_creation(self):
        """Test that default configuration can be created."""
        config = AppConfig()
        assert config.app_name == "YouTube Summarizer"
        assert config.environment == "development"
        assert config.port == 8000
        assert config.debug is False
    
    def test_config_from_env_vars(self):
        """Test configuration loading from environment variables."""
        with patch.dict(os.environ, {
            'APP_NAME': 'Test App',
            'PORT': '9000',
            'DEBUG': 'true',
            'ENVIRONMENT': 'production'
        }):
            config = AppConfig()
            assert config.app_name == "Test App"
            assert config.port == 9000
            assert config.debug is True
            assert config.environment == "production"
    
    def test_invalid_environment_raises_error(self):
        """Test that invalid environment raises validation error."""
        with patch.dict(os.environ, {'ENVIRONMENT': 'invalid'}):
            with pytest.raises(ValidationError):
                AppConfig()
    
    def test_invalid_llm_provider_raises_error(self):
        """Test that invalid LLM provider raises validation error."""
        with patch.dict(os.environ, {'DEFAULT_LLM_PROVIDER': 'invalid'}):
            with pytest.raises(ValidationError):
                AppConfig()
    
    def test_invalid_log_level_raises_error(self):
        """Test that invalid log level raises validation error."""
        with patch.dict(os.environ, {'LOG_LEVEL': 'invalid'}):
            with pytest.raises(ValidationError):
                AppConfig()
    
    def test_is_production_property(self):
        """Test is_production property."""
        config = AppConfig()
        assert config.is_production is False
        
        with patch.dict(os.environ, {'ENVIRONMENT': 'production'}):
            config = AppConfig()
            assert config.is_production is True
    
    def test_is_development_property(self):
        """Test is_development property."""
        config = AppConfig()
        assert config.is_development is True
        
        with patch.dict(os.environ, {'ENVIRONMENT': 'production'}):
            config = AppConfig()
            assert config.is_development is False
    
    def test_redis_url_without_password(self):
        """Test Redis URL generation without password."""
        config = AppConfig()
        expected = f"redis://{config.redis_host}:{config.redis_port}/{config.redis_db}"
        assert config.redis_url == expected
    
    def test_redis_url_with_password(self):
        """Test Redis URL generation with password."""
        with patch.dict(os.environ, {'REDIS_PASSWORD': 'testpass'}):
            config = AppConfig()
            expected = f"redis://:testpass@{config.redis_host}:{config.redis_port}/{config.redis_db}"
            assert config.redis_url == expected
    
    def test_get_logging_config(self):
        """Test logging configuration generation."""
        config = AppConfig()
        log_config = config.get_logging_config()
        
        assert 'version' in log_config
        assert 'formatters' in log_config
        assert 'handlers' in log_config
        assert 'loggers' in log_config
        
        # Check that both json and standard formatters exist
        assert 'json' in log_config['formatters']
        assert 'standard' in log_config['formatters']
        
        # Check that console and file handlers exist
        assert 'console' in log_config['handlers']
        assert 'file' in log_config['handlers']
    
    def test_parse_size_methods(self):
        """Test size parsing methods."""
        config = AppConfig()
        
        # Test KB
        assert config._parse_size('10KB') == 10 * 1024
        
        # Test MB
        assert config._parse_size('10MB') == 10 * 1024 * 1024
        
        # Test GB
        assert config._parse_size('10GB') == 10 * 1024 * 1024 * 1024
        
        # Test plain number
        assert config._parse_size('1024') == 1024


class TestSetupLogging:
    """Test cases for setup_logging function."""
    
    @patch('src.config.os.makedirs')
    @patch('src.config.os.path.exists')
    @patch('src.config.logging.config.dictConfig')
    def test_setup_logging_creates_directory(self, mock_dict_config, mock_exists, mock_makedirs):
        """Test that setup_logging creates log directory if it doesn't exist."""
        mock_exists.return_value = False
        
        setup_logging()
        
        mock_makedirs.assert_called_once()
        mock_dict_config.assert_called_once()
    
    @patch('src.config.os.makedirs')
    @patch('src.config.os.path.exists')
    @patch('src.config.logging.config.dictConfig')
    def test_setup_logging_skips_directory_creation(self, mock_dict_config, mock_exists, mock_makedirs):
        """Test that setup_logging skips directory creation if it exists."""
        mock_exists.return_value = True
        
        setup_logging()
        
        mock_makedirs.assert_not_called()
        mock_dict_config.assert_called_once()


class TestGlobalSettings:
    """Test cases for global settings instance."""
    
    def test_global_settings_exists(self):
        """Test that global settings instance exists."""
        assert settings is not None
        assert isinstance(settings, AppConfig)
    
    def test_global_settings_has_default_values(self):
        """Test that global settings has expected default values."""
        assert settings.app_name == "YouTube Summarizer"
        assert settings.app_version == "1.0.0"
        assert settings.port == 8000