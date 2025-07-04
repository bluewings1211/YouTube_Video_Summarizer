"""
Configuration module for YouTube Summarizer application.
Handles environment variables and application settings.
"""

import os
from typing import Optional, List
from decouple import config, Csv
from pydantic import BaseModel, validator
import logging


class AppConfig(BaseModel):
    """Main application configuration."""
    
    # Application
    app_name: str = config('APP_NAME', default='YouTube Summarizer')
    app_version: str = config('APP_VERSION', default='1.0.0')
    environment: str = config('ENVIRONMENT', default='development')
    debug: bool = config('DEBUG', default=False, cast=bool)
    log_level: str = config('LOG_LEVEL', default='info')
    
    # Server
    host: str = config('HOST', default='0.0.0.0')
    port: int = config('PORT', default=8000, cast=int)
    workers: int = config('WORKERS', default=4, cast=int)
    
    # API Keys
    openai_api_key: Optional[str] = config('OPENAI_API_KEY', default=None)
    anthropic_api_key: Optional[str] = config('ANTHROPIC_API_KEY', default=None)
    
    # LLM Configuration
    default_llm_provider: str = config('DEFAULT_LLM_PROVIDER', default='openai')
    openai_model: str = config('OPENAI_MODEL', default='gpt-4-turbo-preview')
    anthropic_model: str = config('ANTHROPIC_MODEL', default='claude-3-sonnet-20240229')
    max_tokens: int = config('MAX_TOKENS', default=4000, cast=int)
    temperature: float = config('TEMPERATURE', default=0.7, cast=float)
    llm_timeout: int = config('LLM_TIMEOUT', default=60, cast=int)
    llm_summarization_timeout: int = config('LLM_SUMMARIZATION_TIMEOUT', default=120, cast=int)
    llm_keyword_timeout: int = config('LLM_KEYWORD_TIMEOUT', default=45, cast=int)
    llm_timestamp_timeout: int = config('LLM_TIMESTAMP_TIMEOUT', default=90, cast=int)
    
    # YouTube API
    youtube_api_timeout: int = config('YOUTUBE_API_TIMEOUT', default=30, cast=int)
    youtube_metadata_timeout: int = config('YOUTUBE_METADATA_TIMEOUT', default=15, cast=int)
    transcript_fetch_timeout: int = config('TRANSCRIPT_FETCH_TIMEOUT', default=45, cast=int)
    max_video_duration: int = config('MAX_VIDEO_DURATION', default=1800, cast=int)
    retry_attempts: int = config('RETRY_ATTEMPTS', default=3, cast=int)
    retry_delay: int = config('RETRY_DELAY', default=1, cast=int)
    
    # Redis
    redis_host: str = config('REDIS_HOST', default='localhost')
    redis_port: int = config('REDIS_PORT', default=6379, cast=int)
    redis_db: int = config('REDIS_DB', default=0, cast=int)
    redis_password: Optional[str] = config('REDIS_PASSWORD', default=None)
    redis_cache_ttl: int = config('REDIS_CACHE_TTL', default=3600, cast=int)
    
    # Security
    secret_key: str = config('SECRET_KEY', default='dev-secret-key-change-in-production')
    cors_origins: List[str] = config('CORS_ORIGINS', default='http://localhost:3000,http://localhost:8000', cast=Csv())
    api_rate_limit: int = config('API_RATE_LIMIT', default=100, cast=int)
    
    # Logging
    log_format: str = config('LOG_FORMAT', default='json')
    log_file_path: str = config('LOG_FILE_PATH', default='logs/app.log')
    log_max_size: str = config('LOG_MAX_SIZE', default='10MB')
    log_backup_count: int = config('LOG_BACKUP_COUNT', default=5, cast=int)
    
    # Performance
    request_timeout: int = config('REQUEST_TIMEOUT', default=300, cast=int)
    workflow_timeout: int = config('WORKFLOW_TIMEOUT', default=600, cast=int)
    node_timeout: int = config('NODE_TIMEOUT', default=180, cast=int)
    max_content_length: str = config('MAX_CONTENT_LENGTH', default='50MB')
    enable_caching: bool = config('ENABLE_CACHING', default=True, cast=bool)
    
    # Development
    hot_reload: bool = config('HOT_RELOAD', default=True, cast=bool)
    auto_reload: bool = config('AUTO_RELOAD', default=True, cast=bool)
    pytest_timeout: int = config('PYTEST_TIMEOUT', default=30, cast=int)
    
    @validator('environment')
    def validate_environment(cls, v):
        """Validate environment value."""
        valid_environments = ['development', 'staging', 'production']
        if v not in valid_environments:
            raise ValueError(f'Environment must be one of: {valid_environments}')
        return v
    
    @validator('default_llm_provider')
    def validate_llm_provider(cls, v):
        """Validate LLM provider."""
        valid_providers = ['openai', 'anthropic']
        if v not in valid_providers:
            raise ValueError(f'LLM provider must be one of: {valid_providers}')
        return v
    
    @validator('log_level')
    def validate_log_level(cls, v):
        """Validate log level."""
        valid_levels = ['debug', 'info', 'warning', 'error', 'critical']
        if v.lower() not in valid_levels:
            raise ValueError(f'Log level must be one of: {valid_levels}')
        return v.lower()
    
    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.environment == 'production'
    
    @property
    def is_development(self) -> bool:
        """Check if running in development."""
        return self.environment == 'development'
    
    @property
    def redis_url(self) -> str:
        """Get Redis connection URL."""
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/{self.redis_db}"
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"
    
    def get_logging_config(self) -> dict:
        """Get logging configuration."""
        return {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'json': {
                    'class': 'pythonjsonlogger.jsonlogger.JsonFormatter',
                    'format': '%(asctime)s %(name)s %(levelname)s %(message)s'
                },
                'standard': {
                    'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
                }
            },
            'handlers': {
                'console': {
                    'class': 'logging.StreamHandler',
                    'formatter': 'json' if self.log_format == 'json' else 'standard',
                    'level': self.log_level.upper()
                },
                'file': {
                    'class': 'logging.handlers.RotatingFileHandler',
                    'filename': self.log_file_path,
                    'maxBytes': self._parse_size(self.log_max_size),
                    'backupCount': self.log_backup_count,
                    'formatter': 'json' if self.log_format == 'json' else 'standard',
                    'level': self.log_level.upper()
                }
            },
            'loggers': {
                '': {
                    'handlers': ['console', 'file'],
                    'level': self.log_level.upper(),
                    'propagate': False
                }
            }
        }
    
    def _parse_size(self, size_str: str) -> int:
        """Parse size string to bytes."""
        size_str = size_str.upper()
        if size_str.endswith('KB'):
            return int(size_str[:-2]) * 1024
        elif size_str.endswith('MB'):
            return int(size_str[:-2]) * 1024 * 1024
        elif size_str.endswith('GB'):
            return int(size_str[:-2]) * 1024 * 1024 * 1024
        else:
            return int(size_str)


# Global configuration instance
settings = AppConfig()


def setup_logging():
    """Set up logging configuration."""
    # Ensure log directory exists
    log_dir = os.path.dirname(settings.log_file_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Configure logging
    logging.config.dictConfig(settings.get_logging_config())
    
    # Log startup information
    logger = logging.getLogger(__name__)
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Debug mode: {settings.debug}")
    logger.info(f"Log level: {settings.log_level}")


if __name__ == "__main__":
    # For testing configuration
    print("Configuration loaded successfully:")
    print(f"App Name: {settings.app_name}")
    print(f"Environment: {settings.environment}")
    print(f"Debug: {settings.debug}")
    print(f"Redis URL: {settings.redis_url}")
    print(f"Is Production: {settings.is_production}")
    print(f"Is Development: {settings.is_development}")