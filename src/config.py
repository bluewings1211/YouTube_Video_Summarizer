"""
Configuration module for YouTube Summarizer application.
Handles environment variables and application settings.
"""

import os
from typing import Optional, List, Dict, Any
from decouple import config, Csv
from pydantic import BaseModel, validator
import logging
import logging.config
try:
    from .utils.credential_manager import get_credential_manager, secure_credential_fallback
except ImportError:
    # Fallback if credential manager is not available
    def get_credential_manager():
        return None
    
    def secure_credential_fallback(env_var: str, category: str, key: str, default=None):
        return os.environ.get(env_var, default)


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
    
    # API Keys (with secure credential fallback)
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    
    # LLM Configuration
    default_llm_provider: str = config('DEFAULT_LLM_PROVIDER', default='ollama')
    openai_model: str = config('OPENAI_MODEL', default='gpt-4-turbo-preview')
    anthropic_model: str = config('ANTHROPIC_MODEL', default='claude-3-sonnet-20240229')
    ollama_model: str = config('OLLAMA_MODEL', default='llama3.1:8b')
    max_tokens: int = config('MAX_TOKENS', default=4000, cast=int)
    temperature: float = config('TEMPERATURE', default=0.7, cast=float)
    llm_timeout: int = config('LLM_TIMEOUT', default=60, cast=int)
    llm_summarization_timeout: int = config('LLM_SUMMARIZATION_TIMEOUT', default=120, cast=int)
    llm_keyword_timeout: int = config('LLM_KEYWORD_TIMEOUT', default=45, cast=int)
    llm_timestamp_timeout: int = config('LLM_TIMESTAMP_TIMEOUT', default=90, cast=int)
    
    # Ollama Configuration
    ollama_host: str = config('OLLAMA_HOST', default='http://localhost:11434')
    ollama_keep_alive: str = config('OLLAMA_KEEP_ALIVE', default='5m')
    ollama_fallback_enabled: bool = config('OLLAMA_FALLBACK_ENABLED', default=True, cast=bool)
    ollama_fallback_provider: str = config('OLLAMA_FALLBACK_PROVIDER', default='openai')
    ollama_connection_timeout: int = config('OLLAMA_CONNECTION_TIMEOUT', default=10, cast=int)
    
    # Model Selection Strategy
    model_selection_strategy: str = config('MODEL_SELECTION_STRATEGY', default='auto')  # auto, prefer_local, prefer_cloud, cloud_only
    chinese_language_model: str = config('CHINESE_LANGUAGE_MODEL', default='qwen2.5:7b')
    performance_model: str = config('PERFORMANCE_MODEL', default='mistral:7b')
    lightweight_model: str = config('LIGHTWEIGHT_MODEL', default='llama3.2:3b')
    
    # Advanced Model Selection Configuration
    fallback_model_chain: List[str] = config('FALLBACK_MODEL_CHAIN', default='llama3.1:8b,mistral:7b,llama3.2:3b', cast=Csv())
    model_priority_by_task: Dict[str, str] = {}  # Will be populated from environment
    model_load_balancing: bool = config('MODEL_LOAD_BALANCING', default=False, cast=bool)
    model_health_check_enabled: bool = config('MODEL_HEALTH_CHECK_ENABLED', default=True, cast=bool)
    model_health_check_interval: int = config('MODEL_HEALTH_CHECK_INTERVAL', default=300, cast=int)
    model_response_time_threshold: float = config('MODEL_RESPONSE_TIME_THRESHOLD', default=30.0, cast=float)
    model_availability_threshold: float = config('MODEL_AVAILABILITY_THRESHOLD', default=0.95, cast=float)
    
    # Cloud Provider Specific Models
    openai_gpt4_model: str = config('OPENAI_GPT4_MODEL', default='gpt-4o')
    openai_gpt35_model: str = config('OPENAI_GPT35_MODEL', default='gpt-3.5-turbo')
    anthropic_claude_model: str = config('ANTHROPIC_CLAUDE_MODEL', default='claude-3-5-sonnet-20241022')
    anthropic_haiku_model: str = config('ANTHROPIC_HAIKU_MODEL', default='claude-3-haiku-20240307')
    
    # Local Model Management
    ollama_auto_pull: bool = config('OLLAMA_AUTO_PULL', default=True, cast=bool)
    ollama_model_cache_size: int = config('OLLAMA_MODEL_CACHE_SIZE', default=5, cast=int)
    ollama_gpu_memory_fraction: float = config('OLLAMA_GPU_MEMORY_FRACTION', default=0.8, cast=float)
    ollama_cpu_threads: int = config('OLLAMA_CPU_THREADS', default=0, cast=int)  # 0 means auto
    ollama_context_size: int = config('OLLAMA_CONTEXT_SIZE', default=4096, cast=int)
    
    # Model Selection Criteria
    preferred_model_size: str = config('PREFERRED_MODEL_SIZE', default='medium')  # small, medium, large
    max_model_memory_gb: int = config('MAX_MODEL_MEMORY_GB', default=8, cast=int)
    prefer_quantized_models: bool = config('PREFER_QUANTIZED_MODELS', default=True, cast=bool)
    language_specific_routing: bool = config('LANGUAGE_SPECIFIC_ROUTING', default=True, cast=bool)
    
    # YouTube API
    youtube_api_timeout: int = config('YOUTUBE_API_TIMEOUT', default=30, cast=int)
    youtube_metadata_timeout: int = config('YOUTUBE_METADATA_TIMEOUT', default=15, cast=int)
    transcript_fetch_timeout: int = config('TRANSCRIPT_FETCH_TIMEOUT', default=45, cast=int)
    max_video_duration: int = config('MAX_VIDEO_DURATION', default=1800, cast=int)
    retry_attempts: int = config('RETRY_ATTEMPTS', default=3, cast=int)
    retry_delay: int = config('RETRY_DELAY', default=1, cast=int)
    
    # Proxy Configuration
    proxy_enabled: bool = config('PROXY_ENABLED', default=False, cast=bool)
    proxy_urls: List[str] = config('PROXY_URLS', default='', cast=Csv())
    proxy_username: Optional[str] = None
    proxy_password: Optional[str] = None
    proxy_timeout: int = config('PROXY_TIMEOUT', default=30, cast=int)
    proxy_health_check_interval: int = config('PROXY_HEALTH_CHECK_INTERVAL', default=300, cast=int)
    proxy_health_check_timeout: int = config('PROXY_HEALTH_CHECK_TIMEOUT', default=10, cast=int)
    proxy_max_failures: int = config('PROXY_MAX_FAILURES', default=3, cast=int)
    proxy_rotation_enabled: bool = config('PROXY_ROTATION_ENABLED', default=True, cast=bool)
    proxy_pool_size: int = config('PROXY_POOL_SIZE', default=10, cast=int)
    
    # Enhanced Proxy Configuration
    proxy_auth_type: str = config('PROXY_AUTH_TYPE', default='basic')  # basic, digest, ntlm
    proxy_socks_version: str = config('PROXY_SOCKS_VERSION', default='5')  # 4, 5
    proxy_ssl_verify: bool = config('PROXY_SSL_VERIFY', default=True, cast=bool)
    proxy_user_agent: str = config('PROXY_USER_AGENT', default='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
    proxy_connection_retry_attempts: int = config('PROXY_CONNECTION_RETRY_ATTEMPTS', default=3, cast=int)
    proxy_connection_retry_delay: float = config('PROXY_CONNECTION_RETRY_DELAY', default=1.0, cast=float)
    proxy_dns_resolution: str = config('PROXY_DNS_RESOLUTION', default='remote')  # local, remote
    proxy_keep_alive: bool = config('PROXY_KEEP_ALIVE', default=True, cast=bool)
    proxy_max_connections: int = config('PROXY_MAX_CONNECTIONS', default=100, cast=int)
    proxy_max_connections_per_host: int = config('PROXY_MAX_CONNECTIONS_PER_HOST', default=10, cast=int)
    
    # Proxy Security
    proxy_whitelist_domains: List[str] = config('PROXY_WHITELIST_DOMAINS', default='', cast=Csv())
    proxy_blacklist_domains: List[str] = config('PROXY_BLACKLIST_DOMAINS', default='', cast=Csv())
    proxy_enable_logging: bool = config('PROXY_ENABLE_LOGGING', default=False, cast=bool)
    proxy_log_level: str = config('PROXY_LOG_LEVEL', default='warning')
    proxy_headers: Dict[str, str] = {}  # Will be populated from env vars with PROXY_HEADER_ prefix
    
    # Rate Limiting
    rate_limit_enabled: bool = config('RATE_LIMIT_ENABLED', default=True, cast=bool)
    rate_limit_min_interval: int = config('RATE_LIMIT_MIN_INTERVAL', default=10, cast=int)
    rate_limit_max_requests_per_minute: int = config('RATE_LIMIT_MAX_REQUESTS_PER_MINUTE', default=6, cast=int)
    rate_limit_burst_requests: int = config('RATE_LIMIT_BURST_REQUESTS', default=3, cast=int)
    
    # Retry Configuration
    retry_enabled: bool = config('RETRY_ENABLED', default=True, cast=bool)
    retry_max_attempts: int = config('RETRY_MAX_ATTEMPTS', default=5, cast=int)
    retry_base_delay: float = config('RETRY_BASE_DELAY', default=1.0, cast=float)
    retry_max_delay: float = config('RETRY_MAX_DELAY', default=60.0, cast=float)
    retry_backoff_multiplier: float = config('RETRY_BACKOFF_MULTIPLIER', default=2.0, cast=float)
    retry_jitter_enabled: bool = config('RETRY_JITTER_ENABLED', default=True, cast=bool)
    
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
    
    # Enhanced Security Configuration
    encryption_key: Optional[str] = config('ENCRYPTION_KEY', default=None)
    ssl_cert_path: Optional[str] = config('SSL_CERT_PATH', default=None)
    ssl_key_path: Optional[str] = config('SSL_KEY_PATH', default=None)
    ssl_ca_path: Optional[str] = config('SSL_CA_PATH', default=None)
    jwt_secret_key: Optional[str] = config('JWT_SECRET_KEY', default=None)
    jwt_algorithm: str = config('JWT_ALGORITHM', default='HS256')
    jwt_expiration_hours: int = config('JWT_EXPIRATION_HOURS', default=24, cast=int)
    api_key_header: str = config('API_KEY_HEADER', default='X-API-Key')
    max_request_size: str = config('MAX_REQUEST_SIZE', default='10MB')
    enable_https_redirect: bool = config('ENABLE_HTTPS_REDIRECT', default=False, cast=bool)
    enable_security_headers: bool = config('ENABLE_SECURITY_HEADERS', default=True, cast=bool)
    
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
    
    def __init__(self, **kwargs):
        """Initialize configuration and load proxy headers from environment."""
        super().__init__(**kwargs)
        self._load_proxy_headers()
        self._load_model_priority_config()
        self._load_secure_credentials()
    
    def _load_proxy_headers(self):
        """Load proxy headers from environment variables with PROXY_HEADER_ prefix."""
        proxy_headers = {}
        for key, value in os.environ.items():
            if key.startswith('PROXY_HEADER_'):
                header_name = key[13:].replace('_', '-')  # Remove PROXY_HEADER_ prefix and convert _ to -
                proxy_headers[header_name] = value
        self.proxy_headers = proxy_headers
    
    def _load_model_priority_config(self):
        """Load model priority configuration from environment variables with MODEL_PRIORITY_ prefix."""
        model_priority = {}
        for key, value in os.environ.items():
            if key.startswith('MODEL_PRIORITY_'):
                task_name = key[15:].lower()  # Remove MODEL_PRIORITY_ prefix and convert to lowercase
                model_priority[task_name] = value
        self.model_priority_by_task = model_priority
    
    def _load_secure_credentials(self):
        """Load secure credentials using credential manager."""
        try:
            # Load API keys with secure fallback
            self.openai_api_key = secure_credential_fallback(
                'OPENAI_API_KEY', 'api_keys', 'openai'
            )
            self.anthropic_api_key = secure_credential_fallback(
                'ANTHROPIC_API_KEY', 'api_keys', 'anthropic'
            )
            
            # Load proxy credentials with secure fallback
            self.proxy_username = secure_credential_fallback(
                'PROXY_USERNAME', 'proxy', 'username'
            )
            self.proxy_password = secure_credential_fallback(
                'PROXY_PASSWORD', 'proxy', 'password'
            )
            
        except Exception as e:
            # If credential manager fails, fall back to environment variables
            logger.warning(f"Failed to load secure credentials, using environment variables: {e}")
            self.openai_api_key = config('OPENAI_API_KEY', default=None)
            self.anthropic_api_key = config('ANTHROPIC_API_KEY', default=None)
            self.proxy_username = config('PROXY_USERNAME', default=None)
            self.proxy_password = config('PROXY_PASSWORD', default=None)
    
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
        valid_providers = ['openai', 'anthropic', 'ollama']
        if v not in valid_providers:
            raise ValueError(f'LLM provider must be one of: {valid_providers}')
        return v
    
    @validator('ollama_fallback_provider')
    def validate_ollama_fallback_provider(cls, v):
        """Validate Ollama fallback provider."""
        valid_providers = ['openai', 'anthropic']
        if v not in valid_providers:
            raise ValueError(f'Ollama fallback provider must be one of: {valid_providers}')
        return v
    
    @validator('model_selection_strategy')
    def validate_model_selection_strategy(cls, v):
        """Validate model selection strategy."""
        valid_strategies = ['auto', 'prefer_local', 'prefer_cloud', 'cloud_only']
        if v not in valid_strategies:
            raise ValueError(f'Model selection strategy must be one of: {valid_strategies}')
        return v
    
    @validator('log_level')
    def validate_log_level(cls, v):
        """Validate log level."""
        valid_levels = ['debug', 'info', 'warning', 'error', 'critical']
        if v.lower() not in valid_levels:
            raise ValueError(f'Log level must be one of: {valid_levels}')
        return v.lower()
    
    @validator('proxy_urls')
    def validate_proxy_urls(cls, v):
        """Validate proxy URLs format."""
        if not v:
            return []
        
        validated_urls = []
        for url in v:
            url = url.strip()
            if not url:
                continue
                
            # Check if URL has a scheme, if not add http://
            if not url.startswith(('http://', 'https://', 'socks4://', 'socks5://')):
                url = f'http://{url}'
            
            # Basic URL validation
            try:
                from urllib.parse import urlparse
                parsed = urlparse(url)
                if not parsed.hostname:
                    raise ValueError(f'Invalid proxy URL: {url}')
            except Exception:
                raise ValueError(f'Invalid proxy URL format: {url}')
            
            validated_urls.append(url)
        
        return validated_urls
    
    @validator('rate_limit_min_interval')
    def validate_rate_limit_interval(cls, v):
        """Validate rate limit minimum interval."""
        if v < 1:
            raise ValueError('Rate limit minimum interval must be at least 1 second')
        return v
    
    @validator('retry_max_attempts')
    def validate_retry_attempts(cls, v):
        """Validate retry max attempts."""
        if v < 1:
            raise ValueError('Retry max attempts must be at least 1')
        if v > 10:
            raise ValueError('Retry max attempts should not exceed 10')
        return v
    
    @validator('retry_backoff_multiplier')
    def validate_backoff_multiplier(cls, v):
        """Validate retry backoff multiplier."""
        if v < 1.0:
            raise ValueError('Retry backoff multiplier must be at least 1.0')
        if v > 5.0:
            raise ValueError('Retry backoff multiplier should not exceed 5.0')
        return v
    
    @validator('proxy_auth_type')
    def validate_proxy_auth_type(cls, v):
        """Validate proxy authentication type."""
        valid_types = ['basic', 'digest', 'ntlm']
        if v not in valid_types:
            raise ValueError(f'Proxy auth type must be one of: {valid_types}')
        return v
    
    @validator('proxy_socks_version')
    def validate_proxy_socks_version(cls, v):
        """Validate SOCKS proxy version."""
        valid_versions = ['4', '5']
        if v not in valid_versions:
            raise ValueError(f'SOCKS proxy version must be one of: {valid_versions}')
        return v
    
    @validator('proxy_dns_resolution')
    def validate_proxy_dns_resolution(cls, v):
        """Validate proxy DNS resolution method."""
        valid_methods = ['local', 'remote']
        if v not in valid_methods:
            raise ValueError(f'Proxy DNS resolution must be one of: {valid_methods}')
        return v
    
    @validator('proxy_log_level')
    def validate_proxy_log_level(cls, v):
        """Validate proxy log level."""
        valid_levels = ['debug', 'info', 'warning', 'error', 'critical']
        if v.lower() not in valid_levels:
            raise ValueError(f'Proxy log level must be one of: {valid_levels}')
        return v.lower()
    
    @validator('jwt_algorithm')
    def validate_jwt_algorithm(cls, v):
        """Validate JWT algorithm."""
        valid_algorithms = ['HS256', 'HS384', 'HS512', 'RS256', 'RS384', 'RS512', 'ES256', 'ES384', 'ES512']
        if v not in valid_algorithms:
            raise ValueError(f'JWT algorithm must be one of: {valid_algorithms}')
        return v
    
    @validator('preferred_model_size')
    def validate_preferred_model_size(cls, v):
        """Validate preferred model size."""
        valid_sizes = ['small', 'medium', 'large']
        if v not in valid_sizes:
            raise ValueError(f'Preferred model size must be one of: {valid_sizes}')
        return v
    
    @validator('ollama_gpu_memory_fraction')
    def validate_gpu_memory_fraction(cls, v):
        """Validate GPU memory fraction."""
        if v < 0.1 or v > 1.0:
            raise ValueError('GPU memory fraction must be between 0.1 and 1.0')
        return v
    
    @validator('model_availability_threshold')
    def validate_availability_threshold(cls, v):
        """Validate model availability threshold."""
        if v < 0.0 or v > 1.0:
            raise ValueError('Model availability threshold must be between 0.0 and 1.0')
        return v
    
    @validator('model_response_time_threshold')
    def validate_response_time_threshold(cls, v):
        """Validate model response time threshold."""
        if v < 1.0:
            raise ValueError('Model response time threshold must be at least 1.0 seconds')
        return v
    
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
    
    @property
    def proxy_credentials(self) -> Optional[Dict[str, str]]:
        """Get proxy credentials if configured."""
        if self.proxy_username and self.proxy_password:
            return {
                'username': self.proxy_username,
                'password': self.proxy_password
            }
        return None
    
    @property
    def has_proxy_config(self) -> bool:
        """Check if proxy configuration is available."""
        return self.proxy_enabled and len(self.proxy_urls) > 0
    
    @property
    def proxy_config(self) -> Dict[str, Any]:
        """Get complete proxy configuration."""
        return {
            'enabled': self.proxy_enabled,
            'urls': self.proxy_urls,
            'credentials': self.proxy_credentials,
            'timeout': self.proxy_timeout,
            'health_check_interval': self.proxy_health_check_interval,
            'health_check_timeout': self.proxy_health_check_timeout,
            'max_failures': self.proxy_max_failures,
            'rotation_enabled': self.proxy_rotation_enabled,
            'pool_size': self.proxy_pool_size,
            'auth_type': self.proxy_auth_type,
            'socks_version': self.proxy_socks_version,
            'ssl_verify': self.proxy_ssl_verify,
            'user_agent': self.proxy_user_agent,
            'connection_retry_attempts': self.proxy_connection_retry_attempts,
            'connection_retry_delay': self.proxy_connection_retry_delay,
            'dns_resolution': self.proxy_dns_resolution,
            'keep_alive': self.proxy_keep_alive,
            'max_connections': self.proxy_max_connections,
            'max_connections_per_host': self.proxy_max_connections_per_host,
            'whitelist_domains': self.proxy_whitelist_domains,
            'blacklist_domains': self.proxy_blacklist_domains,
            'enable_logging': self.proxy_enable_logging,
            'log_level': self.proxy_log_level,
            'headers': self.proxy_headers
        }
    
    @property
    def rate_limit_config(self) -> Dict[str, Any]:
        """Get complete rate limiting configuration."""
        return {
            'enabled': self.rate_limit_enabled,
            'min_interval': self.rate_limit_min_interval,
            'max_requests_per_minute': self.rate_limit_max_requests_per_minute,
            'burst_requests': self.rate_limit_burst_requests
        }
    
    @property
    def retry_config(self) -> Dict[str, Any]:
        """Get complete retry configuration."""
        return {
            'enabled': self.retry_enabled,
            'max_attempts': self.retry_max_attempts,
            'base_delay': self.retry_base_delay,
            'max_delay': self.retry_max_delay,
            'backoff_multiplier': self.retry_backoff_multiplier,
            'jitter_enabled': self.retry_jitter_enabled
        }
    
    @property
    def ollama_config(self) -> Dict[str, Any]:
        """Get complete Ollama configuration."""
        return {
            'host': self.ollama_host,
            'model': self.ollama_model,
            'keep_alive': self.ollama_keep_alive,
            'fallback_enabled': self.ollama_fallback_enabled,
            'fallback_provider': self.ollama_fallback_provider,
            'connection_timeout': self.ollama_connection_timeout,
            'auto_pull': self.ollama_auto_pull,
            'model_cache_size': self.ollama_model_cache_size,
            'gpu_memory_fraction': self.ollama_gpu_memory_fraction,
            'cpu_threads': self.ollama_cpu_threads,
            'context_size': self.ollama_context_size
        }
    
    @property
    def llm_config(self) -> Dict[str, Any]:
        """Get complete LLM configuration."""
        return {
            'default_provider': self.default_llm_provider,
            'openai_model': self.openai_model,
            'anthropic_model': self.anthropic_model,
            'ollama_model': self.ollama_model,
            'max_tokens': self.max_tokens,
            'temperature': self.temperature,
            'timeout': self.llm_timeout,
            'model_selection_strategy': self.model_selection_strategy,
            'chinese_language_model': self.chinese_language_model,
            'performance_model': self.performance_model,
            'lightweight_model': self.lightweight_model,
            'fallback_model_chain': self.fallback_model_chain,
            'model_priority_by_task': self.model_priority_by_task,
            'model_load_balancing': self.model_load_balancing,
            'model_health_check_enabled': self.model_health_check_enabled,
            'model_health_check_interval': self.model_health_check_interval,
            'model_response_time_threshold': self.model_response_time_threshold,
            'model_availability_threshold': self.model_availability_threshold,
            'openai_gpt4_model': self.openai_gpt4_model,
            'openai_gpt35_model': self.openai_gpt35_model,
            'anthropic_claude_model': self.anthropic_claude_model,
            'anthropic_haiku_model': self.anthropic_haiku_model,
            'preferred_model_size': self.preferred_model_size,
            'max_model_memory_gb': self.max_model_memory_gb,
            'prefer_quantized_models': self.prefer_quantized_models,
            'language_specific_routing': self.language_specific_routing,
            'ollama': self.ollama_config
        }
    
    @property
    def security_config(self) -> Dict[str, Any]:
        """Get complete security configuration."""
        return {
            'secret_key': self.secret_key,
            'cors_origins': self.cors_origins,
            'api_rate_limit': self.api_rate_limit,
            'encryption_key': self.encryption_key,
            'ssl_cert_path': self.ssl_cert_path,
            'ssl_key_path': self.ssl_key_path,
            'ssl_ca_path': self.ssl_ca_path,
            'jwt_secret_key': self.jwt_secret_key,
            'jwt_algorithm': self.jwt_algorithm,
            'jwt_expiration_hours': self.jwt_expiration_hours,
            'api_key_header': self.api_key_header,
            'max_request_size': self.max_request_size,
            'enable_https_redirect': self.enable_https_redirect,
            'enable_security_headers': self.enable_security_headers
        }
    
    @property
    def has_ssl_config(self) -> bool:
        """Check if SSL configuration is available."""
        return self.ssl_cert_path is not None and self.ssl_key_path is not None
    
    @property
    def has_jwt_config(self) -> bool:
        """Check if JWT configuration is available."""
        return self.jwt_secret_key is not None
    
    def get_credential_manager(self):
        """Get credential manager instance."""
        return get_credential_manager()
    
    def store_secure_credential(self, category: str, key: str, value: str):
        """Store a credential securely."""
        cm = self.get_credential_manager()
        if cm:
            cm.store_credential(category, key, value)
        else:
            logger.warning(f"Credential manager not available, cannot store {category}.{key}")
    
    def get_secure_credential(self, category: str, key: str) -> Optional[str]:
        """Get a secure credential."""
        cm = self.get_credential_manager()
        if cm:
            return cm.get_credential(category, key)
        return None
    
    def get_legacy_config(self) -> Dict[str, Any]:
        """Get configuration in legacy format for backward compatibility."""
        return {
            'app_name': self.app_name,
            'app_version': self.app_version,
            'environment': self.environment,
            'debug': self.debug,
            'host': self.host,
            'port': self.port,
            'workers': self.workers,
            'openai_api_key': self.openai_api_key,
            'anthropic_api_key': self.anthropic_api_key,
            'default_llm_provider': self.default_llm_provider,
            'openai_model': self.openai_model,
            'anthropic_model': self.anthropic_model,
            'ollama_model': self.ollama_model,
            'ollama_host': self.ollama_host,
            'max_tokens': self.max_tokens,
            'temperature': self.temperature,
            'proxy_enabled': self.proxy_enabled,
            'proxy_urls': self.proxy_urls,
            'proxy_username': self.proxy_username,
            'proxy_password': self.proxy_password,
            'redis_host': self.redis_host,
            'redis_port': self.redis_port,
            'redis_db': self.redis_db,
            'redis_password': self.redis_password,
            'secret_key': self.secret_key,
            'cors_origins': self.cors_origins,
            'log_level': self.log_level,
            'log_file_path': self.log_file_path
        }
    
    def migrate_from_legacy_config(self, legacy_config: Dict[str, Any]):
        """Migrate configuration from legacy format."""
        for key, value in legacy_config.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        # Store sensitive credentials in credential manager
        if self.openai_api_key:
            self.store_secure_credential('api_keys', 'openai', self.openai_api_key)
        if self.anthropic_api_key:
            self.store_secure_credential('api_keys', 'anthropic', self.anthropic_api_key)
        if self.proxy_username and self.proxy_password:
            self.store_secure_credential('proxy', 'username', self.proxy_username)
            self.store_secure_credential('proxy', 'password', self.proxy_password)
    
    def validate_configuration(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []
        
        # Check required fields based on provider
        if self.default_llm_provider == 'openai' and not self.openai_api_key:
            issues.append("OpenAI API key is required when using OpenAI provider")
        
        if self.default_llm_provider == 'anthropic' and not self.anthropic_api_key:
            issues.append("Anthropic API key is required when using Anthropic provider")
        
        if self.proxy_enabled and not self.proxy_urls:
            issues.append("Proxy URLs are required when proxy is enabled")
        
        if self.proxy_enabled and self.proxy_username and not self.proxy_password:
            issues.append("Proxy password is required when proxy username is provided")
        
        # Check SSL configuration
        if self.ssl_cert_path and not self.ssl_key_path:
            issues.append("SSL key path is required when SSL cert path is provided")
        
        if self.ssl_key_path and not self.ssl_cert_path:
            issues.append("SSL cert path is required when SSL key path is provided")
        
        # Check JWT configuration
        if self.jwt_secret_key and self.jwt_algorithm.startswith('RS') and not self.ssl_key_path:
            issues.append("RSA private key is required for RSA JWT algorithms")
        
        # Check resource limits
        if self.max_model_memory_gb < 1:
            issues.append("Maximum model memory must be at least 1GB")
        
        if self.workers < 1:
            issues.append("Number of workers must be at least 1")
        
        if self.port < 1 or self.port > 65535:
            issues.append("Port must be between 1 and 65535")
        
        return issues
    
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