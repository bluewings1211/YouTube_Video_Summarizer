"""
Configuration validation utilities with detailed error messages and fix suggestions.
"""

import os
import re
from typing import Dict, List, Optional, Tuple, Any
from urllib.parse import urlparse
import logging

logger = logging.getLogger(__name__)


class ConfigurationValidator:
    """Advanced configuration validator with detailed error messages and fix suggestions."""
    
    def __init__(self, config):
        """Initialize validator with configuration object."""
        self.config = config
        self.issues = []
        self.warnings = []
        self.recommendations = []
    
    def validate_all(self) -> Tuple[List[str], List[str], List[str]]:
        """
        Validate all configuration aspects.
        
        Returns:
            Tuple of (issues, warnings, recommendations)
        """
        self.issues = []
        self.warnings = []
        self.recommendations = []
        
        # Run all validation checks
        self._validate_basic_config()
        self._validate_llm_config()
        self._validate_proxy_config()
        self._validate_security_config()
        self._validate_performance_config()
        self._validate_logging_config()
        self._validate_redis_config()
        self._validate_file_paths()
        self._validate_network_config()
        self._validate_environment_specific()
        
        return self.issues, self.warnings, self.recommendations
    
    def _validate_basic_config(self):
        """Validate basic application configuration."""
        # Environment validation
        if self.config.environment not in ['development', 'staging', 'production']:
            self.issues.append(
                f"Invalid environment '{self.config.environment}'. "
                f"Must be one of: development, staging, production. "
                f"Set ENVIRONMENT=development in your .env file."
            )
        
        # Port validation
        if self.config.port < 1 or self.config.port > 65535:
            self.issues.append(
                f"Invalid port {self.config.port}. Must be between 1 and 65535. "
                f"Set PORT=8000 in your .env file."
            )
        
        # Workers validation
        if self.config.workers < 1:
            self.issues.append(
                f"Invalid worker count {self.config.workers}. Must be at least 1. "
                f"Set WORKERS=4 in your .env file."
            )
        elif self.config.workers > 16:
            self.warnings.append(
                f"High worker count {self.config.workers} may cause resource issues. "
                f"Consider reducing to 4-8 workers."
            )
        
        # Host validation
        if self.config.host not in ['0.0.0.0', '127.0.0.1', 'localhost']:
            self.warnings.append(
                f"Unusual host configuration '{self.config.host}'. "
                f"Common values are 0.0.0.0, 127.0.0.1, or localhost."
            )
    
    def _validate_llm_config(self):
        """Validate LLM configuration."""
        # Provider validation
        if self.config.default_llm_provider not in ['openai', 'anthropic', 'ollama']:
            self.issues.append(
                f"Invalid LLM provider '{self.config.default_llm_provider}'. "
                f"Must be one of: openai, anthropic, ollama. "
                f"Set DEFAULT_LLM_PROVIDER=ollama in your .env file."
            )
        
        # API key validation
        if self.config.default_llm_provider == 'openai' and not self.config.openai_api_key:
            self.issues.append(
                "OpenAI API key is required when using OpenAI provider. "
                "Set OPENAI_API_KEY=your-api-key in your .env file or use the credential manager."
            )
        
        if self.config.default_llm_provider == 'anthropic' and not self.config.anthropic_api_key:
            self.issues.append(
                "Anthropic API key is required when using Anthropic provider. "
                "Set ANTHROPIC_API_KEY=your-api-key in your .env file or use the credential manager."
            )
        
        # Ollama configuration
        if self.config.default_llm_provider == 'ollama':
            if not self.config.ollama_host.startswith(('http://', 'https://')):
                self.issues.append(
                    f"Invalid Ollama host '{self.config.ollama_host}'. "
                    f"Must start with http:// or https://. "
                    f"Set OLLAMA_HOST=http://localhost:11434 in your .env file."
                )
            
            # Check if Ollama is accessible (optional check)
            self.recommendations.append(
                f"Ensure Ollama is running on {self.config.ollama_host}. "
                f"Run 'ollama serve' to start the Ollama server."
            )
        
        # Model validation
        if self.config.max_tokens < 1:
            self.issues.append(
                f"Invalid max tokens {self.config.max_tokens}. Must be at least 1. "
                f"Set MAX_TOKENS=4000 in your .env file."
            )
        elif self.config.max_tokens > 128000:
            self.warnings.append(
                f"Very high max tokens {self.config.max_tokens} may cause memory issues. "
                f"Consider reducing to 4000-8000 tokens."
            )
        
        if self.config.temperature < 0.0 or self.config.temperature > 2.0:
            self.issues.append(
                f"Invalid temperature {self.config.temperature}. Must be between 0.0 and 2.0. "
                f"Set TEMPERATURE=0.7 in your .env file."
            )
        
        # Model selection strategy
        if self.config.model_selection_strategy not in ['auto', 'prefer_local', 'prefer_cloud', 'cloud_only']:
            self.issues.append(
                f"Invalid model selection strategy '{self.config.model_selection_strategy}'. "
                f"Must be one of: auto, prefer_local, prefer_cloud, cloud_only. "
                f"Set MODEL_SELECTION_STRATEGY=auto in your .env file."
            )
    
    def _validate_proxy_config(self):
        """Validate proxy configuration."""
        if not self.config.proxy_enabled:
            return
        
        # Proxy URLs validation
        if not self.config.proxy_urls:
            self.issues.append(
                "Proxy is enabled but no proxy URLs are configured. "
                "Set PROXY_URLS=http://proxy1:port,http://proxy2:port in your .env file."
            )
        
        # Validate each proxy URL
        for i, url in enumerate(self.config.proxy_urls):
            try:
                parsed = urlparse(url)
                if not parsed.scheme:
                    self.issues.append(
                        f"Proxy URL #{i+1} '{url}' is missing scheme. "
                        f"Use format: http://proxy:port or socks5://proxy:port"
                    )
                elif parsed.scheme not in ['http', 'https', 'socks4', 'socks5']:
                    self.issues.append(
                        f"Proxy URL #{i+1} '{url}' has invalid scheme '{parsed.scheme}'. "
                        f"Supported schemes: http, https, socks4, socks5"
                    )
                
                if not parsed.hostname:
                    self.issues.append(
                        f"Proxy URL #{i+1} '{url}' is missing hostname."
                    )
                
                if not parsed.port:
                    self.warnings.append(
                        f"Proxy URL #{i+1} '{url}' is missing port. "
                        f"Consider specifying explicit port."
                    )
                
            except Exception as e:
                self.issues.append(
                    f"Proxy URL #{i+1} '{url}' is malformed: {str(e)}"
                )
        
        # Authentication validation
        if self.config.proxy_username and not self.config.proxy_password:
            self.issues.append(
                "Proxy username is set but password is missing. "
                "Set PROXY_PASSWORD=your-password in your .env file or use the credential manager."
            )
        
        if self.config.proxy_password and not self.config.proxy_username:
            self.issues.append(
                "Proxy password is set but username is missing. "
                "Set PROXY_USERNAME=your-username in your .env file or use the credential manager."
            )
        
        # Proxy configuration validation
        if self.config.proxy_timeout < 1:
            self.issues.append(
                f"Invalid proxy timeout {self.config.proxy_timeout}. Must be at least 1 second. "
                f"Set PROXY_TIMEOUT=30 in your .env file."
            )
        
        if self.config.proxy_max_failures < 1:
            self.issues.append(
                f"Invalid proxy max failures {self.config.proxy_max_failures}. Must be at least 1. "
                f"Set PROXY_MAX_FAILURES=3 in your .env file."
            )
        
        if self.config.proxy_pool_size < 1:
            self.issues.append(
                f"Invalid proxy pool size {self.config.proxy_pool_size}. Must be at least 1. "
                f"Set PROXY_POOL_SIZE=10 in your .env file."
            )
    
    def _validate_security_config(self):
        """Validate security configuration."""
        # Secret key validation
        if self.config.is_production and self.config.secret_key == 'dev-secret-key-change-in-production':
            self.issues.append(
                "Default secret key is being used in production. "
                "Set SECRET_KEY=your-secure-secret-key in your .env file."
            )
        
        if len(self.config.secret_key) < 32:
            self.warnings.append(
                f"Secret key is short ({len(self.config.secret_key)} characters). "
                f"Consider using a longer key (32+ characters) for better security."
            )
        
        # SSL configuration
        if self.config.ssl_cert_path and not os.path.exists(self.config.ssl_cert_path):
            self.issues.append(
                f"SSL certificate file not found: {self.config.ssl_cert_path}. "
                f"Ensure the file exists or update SSL_CERT_PATH in your .env file."
            )
        
        if self.config.ssl_key_path and not os.path.exists(self.config.ssl_key_path):
            self.issues.append(
                f"SSL key file not found: {self.config.ssl_key_path}. "
                f"Ensure the file exists or update SSL_KEY_PATH in your .env file."
            )
        
        # JWT configuration
        if self.config.jwt_secret_key and len(self.config.jwt_secret_key) < 32:
            self.warnings.append(
                f"JWT secret key is short ({len(self.config.jwt_secret_key)} characters). "
                f"Consider using a longer key (32+ characters) for better security."
            )
        
        if self.config.jwt_expiration_hours < 1:
            self.issues.append(
                f"Invalid JWT expiration {self.config.jwt_expiration_hours} hours. Must be at least 1. "
                f"Set JWT_EXPIRATION_HOURS=24 in your .env file."
            )
        elif self.config.jwt_expiration_hours > 168:  # 7 days
            self.warnings.append(
                f"Long JWT expiration {self.config.jwt_expiration_hours} hours may pose security risks. "
                f"Consider reducing to 24-72 hours."
            )
        
        # CORS validation
        if self.config.is_production and 'localhost' in str(self.config.cors_origins):
            self.warnings.append(
                "CORS origins include localhost in production. "
                "Consider updating CORS_ORIGINS for production domains."
            )
    
    def _validate_performance_config(self):
        """Validate performance configuration."""
        # Timeout validation
        if self.config.request_timeout < 1:
            self.issues.append(
                f"Invalid request timeout {self.config.request_timeout}. Must be at least 1 second. "
                f"Set REQUEST_TIMEOUT=300 in your .env file."
            )
        
        if self.config.workflow_timeout < self.config.request_timeout:
            self.warnings.append(
                f"Workflow timeout ({self.config.workflow_timeout}s) is less than request timeout ({self.config.request_timeout}s). "
                f"Consider increasing WORKFLOW_TIMEOUT."
            )
        
        # Memory validation
        if self.config.max_model_memory_gb < 1:
            self.issues.append(
                f"Invalid max model memory {self.config.max_model_memory_gb}GB. Must be at least 1GB. "
                f"Set MAX_MODEL_MEMORY_GB=8 in your .env file."
            )
        elif self.config.max_model_memory_gb > 32:
            self.warnings.append(
                f"High max model memory {self.config.max_model_memory_gb}GB may cause system issues. "
                f"Ensure your system has sufficient RAM."
            )
        
        # Caching validation
        if not self.config.enable_caching:
            self.recommendations.append(
                "Caching is disabled. Enable caching for better performance: "
                "Set ENABLE_CACHING=true in your .env file."
            )
    
    def _validate_logging_config(self):
        """Validate logging configuration."""
        # Log level validation
        valid_levels = ['debug', 'info', 'warning', 'error', 'critical']
        if self.config.log_level not in valid_levels:
            self.issues.append(
                f"Invalid log level '{self.config.log_level}'. "
                f"Must be one of: {', '.join(valid_levels)}. "
                f"Set LOG_LEVEL=info in your .env file."
            )
        
        # Log file path validation
        log_dir = os.path.dirname(self.config.log_file_path)
        if log_dir and not os.path.exists(log_dir):
            self.warnings.append(
                f"Log directory does not exist: {log_dir}. "
                f"It will be created automatically, but ensure write permissions are available."
            )
        
        # Log format validation
        if self.config.log_format not in ['json', 'standard']:
            self.issues.append(
                f"Invalid log format '{self.config.log_format}'. "
                f"Must be 'json' or 'standard'. "
                f"Set LOG_FORMAT=json in your .env file."
            )
        
        # Production logging recommendations
        if self.config.is_production:
            if self.config.log_level == 'debug':
                self.recommendations.append(
                    "Debug logging is enabled in production. "
                    "Consider setting LOG_LEVEL=info for better performance."
                )
            
            if self.config.log_format != 'json':
                self.recommendations.append(
                    "Consider using JSON logging format in production for better log parsing: "
                    "Set LOG_FORMAT=json in your .env file."
                )
    
    def _validate_redis_config(self):
        """Validate Redis configuration."""
        # Redis connection validation
        if self.config.redis_port < 1 or self.config.redis_port > 65535:
            self.issues.append(
                f"Invalid Redis port {self.config.redis_port}. Must be between 1 and 65535. "
                f"Set REDIS_PORT=6379 in your .env file."
            )
        
        if self.config.redis_db < 0 or self.config.redis_db > 15:
            self.issues.append(
                f"Invalid Redis database {self.config.redis_db}. Must be between 0 and 15. "
                f"Set REDIS_DB=0 in your .env file."
            )
        
        # Redis TTL validation
        if self.config.redis_cache_ttl < 1:
            self.issues.append(
                f"Invalid Redis cache TTL {self.config.redis_cache_ttl}. Must be at least 1 second. "
                f"Set REDIS_CACHE_TTL=3600 in your .env file."
            )
        
        # Redis host validation
        if self.config.redis_host in ['localhost', '127.0.0.1'] and self.config.is_production:
            self.warnings.append(
                "Redis is configured for localhost in production. "
                "Consider using a dedicated Redis instance for production."
            )
    
    def _validate_file_paths(self):
        """Validate file paths and permissions."""
        # Check if log directory is writable
        log_dir = os.path.dirname(self.config.log_file_path)
        if log_dir and os.path.exists(log_dir):
            if not os.access(log_dir, os.W_OK):
                self.issues.append(
                    f"Log directory is not writable: {log_dir}. "
                    f"Check permissions for the log directory."
                )
        
        # Check SSL files if configured
        if self.config.ssl_cert_path:
            if not os.path.exists(self.config.ssl_cert_path):
                self.issues.append(
                    f"SSL certificate file not found: {self.config.ssl_cert_path}"
                )
            elif not os.access(self.config.ssl_cert_path, os.R_OK):
                self.issues.append(
                    f"SSL certificate file is not readable: {self.config.ssl_cert_path}"
                )
        
        if self.config.ssl_key_path:
            if not os.path.exists(self.config.ssl_key_path):
                self.issues.append(
                    f"SSL key file not found: {self.config.ssl_key_path}"
                )
            elif not os.access(self.config.ssl_key_path, os.R_OK):
                self.issues.append(
                    f"SSL key file is not readable: {self.config.ssl_key_path}"
                )
    
    def _validate_network_config(self):
        """Validate network-related configuration."""
        # Rate limiting validation
        if self.config.rate_limit_enabled:
            if self.config.rate_limit_min_interval < 1:
                self.issues.append(
                    f"Invalid rate limit interval {self.config.rate_limit_min_interval}. Must be at least 1 second. "
                    f"Set RATE_LIMIT_MIN_INTERVAL=10 in your .env file."
                )
            
            if self.config.rate_limit_max_requests_per_minute < 1:
                self.issues.append(
                    f"Invalid rate limit max requests {self.config.rate_limit_max_requests_per_minute}. Must be at least 1. "
                    f"Set RATE_LIMIT_MAX_REQUESTS_PER_MINUTE=6 in your .env file."
                )
        
        # API rate limit validation
        if self.config.api_rate_limit < 1:
            self.issues.append(
                f"Invalid API rate limit {self.config.api_rate_limit}. Must be at least 1. "
                f"Set API_RATE_LIMIT=100 in your .env file."
            )
    
    def _validate_environment_specific(self):
        """Validate environment-specific configurations."""
        if self.config.is_production:
            # Production-specific validations
            if self.config.debug:
                self.warnings.append(
                    "Debug mode is enabled in production. "
                    "Set DEBUG=false in your .env file for production."
                )
            
            if self.config.hot_reload:
                self.warnings.append(
                    "Hot reload is enabled in production. "
                    "Set HOT_RELOAD=false in your .env file for production."
                )
            
            if not self.config.enable_security_headers:
                self.warnings.append(
                    "Security headers are disabled in production. "
                    "Set ENABLE_SECURITY_HEADERS=true in your .env file for production."
                )
        
        elif self.config.is_development:
            # Development-specific recommendations
            if not self.config.debug:
                self.recommendations.append(
                    "Debug mode is disabled in development. "
                    "Consider enabling debug mode for better development experience: "
                    "Set DEBUG=true in your .env file."
                )
            
            if self.config.log_level in ['warning', 'error']:
                self.recommendations.append(
                    f"Log level is set to '{self.config.log_level}' in development. "
                    f"Consider using 'debug' or 'info' for better development visibility."
                )
    
    def generate_report(self) -> str:
        """Generate a comprehensive configuration validation report."""
        issues, warnings, recommendations = self.validate_all()
        
        report = []
        report.append("=" * 80)
        report.append("CONFIGURATION VALIDATION REPORT")
        report.append("=" * 80)
        report.append(f"Environment: {self.config.environment}")
        report.append(f"LLM Provider: {self.config.default_llm_provider}")
        report.append(f"Proxy Enabled: {self.config.proxy_enabled}")
        report.append("")
        
        if issues:
            report.append("CRITICAL ISSUES (must be fixed):")
            report.append("-" * 40)
            for i, issue in enumerate(issues, 1):
                report.append(f"{i}. {issue}")
            report.append("")
        
        if warnings:
            report.append("WARNINGS (should be reviewed):")
            report.append("-" * 40)
            for i, warning in enumerate(warnings, 1):
                report.append(f"{i}. {warning}")
            report.append("")
        
        if recommendations:
            report.append("RECOMMENDATIONS (for optimization):")
            report.append("-" * 40)
            for i, recommendation in enumerate(recommendations, 1):
                report.append(f"{i}. {recommendation}")
            report.append("")
        
        if not issues and not warnings:
            report.append("✅ Configuration validation passed!")
            report.append("No critical issues or warnings found.")
        
        report.append("=" * 80)
        
        return "\n".join(report)


def validate_configuration(config) -> Tuple[List[str], List[str], List[str]]:
    """
    Validate configuration and return issues, warnings, and recommendations.
    
    Args:
        config: Configuration object to validate
        
    Returns:
        Tuple of (issues, warnings, recommendations)
    """
    validator = ConfigurationValidator(config)
    return validator.validate_all()


def generate_config_report(config) -> str:
    """
    Generate a comprehensive configuration validation report.
    
    Args:
        config: Configuration object to validate
        
    Returns:
        Formatted validation report
    """
    validator = ConfigurationValidator(config)
    return validator.generate_report()