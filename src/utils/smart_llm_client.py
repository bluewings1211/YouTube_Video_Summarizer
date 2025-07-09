"""
Smart LLM client with automatic model selection and fallback logic.

This module provides intelligent model selection between local Ollama models
and cloud-based models (OpenAI/Anthropic) based on availability, task requirements,
and user configuration preferences.
"""

import logging
import os
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum

from .call_llm import (
    LLMClient, LLMConfig, LLMProvider, LLMError, LLMClientError,
    create_llm_client, check_ollama_availability, get_recommended_ollama_model,
    detect_provider_from_model, OllamaConnectionError, OllamaModelNotFoundError
)
from .language_detector import (
    detect_language, is_chinese_text, ensure_chinese_encoding, 
    optimize_chinese_content_for_llm
)

# Configure logging
logger = logging.getLogger(__name__)


class ModelSelectionStrategy(Enum):
    """Model selection strategies."""
    AUTO = "auto"                # Automatically choose best available model
    PREFER_LOCAL = "prefer_local"    # Prefer local Ollama models
    PREFER_CLOUD = "prefer_cloud"    # Prefer cloud models
    CLOUD_ONLY = "cloud_only"       # Only use cloud models


@dataclass
class TaskRequirements:
    """Requirements for a specific task."""
    task_type: str              # "summarization", "keywords", "timestamps"
    language: Optional[str] = None
    text_length: Optional[int] = None
    quality_level: str = "medium"    # "low", "medium", "high"
    speed_priority: bool = False


class SmartLLMClient:
    """
    Smart LLM client that automatically selects the best model based on task requirements,
    availability, and configuration preferences.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize SmartLLMClient with configuration.
        
        Args:
            config: Configuration dictionary with LLM settings
        """
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(f"{__name__}.SmartLLMClient")
        
        # Cache for model availability status
        self._ollama_status = None
        self._model_cache = {}
        
        # Initialize based on strategy
        self.strategy = ModelSelectionStrategy(self.config.get('model_selection_strategy', 'auto'))
        self.logger.info(f"Initialized SmartLLMClient with strategy: {self.strategy.value}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration from environment and config module."""
        try:
            from ..config import settings
            return settings.llm_config
        except ImportError:
            # Fallback configuration if config module not available
            return {
                'default_provider': os.getenv('DEFAULT_LLM_PROVIDER', 'ollama'),
                'ollama_model': os.getenv('OLLAMA_MODEL', 'llama3.1:8b'),
                'openai_model': os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo'),
                'anthropic_model': os.getenv('ANTHROPIC_MODEL', 'claude-3-sonnet-20240229'),
                'model_selection_strategy': os.getenv('MODEL_SELECTION_STRATEGY', 'auto'),
                'chinese_language_model': os.getenv('CHINESE_LANGUAGE_MODEL', 'qwen2.5:7b'),
                'performance_model': os.getenv('PERFORMANCE_MODEL', 'mistral:7b'),
                'lightweight_model': os.getenv('LIGHTWEIGHT_MODEL', 'llama3.2:3b'),
                'ollama': {
                    'host': os.getenv('OLLAMA_HOST', 'http://localhost:11434'),
                    'keep_alive': os.getenv('OLLAMA_KEEP_ALIVE', '5m'),
                    'fallback_enabled': os.getenv('OLLAMA_FALLBACK_ENABLED', 'true').lower() == 'true',
                    'fallback_provider': os.getenv('OLLAMA_FALLBACK_PROVIDER', 'openai')
                },
                'max_tokens': int(os.getenv('MAX_TOKENS', '1000')),
                'temperature': float(os.getenv('TEMPERATURE', '0.7'))
            }
    
    def get_ollama_status(self, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Get Ollama server status with caching.
        
        Args:
            force_refresh: Force refresh of cached status
            
        Returns:
            Dictionary containing Ollama availability status
        """
        if self._ollama_status is None or force_refresh:
            ollama_config = self.config.get('ollama', {})
            host = ollama_config.get('host', 'http://localhost:11434')
            self._ollama_status = check_ollama_availability(host)
            
            if self._ollama_status['available']:
                self.logger.info(f"Ollama available with {self._ollama_status['model_count']} models")
            else:
                self.logger.warning(f"Ollama unavailable: {self._ollama_status['reason']}")
        
        return self._ollama_status
    
    def select_optimal_model(self, task_requirements: TaskRequirements) -> Tuple[str, str, Dict[str, Any]]:
        """
        Select the optimal model and provider for the given task requirements.
        
        Args:
            task_requirements: Task requirements for model selection
            
        Returns:
            Tuple of (provider, model, model_config)
        """
        self.logger.debug(f"Selecting model for task: {task_requirements.task_type}")
        
        # Handle cloud-only strategy
        if self.strategy == ModelSelectionStrategy.CLOUD_ONLY:
            return self._select_cloud_model(task_requirements)
        
        # Check Ollama availability for local models
        ollama_status = self.get_ollama_status()
        
        # Handle different strategies
        if self.strategy == ModelSelectionStrategy.PREFER_CLOUD:
            # Try cloud first, fallback to local
            try:
                return self._select_cloud_model(task_requirements)
            except Exception as e:
                self.logger.warning(f"Cloud model selection failed: {str(e)}")
                if ollama_status['available']:
                    return self._select_ollama_model(task_requirements, ollama_status)
                raise
        
        elif self.strategy == ModelSelectionStrategy.PREFER_LOCAL:
            # Try local first, fallback to cloud
            if ollama_status['available']:
                try:
                    return self._select_ollama_model(task_requirements, ollama_status)
                except Exception as e:
                    self.logger.warning(f"Local model selection failed: {str(e)}")
                    return self._select_cloud_model(task_requirements)
            else:
                return self._select_cloud_model(task_requirements)
        
        else:  # AUTO strategy
            return self._select_auto_model(task_requirements, ollama_status)
    
    def _select_auto_model(self, task_requirements: TaskRequirements, ollama_status: Dict[str, Any]) -> Tuple[str, str, Dict[str, Any]]:
        """
        Automatically select the best model based on task requirements and availability.
        
        Args:
            task_requirements: Task requirements
            ollama_status: Ollama availability status
            
        Returns:
            Tuple of (provider, model, model_config)
        """
        # Decision matrix for AUTO selection
        use_local = False
        
        # Prefer local for speed-priority tasks
        if task_requirements.speed_priority and ollama_status['available']:
            use_local = True
        
        # Use specialized models for specific languages
        if task_requirements.language and task_requirements.language.startswith('zh'):
            # Chinese content - prefer qwen models if available
            chinese_model = self.config.get('chinese_language_model', 'qwen2.5:7b')
            if ollama_status['available'] and any(chinese_model in model for model in ollama_status['models']):
                use_local = True
        
        # For high-quality requirements, prefer cloud unless explicitly using performance local models
        if task_requirements.quality_level == "high":
            performance_model = self.config.get('performance_model', 'mistral:7b')
            if not (ollama_status['available'] and any(performance_model in model for model in ollama_status['models'])):
                use_local = False
        
        # For low-resource environments, prefer lightweight local models
        if task_requirements.quality_level == "low" and ollama_status['available']:
            lightweight_model = self.config.get('lightweight_model', 'llama3.2:3b')
            if any(lightweight_model in model for model in ollama_status['models']):
                use_local = True
        
        # Default to local if available, cloud otherwise
        if ollama_status['available'] and use_local:
            return self._select_ollama_model(task_requirements, ollama_status)
        else:
            # Fall back to cloud, but still try local if cloud fails
            try:
                return self._select_cloud_model(task_requirements)
            except Exception as e:
                if ollama_status['available']:
                    self.logger.warning(f"Cloud fallback to local due to: {str(e)}")
                    return self._select_ollama_model(task_requirements, ollama_status)
                raise
    
    def _select_ollama_model(self, task_requirements: TaskRequirements, ollama_status: Dict[str, Any]) -> Tuple[str, str, Dict[str, Any]]:
        """
        Select the best Ollama model for the task requirements.
        
        Args:
            task_requirements: Task requirements
            ollama_status: Ollama availability status
            
        Returns:
            Tuple of (provider, model, model_config)
        """
        available_models = ollama_status['models']
        
        # Language-specific model selection
        if task_requirements.language and task_requirements.language.startswith('zh'):
            chinese_model = self.config.get('chinese_language_model', 'qwen2.5:7b')
            if any(chinese_model in model for model in available_models):
                selected_model = chinese_model
            else:
                # Fallback to best available model for multilingual support
                selected_model = self._find_best_available_model(available_models, task_requirements)
        else:
            # Quality-based selection
            if task_requirements.quality_level == "high":
                preferred_model = self.config.get('performance_model', 'mistral:7b')
            elif task_requirements.quality_level == "low":
                preferred_model = self.config.get('lightweight_model', 'llama3.2:3b')
            else:
                preferred_model = self.config.get('ollama_model', 'llama3.1:8b')
            
            # Find exact or partial match
            selected_model = None
            for model in available_models:
                if preferred_model == model or preferred_model in model:
                    selected_model = preferred_model
                    break
            
            if not selected_model:
                selected_model = self._find_best_available_model(available_models, task_requirements)
        
        # Build configuration
        ollama_config = self.config.get('ollama', {})
        model_config = {
            'host': ollama_config.get('host', 'http://localhost:11434'),
            'keep_alive': ollama_config.get('keep_alive', '5m'),
            'max_tokens': self.config.get('max_tokens', 1000),
            'temperature': self.config.get('temperature', 0.7)
        }
        
        self.logger.info(f"Selected Ollama model: {selected_model}")
        return 'ollama', selected_model, model_config
    
    def _select_cloud_model(self, task_requirements: TaskRequirements) -> Tuple[str, str, Dict[str, Any]]:
        """
        Select the best cloud model for the task requirements.
        
        Args:
            task_requirements: Task requirements
            
        Returns:
            Tuple of (provider, model, model_config)
        """
        # Choose provider based on default configuration and availability
        default_provider = self.config.get('default_provider', 'openai')
        
        # For high-quality tasks, prefer specific providers
        if task_requirements.quality_level == "high":
            if task_requirements.task_type == "summarization":
                provider = 'anthropic'  # Claude is often better for summarization
                model = self.config.get('anthropic_model', 'claude-3-sonnet-20240229')
            else:
                provider = default_provider
                model = self._get_default_model_for_provider(provider)
        else:
            provider = default_provider
            model = self._get_default_model_for_provider(provider)
        
        # Build configuration
        model_config = {
            'max_tokens': self.config.get('max_tokens', 1000),
            'temperature': self.config.get('temperature', 0.7)
        }
        
        self.logger.info(f"Selected cloud model: {provider}/{model}")
        return provider, model, model_config
    
    def _get_default_model_for_provider(self, provider: str) -> str:
        """Get default model for a provider."""
        if provider == 'openai':
            return self.config.get('openai_model', 'gpt-3.5-turbo')
        elif provider == 'anthropic':
            return self.config.get('anthropic_model', 'claude-3-sonnet-20240229')
        else:
            return self.config.get('ollama_model', 'llama3.1:8b')
    
    def _find_best_available_model(self, available_models: List[str], task_requirements: TaskRequirements) -> str:
        """
        Find the best available model from the list.
        
        Args:
            available_models: List of available models
            task_requirements: Task requirements
            
        Returns:
            Best available model name
        """
        # Priority order based on task requirements
        if task_requirements.quality_level == "high":
            priority_order = ['mistral', 'llama3.3', 'llama3.1', 'qwen2.5', 'llama3.2']
        elif task_requirements.quality_level == "low":
            priority_order = ['llama3.2', 'llama3.1', 'mistral', 'qwen2.5']
        else:
            priority_order = ['llama3.1', 'mistral', 'qwen2.5', 'llama3.2', 'llama3.3']
        
        # Find first available model in priority order
        for priority_model in priority_order:
            for model in available_models:
                if priority_model in model.lower():
                    return model
        
        # Fallback to first available model
        return available_models[0] if available_models else 'llama3.1:8b'
    
    def create_client(self, task_requirements: TaskRequirements) -> LLMClient:
        """
        Create an LLM client optimized for the given task requirements.
        
        Args:
            task_requirements: Task requirements for model selection
            
        Returns:
            Configured LLMClient instance
        """
        provider, model, model_config = self.select_optimal_model(task_requirements)
        
        # Create LLM client with fallback handling
        try:
            if provider == 'ollama':
                return create_llm_client(
                    provider=provider,
                    model=model,
                    max_tokens=model_config.get('max_tokens', 1000),
                    temperature=model_config.get('temperature', 0.7),
                    ollama_host=model_config.get('host', 'http://localhost:11434'),
                    ollama_keep_alive=model_config.get('keep_alive', '5m')
                )
            else:
                return create_llm_client(
                    provider=provider,
                    model=model,
                    max_tokens=model_config.get('max_tokens', 1000),
                    temperature=model_config.get('temperature', 0.7)
                )
                
        except (OllamaConnectionError, OllamaModelNotFoundError, LLMClientError) as e:
            # Handle Ollama-specific errors with fallback
            if provider == 'ollama' and self.config.get('ollama', {}).get('fallback_enabled', True):
                fallback_provider = self.config.get('ollama', {}).get('fallback_provider', 'openai')
                self.logger.warning(f"Ollama failed, falling back to {fallback_provider}: {str(e)}")
                return self._create_fallback_client(task_requirements, fallback_provider)
            else:
                raise
    
    def check_provider_health(self, provider: str) -> Dict[str, Any]:
        """
        Check the health status of a specific provider.
        
        Args:
            provider: Provider to check ('ollama', 'openai', 'anthropic')
            
        Returns:
            Dictionary with health status information
        """
        health_status = {
            'provider': provider,
            'available': False,
            'error': None,
            'models': [],
            'response_time': None
        }
        
        try:
            import time
            start_time = time.time()
            
            if provider == 'ollama':
                ollama_status = self.get_ollama_status(force_refresh=True)
                health_status['available'] = ollama_status['available']
                health_status['models'] = ollama_status.get('models', [])
                if not ollama_status['available']:
                    health_status['error'] = ollama_status.get('reason', 'Unknown error')
            
            elif provider in ['openai', 'anthropic']:
                # Test with a minimal request
                test_requirements = TaskRequirements(task_type="test", quality_level="low")
                try:
                    test_client = create_llm_client(
                        provider=provider,
                        model=self._get_default_model_for_provider(provider),
                        max_tokens=1,
                        temperature=0.1
                    )
                    # Don't actually generate, just check if client initializes
                    health_status['available'] = True
                    health_status['models'] = [self._get_default_model_for_provider(provider)]
                except Exception as e:
                    health_status['error'] = str(e)
            
            health_status['response_time'] = round((time.time() - start_time) * 1000, 2)
            
        except Exception as e:
            health_status['error'] = str(e)
        
        return health_status
    
    def get_provider_fallback_chain(self, primary_provider: str) -> List[str]:
        """
        Get the fallback chain for a given primary provider.
        
        Args:
            primary_provider: Primary provider that failed
            
        Returns:
            List of providers to try in order
        """
        all_providers = ['ollama', 'openai', 'anthropic']
        
        # Remove the failed primary provider
        fallback_providers = [p for p in all_providers if p != primary_provider]
        
        # Check availability and sort by preference
        available_providers = []
        for provider in fallback_providers:
            try:
                health = self.check_provider_health(provider)
                if health['available']:
                    available_providers.append((provider, health['response_time'] or 999))
            except:
                continue
        
        # Sort by response time (faster providers first)
        available_providers.sort(key=lambda x: x[1])
        
        return [provider for provider, _ in available_providers]
    
    def _create_fallback_client(self, task_requirements: TaskRequirements, fallback_provider: str) -> LLMClient:
        """
        Create a fallback client when primary selection fails.
        
        Args:
            task_requirements: Task requirements
            fallback_provider: Fallback provider to use
            
        Returns:
            Configured fallback LLMClient
        """
        fallback_model = self._get_default_model_for_provider(fallback_provider)
        
        return create_llm_client(
            provider=fallback_provider,
            model=fallback_model,
            max_tokens=self.config.get('max_tokens', 1000),
            temperature=self.config.get('temperature', 0.7)
        )
    
    def generate_text_with_chinese_optimization(
        self,
        text: str,
        task_requirements: TaskRequirements,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Generate text with Chinese language optimization.
        
        Args:
            text: Input text content (may be Chinese)
            task_requirements: Task requirements for model selection
            max_tokens: Override max tokens
            temperature: Override temperature
            
        Returns:
            Generated text response with metadata
        """
        # Check if content is Chinese and optimize accordingly
        if is_chinese_text(text):
            self.logger.info("Detected Chinese content, applying optimization")
            
            # Ensure proper encoding
            optimized_text = ensure_chinese_encoding(text)
            
            # Get Chinese-optimized prompts
            chinese_prompts = optimize_chinese_content_for_llm(
                optimized_text, 
                task_requirements.task_type
            )
            
            # Use optimized prompts
            return self.generate_text_with_fallback(
                prompt=chinese_prompts['user_prompt'],
                task_requirements=task_requirements,
                system_prompt=chinese_prompts['system_prompt'],
                max_tokens=max_tokens,
                temperature=temperature
            )
        else:
            # Regular processing for non-Chinese content
            return self.generate_text_with_fallback(
                prompt=text,
                task_requirements=task_requirements,
                system_prompt=None,
                max_tokens=max_tokens,
                temperature=temperature
            )

    def generate_text_with_fallback(
        self,
        prompt: str,
        task_requirements: TaskRequirements,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Generate text with intelligent model selection and fallback.
        
        Args:
            prompt: Text prompt for generation
            task_requirements: Task requirements for model selection
            system_prompt: Optional system prompt
            max_tokens: Override max tokens
            temperature: Override temperature
            
        Returns:
            Generated text response with metadata
        """
        client = self.create_client(task_requirements)
        
        try:
            return client.generate_text(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )
        except Exception as e:
            # Log the error and attempt fallback if enabled
            self.logger.error(f"Text generation failed with {client.config.provider.value}: {str(e)}")
            
            # If this was already a fallback attempt, don't try again
            if hasattr(client.config, 'is_fallback'):
                raise
            
            # Try fallback if enabled
            if self.config.get('ollama', {}).get('fallback_enabled', True):
                # Get intelligent fallback chain
                fallback_chain = self.get_provider_fallback_chain(client.config.provider.value)
                
                if not fallback_chain:
                    self.logger.error("No available fallback providers")
                    raise
                
                # Try each provider in the fallback chain
                last_error = e
                for fallback_provider in fallback_chain:
                    try:
                        self.logger.info(f"Attempting fallback to {fallback_provider}")
                        
                        fallback_client = self._create_fallback_client(task_requirements, fallback_provider)
                        fallback_client.config.is_fallback = True  # Mark as fallback to prevent infinite recursion
                        
                        result = fallback_client.generate_text(
                            prompt=prompt,
                            system_prompt=system_prompt,
                            max_tokens=max_tokens,
                            temperature=temperature
                        )
                        
                        # Add fallback metadata to the response
                        result['_fallback_used'] = {
                            'original_provider': client.config.provider.value,
                            'fallback_provider': fallback_provider,
                            'original_error': str(e)
                        }
                        
                        self.logger.info(f"Fallback to {fallback_provider} successful")
                        return result
                        
                    except Exception as fallback_error:
                        self.logger.warning(f"Fallback to {fallback_provider} failed: {str(fallback_error)}")
                        last_error = fallback_error
                        continue
                
                # All fallbacks failed
                self.logger.error("All fallback providers failed")
                raise last_error
            else:
                raise


# Convenience functions for common use cases
def create_smart_client(config: Optional[Dict[str, Any]] = None) -> SmartLLMClient:
    """Create a SmartLLMClient with optional configuration."""
    return SmartLLMClient(config)


def detect_task_requirements(
    text: str = "",
    task_type: str = "summarization",
    speed_priority: bool = False,
    quality_level: str = "medium"
) -> TaskRequirements:
    """
    Detect task requirements from input text and parameters.
    
    Args:
        text: Input text to analyze
        task_type: Type of task to perform
        speed_priority: Whether speed is prioritized
        quality_level: Required quality level
        
    Returns:
        TaskRequirements object
    """
    # Detect language if text is provided
    language = None
    if text:
        try:
            language = detect_language(text)
        except Exception:
            language = 'en'  # Default to English
    
    # Determine text length category
    text_length = len(text.split()) if text else 0
    
    return TaskRequirements(
        task_type=task_type,
        language=language,
        text_length=text_length,
        quality_level=quality_level,
        speed_priority=speed_priority
    )