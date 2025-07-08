"""
LLM client utilities for AI-powered content generation.

This module provides a unified interface for calling different LLM providers
including OpenAI and Anthropic, with support for different model types and
configuration options.
"""

import logging
import os
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

try:
    import openai
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"


@dataclass
class LLMConfig:
    """Configuration for LLM client."""
    provider: LLMProvider
    model: str
    api_key: Optional[str] = None
    max_tokens: int = 1000
    temperature: float = 0.7
    timeout: int = 30
    # Ollama-specific settings
    ollama_host: str = "http://localhost:11434"
    ollama_keep_alive: str = "5m"


class LLMError(Exception):
    """Base exception for LLM operations."""
    
    def __init__(self, message: str, provider: str = "", error_code: Optional[str] = None, 
                 retry_after: Optional[int] = None):
        super().__init__(message)
        self.message = message
        self.provider = provider
        self.error_code = error_code
        self.retry_after = retry_after
        self.timestamp = datetime.utcnow().isoformat()


class LLMAuthenticationError(LLMError):
    """Authentication errors (invalid API key, etc.)."""
    
    def __init__(self, provider: str = ""):
        message = f"Authentication failed for {provider}. Check your API key."
        super().__init__(message, provider, "AUTH_ERROR")


class LLMRateLimitError(LLMError):
    """Rate limit exceeded errors."""
    
    def __init__(self, provider: str = "", retry_after: int = 60):
        message = f"Rate limit exceeded for {provider}. Retry after {retry_after} seconds."
        super().__init__(message, provider, "RATE_LIMIT", retry_after)


class LLMQuotaError(LLMError):
    """Quota/usage limit exceeded errors."""
    
    def __init__(self, provider: str = ""):
        message = f"Usage quota exceeded for {provider}. Check your billing and limits."
        super().__init__(message, provider, "QUOTA_EXCEEDED")


class LLMModelError(LLMError):
    """Model-specific errors (unavailable, deprecated, etc.)."""
    
    def __init__(self, model: str = "", provider: str = ""):
        message = f"Model {model} is unavailable or invalid for {provider}"
        super().__init__(message, provider, "MODEL_ERROR")


class LLMContentError(LLMError):
    """Content-related errors (filtered, too long, etc.)."""
    
    def __init__(self, reason: str = "Content filtered", provider: str = ""):
        message = f"Content error: {reason}"
        super().__init__(message, provider, "CONTENT_ERROR")


class LLMTimeoutError(LLMError):
    """Timeout errors."""
    
    def __init__(self, timeout: int = 30, provider: str = ""):
        message = f"Request to {provider} timed out after {timeout} seconds"
        super().__init__(message, provider, "TIMEOUT", retry_after=10)


class LLMServerError(LLMError):
    """Server-side LLM errors."""
    
    def __init__(self, status_code: int = 500, provider: str = ""):
        message = f"Server error from {provider} (HTTP {status_code})"
        super().__init__(message, provider, "SERVER_ERROR", retry_after=30)


class LLMClientError(LLMError):
    """Client-side LLM errors (bad request, etc.)."""
    
    def __init__(self, message: str = "Invalid request", provider: str = ""):
        super().__init__(f"Client error: {message}", provider, "CLIENT_ERROR")


class OllamaConnectionError(LLMError):
    """Ollama server connection errors."""
    
    def __init__(self, host: str = "localhost:11434"):
        message = f"Failed to connect to Ollama server at {host}. Ensure Ollama is running."
        super().__init__(message, "ollama", "CONNECTION_ERROR")


class OllamaModelNotFoundError(LLMError):
    """Ollama model not found errors."""
    
    def __init__(self, model: str = ""):
        message = f"Model '{model}' not found in Ollama. Run 'ollama pull {model}' to download it."
        super().__init__(message, "ollama", "MODEL_NOT_FOUND")


class OllamaInsufficientResourcesError(LLMError):
    """Ollama insufficient resources errors."""
    
    def __init__(self, model: str = ""):
        message = f"Insufficient resources to run model '{model}'. Consider using a smaller model."
        super().__init__(message, "ollama", "INSUFFICIENT_RESOURCES")


class LLMClient:
    """
    Unified LLM client for calling different AI providers.
    
    Supports OpenAI and Anthropic APIs with consistent interface.
    """
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{config.provider.value}")
        
        # Initialize provider-specific client
        if config.provider == LLMProvider.OPENAI:
            self._init_openai_client()
        elif config.provider == LLMProvider.ANTHROPIC:
            self._init_anthropic_client()
        elif config.provider == LLMProvider.OLLAMA:
            self._init_ollama_client()
        else:
            raise LLMClientError(f"Unsupported provider: {config.provider}")
    
    def _init_openai_client(self):
        """Initialize OpenAI client."""
        if not OPENAI_AVAILABLE:
            raise LLMClientError("OpenAI library not installed. Install with: pip install openai")
        
        api_key = self.config.api_key or os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise LLMClientError("OpenAI API key not provided")
        
        self.client = OpenAI(api_key=api_key)
        self.logger.info(f"OpenAI client initialized with model: {self.config.model}")
    
    def _init_anthropic_client(self):
        """Initialize Anthropic client."""
        if not ANTHROPIC_AVAILABLE:
            raise LLMClientError("Anthropic library not installed. Install with: pip install anthropic")
        
        api_key = self.config.api_key or os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            raise LLMClientError("Anthropic API key not provided")
        
        self.client = Anthropic(api_key=api_key)
        self.logger.info(f"Anthropic client initialized with model: {self.config.model}")
    
    def _init_ollama_client(self):
        """Initialize Ollama client."""
        if not OLLAMA_AVAILABLE:
            raise LLMClientError("Ollama library not installed. Install with: pip install ollama")

        # Set up Ollama client with host configuration
        self.client = ollama.Client(host=self.config.ollama_host)

        # Test connection and verify model availability
        try:
            # Check if Ollama server is running
            models = self.client.list()
            self.logger.debug(f"Ollama models response: {models}")

            available_models = []
            model_list = models.get('models', [])
            for model_info in model_list:
                name = None
                if isinstance(model_info, dict):
                    name = model_info.get('name')
                else:
                    # Try to get name from model attribute, fallback to model
                    name = getattr(model_info, 'name', None)
                    if not name:
                        name = getattr(model_info, 'model', None)
                
                if name:
                    available_models.append(name)
                else:
                    self.logger.debug(f"Skipping model entry without name: {model_info}")

            # Check if the requested model is available
            if self.config.model not in available_models:
                # Try to find partial matches (e.g., llama3.1:8b matches llama3.1)
                model_found = any(self.config.model in model for model in available_models)
                if not model_found:
                    self.logger.warning(
                        f"Model '{self.config.model}' not found. Available models: {available_models}"
                    )
                    raise OllamaModelNotFoundError(self.config.model)

            self.logger.info(f"Ollama client initialized with model: {self.config.model} at {self.config.ollama_host}")

        except Exception as e:
            if "connection" in str(e).lower() or "refused" in str(e).lower():
                raise OllamaConnectionError(self.config.ollama_host)
            elif "not found" in str(e).lower():
                raise OllamaModelNotFoundError(self.config.model)
            else:
                self.logger.error(f"Full error when initializing Ollama client: {e}", exc_info=True)
                raise LLMClientError(f"Failed to initialize Ollama client: {str(e)}")
    
    def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Generate text using the configured LLM.
        
        Args:
            prompt: The main prompt for text generation
            system_prompt: Optional system prompt for context
            max_tokens: Override default max tokens
            temperature: Override default temperature
            
        Returns:
            Dictionary containing generated text and metadata
            
        Raises:
            LLMError: If text generation fails
        """
        start_time = datetime.utcnow()
        
        try:
            # Use provided values or fall back to config
            max_tokens = max_tokens or self.config.max_tokens
            temperature = temperature or self.config.temperature
            
            if self.config.provider == LLMProvider.OPENAI:
                result = self._generate_openai(prompt, system_prompt, max_tokens, temperature)
            elif self.config.provider == LLMProvider.ANTHROPIC:
                result = self._generate_anthropic(prompt, system_prompt, max_tokens, temperature)
            elif self.config.provider == LLMProvider.OLLAMA:
                result = self._generate_ollama(prompt, system_prompt, max_tokens, temperature)
            else:
                raise LLMClientError(f"Unsupported provider: {self.config.provider}")
            
            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()
            
            return {
                'text': result['text'],
                'usage': result.get('usage', {}),
                'model': self.config.model,
                'provider': self.config.provider.value,
                'duration_seconds': duration,
                'timestamp': end_time.isoformat(),
                'success': True
            }
            
        except Exception as e:
            self.logger.error(f"Text generation failed: {str(e)}")
            
            # Categorize error for better fallback handling
            error_str = str(e).lower()
            if "rate limit" in error_str:
                raise LLMRateLimitError(self.config.provider.value, retry_after=60)
            elif "authentication" in error_str or "api key" in error_str:
                raise LLMAuthenticationError(self.config.provider.value)
            elif "timeout" in error_str:
                raise LLMTimeoutError(self.config.timeout, self.config.provider.value)
            elif "quota" in error_str or "usage" in error_str:
                raise LLMQuotaError(self.config.provider.value)
            elif "model" in error_str and ("not found" in error_str or "unavailable" in error_str):
                raise LLMModelError(self.config.model, self.config.provider.value)
            elif "content" in error_str and ("filtered" in error_str or "policy" in error_str):
                raise LLMContentError("Content filtered by provider", self.config.provider.value)
            elif "server" in error_str or "500" in error_str or "502" in error_str or "503" in error_str:
                raise LLMServerError(500, self.config.provider.value)
            else:
                raise LLMError(f"LLM generation failed: {str(e)}", self.config.provider.value)
    
    def _generate_openai(
        self,
        prompt: str,
        system_prompt: Optional[str],
        max_tokens: int,
        temperature: float
    ) -> Dict[str, Any]:
        """Generate text using OpenAI API."""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=self.config.timeout
            )
            
            return {
                'text': response.choices[0].message.content,
                'usage': {
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens,
                    'total_tokens': response.usage.total_tokens
                }
            }
            
        except Exception as e:
            self.logger.error(f"OpenAI API call failed: {str(e)}")
            raise
    
    def _generate_anthropic(
        self,
        prompt: str,
        system_prompt: Optional[str],
        max_tokens: int,
        temperature: float
    ) -> Dict[str, Any]:
        """Generate text using Anthropic API."""
        try:
            kwargs = {
                'model': self.config.model,
                'messages': [{"role": "user", "content": prompt}],
                'max_tokens': max_tokens,
                'temperature': temperature
            }
            
            if system_prompt:
                kwargs['system'] = system_prompt
            
            response = self.client.messages.create(**kwargs)
            
            return {
                'text': response.content[0].text,
                'usage': {
                    'prompt_tokens': response.usage.input_tokens,
                    'completion_tokens': response.usage.output_tokens,
                    'total_tokens': response.usage.input_tokens + response.usage.output_tokens
                }
            }
            
        except Exception as e:
            self.logger.error(f"Anthropic API call failed: {str(e)}")
            raise
    
    def _generate_ollama(
        self,
        prompt: str,
        system_prompt: Optional[str],
        max_tokens: int,
        temperature: float
    ) -> Dict[str, Any]:
        """Generate text using Ollama API."""
        try:
            # Build the message structure for Ollama
            messages = []
            
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            messages.append({"role": "user", "content": prompt})
            
            # Prepare options for Ollama
            options = {
                "temperature": temperature,
                "num_predict": max_tokens,  # Ollama uses num_predict instead of max_tokens
            }
            
            # Make the API call to Ollama
            response = self.client.chat(
                model=self.config.model,
                messages=messages,
                options=options,
                keep_alive=self.config.ollama_keep_alive
            )
            
            # Extract the response text
            response_text = response.get('message', {}).get('content', '')
            
            if not response_text:
                raise ValueError("Empty response from Ollama")
            
            # Estimate token usage (Ollama doesn't provide exact counts)
            prompt_tokens = len(' '.join([msg['content'] for msg in messages]).split()) * 1.3  # Rough estimation
            completion_tokens = len(response_text.split()) * 1.3
            
            return {
                'text': response_text,
                'usage': {
                    'prompt_tokens': int(prompt_tokens),
                    'completion_tokens': int(completion_tokens),
                    'total_tokens': int(prompt_tokens + completion_tokens)
                },
                # Additional Ollama-specific metadata (for debugging/optimization)
                '_ollama_metadata': {
                    'model': response.get('model', self.config.model),
                    'created_at': response.get('created_at', ''),
                    'done': response.get('done', True),
                    'total_duration': response.get('total_duration', 0),
                    'load_duration': response.get('load_duration', 0),
                    'prompt_eval_count': response.get('prompt_eval_count', 0),
                    'prompt_eval_duration': response.get('prompt_eval_duration', 0),
                    'eval_count': response.get('eval_count', 0),
                    'eval_duration': response.get('eval_duration', 0)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Ollama API call failed: {str(e)}")
            
            # Handle specific Ollama errors
            error_str = str(e).lower()
            if "connection" in error_str or "refused" in error_str:
                raise OllamaConnectionError(self.config.ollama_host)
            elif "not found" in error_str or "no such file" in error_str:
                raise OllamaModelNotFoundError(self.config.model)
            elif "out of memory" in error_str or "insufficient" in error_str:
                raise OllamaInsufficientResourcesError(self.config.model)
            else:
                raise LLMError(f"Ollama generation failed: {str(e)}")
    
    def summarize_text(
        self,
        text: str,
        target_length: int = 500,
        style: str = "professional"
    ) -> Dict[str, Any]:
        """
        Summarize text to approximately the target length.
        
        Args:
            text: Text to summarize
            target_length: Target word count for summary
            style: Style of summary (professional, casual, academic)
            
        Returns:
            Dictionary containing summary and metadata
        """
        system_prompt = f"""You are an expert content summarizer. Create a {style} summary of the provided text that is approximately {target_length} words long.

Guidelines:
- Focus on the main points and key insights
- Maintain the original tone and important context
- Use clear, concise language
- Include specific details when relevant
- Ensure the summary is coherent and well-structured"""

        user_prompt = f"""Please summarize the following text in approximately {target_length} words:

{text}

Summary:"""

        try:
            result = self.generate_text(
                prompt=user_prompt,
                system_prompt=system_prompt,
                max_tokens=int(target_length * 2),  # Allow extra tokens for safety
                temperature=0.3  # Lower temperature for more focused summaries
            )
            
            summary_text = result['text'].strip()
            word_count = len(summary_text.split())
            
            return {
                'summary': summary_text,
                'word_count': word_count,
                'target_length': target_length,
                'style': style,
                'original_length': len(text.split()),
                'compression_ratio': len(text.split()) / word_count if word_count > 0 else 0,
                'generation_metadata': result,
                'success': True
            }
            
        except Exception as e:
            self.logger.error(f"Text summarization failed: {str(e)}")
            raise
    
    def extract_keywords(
        self,
        text: str,
        count: int = 8,
        include_phrases: bool = True
    ) -> Dict[str, Any]:
        """
        Extract keywords from text.
        
        Args:
            text: Text to extract keywords from
            count: Number of keywords to extract
            include_phrases: Whether to include key phrases
            
        Returns:
            Dictionary containing keywords and metadata
        """
        phrase_instruction = "and key phrases (2-3 words)" if include_phrases else ""
        
        system_prompt = f"""You are an expert at extracting relevant keywords from text. Extract the {count} most important keywords {phrase_instruction} from the provided text.

Guidelines:
- Focus on the most relevant and significant terms
- Include both single words and short phrases if requested
- Avoid common stop words unless they're part of important phrases
- Consider the context and domain of the text
- Provide keywords that would be useful for search and categorization"""

        user_prompt = f"""Extract the {count} most important keywords from this text:

{text}

Please provide exactly {count} keywords, one per line, without numbering or bullets:"""

        try:
            result = self.generate_text(
                prompt=user_prompt,
                system_prompt=system_prompt,
                max_tokens=200,
                temperature=0.2  # Low temperature for consistent keyword extraction
            )
            
            keywords_text = result['text'].strip()
            keywords = [kw.strip() for kw in keywords_text.split('\n') if kw.strip()]
            
            # Ensure we have the right number of keywords
            keywords = keywords[:count]
            
            return {
                'keywords': keywords,
                'count': len(keywords),
                'requested_count': count,
                'include_phrases': include_phrases,
                'generation_metadata': result,
                'success': True
            }
            
        except Exception as e:
            self.logger.error(f"Keyword extraction failed: {str(e)}")
            raise
    
    def generate_timestamps(
        self,
        transcript_data: List[Dict[str, Any]],
        video_id: str,
        count: int = 5
    ) -> Dict[str, Any]:
        """
        Generate important timestamps with descriptions.
        
        Args:
            transcript_data: Raw transcript data with timestamps
            video_id: YouTube video ID
            count: Number of timestamps to generate
            
        Returns:
            Dictionary containing timestamps and metadata
        """
        # Create a readable transcript with timestamps
        transcript_text = ""
        for i, entry in enumerate(transcript_data[:50]):  # Use first 50 entries
            start_time = entry.get('start', 0)
            text = entry.get('text', '')
            transcript_text += f"[{start_time:.1f}s] {text}\n"
        
        system_prompt = f"""You are an expert at identifying the most important moments in video content. Analyze the provided transcript and identify the {count} most significant timestamps.

Guidelines:
- Focus on key insights, important announcements, or turning points
- Include a brief description of what happens at each timestamp
- Rate the importance on a scale of 1-10
- Provide timestamps in the format: [timestamp]s
- Choose moments that viewers would want to jump to directly"""

        user_prompt = f"""Analyze this transcript and identify the {count} most important timestamps:

{transcript_text}

For each timestamp, provide:
1. The timestamp in seconds
2. A brief description (10-20 words)
3. Importance rating (1-10)

Format each entry as:
[XXX.X]s (Rating: X/10) - Description"""

        try:
            result = self.generate_text(
                prompt=user_prompt,
                system_prompt=system_prompt,
                max_tokens=400,
                temperature=0.3
            )
            
            timestamps_text = result['text'].strip()
            timestamps = self._parse_timestamps(timestamps_text, video_id)
            
            return {
                'timestamps': timestamps,
                'count': len(timestamps),
                'requested_count': count,
                'video_id': video_id,
                'generation_metadata': result,
                'success': True
            }
            
        except Exception as e:
            self.logger.error(f"Timestamp generation failed: {str(e)}")
            raise
    
    def _parse_timestamps(self, timestamps_text: str, video_id: str) -> List[Dict[str, Any]]:
        """Parse generated timestamps into structured format."""
        timestamps = []
        
        for line in timestamps_text.split('\n'):
            if not line.strip():
                continue
            
            # Extract timestamp, rating, and description
            import re
            match = re.match(r'\[(\d+(?:\.\d+)?)\]s\s*\(Rating:\s*(\d+)/10\)\s*-\s*(.+)', line.strip())
            
            if match:
                timestamp_seconds = float(match.group(1))
                rating = int(match.group(2))
                description = match.group(3).strip()
                
                # Generate YouTube URL with timestamp
                youtube_url = f"https://www.youtube.com/watch?v={video_id}&t={int(timestamp_seconds)}s"
                
                timestamps.append({
                    'timestamp_seconds': timestamp_seconds,
                    'timestamp_formatted': self._format_timestamp(timestamp_seconds),
                    'description': description,
                    'importance_rating': rating,
                    'youtube_url': youtube_url
                })
        
        # Sort by timestamp
        timestamps.sort(key=lambda x: x['timestamp_seconds'])
        
        return timestamps
    
    def _format_timestamp(self, seconds: float) -> str:
        """Format timestamp in MM:SS or HH:MM:SS format."""
        total_seconds = int(seconds)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        
        if hours > 0:
            return f"{hours}:{minutes:02d}:{seconds:02d}"
        else:
            return f"{minutes}:{seconds:02d}"


# Convenience functions for common operations
def create_llm_client(
    provider: str = "openai",
    model: str = "gpt-3.5-turbo",
    api_key: Optional[str] = None,
    max_tokens: int = 1000,
    temperature: float = 0.7,
    ollama_host: str = "http://localhost:11434",
    ollama_keep_alive: str = "5m"
) -> LLMClient:
    """
    Create an LLM client with the specified configuration.
    
    Args:
        provider: LLM provider ("openai", "anthropic", or "ollama")
        model: Model name to use
        api_key: API key (optional, can use environment variable)
        max_tokens: Maximum tokens for generation
        temperature: Temperature for generation
        ollama_host: Ollama server host (for ollama provider)
        ollama_keep_alive: How long to keep model loaded (for ollama provider)
        
    Returns:
        Configured LLMClient instance
    """
    provider_enum = LLMProvider(provider.lower())
    
    config = LLMConfig(
        provider=provider_enum,
        model=model,
        api_key=api_key,
        max_tokens=max_tokens,
        temperature=temperature,
        ollama_host=ollama_host,
        ollama_keep_alive=ollama_keep_alive
    )
    
    return LLMClient(config)


def get_default_models() -> Dict[str, List[str]]:
    """Get default model options for each provider."""
    return {
        'openai': [
            'gpt-4',
            'gpt-4-turbo',
            'gpt-3.5-turbo',
            'gpt-3.5-turbo-16k'
        ],
        'anthropic': [
            'claude-3-opus-20240229',
            'claude-3-sonnet-20240229',
            'claude-3-haiku-20240307'
        ],
        'ollama': [
            'llama3.1:8b',
            'llama3.2:3b', 
            'mistral:7b',
            'qwen2.5:7b',
            'mistral-small:24b',
            'llama3.3:70b'
        ]
    }


def detect_provider_from_model(model: str) -> str:
    """Detect provider from model name."""
    if model.startswith('gpt-'):
        return 'openai'
    elif model.startswith('claude-'):
        return 'anthropic'
    elif any(model.startswith(prefix) for prefix in ['llama', 'mistral', 'qwen', 'phi', 'gemma']):
        return 'ollama'
    else:
        return 'openai'  # Default to OpenAI


def check_ollama_availability(host: str = "http://localhost:11434") -> Dict[str, Any]:
    """
    Check if Ollama server is available and return status information.
    
    Args:
        host: Ollama server host URL
        
    Returns:
        Dictionary containing availability status and model information
    """
    if not OLLAMA_AVAILABLE:
        return {
            'available': False,
            'reason': 'Ollama library not installed',
            'models': [],
            'host': host
        }
    
    try:
        client = ollama.Client(host=host)
        models_response = client.list()
        
        # Handle different response formats
        if hasattr(models_response, 'models'):
            # New format: ListResponse object with models attribute
            model_list = models_response.models
        elif isinstance(models_response, dict):
            # Old format: dictionary with 'models' key
            model_list = models_response.get('models', [])
        else:
            # Fallback
            model_list = []
        
        available_models = []
        for model_info in model_list:
            name = None
            if isinstance(model_info, dict):
                # Handle dict format
                name = model_info.get('name') or model_info.get('model')
            else:
                # Handle object format - try multiple possible attributes
                name = (getattr(model_info, 'name', None) or 
                       getattr(model_info, 'model', None))
            
            if name:
                available_models.append(name)
        
        return {
            'available': True,
            'models': available_models,
            'model_count': len(available_models),
            'host': host,
            'status': 'connected'
        }
        
    except Exception as e:
        error_str = str(e).lower()
        if "connection" in error_str or "refused" in error_str:
            reason = f"Cannot connect to Ollama server at {host}"
        else:
            reason = f"Ollama error: {str(e)}"
            
        return {
            'available': False,
            'reason': reason,
            'models': [],
            'host': host,
            'error': str(e)
        }


def get_recommended_ollama_model(task: str = "summarization", resource_level: str = "medium") -> str:
    """
    Get recommended Ollama model based on task and resource constraints.
    
    Args:
        task: Type of task ("summarization", "general", "multilingual")
        resource_level: Resource level ("low", "medium", "high")
        
    Returns:
        Recommended model name
    """
    recommendations = {
        "summarization": {
            "low": "llama3.2:3b",
            "medium": "llama3.1:8b", 
            "high": "mistral-small:24b"
        },
        "multilingual": {
            "low": "qwen2.5:7b",
            "medium": "qwen2.5:7b",
            "high": "llama3.3:70b"
        },
        "general": {
            "low": "llama3.2:3b",
            "medium": "mistral:7b",
            "high": "llama3.1:8b"
        }
    }
    
    return recommendations.get(task, recommendations["general"]).get(resource_level, "llama3.1:8b")