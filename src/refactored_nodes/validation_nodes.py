"""
Base classes and validation nodes for PocketFlow processing.

This module contains the base processing node class and error handling
that is shared across all node types in the YouTube summarization workflow.
"""

import logging
import time
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

try:
    from pocketflow import Node, NodeState, Store
except ImportError:
    # Fallback base classes for development
    class Node(ABC):
        def __init__(self, name: str):
            self.name = name
            self.logger = logging.getLogger(f"{__name__}.{name}")
        
        @abstractmethod
        def prep(self, store: "Store") -> Dict[str, Any]:
            pass
        
        @abstractmethod
        def exec(self, store: "Store", prep_result: Dict[str, Any]) -> Dict[str, Any]:
            pass
        
        @abstractmethod
        def post(self, store: "Store", prep_result: Dict[str, Any], exec_result: Dict[str, Any]) -> Dict[str, Any]:
            pass
    
    class Store(dict):
        def __init__(self):
            super().__init__()
    
    class NodeState:
        PENDING = "pending"
        PREP = "prep"
        EXEC = "exec"
        POST = "post"
        SUCCESS = "success"
        FAILED = "failed"

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class NodeError:
    """Structured error information for nodes."""
    node_name: str
    error_type: str
    message: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    retry_count: int = 0
    is_recoverable: bool = True


class BaseProcessingNode(Node):
    """Base class for all processing nodes with common functionality."""
    
    def __init__(self, name: str, max_retries: int = 3, retry_delay: float = 1.0):
        super().__init__(name)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.logger = logging.getLogger(f"{__name__}.{name}")
    
    def _handle_error(self, error: Exception, context: str, retry_count: int = 0) -> NodeError:
        """Handle and log errors with structured information."""
        # Determine if error is recoverable based on error type
        is_recoverable = self._is_error_recoverable(error) and retry_count < self.max_retries
        
        error_info = NodeError(
            node_name=self.name,
            error_type=type(error).__name__,
            message=str(error),
            retry_count=retry_count,
            is_recoverable=is_recoverable
        )
        
        log_msg = f"{context}: {error_info.error_type} - {error_info.message}"
        if retry_count > 0:
            log_msg += f" (retry {retry_count}/{self.max_retries})"
        
        # Include error details for debugging
        if hasattr(error, 'error_code'):
            log_msg += f" [Code: {error.error_code}]"
        
        if is_recoverable:
            self.logger.warning(log_msg)
        else:
            self.logger.error(log_msg, exc_info=True)
        
        return error_info
    
    def _is_error_recoverable(self, error: Exception) -> bool:
        """Determine if an error is recoverable and worth retrying."""
        # Import here to avoid circular imports
        try:
            from ..utils.youtube_api import (
                PrivateVideoError, LiveVideoError, NoTranscriptAvailableError, 
                VideoTooLongError, NetworkTimeoutError, RateLimitError
            )
            from ..utils.call_llm import (
                LLMRateLimitError, LLMTimeoutError, LLMServerError, 
                LLMAuthenticationError, LLMQuotaError, OllamaConnectionError,
                OllamaModelNotFoundError, OllamaInsufficientResourcesError
            )
        except ImportError:
            # Fallback error types
            class PrivateVideoError(Exception): pass
            class LiveVideoError(Exception): pass
            class NoTranscriptAvailableError(Exception): pass
            class VideoTooLongError(Exception): pass
            class NetworkTimeoutError(Exception): pass
            class RateLimitError(Exception): pass
            class LLMRateLimitError(Exception): pass
            class LLMTimeoutError(Exception): pass
            class LLMServerError(Exception): pass
            class LLMAuthenticationError(Exception): pass
            class LLMQuotaError(Exception): pass
            class OllamaConnectionError(Exception): pass
            class OllamaModelNotFoundError(Exception): pass
            class OllamaInsufficientResourcesError(Exception): pass
        
        # Non-recoverable errors
        non_recoverable_errors = (
            PrivateVideoError,
            LiveVideoError, 
            NoTranscriptAvailableError,
            VideoTooLongError,
            LLMAuthenticationError,
            LLMQuotaError,
            OllamaModelNotFoundError,  # Model not found requires manual intervention
            OllamaInsufficientResourcesError,  # Resource issues require system changes
            ValueError,  # Usually validation errors
            TypeError,   # Usually programming errors
        )
        
        if isinstance(error, non_recoverable_errors):
            return False
        
        # Recoverable errors (worth retrying)
        recoverable_errors = (
            NetworkTimeoutError,
            RateLimitError,
            LLMRateLimitError,
            LLMTimeoutError,
            LLMServerError,
            OllamaConnectionError,  # Connection issues may be temporary
            ConnectionError,
            TimeoutError,
        )
        
        if isinstance(error, recoverable_errors):
            return True
        
        # For generic exceptions, check the error message
        error_message = str(error).lower()
        if any(keyword in error_message for keyword in ['timeout', 'connection', 'network', 'rate limit']):
            return True
        
        # Default to non-recoverable for unknown errors
        return False
    
    def _retry_with_delay(self, retry_count: int) -> None:
        """Sleep with exponential backoff for retries."""
        if retry_count > 0:
            delay = self.retry_delay * (2 ** (retry_count - 1))
            self.logger.info(f"Retrying in {delay:.2f} seconds...")
            time.sleep(delay)
    
    def _validate_store_data(self, store: Store, required_keys: List[str]) -> Tuple[bool, List[str]]:
        """Validate that required data is present in the store."""
        missing_keys = []
        for key in required_keys:
            if key not in store:
                missing_keys.append(key)
        
        return len(missing_keys) == 0, missing_keys
    
    def _safe_store_update(self, store: Store, data: Dict[str, Any]) -> None:
        """Safely update store with new data."""
        try:
            store.update(data)
            self.logger.debug(f"Store updated with keys: {list(data.keys())}")
        except Exception as e:
            self.logger.error(f"Failed to update store: {str(e)}")
            raise
    
    def _execute_with_retry(self, operation, *args, **kwargs):
        """Execute an operation with retry logic and proper error handling."""
        last_error = None
        
        for attempt in range(self.max_retries + 1):
            try:
                self.logger.debug(f"Executing operation (attempt {attempt + 1}/{self.max_retries + 1})")
                return operation(*args, **kwargs)
                
            except Exception as e:
                last_error = e
                error_info = self._handle_error(e, f"Operation failed", attempt)
                
                # If not recoverable or max retries reached, raise the error
                if not error_info.is_recoverable or attempt >= self.max_retries:
                    raise e
                
                # Wait before retrying
                self._retry_with_delay(attempt)
        
        # This should not be reached, but just in case
        if last_error:
            raise last_error
        else:
            raise RuntimeError("Operation failed without specific error")
    
    def _validate_inputs(self, required_keys: List[str], store: Store) -> None:
        """Validate that all required inputs are present in the store."""
        missing_keys = []
        for key in required_keys:
            if key not in store or store[key] is None:
                missing_keys.append(key)
        
        if missing_keys:
            raise ValueError(f"Missing required inputs: {missing_keys}")
    
    def _measure_execution_time(self, start_time: str, end_time: str) -> float:
        """Measure execution time between two ISO format timestamps."""
        try:
            start = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
            end = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
            
            return (end - start).total_seconds()
        except Exception:
            return 0.0