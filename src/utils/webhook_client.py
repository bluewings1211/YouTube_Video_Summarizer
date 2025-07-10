"""
Webhook client utilities for sending HTTP notifications.

This module provides a comprehensive webhook client for sending HTTP notifications
with support for authentication, retry logic, error handling, and performance monitoring.
"""

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union, Tuple
from dataclasses import dataclass
from enum import Enum

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)


class WebhookAuthType(Enum):
    """Supported webhook authentication types."""
    NONE = "none"
    BASIC = "basic"
    BEARER = "bearer"
    API_KEY = "api_key"
    HMAC_SHA256 = "hmac_sha256"
    CUSTOM = "custom"


class WebhookStatus(Enum):
    """Webhook delivery status."""
    PENDING = "pending"
    SENDING = "sending"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    RATE_LIMITED = "rate_limited"
    AUTHENTICATION_FAILED = "authentication_failed"
    INVALID_URL = "invalid_url"
    CONNECTION_ERROR = "connection_error"


@dataclass
class WebhookConfig:
    """Configuration for webhook client."""
    url: str
    method: str = "POST"
    auth_type: WebhookAuthType = WebhookAuthType.NONE
    auth_config: Optional[Dict[str, Any]] = None
    headers: Optional[Dict[str, str]] = None
    timeout_seconds: int = 30
    max_retries: int = 3
    retry_delay_seconds: int = 1
    retry_backoff_multiplier: float = 2.0
    max_retry_delay_seconds: int = 300
    verify_ssl: bool = True
    follow_redirects: bool = True
    user_agent: str = "YouTube-Summarizer-Webhook/1.0"


@dataclass
class WebhookRequest:
    """Webhook request data."""
    payload: Dict[str, Any]
    headers: Optional[Dict[str, str]] = None
    timestamp: Optional[datetime] = None
    request_id: Optional[str] = None
    signature: Optional[str] = None
    event_type: Optional[str] = None
    source: Optional[str] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        if self.request_id is None:
            self.request_id = f"req_{uuid.uuid4().hex[:12]}"


@dataclass
class WebhookResponse:
    """Webhook response data."""
    status: WebhookStatus
    status_code: Optional[int] = None
    response_text: Optional[str] = None
    response_headers: Optional[Dict[str, str]] = None
    response_time_ms: Optional[int] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    final_url: Optional[str] = None
    request_id: Optional[str] = None
    timestamp: Optional[datetime] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()

    @property
    def is_success(self) -> bool:
        """Check if the webhook was successful."""
        return self.status == WebhookStatus.SUCCESS

    @property
    def is_retryable(self) -> bool:
        """Check if the webhook failure is retryable."""
        return self.status in [
            WebhookStatus.TIMEOUT,
            WebhookStatus.CONNECTION_ERROR,
            WebhookStatus.RATE_LIMITED
        ]


class WebhookError(Exception):
    """Base exception for webhook operations."""

    def __init__(self, message: str, status: WebhookStatus = WebhookStatus.FAILED,
                 status_code: Optional[int] = None, retry_after: Optional[int] = None):
        super().__init__(message)
        self.message = message
        self.status = status
        self.status_code = status_code
        self.retry_after = retry_after
        self.timestamp = datetime.utcnow()


class WebhookAuthenticationError(WebhookError):
    """Authentication errors for webhooks."""

    def __init__(self, message: str = "Webhook authentication failed"):
        super().__init__(message, WebhookStatus.AUTHENTICATION_FAILED, 401)


class WebhookTimeoutError(WebhookError):
    """Timeout errors for webhooks."""

    def __init__(self, message: str = "Webhook request timed out"):
        super().__init__(message, WebhookStatus.TIMEOUT, 408)


class WebhookRateLimitError(WebhookError):
    """Rate limit errors for webhooks."""

    def __init__(self, message: str = "Webhook rate limit exceeded", retry_after: int = 60):
        super().__init__(message, WebhookStatus.RATE_LIMITED, 429, retry_after)


class WebhookClient:
    """
    Comprehensive webhook client for sending HTTP notifications.

    Supports multiple authentication methods, retry logic, error handling,
    and performance monitoring.
    """

    def __init__(self, config: WebhookConfig):
        """
        Initialize the webhook client.

        Args:
            config: Webhook configuration
        """
        self.config = config
        self._validate_config()
        self._client = None
        self._stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_response_time_ms': 0,
            'last_request_time': None
        }

    def _validate_config(self):
        """Validate webhook configuration."""
        if not self.config.url:
            raise ValueError("Webhook URL cannot be empty")

        if not self.config.url.startswith(('http://', 'https://')):
            raise ValueError("Webhook URL must start with http:// or https://")

        if self.config.method.upper() not in ['GET', 'POST', 'PUT', 'PATCH', 'DELETE']:
            raise ValueError(f"Invalid HTTP method: {self.config.method}")

        if self.config.timeout_seconds <= 0:
            raise ValueError("Timeout must be positive")

        if self.config.max_retries < 0:
            raise ValueError("Max retries cannot be negative")

    def _get_http_client(self):
        """Get HTTP client (prefer httpx, fallback to requests)."""
        if HTTPX_AVAILABLE:
            if self._client is None:
                self._client = httpx.Client(
                    timeout=self.config.timeout_seconds,
                    verify=self.config.verify_ssl,
                    follow_redirects=self.config.follow_redirects
                )
            return self._client
        elif REQUESTS_AVAILABLE:
            return requests.Session()
        else:
            raise RuntimeError("No HTTP client available. Install httpx or requests.")

    def _prepare_headers(self, request: WebhookRequest) -> Dict[str, str]:
        """Prepare headers for the webhook request."""
        headers = {
            'User-Agent': self.config.user_agent,
            'Content-Type': 'application/json',
            'X-Request-ID': request.request_id,
            'X-Timestamp': request.timestamp.isoformat()
        }

        # Add configuration headers
        if self.config.headers:
            headers.update(self.config.headers)

        # Add request-specific headers
        if request.headers:
            headers.update(request.headers)

        # Add event information if available
        if request.event_type:
            headers['X-Event-Type'] = request.event_type

        if request.source:
            headers['X-Event-Source'] = request.source

        # Add authentication headers
        auth_headers = self._get_auth_headers(request)
        if auth_headers:
            headers.update(auth_headers)

        return headers

    def _get_auth_headers(self, request: WebhookRequest) -> Optional[Dict[str, str]]:
        """Get authentication headers based on auth type."""
        if self.config.auth_type == WebhookAuthType.NONE:
            return None

        if not self.config.auth_config:
            logger.warning("Auth type specified but no auth config provided")
            return None

        auth_config = self.config.auth_config
        headers = {}

        try:
            if self.config.auth_type == WebhookAuthType.BASIC:
                username = auth_config.get('username', '')
                password = auth_config.get('password', '')
                credentials = f"{username}:{password}"
                encoded_credentials = base64.b64encode(credentials.encode()).decode()
                headers['Authorization'] = f"Basic {encoded_credentials}"

            elif self.config.auth_type == WebhookAuthType.BEARER:
                token = auth_config.get('token', '')
                headers['Authorization'] = f"Bearer {token}"

            elif self.config.auth_type == WebhookAuthType.API_KEY:
                api_key = auth_config.get('api_key', '')
                header_name = auth_config.get('header_name', 'X-API-Key')
                headers[header_name] = api_key

            elif self.config.auth_type == WebhookAuthType.HMAC_SHA256:
                secret = auth_config.get('secret', '')
                payload_str = json.dumps(request.payload, sort_keys=True, separators=(',', ':'))
                signature = hmac.new(
                    secret.encode(),
                    payload_str.encode(),
                    hashlib.sha256
                ).hexdigest()
                header_name = auth_config.get('header_name', 'X-Signature-SHA256')
                headers[header_name] = f"sha256={signature}"
                request.signature = signature

            elif self.config.auth_type == WebhookAuthType.CUSTOM:
                # For custom auth, expect headers to be provided in auth_config
                custom_headers = auth_config.get('headers', {})
                headers.update(custom_headers)

        except Exception as e:
            logger.error(f"Failed to generate auth headers: {str(e)}")
            raise WebhookAuthenticationError(f"Authentication header generation failed: {str(e)}")

        return headers

    def _send_request_httpx(self, request: WebhookRequest, headers: Dict[str, str]) -> WebhookResponse:
        """Send request using httpx client."""
        client = self._get_http_client()
        start_time = time.time()

        try:
            response = client.request(
                method=self.config.method.upper(),
                url=self.config.url,
                json=request.payload,
                headers=headers,
                timeout=self.config.timeout_seconds
            )

            end_time = time.time()
            response_time_ms = int((end_time - start_time) * 1000)

            # Check for rate limiting
            if response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', 60))
                raise WebhookRateLimitError(
                    f"Rate limit exceeded: {response.status_code}",
                    retry_after
                )

            # Check for authentication errors
            if response.status_code == 401:
                raise WebhookAuthenticationError(f"Authentication failed: {response.status_code}")

            # Determine status
            if 200 <= response.status_code < 300:
                status = WebhookStatus.SUCCESS
            else:
                status = WebhookStatus.FAILED

            return WebhookResponse(
                status=status,
                status_code=response.status_code,
                response_text=response.text[:1000],  # Limit response text size
                response_headers=dict(response.headers),
                response_time_ms=response_time_ms,
                final_url=str(response.url),
                request_id=request.request_id
            )

        except httpx.TimeoutException:
            raise WebhookTimeoutError("Request timed out")
        except httpx.ConnectError as e:
            raise WebhookError(f"Connection error: {str(e)}", WebhookStatus.CONNECTION_ERROR)
        except Exception as e:
            raise WebhookError(f"Unexpected error: {str(e)}", WebhookStatus.FAILED)

    def _send_request_requests(self, request: WebhookRequest, headers: Dict[str, str]) -> WebhookResponse:
        """Send request using requests library."""
        session = self._get_http_client()
        start_time = time.time()

        try:
            response = session.request(
                method=self.config.method.upper(),
                url=self.config.url,
                json=request.payload,
                headers=headers,
                timeout=self.config.timeout_seconds,
                verify=self.config.verify_ssl,
                allow_redirects=self.config.follow_redirects
            )

            end_time = time.time()
            response_time_ms = int((end_time - start_time) * 1000)

            # Check for rate limiting
            if response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', 60))
                raise WebhookRateLimitError(
                    f"Rate limit exceeded: {response.status_code}",
                    retry_after
                )

            # Check for authentication errors
            if response.status_code == 401:
                raise WebhookAuthenticationError(f"Authentication failed: {response.status_code}")

            # Determine status
            if 200 <= response.status_code < 300:
                status = WebhookStatus.SUCCESS
            else:
                status = WebhookStatus.FAILED

            return WebhookResponse(
                status=status,
                status_code=response.status_code,
                response_text=response.text[:1000],  # Limit response text size
                response_headers=dict(response.headers),
                response_time_ms=response_time_ms,
                final_url=response.url,
                request_id=request.request_id
            )

        except requests.exceptions.Timeout:
            raise WebhookTimeoutError("Request timed out")
        except requests.exceptions.ConnectionError as e:
            raise WebhookError(f"Connection error: {str(e)}", WebhookStatus.CONNECTION_ERROR)
        except Exception as e:
            raise WebhookError(f"Unexpected error: {str(e)}", WebhookStatus.FAILED)

    def send(self, request: WebhookRequest) -> WebhookResponse:
        """
        Send a webhook request with retry logic.

        Args:
            request: Webhook request data

        Returns:
            WebhookResponse: Response from the webhook

        Raises:
            WebhookError: If the webhook fails after all retries
        """
        headers = self._prepare_headers(request)
        last_exception = None
        retry_count = 0

        self._stats['total_requests'] += 1
        self._stats['last_request_time'] = datetime.utcnow()

        for attempt in range(self.config.max_retries + 1):
            try:
                # Send the request
                if HTTPX_AVAILABLE:
                    response = self._send_request_httpx(request, headers)
                else:
                    response = self._send_request_requests(request, headers)

                response.retry_count = retry_count

                # Update statistics
                if response.is_success:
                    self._stats['successful_requests'] += 1
                else:
                    self._stats['failed_requests'] += 1

                if response.response_time_ms:
                    self._stats['total_response_time_ms'] += response.response_time_ms

                # Log the result
                self._log_request(request, response, attempt)

                return response

            except WebhookError as e:
                last_exception = e
                retry_count = attempt

                # Check if we should retry
                if attempt < self.config.max_retries and e.status in [
                    WebhookStatus.TIMEOUT,
                    WebhookStatus.CONNECTION_ERROR,
                    WebhookStatus.RATE_LIMITED
                ]:
                    # Calculate retry delay
                    if e.status == WebhookStatus.RATE_LIMITED and e.retry_after:
                        delay = min(e.retry_after, self.config.max_retry_delay_seconds)
                    else:
                        delay = min(
                            self.config.retry_delay_seconds * (self.config.retry_backoff_multiplier ** attempt),
                            self.config.max_retry_delay_seconds
                        )

                    logger.warning(f"Webhook request failed (attempt {attempt + 1}), retrying in {delay}s: {str(e)}")
                    time.sleep(delay)
                    continue
                else:
                    # Don't retry for authentication errors or if max retries reached
                    break

        # All retries exhausted
        self._stats['failed_requests'] += 1

        if last_exception:
            response = WebhookResponse(
                status=last_exception.status,
                status_code=last_exception.status_code,
                error_message=last_exception.message,
                retry_count=retry_count,
                request_id=request.request_id
            )
        else:
            response = WebhookResponse(
                status=WebhookStatus.FAILED,
                error_message="Unknown error occurred",
                retry_count=retry_count,
                request_id=request.request_id
            )

        self._log_request(request, response, retry_count)
        return response

    def send_async(self, request: WebhookRequest) -> WebhookResponse:
        """
        Send a webhook request asynchronously.

        Args:
            request: Webhook request data

        Returns:
            WebhookResponse: Response from the webhook
        """
        # For now, this is a simple wrapper around the sync method
        # In a full async implementation, you would use httpx.AsyncClient
        return self.send(request)

    def health_check(self, health_check_url: Optional[str] = None) -> WebhookResponse:
        """
        Perform a health check on the webhook endpoint.

        Args:
            health_check_url: Optional specific URL for health check

        Returns:
            WebhookResponse: Health check response
        """
        # Use health check URL if provided, otherwise use main URL
        original_url = self.config.url
        if health_check_url:
            self.config.url = health_check_url

        try:
            # Create a simple health check request
            health_request = WebhookRequest(
                payload={'health_check': True, 'timestamp': datetime.utcnow().isoformat()},
                event_type='health_check'
            )

            # Send with minimal retries for health check
            original_retries = self.config.max_retries
            self.config.max_retries = 1

            response = self.send(health_request)

            # Restore original configuration
            self.config.max_retries = original_retries
            self.config.url = original_url

            return response

        except Exception as e:
            # Restore original configuration
            self.config.url = original_url
            return WebhookResponse(
                status=WebhookStatus.FAILED,
                error_message=f"Health check failed: {str(e)}"
            )

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get webhook client statistics.

        Returns:
            Dict with client statistics
        """
        total_requests = self._stats['total_requests']
        success_rate = 0.0
        avg_response_time = 0.0

        if total_requests > 0:
            success_rate = (self._stats['successful_requests'] / total_requests) * 100
            if self._stats['successful_requests'] > 0:
                avg_response_time = self._stats['total_response_time_ms'] / self._stats['successful_requests']

        return {
            'total_requests': total_requests,
            'successful_requests': self._stats['successful_requests'],
            'failed_requests': self._stats['failed_requests'],
            'success_rate_percentage': round(success_rate, 2),
            'average_response_time_ms': round(avg_response_time, 2),
            'last_request_time': self._stats['last_request_time'].isoformat() if self._stats['last_request_time'] else None,
            'configuration': {
                'url': self.config.url,
                'method': self.config.method,
                'timeout_seconds': self.config.timeout_seconds,
                'max_retries': self.config.max_retries,
                'auth_type': self.config.auth_type.value
            }
        }

    def reset_statistics(self):
        """Reset client statistics."""
        self._stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_response_time_ms': 0,
            'last_request_time': None
        }

    def _log_request(self, request: WebhookRequest, response: WebhookResponse, attempt: int):
        """Log webhook request and response."""
        log_level = logging.INFO if response.is_success else logging.WARNING

        log_data = {
            'request_id': request.request_id,
            'url': self.config.url,
            'method': self.config.method,
            'status': response.status.value,
            'status_code': response.status_code,
            'response_time_ms': response.response_time_ms,
            'attempt': attempt + 1,
            'retry_count': response.retry_count,
            'event_type': request.event_type,
            'source': request.source
        }

        if response.is_success:
            logger.log(log_level, f"Webhook sent successfully: {log_data}")
        else:
            log_data['error'] = response.error_message
            logger.log(log_level, f"Webhook failed: {log_data}")

    def close(self):
        """Close the HTTP client if using httpx."""
        if self._client and hasattr(self._client, 'close'):
            self._client.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# Convenience functions

def create_webhook_client(
    url: str,
    auth_type: WebhookAuthType = WebhookAuthType.NONE,
    auth_config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> WebhookClient:
    """
    Create a webhook client with simplified configuration.

    Args:
        url: Webhook URL
        auth_type: Authentication type
        auth_config: Authentication configuration
        **kwargs: Additional configuration options

    Returns:
        WebhookClient: Configured webhook client
    """
    config = WebhookConfig(
        url=url,
        auth_type=auth_type,
        auth_config=auth_config,
        **kwargs
    )
    return WebhookClient(config)


def send_webhook(
    url: str,
    payload: Dict[str, Any],
    method: str = "POST",
    auth_type: WebhookAuthType = WebhookAuthType.NONE,
    auth_config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> WebhookResponse:
    """
    Send a simple webhook with minimal configuration.

    Args:
        url: Webhook URL
        payload: Data to send
        method: HTTP method
        auth_type: Authentication type
        auth_config: Authentication configuration
        **kwargs: Additional configuration options

    Returns:
        WebhookResponse: Response from the webhook
    """
    config = WebhookConfig(
        url=url,
        method=method,
        auth_type=auth_type,
        auth_config=auth_config,
        **kwargs
    )

    request = WebhookRequest(payload=payload)

    with WebhookClient(config) as client:
        return client.send(request)


def test_webhook_connectivity(url: str, timeout_seconds: int = 10) -> bool:
    """
    Test basic connectivity to a webhook endpoint.

    Args:
        url: Webhook URL to test
        timeout_seconds: Timeout for the test

    Returns:
        bool: True if endpoint is reachable, False otherwise
    """
    try:
        config = WebhookConfig(
            url=url,
            timeout_seconds=timeout_seconds,
            max_retries=0
        )

        request = WebhookRequest(
            payload={'test': True, 'timestamp': datetime.utcnow().isoformat()}
        )

        with WebhookClient(config) as client:
            response = client.send(request)
            return response.status_code is not None and response.status_code < 500

    except Exception:
        return False