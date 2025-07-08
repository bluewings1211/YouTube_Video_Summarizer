"""
Proxy rotation and rate limiting management for YouTube API requests.

This module provides comprehensive proxy management with rotation, health checking,
rate limiting, and retry logic with exponential backoff.
"""

import asyncio
import time
import random
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from urllib.parse import urlparse
import threading
from collections import deque
from contextlib import asynccontextmanager, contextmanager
import urllib.request
import urllib.error
import socket
import ssl
from concurrent.futures import ThreadPoolExecutor, as_completed
import aiohttp
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

from ..config import settings

# Configure logging
logger = logging.getLogger(__name__)


class ProxyStatus(Enum):
    """Proxy health status enumeration."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ProxyInfo:
    """Proxy information and health status."""
    url: str
    status: ProxyStatus = ProxyStatus.UNKNOWN
    failure_count: int = 0
    last_used: Optional[datetime] = None
    last_health_check: Optional[datetime] = None
    response_time: Optional[float] = None
    consecutive_failures: int = 0
    total_requests: int = 0
    successful_requests: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate for this proxy."""
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests
    
    @property
    def is_healthy(self) -> bool:
        """Check if proxy is considered healthy."""
        return (
            self.status == ProxyStatus.HEALTHY and
            self.consecutive_failures < settings.proxy_max_failures
        )
    
    @property
    def parsed_url(self) -> dict:
        """Parse proxy URL into components."""
        parsed = urlparse(self.url)
        return {
            'scheme': parsed.scheme,
            'hostname': parsed.hostname,
            'port': parsed.port,
            'username': parsed.username,
            'password': parsed.password
        }


@dataclass
class RateLimitInfo:
    """Rate limiting information."""
    requests_made: int = 0
    last_request_time: Optional[datetime] = None
    request_times: deque = field(default_factory=lambda: deque(maxlen=100))
    burst_count: int = 0
    burst_start_time: Optional[datetime] = None
    
    def add_request(self) -> None:
        """Record a new request."""
        now = datetime.utcnow()
        self.requests_made += 1
        self.last_request_time = now
        self.request_times.append(now)
        
        # Handle burst counting
        if self.burst_start_time is None or (now - self.burst_start_time).total_seconds() > 60:
            self.burst_count = 1
            self.burst_start_time = now
        else:
            self.burst_count += 1
    
    def get_requests_in_last_minute(self) -> int:
        """Get number of requests in the last minute."""
        if not self.request_times:
            return 0
        
        now = datetime.utcnow()
        cutoff_time = now - timedelta(minutes=1)
        
        return sum(1 for req_time in self.request_times if req_time >= cutoff_time)
    
    def time_until_next_request(self) -> float:
        """Calculate time until next request is allowed."""
        if not self.last_request_time:
            return 0.0
        
        time_since_last = (datetime.utcnow() - self.last_request_time).total_seconds()
        min_interval = settings.rate_limit_min_interval
        
        return max(0.0, min_interval - time_since_last)
    
    def can_make_request(self) -> bool:
        """Check if a request can be made now."""
        if not settings.rate_limit_enabled:
            return True
        
        # Check minimum interval
        if self.time_until_next_request() > 0:
            return False
        
        # Check requests per minute limit
        if self.get_requests_in_last_minute() >= settings.rate_limit_max_requests_per_minute:
            return False
        
        # Check burst limit
        if self.burst_count >= settings.rate_limit_burst_requests:
            if self.burst_start_time and (datetime.utcnow() - self.burst_start_time).total_seconds() < 60:
                return False
        
        return True


class ProxyRotationManager:
    """
    Comprehensive proxy rotation and rate limiting manager.
    
    This class handles:
    - Proxy rotation with health checking
    - Rate limiting with configurable intervals
    - Retry logic with exponential backoff
    - Connection pooling for performance
    - Comprehensive logging and monitoring
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the proxy rotation manager.
        
        Args:
            config: Optional configuration dictionary, defaults to settings
        """
        self.config = config or settings.proxy_config
        self.rate_limit_config = settings.rate_limit_config
        self.retry_config = settings.retry_config
        
        # Initialize proxy pool
        self.proxies: List[ProxyInfo] = []
        self.current_proxy_index = 0
        self.proxy_lock = threading.Lock()
        
        # Rate limiting
        self.rate_limit_info = RateLimitInfo()
        self.rate_limit_lock = threading.Lock()
        
        # Connection pools
        self.session_pool: Dict[str, requests.Session] = {}
        self.pool_lock = threading.Lock()
        
        # Health check management
        self.health_check_executor = ThreadPoolExecutor(max_workers=5, thread_name_prefix="proxy-health")
        self.health_check_running = False
        self.health_check_thread: Optional[threading.Thread] = None
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'proxy_switches': 0,
            'rate_limit_hits': 0,
            'retries_performed': 0,
            'health_checks_performed': 0,
            'start_time': datetime.utcnow()
        }
        
        # Initialize
        self._initialize_proxies()
        self._start_health_check_thread()
        
        logger.info(f"ProxyRotationManager initialized with {len(self.proxies)} proxies")
    
    def _initialize_proxies(self) -> None:
        """Initialize proxy pool from configuration."""
        if not self.config.get('enabled', False):
            logger.info("Proxy rotation disabled")
            return
        
        proxy_urls = self.config.get('urls', [])
        if not proxy_urls:
            logger.warning("No proxy URLs configured")
            return
        
        for url in proxy_urls:
            proxy_info = ProxyInfo(url=url)
            self.proxies.append(proxy_info)
            logger.debug(f"Added proxy: {url}")
        
        logger.info(f"Initialized {len(self.proxies)} proxies")
    
    def _start_health_check_thread(self) -> None:
        """Start background health check thread."""
        if not self.config.get('enabled', False) or not self.proxies:
            return
        
        self.health_check_running = True
        self.health_check_thread = threading.Thread(
            target=self._health_check_loop,
            daemon=True,
            name="proxy-health-checker"
        )
        self.health_check_thread.start()
        logger.info("Started proxy health check thread")
    
    def _health_check_loop(self) -> None:
        """Background health check loop."""
        while self.health_check_running:
            try:
                interval = self.config.get('health_check_interval', 300)
                time.sleep(interval)
                
                if self.health_check_running:
                    self._perform_health_checks()
                    
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                time.sleep(30)  # Wait before retrying
    
    def _perform_health_checks(self) -> None:
        """Perform health checks on all proxies."""
        if not self.proxies:
            return
        
        logger.debug("Performing proxy health checks")
        
        # Submit health check tasks
        future_to_proxy = {}
        for proxy in self.proxies:
            future = self.health_check_executor.submit(self._check_proxy_health, proxy)
            future_to_proxy[future] = proxy
        
        # Process results
        for future in as_completed(future_to_proxy, timeout=60):
            proxy = future_to_proxy[future]
            try:
                result = future.result()
                self._update_proxy_health(proxy, result)
                self.stats['health_checks_performed'] += 1
                
            except Exception as e:
                logger.error(f"Health check failed for proxy {proxy.url}: {e}")
                self._update_proxy_health(proxy, False)
        
        # Log health status
        healthy_count = sum(1 for p in self.proxies if p.is_healthy)
        logger.info(f"Health check complete: {healthy_count}/{len(self.proxies)} proxies healthy")
    
    def _check_proxy_health(self, proxy: ProxyInfo) -> bool:
        """
        Check health of a single proxy.
        
        Args:
            proxy: Proxy to check
            
        Returns:
            True if proxy is healthy, False otherwise
        """
        try:
            # Create a test request through the proxy
            test_url = "https://httpbin.org/ip"
            timeout = self.config.get('health_check_timeout', 10)
            
            proxies = {
                'http': proxy.url,
                'https': proxy.url
            }
            
            start_time = time.time()
            
            logger.debug(f"Health check starting for proxy {proxy.url}")
            
            response = requests.get(test_url, proxies=proxies, timeout=timeout)
            end_time = time.time()
            
            response_time = end_time - start_time
            
            if response.status_code == 200:
                proxy.response_time = response_time
                proxy.last_health_check = datetime.utcnow()
                
                # Log detailed health check success
                logger.info(f"Proxy health check PASSED for {proxy.url} "
                          f"(response_time: {response_time:.2f}s, status: {response.status_code})")
                
                # Check if proxy returns expected content
                try:
                    json_response = response.json()
                    proxy_ip = json_response.get('origin', 'unknown')
                    logger.debug(f"Proxy {proxy.url} returned IP: {proxy_ip}")
                except Exception:
                    logger.debug(f"Proxy {proxy.url} returned non-JSON response")
                
                return True
            else:
                logger.warning(f"Proxy health check FAILED for {proxy.url} "
                             f"(status: {response.status_code}, response_time: {response_time:.2f}s)")
                return False
                
        except requests.exceptions.ProxyError as e:
            logger.error(f"Proxy connection FAILED for {proxy.url}: {e}")
            return False
        except requests.exceptions.Timeout as e:
            logger.warning(f"Proxy health check TIMEOUT for {proxy.url} after {timeout}s: {e}")
            return False
        except requests.exceptions.ConnectionError as e:
            logger.warning(f"Proxy connection ERROR for {proxy.url}: {e}")
            return False
        except Exception as e:
            logger.error(f"Proxy health check UNEXPECTED ERROR for {proxy.url}: {e}", exc_info=True)
            return False
    
    def _update_proxy_health(self, proxy: ProxyInfo, is_healthy: bool) -> None:
        """
        Update proxy health status.
        
        Args:
            proxy: Proxy to update
            is_healthy: Whether the proxy is healthy
        """
        with self.proxy_lock:
            if is_healthy:
                proxy.status = ProxyStatus.HEALTHY
                proxy.consecutive_failures = 0
                logger.debug(f"Proxy {proxy.url} is healthy (response time: {proxy.response_time:.2f}s)")
            else:
                proxy.consecutive_failures += 1
                proxy.failure_count += 1
                
                if proxy.consecutive_failures >= settings.proxy_max_failures:
                    proxy.status = ProxyStatus.UNHEALTHY
                    logger.warning(f"Proxy {proxy.url} marked as unhealthy after {proxy.consecutive_failures} failures")
                else:
                    proxy.status = ProxyStatus.DEGRADED
                    logger.debug(f"Proxy {proxy.url} degraded ({proxy.consecutive_failures} failures)")
    
    def get_current_proxy(self) -> Optional[ProxyInfo]:
        """
        Get the current proxy for use.
        
        Returns:
            Current proxy info or None if no proxy available
        """
        if not self.config.get('enabled', False) or not self.proxies:
            return None
        
        with self.proxy_lock:
            # Find next healthy proxy
            healthy_proxies = [p for p in self.proxies if p.is_healthy]
            
            if not healthy_proxies:
                logger.warning("No healthy proxies available")
                return None
            
            # Rotate if enabled
            if self.config.get('rotation_enabled', True):
                self.current_proxy_index = (self.current_proxy_index + 1) % len(healthy_proxies)
                self.stats['proxy_switches'] += 1
            
            proxy = healthy_proxies[self.current_proxy_index]
            proxy.last_used = datetime.utcnow()
            
            return proxy
    
    def get_proxy_dict(self, proxy: ProxyInfo) -> Dict[str, str]:
        """
        Get proxy dictionary for requests.
        
        Args:
            proxy: Proxy information
            
        Returns:
            Dictionary with proxy configuration
        """
        return {
            'http': proxy.url,
            'https': proxy.url
        }
    
    def wait_for_rate_limit(self) -> None:
        """Wait for rate limit if necessary."""
        if not self.rate_limit_config.get('enabled', True):
            return
        
        with self.rate_limit_lock:
            wait_time = self.rate_limit_info.time_until_next_request()
            
            if wait_time > 0:
                self.stats['rate_limit_hits'] += 1
                logger.info(f"Rate limit hit, waiting {wait_time:.2f} seconds")
                time.sleep(wait_time)
            
            self.rate_limit_info.add_request()
    
    def get_session(self, proxy: Optional[ProxyInfo] = None) -> requests.Session:
        """
        Get a configured session for requests.
        
        Args:
            proxy: Optional proxy to use
            
        Returns:
            Configured requests session
        """
        session_key = proxy.url if proxy else "no_proxy"
        
        with self.pool_lock:
            if session_key not in self.session_pool:
                session = requests.Session()
                
                # Configure retry strategy
                retry_strategy = Retry(
                    total=3,
                    backoff_factor=1,
                    status_forcelist=[429, 500, 502, 503, 504],
                )
                
                adapter = HTTPAdapter(max_retries=retry_strategy)
                session.mount("http://", adapter)
                session.mount("https://", adapter)
                
                # Configure proxy
                if proxy:
                    session.proxies = self.get_proxy_dict(proxy)
                
                # Configure timeout
                session.timeout = self.config.get('timeout', 30)
                
                self.session_pool[session_key] = session
            
            return self.session_pool[session_key]
    
    @contextmanager
    def request_context(self):
        """
        Context manager for making requests with proxy rotation and rate limiting.
        
        Yields:
            Tuple of (session, proxy_info)
        """
        request_start_time = time.time()
        proxy = None
        
        try:
            # Wait for rate limit
            self.wait_for_rate_limit()
            
            # Get proxy
            proxy = self.get_current_proxy()
            
            if proxy:
                logger.debug(f"Request using proxy: {proxy.url} "
                           f"(success_rate: {proxy.success_rate:.2f}, "
                           f"consecutive_failures: {proxy.consecutive_failures})")
            else:
                logger.debug("Request using direct connection (no proxy)")
            
            # Get session
            session = self.get_session(proxy)
            
            yield session, proxy
            
            # Calculate request time
            request_time = time.time() - request_start_time
            
            # Update success statistics
            self.stats['total_requests'] += 1
            self.stats['successful_requests'] += 1
            
            if proxy:
                proxy.total_requests += 1
                proxy.successful_requests += 1
                proxy.consecutive_failures = 0  # Reset consecutive failures on success
                
                logger.info(f"Request SUCCESS via proxy {proxy.url} "
                          f"(time: {request_time:.2f}s, success_rate: {proxy.success_rate:.2f})")
            else:
                logger.info(f"Request SUCCESS via direct connection (time: {request_time:.2f}s)")
                
        except requests.exceptions.ProxyError as e:
            request_time = time.time() - request_start_time
            self._handle_request_failure(proxy, e, "PROXY_ERROR", request_time)
            raise
        except requests.exceptions.Timeout as e:
            request_time = time.time() - request_start_time
            self._handle_request_failure(proxy, e, "TIMEOUT", request_time)
            raise
        except requests.exceptions.ConnectionError as e:
            request_time = time.time() - request_start_time
            self._handle_request_failure(proxy, e, "CONNECTION_ERROR", request_time)
            raise
        except requests.exceptions.HTTPError as e:
            request_time = time.time() - request_start_time
            self._handle_request_failure(proxy, e, "HTTP_ERROR", request_time)
            raise
        except Exception as e:
            request_time = time.time() - request_start_time
            self._handle_request_failure(proxy, e, "UNEXPECTED_ERROR", request_time)
            raise
    
    def _handle_request_failure(self, proxy: Optional[ProxyInfo], exception: Exception, 
                               error_type: str, request_time: float) -> None:
        """
        Handle request failure with detailed logging and statistics.
        
        Args:
            proxy: Proxy that was used (if any)
            exception: Exception that occurred
            error_type: Type of error for logging
            request_time: Time taken for the request
        """
        # Update failure statistics
        self.stats['total_requests'] += 1
        self.stats['failed_requests'] += 1
        
        if proxy:
            proxy.total_requests += 1
            proxy.consecutive_failures += 1
            proxy.failure_count += 1
            
            # Log detailed proxy failure
            logger.error(f"Request {error_type} via proxy {proxy.url} "
                        f"(time: {request_time:.2f}s, consecutive_failures: {proxy.consecutive_failures}, "
                        f"success_rate: {proxy.success_rate:.2f}): {exception}")
            
            # Check if proxy should be marked unhealthy
            if proxy.consecutive_failures >= settings.proxy_max_failures:
                old_status = proxy.status
                proxy.status = ProxyStatus.UNHEALTHY
                logger.warning(f"Proxy {proxy.url} marked UNHEALTHY due to {proxy.consecutive_failures} "
                             f"consecutive failures (was {old_status.value})")
                
                # Log statistics for the failed proxy
                logger.info(f"Failed proxy statistics - "
                          f"Total requests: {proxy.total_requests}, "
                          f"Successful: {proxy.successful_requests}, "
                          f"Failed: {proxy.failure_count}, "
                          f"Success rate: {proxy.success_rate:.2f}")
        else:
            logger.error(f"Request {error_type} via direct connection "
                        f"(time: {request_time:.2f}s): {exception}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics.
        
        Returns:
            Dictionary with statistics
        """
        uptime = (datetime.utcnow() - self.stats['start_time']).total_seconds()
        
        proxy_stats = []
        for proxy in self.proxies:
            proxy_stats.append({
                'url': proxy.url,
                'status': proxy.status.value,
                'success_rate': proxy.success_rate,
                'total_requests': proxy.total_requests,
                'successful_requests': proxy.successful_requests,
                'failure_count': proxy.failure_count,
                'consecutive_failures': proxy.consecutive_failures,
                'response_time': proxy.response_time,
                'last_used': proxy.last_used.isoformat() if proxy.last_used else None,
                'last_health_check': proxy.last_health_check.isoformat() if proxy.last_health_check else None
            })
        
        return {
            'uptime_seconds': uptime,
            'total_requests': self.stats['total_requests'],
            'successful_requests': self.stats['successful_requests'],
            'failed_requests': self.stats['failed_requests'],
            'success_rate': self.stats['successful_requests'] / max(1, self.stats['total_requests']),
            'proxy_switches': self.stats['proxy_switches'],
            'rate_limit_hits': self.stats['rate_limit_hits'],
            'retries_performed': self.stats['retries_performed'],
            'health_checks_performed': self.stats['health_checks_performed'],
            'proxy_count': len(self.proxies),
            'healthy_proxy_count': sum(1 for p in self.proxies if p.is_healthy),
            'proxy_details': proxy_stats,
            'rate_limit_info': {
                'requests_made': self.rate_limit_info.requests_made,
                'requests_in_last_minute': self.rate_limit_info.get_requests_in_last_minute(),
                'time_until_next_request': self.rate_limit_info.time_until_next_request(),
                'can_make_request': self.rate_limit_info.can_make_request()
            }
        }
    
    def shutdown(self) -> None:
        """Clean shutdown of proxy manager."""
        logger.info("Shutting down proxy manager")
        
        # Stop health check thread
        self.health_check_running = False
        if self.health_check_thread:
            self.health_check_thread.join(timeout=5)
        
        # Shutdown executor
        self.health_check_executor.shutdown(wait=True)
        
        # Close sessions
        with self.pool_lock:
            for session in self.session_pool.values():
                session.close()
            self.session_pool.clear()
        
        logger.info("Proxy manager shutdown complete")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()


class RetryManager:
    """
    Exponential backoff retry manager for failed requests.
    
    This class implements configurable retry logic with exponential backoff,
    jitter, and comprehensive logging.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize retry manager.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or settings.retry_config
        self.logger = logging.getLogger(f"{__name__}.RetryManager")
    
    def calculate_delay(self, attempt: int) -> float:
        """
        Calculate delay for retry attempt.
        
        Args:
            attempt: Retry attempt number (0-based)
            
        Returns:
            Delay in seconds
        """
        if not self.config.get('enabled', True):
            return 0.0
        
        base_delay = self.config.get('base_delay', 1.0)
        multiplier = self.config.get('backoff_multiplier', 2.0)
        max_delay = self.config.get('max_delay', 60.0)
        
        # Calculate exponential backoff
        delay = base_delay * (multiplier ** attempt)
        delay = min(delay, max_delay)
        
        # Add jitter if enabled
        if self.config.get('jitter_enabled', True):
            jitter = random.uniform(0, 0.1) * delay
            delay += jitter
        
        return delay
    
    def should_retry(self, attempt: int, exception: Exception) -> bool:
        """
        Determine if request should be retried.
        
        Args:
            attempt: Current attempt number (0-based)
            exception: Exception that occurred
            
        Returns:
            True if should retry, False otherwise
        """
        if not self.config.get('enabled', True):
            return False
        
        max_attempts = self.config.get('max_attempts', 5)
        
        # Check attempt limit
        if attempt >= max_attempts:
            return False
        
        # Check exception type
        retryable_exceptions = (
            ConnectionError,
            TimeoutError,
            urllib.error.URLError,
            requests.exceptions.RequestException,
            socket.error,
            ssl.SSLError
        )
        
        return isinstance(exception, retryable_exceptions)
    
    @contextmanager
    def retry_context(self, operation_name: str = "operation"):
        """
        Context manager for retry logic.
        
        Args:
            operation_name: Name of operation for logging
        """
        attempt = 0
        start_time = time.time()
        last_exception = None
        
        while True:
            try:
                self.logger.debug(f"{operation_name} attempt {attempt + 1}")
                yield attempt
                
                # Success case
                total_time = time.time() - start_time
                if attempt > 0:
                    self.logger.info(f"{operation_name} SUCCESS after {attempt + 1} attempts "
                                   f"(total_time: {total_time:.2f}s)")
                else:
                    self.logger.debug(f"{operation_name} SUCCESS on first attempt "
                                    f"(time: {total_time:.2f}s)")
                break  # Success, exit retry loop
                
            except Exception as e:
                last_exception = e
                current_time = time.time() - start_time
                
                # Classify the exception type for better logging
                exception_type = type(e).__name__
                
                if not self.should_retry(attempt, e):
                    # Log final failure with detailed information
                    self.logger.error(f"{operation_name} FINAL FAILURE after {attempt + 1} attempts "
                                    f"(total_time: {current_time:.2f}s, "
                                    f"exception_type: {exception_type}): {e}")
                    
                    # Add retry statistics
                    if attempt > 0:
                        self.logger.info(f"{operation_name} retry statistics - "
                                       f"attempts: {attempt + 1}, "
                                       f"total_time: {current_time:.2f}s, "
                                       f"final_exception: {exception_type}")
                    raise
                
                delay = self.calculate_delay(attempt)
                
                # Log retry with detailed information
                self.logger.warning(f"{operation_name} attempt {attempt + 1} FAILED "
                                  f"(exception_type: {exception_type}, "
                                  f"time_elapsed: {current_time:.2f}s): {e}. "
                                  f"Retrying in {delay:.2f}s (attempt {attempt + 2}/{self.config.get('max_attempts', 5)})")
                
                time.sleep(delay)
                attempt += 1


# Global instances
proxy_manager = ProxyRotationManager()
retry_manager = RetryManager()


def get_proxy_manager() -> ProxyRotationManager:
    """Get the global proxy manager instance."""
    return proxy_manager


def get_retry_manager() -> RetryManager:
    """Get the global retry manager instance."""
    return retry_manager