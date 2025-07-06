"""
Unit tests for proxy rotation and rate limiting functionality.

This module contains comprehensive tests for:
- Proxy rotation mechanism
- Rate limiting enforcement
- Exponential backoff retry logic
- Proxy health checking
- Failure recovery functionality
- Error handling and logging
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock, PropertyMock
from datetime import datetime, timedelta
from contextlib import contextmanager
import requests
from requests.exceptions import ProxyError, Timeout, ConnectionError, HTTPError

from src.utils.proxy_manager import (
    ProxyRotationManager,
    RetryManager,
    ProxyInfo,
    ProxyStatus,
    RateLimitInfo
)
from src.utils.youtube_api import ProxyAwareTranscriptApi
from src.config import settings


class TestProxyInfo:
    """Test ProxyInfo dataclass functionality."""
    
    def test_proxy_info_creation(self):
        """Test creating ProxyInfo instance."""
        proxy = ProxyInfo(url="http://proxy.example.com:8080")
        
        assert proxy.url == "http://proxy.example.com:8080"
        assert proxy.status == ProxyStatus.UNKNOWN
        assert proxy.failure_count == 0
        assert proxy.last_used is None
        assert proxy.total_requests == 0
        assert proxy.successful_requests == 0
        assert proxy.consecutive_failures == 0
    
    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        proxy = ProxyInfo(url="http://proxy.example.com:8080")
        
        # No requests yet
        assert proxy.success_rate == 0.0
        
        # Add some requests
        proxy.total_requests = 10
        proxy.successful_requests = 8
        assert proxy.success_rate == 0.8
        
        # All successful
        proxy.successful_requests = 10
        assert proxy.success_rate == 1.0
    
    def test_is_healthy_property(self):
        """Test is_healthy property logic."""
        proxy = ProxyInfo(url="http://proxy.example.com:8080")
        
        # Default state should not be healthy (status is UNKNOWN)
        assert not proxy.is_healthy
        
        # Set to healthy
        proxy.status = ProxyStatus.HEALTHY
        assert proxy.is_healthy
        
        # Add failures but still under limit
        proxy.consecutive_failures = 2
        assert proxy.is_healthy
        
        # Exceed failure limit
        proxy.consecutive_failures = 5  # Default max is 3
        assert not proxy.is_healthy
    
    def test_parsed_url_property(self):
        """Test URL parsing functionality."""
        proxy = ProxyInfo(url="http://user:pass@proxy.example.com:8080")
        parsed = proxy.parsed_url
        
        assert parsed['scheme'] == 'http'
        assert parsed['hostname'] == 'proxy.example.com'
        assert parsed['port'] == 8080
        assert parsed['username'] == 'user'
        assert parsed['password'] == 'pass'


class TestRateLimitInfo:
    """Test RateLimitInfo functionality."""
    
    def test_rate_limit_creation(self):
        """Test creating RateLimitInfo instance."""
        rate_limit = RateLimitInfo()
        
        assert rate_limit.requests_made == 0
        assert rate_limit.last_request_time is None
        assert rate_limit.burst_count == 0
        assert rate_limit.burst_start_time is None
    
    def test_add_request(self):
        """Test adding request to rate limit tracking."""
        rate_limit = RateLimitInfo()
        
        # Add first request
        rate_limit.add_request()
        assert rate_limit.requests_made == 1
        assert rate_limit.last_request_time is not None
        assert rate_limit.burst_count == 1
        assert rate_limit.burst_start_time is not None
        
        # Add second request
        rate_limit.add_request()
        assert rate_limit.requests_made == 2
        assert rate_limit.burst_count == 2
    
    def test_requests_in_last_minute(self):
        """Test counting requests in the last minute."""
        rate_limit = RateLimitInfo()
        
        # No requests yet
        assert rate_limit.get_requests_in_last_minute() == 0
        
        # Add recent request
        rate_limit.add_request()
        assert rate_limit.get_requests_in_last_minute() == 1
        
        # Add old request (simulate by manually setting old time)
        old_time = datetime.utcnow() - timedelta(minutes=2)
        rate_limit.request_times.append(old_time)
        
        # Should still count only the recent one
        assert rate_limit.get_requests_in_last_minute() == 1
    
    def test_time_until_next_request(self):
        """Test calculating time until next request is allowed."""
        rate_limit = RateLimitInfo()
        
        # No last request, should allow immediately
        assert rate_limit.time_until_next_request() == 0.0
        
        # Add request, should need to wait
        rate_limit.add_request()
        wait_time = rate_limit.time_until_next_request()
        assert wait_time > 0
        assert wait_time <= 10  # Default min interval
    
    @patch('src.config.settings')
    def test_can_make_request(self, mock_settings):
        """Test request permission logic."""
        # Configure mock settings
        mock_settings.rate_limit_enabled = True
        mock_settings.rate_limit_min_interval = 10
        mock_settings.rate_limit_max_requests_per_minute = 6
        mock_settings.rate_limit_burst_requests = 3
        
        rate_limit = RateLimitInfo()
        
        # Should allow first request
        assert rate_limit.can_make_request()
        
        # Add request and check immediately
        rate_limit.add_request()
        # Should not allow immediately due to min interval
        assert not rate_limit.can_make_request()


class TestRetryManager:
    """Test RetryManager functionality."""
    
    def test_retry_manager_creation(self):
        """Test creating RetryManager instance."""
        retry_manager = RetryManager()
        assert retry_manager.config is not None
    
    def test_calculate_delay(self):
        """Test exponential backoff delay calculation."""
        retry_manager = RetryManager()
        
        # First attempt (0) should have base delay
        delay0 = retry_manager.calculate_delay(0)
        assert delay0 >= 1.0  # Base delay
        
        # Second attempt should be higher
        delay1 = retry_manager.calculate_delay(1)
        assert delay1 > delay0
        
        # Should respect max delay
        delay_high = retry_manager.calculate_delay(10)
        assert delay_high <= 60.0  # Default max delay
    
    def test_should_retry_logic(self):
        """Test retry decision logic."""
        retry_manager = RetryManager()
        
        # Should retry on network errors
        assert retry_manager.should_retry(0, ConnectionError("Network error"))
        assert retry_manager.should_retry(1, TimeoutError("Timeout"))
        
        # Should not retry beyond max attempts
        assert not retry_manager.should_retry(10, ConnectionError("Network error"))
        
        # Should not retry on non-retryable errors
        assert not retry_manager.should_retry(0, ValueError("Invalid value"))
    
    def test_retry_context_success(self):
        """Test retry context with successful operation."""
        retry_manager = RetryManager()
        
        attempts = []
        
        with retry_manager.retry_context("test_operation") as attempt:
            attempts.append(attempt)
            # Simulate success on first attempt
            pass
        
        assert len(attempts) == 1
        assert attempts[0] == 0
    
    def test_retry_context_with_retries(self):
        """Test retry context with failures and eventual success."""
        retry_manager = RetryManager()
        
        attempts = []
        
        with patch('time.sleep'):  # Speed up test
            with retry_manager.retry_context("test_operation") as attempt:
                attempts.append(attempt)
                if attempt < 2:  # Fail first two attempts
                    raise ConnectionError("Network error")
                # Success on third attempt
        
        assert len(attempts) == 3
        assert attempts == [0, 1, 2]
    
    def test_retry_context_final_failure(self):
        """Test retry context with ultimate failure."""
        retry_manager = RetryManager({'max_attempts': 2})
        
        attempts = []
        
        with patch('time.sleep'):  # Speed up test
            with pytest.raises(ConnectionError):
                with retry_manager.retry_context("test_operation") as attempt:
                    attempts.append(attempt)
                    raise ConnectionError("Persistent network error")
        
        assert len(attempts) == 2  # Should try max_attempts times


class TestProxyRotationManager:
    """Test ProxyRotationManager functionality."""
    
    def test_proxy_manager_creation_no_proxies(self):
        """Test creating proxy manager with no proxy configuration."""
        config = {'enabled': False, 'urls': []}
        
        with patch('src.utils.proxy_manager.settings') as mock_settings:
            mock_settings.proxy_config = config
            mock_settings.rate_limit_config = {'enabled': True}
            mock_settings.retry_config = {'enabled': True}
            
            manager = ProxyRotationManager(config)
            
            assert len(manager.proxies) == 0
            assert not manager.config.get('enabled', False)
    
    def test_proxy_manager_creation_with_proxies(self):
        """Test creating proxy manager with proxy configuration."""
        config = {
            'enabled': True,
            'urls': ['http://proxy1.example.com:8080', 'http://proxy2.example.com:8080'],
            'rotation_enabled': True
        }
        
        with patch('src.utils.proxy_manager.settings') as mock_settings:
            mock_settings.proxy_config = config
            mock_settings.rate_limit_config = {'enabled': True}
            mock_settings.retry_config = {'enabled': True}
            mock_settings.proxy_max_failures = 3
            
            with patch.object(ProxyRotationManager, '_start_health_check_thread'):
                manager = ProxyRotationManager(config)
                
                assert len(manager.proxies) == 2
                assert manager.proxies[0].url == 'http://proxy1.example.com:8080'
                assert manager.proxies[1].url == 'http://proxy2.example.com:8080'
    
    def test_get_current_proxy_no_proxies(self):
        """Test getting current proxy when none are configured."""
        config = {'enabled': False, 'urls': []}
        
        with patch('src.utils.proxy_manager.settings') as mock_settings:
            mock_settings.proxy_config = config
            mock_settings.rate_limit_config = {'enabled': True}
            mock_settings.retry_config = {'enabled': True}
            
            with patch.object(ProxyRotationManager, '_start_health_check_thread'):
                manager = ProxyRotationManager(config)
                assert manager.get_current_proxy() is None
    
    def test_get_current_proxy_with_healthy_proxies(self):
        """Test getting current proxy when healthy proxies are available."""
        config = {
            'enabled': True,
            'urls': ['http://proxy1.example.com:8080'],
            'rotation_enabled': True
        }
        
        with patch('src.utils.proxy_manager.settings') as mock_settings:
            mock_settings.proxy_config = config
            mock_settings.rate_limit_config = {'enabled': True}
            mock_settings.retry_config = {'enabled': True}
            mock_settings.proxy_max_failures = 3
            
            with patch.object(ProxyRotationManager, '_start_health_check_thread'):
                manager = ProxyRotationManager(config)
                
                # Mark proxy as healthy
                manager.proxies[0].status = ProxyStatus.HEALTHY
                
                proxy = manager.get_current_proxy()
                assert proxy is not None
                assert proxy.url == 'http://proxy1.example.com:8080'
    
    def test_get_current_proxy_no_healthy_proxies(self):
        """Test getting current proxy when no healthy proxies are available."""
        config = {
            'enabled': True,
            'urls': ['http://proxy1.example.com:8080'],
            'rotation_enabled': True
        }
        
        with patch('src.utils.proxy_manager.settings') as mock_settings:
            mock_settings.proxy_config = config
            mock_settings.rate_limit_config = {'enabled': True}
            mock_settings.retry_config = {'enabled': True}
            mock_settings.proxy_max_failures = 3
            
            with patch.object(ProxyRotationManager, '_start_health_check_thread'):
                manager = ProxyRotationManager(config)
                
                # Mark proxy as unhealthy
                manager.proxies[0].status = ProxyStatus.UNHEALTHY
                manager.proxies[0].consecutive_failures = 5
                
                proxy = manager.get_current_proxy()
                assert proxy is None
    
    def test_proxy_rotation(self):
        """Test proxy rotation functionality."""
        config = {
            'enabled': True,
            'urls': ['http://proxy1.example.com:8080', 'http://proxy2.example.com:8080'],
            'rotation_enabled': True
        }
        
        with patch('src.utils.proxy_manager.settings') as mock_settings:
            mock_settings.proxy_config = config
            mock_settings.rate_limit_config = {'enabled': True}
            mock_settings.retry_config = {'enabled': True}
            mock_settings.proxy_max_failures = 3
            
            with patch.object(ProxyRotationManager, '_start_health_check_thread'):
                manager = ProxyRotationManager(config)
                
                # Mark both proxies as healthy
                for proxy in manager.proxies:
                    proxy.status = ProxyStatus.HEALTHY
                
                # Get first proxy
                proxy1 = manager.get_current_proxy()
                assert proxy1.url == 'http://proxy1.example.com:8080'
                
                # Get second proxy (should rotate)
                proxy2 = manager.get_current_proxy()
                assert proxy2.url == 'http://proxy2.example.com:8080'
                
                # Get third proxy (should rotate back to first)
                proxy3 = manager.get_current_proxy()
                assert proxy3.url == 'http://proxy1.example.com:8080'
    
    @patch('src.utils.proxy_manager.requests.get')
    def test_proxy_health_check_success(self, mock_get, setup_proxy_manager):
        """Test successful proxy health check."""
        manager, config = setup_proxy_manager
        
        # Mock successful health check response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'origin': '1.2.3.4'}
        mock_get.return_value = mock_response
        
        proxy = manager.proxies[0]
        result = manager._check_proxy_health(proxy)
        
        assert result is True
        assert proxy.status == ProxyStatus.HEALTHY
        assert proxy.response_time is not None
        assert proxy.last_health_check is not None
    
    @patch('src.utils.proxy_manager.requests.get')
    def test_proxy_health_check_failure(self, mock_get, setup_proxy_manager):
        """Test failed proxy health check."""
        manager, config = setup_proxy_manager
        
        # Mock failed health check response
        mock_get.side_effect = ProxyError("Proxy connection failed")
        
        proxy = manager.proxies[0]
        result = manager._check_proxy_health(proxy)
        
        assert result is False
    
    @patch('src.utils.proxy_manager.requests.get')
    def test_proxy_health_check_timeout(self, mock_get, setup_proxy_manager):
        """Test proxy health check timeout."""
        manager, config = setup_proxy_manager
        
        # Mock timeout
        mock_get.side_effect = Timeout("Request timed out")
        
        proxy = manager.proxies[0]
        result = manager._check_proxy_health(proxy)
        
        assert result is False
    
    def test_rate_limiting(self):
        """Test rate limiting functionality."""
        config = {'enabled': False, 'urls': []}
        rate_config = {'enabled': True, 'min_interval': 1}
        
        with patch('src.utils.proxy_manager.settings') as mock_settings:
            mock_settings.proxy_config = config
            mock_settings.rate_limit_config = rate_config
            mock_settings.retry_config = {'enabled': True}
            
            with patch.object(ProxyRotationManager, '_start_health_check_thread'):
                manager = ProxyRotationManager(config)
                
                # First call should not wait
                start_time = time.time()
                manager.wait_for_rate_limit()
                elapsed = time.time() - start_time
                assert elapsed < 0.1  # Should be immediate
                
                # Second call should wait
                start_time = time.time()
                manager.wait_for_rate_limit()
                elapsed = time.time() - start_time
                assert elapsed >= 0.9  # Should wait close to 1 second
    
    def test_request_context_success(self, setup_proxy_manager):
        """Test successful request context."""
        manager, config = setup_proxy_manager
        
        with patch.object(manager, 'wait_for_rate_limit'):
            with patch.object(manager, 'get_current_proxy', return_value=None):
                with patch.object(manager, 'get_session') as mock_get_session:
                    mock_session = Mock()
                    mock_get_session.return_value = mock_session
                    
                    with manager.request_context() as (session, proxy):
                        assert session == mock_session
                        assert proxy is None
    
    def test_request_context_with_proxy_error(self, setup_proxy_manager):
        """Test request context with proxy error."""
        manager, config = setup_proxy_manager
        
        proxy = manager.proxies[0]
        proxy.status = ProxyStatus.HEALTHY
        
        with patch.object(manager, 'wait_for_rate_limit'):
            with patch.object(manager, 'get_current_proxy', return_value=proxy):
                with patch.object(manager, 'get_session') as mock_get_session:
                    mock_session = Mock()
                    mock_get_session.return_value = mock_session
                    
                    with pytest.raises(ProxyError):
                        with manager.request_context() as (session, proxy_info):
                            raise ProxyError("Proxy connection failed")
                    
                    # Check that failure statistics were updated
                    assert proxy.consecutive_failures > 0
                    assert proxy.failure_count > 0
    
    def test_statistics_gathering(self, setup_proxy_manager):
        """Test comprehensive statistics gathering."""
        manager, config = setup_proxy_manager
        
        # Add some statistics
        manager.stats['total_requests'] = 10
        manager.stats['successful_requests'] = 8
        manager.stats['failed_requests'] = 2
        
        stats = manager.get_statistics()
        
        assert stats['total_requests'] == 10
        assert stats['successful_requests'] == 8
        assert stats['failed_requests'] == 2
        assert stats['success_rate'] == 0.8
        assert 'uptime_seconds' in stats
        assert 'proxy_count' in stats
        assert 'proxy_details' in stats


class TestProxyAwareTranscriptApi:
    """Test ProxyAwareTranscriptApi functionality."""
    
    def test_proxy_api_creation(self):
        """Test creating ProxyAwareTranscriptApi instance."""
        with patch('src.utils.proxy_manager.get_proxy_manager'):
            with patch('src.utils.proxy_manager.get_retry_manager'):
                api = ProxyAwareTranscriptApi()
                assert api.proxy_manager is not None
                assert api.retry_manager is not None
    
    @patch('urllib.request.ProxyHandler')
    @patch('urllib.request.build_opener')
    @patch('urllib.request.install_opener')
    def test_patch_transcript_api(self, mock_install, mock_build, mock_handler):
        """Test patching transcript API with proxy."""
        proxy_info = ProxyInfo(url="http://proxy.example.com:8080")
        
        with patch('src.utils.proxy_manager.get_proxy_manager'):
            with patch('src.utils.proxy_manager.get_retry_manager'):
                api = ProxyAwareTranscriptApi()
                api._patch_transcript_api(proxy_info)
                
                # Verify proxy handler was created and installed
                mock_handler.assert_called_once()
                mock_build.assert_called_once()
                mock_install.assert_called_once()
    
    @patch('urllib.request.build_opener')
    @patch('urllib.request.install_opener')
    def test_restore_transcript_api(self, mock_install, mock_build):
        """Test restoring original transcript API behavior."""
        with patch('src.utils.proxy_manager.get_proxy_manager'):
            with patch('src.utils.proxy_manager.get_retry_manager'):
                api = ProxyAwareTranscriptApi()
                api._restore_transcript_api()
                
                # Verify opener was built and installed
                mock_build.assert_called_once()
                mock_install.assert_called_once()
    
    @patch('src.utils.youtube_api.YouTubeTranscriptApi.get_transcript')
    def test_get_transcript_success(self, mock_get_transcript):
        """Test successful transcript retrieval."""
        mock_get_transcript.return_value = [{'text': 'Hello', 'start': 0.0, 'duration': 2.0}]
        
        with patch('src.utils.proxy_manager.get_proxy_manager') as mock_proxy_mgr:
            with patch('src.utils.proxy_manager.get_retry_manager') as mock_retry_mgr:
                # Setup mocks
                mock_retry_context = Mock()
                mock_retry_context.__enter__ = Mock(return_value=None)
                mock_retry_context.__exit__ = Mock(return_value=None)
                mock_retry_mgr.return_value.retry_context.return_value = mock_retry_context
                
                mock_request_context = Mock()
                mock_request_context.__enter__ = Mock(return_value=(Mock(), None))
                mock_request_context.__exit__ = Mock(return_value=None)
                mock_proxy_mgr.return_value.request_context.return_value = mock_request_context
                
                api = ProxyAwareTranscriptApi()
                
                with patch.object(api, '_patch_transcript_api'):
                    with patch.object(api, '_restore_transcript_api'):
                        result = api.get_transcript("test_video_id")
                        
                        assert result == [{'text': 'Hello', 'start': 0.0, 'duration': 2.0}]
                        mock_get_transcript.assert_called_once_with("test_video_id")
    
    @patch('src.utils.youtube_api.YouTubeTranscriptApi.get_transcript')
    def test_get_transcript_with_languages(self, mock_get_transcript):
        """Test transcript retrieval with specific languages."""
        mock_get_transcript.return_value = [{'text': 'Hello', 'start': 0.0, 'duration': 2.0}]
        
        with patch('src.utils.proxy_manager.get_proxy_manager') as mock_proxy_mgr:
            with patch('src.utils.proxy_manager.get_retry_manager') as mock_retry_mgr:
                # Setup mocks
                mock_retry_context = Mock()
                mock_retry_context.__enter__ = Mock(return_value=None)
                mock_retry_context.__exit__ = Mock(return_value=None)
                mock_retry_mgr.return_value.retry_context.return_value = mock_retry_context
                
                mock_request_context = Mock()
                mock_request_context.__enter__ = Mock(return_value=(Mock(), None))
                mock_request_context.__exit__ = Mock(return_value=None)
                mock_proxy_mgr.return_value.request_context.return_value = mock_request_context
                
                api = ProxyAwareTranscriptApi()
                
                with patch.object(api, '_patch_transcript_api'):
                    with patch.object(api, '_restore_transcript_api'):
                        result = api.get_transcript("test_video_id", languages=['en', 'es'])
                        
                        assert result == [{'text': 'Hello', 'start': 0.0, 'duration': 2.0}]
                        mock_get_transcript.assert_called_once_with("test_video_id", languages=['en', 'es'])


# Fixtures
@pytest.fixture
def setup_proxy_manager():
    """Setup proxy manager for testing."""
    config = {
        'enabled': True,
        'urls': ['http://proxy1.example.com:8080'],
        'rotation_enabled': True,
        'health_check_interval': 300,
        'health_check_timeout': 10,
        'max_failures': 3
    }
    
    with patch('src.utils.proxy_manager.settings') as mock_settings:
        mock_settings.proxy_config = config
        mock_settings.rate_limit_config = {'enabled': True, 'min_interval': 1}
        mock_settings.retry_config = {'enabled': True, 'max_attempts': 3}
        mock_settings.proxy_max_failures = 3
        
        with patch.object(ProxyRotationManager, '_start_health_check_thread'):
            manager = ProxyRotationManager(config)
            yield manager, config


# Integration tests
class TestProxyIntegration:
    """Integration tests for proxy functionality."""
    
    def test_end_to_end_proxy_flow(self):
        """Test complete proxy flow from configuration to request."""
        config = {
            'enabled': True,
            'urls': ['http://proxy1.example.com:8080'],
            'rotation_enabled': True
        }
        
        with patch('src.utils.proxy_manager.settings') as mock_settings:
            mock_settings.proxy_config = config
            mock_settings.rate_limit_config = {'enabled': True, 'min_interval': 1}
            mock_settings.retry_config = {'enabled': True, 'max_attempts': 3}
            mock_settings.proxy_max_failures = 3
            
            with patch.object(ProxyRotationManager, '_start_health_check_thread'):
                with patch('src.utils.proxy_manager.requests.Session') as mock_session_class:
                    mock_session = Mock()
                    mock_session_class.return_value = mock_session
                    
                    manager = ProxyRotationManager(config)
                    
                    # Mark proxy as healthy
                    manager.proxies[0].status = ProxyStatus.HEALTHY
                    
                    # Test request context
                    with patch.object(manager, 'wait_for_rate_limit'):
                        with manager.request_context() as (session, proxy):
                            assert session == mock_session
                            assert proxy is not None
                            assert proxy.url == 'http://proxy1.example.com:8080'
    
    def test_failure_recovery_flow(self):
        """Test failure recovery and proxy switching."""
        config = {
            'enabled': True,
            'urls': ['http://proxy1.example.com:8080', 'http://proxy2.example.com:8080'],
            'rotation_enabled': True
        }
        
        with patch('src.utils.proxy_manager.settings') as mock_settings:
            mock_settings.proxy_config = config
            mock_settings.rate_limit_config = {'enabled': True}
            mock_settings.retry_config = {'enabled': True}
            mock_settings.proxy_max_failures = 2
            
            with patch.object(ProxyRotationManager, '_start_health_check_thread'):
                manager = ProxyRotationManager(config)
                
                # Mark both proxies as healthy
                for proxy in manager.proxies:
                    proxy.status = ProxyStatus.HEALTHY
                
                # Get first proxy and simulate failure
                proxy1 = manager.get_current_proxy()
                assert proxy1.url == 'http://proxy1.example.com:8080'
                
                # Simulate failures
                proxy1.consecutive_failures = 3
                manager._update_proxy_health(proxy1, False)
                
                # Should now get second proxy
                proxy2 = manager.get_current_proxy()
                assert proxy2.url == 'http://proxy2.example.com:8080'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])