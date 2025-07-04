"""
Comprehensive API integration tests for YouTube Summarizer Web Service.

Tests all API endpoints, error handling, request validation, and response formatting.
"""

import pytest
import json
import time
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient
from fastapi import status

# Import the app - handle import issues gracefully
try:
    from src.app import app, workflow_instance
    from src.flow import WorkflowError
except ImportError:
    pytest.skip("Cannot import app module - dependencies not installed", allow_module_level=True)


class TestAPIEndpoints:
    """Test suite for API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    @pytest.fixture
    def mock_workflow_success(self):
        """Mock successful workflow execution."""
        return {
            'status': 'success',
            'data': {
                'video_id': 'test123',
                'title': 'Test Video Title',
                'duration': 1800,
                'summary': 'This is a test summary of the video content. ' * 20,  # ~500 words
                'timestamps': [
                    {
                        'timestamp': '01:30',
                        'description': 'Introduction to main topic',
                        'importance_rating': 8
                    },
                    {
                        'timestamp': '05:45',
                        'description': 'Key concepts explained',
                        'importance_rating': 9
                    }
                ],
                'keywords': ['test', 'video', 'content', 'analysis', 'summary']
            },
            'metadata': {
                'processing_time': 2.5
            }
        }
    
    @pytest.fixture
    def mock_workflow_instance(self, mock_workflow_success):
        """Mock workflow instance."""
        mock_instance = Mock()
        mock_instance.run.return_value = mock_workflow_success
        return mock_instance

    def test_root_endpoint(self, client):
        """Test root endpoint returns service information."""
        response = client.get("/")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "service" in data
        assert "version" in data
        assert "description" in data
        assert "endpoints" in data
        assert "timestamp" in data
        
        # Check endpoints are listed
        endpoints = data["endpoints"]
        assert "/api/v1/summarize" in endpoints["summarize"]
        assert "/health" in endpoints["health"]
        assert "/api/docs" in endpoints["docs"]

    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data
        assert "workflow_ready" in data

    def test_metrics_endpoint(self, client):
        """Test metrics endpoint."""
        response = client.get("/metrics")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "uptime_seconds" in data
        assert "timestamp" in data
        assert "version" in data
        assert "workflow_status" in data

    @patch('src.app.workflow_instance')
    def test_summarize_endpoint_success(self, mock_workflow, client, mock_workflow_success):
        """Test successful video summarization."""
        # Setup mock
        mock_workflow.run.return_value = mock_workflow_success
        mock_workflow.__bool__ = lambda self: True  # Make workflow_instance truthy
        
        # Test request
        request_data = {
            "youtube_url": "https://www.youtube.com/watch?v=test123"
        }
        
        response = client.post("/api/v1/summarize", json=request_data)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Check response structure
        assert data["video_id"] == "test123"
        assert data["title"] == "Test Video Title"
        assert data["duration"] == 1800
        assert len(data["summary"]) > 0
        assert len(data["timestamped_segments"]) == 2
        assert len(data["keywords"]) == 5
        assert data["processing_time"] > 0
        
        # Check timestamped segments structure
        segment = data["timestamped_segments"][0]
        assert segment["timestamp"] == "01:30"
        assert "youtube.com" in segment["url"]
        assert "t=90s" in segment["url"]
        assert segment["description"] == "Introduction to main topic"
        assert segment["importance_rating"] == 8

    def test_summarize_endpoint_invalid_url(self, client):
        """Test validation for invalid YouTube URLs."""
        invalid_urls = [
            "",
            "not_a_url",
            "https://example.com",
            "https://vimeo.com/123456",
            "https://facebook.com/video/123",
        ]
        
        for url in invalid_urls:
            request_data = {"youtube_url": url}
            response = client.post("/api/v1/summarize", json=request_data)
            
            assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_summarize_endpoint_missing_url(self, client):
        """Test validation for missing YouTube URL."""
        response = client.post("/api/v1/summarize", json={})
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_summarize_endpoint_malformed_json(self, client):
        """Test handling of malformed JSON requests."""
        response = client.post(
            "/api/v1/summarize",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @patch('src.app.workflow_instance', None)
    def test_summarize_endpoint_workflow_unavailable(self, client):
        """Test handling when workflow instance is not available."""
        request_data = {
            "youtube_url": "https://www.youtube.com/watch?v=test123"
        }
        
        response = client.post("/api/v1/summarize", json=request_data)
        
        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        data = response.json()
        assert "workflow engine not ready" in data["detail"].lower()

    @patch('src.app.workflow_instance')
    def test_summarize_endpoint_workflow_error_private_video(self, mock_workflow, client):
        """Test handling of private video errors."""
        mock_workflow.run.side_effect = Exception("Private video cannot be processed")
        mock_workflow.__bool__ = lambda self: True
        
        request_data = {
            "youtube_url": "https://www.youtube.com/watch?v=private123"
        }
        
        response = client.post("/api/v1/summarize", json=request_data)
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        data = response.json()
        assert "private" in data["detail"].lower()

    @patch('src.app.workflow_instance')
    def test_summarize_endpoint_workflow_error_live_stream(self, mock_workflow, client):
        """Test handling of live stream errors."""
        mock_workflow.run.side_effect = Exception("Live stream not supported")
        mock_workflow.__bool__ = lambda self: True
        
        request_data = {
            "youtube_url": "https://www.youtube.com/watch?v=live123"
        }
        
        response = client.post("/api/v1/summarize", json=request_data)
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        data = response.json()
        assert "live" in data["detail"].lower()

    @patch('src.app.workflow_instance')
    def test_summarize_endpoint_workflow_error_no_transcript(self, mock_workflow, client):
        """Test handling of missing transcript errors."""
        mock_workflow.run.side_effect = Exception("Transcript not available")
        mock_workflow.__bool__ = lambda self: True
        
        request_data = {
            "youtube_url": "https://www.youtube.com/watch?v=notranscript123"
        }
        
        response = client.post("/api/v1/summarize", json=request_data)
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        data = response.json()
        assert "transcript" in data["detail"].lower()

    @patch('src.app.workflow_instance')
    def test_summarize_endpoint_workflow_error_duration_limit(self, mock_workflow, client):
        """Test handling of video duration limit errors."""
        mock_workflow.run.side_effect = Exception("Video duration exceeds limit")
        mock_workflow.__bool__ = lambda self: True
        
        request_data = {
            "youtube_url": "https://www.youtube.com/watch?v=toolong123"
        }
        
        response = client.post("/api/v1/summarize", json=request_data)
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        data = response.json()
        assert "duration" in data["detail"].lower()

    @patch('src.app.workflow_instance')
    def test_summarize_endpoint_workflow_failed_status(self, mock_workflow, client):
        """Test handling of workflow failed status."""
        mock_workflow.run.return_value = {
            'status': 'failed',
            'error': {
                'type': 'ProcessingError',
                'message': 'Processing failed due to internal error'
            }
        }
        mock_workflow.__bool__ = lambda self: True
        
        request_data = {
            "youtube_url": "https://www.youtube.com/watch?v=failed123"
        }
        
        response = client.post("/api/v1/summarize", json=request_data)
        
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        data = response.json()
        assert "internal error" in data["detail"].lower()

    @patch('src.app.workflow_instance')
    def test_summarize_endpoint_timeout_error(self, mock_workflow, client):
        """Test handling of timeout errors."""
        mock_workflow.run.return_value = {
            'status': 'failed',
            'error': {
                'type': 'TimeoutError',
                'message': 'Request timeout occurred'
            }
        }
        mock_workflow.__bool__ = lambda self: True
        
        request_data = {
            "youtube_url": "https://www.youtube.com/watch?v=timeout123"
        }
        
        response = client.post("/api/v1/summarize", json=request_data)
        
        assert response.status_code == status.HTTP_408_REQUEST_TIMEOUT

    def test_request_response_headers(self, client):
        """Test that proper headers are set on responses."""
        response = client.get("/health")
        
        # Check that middleware headers are present
        assert "X-Process-Time" in response.headers
        assert "X-Request-ID" in response.headers
        
        # Validate header values
        process_time = float(response.headers["X-Process-Time"])
        assert process_time >= 0
        
        request_id = response.headers["X-Request-ID"]
        assert request_id.startswith("req_")

    def test_cors_headers(self, client):
        """Test CORS headers are properly set."""
        response = client.options("/api/v1/summarize")
        
        # CORS should allow the request
        assert response.status_code in [status.HTTP_200_OK, status.HTTP_405_METHOD_NOT_ALLOWED]

    @patch('src.app.workflow_instance')
    def test_response_performance_tracking(self, mock_workflow, client, mock_workflow_success):
        """Test that response includes performance metrics."""
        mock_workflow.run.return_value = mock_workflow_success
        mock_workflow.__bool__ = lambda self: True
        
        request_data = {
            "youtube_url": "https://www.youtube.com/watch?v=perf123"
        }
        
        start_time = time.time()
        response = client.post("/api/v1/summarize", json=request_data)
        end_time = time.time()
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Check processing time is reasonable
        processing_time = data["processing_time"]
        assert 0 <= processing_time <= (end_time - start_time) + 1  # Allow some margin


class TestUtilityFunctions:
    """Test utility functions in the app module."""
    
    def test_convert_timestamp_to_seconds(self):
        """Test timestamp conversion utility."""
        # Import the function
        from src.app import convert_timestamp_to_seconds
        
        # Test MM:SS format
        assert convert_timestamp_to_seconds("01:30") == 90
        assert convert_timestamp_to_seconds("00:45") == 45
        assert convert_timestamp_to_seconds("10:00") == 600
        
        # Test HH:MM:SS format
        assert convert_timestamp_to_seconds("01:30:45") == 5445
        assert convert_timestamp_to_seconds("00:01:30") == 90
        assert convert_timestamp_to_seconds("02:00:00") == 7200
        
        # Test edge cases
        assert convert_timestamp_to_seconds("") == 0
        assert convert_timestamp_to_seconds("invalid") == 0
        assert convert_timestamp_to_seconds("1:2:3:4") == 0
        assert convert_timestamp_to_seconds(None) == 0


class TestErrorHandling:
    """Test error handling and exception scenarios."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    def test_404_not_found(self, client):
        """Test 404 handling for non-existent endpoints."""
        response = client.get("/nonexistent")
        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_405_method_not_allowed(self, client):
        """Test 405 handling for wrong HTTP methods."""
        response = client.get("/api/v1/summarize")
        assert response.status_code == status.HTTP_405_METHOD_NOT_ALLOWED

    @patch('src.app.workflow_instance')
    def test_unexpected_exception_handling(self, mock_workflow, client):
        """Test handling of unexpected exceptions."""
        mock_workflow.run.side_effect = RuntimeError("Unexpected error")
        mock_workflow.__bool__ = lambda self: True
        
        request_data = {
            "youtube_url": "https://www.youtube.com/watch?v=error123"
        }
        
        response = client.post("/api/v1/summarize", json=request_data)
        
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        data = response.json()
        assert "internal processing error" in data["detail"].lower()


class TestValidation:
    """Test input validation and data formatting."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    def test_valid_youtube_urls(self, client):
        """Test various valid YouTube URL formats."""
        valid_urls = [
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://youtube.com/watch?v=dQw4w9WgXcQ",
            "https://youtu.be/dQw4w9WgXcQ",
            "https://www.youtube.com/embed/dQw4w9WgXcQ",
            "https://www.youtube.com/v/dQw4w9WgXcQ",
            "https://www.youtube.com/shorts/dQw4w9WgXcQ",
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=90s"
        ]
        
        for url in valid_urls:
            request_data = {"youtube_url": url}
            response = client.post("/api/v1/summarize", json=request_data)
            
            # Should pass validation (but may fail at workflow level if workflow not mocked)
            assert response.status_code != status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_url_length_limit(self, client):
        """Test URL length validation."""
        # Very long URL should be rejected
        long_url = "https://www.youtube.com/watch?v=" + "a" * 1000
        request_data = {"youtube_url": long_url}
        response = client.post("/api/v1/summarize", json=request_data)
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_empty_and_whitespace_urls(self, client):
        """Test empty and whitespace-only URLs."""
        invalid_urls = ["", "   ", "\t\n"]
        
        for url in invalid_urls:
            request_data = {"youtube_url": url}
            response = client.post("/api/v1/summarize", json=request_data)
            
            assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


# Integration test configuration
@pytest.fixture(scope="session")
def test_settings():
    """Test configuration settings."""
    return {
        "test_timeout": 30,
        "mock_responses": True,
        "enable_logging": True
    }


if __name__ == "__main__":
    # Run tests when executed directly
    pytest.main([__file__, "-v"])