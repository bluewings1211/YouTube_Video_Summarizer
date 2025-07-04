"""
Integration tests for end-to-end YouTube Summarizer workflow.

This module provides comprehensive integration testing for the complete 
video processing pipeline from API request to final response.
"""

import pytest
import unittest
from unittest.mock import Mock, patch, MagicMock
import json
import time
import os
import sys
from datetime import datetime
from typing import Dict, Any

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from fastapi.testclient import TestClient
from src.app import app


class TestEndToEndIntegration(unittest.TestCase):
    """Comprehensive end-to-end integration tests."""
    
    def setUp(self):
        """Set up test client and mock configurations."""
        self.client = TestClient(app)
        self.test_video_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        self.test_video_id = "dQw4w9WgXcQ"
        
        # Mock successful workflow response
        self.mock_successful_response = {
            'workflow_id': 'test_workflow_123',
            'status': 'success',
            'video_id': self.test_video_id,
            'video_title': 'Test Video Title',
            'video_duration': 180,
            'summary': 'This is a comprehensive summary of the test video content. The video discusses various topics and provides valuable insights for viewers.',
            'timestamps': [
                {
                    'timestamp': '00:30',
                    'description': 'Introduction and overview',
                    'importance': 'high',
                    'url': f'https://www.youtube.com/watch?v={self.test_video_id}&t=30s'
                },
                {
                    'timestamp': '01:45',
                    'description': 'Main content discussion',
                    'importance': 'high',
                    'url': f'https://www.youtube.com/watch?v={self.test_video_id}&t=105s'
                },
                {
                    'timestamp': '02:50',
                    'description': 'Conclusion and next steps',
                    'importance': 'medium',
                    'url': f'https://www.youtube.com/watch?v={self.test_video_id}&t=170s'
                }
            ],
            'keywords': ['tutorial', 'educational', 'example', 'demonstration', 'guide'],
            'processing_time': 25.7,
            'metadata': {
                'transcript_language': 'en',
                'transcript_length': 1250,
                'processing_start_time': datetime.utcnow().isoformat(),
                'llm_provider': 'openai',
                'model_used': 'gpt-4'
            }
        }
    
    @patch('src.app.workflow_instance')
    def test_complete_successful_workflow(self, mock_workflow):
        """Test complete successful end-to-end workflow."""
        # Configure mock workflow
        mock_workflow.run.return_value = self.mock_successful_response
        mock_workflow.__bool__ = lambda self: True
        
        # Send request
        request_data = {"youtube_url": self.test_video_url}
        response = self.client.post("/api/v1/summarize", json=request_data)
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        assert "workflow_id" in data
        assert "status" in data
        assert data["status"] == "success"
        assert "video_id" in data
        assert data["video_id"] == self.test_video_id
        assert "summary" in data
        assert "timestamps" in data
        assert "keywords" in data
        assert "processing_time" in data
        
        # Verify timestamps structure
        assert len(data["timestamps"]) == 3
        for timestamp in data["timestamps"]:
            assert "timestamp" in timestamp
            assert "description" in timestamp
            assert "importance" in timestamp
            assert "url" in timestamp
        
        # Verify keywords
        assert len(data["keywords"]) == 5
        assert all(isinstance(keyword, str) for keyword in data["keywords"])
        
        # Verify workflow was called correctly
        mock_workflow.run.assert_called_once()
        call_args = mock_workflow.run.call_args[0][0]
        assert call_args["youtube_url"] == self.test_video_url
    
    @patch('src.app.workflow_instance')
    def test_workflow_with_different_url_formats(self, mock_workflow):
        """Test workflow with various YouTube URL formats."""
        mock_workflow.run.return_value = self.mock_successful_response
        mock_workflow.__bool__ = lambda self: True
        
        # Test different URL formats
        url_formats = [
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://youtu.be/dQw4w9WgXcQ",
            "https://youtube.com/watch?v=dQw4w9WgXcQ",
            "https://m.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://www.youtube.com/embed/dQw4w9WgXcQ"
        ]
        
        for url in url_formats:
            with self.subTest(url=url):
                request_data = {"youtube_url": url}
                response = self.client.post("/api/v1/summarize", json=request_data)
                
                assert response.status_code == 200
                data = response.json()
                assert data["status"] == "success"
                assert data["video_id"] == self.test_video_id
    
    @patch('src.app.workflow_instance')
    def test_workflow_error_handling_scenarios(self, mock_workflow):
        """Test various error scenarios in the workflow."""
        mock_workflow.__bool__ = lambda self: True
        
        # Test different error scenarios
        error_scenarios = [
            {
                "error": Exception("Video is private"),
                "expected_status": 500,
                "test_name": "private_video"
            },
            {
                "error": Exception("Transcript not available"),
                "expected_status": 500,
                "test_name": "no_transcript"
            },
            {
                "error": Exception("Video too long"),
                "expected_status": 500,
                "test_name": "duration_limit"
            },
            {
                "error": TimeoutError("Processing timeout"),
                "expected_status": 500,
                "test_name": "timeout"
            }
        ]
        
        for scenario in error_scenarios:
            with self.subTest(scenario=scenario["test_name"]):
                mock_workflow.run.side_effect = scenario["error"]
                
                request_data = {"youtube_url": self.test_video_url}
                response = self.client.post("/api/v1/summarize", json=request_data)
                
                assert response.status_code == scenario["expected_status"]
                data = response.json()
                assert "error" in data
    
    def test_api_input_validation_integration(self):
        """Test API input validation integration."""
        # Test invalid URL formats
        invalid_urls = [
            "",
            "not_a_url",
            "https://example.com",
            "https://youtube.com/watch",
            "https://youtube.com/watch?v=",
            "invalid://youtube.com/watch?v=abc123"
        ]
        
        for invalid_url in invalid_urls:
            with self.subTest(url=invalid_url):
                request_data = {"youtube_url": invalid_url}
                response = self.client.post("/api/v1/summarize", json=request_data)
                
                assert response.status_code == 422
                data = response.json()
                assert "detail" in data
    
    def test_api_request_response_headers(self):
        """Test API request and response headers."""
        # Test with valid request
        request_data = {"youtube_url": self.test_video_url}
        
        with patch('src.app.workflow_instance') as mock_workflow:
            mock_workflow.run.return_value = self.mock_successful_response
            mock_workflow.__bool__ = lambda self: True
            
            response = self.client.post(
                "/api/v1/summarize", 
                json=request_data,
                headers={"Content-Type": "application/json"}
            )
            
            # Verify response headers
            assert response.status_code == 200
            assert "application/json" in response.headers.get("content-type", "")
    
    def test_health_check_integration(self):
        """Test health check endpoint integration."""
        response = self.client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify health response structure
        assert "status" in data
        assert "timestamp" in data
        assert "version" in data
        assert "workflow_ready" in data
        assert data["status"] == "healthy"
    
    def test_metrics_endpoint_integration(self):
        """Test metrics endpoint integration."""
        response = self.client.get("/metrics")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify metrics response structure
        assert "timestamp" in data
        assert "uptime_seconds" in data
        assert "version" in data
        assert "workflow_status" in data
        assert isinstance(data["uptime_seconds"], (int, float))
    
    def test_root_endpoint_integration(self):
        """Test root endpoint integration."""
        response = self.client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify root response structure
        assert "service" in data
        assert "version" in data
        assert "timestamp" in data
        assert data["service"] == "YouTube Summarizer"


class TestWorkflowComponentIntegration(unittest.TestCase):
    """Test integration between workflow components."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_url = "https://www.youtube.com/watch?v=test123"
        self.test_video_id = "test123"
    
    @patch('src.flow.YouTubeSummarizerFlow')
    def test_workflow_creation_integration(self, mock_flow_class):
        """Test workflow creation and configuration."""
        # Mock workflow instance
        mock_flow = Mock()
        mock_flow_class.return_value = mock_flow
        
        # Import and test workflow creation
        from src.flow import create_youtube_summarizer_flow
        
        workflow = create_youtube_summarizer_flow()
        assert workflow is not None
        mock_flow_class.assert_called_once()
    
    @patch('src.nodes.YouTubeTranscriptNode')
    @patch('src.nodes.SummarizationNode')
    @patch('src.nodes.TimestampNode')
    @patch('src.nodes.KeywordExtractionNode')
    def test_node_integration_chain(self, mock_keyword, mock_timestamp, mock_summary, mock_transcript):
        """Test integration between processing nodes."""
        # Setup mock nodes
        mock_nodes = [mock_transcript, mock_summary, mock_timestamp, mock_keyword]
        
        for i, mock_node in enumerate(mock_nodes):
            mock_instance = Mock()
            mock_instance.name = f"node_{i}"
            mock_instance.prep.return_value = {"prep": "success"}
            mock_instance.exec.return_value = {"exec": "success"}
            mock_instance.post.return_value = {"post": "success"}
            mock_node.return_value = mock_instance
        
        # Test node creation
        from src.nodes import YouTubeTranscriptNode, SummarizationNode, TimestampNode, KeywordExtractionNode
        
        nodes = [
            YouTubeTranscriptNode("transcript"),
            SummarizationNode("summary"),
            TimestampNode("timestamp"),
            KeywordExtractionNode("keyword")
        ]
        
        assert len(nodes) == 4
        for node in nodes:
            assert hasattr(node, 'prep')
            assert hasattr(node, 'exec')
            assert hasattr(node, 'post')
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'})
    def test_llm_client_integration(self):
        """Test LLM client integration."""
        from src.utils.call_llm import create_llm_client
        
        # Test client creation
        client = create_llm_client("openai")
        assert client is not None
        assert hasattr(client, 'call')
    
    def test_validator_integration(self):
        """Test URL validator integration."""
        from src.utils.validators import YouTubeURLValidator
        
        validator = YouTubeURLValidator()
        
        # Test valid URLs
        valid_urls = [
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://youtu.be/dQw4w9WgXcQ"
        ]
        
        for url in valid_urls:
            assert validator.validate_youtube_url(url) is True
            video_id = validator.extract_video_id(url)
            assert video_id is not None
    
    def test_error_message_integration(self):
        """Test error message system integration."""
        from src.utils.error_messages import ErrorMessageProvider
        
        provider = ErrorMessageProvider()
        
        # Test error message creation
        error = provider.create_validation_error("test_field", "invalid_value")
        assert error is not None
        assert isinstance(error, dict)


class TestPerformanceIntegration(unittest.TestCase):
    """Test performance aspects of integration."""
    
    def setUp(self):
        """Set up performance test environment."""
        self.client = TestClient(app)
    
    def test_api_response_time(self):
        """Test API response time performance."""
        start_time = time.time()
        response = self.client.get("/health")
        end_time = time.time()
        
        response_time = end_time - start_time
        
        assert response.status_code == 200
        assert response_time < 1.0  # Should respond within 1 second
    
    def test_concurrent_health_checks(self):
        """Test concurrent requests to health endpoint."""
        import concurrent.futures
        
        def make_request():
            return self.client.get("/health")
        
        # Make 5 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request) for _ in range(5)]
            responses = [future.result() for future in futures]
        
        # All requests should succeed
        for response in responses:
            assert response.status_code == 200
    
    @patch('src.app.workflow_instance')
    def test_workflow_timeout_handling(self, mock_workflow):
        """Test workflow timeout handling."""
        # Simulate slow workflow
        def slow_workflow(*args, **kwargs):
            time.sleep(0.1)  # Small delay for testing
            return {"status": "success", "video_id": "test123"}
        
        mock_workflow.run.side_effect = slow_workflow
        mock_workflow.__bool__ = lambda self: True
        
        request_data = {"youtube_url": "https://www.youtube.com/watch?v=test123"}
        
        start_time = time.time()
        response = self.client.post("/api/v1/summarize", json=request_data)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        # Should complete successfully but track timing
        assert response.status_code == 200
        assert processing_time < 5.0  # Should complete within reasonable time


if __name__ == '__main__':
    # Configure test environment
    os.environ.setdefault('TESTING', 'true')
    
    unittest.main(verbosity=2)