"""
Unit tests for workflow orchestration in flow.py.

This module provides comprehensive testing for the YouTube summarizer workflow
including configuration, error handling, fallback mechanisms, and performance monitoring.
"""

import unittest
import time
import unittest.mock as mock
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any, List
from datetime import datetime
import pytest

# Test imports
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from flow import (
    YouTubeSummarizerFlow,
    WorkflowConfig,
    NodeConfig,
    DataFlowConfig,
    FallbackStrategy,
    CircuitBreakerConfig,
    CircuitBreaker,
    CircuitBreakerState,
    NodeExecutionMode,
    WorkflowError,
    WorkflowMetrics,
    NodeMetrics,
    ErrorSeverity,
    create_youtube_summarizer_flow,
    create_custom_workflow_config,
    process_youtube_video
)


class TestWorkflowConfiguration(unittest.TestCase):
    """Test workflow configuration classes and factory functions."""
    
    def test_node_config_creation(self):
        """Test NodeConfig creation with default values."""
        config = NodeConfig(name="TestNode")
        
        self.assertEqual(config.name, "TestNode")
        self.assertEqual(config.enabled, True)
        self.assertEqual(config.required, True)
        self.assertEqual(config.max_retries, 3)
        self.assertEqual(config.retry_delay, 1.0)
        self.assertEqual(config.timeout_seconds, 60)
        self.assertEqual(config.dependencies, [])
        self.assertEqual(config.output_keys, [])
    
    def test_data_flow_config_creation(self):
        """Test DataFlowConfig creation with default values."""
        config = DataFlowConfig()
        
        assert config.input_mapping == {}
        assert config.output_mapping == {}
        assert config.data_validation == {}
        assert config.cleanup_keys == []
    
    def test_fallback_strategy_creation(self):
        """Test FallbackStrategy creation with default values."""
        strategy = FallbackStrategy()
        
        assert strategy.enable_transcript_only == True
        assert strategy.enable_summary_fallback == True
        assert strategy.enable_partial_results == True
        assert strategy.enable_retry_with_degraded == True
        assert strategy.max_fallback_attempts == 2
        assert strategy.fallback_timeout_factor == 0.5
    
    def test_circuit_breaker_config_creation(self):
        """Test CircuitBreakerConfig creation with default values."""
        config = CircuitBreakerConfig()
        
        assert config.failure_threshold == 3
        assert config.recovery_timeout == 60
        assert config.success_threshold == 2
        assert config.enabled == True
    
    def test_workflow_config_creation(self):
        """Test WorkflowConfig creation with all components."""
        config = WorkflowConfig()
        
        assert config.execution_mode == NodeExecutionMode.SEQUENTIAL
        assert isinstance(config.node_configs, dict)
        assert isinstance(config.data_flow_config, DataFlowConfig)
        assert isinstance(config.fallback_strategy, FallbackStrategy)
        assert isinstance(config.circuit_breaker_config, CircuitBreakerConfig)
        assert config.enable_monitoring == True
        assert config.enable_fallbacks == True
        assert config.max_retries == 2
        assert config.timeout_seconds == 300
        assert config.store_cleanup == True
    
    def test_create_custom_workflow_config(self):
        """Test custom workflow configuration creation."""
        node_configs = {
            'TestNode': {
                'enabled': False,
                'max_retries': 5
            }
        }
        
        data_flow_overrides = {
            'input_mapping': {'test_input': 'mapped_input'},
            'cleanup_keys': ['temp_data']
        }
        
        config = create_custom_workflow_config(
            node_configs=node_configs,
            data_flow_overrides=data_flow_overrides,
            enable_monitoring=False,
            max_retries=5
        )
        
        assert config.enable_monitoring == False
        assert config.max_retries == 5
        assert 'test_input' in config.data_flow_config.input_mapping
        assert 'temp_data' in config.data_flow_config.cleanup_keys


class TestCircuitBreaker:
    """Test circuit breaker functionality."""
    
    def test_circuit_breaker_initialization(self):
        """Test circuit breaker initial state."""
        config = CircuitBreakerConfig(failure_threshold=3, recovery_timeout=60)
        cb = CircuitBreaker(node_name="TestNode", config=config)
        
        assert cb.node_name == "TestNode"
        assert cb.state == CircuitBreakerState.CLOSED
        assert cb.failure_count == 0
        assert cb.success_count == 0
        assert cb.consecutive_failures == 0
        assert cb.can_execute() == True
    
    def test_circuit_breaker_failure_recording(self):
        """Test failure recording and state transitions."""
        config = CircuitBreakerConfig(failure_threshold=2, recovery_timeout=60)
        cb = CircuitBreaker(node_name="TestNode", config=config)
        
        # First failure
        cb.record_failure()
        assert cb.failure_count == 1
        assert cb.consecutive_failures == 1
        assert cb.state == CircuitBreakerState.CLOSED
        assert cb.can_execute() == True
        
        # Second failure - should open circuit
        cb.record_failure()
        assert cb.failure_count == 2
        assert cb.consecutive_failures == 2
        assert cb.state == CircuitBreakerState.OPEN
        assert cb.can_execute() == False
    
    def test_circuit_breaker_recovery(self):
        """Test circuit breaker recovery mechanism."""
        config = CircuitBreakerConfig(failure_threshold=1, recovery_timeout=0.1)
        cb = CircuitBreaker(node_name="TestNode", config=config)
        
        # Trigger failure to open circuit
        cb.record_failure()
        assert cb.state == CircuitBreakerState.OPEN
        assert cb.can_execute() == False
        
        # Wait for recovery timeout
        time.sleep(0.2)
        
        # Should transition to half-open
        assert cb.can_execute() == True
        assert cb.state == CircuitBreakerState.HALF_OPEN
        
        # Success should close circuit
        cb.record_success()
        cb.record_success()  # Need success_threshold successes
        assert cb.state == CircuitBreakerState.CLOSED
    
    def test_circuit_breaker_disabled(self):
        """Test circuit breaker when disabled."""
        config = CircuitBreakerConfig(enabled=False)
        cb = CircuitBreaker(node_name="TestNode", config=config)
        
        # Should always allow execution when disabled
        cb.record_failure()
        cb.record_failure()
        cb.record_failure()
        
        assert cb.can_execute() == True


class TestWorkflowMetrics:
    """Test workflow metrics and performance monitoring."""
    
    def test_node_metrics_initialization(self):
        """Test NodeMetrics initialization."""
        start_time = time.time()
        metrics = NodeMetrics(node_name="TestNode", start_time=start_time)
        
        assert metrics.node_name == "TestNode"
        assert metrics.start_time == start_time
        assert metrics.end_time is None
        assert metrics.duration == 0.0
        assert metrics.status == "pending"
        assert metrics.retry_count == 0
        assert metrics.error_count == 0
        assert metrics.fallback_used == False
        assert metrics.performance_score is None
    
    def test_workflow_metrics_initialization(self):
        """Test WorkflowMetrics initialization."""
        start_time = time.time()
        metrics = WorkflowMetrics(start_time=start_time)
        
        assert metrics.start_time == start_time
        assert metrics.end_time is None
        assert metrics.total_duration == 0.0
        assert isinstance(metrics.node_metrics, dict)
        assert metrics.success_rate == 0.0
        assert metrics.completion_rate == 0.0
        assert metrics.error_rate == 0.0
        assert metrics.fallback_usage_rate == 0.0
        assert metrics.throughput == 0.0
        assert metrics.efficiency_score == 0.0


class TestWorkflowExecution:
    """Test workflow execution and orchestration."""
    
    @pytest.fixture
    def mock_nodes(self):
        """Create mock nodes for testing."""
        transcript_node = Mock()
        transcript_node.name = "YouTubeTranscriptNode"
        transcript_node.prep.return_value = {'prep_status': 'success'}
        transcript_node.exec.return_value = {'exec_status': 'success', 'retry_count': 0}
        transcript_node.post.return_value = {'post_status': 'success'}
        
        summary_node = Mock()
        summary_node.name = "SummarizationNode"
        summary_node.prep.return_value = {'prep_status': 'success'}
        summary_node.exec.return_value = {'exec_status': 'success', 'retry_count': 0}
        summary_node.post.return_value = {'post_status': 'success'}
        
        return [transcript_node, summary_node]
    
    @pytest.fixture
    def mock_workflow(self, mock_nodes):
        """Create mock workflow for testing."""
        with patch('flow.YouTubeTranscriptNode'), \
             patch('flow.SummarizationNode'), \
             patch('flow.TimestampNode'), \
             patch('flow.KeywordExtractionNode'):
            
            workflow = YouTubeSummarizerFlow(enable_monitoring=True)
            workflow.nodes = mock_nodes
            return workflow
    
    def test_workflow_initialization(self):
        """Test workflow initialization."""
        with patch('flow.YouTubeTranscriptNode'), \
             patch('flow.SummarizationNode'), \
             patch('flow.TimestampNode'), \
             patch('flow.KeywordExtractionNode'):
            
            workflow = YouTubeSummarizerFlow()
            
            assert workflow.name == "YouTubeSummarizerFlow"
            assert workflow.enable_monitoring == True
            assert workflow.enable_fallbacks == True
            assert len(workflow.nodes) > 0
            assert isinstance(workflow.config, WorkflowConfig)
            assert isinstance(workflow.circuit_breakers, dict)
    
    def test_input_validation(self, mock_workflow):
        """Test input validation."""
        # Test missing youtube_url
        with pytest.raises(ValueError, match="Missing required 'youtube_url'"):
            mock_workflow._validate_input({})
        
        # Test empty youtube_url
        with pytest.raises(ValueError, match="youtube_url must be a non-empty string"):
            mock_workflow._validate_input({'youtube_url': ''})
        
        # Test invalid input type
        with pytest.raises(ValueError, match="Input data must be a dictionary"):
            mock_workflow._validate_input("invalid")
        
        # Test valid input
        mock_workflow._validate_input({'youtube_url': 'https://youtube.com/watch?v=test'})
    
    def test_successful_workflow_execution(self, mock_workflow, mock_nodes):
        """Test successful workflow execution."""
        input_data = {'youtube_url': 'https://youtube.com/watch?v=test'}
        
        # Mock store updates
        mock_workflow.store = Mock()
        mock_workflow._execute_workflow_with_timeout = Mock(return_value={
            'YouTubeTranscriptNode': {'status': 'success'},
            'SummarizationNode': {'status': 'success'}
        })
        
        result = mock_workflow.run(input_data)
        
        assert result['status'] == 'success'
        assert 'data' in result
        assert 'metadata' in result
        assert 'performance' in result
    
    def test_workflow_with_node_failure(self, mock_workflow, mock_nodes):
        """Test workflow behavior with node failures."""
        # Configure one node to fail
        mock_nodes[1].post.return_value = {'post_status': 'failed', 'error': 'Test failure'}
        
        # Mark the failing node as optional
        mock_workflow.config.node_configs['SummarizationNode'] = NodeConfig(
            name='SummarizationNode',
            required=False
        )
        
        input_data = {'youtube_url': 'https://youtube.com/watch?v=test'}
        
        with patch.object(mock_workflow, '_execute_workflow_with_timeout') as mock_execute:
            mock_execute.return_value = {
                'YouTubeTranscriptNode': {'status': 'success'},
                'SummarizationNode': {'status': 'failed'}
            }
            
            result = mock_workflow.run(input_data)
            
            # Should still succeed with partial results
            assert result['status'] == 'success'
    
    def test_workflow_timeout(self, mock_workflow):
        """Test workflow timeout handling."""
        mock_workflow.timeout_seconds = 0.1  # Very short timeout
        
        with patch.object(mock_workflow, '_execute_nodes_sequence') as mock_execute:
            mock_execute.side_effect = lambda: time.sleep(0.2)  # Longer than timeout
            
            with pytest.raises(RuntimeError, match="Workflow failed after all retries"):
                mock_workflow._execute_workflow_with_timeout()


class TestErrorHandling:
    """Test error handling and fallback mechanisms."""
    
    def test_error_severity_classification(self):
        """Test error severity classification."""
        workflow = YouTubeSummarizerFlow()
        
        # Test critical errors
        memory_error = MemoryError("Out of memory")
        severity = workflow._classify_error_severity(memory_error, "TestNode")
        assert severity == ErrorSeverity.CRITICAL
        
        # Test high severity errors
        connection_error = ConnectionError("Connection failed")
        severity = workflow._classify_error_severity(connection_error, "TestNode")
        assert severity == ErrorSeverity.HIGH
        
        # Test medium severity for transcript node
        value_error = ValueError("Invalid input")
        severity = workflow._classify_error_severity(value_error, "YouTubeTranscriptNode")
        assert severity == ErrorSeverity.MEDIUM
    
    def test_fallback_decision_logic(self):
        """Test fallback decision logic."""
        config = WorkflowConfig()
        config.fallback_strategy.enable_summary_fallback = True
        
        workflow = YouTubeSummarizerFlow(config=config)
        
        # Create test error
        test_error = WorkflowError(
            flow_name="test",
            error_type="ValueError",
            message="Test error"
        )
        
        # Test fallback for SummarizationNode
        should_fallback = workflow._should_attempt_fallback(test_error, "SummarizationNode")
        assert should_fallback == True
        
        # Test no fallback for disabled strategy
        workflow.config.fallback_strategy.enable_summary_fallback = False
        should_fallback = workflow._should_attempt_fallback(test_error, "SummarizationNode")
        assert should_fallback == False
    
    def test_workflow_error_creation(self):
        """Test enhanced workflow error creation."""
        workflow = YouTubeSummarizerFlow()
        test_exception = ValueError("Test error message")
        
        error_info = workflow._create_enhanced_workflow_error(
            test_exception, "TestNode", "execution"
        )
        
        assert error_info.flow_name == "YouTubeSummarizerFlow"
        assert error_info.error_type == "ValueError"
        assert error_info.message == "Test error message"
        assert error_info.failed_node == "TestNode"
        assert error_info.node_phase == "execution"
        assert error_info.recovery_action is not None
        assert 'severity' in error_info.context


class TestPerformanceMonitoring:
    """Test performance monitoring and metrics."""
    
    def test_node_performance_scoring(self):
        """Test node performance score calculation."""
        workflow = YouTubeSummarizerFlow()
        
        # Test successful node with fast execution
        node_metrics = NodeMetrics(
            node_name="YouTubeTranscriptNode",
            start_time=time.time(),
            duration=20.0,  # Faster than expected 30s
            status="success",
            retry_count=0,
            fallback_used=False
        )
        
        score = workflow._calculate_node_performance_score(node_metrics)
        assert score > 100  # Should get speed bonus
        
        # Test failed node
        node_metrics.status = "failed"
        score = workflow._calculate_node_performance_score(node_metrics)
        assert score == 0.0
        
        # Test node with retries
        node_metrics.status = "success"
        node_metrics.retry_count = 2
        score = workflow._calculate_node_performance_score(node_metrics)
        assert score < 100  # Should have retry penalty
    
    def test_workflow_metrics_calculation(self):
        """Test workflow metrics calculation."""
        workflow = YouTubeSummarizerFlow(enable_monitoring=True)
        workflow.metrics = WorkflowMetrics(start_time=time.time())
        
        # Add some mock node metrics
        workflow.metrics.node_metrics = {
            'Node1': NodeMetrics(
                node_name='Node1',
                start_time=time.time(),
                status='success',
                performance_score=90.0
            ),
            'Node2': NodeMetrics(
                node_name='Node2', 
                start_time=time.time(),
                status='failed',
                performance_score=0.0
            )
        }
        
        workflow.config.node_configs = {
            'Node1': NodeConfig(name='Node1'),
            'Node2': NodeConfig(name='Node2')
        }
        
        workflow._calculate_workflow_metrics()
        
        assert workflow.metrics.completion_rate == 100.0  # Both nodes completed
        assert workflow.metrics.success_rate == 50.0  # 1 of 2 succeeded
        assert workflow.metrics.error_rate == 50.0  # 1 of 2 failed
        assert workflow.metrics.efficiency_score == 45.0  # Average of 90 and 0
    
    def test_real_time_status_monitoring(self):
        """Test real-time status monitoring."""
        workflow = YouTubeSummarizerFlow(enable_monitoring=True)
        workflow.metrics = WorkflowMetrics(start_time=time.time())
        
        # Add a running node
        workflow.metrics.node_metrics['TestNode'] = NodeMetrics(
            node_name='TestNode',
            start_time=time.time(),
            status='running'
        )
        
        status = workflow.get_workflow_status()
        
        assert status['status'] == 'running'
        assert 'progress' in status
        assert 'performance' in status
        assert 'nodes' in status
        assert 'timing' in status
        assert status['progress']['current_node'] == 'TestNode'


class TestFactoryFunctions:
    """Test factory functions and convenience methods."""
    
    def test_create_youtube_summarizer_flow_legacy(self):
        """Test legacy configuration creation."""
        with patch('flow.YouTubeTranscriptNode'), \
             patch('flow.SummarizationNode'), \
             patch('flow.TimestampNode'), \
             patch('flow.KeywordExtractionNode'):
            
            config = {
                'enable_monitoring': False,
                'max_retries': 5,
                'timeout_seconds': 600
            }
            
            flow = create_youtube_summarizer_flow(config)
            
            assert flow.enable_monitoring == False
            assert flow.max_retries == 5
            assert flow.timeout_seconds == 600
    
    def test_create_youtube_summarizer_flow_with_workflow_config(self):
        """Test creation with WorkflowConfig object."""
        with patch('flow.YouTubeTranscriptNode'), \
             patch('flow.SummarizationNode'), \
             patch('flow.TimestampNode'), \
             patch('flow.KeywordExtractionNode'):
            
            config = WorkflowConfig(
                enable_monitoring=False,
                max_retries=5
            )
            
            flow = create_youtube_summarizer_flow(config)
            
            assert flow.enable_monitoring == False
            assert flow.max_retries == 5
    
    def test_process_youtube_video_convenience_function(self):
        """Test convenience function for processing videos."""
        with patch('flow.create_youtube_summarizer_flow') as mock_create:
            mock_flow = Mock()
            mock_flow.run.return_value = {'status': 'success'}
            mock_create.return_value = mock_flow
            
            result = process_youtube_video('https://youtube.com/watch?v=test')
            
            assert result['status'] == 'success'
            mock_create.assert_called_once()
            mock_flow.run.assert_called_once()


class TestIntegrationScenarios:
    """Test complex integration scenarios."""
    
    def test_workflow_with_circuit_breaker_trips(self):
        """Test workflow behavior when circuit breakers trip."""
        config = WorkflowConfig()
        config.circuit_breaker_config.failure_threshold = 1
        
        with patch('flow.YouTubeTranscriptNode'), \
             patch('flow.SummarizationNode'), \
             patch('flow.TimestampNode'), \
             patch('flow.KeywordExtractionNode'):
            
            workflow = YouTubeSummarizerFlow(config=config)
            
            # Simulate circuit breaker trip
            cb = workflow.circuit_breakers.get('SummarizationNode')
            if cb:
                cb.record_failure()  # Should open circuit
                
                # Check that circuit is open
                assert cb.state == CircuitBreakerState.OPEN
                assert not cb.can_execute()
    
    def test_workflow_with_fallback_execution(self):
        """Test workflow execution with fallback mechanisms."""
        workflow = YouTubeSummarizerFlow()
        
        # Test summary fallback
        fallback_result = workflow._fallback_summary_simple(Mock())
        
        # Should return None without transcript data
        assert fallback_result is None
        
        # Add transcript data to store
        workflow.store = {
            'transcript_data': {
                'transcript_text': 'This is a test transcript with enough words to create a summary.'
            }
        }
        
        fallback_result = workflow._fallback_summary_simple(Mock())
        
        # Should return fallback result with transcript data
        assert fallback_result is not None
        assert fallback_result['exec']['fallback_used'] == True
    
    def test_complete_workflow_integration(self):
        """Test complete workflow integration with all features."""
        with patch('flow.YouTubeTranscriptNode') as mock_transcript, \
             patch('flow.SummarizationNode') as mock_summary, \
             patch('flow.TimestampNode') as mock_timestamp, \
             patch('flow.KeywordExtractionNode') as mock_keyword:
            
            # Configure mock nodes
            mock_nodes = [mock_transcript, mock_summary, mock_timestamp, mock_keyword]
            for i, mock_node in enumerate(mock_nodes):
                mock_node.return_value.name = f"Node{i}"
                mock_node.return_value.prep.return_value = {'prep_status': 'success'}
                mock_node.return_value.exec.return_value = {'exec_status': 'success', 'retry_count': 0}
                mock_node.return_value.post.return_value = {'post_status': 'success'}
            
            # Create workflow with monitoring enabled
            config = WorkflowConfig(enable_monitoring=True)
            workflow = YouTubeSummarizerFlow(config=config)
            
            # Mock the _execute_workflow_with_timeout to avoid actual execution
            with patch.object(workflow, '_execute_workflow_with_timeout') as mock_execute:
                mock_execute.return_value = {
                    'Node0': {'status': 'success'},
                    'Node1': {'status': 'success'},
                    'Node2': {'status': 'success'},
                    'Node3': {'status': 'success'}
                }
                
                # Mock store data for result preparation
                workflow.store = {
                    'transcript_data': {'video_id': 'test', 'transcript_text': 'test'},
                    'summary_data': {'summary_text': 'test summary'},
                    'timestamp_data': {'timestamps': []},
                    'keyword_data': {'keywords': []},
                    'video_metadata': {'title': 'Test Video', 'duration_seconds': 300}
                }
                
                result = workflow.run({'youtube_url': 'https://youtube.com/watch?v=test'})
                
                # Verify comprehensive result structure
                assert result['status'] == 'success'
                assert 'data' in result
                assert 'metadata' in result
                assert 'performance' in result
                assert 'node_performance' in result
                
                # Verify performance metrics
                assert 'efficiency_score' in result['performance']
                assert 'total_duration' in result['performance']
                assert 'success_rate' in result['performance']


if __name__ == '__main__':
    pytest.main([__file__])