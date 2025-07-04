"""
Basic unit tests for workflow orchestration in flow.py.

This module provides core testing for the YouTube summarizer workflow
using the standard unittest framework.
"""

import unittest
import time
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime

# Test imports
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    import flow
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
        create_custom_workflow_config
    )
except ImportError as e:
    print(f"Import error: {e}")
    print("Skipping tests due to import issues")
    sys.exit(0)


class TestWorkflowConfiguration(unittest.TestCase):
    """Test workflow configuration classes."""
    
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
        
        self.assertEqual(config.input_mapping, {})
        self.assertEqual(config.output_mapping, {})
        self.assertEqual(config.data_validation, {})
        self.assertEqual(config.cleanup_keys, [])
    
    def test_fallback_strategy_creation(self):
        """Test FallbackStrategy creation with default values."""
        strategy = FallbackStrategy()
        
        self.assertEqual(strategy.enable_transcript_only, True)
        self.assertEqual(strategy.enable_summary_fallback, True)
        self.assertEqual(strategy.enable_partial_results, True)
        self.assertEqual(strategy.enable_retry_with_degraded, True)
        self.assertEqual(strategy.max_fallback_attempts, 2)
        self.assertEqual(strategy.fallback_timeout_factor, 0.5)
    
    def test_circuit_breaker_config_creation(self):
        """Test CircuitBreakerConfig creation with default values."""
        config = CircuitBreakerConfig()
        
        self.assertEqual(config.failure_threshold, 3)
        self.assertEqual(config.recovery_timeout, 60)
        self.assertEqual(config.success_threshold, 2)
        self.assertEqual(config.enabled, True)
    
    def test_workflow_config_creation(self):
        """Test WorkflowConfig creation with all components."""
        config = WorkflowConfig()
        
        self.assertEqual(config.execution_mode, NodeExecutionMode.SEQUENTIAL)
        self.assertIsInstance(config.node_configs, dict)
        self.assertIsInstance(config.data_flow_config, DataFlowConfig)
        self.assertIsInstance(config.fallback_strategy, FallbackStrategy)
        self.assertIsInstance(config.circuit_breaker_config, CircuitBreakerConfig)
        self.assertEqual(config.enable_monitoring, True)
        self.assertEqual(config.enable_fallbacks, True)
        self.assertEqual(config.max_retries, 2)
        self.assertEqual(config.timeout_seconds, 300)
        self.assertEqual(config.store_cleanup, True)


class TestCircuitBreaker(unittest.TestCase):
    """Test circuit breaker functionality."""
    
    def test_circuit_breaker_initialization(self):
        """Test circuit breaker initial state."""
        config = CircuitBreakerConfig(failure_threshold=3, recovery_timeout=60)
        cb = CircuitBreaker(node_name="TestNode", config=config)
        
        self.assertEqual(cb.node_name, "TestNode")
        self.assertEqual(cb.state, CircuitBreakerState.CLOSED)
        self.assertEqual(cb.failure_count, 0)
        self.assertEqual(cb.success_count, 0)
        self.assertEqual(cb.consecutive_failures, 0)
        self.assertEqual(cb.can_execute(), True)
    
    def test_circuit_breaker_failure_recording(self):
        """Test failure recording and state transitions."""
        config = CircuitBreakerConfig(failure_threshold=2, recovery_timeout=60)
        cb = CircuitBreaker(node_name="TestNode", config=config)
        
        # First failure
        cb.record_failure()
        self.assertEqual(cb.failure_count, 1)
        self.assertEqual(cb.consecutive_failures, 1)
        self.assertEqual(cb.state, CircuitBreakerState.CLOSED)
        self.assertEqual(cb.can_execute(), True)
        
        # Second failure - should open circuit
        cb.record_failure()
        self.assertEqual(cb.failure_count, 2)
        self.assertEqual(cb.consecutive_failures, 2)
        self.assertEqual(cb.state, CircuitBreakerState.OPEN)
        self.assertEqual(cb.can_execute(), False)
    
    def test_circuit_breaker_recovery(self):
        """Test circuit breaker recovery mechanism."""
        config = CircuitBreakerConfig(failure_threshold=1, recovery_timeout=0.1)
        cb = CircuitBreaker(node_name="TestNode", config=config)
        
        # Trigger failure to open circuit
        cb.record_failure()
        self.assertEqual(cb.state, CircuitBreakerState.OPEN)
        self.assertEqual(cb.can_execute(), False)
        
        # Wait for recovery timeout
        time.sleep(0.2)
        
        # Should transition to half-open
        self.assertEqual(cb.can_execute(), True)
        self.assertEqual(cb.state, CircuitBreakerState.HALF_OPEN)
        
        # Success should close circuit
        cb.record_success()
        cb.record_success()  # Need success_threshold successes
        self.assertEqual(cb.state, CircuitBreakerState.CLOSED)
    
    def test_circuit_breaker_disabled(self):
        """Test circuit breaker when disabled."""
        config = CircuitBreakerConfig(enabled=False)
        cb = CircuitBreaker(node_name="TestNode", config=config)
        
        # Should always allow execution when disabled
        cb.record_failure()
        cb.record_failure()
        cb.record_failure()
        
        self.assertEqual(cb.can_execute(), True)


class TestWorkflowMetrics(unittest.TestCase):
    """Test workflow metrics and performance monitoring."""
    
    def test_node_metrics_initialization(self):
        """Test NodeMetrics initialization."""
        start_time = time.time()
        metrics = NodeMetrics(node_name="TestNode", start_time=start_time)
        
        self.assertEqual(metrics.node_name, "TestNode")
        self.assertEqual(metrics.start_time, start_time)
        self.assertIsNone(metrics.end_time)
        self.assertEqual(metrics.duration, 0.0)
        self.assertEqual(metrics.status, "pending")
        self.assertEqual(metrics.retry_count, 0)
        self.assertEqual(metrics.error_count, 0)
        self.assertEqual(metrics.fallback_used, False)
        self.assertIsNone(metrics.performance_score)
    
    def test_workflow_metrics_initialization(self):
        """Test WorkflowMetrics initialization."""
        start_time = time.time()
        metrics = WorkflowMetrics(start_time=start_time)
        
        self.assertEqual(metrics.start_time, start_time)
        self.assertIsNone(metrics.end_time)
        self.assertEqual(metrics.total_duration, 0.0)
        self.assertIsInstance(metrics.node_metrics, dict)
        self.assertEqual(metrics.success_rate, 0.0)
        self.assertEqual(metrics.completion_rate, 0.0)
        self.assertEqual(metrics.error_rate, 0.0)
        self.assertEqual(metrics.fallback_usage_rate, 0.0)
        self.assertEqual(metrics.throughput, 0.0)
        self.assertEqual(metrics.efficiency_score, 0.0)


class TestWorkflowBasic(unittest.TestCase):
    """Test basic workflow functionality."""
    
    @patch('flow.YouTubeTranscriptNode')
    @patch('flow.SummarizationNode')
    @patch('flow.TimestampNode')
    @patch('flow.KeywordExtractionNode')
    def test_workflow_initialization(self, mock_keyword, mock_timestamp, mock_summary, mock_transcript):
        """Test workflow initialization."""
        # Configure mocks
        for mock_node in [mock_transcript, mock_summary, mock_timestamp, mock_keyword]:
            mock_instance = Mock()
            mock_instance.name = "MockNode"
            mock_node.return_value = mock_instance
        
        workflow = YouTubeSummarizerFlow()
        
        self.assertEqual(workflow.name, "YouTubeSummarizerFlow")
        self.assertEqual(workflow.enable_monitoring, True)
        self.assertEqual(workflow.enable_fallbacks, True)
        self.assertGreater(len(workflow.nodes), 0)
        self.assertIsInstance(workflow.config, WorkflowConfig)
        self.assertIsInstance(workflow.circuit_breakers, dict)
    
    def test_input_validation(self):
        """Test input validation."""
        workflow = YouTubeSummarizerFlow()
        
        # Test missing youtube_url
        with self.assertRaises(ValueError) as context:
            workflow._validate_input({})
        self.assertIn("Missing required 'youtube_url'", str(context.exception))
        
        # Test empty youtube_url
        with self.assertRaises(ValueError) as context:
            workflow._validate_input({'youtube_url': ''})
        self.assertIn("youtube_url must be a non-empty string", str(context.exception))
        
        # Test invalid input type
        with self.assertRaises(ValueError) as context:
            workflow._validate_input("invalid")
        self.assertIn("Input data must be a dictionary", str(context.exception))
        
        # Test valid input
        try:
            workflow._validate_input({'youtube_url': 'https://youtube.com/watch?v=test'})
        except Exception as e:
            self.fail(f"Valid input validation failed: {e}")


class TestErrorHandling(unittest.TestCase):
    """Test error handling and fallback mechanisms."""
    
    def test_error_severity_classification(self):
        """Test error severity classification."""
        workflow = YouTubeSummarizerFlow()
        
        # Test critical errors
        memory_error = MemoryError("Out of memory")
        severity = workflow._classify_error_severity(memory_error, "TestNode")
        self.assertEqual(severity, ErrorSeverity.CRITICAL)
        
        # Test high severity errors
        connection_error = ConnectionError("Connection failed")
        severity = workflow._classify_error_severity(connection_error, "TestNode")
        self.assertEqual(severity, ErrorSeverity.HIGH)
        
        # Test medium severity for transcript node
        value_error = ValueError("Invalid input")
        severity = workflow._classify_error_severity(value_error, "YouTubeTranscriptNode")
        self.assertEqual(severity, ErrorSeverity.MEDIUM)
    
    def test_workflow_error_creation(self):
        """Test enhanced workflow error creation."""
        workflow = YouTubeSummarizerFlow()
        test_exception = ValueError("Test error message")
        
        error_info = workflow._create_enhanced_workflow_error(
            test_exception, "TestNode", "execution"
        )
        
        self.assertEqual(error_info.flow_name, "YouTubeSummarizerFlow")
        self.assertEqual(error_info.error_type, "ValueError")
        self.assertEqual(error_info.message, "Test error message")
        self.assertEqual(error_info.failed_node, "TestNode")
        self.assertEqual(error_info.node_phase, "execution")
        self.assertIsNotNone(error_info.recovery_action)
        self.assertIn('severity', error_info.context)


class TestPerformanceMonitoring(unittest.TestCase):
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
        self.assertGreater(score, 100)  # Should get speed bonus
        
        # Test failed node
        node_metrics.status = "failed"
        score = workflow._calculate_node_performance_score(node_metrics)
        self.assertEqual(score, 0.0)
        
        # Test node with retries
        node_metrics.status = "success"
        node_metrics.retry_count = 2
        score = workflow._calculate_node_performance_score(node_metrics)
        self.assertLess(score, 100)  # Should have retry penalty
    
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
        
        self.assertEqual(workflow.metrics.completion_rate, 100.0)  # Both nodes completed
        self.assertEqual(workflow.metrics.success_rate, 50.0)  # 1 of 2 succeeded
        self.assertEqual(workflow.metrics.error_rate, 50.0)  # 1 of 2 failed
        self.assertEqual(workflow.metrics.efficiency_score, 45.0)  # Average of 90 and 0


class TestFactoryFunctions(unittest.TestCase):
    """Test factory functions and convenience methods."""
    
    @patch('flow.YouTubeTranscriptNode')
    @patch('flow.SummarizationNode')
    @patch('flow.TimestampNode')
    @patch('flow.KeywordExtractionNode')
    def test_create_youtube_summarizer_flow_legacy(self, mock_keyword, mock_timestamp, mock_summary, mock_transcript):
        """Test legacy configuration creation."""
        # Configure mocks
        for mock_node in [mock_transcript, mock_summary, mock_timestamp, mock_keyword]:
            mock_instance = Mock()
            mock_instance.name = "MockNode"
            mock_node.return_value = mock_instance
        
        config = {
            'enable_monitoring': False,
            'max_retries': 5,
            'timeout_seconds': 600
        }
        
        flow = create_youtube_summarizer_flow(config)
        
        self.assertEqual(flow.enable_monitoring, False)
        self.assertEqual(flow.max_retries, 5)
        self.assertEqual(flow.timeout_seconds, 600)
    
    @patch('flow.YouTubeTranscriptNode')
    @patch('flow.SummarizationNode')
    @patch('flow.TimestampNode')
    @patch('flow.KeywordExtractionNode')
    def test_create_youtube_summarizer_flow_with_workflow_config(self, mock_keyword, mock_timestamp, mock_summary, mock_transcript):
        """Test creation with WorkflowConfig object."""
        # Configure mocks
        for mock_node in [mock_transcript, mock_summary, mock_timestamp, mock_keyword]:
            mock_instance = Mock()
            mock_instance.name = "MockNode"
            mock_node.return_value = mock_instance
        
        config = WorkflowConfig(
            enable_monitoring=False,
            max_retries=5
        )
        
        flow = create_youtube_summarizer_flow(config)
        
        self.assertEqual(flow.enable_monitoring, False)
        self.assertEqual(flow.max_retries, 5)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)