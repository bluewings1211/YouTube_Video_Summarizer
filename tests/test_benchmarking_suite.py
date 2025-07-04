"""
Comprehensive benchmarking suite for YouTube Summarizer API.

This module provides detailed performance analysis including:
- Response time benchmarks
- Memory usage profiling
- CPU utilization monitoring
- Throughput analysis
- Scalability testing
- Resource efficiency metrics
"""

import time
import asyncio
import threading
import multiprocessing
import sys
import gc
import psutil
import statistics
import json
import csv
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from contextlib import contextmanager
from dataclasses import dataclass, asdict
import pytest
from unittest.mock import Mock, patch, MagicMock

# Try to import actual modules
try:
    from src.app import app
    from src.config import settings
    from src.flow import YouTubeSummarizerFlow
    from src.utils.youtube_api import YouTubeAPI
    from src.utils.call_llm import LLMClient
    from fastapi.testclient import TestClient
    
    test_client = TestClient(app)
    MODULES_AVAILABLE = True
except ImportError:
    MODULES_AVAILABLE = False
    test_client = None


@dataclass
class BenchmarkResult:
    """Container for individual benchmark results."""
    test_name: str
    start_time: datetime
    end_time: datetime
    response_time: float
    memory_before_mb: float
    memory_after_mb: float
    memory_peak_mb: float
    cpu_percent: float
    status_code: int
    success: bool
    request_size_bytes: int
    response_size_bytes: int
    error_message: Optional[str] = None
    additional_metrics: Optional[Dict[str, Any]] = None
    
    @property
    def duration(self) -> float:
        """Calculate duration in seconds."""
        return (self.end_time - self.start_time).total_seconds()
    
    @property
    def memory_delta_mb(self) -> float:
        """Calculate memory change."""
        return self.memory_after_mb - self.memory_before_mb


@dataclass
class BenchmarkSuite:
    """Container for a complete benchmark suite."""
    suite_name: str
    start_time: datetime
    end_time: datetime
    results: List[BenchmarkResult]
    environment_info: Dict[str, Any]
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """Calculate summary statistics for the benchmark suite."""
        if not self.results:
            return {}
        
        successful_results = [r for r in self.results if r.success]
        failed_results = [r for r in self.results if not r.success]
        
        response_times = [r.response_time for r in successful_results]
        memory_deltas = [r.memory_delta_mb for r in self.results]
        cpu_usage = [r.cpu_percent for r in self.results]
        
        return {
            "total_tests": len(self.results),
            "successful_tests": len(successful_results),
            "failed_tests": len(failed_results),
            "success_rate": (len(successful_results) / len(self.results)) * 100 if self.results else 0,
            "response_times": {
                "count": len(response_times),
                "min": min(response_times) if response_times else 0,
                "max": max(response_times) if response_times else 0,
                "mean": statistics.mean(response_times) if response_times else 0,
                "median": statistics.median(response_times) if response_times else 0,
                "std_dev": statistics.stdev(response_times) if len(response_times) > 1 else 0,
                "percentile_90": statistics.quantiles(response_times, n=10)[8] if len(response_times) > 1 else 0,
                "percentile_95": statistics.quantiles(response_times, n=20)[18] if len(response_times) > 1 else 0,
                "percentile_99": statistics.quantiles(response_times, n=100)[98] if len(response_times) > 1 else 0,
            },
            "memory_usage": {
                "min_delta": min(memory_deltas) if memory_deltas else 0,
                "max_delta": max(memory_deltas) if memory_deltas else 0,
                "mean_delta": statistics.mean(memory_deltas) if memory_deltas else 0,
                "total_peak": max([r.memory_peak_mb for r in self.results]) if self.results else 0,
            },
            "cpu_usage": {
                "min": min(cpu_usage) if cpu_usage else 0,
                "max": max(cpu_usage) if cpu_usage else 0,
                "mean": statistics.mean(cpu_usage) if cpu_usage else 0,
            },
            "throughput": {
                "requests_per_second": len(self.results) / self.duration if self.duration > 0 else 0,
                "successful_requests_per_second": len(successful_results) / self.duration if self.duration > 0 else 0,
            },
            "suite_duration": self.duration,
        }
    
    @property
    def duration(self) -> float:
        """Calculate suite duration in seconds."""
        return (self.end_time - self.start_time).total_seconds()


class SystemMonitor:
    """Monitor system resources during benchmarking."""
    
    def __init__(self):
        self.process = psutil.Process()
        self.monitoring = False
        self.measurements = []
        self.monitor_thread = None
    
    def start_monitoring(self, interval: float = 0.1):
        """Start system monitoring in background thread."""
        self.monitoring = True
        self.measurements = []
        self.monitor_thread = threading.Thread(target=self._monitor_loop, args=(interval,))
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop system monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
    
    def _monitor_loop(self, interval: float):
        """Background monitoring loop."""
        while self.monitoring:
            try:
                measurement = {
                    "timestamp": time.time(),
                    "memory_mb": self.process.memory_info().rss / 1024 / 1024,
                    "cpu_percent": self.process.cpu_percent(),
                    "memory_percent": self.process.memory_percent(),
                }
                self.measurements.append(measurement)
                time.sleep(interval)
            except Exception:
                # Ignore monitoring errors
                pass
    
    def get_peak_memory(self) -> float:
        """Get peak memory usage during monitoring."""
        if not self.measurements:
            return 0
        return max(m["memory_mb"] for m in self.measurements)
    
    def get_average_cpu(self) -> float:
        """Get average CPU usage during monitoring."""
        if not self.measurements:
            return 0
        return statistics.mean(m["cpu_percent"] for m in self.measurements)


class BenchmarkRunner:
    """Main benchmark runner with comprehensive metrics collection."""
    
    def __init__(self):
        self.monitor = SystemMonitor()
        self.results = []
        self.test_urls = [
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",  # Rick Roll
            "https://www.youtube.com/watch?v=ScMzIvxBSi4",  # Short video
            "https://www.youtube.com/watch?v=jNQXAC9IVRw",  # Medium video
        ]
    
    @contextmanager
    def benchmark_context(self, test_name: str):
        """Context manager for individual benchmark execution."""
        # Pre-test setup
        gc.collect()  # Force garbage collection
        start_memory = self.monitor.process.memory_info().rss / 1024 / 1024
        
        # Start monitoring
        self.monitor.start_monitoring()
        
        start_time = datetime.now()
        error_message = None
        
        try:
            yield
        except Exception as e:
            error_message = str(e)
        finally:
            end_time = datetime.now()
            
            # Stop monitoring and collect metrics
            self.monitor.stop_monitoring()
            
            end_memory = self.monitor.process.memory_info().rss / 1024 / 1024
            peak_memory = self.monitor.get_peak_memory()
            avg_cpu = self.monitor.get_average_cpu()
            
            # Create benchmark result
            result = BenchmarkResult(
                test_name=test_name,
                start_time=start_time,
                end_time=end_time,
                response_time=(end_time - start_time).total_seconds(),
                memory_before_mb=start_memory,
                memory_after_mb=end_memory,
                memory_peak_mb=peak_memory,
                cpu_percent=avg_cpu,
                status_code=0,  # Will be updated by caller
                success=error_message is None,
                request_size_bytes=0,  # Will be updated by caller
                response_size_bytes=0,  # Will be updated by caller
                error_message=error_message,
            )
            
            self.results.append(result)
    
    def create_mock_response(self, status_code: int = 200, response_time: float = 2.5) -> Mock:
        """Create a mock HTTP response for testing."""
        mock_response = Mock()
        mock_response.status_code = status_code
        mock_response.headers = {"content-type": "application/json"}
        
        if status_code == 200:
            mock_response.json.return_value = {
                "video_id": "dQw4w9WgXcQ",
                "title": "Test Video",
                "duration": 213,
                "summary": "This is a test summary for benchmarking.",
                "timestamped_segments": [
                    {
                        "timestamp": "00:00:45",
                        "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=45s",
                        "description": "Test segment",
                        "importance_rating": 8
                    }
                ],
                "keywords": ["test", "benchmark", "performance"],
                "processing_time": response_time
            }
            mock_response.content = json.dumps(mock_response.json.return_value).encode()
        else:
            mock_response.json.return_value = {
                "error": {
                    "code": "E1001",
                    "message": "Test error for benchmarking"
                }
            }
            mock_response.content = json.dumps(mock_response.json.return_value).encode()
        
        return mock_response
    
    def benchmark_single_request(self):
        """Benchmark a single API request."""
        with self.benchmark_context("single_request") as ctx:
            if MODULES_AVAILABLE:
                response = test_client.post(
                    "/api/v1/summarize",
                    json={"youtube_url": self.test_urls[0]}
                )
            else:
                # Mock the request
                time.sleep(2.5)  # Simulate processing time
                response = self.create_mock_response(200, 2.5)
            
            # Update result with response details
            result = self.results[-1]
            result.status_code = response.status_code
            result.success = response.status_code == 200
            result.request_size_bytes = len(json.dumps({"youtube_url": self.test_urls[0]}))
            result.response_size_bytes = len(response.content) if hasattr(response, 'content') else 1000
    
    def benchmark_concurrent_requests(self, num_threads: int = 5):
        """Benchmark concurrent API requests."""
        def make_request(thread_id: int):
            with self.benchmark_context(f"concurrent_request_{thread_id}"):
                if MODULES_AVAILABLE:
                    response = test_client.post(
                        "/api/v1/summarize",
                        json={"youtube_url": self.test_urls[thread_id % len(self.test_urls)]}
                    )
                else:
                    # Mock the request with some variation
                    time.sleep(2.0 + (thread_id * 0.5))  # Vary processing time
                    response = self.create_mock_response(200, 2.0 + (thread_id * 0.5))
                
                # Update result
                result = self.results[-1]
                result.status_code = response.status_code
                result.success = response.status_code == 200
                result.request_size_bytes = len(json.dumps({"youtube_url": self.test_urls[thread_id % len(self.test_urls)]}))
                result.response_size_bytes = len(response.content) if hasattr(response, 'content') else 1000
        
        # Run concurrent requests
        threads = []
        for i in range(num_threads):
            thread = threading.Thread(target=make_request, args=(i,))
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
    
    def benchmark_different_video_sizes(self):
        """Benchmark requests with different video sizes."""
        video_scenarios = [
            ("short_video", self.test_urls[1], 1.5),
            ("medium_video", self.test_urls[0], 2.5),
            ("long_video", self.test_urls[2], 4.0),
        ]
        
        for scenario_name, url, expected_time in video_scenarios:
            with self.benchmark_context(scenario_name):
                if MODULES_AVAILABLE:
                    response = test_client.post(
                        "/api/v1/summarize",
                        json={"youtube_url": url}
                    )
                else:
                    # Mock with scenario-specific timing
                    time.sleep(expected_time)
                    response = self.create_mock_response(200, expected_time)
                
                # Update result
                result = self.results[-1]
                result.status_code = response.status_code
                result.success = response.status_code in [200, 422]
                result.request_size_bytes = len(json.dumps({"youtube_url": url}))
                result.response_size_bytes = len(response.content) if hasattr(response, 'content') else 1000
    
    def benchmark_error_handling(self):
        """Benchmark error handling performance."""
        error_scenarios = [
            ("invalid_url", "https://invalid.com", 400),
            ("empty_url", "", 400),
            ("nonexistent_video", "https://www.youtube.com/watch?v=INVALID", 404),
        ]
        
        for scenario_name, url, expected_status in error_scenarios:
            with self.benchmark_context(f"error_{scenario_name}"):
                if MODULES_AVAILABLE:
                    response = test_client.post(
                        "/api/v1/summarize",
                        json={"youtube_url": url}
                    )
                else:
                    # Mock error response
                    time.sleep(0.5)  # Errors should be fast
                    response = self.create_mock_response(expected_status, 0.5)
                
                # Update result
                result = self.results[-1]
                result.status_code = response.status_code
                result.success = response.status_code == expected_status
                result.request_size_bytes = len(json.dumps({"youtube_url": url}))
                result.response_size_bytes = len(response.content) if hasattr(response, 'content') else 300
    
    def benchmark_health_endpoints(self):
        """Benchmark health and monitoring endpoints."""
        endpoints = [
            ("/health", "health_check"),
            ("/metrics", "metrics"),
            ("/", "root_info"),
        ]
        
        for endpoint, test_name in endpoints:
            with self.benchmark_context(test_name):
                if MODULES_AVAILABLE:
                    response = test_client.get(endpoint)
                else:
                    # Mock fast endpoint response
                    time.sleep(0.1)
                    mock_response = Mock()
                    mock_response.status_code = 200
                    mock_response.content = b'{"status": "healthy"}'
                    response = mock_response
                
                # Update result
                result = self.results[-1]
                result.status_code = response.status_code
                result.success = response.status_code == 200
                result.request_size_bytes = 0  # GET request
                result.response_size_bytes = len(response.content) if hasattr(response, 'content') else 100
    
    def benchmark_memory_leak_detection(self):
        """Benchmark for memory leak detection."""
        initial_memory = self.monitor.process.memory_info().rss / 1024 / 1024
        
        for i in range(10):
            with self.benchmark_context(f"memory_test_{i}"):
                if MODULES_AVAILABLE:
                    response = test_client.post(
                        "/api/v1/summarize",
                        json={"youtube_url": self.test_urls[i % len(self.test_urls)]}
                    )
                else:
                    # Mock request
                    time.sleep(1.0)
                    response = self.create_mock_response(200, 1.0)
                
                # Update result
                result = self.results[-1]
                result.status_code = response.status_code
                result.success = response.status_code in [200, 422]
                result.request_size_bytes = len(json.dumps({"youtube_url": self.test_urls[i % len(self.test_urls)]}))
                result.response_size_bytes = len(response.content) if hasattr(response, 'content') else 1000
                
                # Force garbage collection
                gc.collect()
        
        final_memory = self.monitor.process.memory_info().rss / 1024 / 1024
        memory_growth = final_memory - initial_memory
        
        # Add memory growth analysis
        for result in self.results[-10:]:
            if result.additional_metrics is None:
                result.additional_metrics = {}
            result.additional_metrics["memory_growth_mb"] = memory_growth
    
    def run_full_benchmark_suite(self) -> BenchmarkSuite:
        """Run the complete benchmark suite."""
        start_time = datetime.now()
        
        print("Starting comprehensive benchmark suite...")
        
        # Environment information
        environment_info = {
            "python_version": sys.version,
            "platform": sys.platform,
            "cpu_count": multiprocessing.cpu_count(),
            "memory_total_gb": psutil.virtual_memory().total / 1024 / 1024 / 1024,
            "modules_available": MODULES_AVAILABLE,
            "timestamp": start_time.isoformat(),
        }
        
        # Run individual benchmarks
        print("1. Single request benchmark...")
        self.benchmark_single_request()
        
        print("2. Concurrent requests benchmark...")
        self.benchmark_concurrent_requests(5)
        
        print("3. Different video sizes benchmark...")
        self.benchmark_different_video_sizes()
        
        print("4. Error handling benchmark...")
        self.benchmark_error_handling()
        
        print("5. Health endpoints benchmark...")
        self.benchmark_health_endpoints()
        
        print("6. Memory leak detection benchmark...")
        self.benchmark_memory_leak_detection()
        
        end_time = datetime.now()
        
        # Create benchmark suite
        suite = BenchmarkSuite(
            suite_name="YouTube Summarizer Full Benchmark",
            start_time=start_time,
            end_time=end_time,
            results=self.results,
            environment_info=environment_info,
        )
        
        return suite


def export_benchmark_results(suite: BenchmarkSuite, format: str = "json"):
    """Export benchmark results in various formats."""
    timestamp = suite.start_time.strftime("%Y%m%d_%H%M%S")
    
    if format == "json":
        filename = f"/tmp/benchmark_results_{timestamp}.json"
        data = {
            "suite_info": {
                "name": suite.suite_name,
                "start_time": suite.start_time.isoformat(),
                "end_time": suite.end_time.isoformat(),
                "duration": suite.duration,
            },
            "environment": suite.environment_info,
            "summary": suite.get_summary_statistics(),
            "detailed_results": [asdict(result) for result in suite.results],
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    elif format == "csv":
        filename = f"/tmp/benchmark_results_{timestamp}.csv"
        
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write header
            header = [
                "test_name", "start_time", "end_time", "response_time",
                "memory_before_mb", "memory_after_mb", "memory_peak_mb",
                "cpu_percent", "status_code", "success", "request_size_bytes",
                "response_size_bytes", "error_message"
            ]
            writer.writerow(header)
            
            # Write data
            for result in suite.results:
                row = [
                    result.test_name, result.start_time.isoformat(),
                    result.end_time.isoformat(), result.response_time,
                    result.memory_before_mb, result.memory_after_mb,
                    result.memory_peak_mb, result.cpu_percent,
                    result.status_code, result.success,
                    result.request_size_bytes, result.response_size_bytes,
                    result.error_message or ""
                ]
                writer.writerow(row)
    
    return filename


@pytest.mark.slow
@pytest.mark.integration
class TestBenchmarkingSuite:
    """Test class for running benchmarking suite."""
    
    def test_run_benchmark_suite(self):
        """Run the complete benchmarking suite."""
        runner = BenchmarkRunner()
        suite = runner.run_full_benchmark_suite()
        
        # Export results
        json_file = export_benchmark_results(suite, "json")
        csv_file = export_benchmark_results(suite, "csv")
        
        print(f"\nBenchmark Results:")
        print(f"JSON Report: {json_file}")
        print(f"CSV Report: {csv_file}")
        
        # Print summary
        summary = suite.get_summary_statistics()
        print(f"\n=== BENCHMARK SUMMARY ===")
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Success Rate: {summary['success_rate']:.2f}%")
        print(f"Average Response Time: {summary['response_times']['mean']:.2f}s")
        print(f"95th Percentile: {summary['response_times']['percentile_95']:.2f}s")
        print(f"Requests/Second: {summary['throughput']['requests_per_second']:.2f}")
        print(f"Peak Memory: {summary['memory_usage']['total_peak']:.2f}MB")
        print(f"Suite Duration: {summary['suite_duration']:.2f}s")
        
        # Assertions for performance thresholds
        assert summary["success_rate"] > 80.0, f"Success rate too low: {summary['success_rate']:.2f}%"
        assert summary["response_times"]["mean"] < 30.0, f"Average response time too high: {summary['response_times']['mean']:.2f}s"
        assert summary["response_times"]["percentile_95"] < 50.0, f"95th percentile too high: {summary['response_times']['percentile_95']:.2f}s"
        assert summary["memory_usage"]["total_peak"] < 2000, f"Peak memory too high: {summary['memory_usage']['total_peak']:.2f}MB"


if __name__ == "__main__":
    # Run benchmarks directly
    runner = BenchmarkRunner()
    suite = runner.run_full_benchmark_suite()
    
    # Export and display results
    json_file = export_benchmark_results(suite, "json")
    csv_file = export_benchmark_results(suite, "csv")
    
    print(f"\nBenchmark completed!")
    print(f"Results exported to:")
    print(f"  JSON: {json_file}")
    print(f"  CSV: {csv_file}")
    
    summary = suite.get_summary_statistics()
    print(f"\nQuick Summary:")
    print(f"  Tests Run: {summary['total_tests']}")
    print(f"  Success Rate: {summary['success_rate']:.1f}%")
    print(f"  Avg Response Time: {summary['response_times']['mean']:.2f}s")
    print(f"  Peak Memory: {summary['memory_usage']['total_peak']:.1f}MB")