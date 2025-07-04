"""
Load testing for YouTube Summarizer API using locust framework.

This module provides comprehensive load testing scenarios that can be used
to stress test the API under various load conditions.
"""

import time
import random
import json
from typing import Dict, List, Any
from datetime import datetime

try:
    from locust import HttpUser, task, between, events
    from locust.runners import MasterRunner, WorkerRunner
    LOCUST_AVAILABLE = True
except ImportError:
    # Fallback when locust is not available
    LOCUST_AVAILABLE = False
    
    # Create mock classes for when locust is not available
    class HttpUser:
        def __init__(self):
            self.client = None
            
        def between(self, min_wait, max_wait):
            return lambda: random.uniform(min_wait, max_wait)
    
    def task(func):
        return func
    
    def between(min_wait, max_wait):
        return lambda: random.uniform(min_wait, max_wait)


# Test data - various YouTube URLs for testing
TEST_URLS = [
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ",  # Rick Roll - 3:33
    "https://www.youtube.com/watch?v=ScMzIvxBSi4",  # Short video
    "https://www.youtube.com/watch?v=jNQXAC9IVRw",  # Medium video
    "https://youtu.be/dQw4w9WgXcQ",                # Short URL format
    "https://www.youtube.com/watch?v=9bZkp7q19f0",  # Long video
]

# Invalid URLs for error testing
INVALID_URLS = [
    "https://invalid-url.com/video",
    "https://www.youtube.com/watch?v=NONEXISTENT",
    "not-a-url",
    "",
]


class APIPerformanceMetrics:
    """Track API performance metrics during load testing."""
    
    def __init__(self):
        self.response_times = []
        self.error_count = 0
        self.success_count = 0
        self.status_codes = {}
        self.start_time = time.time()
        
    def record_response(self, response_time: float, status_code: int, success: bool):
        """Record a response for metrics tracking."""
        self.response_times.append(response_time)
        
        if success:
            self.success_count += 1
        else:
            self.error_count += 1
            
        self.status_codes[status_code] = self.status_codes.get(status_code, 0) + 1
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        total_requests = self.success_count + self.error_count
        elapsed_time = time.time() - self.start_time
        
        return {
            "total_requests": total_requests,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "success_rate": (self.success_count / total_requests * 100) if total_requests > 0 else 0,
            "requests_per_second": total_requests / elapsed_time if elapsed_time > 0 else 0,
            "average_response_time": sum(self.response_times) / len(self.response_times) if self.response_times else 0,
            "min_response_time": min(self.response_times) if self.response_times else 0,
            "max_response_time": max(self.response_times) if self.response_times else 0,
            "status_codes": self.status_codes,
            "elapsed_time": elapsed_time,
        }


# Global metrics instance
metrics = APIPerformanceMetrics()


class YouTubeSummarizerUser(HttpUser):
    """Simulates a user interacting with the YouTube Summarizer API."""
    
    wait_time = between(1, 5)  # Wait 1-5 seconds between requests
    host = "http://localhost:8000"
    
    def on_start(self):
        """Called when a user starts."""
        # Optional: Perform user initialization
        self.client.verify = False  # Disable SSL verification for testing
        
    @task(10)  # Weight: 10 (most common task)
    def summarize_video(self):
        """Test video summarization with valid URLs."""
        url = random.choice(TEST_URLS)
        payload = {"youtube_url": url}
        
        start_time = time.time()
        
        with self.client.post(
            "/api/v1/summarize",
            json=payload,
            headers={"Content-Type": "application/json"},
            catch_response=True
        ) as response:
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                # Success - check response structure
                try:
                    data = response.json()
                    required_fields = ["video_id", "title", "summary", "keywords", "timestamped_segments"]
                    
                    if all(field in data for field in required_fields):
                        response.success()
                        metrics.record_response(response_time, response.status_code, True)
                    else:
                        response.failure(f"Missing required fields in response")
                        metrics.record_response(response_time, response.status_code, False)
                        
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
                    metrics.record_response(response_time, response.status_code, False)
                    
            elif response.status_code in [400, 404, 422]:
                # Expected error responses
                response.success()  # These are valid responses for invalid inputs
                metrics.record_response(response_time, response.status_code, True)
                
            else:
                # Unexpected error
                response.failure(f"Unexpected status code: {response.status_code}")
                metrics.record_response(response_time, response.status_code, False)
    
    @task(2)  # Weight: 2 (less common)
    def test_invalid_urls(self):
        """Test API with invalid URLs to verify error handling."""
        url = random.choice(INVALID_URLS)
        payload = {"youtube_url": url}
        
        start_time = time.time()
        
        with self.client.post(
            "/api/v1/summarize",
            json=payload,
            headers={"Content-Type": "application/json"},
            catch_response=True
        ) as response:
            response_time = time.time() - start_time
            
            if response.status_code in [400, 422]:
                # Expected error for invalid URL
                try:
                    data = response.json()
                    if "error" in data:
                        response.success()
                        metrics.record_response(response_time, response.status_code, True)
                    else:
                        response.failure("Error response missing error field")
                        metrics.record_response(response_time, response.status_code, False)
                except json.JSONDecodeError:
                    response.failure("Invalid JSON in error response")
                    metrics.record_response(response_time, response.status_code, False)
            else:
                response.failure(f"Expected 400/422 for invalid URL, got {response.status_code}")
                metrics.record_response(response_time, response.status_code, False)
    
    @task(5)  # Weight: 5 (moderate frequency)
    def health_check(self):
        """Test health check endpoint."""
        start_time = time.time()
        
        with self.client.get("/health", catch_response=True) as response:
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    if "status" in data and data["status"] == "healthy":
                        response.success()
                        metrics.record_response(response_time, response.status_code, True)
                    else:
                        response.failure("Unhealthy status in health check")
                        metrics.record_response(response_time, response.status_code, False)
                except json.JSONDecodeError:
                    response.failure("Invalid JSON in health check response")
                    metrics.record_response(response_time, response.status_code, False)
            else:
                response.failure(f"Health check failed with status {response.status_code}")
                metrics.record_response(response_time, response.status_code, False)
    
    @task(1)  # Weight: 1 (least common)
    def get_metrics(self):
        """Test metrics endpoint."""
        start_time = time.time()
        
        with self.client.get("/metrics", catch_response=True) as response:
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    if "version" in data and "uptime_seconds" in data:
                        response.success()
                        metrics.record_response(response_time, response.status_code, True)
                    else:
                        response.failure("Missing expected fields in metrics")
                        metrics.record_response(response_time, response.status_code, False)
                except json.JSONDecodeError:
                    response.failure("Invalid JSON in metrics response")
                    metrics.record_response(response_time, response.status_code, False)
            else:
                response.failure(f"Metrics endpoint failed with status {response.status_code}")
                metrics.record_response(response_time, response.status_code, False)
    
    def on_stop(self):
        """Called when a user stops."""
        # Optional: Perform cleanup
        pass


class HeavyLoadUser(HttpUser):
    """Simulates heavy load with more aggressive request patterns."""
    
    wait_time = between(0.5, 2)  # Shorter wait times for heavier load
    host = "http://localhost:8000"
    
    @task
    def rapid_fire_requests(self):
        """Make rapid requests to stress test the API."""
        for _ in range(3):  # Make 3 requests in quick succession
            url = random.choice(TEST_URLS)
            payload = {"youtube_url": url}
            
            start_time = time.time()
            
            with self.client.post(
                "/api/v1/summarize",
                json=payload,
                headers={"Content-Type": "application/json"},
                catch_response=True
            ) as response:
                response_time = time.time() - start_time
                
                success = response.status_code in [200, 400, 404, 422]
                metrics.record_response(response_time, response.status_code, success)
                
                if not success:
                    response.failure(f"Unexpected status: {response.status_code}")
                else:
                    response.success()
            
            time.sleep(0.1)  # Very brief pause between requests


class ErrorTestingUser(HttpUser):
    """Specifically tests error conditions and edge cases."""
    
    wait_time = between(2, 4)
    host = "http://localhost:8000"
    
    @task
    def test_malformed_requests(self):
        """Test various malformed requests."""
        malformed_payloads = [
            {},  # Empty payload
            {"wrong_field": "value"},  # Wrong field name
            {"youtube_url": None},  # Null value
            {"youtube_url": 123},  # Wrong type
            {"youtube_url": ""},  # Empty string
        ]
        
        payload = random.choice(malformed_payloads)
        
        start_time = time.time()
        
        with self.client.post(
            "/api/v1/summarize",
            json=payload,
            headers={"Content-Type": "application/json"},
            catch_response=True
        ) as response:
            response_time = time.time() - start_time
            
            # Should return 400 or 422 for malformed requests
            if response.status_code in [400, 422]:
                response.success()
                metrics.record_response(response_time, response.status_code, True)
            else:
                response.failure(f"Expected 400/422 for malformed request, got {response.status_code}")
                metrics.record_response(response_time, response.status_code, False)


# Event handlers for logging and reporting
if LOCUST_AVAILABLE:
    @events.test_start.add_listener
    def on_test_start(environment, **kwargs):
        """Called when the test starts."""
        print(f"Load test starting at {datetime.now()}")
        print(f"Host: {environment.host}")
        
    @events.test_stop.add_listener
    def on_test_stop(environment, **kwargs):
        """Called when the test stops."""
        print(f"\nLoad test completed at {datetime.now()}")
        
        # Print performance summary
        summary = metrics.get_summary()
        print("\n=== PERFORMANCE SUMMARY ===")
        print(f"Total Requests: {summary['total_requests']}")
        print(f"Success Rate: {summary['success_rate']:.2f}%")
        print(f"Requests/Second: {summary['requests_per_second']:.2f}")
        print(f"Average Response Time: {summary['average_response_time']:.2f}s")
        print(f"Min Response Time: {summary['min_response_time']:.2f}s")
        print(f"Max Response Time: {summary['max_response_time']:.2f}s")
        print(f"Status Code Distribution: {summary['status_codes']}")
        print(f"Test Duration: {summary['elapsed_time']:.2f}s")
        
        # Save summary to file
        summary_path = f"/tmp/load_test_summary_{int(time.time())}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nDetailed summary saved to: {summary_path}")


# Custom load testing scenarios
class LoadTestScenarios:
    """Predefined load testing scenarios."""
    
    @staticmethod
    def smoke_test():
        """Light load for smoke testing."""
        return {
            "users": 2,
            "spawn_rate": 1,
            "run_time": "2m",
            "user_class": YouTubeSummarizerUser,
        }
    
    @staticmethod
    def load_test():
        """Standard load test."""
        return {
            "users": 10,
            "spawn_rate": 2,
            "run_time": "5m",
            "user_class": YouTubeSummarizerUser,
        }
    
    @staticmethod
    def stress_test():
        """High load stress test."""
        return {
            "users": 25,
            "spawn_rate": 5,
            "run_time": "10m",
            "user_class": HeavyLoadUser,
        }
    
    @staticmethod
    def spike_test():
        """Rapid scaling test."""
        return {
            "users": 50,
            "spawn_rate": 10,
            "run_time": "3m",
            "user_class": HeavyLoadUser,
        }
    
    @staticmethod
    def endurance_test():
        """Long-running endurance test."""
        return {
            "users": 15,
            "spawn_rate": 3,
            "run_time": "30m",
            "user_class": YouTubeSummarizerUser,
        }


def run_scenario(scenario_name: str):
    """Run a specific load testing scenario."""
    scenarios = {
        "smoke": LoadTestScenarios.smoke_test(),
        "load": LoadTestScenarios.load_test(),
        "stress": LoadTestScenarios.stress_test(),
        "spike": LoadTestScenarios.spike_test(),
        "endurance": LoadTestScenarios.endurance_test(),
    }
    
    if scenario_name not in scenarios:
        print(f"Unknown scenario: {scenario_name}")
        print(f"Available scenarios: {list(scenarios.keys())}")
        return
    
    scenario = scenarios[scenario_name]
    print(f"\nRunning {scenario_name} test scenario:")
    print(f"  Users: {scenario['users']}")
    print(f"  Spawn Rate: {scenario['spawn_rate']}/s")
    print(f"  Duration: {scenario['run_time']}")
    print(f"  User Class: {scenario['user_class'].__name__}")
    
    if LOCUST_AVAILABLE:
        print("\nTo run this scenario with locust:")
        print(f"locust -f {__file__} --users {scenario['users']} --spawn-rate {scenario['spawn_rate']} --run-time {scenario['run_time']} --host http://localhost:8000")
    else:
        print("\nLocust not available. Install with: pip install locust")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        scenario_name = sys.argv[1]
        run_scenario(scenario_name)
    else:
        print("Available load testing scenarios:")
        for name in ["smoke", "load", "stress", "spike", "endurance"]:
            run_scenario(name)
        
        print("\nUsage: python test_load_testing.py <scenario_name>")
        print("Example: python test_load_testing.py smoke")