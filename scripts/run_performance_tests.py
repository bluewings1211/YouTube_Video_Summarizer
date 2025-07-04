#!/usr/bin/env python3
"""
Performance testing runner script for YouTube Summarizer API.

This script provides a convenient way to run various performance tests
and generate comprehensive reports.
"""

import os
import sys
import subprocess
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def ensure_directory_exists(path: str):
    """Ensure directory exists, create if not."""
    Path(path).mkdir(parents=True, exist_ok=True)

def run_command(command: list, capture_output: bool = True, timeout: int = 300):
    """Run a command and return the result."""
    try:
        result = subprocess.run(
            command,
            capture_output=capture_output,
            text=True,
            timeout=timeout,
            cwd=os.path.join(os.path.dirname(__file__), '..')
        )
        return result
    except subprocess.TimeoutExpired:
        print(f"Command timed out after {timeout} seconds: {' '.join(command)}")
        return None
    except Exception as e:
        print(f"Error running command {' '.join(command)}: {e}")
        return None

def run_pytest_benchmarks():
    """Run pytest-based performance benchmarks."""
    print("Running pytest performance benchmarks...")
    
    benchmark_tests = [
        "tests/test_performance_benchmarks.py",
        "tests/test_benchmarking_suite.py",
    ]
    
    results = {}
    
    for test_file in benchmark_tests:
        print(f"\nRunning {test_file}...")
        
        command = [
            sys.executable, "-m", "pytest",
            test_file,
            "-v",
            "-s",
            "--tb=short",
            "--no-header",
            "--disable-warnings",
            "-m", "slow"
        ]
        
        result = run_command(command, capture_output=False)
        
        if result:
            results[test_file] = {
                "return_code": result.returncode,
                "success": result.returncode == 0,
                "duration": "unknown"  # pytest doesn't return duration easily
            }
        else:
            results[test_file] = {
                "return_code": -1,
                "success": False,
                "duration": "timeout"
            }
    
    return results

def run_load_tests(scenario: str = "smoke"):
    """Run locust-based load tests."""
    print(f"\nRunning load tests with {scenario} scenario...")
    
    # Check if locust is available
    check_locust = run_command([sys.executable, "-c", "import locust"])
    if check_locust and check_locust.returncode != 0:
        print("Locust not available. Install with: pip install locust")
        return {"load_test": {"success": False, "error": "locust not available"}}
    
    # Define scenarios
    scenarios = {
        "smoke": {"users": 2, "spawn_rate": 1, "run_time": "30s"},
        "load": {"users": 10, "spawn_rate": 2, "run_time": "2m"},
        "stress": {"users": 25, "spawn_rate": 5, "run_time": "5m"},
    }
    
    if scenario not in scenarios:
        print(f"Unknown scenario: {scenario}. Available: {list(scenarios.keys())}")
        return {"load_test": {"success": False, "error": f"unknown scenario: {scenario}"}}
    
    config = scenarios[scenario]
    
    command = [
        sys.executable, "-m", "locust",
        "-f", "tests/test_load_testing.py",
        "--users", str(config["users"]),
        "--spawn-rate", str(config["spawn_rate"]),
        "--run-time", config["run_time"],
        "--host", "http://localhost:8000",
        "--headless",
        "--html", f"/tmp/locust_report_{scenario}_{int(time.time())}.html",
        "--csv", f"/tmp/locust_data_{scenario}_{int(time.time())}",
        "--only-summary"
    ]
    
    start_time = time.time()
    result = run_command(command, capture_output=False, timeout=600)
    end_time = time.time()
    
    return {
        "load_test": {
            "scenario": scenario,
            "success": result and result.returncode == 0,
            "return_code": result.returncode if result else -1,
            "duration": end_time - start_time,
            "config": config
        }
    }

def check_api_availability():
    """Check if the API is running and responsive."""
    try:
        import requests
        response = requests.get("http://localhost:8000/health", timeout=5)
        return response.status_code == 200
    except Exception:
        return False

def start_api_if_needed():
    """Start the API if it's not running."""
    if check_api_availability():
        print("API is already running.")
        return True
    
    print("API not running. Attempting to start with Docker Compose...")
    
    # Try to start with docker-compose
    result = run_command(["docker-compose", "up", "-d"], timeout=60)
    
    if result and result.returncode == 0:
        # Wait for API to be ready
        for i in range(30):  # Wait up to 30 seconds
            time.sleep(1)
            if check_api_availability():
                print("API started successfully.")
                return True
        
        print("API started but not responding to health checks.")
        return False
    else:
        print("Failed to start API with docker-compose.")
        return False

def generate_performance_report(results: dict):
    """Generate a comprehensive performance report."""
    timestamp = datetime.now()
    
    report = {
        "report_metadata": {
            "generated_at": timestamp.isoformat(),
            "report_type": "performance_testing",
            "version": "1.0.0",
        },
        "test_environment": {
            "api_available": check_api_availability(),
            "python_version": sys.version,
            "platform": sys.platform,
        },
        "test_results": results,
        "summary": {
            "total_test_suites": len(results),
            "successful_suites": sum(1 for r in results.values() if r.get("success", False)),
            "failed_suites": sum(1 for r in results.values() if not r.get("success", False)),
        }
    }
    
    # Calculate overall success rate
    total_tests = len(results)
    successful_tests = sum(1 for r in results.values() if r.get("success", False))
    report["summary"]["overall_success_rate"] = (successful_tests / total_tests * 100) if total_tests > 0 else 0
    
    # Generate report file
    report_filename = f"/tmp/performance_test_report_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
    
    ensure_directory_exists(os.path.dirname(report_filename))
    
    with open(report_filename, 'w') as f:
        json.dump(report, f, indent=2)
    
    return report_filename

def print_summary(results: dict):
    """Print a summary of test results."""
    print("\n" + "="*60)
    print("PERFORMANCE TESTING SUMMARY")
    print("="*60)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result.get("success", False) else "❌ FAILED"
        print(f"{test_name:<40} {status}")
        
        if "duration" in result:
            print(f"{'Duration:':<40} {result['duration']}")
        
        if "error" in result:
            print(f"{'Error:':<40} {result['error']}")
        
        print("-" * 60)
    
    total_tests = len(results)
    successful_tests = sum(1 for r in results.values() if r.get("success", False))
    
    print(f"\nTotal Test Suites: {total_tests}")
    print(f"Successful: {successful_tests}")
    print(f"Failed: {total_tests - successful_tests}")
    print(f"Success Rate: {(successful_tests / total_tests * 100):.1f}%" if total_tests > 0 else "N/A")

def main():
    """Main performance testing function."""
    parser = argparse.ArgumentParser(description="Run performance tests for YouTube Summarizer API")
    
    parser.add_argument(
        "--tests",
        choices=["all", "benchmarks", "load"],
        default="all",
        help="Which tests to run (default: all)"
    )
    
    parser.add_argument(
        "--load-scenario",
        choices=["smoke", "load", "stress"],
        default="smoke",
        help="Load testing scenario (default: smoke)"
    )
    
    parser.add_argument(
        "--start-api",
        action="store_true",
        help="Try to start API if not running"
    )
    
    parser.add_argument(
        "--skip-api-check",
        action="store_true",
        help="Skip API availability check"
    )
    
    args = parser.parse_args()
    
    print("YouTube Summarizer Performance Testing Suite")
    print("=" * 50)
    
    # Check API availability
    if not args.skip_api_check:
        if args.start_api:
            api_available = start_api_if_needed()
        else:
            api_available = check_api_availability()
        
        if not api_available:
            print("⚠️  API is not available. Some tests may fail.")
            print("   Use --start-api to attempt starting the API automatically.")
            print("   Use --skip-api-check to run tests anyway (mock mode).")
        else:
            print("✅ API is available and responsive.")
    
    results = {}
    
    # Run benchmark tests
    if args.tests in ["all", "benchmarks"]:
        print("\n" + "="*50)
        print("RUNNING BENCHMARK TESTS")
        print("="*50)
        
        benchmark_results = run_pytest_benchmarks()
        results.update(benchmark_results)
    
    # Run load tests
    if args.tests in ["all", "load"]:
        print("\n" + "="*50)
        print("RUNNING LOAD TESTS")
        print("="*50)
        
        load_results = run_load_tests(args.load_scenario)
        results.update(load_results)
    
    # Generate comprehensive report
    report_file = generate_performance_report(results)
    
    # Print summary
    print_summary(results)
    
    print(f"\n📊 Detailed report saved to: {report_file}")
    
    # Exit with appropriate code
    failed_tests = sum(1 for r in results.values() if not r.get("success", False))
    sys.exit(failed_tests)

if __name__ == "__main__":
    main()