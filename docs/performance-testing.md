# Performance Testing and Benchmarking Guide

## Overview

This document provides comprehensive guidance for performance testing and benchmarking the YouTube Summarizer API. The testing suite includes response time analysis, memory usage profiling, CPU utilization monitoring, throughput analysis, and scalability testing.

## Performance Testing Tools

### 1. Pytest-based Benchmarks
- **File**: `tests/test_performance_benchmarks.py`
- **Purpose**: Detailed performance analysis with system resource monitoring
- **Features**: Response time measurement, memory leak detection, concurrent load testing

### 2. Locust Load Testing
- **File**: `tests/test_load_testing.py`
- **Purpose**: Simulated user load testing with realistic usage patterns
- **Features**: Concurrent users, realistic request patterns, error handling validation

### 3. Comprehensive Benchmarking Suite
- **File**: `tests/test_benchmarking_suite.py`
- **Purpose**: In-depth performance analysis with detailed metrics collection
- **Features**: System resource monitoring, performance regression detection, export capabilities

### 4. Performance Test Runner
- **File**: `scripts/run_performance_tests.py`
- **Purpose**: Automated test execution with report generation
- **Features**: Multiple test scenarios, automatic API startup, comprehensive reporting

## Running Performance Tests

### Prerequisites

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the API** (choose one method)
   ```bash
   # Option 1: Docker Compose (recommended)
   docker-compose up -d
   
   # Option 2: Direct Python execution
   python -m uvicorn src.app:app --host 0.0.0.0 --port 8000
   ```

3. **Verify API is Running**
   ```bash
   curl http://localhost:8000/health
   ```

### Quick Start

**Run all performance tests:**
```bash
python scripts/run_performance_tests.py
```

**Run specific test types:**
```bash
# Only benchmark tests
python scripts/run_performance_tests.py --tests benchmarks

# Only load tests
python scripts/run_performance_tests.py --tests load

# Load tests with stress scenario
python scripts/run_performance_tests.py --tests load --load-scenario stress
```

**Auto-start API:**
```bash
python scripts/run_performance_tests.py --start-api
```

### Individual Test Execution

**Pytest Benchmarks:**
```bash
# Run all performance benchmarks
pytest tests/test_performance_benchmarks.py -v -s -m slow

# Run specific benchmark
pytest tests/test_performance_benchmarks.py::TestPerformanceBenchmarks::test_single_request_performance -v -s

# Run with coverage
pytest tests/test_performance_benchmarks.py --cov=src --cov-report=html -m slow
```

**Locust Load Testing:**
```bash
# Smoke test (light load)
locust -f tests/test_load_testing.py --users 2 --spawn-rate 1 --run-time 30s --host http://localhost:8000 --headless

# Standard load test
locust -f tests/test_load_testing.py --users 10 --spawn-rate 2 --run-time 5m --host http://localhost:8000 --headless

# Stress test
locust -f tests/test_load_testing.py --users 25 --spawn-rate 5 --run-time 10m --host http://localhost:8000 --headless

# Interactive mode (web UI)
locust -f tests/test_load_testing.py --host http://localhost:8000
# Then open http://localhost:8089 in your browser
```

**Comprehensive Benchmarking:**
```bash
# Run full benchmark suite
pytest tests/test_benchmarking_suite.py::TestBenchmarkingSuite::test_run_benchmark_suite -v -s

# Run benchmark suite directly
python tests/test_benchmarking_suite.py
```

## Performance Metrics and Thresholds

### Response Time Targets

| Endpoint | Target Response Time | 95th Percentile | Maximum |
|----------|---------------------|-----------------|---------|
| `/api/v1/summarize` (short video) | < 15s | < 20s | < 30s |
| `/api/v1/summarize` (medium video) | < 25s | < 35s | < 45s |
| `/api/v1/summarize` (long video) | < 35s | < 50s | < 60s |
| `/health` | < 0.5s | < 1s | < 2s |
| `/metrics` | < 1s | < 2s | < 5s |
| Error responses | < 2s | < 5s | < 10s |

### Resource Usage Targets

| Metric | Target | Warning | Critical |
|--------|--------|---------|----------|
| Memory Usage (per request) | < 500MB | > 1GB | > 2GB |
| CPU Usage (average) | < 70% | > 85% | > 95% |
| Memory Growth (10 requests) | < 50MB | > 100MB | > 200MB |
| Concurrent Requests | 10 users | 25 users | 50 users |

### Throughput Targets

| Load Scenario | Target RPS | Success Rate | Error Rate |
|---------------|------------|--------------|------------|
| Light Load (5 users) | > 1 RPS | > 95% | < 5% |
| Medium Load (10 users) | > 1.5 RPS | > 90% | < 10% |
| Heavy Load (25 users) | > 2 RPS | > 80% | < 20% |

## Test Scenarios

### 1. Single Request Performance Test
- **Purpose**: Baseline performance measurement
- **Metrics**: Response time, memory usage, CPU utilization
- **Expected**: < 30s response time, < 1GB memory usage

### 2. Concurrent Load Test
- **Purpose**: Multi-user performance validation
- **Configuration**: 5 concurrent users, 3 requests each
- **Metrics**: Average response time, 95th percentile, success rate
- **Expected**: < 50s 95th percentile, > 80% success rate

### 3. Video Length Scaling Test
- **Purpose**: Performance scaling with video duration
- **Scenarios**: Short (30s), Medium (5min), Long (10min) videos
- **Expected**: Reasonable scaling (< 5x increase for 20x duration)

### 4. Error Handling Performance Test
- **Purpose**: Error response performance validation
- **Scenarios**: Invalid URLs, nonexistent videos, malformed requests
- **Expected**: < 5s error response time

### 5. Memory Leak Detection Test
- **Purpose**: Memory management validation
- **Configuration**: 10 sequential requests with garbage collection
- **Expected**: < 200MB memory growth

### 6. Health Endpoint Performance Test
- **Purpose**: Monitoring endpoint performance
- **Configuration**: 10 rapid health checks
- **Expected**: < 1s response time, < 0.5s average

## Load Testing Scenarios

### Smoke Test
```yaml
Users: 2
Spawn Rate: 1/second
Duration: 30 seconds
Purpose: Basic functionality validation
```

### Load Test
```yaml
Users: 10
Spawn Rate: 2/second
Duration: 5 minutes
Purpose: Normal load simulation
```

### Stress Test
```yaml
Users: 25
Spawn Rate: 5/second
Duration: 10 minutes
Purpose: High load stress testing
```

### Spike Test
```yaml
Users: 50
Spawn Rate: 10/second
Duration: 3 minutes
Purpose: Rapid scaling validation
```

### Endurance Test
```yaml
Users: 15
Spawn Rate: 3/second
Duration: 30 minutes
Purpose: Long-term stability testing
```

## User Behavior Patterns

### Realistic User Simulation
The load tests simulate realistic user behavior:

- **70% Video Summarization**: Normal API usage
- **20% Health Checks**: Monitoring and health verification
- **10% Error Scenarios**: Invalid inputs and error handling

### Request Patterns
- **Think Time**: 1-5 seconds between requests
- **Session Duration**: Variable user session lengths
- **Request Distribution**: Weighted task distribution based on real usage

## Report Generation and Analysis

### Automated Reports

Performance tests automatically generate reports in multiple formats:

1. **JSON Reports**: Detailed metrics and raw data
   - Location: `/tmp/performance_test_report_YYYYMMDD_HHMMSS.json`
   - Contents: Environment info, test results, summary statistics

2. **CSV Reports**: Tabular data for analysis
   - Location: `/tmp/benchmark_results_YYYYMMDD_HHMMSS.csv`
   - Contents: Individual test results with all metrics

3. **HTML Reports**: Visual load testing results
   - Location: `/tmp/locust_report_scenario_timestamp.html`
   - Contents: Locust-generated visual reports

### Key Metrics in Reports

1. **Response Time Analysis**
   - Mean, median, min, max response times
   - 90th, 95th, 99th percentile response times
   - Response time distribution and trends

2. **Resource Usage Analysis**
   - Memory usage patterns (before, after, peak)
   - CPU utilization during tests
   - Memory growth and leak detection

3. **Throughput Analysis**
   - Requests per second
   - Success rate and error rate
   - Concurrent user handling capacity

4. **Error Analysis**
   - Error distribution by type
   - Error response times
   - Recovery and retry patterns

## Performance Optimization Guidelines

### Response Time Optimization

1. **Caching Strategies**
   - Implement transcript caching for repeated requests
   - Cache LLM responses for identical content
   - Use Redis for distributed caching

2. **Async Processing**
   - Use async/await for concurrent operations
   - Implement background task processing
   - Parallelize independent operations

3. **Request Optimization**
   - Implement request deduplication
   - Use connection pooling for external APIs
   - Optimize payload sizes

### Memory Management

1. **Resource Cleanup**
   - Proper cleanup of temporary objects
   - Garbage collection tuning
   - Connection pooling and reuse

2. **Memory Profiling**
   - Regular memory usage monitoring
   - Leak detection and prevention
   - Resource usage optimization

### Scalability Improvements

1. **Horizontal Scaling**
   - Load balancing configuration
   - Session-less design
   - Database optimization

2. **Vertical Scaling**
   - Resource allocation tuning
   - CPU and memory optimization
   - I/O performance improvement

## Continuous Performance Monitoring

### CI/CD Integration

Integrate performance tests into your CI/CD pipeline:

```yaml
# Example GitHub Actions step
- name: Run Performance Tests
  run: |
    docker-compose up -d
    python scripts/run_performance_tests.py --tests benchmarks
    docker-compose down
```

### Performance Regression Detection

1. **Baseline Establishment**
   - Run performance tests on stable releases
   - Store baseline metrics for comparison
   - Set up automated threshold monitoring

2. **Regression Alerts**
   - Monitor for performance degradation
   - Set up alerting for threshold breaches
   - Implement automated rollback triggers

### Production Monitoring

1. **Real-time Metrics**
   - Response time monitoring
   - Error rate tracking
   - Resource usage alerts

2. **Performance Dashboards**
   - Grafana/Prometheus integration
   - Real-time performance visualization
   - Historical trend analysis

## Troubleshooting Performance Issues

### Common Performance Problems

1. **High Response Times**
   - Check LLM API latency
   - Verify YouTube API performance
   - Monitor database query performance
   - Review network connectivity

2. **Memory Leaks**
   - Check for unclosed connections
   - Review object lifecycle management
   - Monitor garbage collection efficiency
   - Analyze memory allocation patterns

3. **CPU Bottlenecks**
   - Profile CPU-intensive operations
   - Optimize algorithmic complexity
   - Consider parallel processing
   - Review serialization overhead

### Debugging Tools

1. **Performance Profiling**
   ```bash
   # Memory profiling
   python -m memory_profiler src/app.py
   
   # CPU profiling
   python -m cProfile -s cumulative src/app.py
   ```

2. **Load Testing Analysis**
   ```bash
   # Detailed locust analysis
   locust -f tests/test_load_testing.py --csv results --host http://localhost:8000
   ```

3. **System Monitoring**
   ```bash
   # Real-time system monitoring
   htop
   iotop
   nethogs
   ```

## Best Practices

### Test Design

1. **Test Isolation**
   - Each test should be independent
   - Clean up resources after tests
   - Use fresh environments for each run

2. **Realistic Scenarios**
   - Use realistic data and usage patterns
   - Test edge cases and error conditions
   - Simulate production load patterns

3. **Reproducible Tests**
   - Use consistent test data
   - Document test environments
   - Version control test configurations

### Monitoring and Alerting

1. **Proactive Monitoring**
   - Set up performance baselines
   - Monitor trends over time
   - Alert on degradation patterns

2. **Comprehensive Coverage**
   - Test all critical user paths
   - Include error scenarios
   - Monitor resource usage patterns

### Documentation and Reporting

1. **Clear Documentation**
   - Document test purposes and expectations
   - Provide troubleshooting guidance
   - Maintain performance requirements

2. **Regular Reporting**
   - Generate periodic performance reports
   - Track performance metrics over time
   - Share results with stakeholders

This comprehensive performance testing framework ensures the YouTube Summarizer API maintains optimal performance under various load conditions while providing detailed insights for continuous optimization.