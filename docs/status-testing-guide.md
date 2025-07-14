# Status Tracking System Testing Guide

This guide provides comprehensive information about testing the status tracking system, including test structure, execution, and coverage analysis.

## Overview

The status tracking system includes comprehensive tests covering:

- **Unit Tests**: Individual component testing
- **Integration Tests**: Cross-component interaction testing  
- **API Tests**: REST endpoint testing
- **Performance Tests**: Load and stress testing
- **End-to-End Tests**: Complete workflow testing

## Test Structure

### Test Files Organization

```
tests/
├── test_status_tracking_comprehensive.py  # Main comprehensive test suite
├── test_status_api_endpoints.py          # API endpoint tests
└── test_status_performance.py            # Performance and load tests

src/
├── services/
│   ├── status_service.test.py           # StatusService unit tests
│   ├── status_updater.test.py           # StatusUpdater unit tests
│   ├── status_integration.test.py       # Integration components
│   ├── status_filtering.test.py         # Filtering service tests
│   └── status_events.test.py            # Event system tests
└── api/
    └── status.test.py                   # API-specific tests

scripts/
└── run_status_tests.py                 # Test runner script
```

### Test Categories

#### 1. Unit Tests
- **StatusService**: Database operations, status management
- **StatusUpdater**: Batch processing, queue management
- **StatusFilterService**: Advanced filtering and pagination
- **StatusEventManager**: Event processing and handling
- **Database Models**: Model validation and relationships

#### 2. Integration Tests
- **Workflow Integration**: Status tracking in complete workflows
- **Event System Integration**: Events with status changes
- **API Integration**: Endpoints with services
- **Database Integration**: Cross-table operations

#### 3. Performance Tests
- **Load Testing**: High-volume status operations
- **Concurrent Operations**: Multi-threaded status updates
- **Memory Efficiency**: Resource usage under load
- **Query Performance**: Database operation benchmarks

#### 4. API Tests
- **Endpoint Validation**: Request/response validation
- **Error Handling**: Error scenarios and responses
- **Authentication**: Security and access control
- **Rate Limiting**: API usage limits

## Running Tests

### Quick Start

```bash
# Run all status tracking tests
python scripts/run_status_tests.py

# Run with coverage
python scripts/run_status_tests.py --coverage

# Run including performance tests
python scripts/run_status_tests.py --performance --coverage

# Run with parallel execution
python scripts/run_status_tests.py --parallel 4 --verbose
```

### Individual Test Suites

```bash
# Run comprehensive tests
pytest tests/test_status_tracking_comprehensive.py -v

# Run API tests
pytest tests/test_status_api_endpoints.py -v

# Run performance tests
pytest tests/test_status_performance.py -v -s

# Run specific component tests
pytest src/services/status_service.test.py -v
pytest src/services/status_events.test.py -v
```

### Advanced Test Options

```bash
# Run with specific markers
pytest -m "not slow" tests/

# Run with coverage and detailed reporting
pytest --cov=src.services --cov=src.api --cov-report=html --cov-report=term-missing

# Run integration tests only
pytest tests/test_status_tracking_comprehensive.py::TestStatusTrackingIntegration -v

# Run performance benchmarks
pytest tests/test_status_performance.py -v -s --tb=short
```

## Test Configuration

### pytest.ini Configuration

```ini
[tool:pytest]
testpaths = tests src
python_files = test_*.py *_test.py *.test.py
python_classes = Test*
python_functions = test_*
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests
    performance: marks tests as performance tests
    api: marks tests as API tests
    unit: marks tests as unit tests
addopts = 
    --strict-markers
    --disable-warnings
    --tb=short
```

### Coverage Configuration

```ini
[coverage:run]
source = src
omit = 
    */tests/*
    */test_*
    */__init__.py
    */conftest.py

[coverage:report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise AssertionError
    raise NotImplementedError
    if __name__ == .__main__.:
show_missing = True
```

## Test Data and Fixtures

### Common Test Fixtures

```python
@pytest.fixture
def mock_db_session():
    """Mock database session for testing."""
    session = Mock()
    # Configure session mocks
    return session

@pytest.fixture
def sample_status():
    """Sample status for testing."""
    return ProcessingStatus(
        status_id="test_123",
        status=ProcessingStatusType.STARTING,
        priority=ProcessingPriority.NORMAL,
        progress_percentage=0.0
    )

@pytest.fixture
def status_service(mock_db_session):
    """StatusService instance for testing."""
    return StatusService(db_session=mock_db_session)
```

### Test Data Generation

```python
def generate_test_statuses(count=100):
    """Generate test status data."""
    statuses = []
    for i in range(count):
        status = ProcessingStatus(
            status_id=f"test_status_{i}",
            status=random.choice(list(ProcessingStatusType)),
            priority=random.choice(list(ProcessingPriority)),
            progress_percentage=random.uniform(0, 100)
        )
        statuses.append(status)
    return statuses
```

## Performance Testing

### Performance Benchmarks

The performance tests establish benchmarks for:

- **Status Creation**: < 10ms per status
- **Status Updates**: < 5ms per update
- **Query Performance**: < 100ms for complex filters
- **Event Processing**: > 500 events/second
- **Batch Operations**: > 1000 updates/second

### Performance Test Examples

```python
def test_bulk_operations_performance():
    """Test bulk status operations performance."""
    num_operations = 1000
    start_time = time.time()
    
    # Perform bulk operations
    for i in range(num_operations):
        service.create_processing_status(video_id=i)
    
    duration = time.time() - start_time
    
    # Performance assertions
    assert duration < 10.0  # Max 10 seconds
    assert (duration / num_operations) < 0.01  # Max 10ms per operation
```

### Memory Usage Testing

```python
def test_memory_efficiency():
    """Test memory usage under load."""
    import psutil
    
    process = psutil.Process()
    initial_memory = process.memory_info().rss
    
    # Perform memory-intensive operations
    results = perform_large_operations()
    
    final_memory = process.memory_info().rss
    memory_increase = (final_memory - initial_memory) / 1024 / 1024  # MB
    
    assert memory_increase < 100  # Max 100MB increase
```

## API Testing

### REST Endpoint Testing

```python
def test_status_api_endpoints(client):
    """Test status API endpoints."""
    # Test GET endpoint
    response = client.get("/api/status/test_123")
    assert response.status_code == 200
    
    # Test POST endpoint
    data = {"new_status": "completed", "progress_percentage": 100.0}
    response = client.put("/api/status/test_123", json=data)
    assert response.status_code == 200
```

### API Validation Testing

```python
def test_request_validation():
    """Test API request validation."""
    # Test invalid status
    with pytest.raises(ValidationError):
        StatusUpdateRequest(new_status="invalid_status")
    
    # Test invalid progress percentage
    with pytest.raises(ValidationError):
        ProgressUpdateRequest(progress_percentage=150.0)
```

## Error Testing

### Exception Handling

```python
def test_error_handling():
    """Test error handling in services."""
    service = StatusService()
    
    # Test database error handling
    service.db_session.commit.side_effect = SQLAlchemyError("DB Error")
    
    with pytest.raises(SQLAlchemyError):
        service.create_processing_status()
```

### Error Recovery Testing

```python
def test_error_recovery():
    """Test system recovery from errors."""
    # Simulate transient error
    service.record_error(
        status_id="test_123",
        error_info="Transient network error",
        should_retry=True
    )
    
    # Verify retry logic
    status = service.get_processing_status("test_123")
    assert status.status == ProcessingStatusType.RETRY_PENDING
```

## Continuous Integration

### GitHub Actions Workflow

```yaml
name: Status Tracking Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov pytest-asyncio
    
    - name: Run tests
      run: |
        python scripts/run_status_tests.py --coverage --parallel 2
    
    - name: Upload coverage
      uses: codecov/codecov-action@v1
      with:
        file: ./test_reports/coverage.xml
```

### Pre-commit Hooks

```yaml
# .pre-commit-config.yaml
repos:
-   repo: local
    hooks:
    -   id: status-tests
        name: Status Tracking Tests
        entry: python scripts/run_status_tests.py
        language: system
        pass_filenames: false
        always_run: true
```

## Coverage Analysis

### Coverage Requirements

- **Minimum Coverage**: 80% overall
- **Critical Components**: 95% coverage
  - StatusService
  - StatusUpdater
  - Event system core
- **API Endpoints**: 90% coverage
- **Database Models**: 85% coverage

### Coverage Commands

```bash
# Generate coverage report
coverage run -m pytest tests/
coverage report --show-missing

# Generate HTML coverage report
coverage html
open htmlcov/index.html

# Check coverage against requirements
coverage report --fail-under=80
```

### Coverage Exclusions

```python
def method_not_covered():  # pragma: no cover
    """Method excluded from coverage."""
    pass

if TYPE_CHECKING:  # pragma: no cover
    # Type checking imports
    pass
```

## Test Maintenance

### Regular Test Updates

1. **Update test data** when schema changes
2. **Add new test cases** for new features
3. **Review performance benchmarks** quarterly
4. **Update mocks** when dependencies change
5. **Validate test coverage** with new code

### Test Debugging

```bash
# Run single test with debugging
pytest tests/test_status_service.py::test_create_status -vvv -s --pdb

# Run with logging enabled
pytest --log-cli-level=DEBUG tests/

# Profile test execution
pytest --profile tests/
```

## Best Practices

### Test Writing Guidelines

1. **Use descriptive test names** that explain what is being tested
2. **Follow AAA pattern**: Arrange, Act, Assert
3. **Use fixtures** for common setup
4. **Mock external dependencies** appropriately
5. **Test both success and failure scenarios**
6. **Keep tests focused** on single functionality

### Example Test Structure

```python
def test_status_update_success():
    """Test successful status update with all fields."""
    # Arrange
    service = StatusService(mock_session)
    status_id = "test_123"
    new_status = ProcessingStatusType.COMPLETED
    
    # Act
    result = service.update_status(
        status_id=status_id,
        new_status=new_status,
        progress_percentage=100.0
    )
    
    # Assert
    assert result.status == new_status
    assert result.progress_percentage == 100.0
    mock_session.commit.assert_called_once()
```

### Performance Testing Guidelines

1. **Set realistic benchmarks** based on production requirements
2. **Test with representative data volumes**
3. **Monitor memory usage** during tests
4. **Use appropriate timeouts** for async operations
5. **Document performance expectations**

## Troubleshooting

### Common Test Issues

1. **Database Connection Errors**
   - Ensure test database is available
   - Check connection string configuration
   - Use proper transaction isolation

2. **Async Test Issues**
   - Use `pytest-asyncio` for async tests
   - Properly await async operations
   - Handle event loop cleanup

3. **Mock Configuration**
   - Ensure mocks match real interfaces
   - Reset mocks between tests
   - Use `autospec=True` for better validation

4. **Performance Test Variability**
   - Run multiple iterations
   - Account for system load
   - Use relative benchmarks

### Debug Commands

```bash
# Debug specific test
pytest tests/test_status.py::test_function -vvv --pdb

# Show test collection
pytest --collect-only tests/

# Run with maximum verbosity
pytest -vvv --tb=long tests/

# Profile test performance
pytest --durations=10 tests/
```

This comprehensive testing approach ensures the status tracking system is robust, performant, and maintainable.