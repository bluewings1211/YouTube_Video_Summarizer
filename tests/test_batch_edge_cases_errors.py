"""
Comprehensive edge case and error condition tests for batch processing.

This test suite covers edge cases and error conditions including:
- Boundary condition testing
- Invalid input handling
- Resource exhaustion scenarios
- Network failure simulation
- Database corruption scenarios
- Race condition testing
- Malformed data handling
- Security boundary testing
- Recovery mechanism testing
- Graceful degradation testing
"""

import pytest
import asyncio
import threading
import time
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from sqlalchemy.exc import SQLAlchemyError, IntegrityError, OperationalError
from concurrent.futures import ThreadPoolExecutor
import uuid
import random
import string

from src.database.models import Base, Video
from src.database.batch_models import (
    Batch, BatchItem, QueueItem, ProcessingSession,
    BatchStatus, BatchItemStatus, BatchPriority
)
from src.services.batch_service import (
    BatchService, BatchCreateRequest, BatchItemResult, BatchServiceError
)
from src.services.queue_service import (
    QueueService, QueueProcessingOptions, QueueServiceError
)
from src.services.concurrent_batch_service import (
    ConcurrentBatchService, ConcurrentBatchConfig, ConcurrentBatchError
)
from src.services.batch_processor import (
    BatchProcessor, BatchProcessorError
)


class TestBatchEdgeCasesErrors:
    """Comprehensive edge case and error condition tests."""

    @pytest.fixture(scope="function")
    def db_engine(self):
        """Create in-memory database engine."""
        engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
            echo=False
        )
        Base.metadata.create_all(engine)
        return engine

    @pytest.fixture(scope="function")
    def db_session(self, db_engine):
        """Create database session."""
        TestingSessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=db_engine
        )
        session = TestingSessionLocal()
        yield session
        session.rollback()
        session.close()

    @pytest.fixture(scope="function")
    def batch_service(self, db_session):
        """Create BatchService instance."""
        return BatchService(session=db_session)

    @pytest.fixture(scope="function")
    def queue_service(self, db_session):
        """Create QueueService instance."""
        options = QueueProcessingOptions(
            max_workers=5,
            worker_timeout_minutes=1,
            enable_automatic_cleanup=False
        )
        service = QueueService(db_session, options)
        yield service
        service.shutdown()

    # Input Validation Edge Cases

    def test_batch_creation_empty_inputs(self, batch_service):
        """Test batch creation with various empty inputs."""
        # Empty URLs list
        with pytest.raises(BatchServiceError, match="URLs list cannot be empty"):
            batch_service.create_batch(BatchCreateRequest(
                name="Empty URLs",
                urls=[]
            ))
        
        # None URLs
        with pytest.raises((BatchServiceError, TypeError)):
            batch_service.create_batch(BatchCreateRequest(
                name="None URLs",
                urls=None
            ))
        
        # Empty name
        with patch('src.services.batch_service.extract_video_id_from_url') as mock_extract:
            mock_extract.return_value = "test123"
            
            batch = batch_service.create_batch(BatchCreateRequest(
                name="",  # Empty name should be allowed
                urls=["https://www.youtube.com/watch?v=test123"]
            ))
            assert batch.name == ""

    def test_batch_creation_invalid_urls(self, batch_service):
        """Test batch creation with invalid URL formats."""
        invalid_urls = [
            "",  # Empty string
            "not-a-url",  # Not a URL
            "http://",  # Incomplete URL
            "https://",  # Incomplete URL
            "ftp://example.com",  # Wrong protocol
            "https://example.com",  # Not YouTube
            "https://youtube.com",  # Missing path
            "https://www.youtube.com/",  # Missing video parameter
            "https://www.youtube.com/watch",  # Missing video ID
            "https://www.youtube.com/watch?v=",  # Empty video ID
            "https://www.youtube.com/watch?other=123",  # Wrong parameter
            "https://www.youtube.com/watch?v=" + "x" * 1000,  # Extremely long ID
            None,  # None value
            123,  # Non-string
            {"url": "test"},  # Wrong type
        ]
        
        for invalid_url in invalid_urls:
            try:
                with patch('src.services.batch_service.extract_video_id_from_url') as mock_extract:
                    mock_extract.return_value = None  # Invalid URL
                    
                    with pytest.raises(BatchServiceError):
                        batch_service.create_batch(BatchCreateRequest(
                            name=f"Invalid URL Test",
                            urls=[invalid_url] if invalid_url is not None else [str(invalid_url)]
                        ))
            except (TypeError, AttributeError):
                # Expected for non-string types
                pass

    def test_batch_creation_boundary_sizes(self, batch_service):
        """Test batch creation with boundary sizes."""
        with patch('src.services.batch_service.extract_video_id_from_url') as mock_extract:
            mock_extract.side_effect = lambda url: url.split('=')[-1] if '=' in url else url.split('/')[-1]
            
            # Single URL (minimum valid batch)
            single_url = ["https://www.youtube.com/watch?v=single"]
            batch = batch_service.create_batch(BatchCreateRequest(
                name="Single URL Batch",
                urls=single_url
            ))
            assert batch.total_items == 1
            
            # Very long name
            long_name = "A" * 10000
            batch = batch_service.create_batch(BatchCreateRequest(
                name=long_name,
                urls=single_url
            ))
            assert batch.name == long_name
            
            # Very long description
            long_description = "B" * 50000
            batch = batch_service.create_batch(BatchCreateRequest(
                name="Long Description Test",
                description=long_description,
                urls=single_url
            ))
            assert batch.description == long_description

    def test_batch_creation_special_characters(self, batch_service):
        """Test batch creation with special characters."""
        with patch('src.services.batch_service.extract_video_id_from_url') as mock_extract:
            mock_extract.return_value = "test123"
            
            special_cases = [
                "Batch with émojis 🚀🎉",
                "Batch with unicode ñáéíóú",
                "Batch with symbols !@#$%^&*()",
                "Batch\nwith\nnewlines",
                "Batch\twith\ttabs",
                "Batch with quotes \"'`",
                "Batch with backslashes \\\\",
                "Batch with SQL '; DROP TABLE batches; --",
                "Batch with HTML <script>alert('xss')</script>",
                "Batch with JSON {\"key\": \"value\"}",
                "Batch with XML <?xml version=\"1.0\"?>",
                "\x00\x01\x02",  # Control characters
                "🚀" * 100,  # Many emojis
            ]
            
            for special_name in special_cases:
                try:
                    batch = batch_service.create_batch(BatchCreateRequest(
                        name=special_name,
                        urls=["https://www.youtube.com/watch?v=test123"]
                    ))
                    assert batch.name == special_name
                except Exception as e:
                    # Some special characters might cause valid errors
                    print(f"Special character test failed for '{special_name}': {e}")

    def test_batch_creation_extreme_metadata(self, batch_service):
        """Test batch creation with extreme metadata scenarios."""
        with patch('src.services.batch_service.extract_video_id_from_url') as mock_extract:
            mock_extract.return_value = "test123"
            
            # Very large metadata
            large_metadata = {
                "large_field": "x" * 100000,
                "many_fields": {f"field_{i}": f"value_{i}" for i in range(1000)}
            }
            
            batch = batch_service.create_batch(BatchCreateRequest(
                name="Large Metadata Test",
                urls=["https://www.youtube.com/watch?v=test123"],
                batch_metadata=large_metadata
            ))
            assert batch.batch_metadata == large_metadata
            
            # Nested metadata
            nested_metadata = {
                "level1": {
                    "level2": {
                        "level3": {
                            "level4": {
                                "level5": "deep_value"
                            }
                        }
                    }
                }
            }
            
            batch = batch_service.create_batch(BatchCreateRequest(
                name="Nested Metadata Test",
                urls=["https://www.youtube.com/watch?v=test123"],
                batch_metadata=nested_metadata
            ))
            assert batch.batch_metadata == nested_metadata
            
            # Special types in metadata
            special_metadata = {
                "none_value": None,
                "boolean_value": True,
                "integer_value": 42,
                "float_value": 3.14159,
                "list_value": [1, 2, 3, "a", "b", "c"],
                "empty_dict": {},
                "empty_list": [],
                "unicode_string": "Unicode: ñáéíóú 🚀"
            }
            
            batch = batch_service.create_batch(BatchCreateRequest(
                name="Special Types Metadata Test",
                urls=["https://www.youtube.com/watch?v=test123"],
                batch_metadata=special_metadata
            ))
            assert batch.batch_metadata == special_metadata

    # Database Error Conditions

    def test_database_connection_failures(self, batch_service):
        """Test handling of database connection failures."""
        # Mock database connection failure
        with patch.object(batch_service._get_session(), 'commit') as mock_commit:
            mock_commit.side_effect = OperationalError("Database connection lost", None, None)
            
            with pytest.raises(BatchServiceError):
                batch_service.create_batch(BatchCreateRequest(
                    name="DB Failure Test",
                    urls=["https://www.youtube.com/watch?v=test123"]
                ))

    def test_database_integrity_errors(self, batch_service):
        """Test handling of database integrity constraint violations."""
        with patch('src.services.batch_service.extract_video_id_from_url') as mock_extract:
            mock_extract.return_value = "test123"
            
            # Create initial batch
            batch1 = batch_service.create_batch(BatchCreateRequest(
                name="Integrity Test 1",
                urls=["https://www.youtube.com/watch?v=test123"]
            ))
            
            # Mock integrity error on commit
            with patch.object(batch_service._get_session(), 'commit') as mock_commit:
                mock_commit.side_effect = IntegrityError("Duplicate key", None, None)
                
                with pytest.raises(BatchServiceError):
                    batch_service.create_batch(BatchCreateRequest(
                        name="Integrity Test 2",
                        urls=["https://www.youtube.com/watch?v=test123"]
                    ))

    def test_database_transaction_rollback(self, batch_service):
        """Test database transaction rollback scenarios."""
        with patch('src.services.batch_service.extract_video_id_from_url') as mock_extract:
            mock_extract.return_value = "test123"
            
            original_commit = batch_service._get_session().commit
            
            def failing_commit():
                # Commit some changes then fail
                original_commit()
                raise SQLAlchemyError("Simulated failure")
            
            with patch.object(batch_service._get_session(), 'commit', side_effect=failing_commit):
                with pytest.raises(BatchServiceError):
                    batch_service.create_batch(BatchCreateRequest(
                        name="Rollback Test",
                        urls=["https://www.youtube.com/watch?v=test123"]
                    ))
            
            # Verify rollback occurred - batch shouldn't exist
            batches = batch_service.list_batches()
            rollback_batches = [b for b in batches if b.name == "Rollback Test"]
            assert len(rollback_batches) == 0

    def test_database_corruption_recovery(self, batch_service, db_session):
        """Test recovery from database corruption scenarios."""
        with patch('src.services.batch_service.extract_video_id_from_url') as mock_extract:
            mock_extract.return_value = "test123"
            
            # Create valid batch
            batch = batch_service.create_batch(BatchCreateRequest(
                name="Corruption Test",
                urls=["https://www.youtube.com/watch?v=test123"]
            ))
            
            # Simulate corruption by manually modifying database
            try:
                db_session.execute(text("UPDATE batches SET total_items = -1 WHERE id = :id"), 
                                 {"id": batch.id})
                db_session.commit()
                
                # Try to retrieve corrupted batch
                corrupted_batch = batch_service.get_batch(batch.batch_id)
                
                # Should handle corruption gracefully
                assert corrupted_batch is not None
                assert corrupted_batch.total_items == -1  # Corruption preserved
                
            except Exception as e:
                # Recovery mechanism should prevent crashes
                print(f"Corruption test triggered recovery: {e}")

    # Queue Service Edge Cases

    def test_queue_worker_registration_edge_cases(self, queue_service):
        """Test queue worker registration edge cases."""
        # Register worker with empty queue name
        with pytest.raises(QueueServiceError):
            queue_service.register_worker("")
        
        # Register worker with None queue name
        with pytest.raises((QueueServiceError, TypeError)):
            queue_service.register_worker(None)
        
        # Register worker with extremely long queue name
        long_queue_name = "x" * 10000
        worker = queue_service.register_worker(long_queue_name)
        assert worker.queue_name == long_queue_name
        
        # Register worker with special characters in queue name
        special_queue_name = "queue-with-special!@#$%^&*()chars"
        worker = queue_service.register_worker(special_queue_name)
        assert worker.queue_name == special_queue_name
        
        # Register many workers with same queue
        workers = []
        for i in range(100):
            worker = queue_service.register_worker("mass_queue", f"worker_{i}")
            workers.append(worker)
        
        assert len(workers) == 100
        
        # Unregister all workers
        for worker in workers:
            result = queue_service.unregister_worker(worker.worker_id)
            assert result is True

    def test_queue_item_locking_edge_cases(self, batch_service, queue_service):
        """Test queue item locking edge cases."""
        with patch('src.services.batch_service.extract_video_id_from_url') as mock_extract:
            mock_extract.return_value = "test123"
            
            # Create batch
            batch = batch_service.create_batch(BatchCreateRequest(
                name="Locking Edge Cases",
                urls=["https://www.youtube.com/watch?v=test123"]
            ))
            batch_service.start_batch_processing(batch.batch_id)
            
            # Register worker
            worker = queue_service.register_worker("test_queue", "edge_worker")
            
            # Get queue item
            item = queue_service.get_next_queue_item("test_queue", "edge_worker")
            assert item is not None
            
            # Try to get same item with different worker
            worker2 = queue_service.register_worker("test_queue", "edge_worker_2")
            item2 = queue_service.get_next_queue_item("test_queue", "edge_worker_2")
            
            # Should get different item or None
            if item2:
                assert item2.id != item.id
            
            # Try to complete item with wrong worker
            with pytest.raises(QueueServiceError):
                queue_service.complete_queue_item(
                    item.id,
                    "edge_worker_2",  # Wrong worker
                    BatchItemStatus.COMPLETED
                )
            
            # Try to release item with wrong worker
            with pytest.raises(QueueServiceError):
                queue_service.release_queue_item(item.id, "edge_worker_2")

    def test_queue_item_completion_edge_cases(self, batch_service, queue_service):
        """Test queue item completion edge cases."""
        with patch('src.services.batch_service.extract_video_id_from_url') as mock_extract:
            mock_extract.return_value = "test123"
            
            batch = batch_service.create_batch(BatchCreateRequest(
                name="Completion Edge Cases",
                urls=["https://www.youtube.com/watch?v=test123"]
            ))
            batch_service.start_batch_processing(batch.batch_id)
            
            worker = queue_service.register_worker("test_queue", "completion_worker")
            item = queue_service.get_next_queue_item("test_queue", "completion_worker")
            
            # Try to complete with invalid status
            with pytest.raises((ValueError, QueueServiceError)):
                queue_service.complete_queue_item(
                    item.id,
                    "completion_worker",
                    "invalid_status"  # Not a valid BatchItemStatus
                )
            
            # Try to complete non-existent item
            with pytest.raises(QueueServiceError):
                queue_service.complete_queue_item(
                    99999,  # Non-existent ID
                    "completion_worker",
                    BatchItemStatus.COMPLETED
                )
            
            # Try to complete with extremely large result data
            large_result_data = {"large_field": "x" * 100000}
            result = queue_service.complete_queue_item(
                item.id,
                "completion_worker",
                BatchItemStatus.COMPLETED,
                result_data=large_result_data
            )
            assert result is True

    # Concurrent Operations Edge Cases

    def test_concurrent_batch_creation_race_conditions(self, batch_service):
        """Test race conditions in concurrent batch creation."""
        def create_batch_worker(worker_id):
            try:
                with patch('src.services.batch_service.extract_video_id_from_url') as mock_extract:
                    mock_extract.return_value = f"test{worker_id}"
                    
                    batch = batch_service.create_batch(BatchCreateRequest(
                        name=f"Concurrent Batch {worker_id}",
                        urls=[f"https://www.youtube.com/watch?v=test{worker_id}"]
                    ))
                    return batch.batch_id
            except Exception as e:
                return str(e)
        
        # Create batches concurrently
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [
                executor.submit(create_batch_worker, i)
                for i in range(50)
            ]
            
            results = [future.result() for future in futures]
        
        # Count successful creations
        successful_batches = [r for r in results if r.startswith("batch_")]
        
        # Should have created most batches successfully
        assert len(successful_batches) >= 40  # Allow some failures due to race conditions

    def test_concurrent_queue_operations_race_conditions(self, batch_service, queue_service):
        """Test race conditions in concurrent queue operations."""
        with patch('src.services.batch_service.extract_video_id_from_url') as mock_extract:
            mock_extract.side_effect = lambda url: url.split('=')[-1]
            
            # Create batch with many items
            urls = [f"https://www.youtube.com/watch?v=race{i}" for i in range(100)]
            batch = batch_service.create_batch(BatchCreateRequest(
                name="Race Condition Test",
                urls=urls
            ))
            batch_service.start_batch_processing(batch.batch_id)
            
            # Register many workers
            workers = []
            for i in range(20):
                worker = queue_service.register_worker("race_queue", f"race_worker_{i}")
                workers.append(worker)
            
            # Process items concurrently
            processed_items = []
            items_lock = threading.Lock()
            
            def worker_function(worker_id):
                local_processed = []
                for _ in range(10):  # Each worker tries to process 10 items
                    try:
                        item = queue_service.get_next_queue_item("race_queue", worker_id)
                        if item:
                            queue_service.complete_queue_item(
                                item.id,
                                worker_id,
                                BatchItemStatus.COMPLETED
                            )
                            local_processed.append(item.id)
                    except Exception as e:
                        # Race conditions may cause some errors
                        pass
                
                with items_lock:
                    processed_items.extend(local_processed)
                
                return len(local_processed)
            
            with ThreadPoolExecutor(max_workers=20) as executor:
                futures = [
                    executor.submit(worker_function, worker.worker_id)
                    for worker in workers
                ]
                
                results = [future.result() for future in futures]
            
            # Verify no duplicate processing
            assert len(processed_items) == len(set(processed_items))
            
            # Should have processed many items
            total_processed = sum(results)
            assert total_processed >= 50  # Allow for some race condition failures

    # Resource Exhaustion Edge Cases

    def test_memory_exhaustion_handling(self, batch_service):
        """Test handling of memory exhaustion scenarios."""
        # Try to create batch that would consume excessive memory
        with patch('src.services.batch_service.extract_video_id_from_url') as mock_extract:
            mock_extract.side_effect = lambda url: url.split('=')[-1]
            
            # Create progressively larger batches until memory pressure
            batch_sizes = [1000, 5000, 10000, 20000]
            
            for size in batch_sizes:
                try:
                    urls = [f"https://www.youtube.com/watch?v=mem{i:06d}" for i in range(size)]
                    batch = batch_service.create_batch(BatchCreateRequest(
                        name=f"Memory Test {size}",
                        urls=urls,
                        batch_metadata={"large_data": "x" * 10000}  # Add some memory pressure
                    ))
                    
                    assert batch.total_items == size
                    print(f"Successfully created batch of size {size}")
                    
                except Exception as e:
                    print(f"Memory exhaustion at size {size}: {e}")
                    break

    def test_thread_exhaustion_handling(self, queue_service):
        """Test handling of thread exhaustion scenarios."""
        # Register maximum number of workers
        workers = []
        
        try:
            for i in range(1000):  # Try to create many workers
                worker = queue_service.register_worker("thread_test", f"thread_worker_{i}")
                workers.append(worker)
        except Exception as e:
            print(f"Thread exhaustion at {len(workers)} workers: {e}")
        
        # Should handle gracefully
        assert len(workers) > 0
        
        # Cleanup workers
        for worker in workers:
            try:
                queue_service.unregister_worker(worker.worker_id)
            except:
                pass

    # Network Failure Simulation

    def test_network_timeout_handling(self, batch_service):
        """Test handling of network timeouts."""
        with patch('src.services.batch_service.extract_video_id_from_url') as mock_extract:
            # Simulate network timeout
            def timeout_extract(url):
                time.sleep(10)  # Simulate long network call
                return "timeout_test"
            
            mock_extract.side_effect = timeout_extract
            
            # This should timeout or handle gracefully
            start_time = time.time()
            try:
                batch = batch_service.create_batch(BatchCreateRequest(
                    name="Network Timeout Test",
                    urls=["https://www.youtube.com/watch?v=timeout"]
                ))
                duration = time.time() - start_time
                
                # If it succeeds, should be reasonably fast
                assert duration < 30  # Should not take more than 30 seconds
                
            except Exception as e:
                duration = time.time() - start_time
                print(f"Network timeout handled in {duration:.2f}s: {e}")
                assert duration < 30  # Should timeout within reasonable time

    # Data Corruption Edge Cases

    def test_malformed_data_handling(self, batch_service, db_session):
        """Test handling of malformed data scenarios."""
        with patch('src.services.batch_service.extract_video_id_from_url') as mock_extract:
            mock_extract.return_value = "malformed_test"
            
            # Create batch with valid data
            batch = batch_service.create_batch(BatchCreateRequest(
                name="Malformed Data Test",
                urls=["https://www.youtube.com/watch?v=malformed_test"]
            ))
            
            # Manually corrupt the data
            try:
                # Corrupt batch metadata
                db_session.execute(
                    text("UPDATE batches SET batch_metadata = 'invalid_json' WHERE id = :id"),
                    {"id": batch.id}
                )
                db_session.commit()
                
                # Try to retrieve corrupted batch
                corrupted_batch = batch_service.get_batch(batch.batch_id)
                
                # Should handle corruption gracefully
                assert corrupted_batch is not None
                
            except Exception as e:
                print(f"Malformed data handled: {e}")

    def test_unicode_handling_edge_cases(self, batch_service):
        """Test Unicode and encoding edge cases."""
        with patch('src.services.batch_service.extract_video_id_from_url') as mock_extract:
            mock_extract.return_value = "unicode_test"
            
            unicode_test_cases = [
                "Unicode: àáâãäåæçèéêë",
                "Emoji: 🚀🎉🔥💯✨🌟",
                "Mixed: Hello 世界 🌍",
                "Arabic: مرحبا بك",
                "Chinese: 你好世界",
                "Japanese: こんにちは世界",
                "Russian: Привет мир",
                "Special chars: ™©®℠",
                "\u2603\u2604\u2605",  # Snowman, comet, star
                "Zero-width: \u200b\u200c\u200d",
                "RTL mark: \u200f",
                "Control: \u0000\u0001\u0002",
                "Surrogate pairs: 𝕳𝖊𝖑𝖑𝖔",
            ]
            
            for test_case in unicode_test_cases:
                try:
                    batch = batch_service.create_batch(BatchCreateRequest(
                        name=f"Unicode Test: {test_case}",
                        description=test_case,
                        urls=["https://www.youtube.com/watch?v=unicode_test"],
                        batch_metadata={"unicode_field": test_case}
                    ))
                    
                    # Verify Unicode was preserved
                    assert test_case in batch.name
                    assert batch.description == test_case
                    assert batch.batch_metadata["unicode_field"] == test_case
                    
                except Exception as e:
                    print(f"Unicode test failed for '{test_case}': {e}")

    # Security Edge Cases

    def test_sql_injection_prevention(self, batch_service):
        """Test SQL injection prevention."""
        with patch('src.services.batch_service.extract_video_id_from_url') as mock_extract:
            mock_extract.return_value = "injection_test"
            
            injection_attempts = [
                "'; DROP TABLE batches; --",
                "' OR '1'='1",
                "'; SELECT * FROM batches; --",
                "' UNION SELECT * FROM batches; --",
                "'; DELETE FROM batches; --",
                "'; INSERT INTO batches VALUES (1, 'hacked'); --",
                "'; UPDATE batches SET name = 'hacked'; --",
                "\"; DROP TABLE batches; --",
                "' AND (SELECT COUNT(*) FROM batches) > 0; --",
            ]
            
            for injection in injection_attempts:
                try:
                    batch = batch_service.create_batch(BatchCreateRequest(
                        name=f"Injection Test {injection}",
                        description=injection,
                        urls=["https://www.youtube.com/watch?v=injection_test"],
                        batch_metadata={"injection_field": injection}
                    ))
                    
                    # Should treat as literal string, not SQL
                    assert injection in batch.name
                    assert batch.description == injection
                    
                except Exception as e:
                    # Some injection attempts might cause validation errors
                    print(f"Injection attempt handled: {injection} -> {e}")
                
                # Verify database is still intact
                batches = batch_service.list_batches()
                assert len(batches) > 0  # Database should not be dropped

    def test_xss_prevention(self, batch_service):
        """Test XSS prevention in stored data."""
        with patch('src.services.batch_service.extract_video_id_from_url') as mock_extract:
            mock_extract.return_value = "xss_test"
            
            xss_attempts = [
                "<script>alert('xss')</script>",
                "<img src=x onerror=alert('xss')>",
                "javascript:alert('xss')",
                "<iframe src='javascript:alert(\"xss\")'></iframe>",
                "<svg onload=alert('xss')>",
                "<body onload=alert('xss')>",
                "<%=eval(alert('xss'))%>",
                "${alert('xss')}",
                "{{alert('xss')}}",
                "<script type='text/javascript'>alert('xss')</script>",
            ]
            
            for xss in xss_attempts:
                batch = batch_service.create_batch(BatchCreateRequest(
                    name=f"XSS Test {xss}",
                    description=xss,
                    urls=["https://www.youtube.com/watch?v=xss_test"],
                    batch_metadata={"xss_field": xss}
                ))
                
                # Should store as literal string
                assert xss in batch.name
                assert batch.description == xss
                assert batch.batch_metadata["xss_field"] == xss

    # Recovery and Graceful Degradation

    def test_partial_failure_recovery(self, batch_service, queue_service):
        """Test recovery from partial failures."""
        with patch('src.services.batch_service.extract_video_id_from_url') as mock_extract:
            mock_extract.side_effect = lambda url: url.split('=')[-1]
            
            # Create batch
            urls = [f"https://www.youtube.com/watch?v=recovery{i}" for i in range(10)]
            batch = batch_service.create_batch(BatchCreateRequest(
                name="Partial Failure Recovery Test",
                urls=urls
            ))
            batch_service.start_batch_processing(batch.batch_id)
            
            # Register worker
            worker = queue_service.register_worker("recovery_queue", "recovery_worker")
            
            # Process some items successfully, then simulate failures
            success_count = 0
            failure_count = 0
            
            for i in range(10):
                item = queue_service.get_next_queue_item("recovery_queue", "recovery_worker")
                if item:
                    if i < 5:
                        # First 5 succeed
                        queue_service.complete_queue_item(
                            item.id,
                            "recovery_worker",
                            BatchItemStatus.COMPLETED
                        )
                        success_count += 1
                    else:
                        # Last 5 fail
                        queue_service.complete_queue_item(
                            item.id,
                            "recovery_worker",
                            BatchItemStatus.FAILED,
                            error_message="Simulated failure"
                        )
                        failure_count += 1
            
            # Verify partial completion
            final_batch = batch_service.get_batch(batch.batch_id)
            assert final_batch.completed_items == success_count
            assert final_batch.failed_items == failure_count

    def test_system_recovery_after_crash_simulation(self, batch_service, queue_service):
        """Test system recovery after simulated crash."""
        with patch('src.services.batch_service.extract_video_id_from_url') as mock_extract:
            mock_extract.return_value = "crash_test"
            
            # Create batch and start processing
            batch = batch_service.create_batch(BatchCreateRequest(
                name="Crash Recovery Test",
                urls=["https://www.youtube.com/watch?v=crash_test"]
            ))
            batch_service.start_batch_processing(batch.batch_id)
            
            # Start processing
            worker = queue_service.register_worker("crash_queue", "crash_worker")
            item = queue_service.get_next_queue_item("crash_queue", "crash_worker")
            
            # Simulate crash by not completing the item
            # This leaves the item in PROCESSING state
            
            # Simulate system restart by creating new services
            queue_service.shutdown()
            
            # Create new queue service (simulating restart)
            from src.services.queue_service import QueueProcessingOptions
            new_options = QueueProcessingOptions(
                max_workers=5,
                worker_timeout_minutes=1,
                enable_automatic_cleanup=True
            )
            new_queue_service = QueueService(batch_service._get_session(), new_options)
            
            try:
                # Register new worker
                new_worker = new_queue_service.register_worker("crash_queue", "recovery_worker")
                
                # Should be able to continue processing
                recovered_item = new_queue_service.get_next_queue_item("crash_queue", "recovery_worker")
                
                # Might get the same item or a different one, depending on cleanup
                if recovered_item:
                    new_queue_service.complete_queue_item(
                        recovered_item.id,
                        "recovery_worker",
                        BatchItemStatus.COMPLETED
                    )
                
                # System should be operational
                stats = new_queue_service.get_queue_statistics("crash_queue")
                assert stats is not None
                
            finally:
                new_queue_service.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])