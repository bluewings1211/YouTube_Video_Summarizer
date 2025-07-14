"""
Unit tests for concurrency manager utilities.

This module tests the comprehensive concurrency control mechanisms including:
- Thread-safe operations and synchronization primitives
- Resource management and locking mechanisms
- Rate limiting and throttling controls
- Deadlock prevention and detection
- Performance monitoring and metrics
"""

import unittest
import threading
import time
import asyncio
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import uuid

from .concurrency_manager import (
    ConcurrencyManager, ThreadSafeLock, ResourceManager, RateLimiter,
    ResourceType, ResourceQuota, LockType, ConcurrencyState,
    ConcurrencyError, DeadlockError, ResourceExhaustedError,
    TimeoutError as ConcurrencyTimeoutError, RateLimitedError,
    get_global_concurrency_manager, acquire_shared_lock, acquire_exclusive_lock,
    allocate_resource, get_concurrency_statistics
)


class TestThreadSafeLock(unittest.TestCase):
    """Test cases for ThreadSafeLock class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.lock = ThreadSafeLock(
            lock_id="test_lock",
            max_shared=5,
            timeout_seconds=2.0,
            enable_deadlock_detection=True
        )
    
    def tearDown(self):
        """Clean up after tests."""
        pass
    
    def test_shared_lock_acquisition(self):
        """Test shared lock acquisition and release."""
        # Test single shared lock
        with self.lock.acquire_shared("owner1", timeout=1.0):
            lock_info = self.lock.get_lock_info()
            self.assertEqual(lock_info['shared_count'], 1)
            self.assertIn("owner1", lock_info['shared_owners'])
        
        # Lock should be released
        lock_info = self.lock.get_lock_info()
        self.assertEqual(lock_info['shared_count'], 0)
        self.assertEqual(len(lock_info['shared_owners']), 0)
    
    def test_multiple_shared_locks(self):
        """Test multiple shared locks can be acquired concurrently."""
        acquired_locks = []
        
        def acquire_shared_lock(owner_id):
            try:
                with self.lock.acquire_shared(owner_id, timeout=1.0):
                    acquired_locks.append(owner_id)
                    time.sleep(0.5)  # Hold lock briefly
            except Exception as e:
                self.fail(f"Failed to acquire shared lock for {owner_id}: {e}")
        
        # Start multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=acquire_shared_lock, args=(f"owner{i}",))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # All locks should have been acquired
        self.assertEqual(len(acquired_locks), 3)
    
    def test_exclusive_lock_acquisition(self):
        """Test exclusive lock acquisition and release."""
        with self.lock.acquire_exclusive("owner1", timeout=1.0):
            lock_info = self.lock.get_lock_info()
            self.assertEqual(lock_info['exclusive_owner'], "owner1")
            self.assertEqual(lock_info['shared_count'], 0)
        
        # Lock should be released
        lock_info = self.lock.get_lock_info()
        self.assertIsNone(lock_info['exclusive_owner'])
    
    def test_exclusive_blocks_shared(self):
        """Test that exclusive lock blocks shared locks."""
        acquired_shared = threading.Event()
        
        def try_acquire_shared():
            try:
                with self.lock.acquire_shared("shared_owner", timeout=0.5):
                    acquired_shared.set()
            except ConcurrencyTimeoutError:
                pass  # Expected
        
        # Acquire exclusive lock
        with self.lock.acquire_exclusive("exclusive_owner", timeout=1.0):
            # Try to acquire shared lock in another thread
            thread = threading.Thread(target=try_acquire_shared)
            thread.start()
            thread.join()
        
        # Shared lock should not have been acquired
        self.assertFalse(acquired_shared.is_set())
    
    def test_shared_blocks_exclusive(self):
        """Test that shared locks block exclusive locks."""
        acquired_exclusive = threading.Event()
        
        def try_acquire_exclusive():
            try:
                with self.lock.acquire_exclusive("exclusive_owner", timeout=0.5):
                    acquired_exclusive.set()
            except ConcurrencyTimeoutError:
                pass  # Expected
        
        # Acquire shared lock
        with self.lock.acquire_shared("shared_owner", timeout=1.0):
            # Try to acquire exclusive lock in another thread
            thread = threading.Thread(target=try_acquire_exclusive)
            thread.start()
            thread.join()
        
        # Exclusive lock should not have been acquired
        self.assertFalse(acquired_exclusive.is_set())
    
    def test_lock_timeout(self):
        """Test lock acquisition timeout."""
        # Acquire exclusive lock
        with self.lock.acquire_exclusive("owner1", timeout=1.0):
            # Try to acquire another exclusive lock with timeout
            with self.assertRaises(ConcurrencyTimeoutError):
                with self.lock.acquire_exclusive("owner2", timeout=0.1):
                    pass
    
    def test_deadlock_detection(self):
        """Test deadlock detection."""
        # Try to acquire the same lock twice from the same owner
        with self.lock.acquire_shared("owner1", timeout=1.0):
            with self.assertRaises(DeadlockError):
                with self.lock.acquire_exclusive("owner1", timeout=1.0):
                    pass
    
    def test_max_shared_limit(self):
        """Test maximum shared lock limit."""
        acquired_locks = []
        timeout_errors = []
        
        def try_acquire_shared(owner_id):
            try:
                with self.lock.acquire_shared(owner_id, timeout=0.5):
                    acquired_locks.append(owner_id)
                    time.sleep(1.0)  # Hold lock
            except ConcurrencyTimeoutError:
                timeout_errors.append(owner_id)
        
        # Try to acquire more than max_shared locks
        threads = []
        for i in range(7):  # max_shared is 5
            thread = threading.Thread(target=try_acquire_shared, args=(f"owner{i}",))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Should have acquired max_shared locks and timed out on others
        self.assertEqual(len(acquired_locks), 5)
        self.assertEqual(len(timeout_errors), 2)


class TestResourceManager(unittest.TestCase):
    """Test cases for ResourceManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.resource_manager = ResourceManager(
            manager_id="test_manager",
            default_quota=ResourceQuota(
                max_concurrent=3,
                max_per_second=2.0,
                burst_capacity=5,
                timeout_seconds=2.0
            )
        )
    
    def tearDown(self):
        """Clean up after tests."""
        pass
    
    def test_resource_allocation(self):
        """Test basic resource allocation."""
        with self.resource_manager.allocate_resource(
            ResourceType.DATABASE_CONNECTION,
            "owner1",
            priority="normal",
            timeout=1.0
        ) as allocation:
            self.assertIsNotNone(allocation)
            self.assertEqual(allocation.resource_type, ResourceType.DATABASE_CONNECTION)
            self.assertEqual(allocation.allocated_to, "owner1")
            
            # Check statistics
            stats = self.resource_manager.get_resource_statistics()
            self.assertEqual(stats['total_allocated_resources'], 1)
    
    def test_resource_quota_enforcement(self):
        """Test that resource quotas are enforced."""
        allocations = []
        
        # Allocate up to the quota
        try:
            for i in range(3):  # max_concurrent is 3
                allocation = self.resource_manager.allocate_resource(
                    ResourceType.DATABASE_CONNECTION,
                    f"owner{i}",
                    priority="normal",
                    timeout=1.0
                )
                allocations.append(allocation)
                allocation.__enter__()
        except Exception as e:
            self.fail(f"Failed to allocate resources within quota: {e}")
        
        # Try to allocate beyond quota
        with self.assertRaises(ResourceExhaustedError):
            with self.resource_manager.allocate_resource(
                ResourceType.DATABASE_CONNECTION,
                "owner_extra",
                priority="normal",
                timeout=0.1
            ):
                pass
        
        # Clean up
        for allocation in allocations:
            allocation.__exit__(None, None, None)
    
    def test_resource_cleanup(self):
        """Test resource cleanup functionality."""
        # Allocate some resources
        allocations = []
        for i in range(2):
            allocation = self.resource_manager.allocate_resource(
                ResourceType.DATABASE_CONNECTION,
                f"owner{i}",
                priority="normal",
                timeout=1.0
            )
            allocations.append(allocation)
            allocation.__enter__()
        
        # Simulate stale allocations by manipulating timestamps
        for allocation in allocations:
            allocation.allocated_at = datetime.utcnow() - timedelta(seconds=400)
        
        # Clean up stale allocations
        cleaned_count = self.resource_manager.cleanup_stale_allocations(max_age_seconds=300)
        self.assertEqual(cleaned_count, 2)
        
        # Check that resources were cleaned up
        stats = self.resource_manager.get_resource_statistics()
        self.assertEqual(stats['total_allocated_resources'], 0)
    
    def test_resource_statistics(self):
        """Test resource statistics collection."""
        # Allocate some resources
        with self.resource_manager.allocate_resource(
            ResourceType.DATABASE_CONNECTION,
            "owner1",
            priority="normal"
        ):
            with self.resource_manager.allocate_resource(
                ResourceType.API_REQUEST,
                "owner2",
                priority="high"
            ):
                stats = self.resource_manager.get_resource_statistics()
                
                self.assertEqual(stats['total_allocated_resources'], 2)
                self.assertIn('resource_breakdown', stats)
                self.assertIn('metrics', stats)
                
                # Check resource breakdown
                breakdown = stats['resource_breakdown']
                self.assertIn(ResourceType.DATABASE_CONNECTION.value, breakdown)
                self.assertIn(ResourceType.API_REQUEST.value, breakdown)


class TestRateLimiter(unittest.TestCase):
    """Test cases for RateLimiter class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.rate_limiter = RateLimiter(
            max_rate=2.0,  # 2 requests per second
            burst_capacity=5,
            refill_period=1.0
        )
    
    def tearDown(self):
        """Clean up after tests."""
        pass
    
    def test_basic_rate_limiting(self):
        """Test basic rate limiting functionality."""
        # Should allow requests within burst capacity
        for i in range(5):
            self.assertTrue(self.rate_limiter.acquire())
        
        # Should deny requests beyond burst capacity
        self.assertFalse(self.rate_limiter.acquire())
    
    def test_token_refill(self):
        """Test token bucket refill mechanism."""
        # Exhaust burst capacity
        for i in range(5):
            self.assertTrue(self.rate_limiter.acquire())
        
        # Should be denied immediately
        self.assertFalse(self.rate_limiter.acquire())
        
        # Wait for refill
        time.sleep(1.5)  # Wait for > 1 second
        
        # Should allow requests again
        self.assertTrue(self.rate_limiter.acquire())
    
    def test_rate_limiter_statistics(self):
        """Test rate limiter statistics."""
        # Make some requests
        for i in range(3):
            self.rate_limiter.acquire()
        
        # Make a denied request
        self.rate_limiter.acquire()  # Should be denied
        
        stats = self.rate_limiter.get_statistics()
        self.assertEqual(stats['total_requests'], 4)
        self.assertEqual(stats['allowed_requests'], 3)
        self.assertEqual(stats['denied_requests'], 1)
        self.assertEqual(stats['success_rate'], 3/4)


class TestConcurrencyManager(unittest.TestCase):
    """Test cases for ConcurrencyManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.manager = ConcurrencyManager(
            manager_id="test_manager",
            enable_monitoring=True,
            cleanup_interval=1
        )
    
    def tearDown(self):
        """Clean up after tests."""
        self.manager.shutdown()
    
    def test_lock_creation_and_retrieval(self):
        """Test lock creation and retrieval."""
        lock1 = self.manager.get_lock("test_lock_1")
        lock2 = self.manager.get_lock("test_lock_1")  # Same lock
        lock3 = self.manager.get_lock("test_lock_2")  # Different lock
        
        self.assertIs(lock1, lock2)  # Same instance
        self.assertIsNot(lock1, lock3)  # Different instances
    
    def test_resource_quota_configuration(self):
        """Test resource quota configuration."""
        quota = ResourceQuota(
            max_concurrent=10,
            max_per_second=5.0,
            burst_capacity=15,
            timeout_seconds=30.0
        )
        
        self.manager.configure_resource_quota(ResourceType.DATABASE_CONNECTION, quota)
        
        # Test that quota is applied
        allocations = []
        try:
            for i in range(10):
                allocation = self.manager.allocate_resource(
                    ResourceType.DATABASE_CONNECTION,
                    f"owner{i}",
                    priority="normal",
                    timeout=1.0
                )
                allocations.append(allocation)
                allocation.__enter__()
        except Exception as e:
            self.fail(f"Failed to allocate resources within quota: {e}")
        
        # Should fail beyond quota
        with self.assertRaises(ResourceExhaustedError):
            with self.manager.allocate_resource(
                ResourceType.DATABASE_CONNECTION,
                "owner_extra",
                priority="normal",
                timeout=0.1
            ):
                pass
        
        # Clean up
        for allocation in allocations:
            allocation.__exit__(None, None, None)
    
    def test_comprehensive_statistics(self):
        """Test comprehensive statistics collection."""
        # Create some locks and allocate resources
        lock = self.manager.get_lock("test_lock")
        
        with lock.acquire_shared("owner1", timeout=1.0):
            with self.manager.allocate_resource(
                ResourceType.DATABASE_CONNECTION,
                "owner1",
                priority="normal"
            ):
                stats = self.manager.get_comprehensive_statistics()
                
                self.assertIn('manager_id', stats)
                self.assertIn('active_locks', stats)
                self.assertIn('lock_details', stats)
                self.assertIn('resource_statistics', stats)
                self.assertIn('global_metrics', stats)
                
                self.assertEqual(stats['manager_id'], "test_manager")
                self.assertEqual(stats['active_locks'], 1)
                self.assertIn('test_lock', stats['lock_details'])
    
    def test_context_manager(self):
        """Test context manager functionality."""
        with ConcurrencyManager(manager_id="context_test") as manager:
            lock = manager.get_lock("test_lock")
            with lock.acquire_shared("owner1", timeout=1.0):
                pass
        
        # Manager should be shut down automatically


class TestGlobalConcurrencyManager(unittest.TestCase):
    """Test cases for global concurrency manager functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        pass
    
    def tearDown(self):
        """Clean up after tests."""
        pass
    
    def test_global_manager_singleton(self):
        """Test that global manager is a singleton."""
        manager1 = get_global_concurrency_manager()
        manager2 = get_global_concurrency_manager()
        
        self.assertIs(manager1, manager2)
    
    def test_convenience_functions(self):
        """Test convenience functions."""
        # Test shared lock acquisition
        with acquire_shared_lock("test_lock", "owner1", timeout=1.0):
            pass
        
        # Test exclusive lock acquisition
        with acquire_exclusive_lock("test_lock", "owner1", timeout=1.0):
            pass
        
        # Test resource allocation
        with allocate_resource(
            ResourceType.DATABASE_CONNECTION,
            "owner1",
            priority="normal",
            timeout=1.0
        ):
            pass
        
        # Test statistics retrieval
        stats = get_concurrency_statistics()
        self.assertIsInstance(stats, dict)
        self.assertIn('manager_id', stats)


class TestConcurrencyIntegration(unittest.TestCase):
    """Integration tests for concurrency management."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.manager = ConcurrencyManager(
            manager_id="integration_test",
            enable_monitoring=True,
            cleanup_interval=1
        )
    
    def tearDown(self):
        """Clean up after tests."""
        self.manager.shutdown()
    
    def test_concurrent_operations(self):
        """Test concurrent operations with multiple threads."""
        results = []
        errors = []
        
        def worker_function(worker_id):
            try:
                # Acquire lock
                lock = self.manager.get_lock("shared_resource")
                with lock.acquire_shared(f"worker_{worker_id}", timeout=2.0):
                    # Allocate resource
                    with self.manager.allocate_resource(
                        ResourceType.DATABASE_CONNECTION,
                        f"worker_{worker_id}",
                        priority="normal",
                        timeout=2.0
                    ):
                        # Simulate work
                        time.sleep(0.1)
                        results.append(worker_id)
            except Exception as e:
                errors.append((worker_id, str(e)))
        
        # Start multiple workers
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker_function, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # All workers should have completed successfully
        self.assertEqual(len(results), 5)
        self.assertEqual(len(errors), 0)
    
    def test_resource_contention(self):
        """Test resource contention handling."""
        # Configure tight resource limits
        quota = ResourceQuota(
            max_concurrent=2,
            max_per_second=1.0,
            burst_capacity=3,
            timeout_seconds=1.0
        )
        self.manager.configure_resource_quota(ResourceType.DATABASE_CONNECTION, quota)
        
        successful_allocations = []
        failed_allocations = []
        
        def try_allocate_resource(worker_id):
            try:
                with self.manager.allocate_resource(
                    ResourceType.DATABASE_CONNECTION,
                    f"worker_{worker_id}",
                    priority="normal",
                    timeout=0.5
                ):
                    successful_allocations.append(worker_id)
                    time.sleep(0.2)  # Hold resource briefly
            except (ResourceExhaustedError, ConcurrencyTimeoutError):
                failed_allocations.append(worker_id)
        
        # Start more workers than available resources
        threads = []
        for i in range(5):
            thread = threading.Thread(target=try_allocate_resource, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Some should succeed, some should fail
        self.assertTrue(len(successful_allocations) > 0)
        self.assertTrue(len(failed_allocations) > 0)
        self.assertEqual(len(successful_allocations) + len(failed_allocations), 5)
    
    def test_deadlock_prevention(self):
        """Test deadlock prevention mechanisms."""
        lock1 = self.manager.get_lock("lock1")
        lock2 = self.manager.get_lock("lock2")
        
        deadlock_detected = threading.Event()
        
        def worker1():
            try:
                with lock1.acquire_exclusive("worker1", timeout=2.0):
                    time.sleep(0.1)
                    with lock2.acquire_exclusive("worker1", timeout=1.0):
                        pass
            except DeadlockError:
                deadlock_detected.set()
        
        def worker2():
            try:
                with lock2.acquire_exclusive("worker2", timeout=2.0):
                    time.sleep(0.1)
                    with lock1.acquire_exclusive("worker2", timeout=1.0):
                        pass
            except (DeadlockError, ConcurrencyTimeoutError):
                pass  # Expected
        
        # Start both workers
        thread1 = threading.Thread(target=worker1)
        thread2 = threading.Thread(target=worker2)
        
        thread1.start()
        thread2.start()
        
        thread1.join()
        thread2.join()
        
        # One of the operations should have detected potential deadlock or timed out
        # This test verifies that the system doesn't hang indefinitely


class TestPerformanceAndStress(unittest.TestCase):
    """Performance and stress tests for concurrency management."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.manager = ConcurrencyManager(
            manager_id="performance_test",
            enable_monitoring=True,
            cleanup_interval=5
        )
    
    def tearDown(self):
        """Clean up after tests."""
        self.manager.shutdown()
    
    def test_high_concurrency_locks(self):
        """Test lock performance with high concurrency."""
        num_threads = 50
        num_operations = 100
        lock = self.manager.get_lock("performance_lock", max_shared=20)
        
        completed_operations = []
        
        def worker_function(worker_id):
            for i in range(num_operations):
                try:
                    with lock.acquire_shared(f"worker_{worker_id}", timeout=5.0):
                        # Simulate brief work
                        time.sleep(0.001)
                        completed_operations.append((worker_id, i))
                except Exception as e:
                    self.fail(f"Worker {worker_id} failed: {e}")
        
        # Start all workers
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(worker_function, i)
                for i in range(num_threads)
            ]
            
            # Wait for all to complete
            for future in as_completed(futures):
                future.result()
        
        end_time = time.time()
        
        # Verify all operations completed
        expected_operations = num_threads * num_operations
        self.assertEqual(len(completed_operations), expected_operations)
        
        # Log performance metrics
        duration = end_time - start_time
        ops_per_second = expected_operations / duration
        
        self.assertGreater(ops_per_second, 100)  # Should handle at least 100 ops/sec
        print(f"Performance test: {ops_per_second:.2f} operations/second")
    
    def test_resource_allocation_stress(self):
        """Test resource allocation under stress."""
        num_threads = 30
        num_allocations = 50
        
        # Configure resource quota
        quota = ResourceQuota(
            max_concurrent=10,
            max_per_second=50.0,
            burst_capacity=20,
            timeout_seconds=5.0
        )
        self.manager.configure_resource_quota(ResourceType.DATABASE_CONNECTION, quota)
        
        completed_allocations = []
        failed_allocations = []
        
        def worker_function(worker_id):
            for i in range(num_allocations):
                try:
                    with self.manager.allocate_resource(
                        ResourceType.DATABASE_CONNECTION,
                        f"worker_{worker_id}_{i}",
                        priority="normal",
                        timeout=2.0
                    ):
                        # Simulate work
                        time.sleep(0.01)
                        completed_allocations.append((worker_id, i))
                except (ResourceExhaustedError, ConcurrencyTimeoutError):
                    failed_allocations.append((worker_id, i))
        
        # Start all workers
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(worker_function, i)
                for i in range(num_threads)
            ]
            
            # Wait for all to complete
            for future in as_completed(futures):
                future.result()
        
        # Verify that some operations completed
        total_attempts = num_threads * num_allocations
        self.assertGreater(len(completed_allocations), 0)
        self.assertEqual(len(completed_allocations) + len(failed_allocations), total_attempts)
        
        # Log results
        success_rate = len(completed_allocations) / total_attempts
        print(f"Stress test: {success_rate:.2f} success rate under resource contention")


if __name__ == '__main__':
    unittest.main()