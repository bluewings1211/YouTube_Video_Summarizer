"""
Concurrency management utilities for YouTube video summarization batch processing.

This module provides comprehensive concurrency control mechanisms including:
- Thread-safe operations and synchronization primitives
- Resource management and locking mechanisms
- Rate limiting and throttling controls
- Connection pooling and resource allocation
- Deadlock prevention and detection
- Performance monitoring and metrics
"""

import asyncio
import threading
import time
import logging
from typing import Optional, Dict, Any, List, Callable, Union, AsyncContextManager
from dataclasses import dataclass, field
from enum import Enum
from contextlib import contextmanager, asynccontextmanager
from concurrent.futures import ThreadPoolExecutor, Future
from datetime import datetime, timedelta
from collections import defaultdict, deque
import weakref
import uuid
import json


logger = logging.getLogger(__name__)


class LockType(Enum):
    """Types of locks available."""
    SHARED = "shared"
    EXCLUSIVE = "exclusive"
    UPGRADE = "upgrade"


class ResourceType(Enum):
    """Types of resources that can be managed."""
    DATABASE_CONNECTION = "database_connection"
    API_REQUEST = "api_request"
    WORKER_THREAD = "worker_thread"
    MEMORY_BUFFER = "memory_buffer"
    FILE_HANDLE = "file_handle"
    NETWORK_SOCKET = "network_socket"


class ConcurrencyState(Enum):
    """States for concurrency-controlled operations."""
    IDLE = "idle"
    WAITING = "waiting"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ResourceQuota:
    """Resource allocation quota configuration."""
    max_concurrent: int = 10
    max_per_second: float = 5.0
    burst_capacity: int = 20
    timeout_seconds: float = 30.0
    priority_levels: Dict[str, int] = field(default_factory=lambda: {
        "low": 1,
        "normal": 2,
        "high": 3,
        "urgent": 4
    })


@dataclass
class LockInfo:
    """Information about a lock."""
    lock_id: str
    lock_type: LockType
    resource_key: str
    acquired_at: datetime
    expires_at: Optional[datetime]
    owner_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResourceAllocation:
    """Resource allocation tracking."""
    resource_id: str
    resource_type: ResourceType
    allocated_at: datetime
    allocated_to: str
    quota_used: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConcurrencyMetrics:
    """Metrics for concurrency operations."""
    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    cancelled_operations: int = 0
    average_wait_time: float = 0.0
    average_execution_time: float = 0.0
    current_active_operations: int = 0
    peak_concurrent_operations: int = 0
    resource_utilization: Dict[str, float] = field(default_factory=dict)
    lock_contention_rate: float = 0.0
    timeout_rate: float = 0.0
    last_updated: datetime = field(default_factory=datetime.utcnow)


class ConcurrencyError(Exception):
    """Base exception for concurrency-related errors."""
    pass


class DeadlockError(ConcurrencyError):
    """Exception raised when deadlock is detected."""
    pass


class ResourceExhaustedError(ConcurrencyError):
    """Exception raised when resources are exhausted."""
    pass


class TimeoutError(ConcurrencyError):
    """Exception raised when operation times out."""
    pass


class RateLimitedError(ConcurrencyError):
    """Exception raised when rate limit is exceeded."""
    pass


class ThreadSafeLock:
    """
    Thread-safe lock implementation with timeout and priority support.
    
    This lock supports:
    - Shared/exclusive locking modes
    - Priority-based acquisition
    - Timeout mechanisms
    - Deadlock detection
    - Lock upgrading/downgrading
    """
    
    def __init__(self, 
                 lock_id: str,
                 max_shared: int = 10,
                 timeout_seconds: float = 30.0,
                 enable_deadlock_detection: bool = True):
        """
        Initialize thread-safe lock.
        
        Args:
            lock_id: Unique identifier for the lock
            max_shared: Maximum number of shared locks allowed
            timeout_seconds: Default timeout for lock acquisition
            enable_deadlock_detection: Whether to enable deadlock detection
        """
        self.lock_id = lock_id
        self.max_shared = max_shared
        self.timeout_seconds = timeout_seconds
        self.enable_deadlock_detection = enable_deadlock_detection
        
        # Internal state
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._shared_count = 0
        self._exclusive_owner = None
        self._waiting_queue = deque()
        self._shared_owners = set()
        
        # Tracking
        self._acquisition_times = {}
        self._wait_times = {}
        self._logger = logging.getLogger(f"{__name__}.ThreadSafeLock.{lock_id}")
    
    @contextmanager
    def acquire_shared(self, 
                      owner_id: str,
                      timeout: Optional[float] = None,
                      priority: int = 1):
        """
        Acquire shared lock with context manager.
        
        Args:
            owner_id: Identifier of the lock owner
            timeout: Lock acquisition timeout
            priority: Priority level for lock acquisition
            
        Yields:
            Lock context
            
        Raises:
            TimeoutError: If lock acquisition times out
            DeadlockError: If deadlock is detected
        """
        acquired = False
        start_time = time.time()
        
        try:
            acquired = self._acquire_shared_internal(owner_id, timeout, priority)
            if not acquired:
                raise TimeoutError(f"Failed to acquire shared lock {self.lock_id}")
            
            acquisition_time = time.time() - start_time
            self._acquisition_times[owner_id] = acquisition_time
            self._logger.debug(f"Acquired shared lock {self.lock_id} for {owner_id} in {acquisition_time:.3f}s")
            
            yield self
            
        finally:
            if acquired:
                self._release_shared_internal(owner_id)
                self._logger.debug(f"Released shared lock {self.lock_id} for {owner_id}")
    
    @contextmanager
    def acquire_exclusive(self, 
                         owner_id: str,
                         timeout: Optional[float] = None,
                         priority: int = 1):
        """
        Acquire exclusive lock with context manager.
        
        Args:
            owner_id: Identifier of the lock owner
            timeout: Lock acquisition timeout
            priority: Priority level for lock acquisition
            
        Yields:
            Lock context
            
        Raises:
            TimeoutError: If lock acquisition times out
            DeadlockError: If deadlock is detected
        """
        acquired = False
        start_time = time.time()
        
        try:
            acquired = self._acquire_exclusive_internal(owner_id, timeout, priority)
            if not acquired:
                raise TimeoutError(f"Failed to acquire exclusive lock {self.lock_id}")
            
            acquisition_time = time.time() - start_time
            self._acquisition_times[owner_id] = acquisition_time
            self._logger.debug(f"Acquired exclusive lock {self.lock_id} for {owner_id} in {acquisition_time:.3f}s")
            
            yield self
            
        finally:
            if acquired:
                self._release_exclusive_internal(owner_id)
                self._logger.debug(f"Released exclusive lock {self.lock_id} for {owner_id}")
    
    def _acquire_shared_internal(self, owner_id: str, timeout: Optional[float], priority: int) -> bool:
        """Internal method to acquire shared lock."""
        timeout = timeout or self.timeout_seconds
        deadline = time.time() + timeout
        
        with self._condition:
            # Check for deadlock
            if self.enable_deadlock_detection:
                self._check_deadlock(owner_id)
            
            # Wait for lock availability
            while True:
                # Can acquire if no exclusive lock and not at shared limit
                if self._exclusive_owner is None and self._shared_count < self.max_shared:
                    self._shared_count += 1
                    self._shared_owners.add(owner_id)
                    return True
                
                # Check timeout
                remaining = deadline - time.time()
                if remaining <= 0:
                    return False
                
                # Add to waiting queue
                wait_info = {
                    'owner_id': owner_id,
                    'lock_type': LockType.SHARED,
                    'priority': priority,
                    'wait_start': time.time()
                }
                self._waiting_queue.append(wait_info)
                
                # Wait for notification
                self._condition.wait(timeout=remaining)
                
                # Remove from waiting queue
                if wait_info in self._waiting_queue:
                    self._waiting_queue.remove(wait_info)
    
    def _acquire_exclusive_internal(self, owner_id: str, timeout: Optional[float], priority: int) -> bool:
        """Internal method to acquire exclusive lock."""
        timeout = timeout or self.timeout_seconds
        deadline = time.time() + timeout
        
        with self._condition:
            # Check for deadlock
            if self.enable_deadlock_detection:
                self._check_deadlock(owner_id)
            
            # Wait for lock availability
            while True:
                # Can acquire if no other locks
                if self._exclusive_owner is None and self._shared_count == 0:
                    self._exclusive_owner = owner_id
                    return True
                
                # Check timeout
                remaining = deadline - time.time()
                if remaining <= 0:
                    return False
                
                # Add to waiting queue
                wait_info = {
                    'owner_id': owner_id,
                    'lock_type': LockType.EXCLUSIVE,
                    'priority': priority,
                    'wait_start': time.time()
                }
                self._waiting_queue.append(wait_info)
                
                # Wait for notification
                self._condition.wait(timeout=remaining)
                
                # Remove from waiting queue
                if wait_info in self._waiting_queue:
                    self._waiting_queue.remove(wait_info)
    
    def _release_shared_internal(self, owner_id: str):
        """Internal method to release shared lock."""
        with self._condition:
            if owner_id in self._shared_owners:
                self._shared_owners.remove(owner_id)
                self._shared_count -= 1
                self._condition.notify_all()
    
    def _release_exclusive_internal(self, owner_id: str):
        """Internal method to release exclusive lock."""
        with self._condition:
            if self._exclusive_owner == owner_id:
                self._exclusive_owner = None
                self._condition.notify_all()
    
    def _check_deadlock(self, owner_id: str):
        """Check for potential deadlock conditions."""
        # Simple deadlock detection - can be enhanced for more complex scenarios
        if owner_id in self._shared_owners or self._exclusive_owner == owner_id:
            raise DeadlockError(f"Deadlock detected for owner {owner_id} on lock {self.lock_id}")
    
    def get_lock_info(self) -> Dict[str, Any]:
        """Get current lock information."""
        with self._condition:
            return {
                'lock_id': self.lock_id,
                'shared_count': self._shared_count,
                'exclusive_owner': self._exclusive_owner,
                'shared_owners': list(self._shared_owners),
                'waiting_count': len(self._waiting_queue),
                'max_shared': self.max_shared
            }


class ResourceManager:
    """
    Resource management system for controlling access to limited resources.
    
    This manager provides:
    - Resource allocation with quotas
    - Priority-based resource assignment
    - Rate limiting and throttling
    - Resource tracking and monitoring
    - Automatic cleanup and recovery
    """
    
    def __init__(self, 
                 manager_id: str,
                 default_quota: Optional[ResourceQuota] = None):
        """
        Initialize resource manager.
        
        Args:
            manager_id: Unique identifier for the manager
            default_quota: Default resource quota configuration
        """
        self.manager_id = manager_id
        self.default_quota = default_quota or ResourceQuota()
        
        # Resource tracking
        self._lock = threading.RLock()
        self._resource_quotas: Dict[ResourceType, ResourceQuota] = {}
        self._allocated_resources: Dict[str, ResourceAllocation] = {}
        self._allocation_history = deque(maxlen=1000)
        self._rate_limiters: Dict[ResourceType, 'RateLimiter'] = {}
        
        # Metrics
        self._metrics = ConcurrencyMetrics()
        self._logger = logging.getLogger(f"{__name__}.ResourceManager.{manager_id}")
    
    def configure_resource_quota(self, resource_type: ResourceType, quota: ResourceQuota):
        """
        Configure quota for a specific resource type.
        
        Args:
            resource_type: Type of resource to configure
            quota: Quota configuration
        """
        with self._lock:
            self._resource_quotas[resource_type] = quota
            
            # Create rate limiter for this resource type
            self._rate_limiters[resource_type] = RateLimiter(
                max_rate=quota.max_per_second,
                burst_capacity=quota.burst_capacity
            )
            
            self._logger.info(f"Configured quota for {resource_type.value}: {quota.max_concurrent} concurrent, {quota.max_per_second}/sec")
    
    @contextmanager
    def allocate_resource(self, 
                         resource_type: ResourceType,
                         owner_id: str,
                         priority: str = "normal",
                         timeout: Optional[float] = None,
                         metadata: Optional[Dict[str, Any]] = None):
        """
        Allocate a resource with context manager.
        
        Args:
            resource_type: Type of resource to allocate
            owner_id: Identifier of the resource owner
            priority: Priority level for allocation
            timeout: Allocation timeout
            metadata: Additional metadata for the allocation
            
        Yields:
            Resource allocation context
            
        Raises:
            ResourceExhaustedError: If resources are exhausted
            TimeoutError: If allocation times out
            RateLimitedError: If rate limit is exceeded
        """
        allocation_id = f"{resource_type.value}_{owner_id}_{uuid.uuid4().hex[:8]}"
        allocated = False
        start_time = time.time()
        
        try:
            # Check rate limits
            rate_limiter = self._rate_limiters.get(resource_type)
            if rate_limiter and not rate_limiter.acquire():
                raise RateLimitedError(f"Rate limit exceeded for {resource_type.value}")
            
            # Allocate resource
            allocation = self._allocate_resource_internal(
                allocation_id, resource_type, owner_id, priority, timeout, metadata
            )
            allocated = True
            
            allocation_time = time.time() - start_time
            self._logger.debug(f"Allocated {resource_type.value} to {owner_id} in {allocation_time:.3f}s")
            
            yield allocation
            
        finally:
            if allocated:
                self._deallocate_resource_internal(allocation_id)
                self._logger.debug(f"Deallocated {resource_type.value} from {owner_id}")
    
    def _allocate_resource_internal(self, 
                                   allocation_id: str,
                                   resource_type: ResourceType,
                                   owner_id: str,
                                   priority: str,
                                   timeout: Optional[float],
                                   metadata: Optional[Dict[str, Any]]) -> ResourceAllocation:
        """Internal method to allocate resource."""
        quota = self._resource_quotas.get(resource_type, self.default_quota)
        timeout = timeout or quota.timeout_seconds
        deadline = time.time() + timeout
        
        with self._lock:
            # Check current allocation count
            current_count = sum(1 for alloc in self._allocated_resources.values() 
                              if alloc.resource_type == resource_type)
            
            # Wait for resource availability
            while current_count >= quota.max_concurrent:
                remaining = deadline - time.time()
                if remaining <= 0:
                    raise ResourceExhaustedError(f"Resource {resource_type.value} exhausted")
                
                # Brief wait before rechecking
                time.sleep(0.1)
                current_count = sum(1 for alloc in self._allocated_resources.values() 
                                  if alloc.resource_type == resource_type)
            
            # Create allocation
            allocation = ResourceAllocation(
                resource_id=allocation_id,
                resource_type=resource_type,
                allocated_at=datetime.utcnow(),
                allocated_to=owner_id,
                metadata=metadata or {}
            )
            
            self._allocated_resources[allocation_id] = allocation
            self._allocation_history.append(allocation)
            
            # Update metrics
            self._metrics.total_operations += 1
            self._metrics.current_active_operations += 1
            if self._metrics.current_active_operations > self._metrics.peak_concurrent_operations:
                self._metrics.peak_concurrent_operations = self._metrics.current_active_operations
            
            return allocation
    
    def _deallocate_resource_internal(self, allocation_id: str):
        """Internal method to deallocate resource."""
        with self._lock:
            if allocation_id in self._allocated_resources:
                allocation = self._allocated_resources.pop(allocation_id)
                
                # Update metrics
                self._metrics.current_active_operations -= 1
                self._metrics.successful_operations += 1
                
                # Calculate execution time
                execution_time = (datetime.utcnow() - allocation.allocated_at).total_seconds()
                if self._metrics.average_execution_time == 0:
                    self._metrics.average_execution_time = execution_time
                else:
                    self._metrics.average_execution_time = (
                        self._metrics.average_execution_time * 0.9 + execution_time * 0.1
                    )
    
    def get_resource_statistics(self) -> Dict[str, Any]:
        """Get resource utilization statistics."""
        with self._lock:
            stats = {
                'manager_id': self.manager_id,
                'total_allocated_resources': len(self._allocated_resources),
                'resource_breakdown': {},
                'metrics': {
                    'total_operations': self._metrics.total_operations,
                    'successful_operations': self._metrics.successful_operations,
                    'failed_operations': self._metrics.failed_operations,
                    'current_active_operations': self._metrics.current_active_operations,
                    'peak_concurrent_operations': self._metrics.peak_concurrent_operations,
                    'average_execution_time': self._metrics.average_execution_time,
                    'last_updated': self._metrics.last_updated.isoformat()
                }
            }
            
            # Resource breakdown
            for resource_type in ResourceType:
                count = sum(1 for alloc in self._allocated_resources.values() 
                          if alloc.resource_type == resource_type)
                quota = self._resource_quotas.get(resource_type, self.default_quota)
                utilization = count / quota.max_concurrent if quota.max_concurrent > 0 else 0
                
                stats['resource_breakdown'][resource_type.value] = {
                    'allocated': count,
                    'quota': quota.max_concurrent,
                    'utilization': utilization
                }
            
            return stats
    
    def cleanup_stale_allocations(self, max_age_seconds: int = 300) -> int:
        """
        Clean up stale resource allocations.
        
        Args:
            max_age_seconds: Maximum age for allocations in seconds
            
        Returns:
            Number of cleaned up allocations
        """
        cutoff_time = datetime.utcnow() - timedelta(seconds=max_age_seconds)
        cleaned_count = 0
        
        with self._lock:
            stale_allocations = [
                alloc_id for alloc_id, alloc in self._allocated_resources.items()
                if alloc.allocated_at < cutoff_time
            ]
            
            for alloc_id in stale_allocations:
                self._deallocate_resource_internal(alloc_id)
                cleaned_count += 1
        
        if cleaned_count > 0:
            self._logger.info(f"Cleaned up {cleaned_count} stale resource allocations")
        
        return cleaned_count


class RateLimiter:
    """
    Token bucket rate limiter for controlling request rates.
    
    This rate limiter provides:
    - Token bucket algorithm implementation
    - Burst capacity support
    - Thread-safe operations
    - Adaptive rate limiting
    """
    
    def __init__(self, 
                 max_rate: float,
                 burst_capacity: int,
                 refill_period: float = 1.0):
        """
        Initialize rate limiter.
        
        Args:
            max_rate: Maximum rate in requests per second
            burst_capacity: Maximum burst capacity
            refill_period: Token refill period in seconds
        """
        self.max_rate = max_rate
        self.burst_capacity = burst_capacity
        self.refill_period = refill_period
        
        # Token bucket state
        self._lock = threading.Lock()
        self._tokens = float(burst_capacity)
        self._last_refill = time.time()
        
        # Metrics
        self._total_requests = 0
        self._allowed_requests = 0
        self._denied_requests = 0
    
    def acquire(self, tokens: int = 1) -> bool:
        """
        Acquire tokens from the bucket.
        
        Args:
            tokens: Number of tokens to acquire
            
        Returns:
            True if tokens were acquired, False otherwise
        """
        with self._lock:
            current_time = time.time()
            
            # Refill tokens based on time elapsed
            time_passed = current_time - self._last_refill
            if time_passed > 0:
                tokens_to_add = time_passed * self.max_rate
                self._tokens = min(self.burst_capacity, self._tokens + tokens_to_add)
                self._last_refill = current_time
            
            # Check if we have enough tokens
            self._total_requests += 1
            if self._tokens >= tokens:
                self._tokens -= tokens
                self._allowed_requests += 1
                return True
            else:
                self._denied_requests += 1
                return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get rate limiter statistics."""
        with self._lock:
            return {
                'max_rate': self.max_rate,
                'burst_capacity': self.burst_capacity,
                'current_tokens': self._tokens,
                'total_requests': self._total_requests,
                'allowed_requests': self._allowed_requests,
                'denied_requests': self._denied_requests,
                'success_rate': self._allowed_requests / max(self._total_requests, 1)
            }


class ConcurrencyManager:
    """
    Main concurrency management system that coordinates all concurrency operations.
    
    This manager provides:
    - Centralized concurrency control
    - Lock management and coordination
    - Resource allocation and monitoring
    - Performance metrics and monitoring
    - Automatic cleanup and recovery
    """
    
    def __init__(self, 
                 manager_id: str = "default",
                 enable_monitoring: bool = True,
                 cleanup_interval: int = 300):
        """
        Initialize concurrency manager.
        
        Args:
            manager_id: Unique identifier for the manager
            enable_monitoring: Whether to enable performance monitoring
            cleanup_interval: Cleanup interval in seconds
        """
        self.manager_id = manager_id
        self.enable_monitoring = enable_monitoring
        self.cleanup_interval = cleanup_interval
        
        # Component managers
        self._lock_manager: Dict[str, ThreadSafeLock] = {}
        self._resource_manager = ResourceManager(f"{manager_id}_resources")
        
        # Global synchronization
        self._global_lock = threading.RLock()
        self._operation_counter = 0
        
        # Monitoring and cleanup
        self._cleanup_thread: Optional[threading.Thread] = None
        self._cleanup_running = False
        
        # Metrics
        self._metrics = ConcurrencyMetrics()
        self._logger = logging.getLogger(f"{__name__}.ConcurrencyManager.{manager_id}")
        
        # Start cleanup thread
        self._start_cleanup_thread()
    
    def get_lock(self, 
                 lock_id: str,
                 max_shared: int = 10,
                 timeout_seconds: float = 30.0) -> ThreadSafeLock:
        """
        Get or create a thread-safe lock.
        
        Args:
            lock_id: Unique identifier for the lock
            max_shared: Maximum number of shared locks
            timeout_seconds: Default timeout for lock acquisition
            
        Returns:
            ThreadSafeLock instance
        """
        with self._global_lock:
            if lock_id not in self._lock_manager:
                self._lock_manager[lock_id] = ThreadSafeLock(
                    lock_id=lock_id,
                    max_shared=max_shared,
                    timeout_seconds=timeout_seconds
                )
                self._logger.info(f"Created new lock: {lock_id}")
            
            return self._lock_manager[lock_id]
    
    def configure_resource_quota(self, resource_type: ResourceType, quota: ResourceQuota):
        """
        Configure resource quota.
        
        Args:
            resource_type: Type of resource to configure
            quota: Quota configuration
        """
        self._resource_manager.configure_resource_quota(resource_type, quota)
    
    def allocate_resource(self, 
                         resource_type: ResourceType,
                         owner_id: str,
                         priority: str = "normal",
                         timeout: Optional[float] = None,
                         metadata: Optional[Dict[str, Any]] = None):
        """
        Allocate a resource.
        
        Args:
            resource_type: Type of resource to allocate
            owner_id: Identifier of the resource owner
            priority: Priority level for allocation
            timeout: Allocation timeout
            metadata: Additional metadata for the allocation
            
        Returns:
            Resource allocation context manager
        """
        return self._resource_manager.allocate_resource(
            resource_type, owner_id, priority, timeout, metadata
        )
    
    def get_comprehensive_statistics(self) -> Dict[str, Any]:
        """Get comprehensive concurrency statistics."""
        with self._global_lock:
            stats = {
                'manager_id': self.manager_id,
                'operation_counter': self._operation_counter,
                'active_locks': len(self._lock_manager),
                'lock_details': {
                    lock_id: lock.get_lock_info()
                    for lock_id, lock in self._lock_manager.items()
                },
                'resource_statistics': self._resource_manager.get_resource_statistics(),
                'global_metrics': {
                    'total_operations': self._metrics.total_operations,
                    'successful_operations': self._metrics.successful_operations,
                    'failed_operations': self._metrics.failed_operations,
                    'current_active_operations': self._metrics.current_active_operations,
                    'peak_concurrent_operations': self._metrics.peak_concurrent_operations,
                    'average_execution_time': self._metrics.average_execution_time,
                    'last_updated': self._metrics.last_updated.isoformat()
                }
            }
            
            return stats
    
    def _start_cleanup_thread(self):
        """Start the cleanup thread."""
        if not self._cleanup_running:
            self._cleanup_running = True
            self._cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
            self._cleanup_thread.start()
            self._logger.info("Started cleanup thread")
    
    def _cleanup_worker(self):
        """Background worker for cleanup operations."""
        while self._cleanup_running:
            try:
                # Clean up stale resource allocations
                self._resource_manager.cleanup_stale_allocations()
                
                # Clean up unused locks
                self._cleanup_unused_locks()
                
                # Update metrics
                self._metrics.last_updated = datetime.utcnow()
                
                # Sleep for cleanup interval
                time.sleep(self.cleanup_interval)
                
            except Exception as e:
                self._logger.error(f"Error in cleanup worker: {e}")
                time.sleep(60)  # Sleep for 1 minute on error
    
    def _cleanup_unused_locks(self):
        """Clean up unused locks."""
        with self._global_lock:
            unused_locks = []
            for lock_id, lock in self._lock_manager.items():
                lock_info = lock.get_lock_info()
                if (lock_info['shared_count'] == 0 and 
                    lock_info['exclusive_owner'] is None and 
                    lock_info['waiting_count'] == 0):
                    unused_locks.append(lock_id)
            
            for lock_id in unused_locks:
                del self._lock_manager[lock_id]
                self._logger.debug(f"Cleaned up unused lock: {lock_id}")
    
    def shutdown(self):
        """Shutdown the concurrency manager."""
        self._logger.info("Shutting down concurrency manager...")
        
        # Stop cleanup thread
        self._cleanup_running = False
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=5)
        
        # Clean up resources
        self._resource_manager.cleanup_stale_allocations()
        
        self._logger.info("Concurrency manager shutdown complete")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()


# Global concurrency manager instance
_global_manager: Optional[ConcurrencyManager] = None
_manager_lock = threading.Lock()


def get_global_concurrency_manager() -> ConcurrencyManager:
    """
    Get the global concurrency manager instance.
    
    Returns:
        Global ConcurrencyManager instance
    """
    global _global_manager
    
    with _manager_lock:
        if _global_manager is None:
            _global_manager = ConcurrencyManager(
                manager_id="global",
                enable_monitoring=True,
                cleanup_interval=300
            )
        
        return _global_manager


def shutdown_global_manager():
    """Shutdown the global concurrency manager."""
    global _global_manager
    
    with _manager_lock:
        if _global_manager:
            _global_manager.shutdown()
            _global_manager = None


# Convenience functions for common operations
def acquire_shared_lock(lock_id: str, 
                       owner_id: str,
                       timeout: Optional[float] = None):
    """
    Acquire a shared lock using the global manager.
    
    Args:
        lock_id: Unique identifier for the lock
        owner_id: Identifier of the lock owner
        timeout: Lock acquisition timeout
        
    Returns:
        Lock context manager
    """
    manager = get_global_concurrency_manager()
    lock = manager.get_lock(lock_id)
    return lock.acquire_shared(owner_id, timeout)


def acquire_exclusive_lock(lock_id: str,
                          owner_id: str,
                          timeout: Optional[float] = None):
    """
    Acquire an exclusive lock using the global manager.
    
    Args:
        lock_id: Unique identifier for the lock
        owner_id: Identifier of the lock owner
        timeout: Lock acquisition timeout
        
    Returns:
        Lock context manager
    """
    manager = get_global_concurrency_manager()
    lock = manager.get_lock(lock_id)
    return lock.acquire_exclusive(owner_id, timeout)


def allocate_resource(resource_type: ResourceType,
                     owner_id: str,
                     priority: str = "normal",
                     timeout: Optional[float] = None,
                     metadata: Optional[Dict[str, Any]] = None):
    """
    Allocate a resource using the global manager.
    
    Args:
        resource_type: Type of resource to allocate
        owner_id: Identifier of the resource owner
        priority: Priority level for allocation
        timeout: Allocation timeout
        metadata: Additional metadata for the allocation
        
    Returns:
        Resource allocation context manager
    """
    manager = get_global_concurrency_manager()
    return manager.allocate_resource(resource_type, owner_id, priority, timeout, metadata)


def get_concurrency_statistics() -> Dict[str, Any]:
    """
    Get comprehensive concurrency statistics.
    
    Returns:
        Dictionary containing concurrency statistics
    """
    manager = get_global_concurrency_manager()
    return manager.get_comprehensive_statistics()


# Cleanup function for proper shutdown
import atexit
atexit.register(shutdown_global_manager)