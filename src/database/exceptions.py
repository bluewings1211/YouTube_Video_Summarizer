"""
Database-specific exception classes for error handling.

This module contains custom exception classes for database operations,
providing detailed error context and recovery suggestions.
"""

import logging
from typing import Optional, Dict, Any
from enum import Enum


class DatabaseErrorSeverity(Enum):
    """Severity levels for database errors."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DatabaseErrorCategory(Enum):
    """Categories of database errors."""
    CONNECTION = "connection"
    QUERY = "query"
    CONSTRAINT = "constraint"
    TIMEOUT = "timeout"
    PERMISSION = "permission"
    DATA_INTEGRITY = "data_integrity"
    UNKNOWN = "unknown"


class DatabaseError(Exception):
    """
    Base class for database-related errors.
    
    Provides structured error information with context and recovery suggestions.
    """
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        severity: DatabaseErrorSeverity = DatabaseErrorSeverity.MEDIUM,
        category: DatabaseErrorCategory = DatabaseErrorCategory.UNKNOWN,
        original_error: Optional[Exception] = None,
        context: Optional[Dict[str, Any]] = None,
        recovery_suggestion: Optional[str] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.severity = severity
        self.category = category
        self.original_error = original_error
        self.context = context or {}
        self.recovery_suggestion = recovery_suggestion
        
        # Log the error
        logger = logging.getLogger(__name__)
        log_level = self._get_log_level()
        logger.log(log_level, f"Database error: {message}", extra={
            'error_code': error_code,
            'severity': severity.value,
            'category': category.value,
            'context': context
        })
    
    def _get_log_level(self) -> int:
        """Get appropriate log level for error severity."""
        if self.severity == DatabaseErrorSeverity.CRITICAL:
            return logging.CRITICAL
        elif self.severity == DatabaseErrorSeverity.HIGH:
            return logging.ERROR
        elif self.severity == DatabaseErrorSeverity.MEDIUM:
            return logging.WARNING
        else:
            return logging.INFO
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for serialization."""
        return {
            'error_type': self.__class__.__name__,
            'message': self.message,
            'error_code': self.error_code,
            'severity': self.severity.value,
            'category': self.category.value,
            'context': self.context,
            'recovery_suggestion': self.recovery_suggestion,
            'original_error': str(self.original_error) if self.original_error else None
        }


class DatabaseConnectionError(DatabaseError):
    """Error connecting to the database."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            error_code="DB_CONNECTION_FAILED",
            severity=DatabaseErrorSeverity.HIGH,
            category=DatabaseErrorCategory.CONNECTION,
            recovery_suggestion="Check database server status and connection configuration",
            **kwargs
        )


class DatabaseQueryError(DatabaseError):
    """Error executing a database query."""
    
    def __init__(self, message: str, query: Optional[str] = None, **kwargs):
        context = kwargs.get('context', {})
        if query:
            context['query'] = query
        kwargs['context'] = context
        
        super().__init__(
            message,
            error_code="DB_QUERY_FAILED",
            severity=DatabaseErrorSeverity.MEDIUM,
            category=DatabaseErrorCategory.QUERY,
            recovery_suggestion="Check query syntax and database schema",
            **kwargs
        )


class DatabaseConstraintError(DatabaseError):
    """Database constraint violation error."""
    
    def __init__(self, message: str, constraint_name: Optional[str] = None, **kwargs):
        context = kwargs.get('context', {})
        if constraint_name:
            context['constraint'] = constraint_name
        kwargs['context'] = context
        
        super().__init__(
            message,
            error_code="DB_CONSTRAINT_VIOLATION",
            severity=DatabaseErrorSeverity.MEDIUM,
            category=DatabaseErrorCategory.CONSTRAINT,
            recovery_suggestion="Check data integrity and constraint requirements",
            **kwargs
        )


class DatabaseTimeoutError(DatabaseError):
    """Database operation timeout error."""
    
    def __init__(self, message: str, timeout_seconds: Optional[float] = None, **kwargs):
        context = kwargs.get('context', {})
        if timeout_seconds:
            context['timeout_seconds'] = timeout_seconds
        kwargs['context'] = context
        
        super().__init__(
            message,
            error_code="DB_TIMEOUT",
            severity=DatabaseErrorSeverity.HIGH,
            category=DatabaseErrorCategory.TIMEOUT,
            recovery_suggestion="Increase timeout or optimize query performance",
            **kwargs
        )


class DatabasePermissionError(DatabaseError):
    """Database permission/authorization error."""
    
    def __init__(self, message: str, operation: Optional[str] = None, **kwargs):
        context = kwargs.get('context', {})
        if operation:
            context['operation'] = operation
        kwargs['context'] = context
        
        super().__init__(
            message,
            error_code="DB_PERMISSION_DENIED",
            severity=DatabaseErrorSeverity.HIGH,
            category=DatabaseErrorCategory.PERMISSION,
            recovery_suggestion="Check database user permissions and access rights",
            **kwargs
        )


class DatabaseIntegrityError(DatabaseError):
    """Data integrity constraint violation."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            error_code="DB_INTEGRITY_ERROR",
            severity=DatabaseErrorSeverity.MEDIUM,
            category=DatabaseErrorCategory.DATA_INTEGRITY,
            recovery_suggestion="Verify data consistency and foreign key relationships",
            **kwargs
        )


class DatabaseUnavailableError(DatabaseError):
    """Database service unavailable error."""
    
    def __init__(self, message: str = "Database service is unavailable", **kwargs):
        super().__init__(
            message,
            error_code="DB_UNAVAILABLE",
            severity=DatabaseErrorSeverity.CRITICAL,
            category=DatabaseErrorCategory.CONNECTION,
            recovery_suggestion="Check database server status and retry operation",
            **kwargs
        )


# Error classification helpers
def classify_database_error(error: Exception) -> DatabaseError:
    """
    Classify a generic database error into a specific DatabaseError subclass.
    
    Args:
        error: The original exception
        
    Returns:
        Classified DatabaseError instance
    """
    error_message = str(error).lower()
    
    # Connection errors
    if any(term in error_message for term in [
        'connection', 'connect', 'timeout', 'unreachable', 'refused'
    ]):
        return DatabaseConnectionError(str(error), original_error=error)
    
    # Constraint errors
    if any(term in error_message for term in [
        'constraint', 'unique', 'foreign key', 'not null', 'check'
    ]):
        return DatabaseConstraintError(str(error), original_error=error)
    
    # Permission errors
    if any(term in error_message for term in [
        'permission', 'access', 'denied', 'unauthorized', 'forbidden'
    ]):
        return DatabasePermissionError(str(error), original_error=error)
    
    # Timeout errors
    if any(term in error_message for term in [
        'timeout', 'time out', 'timed out', 'deadline'
    ]):
        return DatabaseTimeoutError(str(error), original_error=error)
    
    # Integrity errors
    if any(term in error_message for term in [
        'integrity', 'duplicate', 'violates', 'invalid'
    ]):
        return DatabaseIntegrityError(str(error), original_error=error)
    
    # Default to generic query error
    return DatabaseQueryError(str(error), original_error=error)


def is_recoverable_error(error: DatabaseError) -> bool:
    """
    Determine if a database error is potentially recoverable.
    
    Args:
        error: DatabaseError instance
        
    Returns:
        True if error might be recoverable with retry
    """
    # Timeout and connection errors are often recoverable
    if error.category in [DatabaseErrorCategory.TIMEOUT, DatabaseErrorCategory.CONNECTION]:
        return True
    
    # Constraint and integrity errors are typically not recoverable without data changes
    if error.category in [DatabaseErrorCategory.CONSTRAINT, DatabaseErrorCategory.DATA_INTEGRITY]:
        return False
    
    # Permission errors are not recoverable without authorization changes
    if error.category == DatabaseErrorCategory.PERMISSION:
        return False
    
    # Query errors might be recoverable depending on the issue
    if error.category == DatabaseErrorCategory.QUERY:
        return error.severity != DatabaseErrorSeverity.CRITICAL
    
    # Default to not recoverable for unknown errors
    return False


def should_retry_operation(error: DatabaseError, retry_count: int, max_retries: int) -> bool:
    """
    Determine if a database operation should be retried.
    
    Args:
        error: DatabaseError that occurred
        retry_count: Current retry attempt number
        max_retries: Maximum number of retries allowed
        
    Returns:
        True if operation should be retried
    """
    if retry_count >= max_retries:
        return False
    
    if not is_recoverable_error(error):
        return False
    
    # Don't retry critical errors
    if error.severity == DatabaseErrorSeverity.CRITICAL:
        return False
    
    return True