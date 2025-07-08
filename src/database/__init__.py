"""
Database package for YouTube Summarizer application.
Contains database models, connection management, maintenance utilities, and monitoring.
"""

from .models import (
    Video,
    Transcript,
    Summary,
    Keyword,
    TimestampedSegment,
    ProcessingMetadata,
    Base
)
from .connection import (
    get_database_session,
    get_database_engine,
    init_database,
    close_database_connections,
    check_database_health,
    get_database_session_dependency
)
from .maintenance import (
    DatabaseMaintenance,
    cleanup_old_records,
    get_database_health_detailed,
    run_maintenance_tasks
)
from .monitor import db_monitor, MonitoredOperation
from .exceptions import (
    DatabaseError,
    DatabaseConnectionError,
    DatabaseQueryError,
    DatabaseConstraintError,
    DatabaseTimeoutError,
    DatabaseUnavailableError,
    classify_database_error
)

__all__ = [
    # Models
    'Video',
    'Transcript',
    'Summary',
    'Keyword',
    'TimestampedSegment',
    'ProcessingMetadata',
    'Base',
    
    # Connection management
    'get_database_session',
    'get_database_session_dependency',
    'get_database_engine',
    'init_database',
    'close_database_connections',
    'check_database_health',
    
    # Maintenance
    'DatabaseMaintenance',
    'cleanup_old_records',
    'get_database_health_detailed',
    'run_maintenance_tasks',
    
    # Monitoring
    'db_monitor',
    'MonitoredOperation',
    
    # Exceptions
    'DatabaseError',
    'DatabaseConnectionError',
    'DatabaseQueryError',
    'DatabaseConstraintError',
    'DatabaseTimeoutError',
    'DatabaseUnavailableError',
    'classify_database_error',
]