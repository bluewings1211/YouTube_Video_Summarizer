"""
Database package for YouTube Summarizer application.
Contains database models, connection management, and migration utilities.
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
    check_database_health
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
    'get_database_engine',
    'init_database',
    'close_database_connections',
    'check_database_health',
]