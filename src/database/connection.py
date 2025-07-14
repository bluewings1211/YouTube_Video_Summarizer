"""
Database connection management for YouTube Summarizer application.
Handles synchronous database sessions, connection pooling, and health checks.
"""

import logging
from contextlib import contextmanager
from typing import Generator, Optional, Dict, Any
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy import create_engine, Engine, text
from sqlalchemy.exc import SQLAlchemyError
from .models import Base
from ..config import settings

logger = logging.getLogger(__name__)

# Global engine and session factory
_engine: Optional[Engine] = None
_session_factory: Optional[sessionmaker[Session]] = None


def get_database_url() -> str:
    """Get database URL from configuration."""
    return settings.database_url


def create_database_engine() -> Engine:
    """Create and configure synchronous database engine."""
    database_url = get_database_url()
    
    # Convert asyncpg URL to psycopg2 URL for synchronous operations
    if database_url.startswith('postgresql+asyncpg://'):
        database_url = database_url.replace('postgresql+asyncpg://', 'postgresql://')
    
    # Engine configuration for synchronous engine
    engine_config = {
        'echo': settings.database_echo,
        'echo_pool': settings.database_echo_pool,
        'pool_size': settings.database_pool_size,
        'max_overflow': settings.database_max_overflow,
        'pool_timeout': settings.database_pool_timeout,
        'pool_recycle': settings.database_pool_recycle,
        'pool_pre_ping': True,
        'pool_reset_on_return': 'commit',
        'connect_args': {
            'application_name': 'youtube_summarizer'
        }
    }
    
    # Create synchronous engine
    engine = create_engine(database_url, **engine_config)
    
    logger.info(f"Created database engine with URL: {database_url.split('@')[0]}@***")
    return engine


def get_database_engine() -> Engine:
    """Get or create the global database engine."""
    global _engine
    if _engine is None:
        _engine = create_database_engine()
    return _engine


def create_session_factory() -> sessionmaker[Session]:
    """Create synchronous session factory."""
    engine = get_database_engine()
    
    # Session configuration
    session_config = {
        'bind': engine,
        'expire_on_commit': False,
        'autocommit': False,
        'autoflush': True,
    }
    
    return sessionmaker(**session_config)


def get_session_factory() -> sessionmaker[Session]:
    """Get or create the global session factory."""
    global _session_factory
    if _session_factory is None:
        _session_factory = create_session_factory()
    return _session_factory


@contextmanager
def get_database_session() -> Generator[Session, None, None]:
    """
    Get synchronous database session with automatic cleanup.
    
    Usage:
        with get_database_session() as session:
            result = session.execute(query)
    """
    session_factory = get_session_factory()
    session = session_factory()
    
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Database session error: {e}")
        raise
    finally:
        session.close()


def get_database_session_dependency() -> Generator[Session, None, None]:
    """
    FastAPI dependency for database sessions.
    
    Usage:
        @app.get("/")
        def endpoint(session: Session = Depends(get_database_session_dependency)):
            # Use session here
    """
    with get_database_session() as session:
        yield session


def init_database() -> bool:
    """
    Initialize database connection and create tables if needed.
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        engine = get_database_engine()
        
        # Test connection
        with engine.begin() as conn:
            # Check if tables exist by trying to query one
            try:
                conn.execute(text("SELECT 1 FROM videos LIMIT 1"))
                logger.info("Database tables already exist")
            except SQLAlchemyError:
                # Tables don't exist, create them
                logger.info("Creating database tables...")
                Base.metadata.create_all(bind=engine)
                logger.info("Database tables created successfully")
        
        logger.info("Database initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        return False


def check_database_health() -> Dict[str, Any]:
    """
    Check database health and connection status.
    
    Returns:
        Dict containing health status information
    """
    health_info = {
        'status': 'unhealthy',
        'timestamp': None,
        'response_time_ms': None,
        'pool_status': None,
        'error': None
    }
    
    try:
        import time
        start_time = time.time()
        
        engine = get_database_engine()
        
        # Test database connection
        with engine.begin() as conn:
            result = conn.execute(text("SELECT 1 AS health_check"))
            row = result.fetchone()
            
            if row and row[0] == 1:
                end_time = time.time()
                response_time = (end_time - start_time) * 1000  # Convert to milliseconds
                
                # Get pool status
                pool_status = {
                    'size': engine.pool.size(),
                    'checked_in': engine.pool.checkedin(),
                    'checked_out': engine.pool.checkedout(),
                }
                
                health_info.update({
                    'status': 'healthy',
                    'timestamp': time.time(),
                    'response_time_ms': round(response_time, 2),
                    'pool_status': pool_status,
                })
                
                logger.debug(f"Database health check passed in {response_time:.2f}ms")
            else:
                health_info['error'] = 'Health check query returned unexpected result'
                
    except Exception as e:
        error_msg = f"Database health check failed: {e}"
        health_info['error'] = error_msg
        logger.error(error_msg)
    
    return health_info


def close_database_connections() -> None:
    """Close all database connections and cleanup resources."""
    global _engine, _session_factory
    
    try:
        if _engine:
            _engine.dispose()
            logger.info("Database engine disposed")
        
        _engine = None
        _session_factory = None
        
        logger.info("Database connections closed successfully")
        
    except Exception as e:
        logger.error(f"Error closing database connections: {e}")


def reset_database_connections() -> bool:
    """Reset database connections (useful for configuration changes)."""
    try:
        close_database_connections()
        
        # Recreate connections
        get_database_engine()
        get_session_factory()
        
        # Test new connections
        health_status = check_database_health()
        if health_status['status'] == 'healthy':
            logger.info("Database connections reset successfully")
            return True
        else:
            logger.error("Database health check failed after reset")
            return False
            
    except Exception as e:
        logger.error(f"Failed to reset database connections: {e}")
        return False


class DatabaseManager:
    """
    Database manager class for handling database operations.
    Provides a higher-level interface for database management.
    """
    
    def __init__(self):
        self._initialized = False
    
    def initialize(self) -> bool:
        """Initialize the database manager."""
        if self._initialized:
            return True
        
        try:
            success = init_database()
            if success:
                self._initialized = True
                logger.info("DatabaseManager initialized successfully")
            return success
        except Exception as e:
            logger.error(f"Failed to initialize DatabaseManager: {e}")
            return False
    
    def health_check(self) -> Dict[str, Any]:
        """Perform database health check."""
        return check_database_health()
    
    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """Get database session through manager."""
        if not self._initialized:
            raise RuntimeError("DatabaseManager not initialized. Call initialize() first.")
        
        with get_database_session() as session:
            yield session
    
    def close(self) -> None:
        """Close database manager and cleanup."""
        close_database_connections()
        self._initialized = False


# Global database manager instance
db_manager = DatabaseManager()

# Alias for backward compatibility
get_db_session = get_database_session


# Utility functions for common database operations
def execute_query(query: str, parameters: Optional[Dict[str, Any]] = None) -> Any:
    """
    Execute a raw SQL query with optional parameters.
    
    Args:
        query: SQL query string
        parameters: Optional query parameters
    
    Returns:
        Query result
    """
    with get_database_session() as session:
        if parameters:
            result = session.execute(text(query), parameters)
        else:
            result = session.execute(text(query))
        return result


def get_table_count(table_name: str) -> int:
    """
    Get count of records in a table.
    
    Args:
        table_name: Name of the table
    
    Returns:
        Number of records in the table
    """
    try:
        result = execute_query(f"SELECT COUNT(*) FROM {table_name}")
        count = result.scalar()
        return count if count is not None else 0
    except Exception as e:
        logger.error(f"Failed to get count for table {table_name}: {e}")
        return 0


def get_database_info() -> Dict[str, Any]:
    """
    Get comprehensive database information.
    
    Returns:
        Dictionary with database statistics and information
    """
    info = {
        'health': check_database_health(),
        'tables': {},
        'total_records': 0,
    }
    
    # Get record counts for all tables
    table_names = ['videos', 'transcripts', 'summaries', 'keywords', 'timestamped_segments', 'processing_metadata']
    
    for table_name in table_names:
        try:
            count = get_table_count(table_name)
            info['tables'][table_name] = count
            info['total_records'] += count
        except Exception as e:
            logger.error(f"Failed to get info for table {table_name}: {e}")
            info['tables'][table_name] = 'error'
    
    return info