"""
Database connection management for YouTube Summarizer application.
Handles async database sessions, connection pooling, and health checks.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional, Dict, Any
from sqlalchemy.ext.asyncio import (
    AsyncSession, async_sessionmaker, create_async_engine, AsyncEngine
)
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.pool import QueuePool
from sqlalchemy import text
from .models import Base
from ..config import settings

logger = logging.getLogger(__name__)

# Global engine and session factory
_engine: Optional[AsyncEngine] = None
_session_factory: Optional[async_sessionmaker[AsyncSession]] = None


def get_database_url() -> str:
    """Get database URL from configuration."""
    return settings.database_url


def create_database_engine() -> AsyncEngine:
    """Create and configure async database engine."""
    database_url = get_database_url()
    
    # Engine configuration
    engine_config = {
        'url': database_url,
        'echo': settings.database_echo,
        'echo_pool': settings.database_echo_pool,
        'pool_size': settings.database_pool_size,
        'max_overflow': settings.database_max_overflow,
        'pool_timeout': settings.database_pool_timeout,
        'pool_recycle': settings.database_pool_recycle,
        'poolclass': QueuePool,
        'pool_pre_ping': True,  # Validate connections before use
        'pool_reset_on_return': 'commit',  # Reset connections on return
    }
    
    # Create async engine
    engine = create_async_engine(**engine_config)
    
    logger.info(f"Created database engine with URL: {database_url.split('@')[0]}@***")
    return engine


def get_database_engine() -> AsyncEngine:
    """Get or create the global database engine."""
    global _engine
    if _engine is None:
        _engine = create_database_engine()
    return _engine


def create_session_factory() -> async_sessionmaker[AsyncSession]:
    """Create async session factory."""
    engine = get_database_engine()
    
    # Session configuration
    session_config = {
        'bind': engine,
        'class_': AsyncSession,
        'expire_on_commit': False,
        'autocommit': False,
        'autoflush': True,
    }
    
    return async_sessionmaker(**session_config)


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Get or create the global session factory."""
    global _session_factory
    if _session_factory is None:
        _session_factory = create_session_factory()
    return _session_factory


@asynccontextmanager
async def get_database_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Get async database session with automatic cleanup.
    
    Usage:
        async with get_database_session() as session:
            result = await session.execute(query)
    """
    session_factory = get_session_factory()
    session = session_factory()
    
    try:
        yield session
        await session.commit()
    except Exception as e:
        await session.rollback()
        logger.error(f"Database session error: {e}")
        raise
    finally:
        await session.close()


async def get_database_session_dependency() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency for database sessions.
    
    Usage:
        @app.get("/")
        async def endpoint(session: AsyncSession = Depends(get_database_session_dependency)):
            # Use session here
    """
    async with get_database_session() as session:
        yield session


async def init_database() -> bool:
    """
    Initialize database connection and create tables if needed.
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        engine = get_database_engine()
        
        # Test connection
        async with engine.begin() as conn:
            # Check if tables exist by trying to query one
            try:
                await conn.execute(text("SELECT 1 FROM videos LIMIT 1"))
                logger.info("Database tables already exist")
            except SQLAlchemyError:
                # Tables don't exist, create them
                logger.info("Creating database tables...")
                await conn.run_sync(Base.metadata.create_all)
                logger.info("Database tables created successfully")
        
        logger.info("Database initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        return False


async def check_database_health() -> Dict[str, Any]:
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
        async with engine.begin() as conn:
            result = await conn.execute(text("SELECT 1 AS health_check"))
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


async def close_database_connections() -> None:
    """Close all database connections and cleanup resources."""
    global _engine, _session_factory
    
    try:
        if _engine:
            await _engine.dispose()
            logger.info("Database engine disposed")
        
        _engine = None
        _session_factory = None
        
        logger.info("Database connections closed successfully")
        
    except Exception as e:
        logger.error(f"Error closing database connections: {e}")


async def reset_database_connections() -> bool:
    """Reset database connections (useful for configuration changes)."""
    try:
        await close_database_connections()
        
        # Recreate connections
        get_database_engine()
        get_session_factory()
        
        # Test new connections
        health_status = await check_database_health()
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
    
    async def initialize(self) -> bool:
        """Initialize the database manager."""
        if self._initialized:
            return True
        
        try:
            success = await init_database()
            if success:
                self._initialized = True
                logger.info("DatabaseManager initialized successfully")
            return success
        except Exception as e:
            logger.error(f"Failed to initialize DatabaseManager: {e}")
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform database health check."""
        return await check_database_health()
    
    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get database session through manager."""
        if not self._initialized:
            raise RuntimeError("DatabaseManager not initialized. Call initialize() first.")
        
        async with get_database_session() as session:
            yield session
    
    async def close(self) -> None:
        """Close database manager and cleanup."""
        await close_database_connections()
        self._initialized = False


# Global database manager instance
db_manager = DatabaseManager()


# Utility functions for common database operations
async def execute_query(query: str, parameters: Optional[Dict[str, Any]] = None) -> Any:
    """
    Execute a raw SQL query with optional parameters.
    
    Args:
        query: SQL query string
        parameters: Optional query parameters
    
    Returns:
        Query result
    """
    async with get_database_session() as session:
        if parameters:
            result = await session.execute(text(query), parameters)
        else:
            result = await session.execute(text(query))
        return result


async def get_table_count(table_name: str) -> int:
    """
    Get count of records in a table.
    
    Args:
        table_name: Name of the table
    
    Returns:
        Number of records in the table
    """
    try:
        result = await execute_query(f"SELECT COUNT(*) FROM {table_name}")
        count = result.scalar()
        return count if count is not None else 0
    except Exception as e:
        logger.error(f"Failed to get count for table {table_name}: {e}")
        return 0


async def get_database_info() -> Dict[str, Any]:
    """
    Get comprehensive database information.
    
    Returns:
        Dictionary with database statistics and information
    """
    info = {
        'health': await check_database_health(),
        'tables': {},
        'total_records': 0,
    }
    
    # Get record counts for all tables
    table_names = ['videos', 'transcripts', 'summaries', 'keywords', 'timestamped_segments', 'processing_metadata']
    
    for table_name in table_names:
        try:
            count = await get_table_count(table_name)
            info['tables'][table_name] = count
            info['total_records'] += count
        except Exception as e:
            logger.error(f"Failed to get info for table {table_name}: {e}")
            info['tables'][table_name] = 'error'
    
    return info