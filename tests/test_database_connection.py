"""
Comprehensive tests for database connection management.

This test suite covers:
- Database engine creation and configuration
- Session factory creation
- Database initialization
- Health checks
- Connection pooling
- Error handling
- Database manager functionality
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.pool import QueuePool

from src.database.connection import (
    get_database_url, create_database_engine, get_database_engine,
    create_session_factory, get_session_factory, get_database_session,
    get_database_session_dependency, init_database, check_database_health,
    close_database_connections, reset_database_connections,
    DatabaseManager, db_manager, execute_query, get_table_count,
    get_database_info
)
from src.database.models import Base


class TestDatabaseURL:
    """Test database URL configuration."""
    
    @patch('src.database.connection.settings')
    def test_get_database_url(self, mock_settings):
        """Test getting database URL from settings."""
        mock_settings.database_url = "postgresql+asyncpg://user:pass@localhost/test"
        
        url = get_database_url()
        
        assert url == "postgresql+asyncpg://user:pass@localhost/test"


class TestDatabaseEngine:
    """Test database engine creation and management."""
    
    @patch('src.database.connection.settings')
    def test_create_database_engine(self, mock_settings):
        """Test creating database engine with configuration."""
        mock_settings.database_url = "postgresql+asyncpg://user:pass@localhost/test"
        mock_settings.database_echo = False
        mock_settings.database_echo_pool = False
        mock_settings.database_pool_size = 5
        mock_settings.database_max_overflow = 10
        mock_settings.database_pool_timeout = 30
        mock_settings.database_pool_recycle = 3600
        
        with patch('src.database.connection.create_async_engine') as mock_create_engine:
            mock_engine = Mock(spec=AsyncEngine)
            mock_create_engine.return_value = mock_engine
            
            engine = create_database_engine()
            
            # Verify create_async_engine was called with correct parameters
            mock_create_engine.assert_called_once()
            call_args = mock_create_engine.call_args[1]
            
            assert call_args['url'] == "postgresql+asyncpg://user:pass@localhost/test"
            assert call_args['echo'] is False
            assert call_args['echo_pool'] is False
            assert call_args['pool_size'] == 5
            assert call_args['max_overflow'] == 10
            assert call_args['pool_timeout'] == 30
            assert call_args['pool_recycle'] == 3600
            assert call_args['poolclass'] == QueuePool
            assert call_args['pool_pre_ping'] is True
            assert call_args['pool_reset_on_return'] == 'commit'
            
            assert engine == mock_engine
    
    @patch('src.database.connection._engine', None)
    def test_get_database_engine_creates_new(self):
        """Test getting database engine creates new engine when none exists."""
        with patch('src.database.connection.create_database_engine') as mock_create:
            mock_engine = Mock(spec=AsyncEngine)
            mock_create.return_value = mock_engine
            
            engine = get_database_engine()
            
            assert engine == mock_engine
            mock_create.assert_called_once()
    
    @patch('src.database.connection._engine')
    def test_get_database_engine_returns_existing(self, mock_existing_engine):
        """Test getting database engine returns existing engine."""
        engine = get_database_engine()
        
        assert engine == mock_existing_engine


class TestSessionFactory:
    """Test session factory creation and management."""
    
    def test_create_session_factory(self):
        """Test creating session factory."""
        mock_engine = Mock(spec=AsyncEngine)
        
        with patch('src.database.connection.get_database_engine', return_value=mock_engine):
            with patch('src.database.connection.async_sessionmaker') as mock_sessionmaker:
                mock_factory = Mock()
                mock_sessionmaker.return_value = mock_factory
                
                factory = create_session_factory()
                
                # Verify async_sessionmaker was called with correct parameters
                mock_sessionmaker.assert_called_once()
                call_args = mock_sessionmaker.call_args[1]
                
                assert call_args['bind'] == mock_engine
                assert call_args['class_'] == AsyncSession
                assert call_args['expire_on_commit'] is False
                assert call_args['autocommit'] is False
                assert call_args['autoflush'] is True
                
                assert factory == mock_factory
    
    @patch('src.database.connection._session_factory', None)
    def test_get_session_factory_creates_new(self):
        """Test getting session factory creates new when none exists."""
        with patch('src.database.connection.create_session_factory') as mock_create:
            mock_factory = Mock()
            mock_create.return_value = mock_factory
            
            factory = get_session_factory()
            
            assert factory == mock_factory
            mock_create.assert_called_once()
    
    @patch('src.database.connection._session_factory')
    def test_get_session_factory_returns_existing(self, mock_existing_factory):
        """Test getting session factory returns existing factory."""
        factory = get_session_factory()
        
        assert factory == mock_existing_factory


class TestDatabaseSession:
    """Test database session management."""
    
    @pytest.mark.asyncio
    async def test_get_database_session_success(self):
        """Test successful database session creation and cleanup."""
        mock_session = AsyncMock(spec=AsyncSession)
        mock_factory = Mock()
        mock_factory.return_value = mock_session
        
        with patch('src.database.connection.get_session_factory', return_value=mock_factory):
            async with get_database_session() as session:
                assert session == mock_session
        
        # Verify session lifecycle
        mock_session.commit.assert_called_once()
        mock_session.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_database_session_with_error(self):
        """Test database session with error handling."""
        mock_session = AsyncMock(spec=AsyncSession)
        mock_factory = Mock()
        mock_factory.return_value = mock_session
        
        with patch('src.database.connection.get_session_factory', return_value=mock_factory):
            with pytest.raises(ValueError, match="Test error"):
                async with get_database_session() as session:
                    assert session == mock_session
                    raise ValueError("Test error")
        
        # Verify session cleanup on error
        mock_session.rollback.assert_called_once()
        mock_session.close.assert_called_once()
        mock_session.commit.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_get_database_session_dependency(self):
        """Test database session dependency for FastAPI."""
        mock_session = AsyncMock(spec=AsyncSession)
        
        with patch('src.database.connection.get_database_session') as mock_get_session:
            # Mock the async context manager
            mock_context = AsyncMock()
            mock_context.__aenter__.return_value = mock_session
            mock_context.__aexit__.return_value = None
            mock_get_session.return_value = mock_context
            
            # Test the dependency
            async_gen = get_database_session_dependency()
            session = await async_gen.__anext__()
            
            assert session == mock_session


class TestDatabaseInitialization:
    """Test database initialization functionality."""
    
    @pytest.mark.asyncio
    async def test_init_database_with_existing_tables(self):
        """Test database initialization when tables already exist."""
        mock_engine = AsyncMock(spec=AsyncEngine)
        mock_connection = AsyncMock()
        mock_engine.begin.return_value.__aenter__.return_value = mock_connection
        
        # Mock successful query (tables exist)
        mock_connection.execute.return_value = None
        
        with patch('src.database.connection.get_database_engine', return_value=mock_engine):
            result = await init_database()
        
        assert result is True
        mock_connection.execute.assert_called_once()
        mock_connection.run_sync.assert_not_called()  # Should not create tables
    
    @pytest.mark.asyncio
    async def test_init_database_create_tables(self):
        """Test database initialization when tables need to be created."""
        mock_engine = AsyncMock(spec=AsyncEngine)
        mock_connection = AsyncMock()
        mock_engine.begin.return_value.__aenter__.return_value = mock_connection
        
        # Mock failed query (tables don't exist), then successful creation
        mock_connection.execute.side_effect = [SQLAlchemyError("Table doesn't exist"), None]
        
        with patch('src.database.connection.get_database_engine', return_value=mock_engine):
            result = await init_database()
        
        assert result is True
        assert mock_connection.execute.call_count == 1
        mock_connection.run_sync.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_init_database_failure(self):
        """Test database initialization failure."""
        mock_engine = AsyncMock(spec=AsyncEngine)
        mock_engine.begin.side_effect = Exception("Connection failed")
        
        with patch('src.database.connection.get_database_engine', return_value=mock_engine):
            result = await init_database()
        
        assert result is False


class TestDatabaseHealthCheck:
    """Test database health check functionality."""
    
    @pytest.mark.asyncio
    async def test_check_database_health_success(self):
        """Test successful database health check."""
        mock_engine = AsyncMock(spec=AsyncEngine)
        mock_connection = AsyncMock()
        mock_engine.begin.return_value.__aenter__.return_value = mock_connection
        
        # Mock successful health check query
        mock_result = Mock()
        mock_row = Mock()
        mock_row.__getitem__.return_value = 1
        mock_result.fetchone.return_value = mock_row
        mock_connection.execute.return_value = mock_result
        
        # Mock pool status
        mock_pool = Mock()
        mock_pool.size.return_value = 5
        mock_pool.checkedin.return_value = 3
        mock_pool.checkedout.return_value = 2
        mock_engine.pool = mock_pool
        
        with patch('src.database.connection.get_database_engine', return_value=mock_engine):
            with patch('time.time', side_effect=[1000.0, 1000.1]):  # 100ms response time
                health_info = await check_database_health()
        
        assert health_info['status'] == 'healthy'
        assert health_info['response_time_ms'] == 100.0
        assert health_info['pool_status']['size'] == 5
        assert health_info['pool_status']['checked_in'] == 3
        assert health_info['pool_status']['checked_out'] == 2
        assert health_info['error'] is None
    
    @pytest.mark.asyncio
    async def test_check_database_health_failure(self):
        """Test database health check failure."""
        mock_engine = AsyncMock(spec=AsyncEngine)
        mock_engine.begin.side_effect = SQLAlchemyError("Connection failed")
        
        with patch('src.database.connection.get_database_engine', return_value=mock_engine):
            health_info = await check_database_health()
        
        assert health_info['status'] == 'unhealthy'
        assert 'Connection failed' in health_info['error']
        assert health_info['response_time_ms'] is None
        assert health_info['pool_status'] is None
    
    @pytest.mark.asyncio
    async def test_check_database_health_unexpected_result(self):
        """Test database health check with unexpected result."""
        mock_engine = AsyncMock(spec=AsyncEngine)
        mock_connection = AsyncMock()
        mock_engine.begin.return_value.__aenter__.return_value = mock_connection
        
        # Mock unexpected result
        mock_result = Mock()
        mock_row = Mock()
        mock_row.__getitem__.return_value = 0  # Unexpected value
        mock_result.fetchone.return_value = mock_row
        mock_connection.execute.return_value = mock_result
        
        with patch('src.database.connection.get_database_engine', return_value=mock_engine):
            health_info = await check_database_health()
        
        assert health_info['status'] == 'unhealthy'
        assert 'unexpected result' in health_info['error']


class TestConnectionManagement:
    """Test connection management functionality."""
    
    @pytest.mark.asyncio
    async def test_close_database_connections(self):
        """Test closing database connections."""
        mock_engine = AsyncMock(spec=AsyncEngine)
        
        with patch('src.database.connection._engine', mock_engine):
            with patch('src.database.connection._session_factory', Mock()):
                await close_database_connections()
        
        mock_engine.dispose.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_close_database_connections_with_error(self):
        """Test closing database connections with error."""
        mock_engine = AsyncMock(spec=AsyncEngine)
        mock_engine.dispose.side_effect = Exception("Disposal failed")
        
        with patch('src.database.connection._engine', mock_engine):
            # Should not raise exception
            await close_database_connections()
        
        mock_engine.dispose.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_reset_database_connections_success(self):
        """Test successful database connection reset."""
        with patch('src.database.connection.close_database_connections') as mock_close:
            with patch('src.database.connection.get_database_engine') as mock_get_engine:
                with patch('src.database.connection.get_session_factory') as mock_get_factory:
                    with patch('src.database.connection.check_database_health') as mock_health:
                        mock_health.return_value = {'status': 'healthy'}
                        
                        result = await reset_database_connections()
        
        assert result is True
        mock_close.assert_called_once()
        mock_get_engine.assert_called_once()
        mock_get_factory.assert_called_once()
        mock_health.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_reset_database_connections_health_check_failure(self):
        """Test database connection reset with health check failure."""
        with patch('src.database.connection.close_database_connections'):
            with patch('src.database.connection.get_database_engine'):
                with patch('src.database.connection.get_session_factory'):
                    with patch('src.database.connection.check_database_health') as mock_health:
                        mock_health.return_value = {'status': 'unhealthy'}
                        
                        result = await reset_database_connections()
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_reset_database_connections_exception(self):
        """Test database connection reset with exception."""
        with patch('src.database.connection.close_database_connections', side_effect=Exception("Reset failed")):
            result = await reset_database_connections()
        
        assert result is False


class TestDatabaseManager:
    """Test DatabaseManager class functionality."""
    
    @pytest.mark.asyncio
    async def test_database_manager_initialization(self):
        """Test DatabaseManager initialization."""
        manager = DatabaseManager()
        assert manager._initialized is False
        
        with patch('src.database.connection.init_database', return_value=True):
            result = await manager.initialize()
        
        assert result is True
        assert manager._initialized is True
    
    @pytest.mark.asyncio
    async def test_database_manager_initialization_failure(self):
        """Test DatabaseManager initialization failure."""
        manager = DatabaseManager()
        
        with patch('src.database.connection.init_database', return_value=False):
            result = await manager.initialize()
        
        assert result is False
        assert manager._initialized is False
    
    @pytest.mark.asyncio
    async def test_database_manager_health_check(self):
        """Test DatabaseManager health check."""
        manager = DatabaseManager()
        expected_health = {'status': 'healthy'}
        
        with patch('src.database.connection.check_database_health', return_value=expected_health):
            health = await manager.health_check()
        
        assert health == expected_health
    
    @pytest.mark.asyncio
    async def test_database_manager_get_session_initialized(self):
        """Test getting session from initialized manager."""
        manager = DatabaseManager()
        manager._initialized = True
        
        mock_session = AsyncMock(spec=AsyncSession)
        
        with patch('src.database.connection.get_database_session') as mock_get_session:
            mock_context = AsyncMock()
            mock_context.__aenter__.return_value = mock_session
            mock_context.__aexit__.return_value = None
            mock_get_session.return_value = mock_context
            
            async with manager.get_session() as session:
                assert session == mock_session
    
    @pytest.mark.asyncio
    async def test_database_manager_get_session_not_initialized(self):
        """Test getting session from uninitialized manager."""
        manager = DatabaseManager()
        
        with pytest.raises(RuntimeError, match="DatabaseManager not initialized"):
            async with manager.get_session() as session:
                pass
    
    @pytest.mark.asyncio
    async def test_database_manager_close(self):
        """Test DatabaseManager close."""
        manager = DatabaseManager()
        manager._initialized = True
        
        with patch('src.database.connection.close_database_connections') as mock_close:
            await manager.close()
        
        mock_close.assert_called_once()
        assert manager._initialized is False


class TestUtilityFunctions:
    """Test utility functions."""
    
    @pytest.mark.asyncio
    async def test_execute_query_without_parameters(self):
        """Test executing query without parameters."""
        mock_session = AsyncMock(spec=AsyncSession)
        mock_result = Mock()
        mock_session.execute.return_value = mock_result
        
        with patch('src.database.connection.get_database_session') as mock_get_session:
            mock_context = AsyncMock()
            mock_context.__aenter__.return_value = mock_session
            mock_context.__aexit__.return_value = None
            mock_get_session.return_value = mock_context
            
            result = await execute_query("SELECT 1")
        
        assert result == mock_result
        mock_session.execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_execute_query_with_parameters(self):
        """Test executing query with parameters."""
        mock_session = AsyncMock(spec=AsyncSession)
        mock_result = Mock()
        mock_session.execute.return_value = mock_result
        
        with patch('src.database.connection.get_database_session') as mock_get_session:
            mock_context = AsyncMock()
            mock_context.__aenter__.return_value = mock_session
            mock_context.__aexit__.return_value = None
            mock_get_session.return_value = mock_context
            
            result = await execute_query("SELECT * FROM videos WHERE id = :id", {"id": 1})
        
        assert result == mock_result
        mock_session.execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_table_count_success(self):
        """Test getting table count successfully."""
        with patch('src.database.connection.execute_query') as mock_execute:
            mock_result = Mock()
            mock_result.scalar.return_value = 42
            mock_execute.return_value = mock_result
            
            count = await get_table_count("videos")
        
        assert count == 42
        mock_execute.assert_called_once_with("SELECT COUNT(*) FROM videos")
    
    @pytest.mark.asyncio
    async def test_get_table_count_error(self):
        """Test getting table count with error."""
        with patch('src.database.connection.execute_query', side_effect=Exception("Query failed")):
            count = await get_table_count("videos")
        
        assert count == 0
    
    @pytest.mark.asyncio
    async def test_get_table_count_none_result(self):
        """Test getting table count with None result."""
        with patch('src.database.connection.execute_query') as mock_execute:
            mock_result = Mock()
            mock_result.scalar.return_value = None
            mock_execute.return_value = mock_result
            
            count = await get_table_count("videos")
        
        assert count == 0
    
    @pytest.mark.asyncio
    async def test_get_database_info_success(self):
        """Test getting database info successfully."""
        mock_health = {'status': 'healthy'}
        
        with patch('src.database.connection.check_database_health', return_value=mock_health):
            with patch('src.database.connection.get_table_count') as mock_count:
                mock_count.side_effect = [10, 5, 3, 2, 1, 0]  # Counts for each table
                
                info = await get_database_info()
        
        assert info['health'] == mock_health
        assert info['total_records'] == 21  # Sum of all counts
        assert info['tables']['videos'] == 10
        assert info['tables']['transcripts'] == 5
        assert info['tables']['summaries'] == 3
        assert info['tables']['keywords'] == 2
        assert info['tables']['timestamped_segments'] == 1
        assert info['tables']['processing_metadata'] == 0
    
    @pytest.mark.asyncio
    async def test_get_database_info_with_errors(self):
        """Test getting database info with some table errors."""
        mock_health = {'status': 'healthy'}
        
        with patch('src.database.connection.check_database_health', return_value=mock_health):
            with patch('src.database.connection.get_table_count') as mock_count:
                mock_count.side_effect = [10, Exception("Error"), 3, 2, 1, 0]
                
                info = await get_database_info()
        
        assert info['health'] == mock_health
        assert info['total_records'] == 16  # Sum excluding error
        assert info['tables']['videos'] == 10
        assert info['tables']['transcripts'] == 'error'
        assert info['tables']['summaries'] == 3


class TestGlobalDatabaseManager:
    """Test global database manager instance."""
    
    @pytest.mark.asyncio
    async def test_global_db_manager_instance(self):
        """Test that global db_manager is a DatabaseManager instance."""
        assert isinstance(db_manager, DatabaseManager)
    
    @pytest.mark.asyncio
    async def test_global_db_manager_functionality(self):
        """Test basic functionality of global db_manager."""
        # Test that it has the expected methods
        assert hasattr(db_manager, 'initialize')
        assert hasattr(db_manager, 'health_check')
        assert hasattr(db_manager, 'get_session')
        assert hasattr(db_manager, 'close')
        
        # Test that it starts uninitialized
        assert db_manager._initialized is False