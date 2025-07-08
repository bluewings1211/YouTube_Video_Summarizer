"""
Comprehensive database operation tests.

This test suite covers:
- Database connection management
- Transaction handling
- CRUD operations for all models
- Error handling and edge cases
- Performance considerations
- Data integrity checks
"""

import pytest
import asyncio
from datetime import datetime, date
from typing import List, Dict, Any
from unittest.mock import Mock, patch, AsyncMock
from sqlalchemy import create_engine, text, select, func
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import IntegrityError, SQLAlchemyError, OperationalError
from sqlalchemy.pool import StaticPool
import json

from src.database.models import (
    Base, Video, Transcript, Summary, Keyword, TimestampedSegment, ProcessingMetadata
)
from src.database.connection import (
    get_database_session, DatabaseManager, get_database_manager,
    test_database_connection, create_database_session
)
from src.database.exceptions import (
    DatabaseError, DatabaseConnectionError, DatabaseQueryError,
    classify_database_error
)


class TestDatabaseConnection:
    """Test database connection functionality."""
    
    @pytest.fixture
    def mock_engine(self):
        """Create mock database engine."""
        engine = Mock()
        engine.connect = Mock()
        engine.dispose = Mock()
        return engine
    
    @pytest.fixture
    def mock_async_engine(self):
        """Create mock async database engine."""
        engine = Mock()
        engine.connect = AsyncMock()
        engine.dispose = AsyncMock()
        return engine
    
    @pytest.mark.asyncio
    async def test_database_connection_success(self, mock_async_engine):
        """Test successful database connection."""
        # Mock successful connection
        mock_connection = Mock()
        mock_async_engine.connect.return_value.__aenter__.return_value = mock_connection
        
        # Test connection
        with patch('src.database.connection.create_async_engine', return_value=mock_async_engine):
            result = await test_database_connection()
            assert result is True
    
    @pytest.mark.asyncio
    async def test_database_connection_failure(self, mock_async_engine):
        """Test database connection failure."""
        # Mock connection failure
        mock_async_engine.connect.side_effect = OperationalError("Connection failed", None, None)
        
        # Test connection failure
        with patch('src.database.connection.create_async_engine', return_value=mock_async_engine):
            result = await test_database_connection()
            assert result is False
    
    @pytest.mark.asyncio
    async def test_database_session_creation(self):
        """Test database session creation."""
        # Create in-memory database for testing
        test_url = "sqlite+aiosqlite:///:memory:"
        engine = create_async_engine(test_url, echo=False)
        
        # Test session creation
        session = await create_database_session(engine)
        assert isinstance(session, AsyncSession)
        await session.close()
        await engine.dispose()
    
    def test_database_manager_initialization(self):
        """Test DatabaseManager initialization."""
        manager = DatabaseManager()
        assert manager.engine is None
        assert manager.session_factory is None
    
    @pytest.mark.asyncio
    async def test_database_manager_initialize(self):
        """Test DatabaseManager initialize method."""
        manager = DatabaseManager()
        test_url = "sqlite+aiosqlite:///:memory:"
        
        await manager.initialize(test_url)
        
        assert manager.engine is not None
        assert manager.session_factory is not None
        
        # Cleanup
        await manager.close()
    
    @pytest.mark.asyncio
    async def test_database_manager_get_session(self):
        """Test DatabaseManager get_session method."""
        manager = DatabaseManager()
        test_url = "sqlite+aiosqlite:///:memory:"
        
        await manager.initialize(test_url)
        
        session = await manager.get_session()
        assert isinstance(session, AsyncSession)
        
        await session.close()
        await manager.close()
    
    @pytest.mark.asyncio
    async def test_database_manager_close(self):
        """Test DatabaseManager close method."""
        manager = DatabaseManager()
        test_url = "sqlite+aiosqlite:///:memory:"
        
        await manager.initialize(test_url)
        await manager.close()
        
        # Engine should be disposed
        assert manager.engine is None


class TestDatabaseCRUDOperations:
    """Test CRUD operations for all database models."""
    
    @pytest.fixture
    async def test_session(self):
        """Create async test database session."""
        # Create in-memory database
        test_url = "sqlite+aiosqlite:///:memory:"
        engine = create_async_engine(test_url, echo=False)
        
        # Create tables
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        # Create session
        async_session = sessionmaker(
            bind=engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        
        session = async_session()
        yield session
        
        # Cleanup
        await session.close()
        await engine.dispose()
    
    @pytest.mark.asyncio
    async def test_video_crud_operations(self, test_session):
        """Test Video CRUD operations."""
        # Create video
        video = Video(
            video_id="dQw4w9WgXcQ",
            title="Test Video",
            duration=212,
            url="https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        )
        
        # Test Create
        test_session.add(video)
        await test_session.commit()
        assert video.id is not None
        
        # Test Read
        stmt = select(Video).where(Video.video_id == "dQw4w9WgXcQ")
        result = await test_session.execute(stmt)
        retrieved_video = result.scalar_one()
        assert retrieved_video.title == "Test Video"
        assert retrieved_video.duration == 212
        
        # Test Update
        retrieved_video.title = "Updated Test Video"
        await test_session.commit()
        
        # Verify update
        stmt = select(Video).where(Video.id == video.id)
        result = await test_session.execute(stmt)
        updated_video = result.scalar_one()
        assert updated_video.title == "Updated Test Video"
        
        # Test Delete
        await test_session.delete(updated_video)
        await test_session.commit()
        
        # Verify deletion
        stmt = select(Video).where(Video.id == video.id)
        result = await test_session.execute(stmt)
        deleted_video = result.scalar_one_or_none()
        assert deleted_video is None
    
    @pytest.mark.asyncio
    async def test_transcript_crud_operations(self, test_session):
        """Test Transcript CRUD operations."""
        # Create video first
        video = Video(
            video_id="dQw4w9WgXcQ",
            title="Test Video",
            url="https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        )
        test_session.add(video)
        await test_session.commit()
        
        # Create transcript
        transcript = Transcript(
            video_id=video.id,
            content="This is a test transcript content.",
            language="en"
        )
        
        # Test Create
        test_session.add(transcript)
        await test_session.commit()
        assert transcript.id is not None
        
        # Test Read
        stmt = select(Transcript).where(Transcript.video_id == video.id)
        result = await test_session.execute(stmt)
        retrieved_transcript = result.scalar_one()
        assert retrieved_transcript.content == "This is a test transcript content."
        assert retrieved_transcript.language == "en"
        
        # Test Update
        retrieved_transcript.content = "Updated transcript content."
        await test_session.commit()
        
        # Verify update
        stmt = select(Transcript).where(Transcript.id == transcript.id)
        result = await test_session.execute(stmt)
        updated_transcript = result.scalar_one()
        assert updated_transcript.content == "Updated transcript content."
    
    @pytest.mark.asyncio
    async def test_summary_crud_operations(self, test_session):
        """Test Summary CRUD operations."""
        # Create video first
        video = Video(
            video_id="dQw4w9WgXcQ",
            title="Test Video",
            url="https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        )
        test_session.add(video)
        await test_session.commit()
        
        # Create summary
        summary = Summary(
            video_id=video.id,
            content="This is a test summary.",
            processing_time=15.5
        )
        
        # Test Create
        test_session.add(summary)
        await test_session.commit()
        assert summary.id is not None
        
        # Test Read
        stmt = select(Summary).where(Summary.video_id == video.id)
        result = await test_session.execute(stmt)
        retrieved_summary = result.scalar_one()
        assert retrieved_summary.content == "This is a test summary."
        assert retrieved_summary.processing_time == 15.5
    
    @pytest.mark.asyncio
    async def test_bulk_operations(self, test_session):
        """Test bulk database operations."""
        # Create multiple videos
        videos = [
            Video(
                video_id=f"test_vid_{i}",
                title=f"Test Video {i}",
                url=f"https://www.youtube.com/watch?v=test_vid_{i}"
            )
            for i in range(10)
        ]
        
        # Bulk insert
        test_session.add_all(videos)
        await test_session.commit()
        
        # Verify all videos were created
        stmt = select(func.count(Video.id))
        result = await test_session.execute(stmt)
        count = result.scalar()
        assert count == 10
        
        # Bulk update
        stmt = select(Video)
        result = await test_session.execute(stmt)
        all_videos = result.scalars().all()
        
        for video in all_videos:
            video.title = f"Updated {video.title}"
        
        await test_session.commit()
        
        # Verify updates
        stmt = select(Video).where(Video.title.like("Updated%"))
        result = await test_session.execute(stmt)
        updated_videos = result.scalars().all()
        assert len(updated_videos) == 10
    
    @pytest.mark.asyncio
    async def test_transaction_rollback(self, test_session):
        """Test transaction rollback functionality."""
        # Create video
        video = Video(
            video_id="dQw4w9WgXcQ",
            title="Test Video",
            url="https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        )
        test_session.add(video)
        await test_session.commit()
        
        # Start transaction that will fail
        try:
            # Add duplicate video (should fail due to unique constraint)
            duplicate_video = Video(
                video_id="dQw4w9WgXcQ",  # Same video_id
                title="Duplicate Video",
                url="https://www.youtube.com/watch?v=dQw4w9WgXcQ"
            )
            test_session.add(duplicate_video)
            await test_session.commit()
            
            # This should not be reached
            assert False, "Expected IntegrityError"
        except IntegrityError:
            await test_session.rollback()
        
        # Verify original video still exists
        stmt = select(Video).where(Video.video_id == "dQw4w9WgXcQ")
        result = await test_session.execute(stmt)
        video = result.scalar_one()
        assert video.title == "Test Video"
    
    @pytest.mark.asyncio
    async def test_relationship_operations(self, test_session):
        """Test operations involving model relationships."""
        # Create video
        video = Video(
            video_id="dQw4w9WgXcQ",
            title="Test Video",
            url="https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        )
        test_session.add(video)
        await test_session.commit()
        
        # Create related records
        transcript = Transcript(
            video_id=video.id,
            content="Test transcript",
            language="en"
        )
        summary = Summary(
            video_id=video.id,
            content="Test summary",
            processing_time=10.0
        )
        keyword = Keyword(
            video_id=video.id,
            keywords_json=[{"keyword": "test", "score": 0.9}]
        )
        
        test_session.add_all([transcript, summary, keyword])
        await test_session.commit()
        
        # Test reading with relationships
        stmt = select(Video).where(Video.id == video.id)
        result = await test_session.execute(stmt)
        video_with_relations = result.scalar_one()
        
        # Load relationships
        await test_session.refresh(video_with_relations, ['transcripts', 'summaries', 'keywords'])
        
        assert len(video_with_relations.transcripts) == 1
        assert len(video_with_relations.summaries) == 1
        assert len(video_with_relations.keywords) == 1
        
        # Test cascade delete
        await test_session.delete(video_with_relations)
        await test_session.commit()
        
        # Verify all related records are deleted
        stmt = select(func.count(Transcript.id))
        result = await test_session.execute(stmt)
        transcript_count = result.scalar()
        assert transcript_count == 0
        
        stmt = select(func.count(Summary.id))
        result = await test_session.execute(stmt)
        summary_count = result.scalar()
        assert summary_count == 0
        
        stmt = select(func.count(Keyword.id))
        result = await test_session.execute(stmt)
        keyword_count = result.scalar()
        assert keyword_count == 0


class TestDatabaseErrorHandling:
    """Test database error handling and classification."""
    
    def test_classify_database_error_connection(self):
        """Test classification of connection errors."""
        # Mock connection error
        error = OperationalError("Connection failed", None, None)
        classified = classify_database_error(error)
        
        assert isinstance(classified, DatabaseConnectionError)
        assert "Connection failed" in str(classified)
    
    def test_classify_database_error_query(self):
        """Test classification of query errors."""
        # Mock query error
        error = IntegrityError("Integrity constraint failed", None, None)
        classified = classify_database_error(error)
        
        assert isinstance(classified, DatabaseQueryError)
        assert "Integrity constraint failed" in str(classified)
    
    def test_classify_database_error_generic(self):
        """Test classification of generic database errors."""
        # Mock generic error
        error = SQLAlchemyError("Generic database error")
        classified = classify_database_error(error)
        
        assert isinstance(classified, DatabaseError)
        assert "Generic database error" in str(classified)
    
    def test_database_error_hierarchy(self):
        """Test database error class hierarchy."""
        # Test inheritance
        assert issubclass(DatabaseConnectionError, DatabaseError)
        assert issubclass(DatabaseQueryError, DatabaseError)
        
        # Test instantiation
        conn_error = DatabaseConnectionError("Connection failed")
        query_error = DatabaseQueryError("Query failed")
        
        assert str(conn_error) == "Connection failed"
        assert str(query_error) == "Query failed"
    
    @pytest.mark.asyncio
    async def test_error_handling_in_operations(self):
        """Test error handling in database operations."""
        # Create invalid async session (should fail)
        invalid_engine = create_async_engine("invalid://connection", echo=False)
        
        with pytest.raises(Exception):
            # This should raise an error
            async with invalid_engine.connect() as conn:
                await conn.execute(text("SELECT 1"))


class TestDatabasePerformance:
    """Test database performance considerations."""
    
    @pytest.fixture
    async def performance_test_session(self):
        """Create test session with performance monitoring."""
        test_url = "sqlite+aiosqlite:///:memory:"
        engine = create_async_engine(test_url, echo=False)
        
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        async_session = sessionmaker(
            bind=engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        
        session = async_session()
        yield session
        
        await session.close()
        await engine.dispose()
    
    @pytest.mark.asyncio
    async def test_bulk_insert_performance(self, performance_test_session):
        """Test bulk insert performance."""
        import time
        
        # Create large number of videos
        videos = [
            Video(
                video_id=f"perf_test_{i:06d}",
                title=f"Performance Test Video {i}",
                url=f"https://www.youtube.com/watch?v=perf_test_{i:06d}"
            )
            for i in range(1000)
        ]
        
        # Measure bulk insert time
        start_time = time.time()
        performance_test_session.add_all(videos)
        await performance_test_session.commit()
        end_time = time.time()
        
        insert_time = end_time - start_time
        assert insert_time < 5.0  # Should complete within 5 seconds
        
        # Verify all records were inserted
        stmt = select(func.count(Video.id))
        result = await performance_test_session.execute(stmt)
        count = result.scalar()
        assert count == 1000
    
    @pytest.mark.asyncio
    async def test_query_performance(self, performance_test_session):
        """Test query performance with indexes."""
        # Create test data
        videos = [
            Video(
                video_id=f"query_test_{i:06d}",
                title=f"Query Test Video {i}",
                url=f"https://www.youtube.com/watch?v=query_test_{i:06d}"
            )
            for i in range(100)
        ]
        
        performance_test_session.add_all(videos)
        await performance_test_session.commit()
        
        # Test indexed query performance
        import time
        start_time = time.time()
        
        # Query by video_id (indexed)
        stmt = select(Video).where(Video.video_id == "query_test_000050")
        result = await performance_test_session.execute(stmt)
        video = result.scalar_one()
        
        end_time = time.time()
        query_time = end_time - start_time
        
        assert query_time < 0.1  # Should be very fast with index
        assert video.video_id == "query_test_000050"
    
    @pytest.mark.asyncio
    async def test_relationship_loading_performance(self, performance_test_session):
        """Test relationship loading performance."""
        # Create video with many related records
        video = Video(
            video_id="perf_video_1",
            title="Performance Video",
            url="https://www.youtube.com/watch?v=perf_video_1"
        )
        performance_test_session.add(video)
        await performance_test_session.commit()
        
        # Create many related records
        transcripts = [
            Transcript(
                video_id=video.id,
                content=f"Transcript segment {i}",
                language="en"
            )
            for i in range(100)
        ]
        
        summaries = [
            Summary(
                video_id=video.id,
                content=f"Summary {i}",
                processing_time=float(i)
            )
            for i in range(50)
        ]
        
        performance_test_session.add_all(transcripts + summaries)
        await performance_test_session.commit()
        
        # Test loading with relationships
        import time
        start_time = time.time()
        
        stmt = select(Video).where(Video.id == video.id)
        result = await performance_test_session.execute(stmt)
        video_with_relations = result.scalar_one()
        
        # Load relationships
        await performance_test_session.refresh(video_with_relations, ['transcripts', 'summaries'])
        
        end_time = time.time()
        loading_time = end_time - start_time
        
        assert loading_time < 1.0  # Should complete within 1 second
        assert len(video_with_relations.transcripts) == 100
        assert len(video_with_relations.summaries) == 50


class TestDatabaseIntegrity:
    """Test database integrity and constraints."""
    
    @pytest.fixture
    async def integrity_test_session(self):
        """Create test session for integrity testing."""
        test_url = "sqlite+aiosqlite:///:memory:"
        engine = create_async_engine(test_url, echo=False)
        
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        async_session = sessionmaker(
            bind=engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        
        session = async_session()
        yield session
        
        await session.close()
        await engine.dispose()
    
    @pytest.mark.asyncio
    async def test_unique_constraints(self, integrity_test_session):
        """Test unique constraints enforcement."""
        # Create video
        video1 = Video(
            video_id="unique_test_1",
            title="Unique Test Video 1",
            url="https://www.youtube.com/watch?v=unique_test_1"
        )
        integrity_test_session.add(video1)
        await integrity_test_session.commit()
        
        # Try to create video with same video_id
        video2 = Video(
            video_id="unique_test_1",  # Same video_id
            title="Unique Test Video 2",
            url="https://www.youtube.com/watch?v=unique_test_1"
        )
        integrity_test_session.add(video2)
        
        with pytest.raises(IntegrityError):
            await integrity_test_session.commit()
    
    @pytest.mark.asyncio
    async def test_foreign_key_constraints(self, integrity_test_session):
        """Test foreign key constraints enforcement."""
        # Try to create transcript without valid video
        transcript = Transcript(
            video_id=999,  # Non-existent video ID
            content="Test transcript",
            language="en"
        )
        integrity_test_session.add(transcript)
        
        with pytest.raises(IntegrityError):
            await integrity_test_session.commit()
    
    @pytest.mark.asyncio
    async def test_cascade_operations(self, integrity_test_session):
        """Test cascade delete operations."""
        # Create video with related records
        video = Video(
            video_id="cascade_test_1",
            title="Cascade Test Video",
            url="https://www.youtube.com/watch?v=cascade_test_1"
        )
        integrity_test_session.add(video)
        await integrity_test_session.commit()
        
        # Create related records
        transcript = Transcript(
            video_id=video.id,
            content="Test transcript",
            language="en"
        )
        summary = Summary(
            video_id=video.id,
            content="Test summary",
            processing_time=10.0
        )
        
        integrity_test_session.add_all([transcript, summary])
        await integrity_test_session.commit()
        
        # Store IDs for verification
        transcript_id = transcript.id
        summary_id = summary.id
        
        # Delete video
        await integrity_test_session.delete(video)
        await integrity_test_session.commit()
        
        # Verify related records are deleted
        stmt = select(Transcript).where(Transcript.id == transcript_id)
        result = await integrity_test_session.execute(stmt)
        assert result.scalar_one_or_none() is None
        
        stmt = select(Summary).where(Summary.id == summary_id)
        result = await integrity_test_session.execute(stmt)
        assert result.scalar_one_or_none() is None
    
    @pytest.mark.asyncio
    async def test_data_validation_constraints(self, integrity_test_session):
        """Test data validation constraints."""
        # Test video_id length validation
        with pytest.raises(ValueError, match="Video ID must be exactly 11 characters long"):
            video = Video(
                video_id="short",  # Too short
                title="Test Video",
                url="https://www.youtube.com/watch?v=short"
            )
            integrity_test_session.add(video)
            await integrity_test_session.commit()
        
        # Test duration validation
        with pytest.raises(ValueError, match="Duration must be positive"):
            video = Video(
                video_id="duration_test",
                title="Test Video",
                duration=-10,  # Negative duration
                url="https://www.youtube.com/watch?v=duration_test"
            )
            integrity_test_session.add(video)
            await integrity_test_session.commit()
        
        # Test processing time validation
        video = Video(
            video_id="proc_time_test",
            title="Test Video",
            url="https://www.youtube.com/watch?v=proc_time_test"
        )
        integrity_test_session.add(video)
        await integrity_test_session.commit()
        
        with pytest.raises(ValueError, match="Processing time cannot be negative"):
            summary = Summary(
                video_id=video.id,
                content="Test summary",
                processing_time=-5.0  # Negative processing time
            )
            integrity_test_session.add(summary)
            await integrity_test_session.commit()