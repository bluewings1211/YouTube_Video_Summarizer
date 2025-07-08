"""
Comprehensive tests for database models.

This test suite covers:
- Model creation and validation
- Model relationships
- Model validators
- Model constraints
- Model serialization
"""

import pytest
from datetime import datetime, timezone
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import IntegrityError, StatementError
from sqlalchemy.pool import StaticPool
import json

from src.database.models import (
    Base, Video, Transcript, Summary, Keyword, TimestampedSegment, ProcessingMetadata,
    get_model_by_name, get_all_models, create_tables, drop_tables
)


@pytest.fixture(scope="function")
def test_engine():
    """Create test database engine with SQLite in-memory database."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=False
    )
    return engine


@pytest.fixture(scope="function")
def test_session(test_engine):
    """Create test database session."""
    # Create tables
    Base.metadata.create_all(test_engine)
    
    # Create session
    TestSession = sessionmaker(bind=test_engine)
    session = TestSession()
    
    yield session
    
    # Cleanup
    session.close()
    Base.metadata.drop_all(test_engine)


class TestVideoModel:
    """Test Video model functionality."""
    
    def test_video_creation(self, test_session):
        """Test creating a video record."""
        video = Video(
            video_id="dQw4w9WgXcQ",
            title="Never Gonna Give You Up",
            duration=212,
            url="https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        )
        
        test_session.add(video)
        test_session.commit()
        
        assert video.id is not None
        assert video.video_id == "dQw4w9WgXcQ"
        assert video.title == "Never Gonna Give You Up"
        assert video.duration == 212
        assert video.url == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        assert video.created_at is not None
        assert video.updated_at is not None
    
    def test_video_id_validation(self, test_session):
        """Test video ID validation."""
        # Test invalid video ID length
        with pytest.raises(ValueError, match="Video ID must be exactly 11 characters long"):
            video = Video(
                video_id="short",
                title="Test Video",
                url="https://www.youtube.com/watch?v=short"
            )
            test_session.add(video)
            test_session.commit()
        
        # Test empty video ID
        with pytest.raises(ValueError, match="Video ID must be exactly 11 characters long"):
            video = Video(
                video_id="",
                title="Test Video",
                url="https://www.youtube.com/watch?v="
            )
            test_session.add(video)
            test_session.commit()
    
    def test_duration_validation(self, test_session):
        """Test duration validation."""
        # Test negative duration
        with pytest.raises(ValueError, match="Duration must be positive"):
            video = Video(
                video_id="dQw4w9WgXcQ",
                title="Test Video",
                duration=-10,
                url="https://www.youtube.com/watch?v=dQw4w9WgXcQ"
            )
            test_session.add(video)
            test_session.commit()
        
        # Test zero duration
        with pytest.raises(ValueError, match="Duration must be positive"):
            video = Video(
                video_id="dQw4w9WgXcQ",
                title="Test Video",
                duration=0,
                url="https://www.youtube.com/watch?v=dQw4w9WgXcQ"
            )
            test_session.add(video)
            test_session.commit()
    
    def test_video_unique_constraint(self, test_session):
        """Test video_id unique constraint."""
        video1 = Video(
            video_id="dQw4w9WgXcQ",
            title="Video 1",
            url="https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        )
        test_session.add(video1)
        test_session.commit()
        
        # Try to create another video with same video_id
        video2 = Video(
            video_id="dQw4w9WgXcQ",
            title="Video 2",
            url="https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        )
        test_session.add(video2)
        
        with pytest.raises(IntegrityError):
            test_session.commit()
    
    def test_video_repr(self, test_session):
        """Test video string representation."""
        video = Video(
            video_id="dQw4w9WgXcQ",
            title="Never Gonna Give You Up - Rick Astley (Official Music Video)",
            url="https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        )
        test_session.add(video)
        test_session.commit()
        
        repr_str = repr(video)
        assert "Video(" in repr_str
        assert "dQw4w9WgXcQ" in repr_str
        assert "Never Gonna Give You Up - Rick Astley (Official M" in repr_str


class TestTranscriptModel:
    """Test Transcript model functionality."""
    
    def test_transcript_creation(self, test_session):
        """Test creating a transcript record."""
        # Create video first
        video = Video(
            video_id="dQw4w9WgXcQ",
            title="Test Video",
            url="https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        )
        test_session.add(video)
        test_session.commit()
        
        # Create transcript
        transcript = Transcript(
            video_id=video.id,
            content="This is a test transcript content.",
            language="en"
        )
        test_session.add(transcript)
        test_session.commit()
        
        assert transcript.id is not None
        assert transcript.video_id == video.id
        assert transcript.content == "This is a test transcript content."
        assert transcript.language == "en"
        assert transcript.created_at is not None
    
    def test_transcript_language_validation(self, test_session):
        """Test transcript language validation."""
        # Create video first
        video = Video(
            video_id="dQw4w9WgXcQ",
            title="Test Video",
            url="https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        )
        test_session.add(video)
        test_session.commit()
        
        # Test invalid language code (too long)
        with pytest.raises(ValueError, match="Language code must be 10 characters or less"):
            transcript = Transcript(
                video_id=video.id,
                content="Test content",
                language="this-is-too-long"
            )
            test_session.add(transcript)
            test_session.commit()
    
    def test_transcript_relationship(self, test_session):
        """Test transcript-video relationship."""
        # Create video
        video = Video(
            video_id="dQw4w9WgXcQ",
            title="Test Video",
            url="https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        )
        test_session.add(video)
        test_session.commit()
        
        # Create transcript
        transcript = Transcript(
            video_id=video.id,
            content="Test content",
            language="en"
        )
        test_session.add(transcript)
        test_session.commit()
        
        # Test relationship
        assert transcript.video == video
        assert transcript in video.transcripts
    
    def test_transcript_cascade_delete(self, test_session):
        """Test cascade delete when video is deleted."""
        # Create video with transcript
        video = Video(
            video_id="dQw4w9WgXcQ",
            title="Test Video",
            url="https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        )
        test_session.add(video)
        test_session.commit()
        
        transcript = Transcript(
            video_id=video.id,
            content="Test content",
            language="en"
        )
        test_session.add(transcript)
        test_session.commit()
        
        transcript_id = transcript.id
        
        # Delete video
        test_session.delete(video)
        test_session.commit()
        
        # Check that transcript is also deleted
        deleted_transcript = test_session.query(Transcript).filter_by(id=transcript_id).first()
        assert deleted_transcript is None


class TestSummaryModel:
    """Test Summary model functionality."""
    
    def test_summary_creation(self, test_session):
        """Test creating a summary record."""
        # Create video first
        video = Video(
            video_id="dQw4w9WgXcQ",
            title="Test Video",
            url="https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        )
        test_session.add(video)
        test_session.commit()
        
        # Create summary
        summary = Summary(
            video_id=video.id,
            content="This is a test summary of the video.",
            processing_time=15.5
        )
        test_session.add(summary)
        test_session.commit()
        
        assert summary.id is not None
        assert summary.video_id == video.id
        assert summary.content == "This is a test summary of the video."
        assert summary.processing_time == 15.5
        assert summary.created_at is not None
    
    def test_summary_processing_time_validation(self, test_session):
        """Test processing time validation."""
        # Create video first
        video = Video(
            video_id="dQw4w9WgXcQ",
            title="Test Video",
            url="https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        )
        test_session.add(video)
        test_session.commit()
        
        # Test negative processing time
        with pytest.raises(ValueError, match="Processing time cannot be negative"):
            summary = Summary(
                video_id=video.id,
                content="Test summary",
                processing_time=-5.0
            )
            test_session.add(summary)
            test_session.commit()
    
    def test_summary_relationship(self, test_session):
        """Test summary-video relationship."""
        # Create video
        video = Video(
            video_id="dQw4w9WgXcQ",
            title="Test Video",
            url="https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        )
        test_session.add(video)
        test_session.commit()
        
        # Create summary
        summary = Summary(
            video_id=video.id,
            content="Test summary",
            processing_time=10.0
        )
        test_session.add(summary)
        test_session.commit()
        
        # Test relationship
        assert summary.video == video
        assert summary in video.summaries


class TestKeywordModel:
    """Test Keyword model functionality."""
    
    def test_keyword_creation(self, test_session):
        """Test creating a keyword record."""
        # Create video first
        video = Video(
            video_id="dQw4w9WgXcQ",
            title="Test Video",
            url="https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        )
        test_session.add(video)
        test_session.commit()
        
        # Create keyword
        keywords_data = [
            {"keyword": "test", "score": 0.9},
            {"keyword": "video", "score": 0.8}
        ]
        keyword = Keyword(
            video_id=video.id,
            keywords_json=keywords_data
        )
        test_session.add(keyword)
        test_session.commit()
        
        assert keyword.id is not None
        assert keyword.video_id == video.id
        assert keyword.keywords_json == keywords_data
        assert keyword.created_at is not None
    
    def test_keyword_json_validation(self, test_session):
        """Test keywords JSON validation."""
        # Create video first
        video = Video(
            video_id="dQw4w9WgXcQ",
            title="Test Video",
            url="https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        )
        test_session.add(video)
        test_session.commit()
        
        # Test invalid JSON (string instead of dict/list)
        with pytest.raises(ValueError, match="Keywords must be a valid JSON object or array"):
            keyword = Keyword(
                video_id=video.id,
                keywords_json="invalid json string"
            )
            test_session.add(keyword)
            test_session.commit()
    
    def test_keyword_repr(self, test_session):
        """Test keyword string representation."""
        # Create video first
        video = Video(
            video_id="dQw4w9WgXcQ",
            title="Test Video",
            url="https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        )
        test_session.add(video)
        test_session.commit()
        
        # Create keyword
        keywords_data = [
            {"keyword": "test", "score": 0.9},
            {"keyword": "video", "score": 0.8}
        ]
        keyword = Keyword(
            video_id=video.id,
            keywords_json=keywords_data
        )
        test_session.add(keyword)
        test_session.commit()
        
        repr_str = repr(keyword)
        assert "Keyword(" in repr_str
        assert "count=2" in repr_str


class TestTimestampedSegmentModel:
    """Test TimestampedSegment model functionality."""
    
    def test_timestamped_segment_creation(self, test_session):
        """Test creating a timestamped segment record."""
        # Create video first
        video = Video(
            video_id="dQw4w9WgXcQ",
            title="Test Video",
            url="https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        )
        test_session.add(video)
        test_session.commit()
        
        # Create timestamped segment
        segments_data = [
            {"start": 0, "end": 30, "text": "First segment"},
            {"start": 30, "end": 60, "text": "Second segment"}
        ]
        segment = TimestampedSegment(
            video_id=video.id,
            segments_json=segments_data
        )
        test_session.add(segment)
        test_session.commit()
        
        assert segment.id is not None
        assert segment.video_id == video.id
        assert segment.segments_json == segments_data
        assert segment.created_at is not None
    
    def test_segments_json_validation(self, test_session):
        """Test segments JSON validation."""
        # Create video first
        video = Video(
            video_id="dQw4w9WgXcQ",
            title="Test Video",
            url="https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        )
        test_session.add(video)
        test_session.commit()
        
        # Test invalid JSON (string instead of dict/list)
        with pytest.raises(ValueError, match="Segments must be a valid JSON object or array"):
            segment = TimestampedSegment(
                video_id=video.id,
                segments_json="invalid json string"
            )
            test_session.add(segment)
            test_session.commit()


class TestProcessingMetadataModel:
    """Test ProcessingMetadata model functionality."""
    
    def test_processing_metadata_creation(self, test_session):
        """Test creating a processing metadata record."""
        # Create video first
        video = Video(
            video_id="dQw4w9WgXcQ",
            title="Test Video",
            url="https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        )
        test_session.add(video)
        test_session.commit()
        
        # Create processing metadata
        workflow_params = {
            "model": "gpt-3.5-turbo",
            "temperature": 0.7,
            "max_tokens": 1000
        }
        metadata = ProcessingMetadata(
            video_id=video.id,
            workflow_params=workflow_params,
            status="completed"
        )
        test_session.add(metadata)
        test_session.commit()
        
        assert metadata.id is not None
        assert metadata.video_id == video.id
        assert metadata.workflow_params == workflow_params
        assert metadata.status == "completed"
        assert metadata.error_info is None
        assert metadata.created_at is not None
    
    def test_status_validation(self, test_session):
        """Test status validation."""
        # Create video first
        video = Video(
            video_id="dQw4w9WgXcQ",
            title="Test Video",
            url="https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        )
        test_session.add(video)
        test_session.commit()
        
        # Test invalid status
        with pytest.raises(ValueError, match="Status must be one of"):
            metadata = ProcessingMetadata(
                video_id=video.id,
                status="invalid_status"
            )
            test_session.add(metadata)
            test_session.commit()
    
    def test_valid_statuses(self, test_session):
        """Test all valid statuses."""
        # Create video first
        video = Video(
            video_id="dQw4w9WgXcQ",
            title="Test Video",
            url="https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        )
        test_session.add(video)
        test_session.commit()
        
        valid_statuses = ['pending', 'processing', 'completed', 'failed']
        
        for status in valid_statuses:
            metadata = ProcessingMetadata(
                video_id=video.id,
                status=status
            )
            test_session.add(metadata)
            test_session.commit()
            
            assert metadata.status == status
            test_session.delete(metadata)
            test_session.commit()


class TestModelUtilities:
    """Test model utility functions."""
    
    def test_get_model_by_name(self):
        """Test getting model by name."""
        assert get_model_by_name('Video') == Video
        assert get_model_by_name('Transcript') == Transcript
        assert get_model_by_name('Summary') == Summary
        assert get_model_by_name('Keyword') == Keyword
        assert get_model_by_name('TimestampedSegment') == TimestampedSegment
        assert get_model_by_name('ProcessingMetadata') == ProcessingMetadata
        assert get_model_by_name('NonExistentModel') is None
    
    def test_get_all_models(self):
        """Test getting all models."""
        models = get_all_models()
        expected_models = [Video, Transcript, Summary, Keyword, TimestampedSegment, ProcessingMetadata]
        
        assert len(models) == len(expected_models)
        for model in expected_models:
            assert model in models
    
    def test_create_drop_tables(self, test_engine):
        """Test creating and dropping tables."""
        # Initially, tables should not exist
        with test_engine.connect() as conn:
            try:
                conn.execute(text("SELECT 1 FROM videos LIMIT 1"))
                tables_exist = True
            except:
                tables_exist = False
        
        assert not tables_exist
        
        # Create tables
        create_tables(test_engine)
        
        # Now tables should exist
        with test_engine.connect() as conn:
            result = conn.execute(text("SELECT 1 FROM videos LIMIT 1"))
            # Should not raise an exception
        
        # Drop tables
        drop_tables(test_engine)
        
        # Tables should no longer exist
        with test_engine.connect() as conn:
            try:
                conn.execute(text("SELECT 1 FROM videos LIMIT 1"))
                tables_exist = True
            except:
                tables_exist = False
        
        assert not tables_exist


class TestModelRelationships:
    """Test model relationships and foreign key constraints."""
    
    def test_video_relationships(self, test_session):
        """Test video model relationships."""
        # Create video
        video = Video(
            video_id="dQw4w9WgXcQ",
            title="Test Video",
            url="https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        )
        test_session.add(video)
        test_session.commit()
        
        # Create related records
        transcript = Transcript(video_id=video.id, content="Test transcript")
        summary = Summary(video_id=video.id, content="Test summary")
        keyword = Keyword(video_id=video.id, keywords_json=[])
        segment = TimestampedSegment(video_id=video.id, segments_json=[])
        metadata = ProcessingMetadata(video_id=video.id, status="pending")
        
        test_session.add_all([transcript, summary, keyword, segment, metadata])
        test_session.commit()
        
        # Refresh video to load relationships
        test_session.refresh(video)
        
        # Test relationships
        assert len(video.transcripts) == 1
        assert len(video.summaries) == 1
        assert len(video.keywords) == 1
        assert len(video.timestamped_segments) == 1
        assert len(video.processing_metadata) == 1
        
        assert video.transcripts[0] == transcript
        assert video.summaries[0] == summary
        assert video.keywords[0] == keyword
        assert video.timestamped_segments[0] == segment
        assert video.processing_metadata[0] == metadata
    
    def test_foreign_key_constraints(self, test_session):
        """Test foreign key constraints."""
        # Try to create transcript without video
        transcript = Transcript(
            video_id=999,  # Non-existent video ID
            content="Test transcript"
        )
        test_session.add(transcript)
        
        with pytest.raises(IntegrityError):
            test_session.commit()
    
    def test_cascade_delete_all_relationships(self, test_session):
        """Test cascade delete for all relationships."""
        # Create video with all related records
        video = Video(
            video_id="dQw4w9WgXcQ",
            title="Test Video",
            url="https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        )
        test_session.add(video)
        test_session.commit()
        
        # Create multiple related records
        transcript = Transcript(video_id=video.id, content="Test transcript")
        summary = Summary(video_id=video.id, content="Test summary")
        keyword = Keyword(video_id=video.id, keywords_json=[])
        segment = TimestampedSegment(video_id=video.id, segments_json=[])
        metadata = ProcessingMetadata(video_id=video.id, status="pending")
        
        test_session.add_all([transcript, summary, keyword, segment, metadata])
        test_session.commit()
        
        # Store IDs for checking
        related_ids = {
            'transcript': transcript.id,
            'summary': summary.id,
            'keyword': keyword.id,
            'segment': segment.id,
            'metadata': metadata.id
        }
        
        # Delete video
        test_session.delete(video)
        test_session.commit()
        
        # Check that all related records are deleted
        assert test_session.query(Transcript).filter_by(id=related_ids['transcript']).first() is None
        assert test_session.query(Summary).filter_by(id=related_ids['summary']).first() is None
        assert test_session.query(Keyword).filter_by(id=related_ids['keyword']).first() is None
        assert test_session.query(TimestampedSegment).filter_by(id=related_ids['segment']).first() is None
        assert test_session.query(ProcessingMetadata).filter_by(id=related_ids['metadata']).first() is None