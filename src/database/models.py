"""
Database models for YouTube Summarizer application.
Defines SQLAlchemy models for storing video processing data.
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
from sqlalchemy import (
    Column, Integer, String, Text, DateTime, JSON, Float, Boolean,
    ForeignKey, UniqueConstraint, Index
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, validates
from sqlalchemy.sql import func

Base = declarative_base()


class Video(Base):
    """
    Video model for storing YouTube video metadata.
    
    Attributes:
        id: Primary key
        video_id: YouTube video ID (unique)
        title: Video title
        duration: Video duration in seconds
        url: YouTube video URL
        created_at: Record creation timestamp
        updated_at: Record last update timestamp
    """
    __tablename__ = 'videos'
    
    id = Column(Integer, primary_key=True, index=True)
    video_id = Column(String(255), unique=True, nullable=False, index=True)
    title = Column(String(1000), nullable=False)
    duration = Column(Integer, nullable=True)  # Duration in seconds
    url = Column(String(2000), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    
    # Relationships
    transcripts = relationship("Transcript", back_populates="video", cascade="all, delete-orphan")
    summaries = relationship("Summary", back_populates="video", cascade="all, delete-orphan")
    keywords = relationship("Keyword", back_populates="video", cascade="all, delete-orphan")
    timestamped_segments = relationship("TimestampedSegment", back_populates="video", cascade="all, delete-orphan")
    processing_metadata = relationship("ProcessingMetadata", back_populates="video", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_video_id', 'video_id'),
        Index('idx_created_at', 'created_at'),
    )
    
    @validates('video_id')
    def validate_video_id(self, key, video_id):
        """Validate YouTube video ID format."""
        if not video_id or len(video_id) != 11:
            raise ValueError("Video ID must be exactly 11 characters long")
        return video_id
    
    @validates('duration')
    def validate_duration(self, key, duration):
        """Validate video duration."""
        if duration is not None and duration <= 0:
            raise ValueError("Duration must be positive")
        return duration
    
    def __repr__(self):
        return f"<Video(id={self.id}, video_id='{self.video_id}', title='{self.title[:50]}...')>"


class Transcript(Base):
    """
    Transcript model for storing video transcripts.
    
    Attributes:
        id: Primary key
        video_id: Foreign key to Video
        content: Full transcript text
        language: Detected language code
        created_at: Record creation timestamp
    """
    __tablename__ = 'transcripts'
    
    id = Column(Integer, primary_key=True, index=True)
    video_id = Column(Integer, ForeignKey('videos.id', ondelete='CASCADE'), nullable=False)
    content = Column(Text, nullable=False)
    language = Column(String(10), nullable=True)  # ISO language code
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Relationships
    video = relationship("Video", back_populates="transcripts")
    
    # Indexes
    __table_args__ = (
        Index('idx_transcript_video_id', 'video_id'),
        Index('idx_transcript_language', 'language'),
    )
    
    @validates('language')
    def validate_language(self, key, language):
        """Validate language code format."""
        if language is not None and len(language) > 10:
            raise ValueError("Language code must be 10 characters or less")
        return language
    
    def __repr__(self):
        return f"<Transcript(id={self.id}, video_id={self.video_id}, language='{self.language}')>"


class Summary(Base):
    """
    Summary model for storing video summaries.
    
    Attributes:
        id: Primary key
        video_id: Foreign key to Video
        content: Summary text
        processing_time: Time taken to generate summary (seconds)
        created_at: Record creation timestamp
    """
    __tablename__ = 'summaries'
    
    id = Column(Integer, primary_key=True, index=True)
    video_id = Column(Integer, ForeignKey('videos.id', ondelete='CASCADE'), nullable=False)
    content = Column(Text, nullable=False)
    processing_time = Column(Float, nullable=True)  # Processing time in seconds
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Relationships
    video = relationship("Video", back_populates="summaries")
    
    # Indexes
    __table_args__ = (
        Index('idx_summary_video_id', 'video_id'),
        Index('idx_summary_created_at', 'created_at'),
    )
    
    @validates('processing_time')
    def validate_processing_time(self, key, processing_time):
        """Validate processing time."""
        if processing_time is not None and processing_time < 0:
            raise ValueError("Processing time cannot be negative")
        return processing_time
    
    def __repr__(self):
        return f"<Summary(id={self.id}, video_id={self.video_id}, processing_time={self.processing_time})>"


class Keyword(Base):
    """
    Keyword model for storing extracted keywords from videos.
    
    Attributes:
        id: Primary key
        video_id: Foreign key to Video
        keywords_json: JSON array of keywords with scores
        created_at: Record creation timestamp
    """
    __tablename__ = 'keywords'
    
    id = Column(Integer, primary_key=True, index=True)
    video_id = Column(Integer, ForeignKey('videos.id', ondelete='CASCADE'), nullable=False)
    keywords_json = Column(JSON, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Relationships
    video = relationship("Video", back_populates="keywords")
    
    # Indexes
    __table_args__ = (
        Index('idx_keyword_video_id', 'video_id'),
    )
    
    @validates('keywords_json')
    def validate_keywords_json(self, key, keywords_json):
        """Validate keywords JSON structure."""
        if not isinstance(keywords_json, (list, dict)):
            raise ValueError("Keywords must be a valid JSON object or array")
        return keywords_json
    
    def __repr__(self):
        keyword_count = len(self.keywords_json) if isinstance(self.keywords_json, list) else 0
        return f"<Keyword(id={self.id}, video_id={self.video_id}, count={keyword_count})>"


class TimestampedSegment(Base):
    """
    Timestamped segment model for storing video segments with timestamps.
    
    Attributes:
        id: Primary key
        video_id: Foreign key to Video
        segments_json: JSON array of segments with timestamps
        created_at: Record creation timestamp
    """
    __tablename__ = 'timestamped_segments'
    
    id = Column(Integer, primary_key=True, index=True)
    video_id = Column(Integer, ForeignKey('videos.id', ondelete='CASCADE'), nullable=False)
    segments_json = Column(JSON, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Relationships
    video = relationship("Video", back_populates="timestamped_segments")
    
    # Indexes
    __table_args__ = (
        Index('idx_timestamped_segment_video_id', 'video_id'),
    )
    
    @validates('segments_json')
    def validate_segments_json(self, key, segments_json):
        """Validate segments JSON structure."""
        if not isinstance(segments_json, (list, dict)):
            raise ValueError("Segments must be a valid JSON object or array")
        return segments_json
    
    def __repr__(self):
        segment_count = len(self.segments_json) if isinstance(self.segments_json, list) else 0
        return f"<TimestampedSegment(id={self.id}, video_id={self.video_id}, count={segment_count})>"


class ProcessingMetadata(Base):
    """
    Processing metadata model for storing workflow processing information.
    
    Attributes:
        id: Primary key
        video_id: Foreign key to Video
        workflow_params: JSON object with workflow parameters
        status: Processing status (pending, processing, completed, failed)
        error_info: Error information if processing failed
        created_at: Record creation timestamp
    """
    __tablename__ = 'processing_metadata'
    
    id = Column(Integer, primary_key=True, index=True)
    video_id = Column(Integer, ForeignKey('videos.id', ondelete='CASCADE'), nullable=False)
    workflow_params = Column(JSON, nullable=True)
    status = Column(String(50), nullable=False, default='pending')
    error_info = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Relationships
    video = relationship("Video", back_populates="processing_metadata")
    
    # Indexes
    __table_args__ = (
        Index('idx_processing_metadata_video_id', 'video_id'),
        Index('idx_processing_metadata_status', 'status'),
        Index('idx_processing_metadata_created_at', 'created_at'),
    )
    
    @validates('status')
    def validate_status(self, key, status):
        """Validate processing status."""
        valid_statuses = ['pending', 'processing', 'completed', 'failed']
        if status not in valid_statuses:
            raise ValueError(f"Status must be one of: {valid_statuses}")
        return status
    
    def __repr__(self):
        return f"<ProcessingMetadata(id={self.id}, video_id={self.video_id}, status='{self.status}')>"


# Model utilities
def get_model_by_name(model_name: str):
    """Get model class by name."""
    models = {
        'Video': Video,
        'Transcript': Transcript,
        'Summary': Summary,
        'Keyword': Keyword,
        'TimestampedSegment': TimestampedSegment,
        'ProcessingMetadata': ProcessingMetadata,
    }
    return models.get(model_name)


def get_all_models():
    """Get all model classes."""
    return [Video, Transcript, Summary, Keyword, TimestampedSegment, ProcessingMetadata]


def create_tables(engine):
    """Create all tables in the database."""
    Base.metadata.create_all(bind=engine)


def drop_tables(engine):
    """Drop all tables from the database."""
    Base.metadata.drop_all(bind=engine)