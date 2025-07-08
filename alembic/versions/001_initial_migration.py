"""Initial migration - Create all database tables

Revision ID: 001
Revises: 
Create Date: 2025-07-08 14:50:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '001'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create videos table
    op.create_table(
        'videos',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('video_id', sa.String(length=255), nullable=False),
        sa.Column('title', sa.String(length=1000), nullable=False),
        sa.Column('duration', sa.Integer(), nullable=True),
        sa.Column('url', sa.String(length=2000), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('video_id')
    )
    
    # Create indexes for videos table
    op.create_index('idx_video_id', 'videos', ['video_id'], unique=False)
    op.create_index('idx_created_at', 'videos', ['created_at'], unique=False)
    
    # Create transcripts table
    op.create_table(
        'transcripts',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('video_id', sa.Integer(), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('language', sa.String(length=10), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['video_id'], ['videos.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indexes for transcripts table
    op.create_index('idx_transcript_video_id', 'transcripts', ['video_id'], unique=False)
    op.create_index('idx_transcript_language', 'transcripts', ['language'], unique=False)
    
    # Create summaries table
    op.create_table(
        'summaries',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('video_id', sa.Integer(), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('processing_time', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['video_id'], ['videos.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indexes for summaries table
    op.create_index('idx_summary_video_id', 'summaries', ['video_id'], unique=False)
    op.create_index('idx_summary_created_at', 'summaries', ['created_at'], unique=False)
    
    # Create keywords table
    op.create_table(
        'keywords',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('video_id', sa.Integer(), nullable=False),
        sa.Column('keywords_json', sa.JSON(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['video_id'], ['videos.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indexes for keywords table
    op.create_index('idx_keyword_video_id', 'keywords', ['video_id'], unique=False)
    
    # Create timestamped_segments table
    op.create_table(
        'timestamped_segments',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('video_id', sa.Integer(), nullable=False),
        sa.Column('segments_json', sa.JSON(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['video_id'], ['videos.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indexes for timestamped_segments table
    op.create_index('idx_timestamped_segment_video_id', 'timestamped_segments', ['video_id'], unique=False)
    
    # Create processing_metadata table
    op.create_table(
        'processing_metadata',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('video_id', sa.Integer(), nullable=False),
        sa.Column('workflow_params', sa.JSON(), nullable=True),
        sa.Column('status', sa.String(length=50), nullable=False),
        sa.Column('error_info', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['video_id'], ['videos.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indexes for processing_metadata table
    op.create_index('idx_processing_metadata_video_id', 'processing_metadata', ['video_id'], unique=False)
    op.create_index('idx_processing_metadata_status', 'processing_metadata', ['status'], unique=False)
    op.create_index('idx_processing_metadata_created_at', 'processing_metadata', ['created_at'], unique=False)


def downgrade() -> None:
    # Drop tables in reverse order due to foreign key constraints
    op.drop_table('processing_metadata')
    op.drop_table('timestamped_segments')
    op.drop_table('keywords')
    op.drop_table('summaries')
    op.drop_table('transcripts')
    op.drop_table('videos')