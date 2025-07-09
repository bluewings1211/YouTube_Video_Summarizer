"""Add batch processing tables

Revision ID: 003_add_batch_processing
Revises: 002_enhance_cascade_delete
Create Date: 2025-01-09 07:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '003_add_batch_processing'
down_revision = '002_enhance_cascade_delete'
branch_labels = None
depends_on = None


def upgrade():
    """Add batch processing tables."""
    
    # Create batch status enum
    batch_status_enum = postgresql.ENUM(
        'PENDING', 'PROCESSING', 'COMPLETED', 'FAILED', 'CANCELLED',
        name='batchstatus'
    )
    batch_status_enum.create(op.get_bind())
    
    # Create batch item status enum
    batch_item_status_enum = postgresql.ENUM(
        'QUEUED', 'PROCESSING', 'METADATA_PROCESSING', 'SUMMARIZING',
        'KEYWORD_EXTRACTION', 'COMPLETED', 'FAILED', 'CANCELLED',
        name='batchitemstatus'
    )
    batch_item_status_enum.create(op.get_bind())
    
    # Create batch priority enum
    batch_priority_enum = postgresql.ENUM(
        'LOW', 'NORMAL', 'HIGH', 'URGENT',
        name='batchpriority'
    )
    batch_priority_enum.create(op.get_bind())
    
    # Create batches table
    op.create_table(
        'batches',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('batch_id', sa.String(length=255), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('status', batch_status_enum, nullable=False),
        sa.Column('priority', batch_priority_enum, nullable=False),
        sa.Column('total_items', sa.Integer(), nullable=False),
        sa.Column('completed_items', sa.Integer(), nullable=False),
        sa.Column('failed_items', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('webhook_url', sa.String(length=2000), nullable=True),
        sa.Column('batch_metadata', sa.JSON(), nullable=True),
        sa.Column('error_info', sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('batch_id')
    )
    
    # Create batch_items table
    op.create_table(
        'batch_items',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('batch_id', sa.Integer(), nullable=False),
        sa.Column('video_id', sa.Integer(), nullable=True),
        sa.Column('url', sa.String(length=2000), nullable=False),
        sa.Column('status', batch_item_status_enum, nullable=False),
        sa.Column('priority', batch_priority_enum, nullable=False),
        sa.Column('processing_order', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('retry_count', sa.Integer(), nullable=False),
        sa.Column('max_retries', sa.Integer(), nullable=False),
        sa.Column('error_info', sa.Text(), nullable=True),
        sa.Column('processing_data', sa.JSON(), nullable=True),
        sa.Column('result_data', sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(['batch_id'], ['batches.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['video_id'], ['videos.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create queue_items table
    op.create_table(
        'queue_items',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('batch_item_id', sa.Integer(), nullable=False),
        sa.Column('queue_name', sa.String(length=255), nullable=False),
        sa.Column('priority', batch_priority_enum, nullable=False),
        sa.Column('scheduled_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('locked_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('locked_by', sa.String(length=255), nullable=True),
        sa.Column('lock_expires_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('retry_count', sa.Integer(), nullable=False),
        sa.Column('max_retries', sa.Integer(), nullable=False),
        sa.Column('error_info', sa.Text(), nullable=True),
        sa.Column('queue_metadata', sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(['batch_item_id'], ['batch_items.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create processing_sessions table
    op.create_table(
        'processing_sessions',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('session_id', sa.String(length=255), nullable=False),
        sa.Column('batch_item_id', sa.Integer(), nullable=False),
        sa.Column('worker_id', sa.String(length=255), nullable=False),
        sa.Column('started_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('heartbeat_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('status', batch_item_status_enum, nullable=False),
        sa.Column('progress_percentage', sa.Float(), nullable=False),
        sa.Column('current_step', sa.String(length=255), nullable=True),
        sa.Column('session_metadata', sa.JSON(), nullable=True),
        sa.Column('error_info', sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(['batch_item_id'], ['batch_items.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('session_id')
    )
    
    # Create indexes for batches table
    op.create_index('idx_batch_id', 'batches', ['batch_id'])
    op.create_index('idx_batch_status', 'batches', ['status'])
    op.create_index('idx_batch_priority', 'batches', ['priority'])
    op.create_index('idx_batch_created_at', 'batches', ['created_at'])
    op.create_index('idx_batch_status_priority', 'batches', ['status', 'priority'])
    
    # Create indexes for batch_items table
    op.create_index('idx_batch_item_batch_id', 'batch_items', ['batch_id'])
    op.create_index('idx_batch_item_video_id', 'batch_items', ['video_id'])
    op.create_index('idx_batch_item_status', 'batch_items', ['status'])
    op.create_index('idx_batch_item_priority', 'batch_items', ['priority'])
    op.create_index('idx_batch_item_processing_order', 'batch_items', ['processing_order'])
    op.create_index('idx_batch_item_created_at', 'batch_items', ['created_at'])
    op.create_index('idx_batch_item_status_priority', 'batch_items', ['status', 'priority'])
    op.create_index('idx_batch_item_batch_status', 'batch_items', ['batch_id', 'status'])
    
    # Create indexes for queue_items table
    op.create_index('idx_queue_item_batch_item_id', 'queue_items', ['batch_item_id'])
    op.create_index('idx_queue_item_queue_name', 'queue_items', ['queue_name'])
    op.create_index('idx_queue_item_priority', 'queue_items', ['priority'])
    op.create_index('idx_queue_item_scheduled_at', 'queue_items', ['scheduled_at'])
    op.create_index('idx_queue_item_locked_at', 'queue_items', ['locked_at'])
    op.create_index('idx_queue_item_locked_by', 'queue_items', ['locked_by'])
    op.create_index('idx_queue_item_lock_expires_at', 'queue_items', ['lock_expires_at'])
    op.create_index('idx_queue_item_queue_priority', 'queue_items', ['queue_name', 'priority'])
    op.create_index('idx_queue_item_queue_scheduled', 'queue_items', ['queue_name', 'scheduled_at'])
    op.create_index('idx_queue_item_available', 'queue_items', ['queue_name', 'scheduled_at', 'locked_at'])
    
    # Create indexes for processing_sessions table
    op.create_index('idx_processing_session_session_id', 'processing_sessions', ['session_id'])
    op.create_index('idx_processing_session_batch_item_id', 'processing_sessions', ['batch_item_id'])
    op.create_index('idx_processing_session_worker_id', 'processing_sessions', ['worker_id'])
    op.create_index('idx_processing_session_heartbeat_at', 'processing_sessions', ['heartbeat_at'])
    op.create_index('idx_processing_session_status', 'processing_sessions', ['status'])
    op.create_index('idx_processing_session_started_at', 'processing_sessions', ['started_at'])


def downgrade():
    """Remove batch processing tables."""
    
    # Drop indexes
    op.drop_index('idx_processing_session_started_at', table_name='processing_sessions')
    op.drop_index('idx_processing_session_status', table_name='processing_sessions')
    op.drop_index('idx_processing_session_heartbeat_at', table_name='processing_sessions')
    op.drop_index('idx_processing_session_worker_id', table_name='processing_sessions')
    op.drop_index('idx_processing_session_batch_item_id', table_name='processing_sessions')
    op.drop_index('idx_processing_session_session_id', table_name='processing_sessions')
    
    op.drop_index('idx_queue_item_available', table_name='queue_items')
    op.drop_index('idx_queue_item_queue_scheduled', table_name='queue_items')
    op.drop_index('idx_queue_item_queue_priority', table_name='queue_items')
    op.drop_index('idx_queue_item_lock_expires_at', table_name='queue_items')
    op.drop_index('idx_queue_item_locked_by', table_name='queue_items')
    op.drop_index('idx_queue_item_locked_at', table_name='queue_items')
    op.drop_index('idx_queue_item_scheduled_at', table_name='queue_items')
    op.drop_index('idx_queue_item_priority', table_name='queue_items')
    op.drop_index('idx_queue_item_queue_name', table_name='queue_items')
    op.drop_index('idx_queue_item_batch_item_id', table_name='queue_items')
    
    op.drop_index('idx_batch_item_batch_status', table_name='batch_items')
    op.drop_index('idx_batch_item_status_priority', table_name='batch_items')
    op.drop_index('idx_batch_item_created_at', table_name='batch_items')
    op.drop_index('idx_batch_item_processing_order', table_name='batch_items')
    op.drop_index('idx_batch_item_priority', table_name='batch_items')
    op.drop_index('idx_batch_item_status', table_name='batch_items')
    op.drop_index('idx_batch_item_video_id', table_name='batch_items')
    op.drop_index('idx_batch_item_batch_id', table_name='batch_items')
    
    op.drop_index('idx_batch_status_priority', table_name='batches')
    op.drop_index('idx_batch_created_at', table_name='batches')
    op.drop_index('idx_batch_priority', table_name='batches')
    op.drop_index('idx_batch_status', table_name='batches')
    op.drop_index('idx_batch_id', table_name='batches')
    
    # Drop tables
    op.drop_table('processing_sessions')
    op.drop_table('queue_items')
    op.drop_table('batch_items')
    op.drop_table('batches')
    
    # Drop enums
    batch_priority_enum = postgresql.ENUM(name='batchpriority')
    batch_priority_enum.drop(op.get_bind())
    
    batch_item_status_enum = postgresql.ENUM(name='batchitemstatus')
    batch_item_status_enum.drop(op.get_bind())
    
    batch_status_enum = postgresql.ENUM(name='batchstatus')
    batch_status_enum.drop(op.get_bind())