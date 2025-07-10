"""Add status tracking tables

Revision ID: 004_add_status_tracking
Revises: 003_add_batch_processing
Create Date: 2025-01-10 08:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '004_add_status_tracking'
down_revision = '003_add_batch_processing'
branch_labels = None
depends_on = None


def upgrade():
    """Add status tracking tables."""
    
    # Create processing status type enum
    processing_status_type_enum = postgresql.ENUM(
        'QUEUED', 'STARTING', 'YOUTUBE_METADATA', 'TRANSCRIPT_EXTRACTION',
        'LANGUAGE_DETECTION', 'SUMMARY_GENERATION', 'KEYWORD_EXTRACTION',
        'TIMESTAMPED_SEGMENTS', 'FINALIZING', 'COMPLETED', 'FAILED',
        'CANCELLED', 'RETRY_PENDING',
        name='processingstatustype'
    )
    processing_status_type_enum.create(op.get_bind())
    
    # Create processing priority enum (reuse batch priority if needed)
    processing_priority_enum = postgresql.ENUM(
        'LOW', 'NORMAL', 'HIGH', 'URGENT',
        name='processingpriority'
    )
    processing_priority_enum.create(op.get_bind())
    
    # Create status change type enum
    status_change_type_enum = postgresql.ENUM(
        'STATUS_UPDATE', 'PROGRESS_UPDATE', 'ERROR_OCCURRED',
        'RETRY_SCHEDULED', 'MANUAL_INTERVENTION', 'SYSTEM_EVENT',
        name='statuschangetype'
    )
    status_change_type_enum.create(op.get_bind())
    
    # Create processing_status table
    op.create_table(
        'processing_status',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('status_id', sa.String(length=255), nullable=False),
        sa.Column('video_id', sa.Integer(), nullable=True),
        sa.Column('batch_item_id', sa.Integer(), nullable=True),
        sa.Column('processing_session_id', sa.Integer(), nullable=True),
        sa.Column('status', processing_status_type_enum, nullable=False),
        sa.Column('substatus', sa.String(length=255), nullable=True),
        sa.Column('priority', processing_priority_enum, nullable=False),
        sa.Column('progress_percentage', sa.Float(), nullable=False),
        sa.Column('current_step', sa.String(length=255), nullable=True),
        sa.Column('total_steps', sa.Integer(), nullable=True),
        sa.Column('completed_steps', sa.Integer(), nullable=False),
        sa.Column('estimated_completion_time', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('worker_id', sa.String(length=255), nullable=True),
        sa.Column('heartbeat_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('retry_count', sa.Integer(), nullable=False),
        sa.Column('max_retries', sa.Integer(), nullable=False),
        sa.Column('processing_metadata', sa.JSON(), nullable=True),
        sa.Column('result_metadata', sa.JSON(), nullable=True),
        sa.Column('error_info', sa.Text(), nullable=True),
        sa.Column('tags', sa.JSON(), nullable=True),
        sa.Column('external_id', sa.String(length=255), nullable=True),
        sa.ForeignKeyConstraint(['video_id'], ['videos.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['batch_item_id'], ['batch_items.id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['processing_session_id'], ['processing_sessions.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('status_id', name='uq_processing_status_status_id')
    )
    
    # Create status_history table
    op.create_table(
        'status_history',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('processing_status_id', sa.Integer(), nullable=False),
        sa.Column('change_type', status_change_type_enum, nullable=False),
        sa.Column('previous_status', processing_status_type_enum, nullable=True),
        sa.Column('new_status', processing_status_type_enum, nullable=False),
        sa.Column('previous_progress', sa.Float(), nullable=True),
        sa.Column('new_progress', sa.Float(), nullable=False),
        sa.Column('change_reason', sa.String(length=500), nullable=True),
        sa.Column('change_metadata', sa.JSON(), nullable=True),
        sa.Column('worker_id', sa.String(length=255), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('duration_seconds', sa.Integer(), nullable=True),
        sa.Column('error_info', sa.Text(), nullable=True),
        sa.Column('external_trigger', sa.String(length=255), nullable=True),
        sa.ForeignKeyConstraint(['processing_status_id'], ['processing_status.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create status_metrics table
    op.create_table(
        'status_metrics',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('metric_date', sa.DateTime(timezone=True), nullable=False),
        sa.Column('metric_hour', sa.Integer(), nullable=True),
        sa.Column('total_items', sa.Integer(), nullable=False),
        sa.Column('completed_items', sa.Integer(), nullable=False),
        sa.Column('failed_items', sa.Integer(), nullable=False),
        sa.Column('cancelled_items', sa.Integer(), nullable=False),
        sa.Column('average_processing_time_seconds', sa.Float(), nullable=True),
        sa.Column('median_processing_time_seconds', sa.Float(), nullable=True),
        sa.Column('max_processing_time_seconds', sa.Float(), nullable=True),
        sa.Column('min_processing_time_seconds', sa.Float(), nullable=True),
        sa.Column('retry_rate_percentage', sa.Float(), nullable=True),
        sa.Column('success_rate_percentage', sa.Float(), nullable=True),
        sa.Column('queue_wait_time_seconds', sa.Float(), nullable=True),
        sa.Column('worker_utilization_percentage', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('metrics_metadata', sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('metric_date', 'metric_hour', name='uq_status_metrics_date_hour')
    )
    
    # Create indexes for processing_status table
    op.create_index('idx_processing_status_status_id', 'processing_status', ['status_id'])
    op.create_index('idx_processing_status_video_id', 'processing_status', ['video_id'])
    op.create_index('idx_processing_status_batch_item_id', 'processing_status', ['batch_item_id'])
    op.create_index('idx_processing_status_processing_session_id', 'processing_status', ['processing_session_id'])
    op.create_index('idx_processing_status_status', 'processing_status', ['status'])
    op.create_index('idx_processing_status_priority', 'processing_status', ['priority'])
    op.create_index('idx_processing_status_created_at', 'processing_status', ['created_at'])
    op.create_index('idx_processing_status_updated_at', 'processing_status', ['updated_at'])
    op.create_index('idx_processing_status_heartbeat_at', 'processing_status', ['heartbeat_at'])
    op.create_index('idx_processing_status_worker_id', 'processing_status', ['worker_id'])
    op.create_index('idx_processing_status_external_id', 'processing_status', ['external_id'])
    op.create_index('idx_processing_status_status_priority', 'processing_status', ['status', 'priority'])
    op.create_index('idx_processing_status_status_updated', 'processing_status', ['status', 'updated_at'])
    op.create_index('idx_processing_status_progress', 'processing_status', ['progress_percentage'])
    
    # Create indexes for status_history table
    op.create_index('idx_status_history_processing_status_id', 'status_history', ['processing_status_id'])
    op.create_index('idx_status_history_change_type', 'status_history', ['change_type'])
    op.create_index('idx_status_history_new_status', 'status_history', ['new_status'])
    op.create_index('idx_status_history_created_at', 'status_history', ['created_at'])
    op.create_index('idx_status_history_worker_id', 'status_history', ['worker_id'])
    op.create_index('idx_status_history_external_trigger', 'status_history', ['external_trigger'])
    op.create_index('idx_status_history_status_created', 'status_history', ['processing_status_id', 'created_at'])
    op.create_index('idx_status_history_status_type', 'status_history', ['new_status', 'change_type'])
    
    # Create indexes for status_metrics table
    op.create_index('idx_status_metrics_metric_date', 'status_metrics', ['metric_date'])
    op.create_index('idx_status_metrics_metric_hour', 'status_metrics', ['metric_hour'])
    op.create_index('idx_status_metrics_date_hour', 'status_metrics', ['metric_date', 'metric_hour'])
    op.create_index('idx_status_metrics_success_rate', 'status_metrics', ['success_rate_percentage'])
    op.create_index('idx_status_metrics_total_items', 'status_metrics', ['total_items'])


def downgrade():
    """Remove status tracking tables."""
    
    # Drop indexes for status_metrics table
    op.drop_index('idx_status_metrics_total_items', table_name='status_metrics')
    op.drop_index('idx_status_metrics_success_rate', table_name='status_metrics')
    op.drop_index('idx_status_metrics_date_hour', table_name='status_metrics')
    op.drop_index('idx_status_metrics_metric_hour', table_name='status_metrics')
    op.drop_index('idx_status_metrics_metric_date', table_name='status_metrics')
    
    # Drop indexes for status_history table
    op.drop_index('idx_status_history_status_type', table_name='status_history')
    op.drop_index('idx_status_history_status_created', table_name='status_history')
    op.drop_index('idx_status_history_external_trigger', table_name='status_history')
    op.drop_index('idx_status_history_worker_id', table_name='status_history')
    op.drop_index('idx_status_history_created_at', table_name='status_history')
    op.drop_index('idx_status_history_new_status', table_name='status_history')
    op.drop_index('idx_status_history_change_type', table_name='status_history')
    op.drop_index('idx_status_history_processing_status_id', table_name='status_history')
    
    # Drop indexes for processing_status table
    op.drop_index('idx_processing_status_progress', table_name='processing_status')
    op.drop_index('idx_processing_status_status_updated', table_name='processing_status')
    op.drop_index('idx_processing_status_status_priority', table_name='processing_status')
    op.drop_index('idx_processing_status_external_id', table_name='processing_status')
    op.drop_index('idx_processing_status_worker_id', table_name='processing_status')
    op.drop_index('idx_processing_status_heartbeat_at', table_name='processing_status')
    op.drop_index('idx_processing_status_updated_at', table_name='processing_status')
    op.drop_index('idx_processing_status_created_at', table_name='processing_status')
    op.drop_index('idx_processing_status_priority', table_name='processing_status')
    op.drop_index('idx_processing_status_status', table_name='processing_status')
    op.drop_index('idx_processing_status_processing_session_id', table_name='processing_status')
    op.drop_index('idx_processing_status_batch_item_id', table_name='processing_status')
    op.drop_index('idx_processing_status_video_id', table_name='processing_status')
    op.drop_index('idx_processing_status_status_id', table_name='processing_status')
    
    # Drop tables
    op.drop_table('status_metrics')
    op.drop_table('status_history')
    op.drop_table('processing_status')
    
    # Drop enums
    status_change_type_enum = postgresql.ENUM(name='statuschangetype')
    status_change_type_enum.drop(op.get_bind())
    
    processing_priority_enum = postgresql.ENUM(name='processingpriority')
    processing_priority_enum.drop(op.get_bind())
    
    processing_status_type_enum = postgresql.ENUM(name='processingstatustype')
    processing_status_type_enum.drop(op.get_bind())