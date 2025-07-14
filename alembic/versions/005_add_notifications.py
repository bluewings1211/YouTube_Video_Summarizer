"""Add notification and webhook tables

Revision ID: 005_add_notifications
Revises: 004_add_status_tracking
Create Date: 2025-07-10 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '005_add_notifications'
down_revision = '004_add_status_tracking'
branch_labels = None
depends_on = None


def upgrade():
    """Add notification and webhook tables."""
    
    # Create notification type enum
    notification_type_enum = postgresql.ENUM(
        'EMAIL', 'WEBHOOK', 'SMS', 'SLACK', 'DISCORD', 'TEAMS', 'CUSTOM',
        name='notificationtype'
    )
    notification_type_enum.create(op.get_bind())
    
    # Create notification event enum
    notification_event_enum = postgresql.ENUM(
        'PROCESSING_STARTED', 'PROCESSING_COMPLETED', 'PROCESSING_FAILED',
        'PROCESSING_CANCELLED', 'BATCH_STARTED', 'BATCH_COMPLETED',
        'BATCH_FAILED', 'BATCH_PROGRESS_UPDATE', 'VIDEO_PROCESSED',
        'ERROR_OCCURRED', 'RETRY_EXHAUSTED', 'SYSTEM_MAINTENANCE',
        'CUSTOM_EVENT',
        name='notificationevent'
    )
    notification_event_enum.create(op.get_bind())
    
    # Create notification status enum
    notification_status_enum = postgresql.ENUM(
        'PENDING', 'SENDING', 'SENT', 'DELIVERED', 'FAILED',
        'RETRY_PENDING', 'CANCELLED',
        name='notificationstatus'
    )
    notification_status_enum.create(op.get_bind())
    
    # Create notification priority enum
    notification_priority_enum = postgresql.ENUM(
        'LOW', 'NORMAL', 'HIGH', 'URGENT',
        name='notificationpriority'
    )
    notification_priority_enum.create(op.get_bind())
    
    # Create notification_configs table
    op.create_table(
        'notification_configs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('config_id', sa.String(length=255), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('user_id', sa.String(length=255), nullable=True),
        sa.Column('notification_type', notification_type_enum, nullable=False),
        sa.Column('event_triggers', sa.JSON(), nullable=False),
        sa.Column('target_address', sa.String(length=2000), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False),
        sa.Column('priority', notification_priority_enum, nullable=False),
        sa.Column('rate_limit_per_hour', sa.Integer(), nullable=True),
        sa.Column('rate_limit_per_day', sa.Integer(), nullable=True),
        sa.Column('template_config', sa.JSON(), nullable=True),
        sa.Column('filter_conditions', sa.JSON(), nullable=True),
        sa.Column('retry_config', sa.JSON(), nullable=True),
        sa.Column('auth_config', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('last_triggered_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('trigger_count_today', sa.Integer(), nullable=False),
        sa.Column('trigger_count_total', sa.Integer(), nullable=False),
        sa.Column('metadata', sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('config_id', name='uq_notification_config_config_id')
    )
    
    # Create notifications table
    op.create_table(
        'notifications',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('notification_id', sa.String(length=255), nullable=False),
        sa.Column('config_id', sa.Integer(), nullable=False),
        sa.Column('event_type', notification_event_enum, nullable=False),
        sa.Column('event_source', sa.String(length=255), nullable=True),
        sa.Column('event_metadata', sa.JSON(), nullable=True),
        sa.Column('status', notification_status_enum, nullable=False),
        sa.Column('priority', notification_priority_enum, nullable=False),
        sa.Column('target_address', sa.String(length=2000), nullable=False),
        sa.Column('subject', sa.String(length=1000), nullable=True),
        sa.Column('message', sa.Text(), nullable=False),
        sa.Column('payload', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('scheduled_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('sent_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('delivered_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('retry_count', sa.Integer(), nullable=False),
        sa.Column('max_retries', sa.Integer(), nullable=False),
        sa.Column('retry_schedule', sa.JSON(), nullable=True),
        sa.Column('error_info', sa.Text(), nullable=True),
        sa.Column('delivery_metadata', sa.JSON(), nullable=True),
        sa.Column('webhook_response', sa.JSON(), nullable=True),
        sa.Column('external_id', sa.String(length=255), nullable=True),
        sa.ForeignKeyConstraint(['config_id'], ['notification_configs.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('notification_id', name='uq_notification_notification_id')
    )
    
    # Create notification_logs table
    op.create_table(
        'notification_logs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('notification_id', sa.Integer(), nullable=False),
        sa.Column('log_level', sa.String(length=20), nullable=False),
        sa.Column('message', sa.Text(), nullable=False),
        sa.Column('log_data', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('worker_id', sa.String(length=255), nullable=True),
        sa.Column('execution_context', sa.JSON(), nullable=True),
        sa.Column('stack_trace', sa.Text(), nullable=True),
        sa.Column('external_references', sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(['notification_id'], ['notifications.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create webhook_endpoints table
    op.create_table(
        'webhook_endpoints',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('endpoint_id', sa.String(length=255), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('url', sa.String(length=2000), nullable=False),
        sa.Column('http_method', sa.String(length=10), nullable=False),
        sa.Column('auth_type', sa.String(length=50), nullable=False),
        sa.Column('auth_config', sa.JSON(), nullable=True),
        sa.Column('headers', sa.JSON(), nullable=True),
        sa.Column('timeout_seconds', sa.Integer(), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False),
        sa.Column('health_check_enabled', sa.Boolean(), nullable=False),
        sa.Column('health_check_url', sa.String(length=2000), nullable=True),
        sa.Column('health_check_interval_minutes', sa.Integer(), nullable=True),
        sa.Column('last_health_check_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('health_status', sa.String(length=50), nullable=True),
        sa.Column('retry_config', sa.JSON(), nullable=True),
        sa.Column('rate_limit_config', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('last_used_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('success_count', sa.Integer(), nullable=False),
        sa.Column('failure_count', sa.Integer(), nullable=False),
        sa.Column('average_response_time_ms', sa.Float(), nullable=True),
        sa.Column('metadata', sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('endpoint_id', name='uq_webhook_endpoint_endpoint_id')
    )
    
    # Create indexes for notification_configs table
    op.create_index('idx_notification_config_config_id', 'notification_configs', ['config_id'])
    op.create_index('idx_notification_config_user_id', 'notification_configs', ['user_id'])
    op.create_index('idx_notification_config_notification_type', 'notification_configs', ['notification_type'])
    op.create_index('idx_notification_config_is_active', 'notification_configs', ['is_active'])
    op.create_index('idx_notification_config_priority', 'notification_configs', ['priority'])
    op.create_index('idx_notification_config_created_at', 'notification_configs', ['created_at'])
    op.create_index('idx_notification_config_last_triggered_at', 'notification_configs', ['last_triggered_at'])
    op.create_index('idx_notification_config_user_active', 'notification_configs', ['user_id', 'is_active'])
    op.create_index('idx_notification_config_type_active', 'notification_configs', ['notification_type', 'is_active'])
    
    # Create indexes for notifications table
    op.create_index('idx_notification_notification_id', 'notifications', ['notification_id'])
    op.create_index('idx_notification_config_id', 'notifications', ['config_id'])
    op.create_index('idx_notification_event_type', 'notifications', ['event_type'])
    op.create_index('idx_notification_event_source', 'notifications', ['event_source'])
    op.create_index('idx_notification_status', 'notifications', ['status'])
    op.create_index('idx_notification_priority', 'notifications', ['priority'])
    op.create_index('idx_notification_created_at', 'notifications', ['created_at'])
    op.create_index('idx_notification_scheduled_at', 'notifications', ['scheduled_at'])
    op.create_index('idx_notification_sent_at', 'notifications', ['sent_at'])
    op.create_index('idx_notification_external_id', 'notifications', ['external_id'])
    op.create_index('idx_notification_status_scheduled', 'notifications', ['status', 'scheduled_at'])
    op.create_index('idx_notification_config_event', 'notifications', ['config_id', 'event_type'])
    op.create_index('idx_notification_source_event', 'notifications', ['event_source', 'event_type'])
    
    # Create indexes for notification_logs table
    op.create_index('idx_notification_log_notification_id', 'notification_logs', ['notification_id'])
    op.create_index('idx_notification_log_log_level', 'notification_logs', ['log_level'])
    op.create_index('idx_notification_log_created_at', 'notification_logs', ['created_at'])
    op.create_index('idx_notification_log_worker_id', 'notification_logs', ['worker_id'])
    op.create_index('idx_notification_log_notification_created', 'notification_logs', ['notification_id', 'created_at'])
    op.create_index('idx_notification_log_level_created', 'notification_logs', ['log_level', 'created_at'])
    
    # Create indexes for webhook_endpoints table
    op.create_index('idx_webhook_endpoint_endpoint_id', 'webhook_endpoints', ['endpoint_id'])
    op.create_index('idx_webhook_endpoint_is_active', 'webhook_endpoints', ['is_active'])
    op.create_index('idx_webhook_endpoint_health_status', 'webhook_endpoints', ['health_status'])
    op.create_index('idx_webhook_endpoint_created_at', 'webhook_endpoints', ['created_at'])
    op.create_index('idx_webhook_endpoint_last_used_at', 'webhook_endpoints', ['last_used_at'])
    op.create_index('idx_webhook_endpoint_last_health_check_at', 'webhook_endpoints', ['last_health_check_at'])
    op.create_index('idx_webhook_endpoint_active_health', 'webhook_endpoints', ['is_active', 'health_status'])


def downgrade():
    """Remove notification and webhook tables."""
    
    # Drop indexes for webhook_endpoints table
    op.drop_index('idx_webhook_endpoint_active_health', table_name='webhook_endpoints')
    op.drop_index('idx_webhook_endpoint_last_health_check_at', table_name='webhook_endpoints')
    op.drop_index('idx_webhook_endpoint_last_used_at', table_name='webhook_endpoints')
    op.drop_index('idx_webhook_endpoint_created_at', table_name='webhook_endpoints')
    op.drop_index('idx_webhook_endpoint_health_status', table_name='webhook_endpoints')
    op.drop_index('idx_webhook_endpoint_is_active', table_name='webhook_endpoints')
    op.drop_index('idx_webhook_endpoint_endpoint_id', table_name='webhook_endpoints')
    
    # Drop indexes for notification_logs table
    op.drop_index('idx_notification_log_level_created', table_name='notification_logs')
    op.drop_index('idx_notification_log_notification_created', table_name='notification_logs')
    op.drop_index('idx_notification_log_worker_id', table_name='notification_logs')
    op.drop_index('idx_notification_log_created_at', table_name='notification_logs')
    op.drop_index('idx_notification_log_log_level', table_name='notification_logs')
    op.drop_index('idx_notification_log_notification_id', table_name='notification_logs')
    
    # Drop indexes for notifications table
    op.drop_index('idx_notification_source_event', table_name='notifications')
    op.drop_index('idx_notification_config_event', table_name='notifications')
    op.drop_index('idx_notification_status_scheduled', table_name='notifications')
    op.drop_index('idx_notification_external_id', table_name='notifications')
    op.drop_index('idx_notification_sent_at', table_name='notifications')
    op.drop_index('idx_notification_scheduled_at', table_name='notifications')
    op.drop_index('idx_notification_created_at', table_name='notifications')
    op.drop_index('idx_notification_priority', table_name='notifications')
    op.drop_index('idx_notification_status', table_name='notifications')
    op.drop_index('idx_notification_event_source', table_name='notifications')
    op.drop_index('idx_notification_event_type', table_name='notifications')
    op.drop_index('idx_notification_config_id', table_name='notifications')
    op.drop_index('idx_notification_notification_id', table_name='notifications')
    
    # Drop indexes for notification_configs table
    op.drop_index('idx_notification_config_type_active', table_name='notification_configs')
    op.drop_index('idx_notification_config_user_active', table_name='notification_configs')
    op.drop_index('idx_notification_config_last_triggered_at', table_name='notification_configs')
    op.drop_index('idx_notification_config_created_at', table_name='notification_configs')
    op.drop_index('idx_notification_config_priority', table_name='notification_configs')
    op.drop_index('idx_notification_config_is_active', table_name='notification_configs')
    op.drop_index('idx_notification_config_notification_type', table_name='notification_configs')
    op.drop_index('idx_notification_config_user_id', table_name='notification_configs')
    op.drop_index('idx_notification_config_config_id', table_name='notification_configs')
    
    # Drop tables
    op.drop_table('webhook_endpoints')
    op.drop_table('notification_logs')
    op.drop_table('notifications')
    op.drop_table('notification_configs')
    
    # Drop enums
    notification_priority_enum = postgresql.ENUM(name='notificationpriority')
    notification_priority_enum.drop(op.get_bind())
    
    notification_status_enum = postgresql.ENUM(name='notificationstatus')
    notification_status_enum.drop(op.get_bind())
    
    notification_event_enum = postgresql.ENUM(name='notificationevent')
    notification_event_enum.drop(op.get_bind())
    
    notification_type_enum = postgresql.ENUM(name='notificationtype')
    notification_type_enum.drop(op.get_bind())