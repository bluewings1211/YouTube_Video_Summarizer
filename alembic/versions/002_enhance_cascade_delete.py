"""Enhance cascade delete functionality

Revision ID: 002
Revises: 001
Create Date: 2025-01-09 06:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '002'
down_revision: Union[str, None] = '001'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """
    Enhance cascade delete functionality with additional constraints and triggers.
    """
    
    # Add audit trigger function for deletion logging
    op.execute("""
        CREATE OR REPLACE FUNCTION log_video_deletion() RETURNS TRIGGER AS $$
        BEGIN
            INSERT INTO deletion_audit_log (
                video_id, 
                video_youtube_id, 
                video_title, 
                deleted_at, 
                deletion_type
            ) VALUES (
                OLD.id,
                OLD.video_id,
                OLD.title,
                NOW(),
                'cascade_delete'
            );
            RETURN OLD;
        END;
        $$ LANGUAGE plpgsql;
    """)
    
    # Create deletion audit log table
    op.create_table(
        'deletion_audit_log',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('video_id', sa.Integer(), nullable=False),
        sa.Column('video_youtube_id', sa.String(length=255), nullable=False),
        sa.Column('video_title', sa.String(length=1000), nullable=False),
        sa.Column('deleted_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('deletion_type', sa.String(length=50), nullable=False),
        sa.Column('deleted_by', sa.String(length=255), nullable=True),
        sa.Column('related_records_count', sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indexes for deletion audit log
    op.create_index('idx_deletion_audit_video_id', 'deletion_audit_log', ['video_id'])
    op.create_index('idx_deletion_audit_deleted_at', 'deletion_audit_log', ['deleted_at'])
    op.create_index('idx_deletion_audit_deletion_type', 'deletion_audit_log', ['deletion_type'])
    
    # Add trigger to videos table for deletion logging
    op.execute("""
        CREATE TRIGGER video_deletion_audit_trigger
        BEFORE DELETE ON videos
        FOR EACH ROW
        EXECUTE FUNCTION log_video_deletion();
    """)
    
    # Add check constraint to ensure video_id format is valid
    op.execute("""
        ALTER TABLE videos 
        ADD CONSTRAINT check_video_id_format 
        CHECK (LENGTH(video_id) = 11 AND video_id ~ '^[A-Za-z0-9_-]+$');
    """)
    
    # Add check constraint to ensure processing status is valid
    op.execute("""
        ALTER TABLE processing_metadata 
        ADD CONSTRAINT check_processing_status 
        CHECK (status IN ('pending', 'processing', 'completed', 'failed', 'cancelled'));
    """)
    
    # Create function to get cascade delete impact
    op.execute("""
        CREATE OR REPLACE FUNCTION get_cascade_delete_impact(target_video_id INTEGER)
        RETURNS JSON AS $$
        DECLARE
            result JSON;
        BEGIN
            SELECT json_build_object(
                'video_id', target_video_id,
                'related_counts', json_build_object(
                    'transcripts', (SELECT COUNT(*) FROM transcripts WHERE video_id = target_video_id),
                    'summaries', (SELECT COUNT(*) FROM summaries WHERE video_id = target_video_id),
                    'keywords', (SELECT COUNT(*) FROM keywords WHERE video_id = target_video_id),
                    'timestamped_segments', (SELECT COUNT(*) FROM timestamped_segments WHERE video_id = target_video_id),
                    'processing_metadata', (SELECT COUNT(*) FROM processing_metadata WHERE video_id = target_video_id)
                ),
                'total_related_records', (
                    (SELECT COUNT(*) FROM transcripts WHERE video_id = target_video_id) +
                    (SELECT COUNT(*) FROM summaries WHERE video_id = target_video_id) +
                    (SELECT COUNT(*) FROM keywords WHERE video_id = target_video_id) +
                    (SELECT COUNT(*) FROM timestamped_segments WHERE video_id = target_video_id) +
                    (SELECT COUNT(*) FROM processing_metadata WHERE video_id = target_video_id)
                ),
                'video_exists', EXISTS(SELECT 1 FROM videos WHERE id = target_video_id)
            ) INTO result;
            
            RETURN result;
        END;
        $$ LANGUAGE plpgsql;
    """)
    
    # Create function to validate cascade delete safety
    op.execute("""
        CREATE OR REPLACE FUNCTION validate_cascade_delete_safety(target_video_id INTEGER)
        RETURNS JSON AS $$
        DECLARE
            result JSON;
            active_processing_count INTEGER;
            total_related_count INTEGER;
            issues TEXT[];
        BEGIN
            -- Check for active processing
            SELECT COUNT(*) INTO active_processing_count
            FROM processing_metadata 
            WHERE video_id = target_video_id AND status IN ('processing', 'pending');
            
            -- Get total related records
            SELECT (
                (SELECT COUNT(*) FROM transcripts WHERE video_id = target_video_id) +
                (SELECT COUNT(*) FROM summaries WHERE video_id = target_video_id) +
                (SELECT COUNT(*) FROM keywords WHERE video_id = target_video_id) +
                (SELECT COUNT(*) FROM timestamped_segments WHERE video_id = target_video_id) +
                (SELECT COUNT(*) FROM processing_metadata WHERE video_id = target_video_id)
            ) INTO total_related_count;
            
            -- Collect potential issues
            issues := ARRAY[]::TEXT[];
            
            IF active_processing_count > 0 THEN
                issues := array_append(issues, 'Active processing records exist');
            END IF;
            
            IF total_related_count > 10000 THEN
                issues := array_append(issues, 'Large number of related records');
            END IF;
            
            -- Build result
            SELECT json_build_object(
                'video_id', target_video_id,
                'can_delete_safely', array_length(issues, 1) IS NULL,
                'issues', issues,
                'active_processing_count', active_processing_count,
                'total_related_count', total_related_count,
                'video_exists', EXISTS(SELECT 1 FROM videos WHERE id = target_video_id)
            ) INTO result;
            
            RETURN result;
        END;
        $$ LANGUAGE plpgsql;
    """)
    
    # Create function to clean up orphaned records
    op.execute("""
        CREATE OR REPLACE FUNCTION cleanup_orphaned_records(target_video_id INTEGER)
        RETURNS JSON AS $$
        DECLARE
            result JSON;
            deleted_counts JSON;
        BEGIN
            -- Clean up orphaned records in the correct order
            WITH cleanup_results AS (
                SELECT 
                    'processing_metadata' as table_name,
                    (SELECT COUNT(*) FROM processing_metadata WHERE video_id = target_video_id AND NOT EXISTS (SELECT 1 FROM videos WHERE id = target_video_id)) as orphaned_count
                UNION ALL
                SELECT 
                    'timestamped_segments' as table_name,
                    (SELECT COUNT(*) FROM timestamped_segments WHERE video_id = target_video_id AND NOT EXISTS (SELECT 1 FROM videos WHERE id = target_video_id)) as orphaned_count
                UNION ALL
                SELECT 
                    'keywords' as table_name,
                    (SELECT COUNT(*) FROM keywords WHERE video_id = target_video_id AND NOT EXISTS (SELECT 1 FROM videos WHERE id = target_video_id)) as orphaned_count
                UNION ALL
                SELECT 
                    'summaries' as table_name,
                    (SELECT COUNT(*) FROM summaries WHERE video_id = target_video_id AND NOT EXISTS (SELECT 1 FROM videos WHERE id = target_video_id)) as orphaned_count
                UNION ALL
                SELECT 
                    'transcripts' as table_name,
                    (SELECT COUNT(*) FROM transcripts WHERE video_id = target_video_id AND NOT EXISTS (SELECT 1 FROM videos WHERE id = target_video_id)) as orphaned_count
            ),
            deletion_results AS (
                SELECT 
                    json_object_agg(table_name, orphaned_count) as counts
                FROM cleanup_results
                WHERE orphaned_count > 0
            )
            SELECT 
                json_build_object(
                    'video_id', target_video_id,
                    'orphaned_records_found', COALESCE(counts, '{}'::json),
                    'cleanup_timestamp', NOW()
                ) INTO result
            FROM deletion_results;
            
            -- Actually delete orphaned records
            DELETE FROM processing_metadata WHERE video_id = target_video_id AND NOT EXISTS (SELECT 1 FROM videos WHERE id = target_video_id);
            DELETE FROM timestamped_segments WHERE video_id = target_video_id AND NOT EXISTS (SELECT 1 FROM videos WHERE id = target_video_id);
            DELETE FROM keywords WHERE video_id = target_video_id AND NOT EXISTS (SELECT 1 FROM videos WHERE id = target_video_id);
            DELETE FROM summaries WHERE video_id = target_video_id AND NOT EXISTS (SELECT 1 FROM videos WHERE id = target_video_id);
            DELETE FROM transcripts WHERE video_id = target_video_id AND NOT EXISTS (SELECT 1 FROM videos WHERE id = target_video_id);
            
            RETURN result;
        END;
        $$ LANGUAGE plpgsql;
    """)
    
    # Add partial index for active processing records
    op.execute("""
        CREATE INDEX CONCURRENTLY idx_processing_metadata_active_status 
        ON processing_metadata (video_id, status) 
        WHERE status IN ('processing', 'pending');
    """)
    
    # Add partial index for recent videos
    op.execute("""
        CREATE INDEX CONCURRENTLY idx_videos_recent 
        ON videos (created_at DESC) 
        WHERE created_at > NOW() - INTERVAL '30 days';
    """)


def downgrade() -> None:
    """
    Remove enhanced cascade delete functionality.
    """
    
    # Drop partial indexes
    op.drop_index('idx_videos_recent', table_name='videos')
    op.drop_index('idx_processing_metadata_active_status', table_name='processing_metadata')
    
    # Drop functions
    op.execute("DROP FUNCTION IF EXISTS cleanup_orphaned_records(INTEGER);")
    op.execute("DROP FUNCTION IF EXISTS validate_cascade_delete_safety(INTEGER);")
    op.execute("DROP FUNCTION IF EXISTS get_cascade_delete_impact(INTEGER);")
    
    # Drop constraints
    op.execute("ALTER TABLE processing_metadata DROP CONSTRAINT IF EXISTS check_processing_status;")
    op.execute("ALTER TABLE videos DROP CONSTRAINT IF EXISTS check_video_id_format;")
    
    # Drop trigger
    op.execute("DROP TRIGGER IF EXISTS video_deletion_audit_trigger ON videos;")
    
    # Drop audit function
    op.execute("DROP FUNCTION IF EXISTS log_video_deletion();")
    
    # Drop audit table
    op.drop_table('deletion_audit_log')