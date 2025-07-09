"""
Database maintenance and cleanup utilities.

This module provides utilities for database maintenance tasks including
cleanup of old records, health checks, and optimization operations.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import select, delete, func, text
from sqlalchemy.exc import SQLAlchemyError

from .models import Video, Transcript, Summary, Keyword, TimestampedSegment, ProcessingMetadata
from .connection import get_database_session, get_database_info, check_database_health
from .exceptions import DatabaseError, classify_database_error
from .monitor import db_monitor, MonitoredOperation

logger = logging.getLogger(__name__)


class DatabaseMaintenance:
    """
    Database maintenance and cleanup operations.
    
    Provides methods for cleaning up old records, checking database health,
    and performing optimization tasks.
    """
    
    def __init__(self, session: Optional[Session] = None):
        """
        Initialize database maintenance utility.
        
        Args:
            session: Optional database session
        """
        self.session = session
        self.logger = logging.getLogger(f"{__name__}.DatabaseMaintenance")
    
    def _get_session(self) -> Session:
        """Get database session (internal method)."""
        if self.session:
            return self.session
        else:
            # Use context manager for session - this is not ideal, should use dependency injection
            raise RuntimeError("No session provided. Use context manager for database operations.")
    
    def cleanup_old_records(
        self,
        days_to_keep: int = 30,
        dry_run: bool = False,
        batch_size: int = 1000
    ) -> Dict[str, Any]:
        """
        Clean up old database records beyond the retention period.
        
        Args:
            days_to_keep: Number of days to keep records
            dry_run: If True, only count records without deleting
            batch_size: Number of records to process in each batch
            
        Returns:
            Dictionary with cleanup results
        """
        with MonitoredOperation("cleanup_old_records"):
            try:
                cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
                results = {
                    'cutoff_date': cutoff_date.isoformat(),
                    'days_to_keep': days_to_keep,
                    'dry_run': dry_run,
                    'deleted_counts': {},
                    'total_deleted': 0,
                    'errors': []
                }
                
                with get_database_session() as session:
                    # Clean up processing metadata first (has foreign keys)
                    deleted_metadata = self._cleanup_table_records(
                        session, ProcessingMetadata, cutoff_date, dry_run, batch_size
                    )
                    results['deleted_counts']['processing_metadata'] = deleted_metadata
                    
                    # Clean up other related tables
                    tables_to_clean = [
                        (TimestampedSegment, 'timestamped_segments'),
                        (Keyword, 'keywords'),
                        (Summary, 'summaries'),
                        (Transcript, 'transcripts')
                    ]
                    
                    for model_class, table_name in tables_to_clean:
                        deleted_count = self._cleanup_table_records(
                            session, model_class, cutoff_date, dry_run, batch_size
                        )
                        results['deleted_counts'][table_name] = deleted_count
                    
                    # Finally clean up videos that have no related records
                    deleted_videos = self._cleanup_orphaned_videos(
                        session, cutoff_date, dry_run, batch_size
                    )
                    results['deleted_counts']['videos'] = deleted_videos
                    
                    # Calculate total
                    results['total_deleted'] = sum(results['deleted_counts'].values())
                    
                    if not dry_run and results['total_deleted'] > 0:
                        session.commit()
                        self.logger.info(f"Cleaned up {results['total_deleted']} old records")
                    elif dry_run:
                        self.logger.info(f"Dry run: would delete {results['total_deleted']} records")
                
                return results
                
            except Exception as e:
                db_error = classify_database_error(e)
                self.logger.error(f"Error during cleanup: {db_error}")
                results['errors'].append(str(db_error))
                return results
    
    def _cleanup_table_records(
        self,
        session: Session,
        model_class,
        cutoff_date: datetime,
        dry_run: bool,
        batch_size: int
    ) -> int:
        """Clean up records from a specific table."""
        total_deleted = 0
        
        try:
            while True:
                # Find old records in batches
                if dry_run:
                    # Count records that would be deleted
                    stmt = select(func.count(model_class.id)).where(
                        model_class.created_at < cutoff_date
                    )
                    result = session.execute(stmt)
                    count = result.scalar() or 0
                    total_deleted += count
                    break
                else:
                    # Delete records in batches
                    stmt = select(model_class.id).where(
                        model_class.created_at < cutoff_date
                    ).limit(batch_size)
                    result = session.execute(stmt)
                    ids_to_delete = [row[0] for row in result.fetchall()]
                    
                    if not ids_to_delete:
                        break
                    
                    # Delete the batch
                    delete_stmt = delete(model_class).where(
                        model_class.id.in_(ids_to_delete)
                    )
                    delete_result = session.execute(delete_stmt)
                    deleted_count = delete_result.rowcount
                    total_deleted += deleted_count
                    
                    self.logger.debug(f"Deleted {deleted_count} records from {model_class.__tablename__}")
                    
                    # If we deleted fewer than batch_size, we're done
                    if deleted_count < batch_size:
                        break
            
        except Exception as e:
            self.logger.error(f"Error cleaning up {model_class.__tablename__}: {e}")
            
        return total_deleted
    
    def _cleanup_orphaned_videos(
        self,
        session: Session,
        cutoff_date: datetime,
        dry_run: bool,
        batch_size: int
    ) -> int:
        """Clean up videos that have no related records and are old."""
        total_deleted = 0
        
        try:
            # Find videos older than cutoff with no related records
            subquery_transcripts = select(Transcript.video_id).where(Transcript.video_id == Video.id)
            subquery_summaries = select(Summary.video_id).where(Summary.video_id == Video.id)
            subquery_keywords = select(Keyword.video_id).where(Keyword.video_id == Video.id)
            subquery_segments = select(TimestampedSegment.video_id).where(TimestampedSegment.video_id == Video.id)
            subquery_metadata = select(ProcessingMetadata.video_id).where(ProcessingMetadata.video_id == Video.id)
            
            if dry_run:
                # Count orphaned videos
                stmt = select(func.count(Video.id)).where(
                    Video.created_at < cutoff_date,
                    ~subquery_transcripts.exists(),
                    ~subquery_summaries.exists(),
                    ~subquery_keywords.exists(),
                    ~subquery_segments.exists(),
                    ~subquery_metadata.exists()
                )
                result = session.execute(stmt)
                total_deleted = result.scalar() or 0
            else:
                # Delete orphaned videos in batches
                while True:
                    stmt = select(Video.id).where(
                        Video.created_at < cutoff_date,
                        ~subquery_transcripts.exists(),
                        ~subquery_summaries.exists(),
                        ~subquery_keywords.exists(),
                        ~subquery_segments.exists(),
                        ~subquery_metadata.exists()
                    ).limit(batch_size)
                    
                    result = session.execute(stmt)
                    ids_to_delete = [row[0] for row in result.fetchall()]
                    
                    if not ids_to_delete:
                        break
                    
                    # Delete the batch
                    delete_stmt = delete(Video).where(Video.id.in_(ids_to_delete))
                    delete_result = session.execute(delete_stmt)
                    deleted_count = delete_result.rowcount
                    total_deleted += deleted_count
                    
                    self.logger.debug(f"Deleted {deleted_count} orphaned videos")
                    
                    if deleted_count < batch_size:
                        break
                        
        except Exception as e:
            self.logger.error(f"Error cleaning up orphaned videos: {e}")
            
        return total_deleted
    
    def get_database_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive database statistics.
        
        Returns:
            Dictionary with database statistics
        """
        with MonitoredOperation("get_database_statistics"):
            try:
                with get_database_session() as session:
                    stats = {
                        'timestamp': datetime.utcnow().isoformat(),
                        'table_counts': {},
                        'table_sizes': {},
                        'oldest_records': {},
                        'newest_records': {},
                        'total_records': 0
                    }
                    
                    # Get counts and date ranges for each table
                    tables = [
                        (Video, 'videos'),
                        (Transcript, 'transcripts'),
                        (Summary, 'summaries'),
                        (Keyword, 'keywords'),
                        (TimestampedSegment, 'timestamped_segments'),
                        (ProcessingMetadata, 'processing_metadata')
                    ]
                    
                    for model_class, table_name in tables:
                        # Count records
                        count_stmt = select(func.count(model_class.id))
                        count_result = session.execute(count_stmt)
                        count = count_result.scalar() or 0
                        stats['table_counts'][table_name] = count
                        stats['total_records'] += count
                        
                        if count > 0:
                            # Get oldest record
                            oldest_stmt = select(func.min(model_class.created_at))
                            oldest_result = session.execute(oldest_stmt)
                            oldest = oldest_result.scalar()
                            stats['oldest_records'][table_name] = oldest.isoformat() if oldest else None
                            
                            # Get newest record
                            newest_stmt = select(func.max(model_class.created_at))
                            newest_result = session.execute(newest_stmt)
                            newest = newest_result.scalar()
                            stats['newest_records'][table_name] = newest.isoformat() if newest else None
                            
                            # Get table size (PostgreSQL specific)
                            try:
                                size_stmt = text(f"SELECT pg_total_relation_size('{table_name}') AS size")
                                size_result = session.execute(size_stmt)
                                size = size_result.scalar()
                                stats['table_sizes'][table_name] = size
                            except Exception:
                                # Not PostgreSQL or permission issue
                                stats['table_sizes'][table_name] = None
                        else:
                            stats['oldest_records'][table_name] = None
                            stats['newest_records'][table_name] = None
                            stats['table_sizes'][table_name] = None
                    
                    return stats
                    
            except Exception as e:
                db_error = classify_database_error(e)
                self.logger.error(f"Error getting database statistics: {db_error}")
                return {
                    'error': str(db_error),
                    'timestamp': datetime.utcnow().isoformat()
                }
    
    def vacuum_analyze_tables(self) -> Dict[str, Any]:
        """
        Perform VACUUM ANALYZE on database tables (PostgreSQL specific).
        
        Returns:
            Dictionary with vacuum results
        """
        with MonitoredOperation("vacuum_analyze_tables"):
            try:
                results = {
                    'timestamp': datetime.utcnow().isoformat(),
                    'tables_processed': [],
                    'errors': []
                }
                
                with get_database_session() as session:
                    table_names = ['videos', 'transcripts', 'summaries', 'keywords', 
                                 'timestamped_segments', 'processing_metadata']
                    
                    for table_name in table_names:
                        try:
                            vacuum_stmt = text(f"VACUUM ANALYZE {table_name}")
                            session.execute(vacuum_stmt)
                            results['tables_processed'].append(table_name)
                            self.logger.debug(f"VACUUM ANALYZE completed for {table_name}")
                        except Exception as e:
                            error_msg = f"Error vacuum analyzing {table_name}: {str(e)}"
                            results['errors'].append(error_msg)
                            self.logger.error(error_msg)
                    
                    session.commit()
                
                return results
                
            except Exception as e:
                db_error = classify_database_error(e)
                self.logger.error(f"Error during vacuum analyze: {db_error}")
                return {
                    'error': str(db_error),
                    'timestamp': datetime.utcnow().isoformat()
                }
    
    def check_data_integrity(self) -> Dict[str, Any]:
        """
        Check database data integrity and consistency.
        
        Returns:
            Dictionary with integrity check results
        """
        with MonitoredOperation("check_data_integrity"):
            try:
                results = {
                    'timestamp': datetime.utcnow().isoformat(),
                    'checks': {},
                    'issues_found': [],
                    'total_issues': 0
                }
                
                with get_database_session() as session:
                    # Check for orphaned records
                    orphan_checks = [
                        (Transcript, Video, 'transcripts_without_videos'),
                        (Summary, Video, 'summaries_without_videos'),
                        (Keyword, Video, 'keywords_without_videos'),
                        (TimestampedSegment, Video, 'segments_without_videos'),
                        (ProcessingMetadata, Video, 'metadata_without_videos')
                    ]
                    
                    for child_model, parent_model, check_name in orphan_checks:
                        # Find orphaned records
                        subquery = select(parent_model.id).where(parent_model.id == child_model.video_id)
                        stmt = select(func.count(child_model.id)).where(~subquery.exists())
                        result = session.execute(stmt)
                        orphan_count = result.scalar() or 0
                        
                        results['checks'][check_name] = orphan_count
                        if orphan_count > 0:
                            issue = f"Found {orphan_count} orphaned records in {child_model.__tablename__}"
                            results['issues_found'].append(issue)
                            results['total_issues'] += orphan_count
                    
                    # Check for videos with duplicate video_ids
                    duplicate_stmt = select(
                        Video.video_id,
                        func.count(Video.id).label('count')
                    ).group_by(Video.video_id).having(func.count(Video.id) > 1)
                    
                    duplicate_result = session.execute(duplicate_stmt)
                    duplicates = duplicate_result.fetchall()
                    
                    results['checks']['duplicate_video_ids'] = len(duplicates)
                    if duplicates:
                        for video_id, count in duplicates:
                            issue = f"Video ID '{video_id}' has {count} duplicate records"
                            results['issues_found'].append(issue)
                            results['total_issues'] += count - 1  # Subtract 1 for the original
                    
                    # Check for records with invalid JSON
                    json_checks = [
                        (Keyword, 'keywords_json', 'invalid_keywords_json'),
                        (TimestampedSegment, 'segments_json', 'invalid_segments_json'),
                        (ProcessingMetadata, 'workflow_params', 'invalid_workflow_params')
                    ]
                    
                    for model_class, json_field, check_name in json_checks:
                        # This is a simplified check - in practice you might want more sophisticated JSON validation
                        stmt = select(func.count(model_class.id)).where(
                            getattr(model_class, json_field).is_(None)
                        )
                        result = session.execute(stmt)
                        null_count = result.scalar() or 0
                        
                        results['checks'][check_name] = null_count
                        if null_count > 0:
                            issue = f"Found {null_count} records with null {json_field} in {model_class.__tablename__}"
                            results['issues_found'].append(issue)
                
                return results
                
            except Exception as e:
                db_error = classify_database_error(e)
                self.logger.error(f"Error during integrity check: {db_error}")
                return {
                    'error': str(db_error),
                    'timestamp': datetime.utcnow().isoformat()
                }


# Convenience functions
def cleanup_old_records(days_to_keep: int = 30, dry_run: bool = False) -> Dict[str, Any]:
    """
    Convenience function to clean up old database records.
    
    Args:
        days_to_keep: Number of days to keep records
        dry_run: If True, only count records without deleting
        
    Returns:
        Dictionary with cleanup results
    """
    maintenance = DatabaseMaintenance()
    return maintenance.cleanup_old_records(days_to_keep, dry_run)


def get_database_health_detailed() -> Dict[str, Any]:
    """
    Get detailed database health information.
    
    Returns:
        Dictionary with comprehensive health data
    """
    try:
        # Get basic health info
        health_info = check_database_health()
        
        # Get detailed statistics
        maintenance = DatabaseMaintenance()
        stats = maintenance.get_database_statistics()
        
        # Get monitoring info
        monitor_health = db_monitor.get_health_status()
        
        return {
            'basic_health': health_info,
            'statistics': stats,
            'monitoring': monitor_health,
            'timestamp': datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting detailed health info: {e}")
        return {
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }


def run_maintenance_tasks(
    cleanup_days: int = 30,
    run_vacuum: bool = True,
    check_integrity: bool = True
) -> Dict[str, Any]:
    """
    Run comprehensive database maintenance tasks.
    
    Args:
        cleanup_days: Days to keep for cleanup
        run_vacuum: Whether to run VACUUM ANALYZE
        check_integrity: Whether to check data integrity
        
    Returns:
        Dictionary with all maintenance results
    """
    results = {
        'timestamp': datetime.utcnow().isoformat(),
        'tasks_run': [],
        'cleanup_results': None,
        'vacuum_results': None,
        'integrity_results': None,
        'errors': []
    }
    
    maintenance = DatabaseMaintenance()
    
    try:
        # Cleanup old records
        if cleanup_days > 0:
            results['cleanup_results'] = maintenance.cleanup_old_records(cleanup_days)
            results['tasks_run'].append('cleanup')
        
        # Vacuum analyze tables
        if run_vacuum:
            results['vacuum_results'] = maintenance.vacuum_analyze_tables()
            results['tasks_run'].append('vacuum')
        
        # Check data integrity
        if check_integrity:
            results['integrity_results'] = maintenance.check_data_integrity()
            results['tasks_run'].append('integrity_check')
        
        logger.info(f"Maintenance tasks completed: {results['tasks_run']}")
        
    except Exception as e:
        error_msg = f"Error during maintenance tasks: {e}"
        results['errors'].append(error_msg)
        logger.error(error_msg)
    
    return results