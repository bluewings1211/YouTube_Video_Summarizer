"""
Database cascade delete utilities and enhanced deletion logic.

This module provides advanced cascade delete functionality beyond the basic
SQLAlchemy relationship cascades, including validation, monitoring, and
transaction-safe deletion operations.
"""

import logging
from typing import List, Dict, Any, Optional, Set, Tuple
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy.orm import Session
from sqlalchemy import select, func, text, delete, inspect
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.engine import Engine

from .models import Video, Transcript, Summary, Keyword, TimestampedSegment, ProcessingMetadata
from .exceptions import (
    DatabaseError, DatabaseIntegrityError, DatabaseConstraintError,
    classify_database_error
)

logger = logging.getLogger(__name__)


@dataclass
class CascadeDeleteResult:
    """Result of cascade delete operation."""
    success: bool
    video_id: int
    deleted_counts: Dict[str, int]
    total_deleted: int
    error_message: Optional[str] = None
    execution_time_ms: Optional[float] = None


@dataclass
class CascadeDeleteValidation:
    """Validation result for cascade delete operation."""
    can_delete: bool
    video_exists: bool
    related_counts: Dict[str, int]
    potential_issues: List[str]
    total_related_records: int


class CascadeDeleteManager:
    """
    Advanced cascade delete manager with validation and monitoring.
    
    Provides enhanced deletion functionality with:
    - Pre-deletion validation
    - Detailed deletion tracking
    - Transaction safety
    - Performance monitoring
    - Integrity checking
    """
    
    def __init__(self, session: Session):
        """
        Initialize cascade delete manager.
        
        Args:
            session: Database session
        """
        self.session = session
        self.logger = logging.getLogger(f"{__name__}.CascadeDeleteManager")
        
        # Define table deletion order (important for foreign key constraints)
        self.deletion_order = [
            'processing_metadata',
            'timestamped_segments', 
            'keywords',
            'summaries',
            'transcripts',
            'videos'
        ]
        
        # Model mapping for easier access
        self.model_map = {
            'videos': Video,
            'transcripts': Transcript,
            'summaries': Summary,
            'keywords': Keyword,
            'timestamped_segments': TimestampedSegment,
            'processing_metadata': ProcessingMetadata
        }

    def validate_cascade_delete(self, video_id: int) -> CascadeDeleteValidation:
        """
        Validate that a video can be safely deleted.
        
        Args:
            video_id: Database video ID
            
        Returns:
            CascadeDeleteValidation with validation results
        """
        try:
            # Check if video exists
            video = self.session.get(Video, video_id)
            if not video:
                return CascadeDeleteValidation(
                    can_delete=False,
                    video_exists=False,
                    related_counts={},
                    potential_issues=["Video not found"],
                    total_related_records=0
                )
            
            # Get counts of related records
            related_counts = self._get_related_record_counts(video_id)
            total_related = sum(related_counts.values())
            
            # Check for potential issues
            potential_issues = []
            
            # Check for large number of related records
            if total_related > 10000:
                potential_issues.append(f"Large number of related records ({total_related})")
            
            # Check for any processing metadata with "processing" status
            processing_count = self.session.execute(
                select(func.count(ProcessingMetadata.id)).where(
                    ProcessingMetadata.video_id == video_id,
                    ProcessingMetadata.status == 'processing'
                )
            ).scalar() or 0
            
            if processing_count > 0:
                potential_issues.append(f"Video has {processing_count} active processing records")
            
            # Determine if deletion is safe
            can_delete = len(potential_issues) == 0
            
            return CascadeDeleteValidation(
                can_delete=can_delete,
                video_exists=True,
                related_counts=related_counts,
                potential_issues=potential_issues,
                total_related_records=total_related
            )
            
        except SQLAlchemyError as e:
            self.logger.error(f"Error validating cascade delete for video {video_id}: {e}")
            return CascadeDeleteValidation(
                can_delete=False,
                video_exists=False,
                related_counts={},
                potential_issues=[f"Database error: {str(e)}"],
                total_related_records=0
            )

    def execute_cascade_delete(self, video_id: int, force: bool = False) -> CascadeDeleteResult:
        """
        Execute cascade delete with monitoring and validation.
        
        Args:
            video_id: Database video ID to delete
            force: Skip validation if True
            
        Returns:
            CascadeDeleteResult with operation results
        """
        start_time = datetime.now()
        
        try:
            # Validate deletion unless forced
            if not force:
                validation = self.validate_cascade_delete(video_id)
                if not validation.can_delete:
                    return CascadeDeleteResult(
                        success=False,
                        video_id=video_id,
                        deleted_counts={},
                        total_deleted=0,
                        error_message=f"Validation failed: {', '.join(validation.potential_issues)}"
                    )
            
            # Get initial counts for tracking
            initial_counts = self._get_related_record_counts(video_id)
            
            # Execute deletion with transaction
            with self._transaction_context():
                # Delete the video (cascade will handle related records)
                video = self.session.get(Video, video_id)
                if not video:
                    return CascadeDeleteResult(
                        success=False,
                        video_id=video_id,
                        deleted_counts={},
                        total_deleted=0,
                        error_message="Video not found"
                    )
                
                # Store video info for logging
                video_info = {
                    'id': video.id,
                    'video_id': video.video_id,
                    'title': video.title
                }
                
                # Delete video (cascade will handle related records)
                self.session.delete(video)
                self.session.flush()  # Ensure deletion is executed
                
                # Verify deletion was successful
                final_counts = self._get_related_record_counts(video_id)
                deleted_counts = {
                    table: initial_counts.get(table, 0) - final_counts.get(table, 0)
                    for table in initial_counts.keys()
                }
                deleted_counts['videos'] = 1  # The video itself
                
                total_deleted = sum(deleted_counts.values())
                
                # Calculate execution time
                execution_time = (datetime.now() - start_time).total_seconds() * 1000
                
                self.logger.info(f"Successfully deleted video {video_id} ({video_info})")
                self.logger.info(f"Deleted records: {deleted_counts}")
                
                return CascadeDeleteResult(
                    success=True,
                    video_id=video_id,
                    deleted_counts=deleted_counts,
                    total_deleted=total_deleted,
                    execution_time_ms=execution_time
                )
                
        except SQLAlchemyError as e:
            db_error = classify_database_error(e)
            self.logger.error(f"Database error during cascade delete: {db_error}")
            
            return CascadeDeleteResult(
                success=False,
                video_id=video_id,
                deleted_counts={},
                total_deleted=0,
                error_message=f"Database error: {db_error.message}"
            )
            
        except Exception as e:
            self.logger.error(f"Unexpected error during cascade delete: {e}")
            
            return CascadeDeleteResult(
                success=False,
                video_id=video_id,
                deleted_counts={},
                total_deleted=0,
                error_message=f"Unexpected error: {str(e)}"
            )

    def batch_cascade_delete(self, video_ids: List[int], force: bool = False) -> List[CascadeDeleteResult]:
        """
        Execute batch cascade delete operations.
        
        Args:
            video_ids: List of video IDs to delete
            force: Skip validation if True
            
        Returns:
            List of CascadeDeleteResult for each video
        """
        results = []
        
        for video_id in video_ids:
            result = self.execute_cascade_delete(video_id, force=force)
            results.append(result)
            
            # Log progress for large batches
            if len(video_ids) > 10:
                processed = len(results)
                self.logger.info(f"Batch delete progress: {processed}/{len(video_ids)} videos processed")
        
        # Log batch summary
        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful
        total_deleted = sum(r.total_deleted for r in results)
        
        self.logger.info(f"Batch delete completed: {successful} successful, {failed} failed, {total_deleted} total records deleted")
        
        return results

    def verify_cascade_integrity(self, video_id: int) -> Dict[str, Any]:
        """
        Verify that cascade delete completed successfully.
        
        Args:
            video_id: Video ID that was deleted
            
        Returns:
            Dictionary with integrity check results
        """
        try:
            # Check if video still exists
            video_exists = self.session.get(Video, video_id) is not None
            
            # Check for orphaned records
            orphaned_records = {}
            for table_name, model_class in self.model_map.items():
                if table_name == 'videos':
                    continue
                    
                count = self.session.execute(
                    select(func.count(model_class.id)).where(
                        model_class.video_id == video_id
                    )
                ).scalar() or 0
                
                if count > 0:
                    orphaned_records[table_name] = count
            
            has_orphans = len(orphaned_records) > 0
            
            return {
                'video_exists': video_exists,
                'has_orphaned_records': has_orphans,
                'orphaned_records': orphaned_records,
                'integrity_check_passed': not video_exists and not has_orphans
            }
            
        except SQLAlchemyError as e:
            self.logger.error(f"Error verifying cascade integrity: {e}")
            return {
                'video_exists': None,
                'has_orphaned_records': None,
                'orphaned_records': {},
                'integrity_check_passed': False,
                'error': str(e)
            }

    def cleanup_orphaned_records(self, video_id: int) -> Dict[str, int]:
        """
        Clean up any orphaned records for a deleted video.
        
        Args:
            video_id: Video ID to clean up orphans for
            
        Returns:
            Dictionary with counts of cleaned up records
        """
        cleaned_counts = {}
        
        try:
            with self._transaction_context():
                # Clean up orphaned records in deletion order
                for table_name in self.deletion_order:
                    if table_name == 'videos':
                        continue
                    
                    model_class = self.model_map[table_name]
                    
                    # Delete orphaned records
                    result = self.session.execute(
                        delete(model_class).where(model_class.video_id == video_id)
                    )
                    
                    deleted_count = result.rowcount
                    if deleted_count > 0:
                        cleaned_counts[table_name] = deleted_count
                        self.logger.info(f"Cleaned up {deleted_count} orphaned records from {table_name}")
                
                return cleaned_counts
                
        except SQLAlchemyError as e:
            self.logger.error(f"Error cleaning up orphaned records: {e}")
            raise DatabaseError(f"Failed to clean up orphaned records: {e}")

    def get_cascade_delete_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about potential cascade delete operations.
        
        Returns:
            Dictionary with cascade delete statistics
        """
        try:
            stats = {}
            
            # Get total video count
            total_videos = self.session.execute(
                select(func.count(Video.id))
            ).scalar() or 0
            
            stats['total_videos'] = total_videos
            
            # Get average related record counts
            if total_videos > 0:
                avg_related = {}
                for table_name, model_class in self.model_map.items():
                    if table_name == 'videos':
                        continue
                    
                    total_records = self.session.execute(
                        select(func.count(model_class.id))
                    ).scalar() or 0
                    
                    avg_related[table_name] = {
                        'total_records': total_records,
                        'avg_per_video': round(total_records / total_videos, 2)
                    }
                
                stats['average_related_records'] = avg_related
            
            # Get videos with most related records
            videos_with_most_related = self.session.execute(
                select(
                    Video.id,
                    Video.video_id,
                    Video.title,
                    func.count(Transcript.id).label('transcript_count'),
                    func.count(Summary.id).label('summary_count'),
                    func.count(Keyword.id).label('keyword_count'),
                    func.count(TimestampedSegment.id).label('segment_count'),
                    func.count(ProcessingMetadata.id).label('metadata_count')
                ).select_from(Video)
                .outerjoin(Transcript, Video.id == Transcript.video_id)
                .outerjoin(Summary, Video.id == Summary.video_id)
                .outerjoin(Keyword, Video.id == Keyword.video_id)
                .outerjoin(TimestampedSegment, Video.id == TimestampedSegment.video_id)
                .outerjoin(ProcessingMetadata, Video.id == ProcessingMetadata.video_id)
                .group_by(Video.id, Video.video_id, Video.title)
                .order_by(
                    (func.count(Transcript.id) + func.count(Summary.id) + 
                     func.count(Keyword.id) + func.count(TimestampedSegment.id) + 
                     func.count(ProcessingMetadata.id)).desc()
                )
                .limit(10)
            ).fetchall()
            
            stats['videos_with_most_related'] = [
                {
                    'id': row[0],
                    'video_id': row[1],
                    'title': row[2][:50] + '...' if len(row[2]) > 50 else row[2],
                    'total_related': sum(row[3:]),
                    'breakdown': {
                        'transcripts': row[3],
                        'summaries': row[4],
                        'keywords': row[5],
                        'segments': row[6],
                        'metadata': row[7]
                    }
                }
                for row in videos_with_most_related
            ]
            
            return stats
            
        except SQLAlchemyError as e:
            self.logger.error(f"Error getting cascade delete statistics: {e}")
            raise DatabaseError(f"Failed to get statistics: {e}")

    def _get_related_record_counts(self, video_id: int) -> Dict[str, int]:
        """Get counts of related records for a video."""
        counts = {}
        
        try:
            for table_name, model_class in self.model_map.items():
                if table_name == 'videos':
                    continue
                    
                count = self.session.execute(
                    select(func.count(model_class.id)).where(
                        model_class.video_id == video_id
                    )
                ).scalar() or 0
                
                counts[table_name] = count
            
            return counts
            
        except SQLAlchemyError as e:
            self.logger.error(f"Error getting related record counts: {e}")
            return {}

    @contextmanager
    def _transaction_context(self):
        """Context manager for transaction handling."""
        try:
            yield
            self.session.commit()
        except Exception as e:
            self.session.rollback()
            raise


# Utility functions for easier access
def create_cascade_delete_manager(session: Session) -> CascadeDeleteManager:
    """Create a cascade delete manager instance."""
    return CascadeDeleteManager(session)


def validate_video_deletion(session: Session, video_id: int) -> CascadeDeleteValidation:
    """Validate that a video can be safely deleted."""
    manager = create_cascade_delete_manager(session)
    return manager.validate_cascade_delete(video_id)


def execute_enhanced_cascade_delete(session: Session, video_id: int, force: bool = False) -> CascadeDeleteResult:
    """Execute enhanced cascade delete with validation and monitoring."""
    manager = create_cascade_delete_manager(session)
    return manager.execute_cascade_delete(video_id, force=force)