"""
Status metrics service for aggregating and calculating processing metrics.
Provides performance analytics and monitoring capabilities.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_, case, extract
from statistics import median

from ..database.connection import get_db_session
from ..database.status_models import (
    ProcessingStatus, StatusHistory, StatusMetrics,
    ProcessingStatusType, StatusChangeType
)
from ..utils.error_messages import ErrorMessages


class StatusMetricsService:
    """
    Service for calculating and managing processing status metrics.
    
    This service provides functionality to calculate performance metrics,
    generate reports, and maintain aggregated metrics for dashboard use.
    """
    
    def __init__(self, db_session: Optional[Session] = None):
        """
        Initialize the StatusMetricsService.
        
        Args:
            db_session: Optional database session
        """
        self.db_session = db_session
        self._should_close_session = db_session is None
        self.logger = logging.getLogger(__name__)
    
    def __enter__(self):
        """Context manager entry."""
        if self.db_session is None:
            self.db_session = get_db_session()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self._should_close_session and self.db_session:
            self.db_session.close()
    
    def calculate_hourly_metrics(self, target_date: datetime) -> StatusMetrics:
        """
        Calculate hourly metrics for a specific date and hour.
        
        Args:
            target_date: Target date and hour for metrics calculation
            
        Returns:
            StatusMetrics object with calculated metrics
        """
        try:
            # Round to the hour
            target_hour = target_date.replace(minute=0, second=0, microsecond=0)
            next_hour = target_hour + timedelta(hours=1)
            
            # Get all processing statuses for the hour
            statuses = self.db_session.query(ProcessingStatus).filter(
                and_(
                    ProcessingStatus.created_at >= target_hour,
                    ProcessingStatus.created_at < next_hour
                )
            ).all()
            
            if not statuses:
                return self._create_empty_metrics(target_hour, target_hour.hour)
            
            # Calculate basic counts
            total_items = len(statuses)
            completed_items = sum(1 for s in statuses if s.is_completed)
            failed_items = sum(1 for s in statuses if s.is_failed)
            cancelled_items = sum(1 for s in statuses if s.is_cancelled)
            
            # Calculate processing times for completed items
            processing_times = []
            for status in statuses:
                if status.is_completed and status.started_at and status.completed_at:
                    processing_time = (status.completed_at - status.started_at).total_seconds()
                    processing_times.append(processing_time)
            
            # Calculate time-based metrics
            avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else None
            median_processing_time = median(processing_times) if processing_times else None
            max_processing_time = max(processing_times) if processing_times else None
            min_processing_time = min(processing_times) if processing_times else None
            
            # Calculate retry rate
            retry_count = sum(s.retry_count for s in statuses if s.retry_count > 0)
            retry_rate = (retry_count / total_items * 100) if total_items > 0 else 0.0
            
            # Calculate success rate
            success_rate = (completed_items / total_items * 100) if total_items > 0 else 0.0
            
            # Calculate queue wait time
            queue_wait_times = []
            for status in statuses:
                if status.started_at and status.created_at:
                    wait_time = (status.started_at - status.created_at).total_seconds()
                    queue_wait_times.append(wait_time)
            
            avg_queue_wait_time = sum(queue_wait_times) / len(queue_wait_times) if queue_wait_times else None
            
            # Calculate worker utilization (simplified)
            active_workers = len(set(s.worker_id for s in statuses if s.worker_id))
            worker_utilization = min(100.0, (active_workers / max(1, total_items)) * 100)
            
            # Create or update metrics record
            metrics = StatusMetrics(
                metric_date=target_hour,
                metric_hour=target_hour.hour,
                total_items=total_items,
                completed_items=completed_items,
                failed_items=failed_items,
                cancelled_items=cancelled_items,
                average_processing_time_seconds=avg_processing_time,
                median_processing_time_seconds=median_processing_time,
                max_processing_time_seconds=max_processing_time,
                min_processing_time_seconds=min_processing_time,
                retry_rate_percentage=retry_rate,
                success_rate_percentage=success_rate,
                queue_wait_time_seconds=avg_queue_wait_time,
                worker_utilization_percentage=worker_utilization,
                metrics_metadata={
                    'calculation_timestamp': datetime.utcnow().isoformat(),
                    'active_workers': active_workers,
                    'processing_time_samples': len(processing_times),
                    'queue_wait_time_samples': len(queue_wait_times)
                }
            )
            
            # Save or update existing metrics
            existing_metrics = self.db_session.query(StatusMetrics).filter(
                and_(
                    StatusMetrics.metric_date == target_hour,
                    StatusMetrics.metric_hour == target_hour.hour
                )
            ).first()
            
            if existing_metrics:
                # Update existing metrics
                for attr, value in metrics.__dict__.items():
                    if not attr.startswith('_') and attr not in ['id', 'created_at']:
                        setattr(existing_metrics, attr, value)
                existing_metrics.updated_at = datetime.utcnow()
                metrics = existing_metrics
            else:
                # Create new metrics
                self.db_session.add(metrics)
            
            self.db_session.commit()
            return metrics
            
        except Exception as e:
            self.db_session.rollback()
            self.logger.error(f"Error calculating hourly metrics: {e}")
            raise
    
    def calculate_daily_metrics(self, target_date: datetime) -> StatusMetrics:
        """
        Calculate daily metrics for a specific date.
        
        Args:
            target_date: Target date for metrics calculation
            
        Returns:
            StatusMetrics object with calculated daily metrics
        """
        try:
            # Round to the day
            target_day = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
            next_day = target_day + timedelta(days=1)
            
            # Get all processing statuses for the day
            statuses = self.db_session.query(ProcessingStatus).filter(
                and_(
                    ProcessingStatus.created_at >= target_day,
                    ProcessingStatus.created_at < next_day
                )
            ).all()
            
            if not statuses:
                return self._create_empty_metrics(target_day, None)
            
            # Calculate metrics similar to hourly but for the entire day
            total_items = len(statuses)
            completed_items = sum(1 for s in statuses if s.is_completed)
            failed_items = sum(1 for s in statuses if s.is_failed)
            cancelled_items = sum(1 for s in statuses if s.is_cancelled)
            
            # Calculate processing times
            processing_times = []
            for status in statuses:
                if status.is_completed and status.started_at and status.completed_at:
                    processing_time = (status.completed_at - status.started_at).total_seconds()
                    processing_times.append(processing_time)
            
            # Calculate time-based metrics
            avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else None
            median_processing_time = median(processing_times) if processing_times else None
            max_processing_time = max(processing_times) if processing_times else None
            min_processing_time = min(processing_times) if processing_times else None
            
            # Calculate rates
            retry_count = sum(s.retry_count for s in statuses if s.retry_count > 0)
            retry_rate = (retry_count / total_items * 100) if total_items > 0 else 0.0
            success_rate = (completed_items / total_items * 100) if total_items > 0 else 0.0
            
            # Calculate queue wait time
            queue_wait_times = []
            for status in statuses:
                if status.started_at and status.created_at:
                    wait_time = (status.started_at - status.created_at).total_seconds()
                    queue_wait_times.append(wait_time)
            
            avg_queue_wait_time = sum(queue_wait_times) / len(queue_wait_times) if queue_wait_times else None
            
            # Calculate worker utilization
            active_workers = len(set(s.worker_id for s in statuses if s.worker_id))
            worker_utilization = min(100.0, (active_workers / max(1, total_items)) * 100)
            
            # Create or update metrics record
            metrics = StatusMetrics(
                metric_date=target_day,
                metric_hour=None,  # Daily metrics don't have specific hour
                total_items=total_items,
                completed_items=completed_items,
                failed_items=failed_items,
                cancelled_items=cancelled_items,
                average_processing_time_seconds=avg_processing_time,
                median_processing_time_seconds=median_processing_time,
                max_processing_time_seconds=max_processing_time,
                min_processing_time_seconds=min_processing_time,
                retry_rate_percentage=retry_rate,
                success_rate_percentage=success_rate,
                queue_wait_time_seconds=avg_queue_wait_time,
                worker_utilization_percentage=worker_utilization,
                metrics_metadata={
                    'calculation_timestamp': datetime.utcnow().isoformat(),
                    'active_workers': active_workers,
                    'processing_time_samples': len(processing_times),
                    'queue_wait_time_samples': len(queue_wait_times),
                    'is_daily_metric': True
                }
            )
            
            # Save or update existing metrics
            existing_metrics = self.db_session.query(StatusMetrics).filter(
                and_(
                    StatusMetrics.metric_date == target_day,
                    StatusMetrics.metric_hour.is_(None)
                )
            ).first()
            
            if existing_metrics:
                # Update existing metrics
                for attr, value in metrics.__dict__.items():
                    if not attr.startswith('_') and attr not in ['id', 'created_at']:
                        setattr(existing_metrics, attr, value)
                existing_metrics.updated_at = datetime.utcnow()
                metrics = existing_metrics
            else:
                # Create new metrics
                self.db_session.add(metrics)
            
            self.db_session.commit()
            return metrics
            
        except Exception as e:
            self.db_session.rollback()
            self.logger.error(f"Error calculating daily metrics: {e}")
            raise
    
    def get_metrics_for_period(
        self,
        start_date: datetime,
        end_date: datetime,
        hourly: bool = True
    ) -> List[StatusMetrics]:
        """
        Get metrics for a date range.
        
        Args:
            start_date: Start date for metrics
            end_date: End date for metrics
            hourly: Whether to get hourly or daily metrics
            
        Returns:
            List of StatusMetrics for the period
        """
        try:
            query = self.db_session.query(StatusMetrics).filter(
                and_(
                    StatusMetrics.metric_date >= start_date,
                    StatusMetrics.metric_date <= end_date
                )
            )
            
            if hourly:
                query = query.filter(StatusMetrics.metric_hour.isnot(None))
            else:
                query = query.filter(StatusMetrics.metric_hour.is_(None))
            
            return query.order_by(StatusMetrics.metric_date).all()
            
        except Exception as e:
            self.logger.error(f"Error getting metrics for period: {e}")
            raise
    
    def get_current_performance_summary(self) -> Dict[str, Any]:
        """
        Get current performance summary.
        
        Returns:
            Dictionary with current performance metrics
        """
        try:
            # Get active statuses
            active_statuses = self.db_session.query(ProcessingStatus).filter(
                ProcessingStatus.status.in_([
                    ProcessingStatusType.STARTING,
                    ProcessingStatusType.YOUTUBE_METADATA,
                    ProcessingStatusType.TRANSCRIPT_EXTRACTION,
                    ProcessingStatusType.LANGUAGE_DETECTION,
                    ProcessingStatusType.SUMMARY_GENERATION,
                    ProcessingStatusType.KEYWORD_EXTRACTION,
                    ProcessingStatusType.TIMESTAMPED_SEGMENTS,
                    ProcessingStatusType.FINALIZING
                ])
            ).all()
            
            # Get today's completed statuses
            today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
            tomorrow = today + timedelta(days=1)
            
            today_statuses = self.db_session.query(ProcessingStatus).filter(
                and_(
                    ProcessingStatus.created_at >= today,
                    ProcessingStatus.created_at < tomorrow
                )
            ).all()
            
            # Calculate summary metrics
            active_count = len(active_statuses)
            today_total = len(today_statuses)
            today_completed = sum(1 for s in today_statuses if s.is_completed)
            today_failed = sum(1 for s in today_statuses if s.is_failed)
            
            # Get current workers
            active_workers = set(s.worker_id for s in active_statuses if s.worker_id)
            
            # Calculate average progress of active items
            if active_statuses:
                avg_progress = sum(s.progress_percentage for s in active_statuses) / len(active_statuses)
            else:
                avg_progress = 0.0
            
            # Get recent metrics
            recent_metrics = self.db_session.query(StatusMetrics).filter(
                StatusMetrics.metric_date >= today - timedelta(days=7)
            ).order_by(StatusMetrics.metric_date.desc()).limit(7).all()
            
            # Calculate trend
            if len(recent_metrics) >= 2:
                latest_success_rate = recent_metrics[0].success_rate_percentage or 0
                previous_success_rate = recent_metrics[1].success_rate_percentage or 0
                success_rate_trend = latest_success_rate - previous_success_rate
            else:
                success_rate_trend = 0.0
            
            return {
                'active_processing_count': active_count,
                'active_workers': list(active_workers),
                'worker_count': len(active_workers),
                'average_progress': round(avg_progress, 2),
                'today_total': today_total,
                'today_completed': today_completed,
                'today_failed': today_failed,
                'today_success_rate': round((today_completed / today_total * 100) if today_total > 0 else 0, 2),
                'success_rate_trend': round(success_rate_trend, 2),
                'recent_metrics_count': len(recent_metrics),
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting current performance summary: {e}")
            raise
    
    def get_worker_performance(self, worker_id: str, days: int = 7) -> Dict[str, Any]:
        """
        Get performance metrics for a specific worker.
        
        Args:
            worker_id: Worker identifier
            days: Number of days to analyze
            
        Returns:
            Dictionary with worker performance metrics
        """
        try:
            # Get worker statuses for the period
            start_date = datetime.utcnow() - timedelta(days=days)
            
            worker_statuses = self.db_session.query(ProcessingStatus).filter(
                and_(
                    ProcessingStatus.worker_id == worker_id,
                    ProcessingStatus.created_at >= start_date
                )
            ).all()
            
            if not worker_statuses:
                return {
                    'worker_id': worker_id,
                    'total_processed': 0,
                    'success_rate': 0.0,
                    'average_processing_time': None,
                    'error_rate': 0.0,
                    'days_analyzed': days
                }
            
            # Calculate metrics
            total_processed = len(worker_statuses)
            completed_count = sum(1 for s in worker_statuses if s.is_completed)
            failed_count = sum(1 for s in worker_statuses if s.is_failed)
            
            # Calculate processing times
            processing_times = []
            for status in worker_statuses:
                if status.is_completed and status.started_at and status.completed_at:
                    processing_time = (status.completed_at - status.started_at).total_seconds()
                    processing_times.append(processing_time)
            
            avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else None
            
            # Calculate rates
            success_rate = (completed_count / total_processed * 100) if total_processed > 0 else 0.0
            error_rate = (failed_count / total_processed * 100) if total_processed > 0 else 0.0
            
            return {
                'worker_id': worker_id,
                'total_processed': total_processed,
                'completed_count': completed_count,
                'failed_count': failed_count,
                'success_rate': round(success_rate, 2),
                'error_rate': round(error_rate, 2),
                'average_processing_time': round(avg_processing_time, 2) if avg_processing_time else None,
                'processing_time_samples': len(processing_times),
                'days_analyzed': days,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting worker performance: {e}")
            raise
    
    def get_status_distribution(self, days: int = 7) -> Dict[str, int]:
        """
        Get distribution of processing statuses.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Dictionary with status distribution
        """
        try:
            start_date = datetime.utcnow() - timedelta(days=days)
            
            # Query status distribution
            status_counts = self.db_session.query(
                ProcessingStatus.status,
                func.count(ProcessingStatus.id).label('count')
            ).filter(
                ProcessingStatus.created_at >= start_date
            ).group_by(ProcessingStatus.status).all()
            
            # Convert to dictionary
            distribution = {}
            for status, count in status_counts:
                distribution[status.value] = count
            
            return distribution
            
        except Exception as e:
            self.logger.error(f"Error getting status distribution: {e}")
            raise
    
    def _create_empty_metrics(self, metric_date: datetime, metric_hour: Optional[int]) -> StatusMetrics:
        """
        Create empty metrics record.
        
        Args:
            metric_date: Date for metrics
            metric_hour: Hour for metrics (None for daily)
            
        Returns:
            StatusMetrics with zero values
        """
        return StatusMetrics(
            metric_date=metric_date,
            metric_hour=metric_hour,
            total_items=0,
            completed_items=0,
            failed_items=0,
            cancelled_items=0,
            average_processing_time_seconds=None,
            median_processing_time_seconds=None,
            max_processing_time_seconds=None,
            min_processing_time_seconds=None,
            retry_rate_percentage=0.0,
            success_rate_percentage=0.0,
            queue_wait_time_seconds=None,
            worker_utilization_percentage=0.0,
            metrics_metadata={
                'calculation_timestamp': datetime.utcnow().isoformat(),
                'is_empty_metric': True
            }
        )
    
    def cleanup_old_metrics(self, days_old: int = 90) -> int:
        """
        Clean up old metrics records.
        
        Args:
            days_old: Age threshold in days
            
        Returns:
            Number of records cleaned up
        """
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_old)
            
            count = self.db_session.query(StatusMetrics).filter(
                StatusMetrics.metric_date < cutoff_date
            ).count()
            
            self.db_session.query(StatusMetrics).filter(
                StatusMetrics.metric_date < cutoff_date
            ).delete()
            
            self.db_session.commit()
            return count
            
        except Exception as e:
            self.db_session.rollback()
            self.logger.error(f"Error cleaning up old metrics: {e}")
            raise