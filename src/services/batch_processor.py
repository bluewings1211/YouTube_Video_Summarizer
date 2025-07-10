"""
Batch processor integration for YouTube video summarization workflow.

This module provides the integration between BatchService and the existing
video processing workflow, enabling batch processing of YouTube videos
through the PocketFlow system.
"""

import logging
import asyncio
from typing import Optional, Dict, Any, List
from datetime import datetime
from contextlib import asynccontextmanager

from ..database.connection import get_database_session
from ..database.batch_models import BatchItemStatus, BatchStatus
from ..flow import YouTubeSummarizerFlow, WorkflowConfig
from ..services.batch_service import (
    BatchService, BatchItemResult, BatchServiceError
)
from ..services.video_service import VideoService
from ..utils.youtube_utils import extract_video_id_from_url

logger = logging.getLogger(__name__)


class BatchProcessorError(Exception):
    """Custom exception for batch processor operations."""
    pass


class BatchProcessor:
    """
    Batch processor for integrating BatchService with video processing workflow.
    
    This class handles:
    - Processing individual batch items through the video workflow
    - Progress tracking and session management
    - Error handling and retry logic
    - Integration with existing video processing services
    """

    def __init__(self, 
                 batch_service: Optional[BatchService] = None,
                 video_service: Optional[VideoService] = None,
                 workflow_config: Optional[WorkflowConfig] = None):
        """
        Initialize the batch processor.
        
        Args:
            batch_service: BatchService instance
            video_service: VideoService instance
            workflow_config: Configuration for the video processing workflow
        """
        self.batch_service = batch_service
        self.video_service = video_service
        self.workflow_config = workflow_config or WorkflowConfig()
        self._logger = logging.getLogger(f"{__name__}.BatchProcessor")
        self._active_workers = {}  # Track active workers

    async def process_batch_item(self, batch_item_id: int, worker_id: str) -> BatchItemResult:
        """
        Process a single batch item through the video processing workflow.
        
        Args:
            batch_item_id: ID of the batch item to process
            worker_id: Identifier of the worker processing this item
            
        Returns:
            BatchItemResult containing the processing results
            
        Raises:
            BatchProcessorError: If processing fails
        """
        try:
            # Get database session
            async with self._get_database_session() as session:
                batch_service = self.batch_service or BatchService(session)
                video_service = self.video_service or VideoService(session)
                
                # Get batch item details
                batch_item = session.get(BatchItem, batch_item_id)
                if not batch_item:
                    raise BatchProcessorError(f"Batch item {batch_item_id} not found")
                
                # Extract video ID from URL
                video_id = extract_video_id_from_url(batch_item.url)
                if not video_id:
                    return BatchItemResult(
                        batch_item_id=batch_item_id,
                        status=BatchItemStatus.FAILED,
                        error_message=f"Invalid YouTube URL: {batch_item.url}"
                    )
                
                # Create processing session
                processing_session = batch_service.create_processing_session(
                    batch_item_id, worker_id
                )
                
                self._logger.info(f"Starting batch item {batch_item_id} processing for video {video_id}")
                
                try:
                    # Initialize the workflow
                    workflow = YouTubeSummarizerFlow(config=self.workflow_config)
                    
                    # Process the video through the workflow
                    result = await self._process_video_workflow(
                        workflow, batch_item.url, video_id, processing_session.session_id
                    )
                    
                    # Save results to database
                    video_record = None
                    if result.get('success'):
                        video_record = await self._save_video_results(
                            video_service, video_id, result, batch_item.url
                        )
                    
                    # Update processing session
                    batch_service.update_processing_session(
                        processing_session.session_id,
                        100.0,
                        "Completed"
                    )
                    
                    # Return success result
                    return BatchItemResult(
                        batch_item_id=batch_item_id,
                        status=BatchItemStatus.COMPLETED,
                        video_id=video_record.id if video_record else None,
                        result_data=result
                    )
                    
                except Exception as e:
                    self._logger.error(f"Error processing batch item {batch_item_id}: {e}")
                    
                    # Update processing session with error
                    batch_service.update_processing_session(
                        processing_session.session_id,
                        100.0,
                        f"Failed: {str(e)}"
                    )
                    
                    return BatchItemResult(
                        batch_item_id=batch_item_id,
                        status=BatchItemStatus.FAILED,
                        error_message=str(e)
                    )
                
        except Exception as e:
            self._logger.error(f"Unexpected error processing batch item {batch_item_id}: {e}")
            return BatchItemResult(
                batch_item_id=batch_item_id,
                status=BatchItemStatus.FAILED,
                error_message=f"Unexpected error: {str(e)}"
            )

    async def _process_video_workflow(self, 
                                    workflow: YouTubeSummarizerFlow,
                                    url: str,
                                    video_id: str,
                                    session_id: str) -> Dict[str, Any]:
        """
        Process video through the workflow with progress tracking.
        
        Args:
            workflow: YouTube summarizer workflow instance
            url: YouTube video URL
            video_id: Extracted video ID
            session_id: Processing session ID for progress tracking
            
        Returns:
            Dictionary containing processing results
        """
        try:
            # Update progress: Starting workflow
            if self.batch_service:
                self.batch_service.update_processing_session(
                    session_id, 5.0, "Initializing workflow"
                )
            
            # Execute the workflow
            workflow_result = await workflow.execute({
                'url': url,
                'video_id': video_id,
                'batch_processing': True,
                'session_id': session_id
            })
            
            # Update progress: Workflow completed
            if self.batch_service:
                self.batch_service.update_processing_session(
                    session_id, 95.0, "Workflow completed, saving results"
                )
            
            return {
                'success': True,
                'workflow_result': workflow_result,
                'video_id': video_id,
                'url': url,
                'processed_at': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self._logger.error(f"Workflow execution failed for video {video_id}: {e}")
            return {
                'success': False,
                'error': str(e),
                'video_id': video_id,
                'url': url
            }

    async def _save_video_results(self, 
                                video_service: VideoService,
                                video_id: str,
                                result: Dict[str, Any],
                                url: str) -> Optional[Any]:
        """
        Save video processing results to database.
        
        Args:
            video_service: VideoService instance
            video_id: YouTube video ID
            result: Processing results
            url: YouTube video URL
            
        Returns:
            Created video record or None if saving failed
        """
        try:
            workflow_result = result.get('workflow_result', {})
            
            # Extract data from workflow result
            video_data = workflow_result.get('video_data', {})
            transcript_data = workflow_result.get('transcript_data', {})
            summary_data = workflow_result.get('summary_data', {})
            
            # Create video record
            video_record = video_service.create_video_record(
                video_id=video_id,
                title=video_data.get('title', 'Unknown Title'),
                duration=video_data.get('duration'),
                url=url,
                channel_name=video_data.get('channel_name'),
                description=video_data.get('description'),
                published_date=video_data.get('published_date'),
                view_count=video_data.get('view_count'),
                transcript_content=transcript_data.get('content'),
                transcript_language=transcript_data.get('language')
            )
            
            if video_record:
                # Save summary if available
                if summary_data.get('content'):
                    video_service.save_summary(
                        video_record.id,
                        summary_data['content'],
                        summary_data.get('language', 'unknown'),
                        summary_data.get('model_used', 'unknown')
                    )
                
                # Save keywords if available
                keywords = workflow_result.get('keywords', [])
                if keywords:
                    video_service.save_keywords(video_record.id, keywords)
                
                # Save timestamped segments if available
                segments = workflow_result.get('segments', [])
                if segments:
                    video_service.save_timestamped_segments(video_record.id, segments)
                
                self._logger.info(f"Saved video results for {video_id}")
                return video_record
            
        except Exception as e:
            self._logger.error(f"Error saving video results for {video_id}: {e}")
            return None

    async def start_batch_worker(self, worker_id: str, queue_name: str = 'video_processing'):
        """
        Start a batch processing worker.
        
        Args:
            worker_id: Unique identifier for the worker
            queue_name: Name of the queue to process
        """
        self._logger.info(f"Starting batch worker {worker_id} for queue {queue_name}")
        self._active_workers[worker_id] = {
            'started_at': datetime.utcnow(),
            'queue_name': queue_name,
            'processed_items': 0,
            'failed_items': 0
        }
        
        try:
            while worker_id in self._active_workers:
                async with self._get_database_session() as session:
                    batch_service = self.batch_service or BatchService(session)
                    
                    # Get next queue item
                    queue_item = batch_service.get_next_queue_item(
                        queue_name=queue_name,
                        worker_id=worker_id
                    )
                    
                    if queue_item:
                        # Process the item
                        self._logger.info(f"Worker {worker_id} processing item {queue_item.batch_item_id}")
                        
                        result = await self.process_batch_item(
                            queue_item.batch_item_id,
                            worker_id
                        )
                        
                        # Complete the item
                        batch_service.complete_batch_item(queue_item.batch_item_id, result)
                        
                        # Update worker stats
                        if result.status == BatchItemStatus.COMPLETED:
                            self._active_workers[worker_id]['processed_items'] += 1
                        else:
                            self._active_workers[worker_id]['failed_items'] += 1
                        
                        self._logger.info(f"Worker {worker_id} completed item {queue_item.batch_item_id}")
                    else:
                        # No items available, wait before checking again
                        await asyncio.sleep(5)
                        
        except Exception as e:
            self._logger.error(f"Error in batch worker {worker_id}: {e}")
        finally:
            # Clean up worker
            if worker_id in self._active_workers:
                del self._active_workers[worker_id]
            self._logger.info(f"Batch worker {worker_id} stopped")

    async def stop_batch_worker(self, worker_id: str):
        """
        Stop a batch processing worker.
        
        Args:
            worker_id: Identifier of the worker to stop
        """
        if worker_id in self._active_workers:
            del self._active_workers[worker_id]
            self._logger.info(f"Stopping batch worker {worker_id}")

    def get_worker_stats(self) -> Dict[str, Any]:
        """
        Get statistics for all active workers.
        
        Returns:
            Dictionary containing worker statistics
        """
        stats = {
            'active_workers': len(self._active_workers),
            'workers': {}
        }
        
        for worker_id, worker_info in self._active_workers.items():
            stats['workers'][worker_id] = {
                'started_at': worker_info['started_at'].isoformat(),
                'queue_name': worker_info['queue_name'],
                'processed_items': worker_info['processed_items'],
                'failed_items': worker_info['failed_items'],
                'uptime_seconds': (datetime.utcnow() - worker_info['started_at']).total_seconds()
            }
        
        return stats

    async def process_batch_by_id(self, batch_id: str, num_workers: int = 1) -> Dict[str, Any]:
        """
        Process an entire batch using multiple workers.
        
        Args:
            batch_id: Batch identifier
            num_workers: Number of workers to use
            
        Returns:
            Dictionary containing processing results
        """
        try:
            async with self._get_database_session() as session:
                batch_service = self.batch_service or BatchService(session)
                
                # Get batch info
                batch = batch_service.get_batch(batch_id)
                if not batch:
                    raise BatchProcessorError(f"Batch {batch_id} not found")
                
                # Start batch processing
                batch_service.start_batch_processing(batch_id)
                
                # Create worker tasks
                worker_tasks = []
                for i in range(num_workers):
                    worker_id = f"batch_worker_{batch_id}_{i}"
                    task = asyncio.create_task(
                        self.start_batch_worker(worker_id, 'video_processing')
                    )
                    worker_tasks.append((worker_id, task))
                
                self._logger.info(f"Started {num_workers} workers for batch {batch_id}")
                
                # Monitor batch progress
                while True:
                    progress = batch_service.get_batch_progress(batch_id)
                    if not progress:
                        break
                    
                    if progress.status in [BatchStatus.COMPLETED, BatchStatus.CANCELLED, BatchStatus.FAILED]:
                        break
                    
                    self._logger.info(f"Batch {batch_id} progress: {progress.progress_percentage:.1f}%")
                    await asyncio.sleep(10)
                
                # Stop all workers
                for worker_id, task in worker_tasks:
                    await self.stop_batch_worker(worker_id)
                    task.cancel()
                
                # Get final results
                final_progress = batch_service.get_batch_progress(batch_id)
                
                return {
                    'batch_id': batch_id,
                    'status': final_progress.status.value if final_progress else 'unknown',
                    'total_items': final_progress.total_items if final_progress else 0,
                    'completed_items': final_progress.completed_items if final_progress else 0,
                    'failed_items': final_progress.failed_items if final_progress else 0,
                    'progress_percentage': final_progress.progress_percentage if final_progress else 0,
                    'num_workers_used': num_workers
                }
                
        except Exception as e:
            self._logger.error(f"Error processing batch {batch_id}: {e}")
            raise BatchProcessorError(f"Failed to process batch: {e}")

    @asynccontextmanager
    async def _get_database_session(self):
        """Get database session as async context manager."""
        session = get_database_session()
        try:
            yield session
        finally:
            session.close()


# Factory function for creating batch processor
def create_batch_processor(workflow_config: Optional[WorkflowConfig] = None) -> BatchProcessor:
    """
    Create a BatchProcessor instance.
    
    Args:
        workflow_config: Optional workflow configuration
        
    Returns:
        BatchProcessor instance
    """
    return BatchProcessor(workflow_config=workflow_config)


# Example usage function
async def example_batch_processing():
    """
    Example of how to use the batch processor.
    """
    # Create batch processor
    processor = create_batch_processor()
    
    # Example URLs
    urls = [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://www.youtube.com/watch?v=9bZkp7q19f0",
        "https://youtu.be/oHg5SJYRHA0"
    ]
    
    # Create batch
    async with processor._get_database_session() as session:
        batch_service = BatchService(session)
        
        from .batch_service import BatchCreateRequest, BatchPriority
        
        request = BatchCreateRequest(
            name="Example Batch",
            description="Example batch processing",
            urls=urls,
            priority=BatchPriority.NORMAL
        )
        
        batch = batch_service.create_batch(request)
        
        # Process batch
        results = await processor.process_batch_by_id(batch.batch_id, num_workers=2)
        
        print(f"Batch processing completed: {results}")


if __name__ == "__main__":
    # Run example
    asyncio.run(example_batch_processing())