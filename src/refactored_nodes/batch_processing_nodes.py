"""
Batch processing nodes for PocketFlow workflow integration.

This module provides PocketFlow nodes that enable batch processing capabilities
within the existing YouTube video summarization workflow system.
"""

import logging
import uuid
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field

from .validation_nodes import BaseProcessingNode
from ..services.batch_service import BatchService, BatchCreateRequest, BatchProgressInfo, BatchItemResult
from ..services.queue_service import QueueService, QueueProcessingOptions, WorkerInfo, QueueWorkerStatus
from ..database.batch_models import BatchStatus, BatchItemStatus, BatchPriority
from ..database.connection import get_database_session
from ..flow import YouTubeSummarizerFlow, WorkflowConfig
from ..utils.validators import extract_youtube_video_id as extract_video_id_from_url

logger = logging.getLogger(__name__)


@dataclass
class BatchProcessingConfig:
    """Configuration for batch processing operations."""
    max_workers: int = 5
    worker_timeout_minutes: int = 30
    batch_timeout_minutes: int = 120
    enable_progress_tracking: bool = True
    enable_webhook_notifications: bool = False
    default_priority: BatchPriority = BatchPriority.NORMAL
    enable_concurrent_processing: bool = True
    max_concurrent_batches: int = 3
    enable_retry_failed_items: bool = True
    max_item_retries: int = 3


class BatchCreationNode(BaseProcessingNode):
    """
    Node for creating and initializing batch processing jobs.
    
    This node handles the creation of batch processing jobs from input URLs,
    validates the input data, and sets up the initial batch structure.
    """
    
    def __init__(self, 
                 max_retries: int = 3, 
                 retry_delay: float = 1.0,
                 config: Optional[BatchProcessingConfig] = None):
        super().__init__("BatchCreationNode", max_retries, retry_delay)
        self.config = config or BatchProcessingConfig()
        self.batch_service = None
        
    def prep(self, store: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare for batch creation by validating input URLs and configuration.
        
        Args:
            store: Data store containing 'batch_urls' and optional 'batch_config'
            
        Returns:
            Dict containing prep results and validation status
        """
        self.logger.info("Starting batch creation preparation")
        
        try:
            # Validate required inputs
            required_fields = ['batch_urls']
            if 'batch_urls' not in store:
                raise ValueError("Missing required field: batch_urls")
            
            batch_urls = store['batch_urls']
            if not isinstance(batch_urls, list) or not batch_urls:
                raise ValueError("batch_urls must be a non-empty list")
                
            # Validate URLs
            validated_urls = []
            invalid_urls = []
            
            for url in batch_urls:
                if not isinstance(url, str) or not url.strip():
                    invalid_urls.append(url)
                    continue
                    
                try:
                    video_id = extract_video_id_from_url(url)
                    if not video_id:
                        invalid_urls.append(url)
                    else:
                        validated_urls.append(url.strip())
                except Exception as e:
                    invalid_urls.append(url)
            
            if not validated_urls:
                raise ValueError("No valid YouTube URLs found in batch_urls")
            
            # Extract batch configuration
            batch_config = store.get('batch_config', {})
            batch_name = batch_config.get('name', f"Batch_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}")
            batch_description = batch_config.get('description', f"Batch processing of {len(validated_urls)} videos")
            priority = BatchPriority(batch_config.get('priority', self.config.default_priority.value))
            webhook_url = batch_config.get('webhook_url')
            
            prep_result = {
                'validated_urls': validated_urls,
                'invalid_urls': invalid_urls,
                'batch_name': batch_name,
                'batch_description': batch_description,
                'priority': priority,
                'webhook_url': webhook_url,
                'batch_metadata': batch_config.get('metadata', {}),
                'total_valid_urls': len(validated_urls),
                'total_invalid_urls': len(invalid_urls),
                'validation_status': 'success',
                'prep_timestamp': datetime.utcnow().isoformat()
            }
            
            self.logger.info(f"Batch creation preparation successful: {len(validated_urls)} valid URLs, {len(invalid_urls)} invalid URLs")
            return prep_result
            
        except Exception as e:
            error_info = self._handle_error(e, "Batch creation preparation failed")
            prep_result = {
                'validation_status': 'failed',
                'error': {
                    'type': error_info.error_type,
                    'message': error_info.message,
                    'timestamp': error_info.timestamp,
                    'is_recoverable': error_info.is_recoverable
                },
                'prep_timestamp': datetime.utcnow().isoformat()
            }
            
            return prep_result
    
    def exec(self, store: Dict[str, Any], prep_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute batch creation in the database.
        
        Args:
            store: Data store
            prep_result: Results from preparation phase
            
        Returns:
            Dict containing batch creation results
        """
        self.logger.info("Starting batch creation execution")
        
        try:
            # Check if preparation was successful
            if prep_result.get('validation_status') != 'success':
                raise ValueError("Cannot execute batch creation: preparation failed")
            
            # Initialize batch service
            with get_database_session() as session:
                self.batch_service = BatchService(session)
                
                # Create batch request
                batch_request = BatchCreateRequest(
                    name=prep_result['batch_name'],
                    description=prep_result['batch_description'],
                    urls=prep_result['validated_urls'],
                    priority=prep_result['priority'],
                    webhook_url=prep_result['webhook_url'],
                    batch_metadata=prep_result['batch_metadata']
                )
                
                # Create the batch
                batch = self.batch_service.create_batch(batch_request)
                
                exec_result = {
                    'batch_id': batch.batch_id,
                    'batch_db_id': batch.id,
                    'batch_status': batch.status.value,
                    'total_items': batch.total_items,
                    'created_at': batch.created_at.isoformat(),
                    'batch_metadata': {
                        'name': batch.name,
                        'description': batch.description,
                        'priority': batch.priority.value,
                        'webhook_url': batch.webhook_url
                    },
                    'execution_status': 'success',
                    'execution_timestamp': datetime.utcnow().isoformat()
                }
                
                self.logger.info(f"Batch creation successful: {batch.batch_id} with {batch.total_items} items")
                return exec_result
                
        except Exception as e:
            error_info = self._handle_error(e, "Batch creation execution failed")
            exec_result = {
                'execution_status': 'failed',
                'error': {
                    'type': error_info.error_type,
                    'message': error_info.message,
                    'timestamp': error_info.timestamp,
                    'is_recoverable': error_info.is_recoverable
                },
                'execution_timestamp': datetime.utcnow().isoformat()
            }
            
            return exec_result
    
    def post(self, store: Dict[str, Any], prep_result: Dict[str, Any], exec_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Post-process batch creation and update store.
        
        Args:
            store: Data store
            prep_result: Results from preparation phase
            exec_result: Results from execution phase
            
        Returns:
            Dict containing final processing results
        """
        self.logger.info("Starting batch creation post-processing")
        
        try:
            if exec_result.get('execution_status') == 'success':
                # Update store with batch information
                store_updates = {
                    'batch_id': exec_result['batch_id'],
                    'batch_db_id': exec_result['batch_db_id'],
                    'batch_status': exec_result['batch_status'],
                    'batch_total_items': exec_result['total_items'],
                    'batch_created_at': exec_result['created_at'],
                    'batch_metadata': exec_result['batch_metadata'],
                    'batch_urls': prep_result['validated_urls'],
                    'batch_invalid_urls': prep_result['invalid_urls']
                }
                
                post_result = {
                    'processing_status': 'success',
                    'batch_id': exec_result['batch_id'],
                    'batch_created': True,
                    'total_items': exec_result['total_items'],
                    'store_updated': True,
                    'post_timestamp': datetime.utcnow().isoformat()
                }
                
            else:
                # Handle failed execution
                store_updates = {
                    'batch_creation_error': exec_result.get('error', {}),
                    'batch_urls': prep_result.get('validated_urls', []),
                    'batch_invalid_urls': prep_result.get('invalid_urls', [])
                }
                
                post_result = {
                    'processing_status': 'failed',
                    'batch_created': False,
                    'store_updated': True,
                    'post_timestamp': datetime.utcnow().isoformat(),
                    'error': exec_result.get('error', {})
                }
            
            # Apply store updates
            self._safe_store_update(store, store_updates)
            
            self.logger.info(f"Batch creation post-processing completed: {post_result['processing_status']}")
            return post_result
            
        except Exception as e:
            error_info = self._handle_error(e, "Batch creation post-processing failed")
            post_result = {
                'processing_status': 'failed',
                'batch_created': False,
                'store_updated': False,
                'post_timestamp': datetime.utcnow().isoformat(),
                'error': {
                    'type': error_info.error_type,
                    'message': error_info.message,
                    'timestamp': error_info.timestamp,
                    'is_recoverable': error_info.is_recoverable
                }
            }
            
            return post_result


class BatchProcessingNode(BaseProcessingNode):
    """
    Node for executing batch processing using queue workers.
    
    This node coordinates the processing of batch items through the queue
    system, managing workers and monitoring progress.
    """
    
    def __init__(self, 
                 max_retries: int = 3, 
                 retry_delay: float = 1.0,
                 config: Optional[BatchProcessingConfig] = None):
        super().__init__("BatchProcessingNode", max_retries, retry_delay)
        self.config = config or BatchProcessingConfig()
        self.batch_service = None
        self.queue_service = None
        self.workers = []
        
    def prep(self, store: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare for batch processing by validating batch existence and configuration.
        
        Args:
            store: Data store containing 'batch_id'
            
        Returns:
            Dict containing prep results and validation status
        """
        self.logger.info("Starting batch processing preparation")
        
        try:
            # Validate required inputs
            if 'batch_id' not in store:
                raise ValueError("Missing required field: batch_id")
            
            batch_id = store['batch_id']
            
            # Initialize services
            with get_database_session() as session:
                self.batch_service = BatchService(session)
                
                # Get batch information
                batch = self.batch_service.get_batch(batch_id)
                if not batch:
                    raise ValueError(f"Batch {batch_id} not found")
                
                if batch.status != BatchStatus.PENDING:
                    raise ValueError(f"Batch {batch_id} is not in PENDING status (current: {batch.status.value})")
                
                prep_result = {
                    'batch_id': batch_id,
                    'batch_db_id': batch.id,
                    'batch_status': batch.status.value,
                    'total_items': batch.total_items,
                    'batch_name': batch.name,
                    'batch_priority': batch.priority.value,
                    'processing_config': {
                        'max_workers': self.config.max_workers,
                        'worker_timeout_minutes': self.config.worker_timeout_minutes,
                        'enable_progress_tracking': self.config.enable_progress_tracking,
                        'enable_concurrent_processing': self.config.enable_concurrent_processing
                    },
                    'validation_status': 'success',
                    'prep_timestamp': datetime.utcnow().isoformat()
                }
                
                self.logger.info(f"Batch processing preparation successful for batch {batch_id}")
                return prep_result
                
        except Exception as e:
            error_info = self._handle_error(e, "Batch processing preparation failed")
            prep_result = {
                'validation_status': 'failed',
                'error': {
                    'type': error_info.error_type,
                    'message': error_info.message,
                    'timestamp': error_info.timestamp,
                    'is_recoverable': error_info.is_recoverable
                },
                'prep_timestamp': datetime.utcnow().isoformat()
            }
            
            return prep_result
    
    def exec(self, store: Dict[str, Any], prep_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute batch processing using queue workers.
        
        Args:
            store: Data store
            prep_result: Results from preparation phase
            
        Returns:
            Dict containing batch processing results
        """
        self.logger.info("Starting batch processing execution")
        
        try:
            # Check if preparation was successful
            if prep_result.get('validation_status') != 'success':
                raise ValueError("Cannot execute batch processing: preparation failed")
            
            batch_id = prep_result['batch_id']
            
            with get_database_session() as session:
                self.batch_service = BatchService(session)
                self.queue_service = QueueService(
                    session, 
                    QueueProcessingOptions(
                        max_workers=self.config.max_workers,
                        worker_timeout_minutes=self.config.worker_timeout_minutes,
                        enable_priority_processing=True
                    )
                )
                
                # Start batch processing
                batch_started = self.batch_service.start_batch_processing(batch_id)
                if not batch_started:
                    raise ValueError(f"Failed to start batch processing for {batch_id}")
                
                # Initialize workers if concurrent processing is enabled
                if self.config.enable_concurrent_processing:
                    self._initialize_workers(batch_id, prep_result['processing_config'])
                
                # Process batch items
                processing_results = self._process_batch_items(batch_id, prep_result)
                
                exec_result = {
                    'batch_id': batch_id,
                    'processing_started': True,
                    'workers_initialized': self.config.enable_concurrent_processing,
                    'worker_count': len(self.workers),
                    'processing_results': processing_results,
                    'execution_status': 'success',
                    'execution_timestamp': datetime.utcnow().isoformat()
                }
                
                self.logger.info(f"Batch processing execution successful for batch {batch_id}")
                return exec_result
                
        except Exception as e:
            error_info = self._handle_error(e, "Batch processing execution failed")
            exec_result = {
                'execution_status': 'failed',
                'error': {
                    'type': error_info.error_type,
                    'message': error_info.message,
                    'timestamp': error_info.timestamp,
                    'is_recoverable': error_info.is_recoverable
                },
                'execution_timestamp': datetime.utcnow().isoformat()
            }
            
            return exec_result
    
    def post(self, store: Dict[str, Any], prep_result: Dict[str, Any], exec_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Post-process batch processing results and update store.
        
        Args:
            store: Data store
            prep_result: Results from preparation phase
            exec_result: Results from execution phase
            
        Returns:
            Dict containing final processing results
        """
        self.logger.info("Starting batch processing post-processing")
        
        try:
            if exec_result.get('execution_status') == 'success':
                # Update store with processing results
                store_updates = {
                    'batch_processing_started': exec_result['processing_started'],
                    'batch_worker_count': exec_result['worker_count'],
                    'batch_processing_results': exec_result['processing_results'],
                    'batch_processing_timestamp': exec_result['execution_timestamp']
                }
                
                # Get final batch progress
                batch_progress = self._get_batch_progress(prep_result['batch_id'])
                store_updates['batch_progress'] = batch_progress
                
                post_result = {
                    'processing_status': 'success',
                    'batch_id': prep_result['batch_id'],
                    'processing_completed': True,
                    'final_progress': batch_progress,
                    'store_updated': True,
                    'post_timestamp': datetime.utcnow().isoformat()
                }
                
            else:
                # Handle failed execution
                store_updates = {
                    'batch_processing_error': exec_result.get('error', {}),
                    'batch_processing_started': False
                }
                
                post_result = {
                    'processing_status': 'failed',
                    'batch_id': prep_result.get('batch_id'),
                    'processing_completed': False,
                    'store_updated': True,
                    'post_timestamp': datetime.utcnow().isoformat(),
                    'error': exec_result.get('error', {})
                }
            
            # Apply store updates
            self._safe_store_update(store, store_updates)
            
            # Cleanup workers
            self._cleanup_workers()
            
            self.logger.info(f"Batch processing post-processing completed: {post_result['processing_status']}")
            return post_result
            
        except Exception as e:
            error_info = self._handle_error(e, "Batch processing post-processing failed")
            post_result = {
                'processing_status': 'failed',
                'processing_completed': False,
                'store_updated': False,
                'post_timestamp': datetime.utcnow().isoformat(),
                'error': {
                    'type': error_info.error_type,
                    'message': error_info.message,
                    'timestamp': error_info.timestamp,
                    'is_recoverable': error_info.is_recoverable
                }
            }
            
            return post_result
    
    def _initialize_workers(self, batch_id: str, processing_config: Dict[str, Any]):
        """Initialize worker processes for batch processing."""
        try:
            max_workers = processing_config.get('max_workers', 3)
            
            for i in range(max_workers):
                worker_info = self.queue_service.register_worker(
                    queue_name='video_processing',
                    worker_id=f"batch_{batch_id}_worker_{i}"
                )
                self.workers.append(worker_info)
                
            self.logger.info(f"Initialized {len(self.workers)} workers for batch {batch_id}")
            
        except Exception as e:
            self.logger.error(f"Error initializing workers: {e}")
    
    def _process_batch_items(self, batch_id: str, prep_result: Dict[str, Any]) -> Dict[str, Any]:
        """Process all items in the batch."""
        try:
            processed_items = 0
            failed_items = 0
            total_items = prep_result['total_items']
            
            # Process items using queue workers
            while processed_items + failed_items < total_items:
                # Get next item from queue
                for worker in self.workers:
                    queue_item = self.queue_service.get_next_queue_item(
                        queue_name='video_processing',
                        worker_id=worker.worker_id
                    )
                    
                    if queue_item:
                        # Process the item
                        result = self._process_single_item(queue_item, worker)
                        
                        if result['status'] == BatchItemStatus.COMPLETED:
                            processed_items += 1
                        else:
                            failed_items += 1
                
                # Small delay to prevent tight loop
                import time
                time.sleep(0.1)
            
            return {
                'processed_items': processed_items,
                'failed_items': failed_items,
                'total_items': total_items,
                'success_rate': processed_items / total_items if total_items > 0 else 0
            }
            
        except Exception as e:
            self.logger.error(f"Error processing batch items: {e}")
            return {
                'processed_items': processed_items,
                'failed_items': failed_items,
                'total_items': total_items,
                'error': str(e)
            }
    
    def _process_single_item(self, queue_item, worker: WorkerInfo) -> Dict[str, Any]:
        """Process a single batch item."""
        try:
            # Get the URL from the batch item
            url = queue_item.batch_item.url
            
            # Create a simplified workflow for single video processing
            workflow = YouTubeSummarizerFlow(
                enable_monitoring=False,
                enable_fallbacks=True,
                max_retries=1,
                timeout_seconds=300
            )
            
            # Process the video
            result = workflow.run({'youtube_url': url})
            
            # Determine status based on workflow result
            if result.get('status') == 'success':
                status = BatchItemStatus.COMPLETED
                error_message = None
                result_data = result.get('data', {})
            else:
                status = BatchItemStatus.FAILED
                error_message = str(result.get('error', 'Unknown error'))
                result_data = None
            
            # Complete the queue item
            self.queue_service.complete_queue_item(
                queue_item_id=queue_item.id,
                worker_id=worker.worker_id,
                status=status,
                result_data=result_data,
                error_message=error_message
            )
            
            return {
                'status': status,
                'result_data': result_data,
                'error_message': error_message
            }
            
        except Exception as e:
            self.logger.error(f"Error processing single item: {e}")
            
            # Mark as failed
            self.queue_service.complete_queue_item(
                queue_item_id=queue_item.id,
                worker_id=worker.worker_id,
                status=BatchItemStatus.FAILED,
                error_message=str(e)
            )
            
            return {
                'status': BatchItemStatus.FAILED,
                'error_message': str(e)
            }
    
    def _get_batch_progress(self, batch_id: str) -> Dict[str, Any]:
        """Get current batch progress."""
        try:
            progress_info = self.batch_service.get_batch_progress(batch_id)
            if progress_info:
                return {
                    'batch_id': progress_info.batch_id,
                    'status': progress_info.status.value,
                    'total_items': progress_info.total_items,
                    'completed_items': progress_info.completed_items,
                    'failed_items': progress_info.failed_items,
                    'pending_items': progress_info.pending_items,
                    'progress_percentage': progress_info.progress_percentage,
                    'started_at': progress_info.started_at.isoformat() if progress_info.started_at else None,
                    'estimated_completion': progress_info.estimated_completion.isoformat() if progress_info.estimated_completion else None
                }
            return {}
            
        except Exception as e:
            self.logger.error(f"Error getting batch progress: {e}")
            return {}
    
    def _cleanup_workers(self):
        """Clean up worker resources."""
        try:
            for worker in self.workers:
                self.queue_service.unregister_worker(worker.worker_id)
            self.workers.clear()
            
        except Exception as e:
            self.logger.error(f"Error cleaning up workers: {e}")


class BatchStatusNode(BaseProcessingNode):
    """
    Node for monitoring and reporting batch processing status.
    
    This node provides status updates, progress tracking, and completion
    notifications for batch processing operations.
    """
    
    def __init__(self, 
                 max_retries: int = 3, 
                 retry_delay: float = 1.0,
                 config: Optional[BatchProcessingConfig] = None):
        super().__init__("BatchStatusNode", max_retries, retry_delay)
        self.config = config or BatchProcessingConfig()
        self.batch_service = None
        
    def prep(self, store: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare for batch status monitoring.
        
        Args:
            store: Data store containing 'batch_id'
            
        Returns:
            Dict containing prep results and validation status
        """
        self.logger.info("Starting batch status monitoring preparation")
        
        try:
            # Validate required inputs
            if 'batch_id' not in store:
                raise ValueError("Missing required field: batch_id")
            
            batch_id = store['batch_id']
            
            prep_result = {
                'batch_id': batch_id,
                'monitoring_enabled': self.config.enable_progress_tracking,
                'webhook_enabled': self.config.enable_webhook_notifications,
                'validation_status': 'success',
                'prep_timestamp': datetime.utcnow().isoformat()
            }
            
            self.logger.info(f"Batch status monitoring preparation successful for batch {batch_id}")
            return prep_result
            
        except Exception as e:
            error_info = self._handle_error(e, "Batch status monitoring preparation failed")
            prep_result = {
                'validation_status': 'failed',
                'error': {
                    'type': error_info.error_type,
                    'message': error_info.message,
                    'timestamp': error_info.timestamp,
                    'is_recoverable': error_info.is_recoverable
                },
                'prep_timestamp': datetime.utcnow().isoformat()
            }
            
            return prep_result
    
    def exec(self, store: Dict[str, Any], prep_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute batch status monitoring and reporting.
        
        Args:
            store: Data store
            prep_result: Results from preparation phase
            
        Returns:
            Dict containing batch status results
        """
        self.logger.info("Starting batch status monitoring execution")
        
        try:
            # Check if preparation was successful
            if prep_result.get('validation_status') != 'success':
                raise ValueError("Cannot execute batch status monitoring: preparation failed")
            
            batch_id = prep_result['batch_id']
            
            with get_database_session() as session:
                self.batch_service = BatchService(session)
                
                # Get comprehensive batch status
                batch = self.batch_service.get_batch(batch_id)
                if not batch:
                    raise ValueError(f"Batch {batch_id} not found")
                
                # Get detailed progress information
                progress_info = self.batch_service.get_batch_progress(batch_id)
                
                # Get batch statistics
                batch_stats = self._get_batch_statistics(batch)
                
                exec_result = {
                    'batch_id': batch_id,
                    'batch_status': batch.status.value,
                    'batch_info': {
                        'name': batch.name,
                        'description': batch.description,
                        'created_at': batch.created_at.isoformat(),
                        'started_at': batch.started_at.isoformat() if batch.started_at else None,
                        'completed_at': batch.completed_at.isoformat() if batch.completed_at else None,
                        'total_items': batch.total_items,
                        'completed_items': batch.completed_items,
                        'failed_items': batch.failed_items,
                        'pending_items': batch.pending_items,
                        'progress_percentage': batch.progress_percentage,
                        'priority': batch.priority.value,
                        'webhook_url': batch.webhook_url
                    },
                    'progress_info': {
                        'estimated_completion': progress_info.estimated_completion.isoformat() if progress_info and progress_info.estimated_completion else None,
                        'processing_rate': self._calculate_processing_rate(batch),
                        'remaining_time_estimate': self._estimate_remaining_time(batch)
                    },
                    'batch_statistics': batch_stats,
                    'execution_status': 'success',
                    'execution_timestamp': datetime.utcnow().isoformat()
                }
                
                self.logger.info(f"Batch status monitoring execution successful for batch {batch_id}")
                return exec_result
                
        except Exception as e:
            error_info = self._handle_error(e, "Batch status monitoring execution failed")
            exec_result = {
                'execution_status': 'failed',
                'error': {
                    'type': error_info.error_type,
                    'message': error_info.message,
                    'timestamp': error_info.timestamp,
                    'is_recoverable': error_info.is_recoverable
                },
                'execution_timestamp': datetime.utcnow().isoformat()
            }
            
            return exec_result
    
    def post(self, store: Dict[str, Any], prep_result: Dict[str, Any], exec_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Post-process batch status monitoring results and update store.
        
        Args:
            store: Data store
            prep_result: Results from preparation phase
            exec_result: Results from execution phase
            
        Returns:
            Dict containing final processing results
        """
        self.logger.info("Starting batch status monitoring post-processing")
        
        try:
            if exec_result.get('execution_status') == 'success':
                # Update store with status information
                store_updates = {
                    'batch_final_status': exec_result['batch_status'],
                    'batch_final_info': exec_result['batch_info'],
                    'batch_progress_info': exec_result['progress_info'],
                    'batch_statistics': exec_result['batch_statistics'],
                    'batch_monitoring_timestamp': exec_result['execution_timestamp']
                }
                
                # Determine if batch is complete
                is_complete = exec_result['batch_status'] in ['COMPLETED', 'CANCELLED']
                
                post_result = {
                    'processing_status': 'success',
                    'batch_id': prep_result['batch_id'],
                    'batch_complete': is_complete,
                    'final_status': exec_result['batch_status'],
                    'final_statistics': exec_result['batch_statistics'],
                    'store_updated': True,
                    'post_timestamp': datetime.utcnow().isoformat()
                }
                
                # Send webhook notification if enabled and batch is complete
                if is_complete and self.config.enable_webhook_notifications:
                    self._send_webhook_notification(exec_result)
                
            else:
                # Handle failed execution
                store_updates = {
                    'batch_status_error': exec_result.get('error', {}),
                    'batch_monitoring_failed': True
                }
                
                post_result = {
                    'processing_status': 'failed',
                    'batch_id': prep_result.get('batch_id'),
                    'batch_complete': False,
                    'store_updated': True,
                    'post_timestamp': datetime.utcnow().isoformat(),
                    'error': exec_result.get('error', {})
                }
            
            # Apply store updates
            self._safe_store_update(store, store_updates)
            
            self.logger.info(f"Batch status monitoring post-processing completed: {post_result['processing_status']}")
            return post_result
            
        except Exception as e:
            error_info = self._handle_error(e, "Batch status monitoring post-processing failed")
            post_result = {
                'processing_status': 'failed',
                'batch_complete': False,
                'store_updated': False,
                'post_timestamp': datetime.utcnow().isoformat(),
                'error': {
                    'type': error_info.error_type,
                    'message': error_info.message,
                    'timestamp': error_info.timestamp,
                    'is_recoverable': error_info.is_recoverable
                }
            }
            
            return post_result
    
    def _get_batch_statistics(self, batch) -> Dict[str, Any]:
        """Get detailed batch statistics."""
        try:
            # Calculate processing time
            processing_time = None
            if batch.started_at and batch.completed_at:
                processing_time = (batch.completed_at - batch.started_at).total_seconds()
            elif batch.started_at:
                processing_time = (datetime.utcnow() - batch.started_at).total_seconds()
            
            # Calculate success rate
            total_processed = batch.completed_items + batch.failed_items
            success_rate = batch.completed_items / total_processed if total_processed > 0 else 0
            
            return {
                'processing_time_seconds': processing_time,
                'success_rate': success_rate,
                'failure_rate': 1.0 - success_rate,
                'completion_percentage': (total_processed / batch.total_items) * 100 if batch.total_items > 0 else 0,
                'items_per_second': total_processed / processing_time if processing_time and processing_time > 0 else 0
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating batch statistics: {e}")
            return {}
    
    def _calculate_processing_rate(self, batch) -> float:
        """Calculate current processing rate (items per second)."""
        try:
            if not batch.started_at:
                return 0.0
            
            elapsed_time = (datetime.utcnow() - batch.started_at).total_seconds()
            processed_items = batch.completed_items + batch.failed_items
            
            return processed_items / elapsed_time if elapsed_time > 0 else 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating processing rate: {e}")
            return 0.0
    
    def _estimate_remaining_time(self, batch) -> Optional[float]:
        """Estimate remaining processing time in seconds."""
        try:
            processing_rate = self._calculate_processing_rate(batch)
            if processing_rate <= 0:
                return None
            
            remaining_items = batch.total_items - (batch.completed_items + batch.failed_items)
            return remaining_items / processing_rate
            
        except Exception as e:
            self.logger.error(f"Error estimating remaining time: {e}")
            return None
    
    def _send_webhook_notification(self, exec_result: Dict[str, Any]):
        """Send webhook notification for batch completion."""
        try:
            webhook_url = exec_result['batch_info'].get('webhook_url')
            if not webhook_url:
                return
            
            # Prepare notification payload
            notification_payload = {
                'batch_id': exec_result['batch_id'],
                'status': exec_result['batch_status'],
                'completion_time': exec_result['execution_timestamp'],
                'statistics': exec_result['batch_statistics'],
                'batch_info': exec_result['batch_info']
            }
            
            # Send webhook (implement actual HTTP request)
            # This would typically use requests library
            self.logger.info(f"Webhook notification prepared for {webhook_url}")
            
        except Exception as e:
            self.logger.error(f"Error sending webhook notification: {e}")


# Export all batch processing nodes
__all__ = [
    'BatchCreationNode',
    'BatchProcessingNode', 
    'BatchStatusNode',
    'BatchProcessingConfig'
]