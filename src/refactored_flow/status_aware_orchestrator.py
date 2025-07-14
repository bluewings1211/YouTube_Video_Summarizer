"""
Status-aware workflow orchestrator that integrates with status tracking system.

This module extends the existing orchestrator with comprehensive status tracking
while maintaining compatibility with the existing workflow system.
"""

import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

from .orchestrator import YouTubeSummarizerFlow, YouTubeBatchProcessingFlow
from ..services.status_integration import StatusTrackingMixin, WorkflowStatusManager
from ..services.status_service import StatusService
from ..database.status_models import ProcessingStatusType, ProcessingPriority
from ..database.connection import get_db_session


logger = logging.getLogger(__name__)


class StatusAwareYouTubeSummarizerFlow(StatusTrackingMixin, YouTubeSummarizerFlow):
    """
    Status-aware YouTube summarizer workflow that tracks processing status
    throughout the entire workflow execution.
    """
    
    def __init__(self, *args, **kwargs):
        # Initialize status tracking options
        self.enable_status_tracking = kwargs.pop('enable_status_tracking', True)
        self.status_priority = kwargs.pop('status_priority', ProcessingPriority.NORMAL)
        
        # Initialize parent classes
        super().__init__(*args, **kwargs)
        
        # Initialize workflow status manager
        self.workflow_status_manager: Optional[WorkflowStatusManager] = None
        if self.enable_status_tracking:
            try:
                self.workflow_status_manager = WorkflowStatusManager(
                    workflow_name="YouTubeSummarizerFlow"
                )
            except Exception as e:
                logger.warning(f"Failed to initialize workflow status manager: {e}")
                self.enable_status_tracking = False
    
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the complete workflow with comprehensive status tracking.
        
        Args:
            input_data: Dictionary containing 'youtube_url' and optional configuration
            
        Returns:
            Dictionary containing all processing results with status tracking information
        """
        if not self.enable_status_tracking:
            # Fall back to parent implementation
            return super().run(input_data)
        
        # Extract video metadata for status tracking
        video_id = self._extract_video_id_for_status(input_data)
        youtube_url = input_data.get('youtube_url', '')
        
        try:
            # Start workflow-level status tracking
            workflow_status_id = self.workflow_status_manager.start_workflow(
                video_id=video_id,
                node_names=['YouTubeDataNode', 'SummarizationNode', 'TimestampNode', 'KeywordExtractionNode'],
                priority=self.status_priority
            )
            
            # Store status ID for access by other components
            input_data['workflow_status_id'] = workflow_status_id
            
            logger.info(f"Started status-aware workflow execution for {youtube_url}")
            
            # Execute the parent workflow with status tracking context
            with self._status_tracking_context(
                operation_name="complete_workflow",
                total_steps=4,  # Number of main processing nodes
                video_id=video_id,
                auto_complete=False  # We'll handle completion manually
            ) as status_id:
                
                # Update workflow status to indicate main processing
                self._update_status(
                    ProcessingStatusType.STARTING,
                    current_step="Starting main workflow execution",
                    change_reason="Beginning video processing workflow"
                )
                
                # Execute parent workflow
                result = super().run(input_data)
                
                # Determine final status based on result
                if result.get('status') == 'success':
                    # Mark workflow as completed
                    self.workflow_status_manager.finish_workflow(success=True)
                    self._update_status(
                        ProcessingStatusType.COMPLETED,
                        progress_percentage=100.0,
                        current_step="Workflow completed successfully",
                        change_reason="All processing steps completed successfully"
                    )
                else:
                    # Mark workflow as failed
                    error_info = str(result.get('error', 'Unknown workflow error'))
                    self.workflow_status_manager.finish_workflow(success=False, error_info=error_info)
                    self._record_error(
                        error_info=error_info,
                        error_metadata={'workflow_result': result.get('status', 'unknown')},
                        should_retry=False
                    )
                
                # Add status tracking information to result
                result['status_tracking'] = {
                    'enabled': True,
                    'workflow_status_id': workflow_status_id,
                    'status_id': status_id,
                    'node_status_mapping': self.workflow_status_manager.node_status_mapping.copy()
                }
                
                return result
        
        except Exception as e:
            # Handle workflow error with status tracking
            error_info = f"Workflow execution failed: {str(e)}"
            
            if self.workflow_status_manager:
                self.workflow_status_manager.finish_workflow(success=False, error_info=error_info)
            
            self._record_error(
                error_info=error_info,
                error_metadata={'error_type': type(e).__name__, 'youtube_url': youtube_url},
                should_retry=False
            )
            
            logger.error(f"Status-aware workflow failed: {error_info}")
            raise
    
    def _execute_single_node(self, node_name: str) -> Dict[str, Any]:
        """Execute a single node with comprehensive status tracking."""
        if not self.enable_status_tracking:
            return super()._execute_single_node(node_name)
        
        try:
            # Start node-level status tracking
            video_id = self._extract_video_id_from_store()
            node_status_id = self.workflow_status_manager.start_node(node_name, video_id)
            
            logger.info(f"Starting status-aware execution of {node_name}")
            
            # Execute node with status updates
            result = self._execute_node_with_status_tracking(node_name)
            
            # Mark node as completed
            self.workflow_status_manager.finish_node(node_name, success=True)
            
            return result
        
        except Exception as e:
            # Mark node as failed
            error_info = f"Node {node_name} failed: {str(e)}"
            self.workflow_status_manager.finish_node(node_name, success=False, error_info=error_info)
            
            logger.error(f"Status-aware node execution failed: {error_info}")
            raise
    
    def _execute_node_with_status_tracking(self, node_name: str) -> Dict[str, Any]:
        """Execute a node with detailed status tracking for each phase."""
        node = self.node_instances.get(node_name)
        if not node:
            raise ValueError(f"Node not found: {node_name}")
        
        node_status_id = self.workflow_status_manager.get_node_status_id(node_name)
        
        try:
            # Prep phase with status tracking
            logger.debug(f"Starting prep phase for {node_name}")
            self._update_node_status(node_status_id, "prep", node_name, 0)
            
            prep_result = node.prep(self.store)
            
            self._update_node_status(node_status_id, "prep_completed", node_name, 33)
            
            # Exec phase with status tracking
            logger.debug(f"Starting exec phase for {node_name}")
            self._update_node_status(node_status_id, "exec", node_name, 33)
            
            exec_result = node.exec(self.store, prep_result)
            
            self._update_node_status(node_status_id, "exec_completed", node_name, 66)
            
            # Post phase with status tracking
            logger.debug(f"Starting post phase for {node_name}")
            self._update_node_status(node_status_id, "post", node_name, 66)
            
            post_result = node.post(self.store, prep_result, exec_result)
            
            self._update_node_status(node_status_id, "post_completed", node_name, 100)
            
            return {
                'prep_result': prep_result,
                'exec_result': exec_result,
                'post_result': post_result,
                'node_status': 'success',
                'status_tracking': {
                    'node_status_id': node_status_id,
                    'phases_completed': ['prep', 'exec', 'post']
                }
            }
        
        except Exception as e:
            # Record error for the specific node
            self._record_node_error(node_status_id, node_name, str(e))
            raise
    
    def _update_node_status(self, node_status_id: str, phase: str, node_name: str, progress: float):
        """Update status for a specific node phase."""
        if not node_status_id:
            return
        
        try:
            # Map phases to more descriptive steps
            phase_descriptions = {
                'prep': f'Preparing {node_name}',
                'prep_completed': f'Preparation completed for {node_name}',
                'exec': f'Executing {node_name}',
                'exec_completed': f'Execution completed for {node_name}',
                'post': f'Post-processing {node_name}',
                'post_completed': f'Post-processing completed for {node_name}'
            }
            
            current_step = phase_descriptions.get(phase, f'{phase} {node_name}')
            
            # Update node status
            with get_db_session() as session:
                status_service = StatusService(session)
                status_service.update_progress(
                    status_id=node_status_id,
                    progress_percentage=progress,
                    current_step=current_step,
                    processing_metadata={
                        'phase': phase,
                        'node_name': node_name,
                        'timestamp': datetime.utcnow().isoformat()
                    }
                )
        
        except Exception as e:
            logger.error(f"Failed to update node status for {node_name}: {e}")
    
    def _record_node_error(self, node_status_id: str, node_name: str, error_info: str):
        """Record error for a specific node."""
        if not node_status_id:
            return
        
        try:
            with get_db_session() as session:
                status_service = StatusService(session)
                status_service.record_error(
                    status_id=node_status_id,
                    error_info=error_info,
                    error_metadata={
                        'node_name': node_name,
                        'error_timestamp': datetime.utcnow().isoformat()
                    },
                    should_retry=True
                )
        
        except Exception as e:
            logger.error(f"Failed to record node error for {node_name}: {e}")
    
    def _extract_video_id_for_status(self, input_data: Dict[str, Any]) -> Optional[int]:
        """Extract video ID from input data for status tracking."""
        try:
            # Try to get from existing video_id field
            if 'video_id' in input_data:
                return input_data['video_id']
            
            # Try to extract from YouTube URL
            youtube_url = input_data.get('youtube_url', '')
            extracted_id = self._extract_video_id(youtube_url)
            
            if extracted_id:
                # For status tracking, we may need the database ID, not the YouTube ID
                # This would require a lookup, but for now we'll use None and rely on
                # other identifying information
                return None
            
        except Exception as e:
            logger.warning(f"Failed to extract video ID for status tracking: {e}")
        
        return None
    
    def _extract_video_id_from_store(self) -> Optional[int]:
        """Extract video ID from workflow store."""
        try:
            if hasattr(self, 'store') and self.store:
                return self.store.get('video_id') or self.store.get('database_video_id')
        except Exception as e:
            logger.warning(f"Failed to extract video ID from store: {e}")
        
        return None


class StatusAwareYouTubeBatchProcessingFlow(StatusTrackingMixin, YouTubeBatchProcessingFlow):
    """
    Status-aware YouTube batch processing workflow that tracks processing status
    for batch operations and individual batch items.
    """
    
    def __init__(self, *args, **kwargs):
        # Initialize status tracking options
        self.enable_status_tracking = kwargs.pop('enable_status_tracking', True)
        self.status_priority = kwargs.pop('status_priority', ProcessingPriority.NORMAL)
        
        # Initialize parent classes
        super().__init__(*args, **kwargs)
        
        # Initialize workflow status manager
        self.workflow_status_manager: Optional[WorkflowStatusManager] = None
        if self.enable_status_tracking:
            try:
                self.workflow_status_manager = WorkflowStatusManager(
                    workflow_name="YouTubeBatchProcessingFlow"
                )
            except Exception as e:
                logger.warning(f"Failed to initialize batch workflow status manager: {e}")
                self.enable_status_tracking = False
    
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the complete batch processing workflow with status tracking.
        
        Args:
            input_data: Dictionary containing 'batch_urls' and optional 'batch_config'
            
        Returns:
            Dictionary containing all processing results with status tracking information
        """
        if not self.enable_status_tracking:
            # Fall back to parent implementation
            return super().run(input_data)
        
        # Extract batch information for status tracking
        batch_urls = input_data.get('batch_urls', [])
        batch_size = len(batch_urls)
        
        try:
            # Start workflow-level status tracking for batch processing
            workflow_status_id = self.workflow_status_manager.start_workflow(
                processing_session_id=self._generate_batch_session_id(),
                node_names=['BatchCreationNode', 'BatchProcessingNode', 'BatchStatusNode'],
                priority=self.status_priority
            )
            
            # Store status ID for access by other components
            input_data['workflow_status_id'] = workflow_status_id
            
            logger.info(f"Started status-aware batch workflow execution for {batch_size} URLs")
            
            # Execute the parent workflow with status tracking context
            with self._status_tracking_context(
                operation_name="batch_workflow",
                total_steps=3,  # Number of batch processing nodes
                auto_complete=False  # We'll handle completion manually
            ) as status_id:
                
                # Update workflow status to indicate batch processing
                self._update_status(
                    ProcessingStatusType.STARTING,
                    current_step=f"Starting batch processing of {batch_size} videos",
                    change_reason="Beginning batch processing workflow",
                    change_metadata={'batch_size': batch_size}
                )
                
                # Execute parent workflow
                result = super().run(input_data)
                
                # Determine final status based on result
                if result.get('status') == 'success':
                    # Mark workflow as completed
                    self.workflow_status_manager.finish_workflow(success=True)
                    self._update_status(
                        ProcessingStatusType.COMPLETED,
                        progress_percentage=100.0,
                        current_step="Batch workflow completed successfully",
                        change_reason="All batch processing steps completed successfully"
                    )
                else:
                    # Mark workflow as failed
                    error_info = str(result.get('error', 'Unknown batch workflow error'))
                    self.workflow_status_manager.finish_workflow(success=False, error_info=error_info)
                    self._record_error(
                        error_info=error_info,
                        error_metadata={'batch_result': result.get('status', 'unknown'), 'batch_size': batch_size},
                        should_retry=False
                    )
                
                # Add status tracking information to result
                result['status_tracking'] = {
                    'enabled': True,
                    'workflow_status_id': workflow_status_id,
                    'status_id': status_id,
                    'node_status_mapping': self.workflow_status_manager.node_status_mapping.copy(),
                    'batch_size': batch_size
                }
                
                return result
        
        except Exception as e:
            # Handle workflow error with status tracking
            error_info = f"Batch workflow execution failed: {str(e)}"
            
            if self.workflow_status_manager:
                self.workflow_status_manager.finish_workflow(success=False, error_info=error_info)
            
            self._record_error(
                error_info=error_info,
                error_metadata={'error_type': type(e).__name__, 'batch_size': batch_size},
                should_retry=False
            )
            
            logger.error(f"Status-aware batch workflow failed: {error_info}")
            raise
    
    def _execute_single_node(self, node_name: str) -> Dict[str, Any]:
        """Execute a single batch processing node with status tracking."""
        if not self.enable_status_tracking:
            return super()._execute_single_node(node_name)
        
        try:
            # Start node-level status tracking
            node_status_id = self.workflow_status_manager.start_node(node_name)
            
            logger.info(f"Starting status-aware execution of batch node {node_name}")
            
            # Execute node with the parent implementation
            result = super()._execute_single_node(node_name)
            
            # Mark node as completed
            self.workflow_status_manager.finish_node(node_name, success=True)
            
            # Add status tracking information to result
            result['status_tracking'] = {
                'node_status_id': node_status_id,
                'node_name': node_name
            }
            
            return result
        
        except Exception as e:
            # Mark node as failed
            error_info = f"Batch node {node_name} failed: {str(e)}"
            self.workflow_status_manager.finish_node(node_name, success=False, error_info=error_info)
            
            logger.error(f"Status-aware batch node execution failed: {error_info}")
            raise
    
    def _generate_batch_session_id(self) -> int:
        """Generate a unique session ID for batch processing."""
        import time
        return int(time.time() * 1000)  # Use timestamp as session ID
    
    def cleanup(self):
        """Cleanup resources including status tracking."""
        try:
            if self.workflow_status_manager:
                self.workflow_status_manager.cleanup()
        except Exception as e:
            logger.error(f"Error during status tracking cleanup: {e}")
        
        # Call parent cleanup if it exists
        if hasattr(super(), 'cleanup'):
            super().cleanup()


# Factory functions for creating status-aware workflows
def create_status_aware_summarizer_flow(*args, **kwargs) -> StatusAwareYouTubeSummarizerFlow:
    """
    Factory function to create a status-aware YouTube summarizer flow.
    
    This provides a convenient way to create the status-aware flow with
    sensible defaults while allowing customization.
    """
    return StatusAwareYouTubeSummarizerFlow(*args, **kwargs)


def create_status_aware_batch_flow(*args, **kwargs) -> StatusAwareYouTubeBatchProcessingFlow:
    """
    Factory function to create a status-aware YouTube batch processing flow.
    
    This provides a convenient way to create the status-aware batch flow with
    sensible defaults while allowing customization.
    """
    return StatusAwareYouTubeBatchProcessingFlow(*args, **kwargs)