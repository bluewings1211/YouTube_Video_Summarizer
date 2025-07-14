"""
Status-aware node implementations that integrate with the status tracking system.

This module provides mixins and base classes that existing nodes can use
to add status tracking capabilities without major refactoring.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

from .validation_nodes import BaseProcessingNode, NodeError
from ..services.status_integration import StatusTrackingMixin
from ..database.status_models import ProcessingStatusType


logger = logging.getLogger(__name__)


class StatusAwareNodeMixin(StatusTrackingMixin):
    """
    Mixin to add status tracking capabilities to existing nodes.
    
    This mixin can be added to any node class to provide automatic
    status tracking during node execution phases.
    """
    
    def __init__(self, *args, **kwargs):
        # Extract status tracking configuration
        self.node_enable_status_tracking = kwargs.pop('enable_status_tracking', True)
        self.node_status_immediate_updates = kwargs.pop('status_immediate_updates', False)
        
        # Initialize parent classes
        super().__init__(*args, **kwargs)
        
        # Override status tracking enabled flag
        self._status_tracking_enabled = self.node_enable_status_tracking
        
        if self._status_tracking_enabled:
            logger.debug(f"Status tracking enabled for node {getattr(self, 'name', self.__class__.__name__)}")
    
    def prep(self, store) -> Dict[str, Any]:
        """Prep phase with status tracking."""
        if not self._status_tracking_enabled:
            return super().prep(store)
        
        node_name = getattr(self, 'name', self.__class__.__name__)
        
        try:
            # Check if we have a workflow status context
            workflow_status_id = store.get('workflow_status_id')
            video_id = store.get('video_id') or store.get('database_video_id')
            
            # Create node status if not already exists
            if not self._current_status_id:
                self._create_processing_status(
                    video_id=video_id,
                    external_id=f"node_{node_name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                    tags=[node_name, 'node_execution'],
                    processing_metadata={
                        'node_name': node_name,
                        'workflow_status_id': workflow_status_id,
                        'phase': 'prep'
                    }
                )
            
            # Update status to indicate prep phase
            self._update_status(
                ProcessingStatusType.STARTING,
                progress_percentage=0.0,
                current_step=f"Preparing {node_name}",
                change_reason=f"Starting prep phase for {node_name}",
                immediate=self.node_status_immediate_updates
            )
            
            # Execute parent prep
            prep_result = super().prep(store)
            
            # Update status after prep completion
            self._update_progress(
                progress_percentage=33.0,
                current_step=f"Preparation completed for {node_name}",
                processing_metadata={'phase': 'prep_completed'},
                immediate=self.node_status_immediate_updates
            )
            
            return prep_result
        
        except Exception as e:
            # Record error and re-raise
            self._record_error(
                error_info=f"Prep phase failed for {node_name}: {str(e)}",
                error_metadata={
                    'phase': 'prep',
                    'node_name': node_name,
                    'error_type': type(e).__name__
                },
                should_retry=True,
                immediate=True
            )
            raise
    
    def exec(self, store, prep_result: Dict[str, Any]) -> Dict[str, Any]:
        """Exec phase with status tracking."""
        if not self._status_tracking_enabled:
            return super().exec(store, prep_result)
        
        node_name = getattr(self, 'name', self.__class__.__name__)
        
        try:
            # Update status to indicate exec phase
            self._update_progress(
                progress_percentage=33.0,
                current_step=f"Executing {node_name}",
                processing_metadata={'phase': 'exec'},
                immediate=self.node_status_immediate_updates
            )
            
            # Send heartbeat before potentially long-running operation
            self._send_heartbeat(33.0, f"Starting execution of {node_name}")
            
            # Execute parent exec
            exec_result = super().exec(store, prep_result)
            
            # Update status after exec completion
            self._update_progress(
                progress_percentage=66.0,
                current_step=f"Execution completed for {node_name}",
                processing_metadata={'phase': 'exec_completed'},
                immediate=self.node_status_immediate_updates
            )
            
            return exec_result
        
        except Exception as e:
            # Record error and re-raise
            self._record_error(
                error_info=f"Exec phase failed for {node_name}: {str(e)}",
                error_metadata={
                    'phase': 'exec',
                    'node_name': node_name,
                    'error_type': type(e).__name__
                },
                should_retry=True,
                immediate=True
            )
            raise
    
    def post(self, store, prep_result: Dict[str, Any], exec_result: Dict[str, Any]) -> Dict[str, Any]:
        """Post phase with status tracking."""
        if not self._status_tracking_enabled:
            return super().post(store, prep_result, exec_result)
        
        node_name = getattr(self, 'name', self.__class__.__name__)
        
        try:
            # Update status to indicate post phase
            self._update_progress(
                progress_percentage=66.0,
                current_step=f"Post-processing {node_name}",
                processing_metadata={'phase': 'post'},
                immediate=self.node_status_immediate_updates
            )
            
            # Execute parent post
            post_result = super().post(store, prep_result, exec_result)
            
            # Update status to completed
            self._update_status(
                ProcessingStatusType.COMPLETED,
                progress_percentage=100.0,
                current_step=f"Completed {node_name}",
                completed_steps=3,
                change_reason=f"Successfully completed all phases for {node_name}",
                immediate=True  # Always immediate for completion
            )
            
            return post_result
        
        except Exception as e:
            # Record error and re-raise
            self._record_error(
                error_info=f"Post phase failed for {node_name}: {str(e)}",
                error_metadata={
                    'phase': 'post',
                    'node_name': node_name,
                    'error_type': type(e).__name__
                },
                should_retry=True,
                immediate=True
            )
            raise


class StatusAwareBaseProcessingNode(StatusAwareNodeMixin, BaseProcessingNode):
    """
    Status-aware base processing node that combines status tracking with
    the standard base processing node functionality.
    """
    
    def __init__(self, name: str, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        logger.debug(f"Initialized status-aware base processing node: {name}")
    
    def _handle_error(self, error: Exception, context: str, retry_count: int = 0) -> NodeError:
        """Handle and log errors with both standard and status tracking."""
        # Call parent error handling
        node_error = super()._handle_error(error, context, retry_count)
        
        # Record error in status tracking system
        if self._status_tracking_enabled:
            self._record_error(
                error_info=f"{context}: {str(error)}",
                error_metadata={
                    'error_type': type(error).__name__,
                    'context': context,
                    'retry_count': retry_count,
                    'is_recoverable': node_error.is_recoverable,
                    'node_name': getattr(self, 'name', self.__class__.__name__)
                },
                should_retry=node_error.is_recoverable and retry_count < self.max_retries,
                immediate=True
            )
        
        return node_error


# Factory functions to create status-aware versions of existing nodes

def make_status_aware_node(node_class):
    """
    Factory function to create a status-aware version of any node class.
    
    Args:
        node_class: The original node class to enhance with status tracking
        
    Returns:
        A new class that combines StatusAwareNodeMixin with the original node class
    """
    class StatusAwareNode(StatusAwareNodeMixin, node_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
    
    # Set appropriate name for the new class
    StatusAwareNode.__name__ = f"StatusAware{node_class.__name__}"
    StatusAwareNode.__qualname__ = f"StatusAware{node_class.__name__}"
    
    return StatusAwareNode


def create_status_aware_youtube_data_node(*args, **kwargs):
    """Create a status-aware YouTube data node."""
    try:
        from .youtube_data_node import YouTubeDataNode
        StatusAwareYouTubeDataNode = make_status_aware_node(YouTubeDataNode)
        return StatusAwareYouTubeDataNode(*args, **kwargs)
    except ImportError:
        logger.warning("YouTubeDataNode not available, returning None")
        return None


def create_status_aware_summarization_node(*args, **kwargs):
    """Create a status-aware summarization node."""
    try:
        from .llm_nodes import SummarizationNode
        StatusAwareSummarizationNode = make_status_aware_node(SummarizationNode)
        return StatusAwareSummarizationNode(*args, **kwargs)
    except ImportError:
        logger.warning("SummarizationNode not available, returning None")
        return None


def create_status_aware_keyword_extraction_node(*args, **kwargs):
    """Create a status-aware keyword extraction node."""
    try:
        from .llm_nodes import KeywordExtractionNode
        StatusAwareKeywordExtractionNode = make_status_aware_node(KeywordExtractionNode)
        return StatusAwareKeywordExtractionNode(*args, **kwargs)
    except ImportError:
        logger.warning("KeywordExtractionNode not available, returning None")
        return None


def create_status_aware_timestamp_node(*args, **kwargs):
    """Create a status-aware timestamp node."""
    try:
        from .summary_nodes import TimestampNode
        StatusAwareTimestampNode = make_status_aware_node(TimestampNode)
        return StatusAwareTimestampNode(*args, **kwargs)
    except ImportError:
        logger.warning("TimestampNode not available, returning None")
        return None


# Example of how to create status-aware versions of specific nodes
class StatusAwareYouTubeDataNode(StatusAwareNodeMixin, BaseProcessingNode):
    """
    Example status-aware YouTube data node.
    
    This shows how the mixin can be used with specific node implementations.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__("YouTubeDataNode", *args, **kwargs)
    
    def prep(self, store) -> Dict[str, Any]:
        """Prepare for YouTube data extraction with status tracking."""
        # Custom prep logic with status tracking would go here
        return super().prep(store)
    
    def exec(self, store, prep_result: Dict[str, Any]) -> Dict[str, Any]:
        """Execute YouTube data extraction with status tracking."""
        # Custom exec logic with status tracking would go here
        return super().exec(store, prep_result)
    
    def post(self, store, prep_result: Dict[str, Any], exec_result: Dict[str, Any]) -> Dict[str, Any]:
        """Post-process YouTube data extraction with status tracking."""
        # Custom post logic with status tracking would go here
        return super().post(store, prep_result, exec_result)