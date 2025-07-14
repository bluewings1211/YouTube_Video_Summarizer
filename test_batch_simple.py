#!/usr/bin/env python3
"""
Simple test for batch processing integration without external dependencies.
"""

import sys
import os
import logging

# Add the src directory to the path for importing modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_batch_nodes_import():
    """Test that batch processing nodes can be imported."""
    logger.info("Testing batch processing nodes import")
    
    try:
        from src.refactored_nodes.batch_processing_nodes import (
            BatchCreationNode,
            BatchProcessingNode,
            BatchStatusNode,
            BatchProcessingConfig
        )
        
        logger.info("✓ Batch processing nodes imported successfully")
        
        # Test configuration
        config = BatchProcessingConfig(
            max_workers=2,
            enable_progress_tracking=True
        )
        
        # Test node initialization
        creation_node = BatchCreationNode(config=config)
        processing_node = BatchProcessingNode(config=config)
        status_node = BatchStatusNode(config=config)
        
        logger.info("✓ All batch processing nodes initialized successfully")
        
        # Test node structure
        assert hasattr(creation_node, 'prep')
        assert hasattr(creation_node, 'exec')
        assert hasattr(creation_node, 'post')
        
        logger.info("✓ Batch nodes have required PocketFlow methods")
        
        return True
        
    except Exception as e:
        logger.error(f"Batch nodes import test failed: {str(e)}")
        return False


def test_services_import():
    """Test that batch and queue services can be imported."""
    logger.info("Testing services import")
    
    try:
        from src.services.batch_service import BatchService, BatchCreateRequest
        from src.services.queue_service import QueueService, QueueProcessingOptions
        
        logger.info("✓ Batch and queue services imported successfully")
        
        # Test service structures
        assert hasattr(BatchService, 'create_batch')
        assert hasattr(BatchService, 'get_batch')
        assert hasattr(BatchService, 'start_batch_processing')
        
        assert hasattr(QueueService, 'get_next_queue_item')
        assert hasattr(QueueService, 'complete_queue_item')
        assert hasattr(QueueService, 'register_worker')
        
        logger.info("✓ Services have required methods")
        
        return True
        
    except Exception as e:
        logger.error(f"Services import test failed: {str(e)}")
        return False


def test_database_models_import():
    """Test that batch database models can be imported."""
    logger.info("Testing database models import")
    
    try:
        from src.database.batch_models import (
            Batch,
            BatchItem,
            QueueItem,
            ProcessingSession,
            BatchStatus,
            BatchItemStatus,
            BatchPriority
        )
        
        logger.info("✓ Database models imported successfully")
        
        # Test enum values
        assert BatchStatus.PENDING
        assert BatchStatus.PROCESSING
        assert BatchStatus.COMPLETED
        assert BatchStatus.CANCELLED
        
        assert BatchItemStatus.QUEUED
        assert BatchItemStatus.PROCESSING
        assert BatchItemStatus.COMPLETED
        assert BatchItemStatus.FAILED
        
        assert BatchPriority.LOW
        assert BatchPriority.NORMAL
        assert BatchPriority.HIGH
        
        logger.info("✓ Database models have correct enum values")
        
        return True
        
    except Exception as e:
        logger.error(f"Database models import test failed: {str(e)}")
        return False


def test_basic_node_functionality():
    """Test basic functionality of batch processing nodes."""
    logger.info("Testing basic node functionality")
    
    try:
        from src.refactored_nodes.batch_processing_nodes import BatchCreationNode
        from src.refactored_nodes.validation_nodes import Store
        
        # Create a test node
        node = BatchCreationNode()
        
        # Create a test store with sample data
        store = Store()
        store['batch_urls'] = [
            'https://www.youtube.com/watch?v=dQw4w9WgXcQ',
            'https://www.youtube.com/watch?v=9bZkp7q19f0'
        ]
        store['batch_config'] = {
            'name': 'Test Batch',
            'description': 'Test batch description',
            'priority': 'NORMAL'
        }
        
        # Test prep phase
        prep_result = node.prep(store)
        
        logger.info(f"✓ Prep phase completed: {prep_result.get('validation_status')}")
        
        # Verify prep result structure
        assert 'validation_status' in prep_result
        assert 'prep_timestamp' in prep_result
        
        if prep_result['validation_status'] == 'success':
            assert 'validated_urls' in prep_result
            assert 'batch_name' in prep_result
            assert 'priority' in prep_result
            logger.info("✓ Prep phase returned expected data structure")
        
        return True
        
    except Exception as e:
        logger.error(f"Basic node functionality test failed: {str(e)}")
        return False


def main():
    """Run all tests."""
    logger.info("Starting simple batch processing integration tests")
    
    tests = [
        test_batch_nodes_import,
        test_services_import,
        test_database_models_import,
        test_basic_node_functionality
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            logger.error(f"Test {test.__name__} failed with exception: {e}")
            failed += 1
    
    logger.info(f"Test results: {passed} passed, {failed} failed")
    
    if failed == 0:
        logger.info("✅ All tests passed! Batch processing integration is working correctly.")
        return True
    else:
        logger.error(f"❌ {failed} tests failed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)