"""
Comprehensive test suite for BatchProcessor functionality.

This test suite provides complete coverage for BatchProcessor including:
- Video processing workflow integration
- Progress tracking and session management
- Error handling and retry mechanisms
- Worker management and coordination
- Database integration and transaction handling
- Performance optimization and monitoring
- Edge cases and stress testing
- Integration with BatchService and VideoService
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, MagicMock, AsyncMock, call
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

from src.database.batch_models import (
    BatchItem, ProcessingSession, BatchItemStatus, BatchStatus
)
from src.database.models import Video
from src.services.batch_processor import (
    BatchProcessor, BatchProcessorError, create_batch_processor
)
from src.services.batch_service import BatchService, BatchItemResult
from src.services.video_service import VideoService
from src.flow import YouTubeSummarizerFlow, WorkflowConfig


class TestBatchProcessorComprehensive:
    """Comprehensive test suite for BatchProcessor."""

    @pytest.fixture
    def workflow_config(self):
        """Create workflow configuration for testing."""
        config = WorkflowConfig()
        config.enable_monitoring = True
        config.enable_fallbacks = True
        config.max_retries = 2
        config.timeout_seconds = 300
        return config

    @pytest.fixture
    def mock_batch_service(self):
        """Create mock BatchService."""
        service = Mock(spec=BatchService)
        service.create_processing_session.return_value = Mock(
            session_id="test_session_123",
            id=1
        )
        service.update_processing_session.return_value = True
        service.get_next_queue_item.return_value = None
        service.complete_batch_item.return_value = True
        service.get_batch.return_value = Mock(
            batch_id="test_batch",
            status=BatchStatus.PROCESSING
        )
        service.start_batch_processing.return_value = True
        service.get_batch_progress.return_value = Mock(
            status=BatchStatus.COMPLETED,
            progress_percentage=100.0,
            total_items=1,
            completed_items=1,
            failed_items=0
        )
        return service

    @pytest.fixture
    def mock_video_service(self):
        """Create mock VideoService."""
        service = Mock(spec=VideoService)
        mock_video = Mock()
        mock_video.id = 123
        service.create_video_record.return_value = mock_video
        service.save_summary.return_value = None
        service.save_keywords.return_value = None
        service.save_timestamped_segments.return_value = None
        return service

    @pytest.fixture
    def mock_database_session(self):
        """Create mock database session."""
        session = Mock()
        session.get.return_value = Mock(
            id=1,
            url="https://www.youtube.com/watch?v=test123",
            batch_id=1
        )
        session.commit.return_value = None
        session.close.return_value = None
        return session

    @pytest.fixture
    def batch_processor(self, mock_batch_service, mock_video_service, workflow_config):
        """Create BatchProcessor instance."""
        return BatchProcessor(
            batch_service=mock_batch_service,
            video_service=mock_video_service,
            workflow_config=workflow_config
        )

    @pytest.fixture
    def minimal_batch_processor(self):
        """Create minimal BatchProcessor for testing initialization."""
        return BatchProcessor()

    # Initialization Tests

    def test_batch_processor_initialization_with_services(self, batch_processor, mock_batch_service, mock_video_service, workflow_config):
        """Test BatchProcessor initialization with provided services."""
        assert batch_processor.batch_service == mock_batch_service
        assert batch_processor.video_service == mock_video_service
        assert batch_processor.workflow_config == workflow_config
        assert isinstance(batch_processor._active_workers, dict)
        assert len(batch_processor._active_workers) == 0

    def test_batch_processor_initialization_minimal(self, minimal_batch_processor):
        """Test BatchProcessor initialization with minimal parameters."""
        assert minimal_batch_processor.batch_service is None
        assert minimal_batch_processor.video_service is None
        assert isinstance(minimal_batch_processor.workflow_config, WorkflowConfig)
        assert isinstance(minimal_batch_processor._active_workers, dict)

    def test_batch_processor_initialization_custom_config(self):
        """Test BatchProcessor initialization with custom workflow config."""
        custom_config = WorkflowConfig()
        custom_config.max_retries = 5
        custom_config.timeout_seconds = 600
        
        processor = BatchProcessor(workflow_config=custom_config)
        
        assert processor.workflow_config == custom_config
        assert processor.workflow_config.max_retries == 5
        assert processor.workflow_config.timeout_seconds == 600

    # Video Processing Tests

    @pytest.mark.asyncio
    async def test_process_batch_item_success(self, batch_processor, mock_database_session):
        """Test successful batch item processing."""
        batch_item_id = 1
        worker_id = "test_worker"
        
        # Mock database session context manager
        @asynccontextmanager
        async def mock_get_session():
            yield mock_database_session
        
        batch_processor._get_database_session = mock_get_session
        
        # Mock successful workflow execution
        with patch('src.services.batch_processor.extract_video_id_from_url') as mock_extract:
            mock_extract.return_value = "test123"
            
            with patch.object(batch_processor, '_process_video_workflow') as mock_workflow:
                mock_workflow.return_value = {
                    'success': True,
                    'workflow_result': {
                        'video_data': {'title': 'Test Video', 'duration': 180},
                        'transcript_data': {'content': 'Test transcript'},
                        'summary_data': {'content': 'Test summary'}
                    },
                    'video_id': 'test123',
                    'url': 'https://www.youtube.com/watch?v=test123'
                }
                
                with patch.object(batch_processor, '_save_video_results') as mock_save:
                    mock_video_record = Mock()
                    mock_video_record.id = 123
                    mock_save.return_value = mock_video_record
                    
                    # Execute test
                    result = await batch_processor.process_batch_item(batch_item_id, worker_id)
                    
                    # Verify result
                    assert isinstance(result, BatchItemResult)
                    assert result.batch_item_id == batch_item_id
                    assert result.status == BatchItemStatus.COMPLETED
                    assert result.video_id == 123
                    assert 'workflow_result' in result.result_data

    @pytest.mark.asyncio
    async def test_process_batch_item_invalid_url(self, batch_processor, mock_database_session):
        """Test batch item processing with invalid URL."""
        batch_item_id = 1
        worker_id = "test_worker"
        
        @asynccontextmanager
        async def mock_get_session():
            yield mock_database_session
        
        batch_processor._get_database_session = mock_get_session
        
        # Mock invalid URL extraction
        with patch('src.services.batch_processor.extract_video_id_from_url') as mock_extract:
            mock_extract.return_value = None
            
            result = await batch_processor.process_batch_item(batch_item_id, worker_id)
            
            assert isinstance(result, BatchItemResult)
            assert result.batch_item_id == batch_item_id
            assert result.status == BatchItemStatus.FAILED
            assert "Invalid YouTube URL" in result.error_message

    @pytest.mark.asyncio
    async def test_process_batch_item_not_found(self, batch_processor, mock_database_session):
        """Test batch item processing when item not found."""
        batch_item_id = 999
        worker_id = "test_worker"
        
        # Mock session returning None for batch item
        mock_database_session.get.return_value = None
        
        @asynccontextmanager
        async def mock_get_session():
            yield mock_database_session
        
        batch_processor._get_database_session = mock_get_session
        
        result = await batch_processor.process_batch_item(batch_item_id, worker_id)
        
        assert isinstance(result, BatchItemResult)
        assert result.batch_item_id == batch_item_id
        assert result.status == BatchItemStatus.FAILED
        assert "not found" in result.error_message

    @pytest.mark.asyncio
    async def test_process_batch_item_workflow_failure(self, batch_processor, mock_database_session):
        """Test batch item processing with workflow failure."""
        batch_item_id = 1
        worker_id = "test_worker"
        
        @asynccontextmanager
        async def mock_get_session():
            yield mock_database_session
        
        batch_processor._get_database_session = mock_get_session
        
        with patch('src.services.batch_processor.extract_video_id_from_url') as mock_extract:
            mock_extract.return_value = "test123"
            
            with patch.object(batch_processor, '_process_video_workflow') as mock_workflow:
                mock_workflow.side_effect = Exception("Workflow failed")
                
                result = await batch_processor.process_batch_item(batch_item_id, worker_id)
                
                assert isinstance(result, BatchItemResult)
                assert result.batch_item_id == batch_item_id
                assert result.status == BatchItemStatus.FAILED
                assert "Workflow failed" in result.error_message

    @pytest.mark.asyncio
    async def test_process_video_workflow_success(self, batch_processor):
        """Test successful video workflow processing."""
        url = "https://www.youtube.com/watch?v=test123"
        video_id = "test123"
        session_id = "test_session"
        
        # Mock workflow
        mock_workflow = Mock(spec=YouTubeSummarizerFlow)
        mock_workflow.execute = AsyncMock(return_value={
            'video_data': {'title': 'Test Video', 'duration': 180},
            'transcript_data': {'content': 'Test transcript'},
            'summary_data': {'content': 'Test summary'}
        })
        
        result = await batch_processor._process_video_workflow(
            mock_workflow, url, video_id, session_id
        )
        
        assert result['success'] is True
        assert result['video_id'] == video_id
        assert result['url'] == url
        assert 'workflow_result' in result
        assert 'processed_at' in result
        
        # Verify workflow was called with correct parameters
        mock_workflow.execute.assert_called_once_with({
            'url': url,
            'video_id': video_id,
            'batch_processing': True,
            'session_id': session_id
        })

    @pytest.mark.asyncio
    async def test_process_video_workflow_failure(self, batch_processor):
        """Test video workflow processing with failure."""
        url = "https://www.youtube.com/watch?v=test123"
        video_id = "test123"
        session_id = "test_session"
        
        # Mock workflow with failure
        mock_workflow = Mock(spec=YouTubeSummarizerFlow)
        mock_workflow.execute = AsyncMock(side_effect=Exception("Workflow error"))
        
        result = await batch_processor._process_video_workflow(
            mock_workflow, url, video_id, session_id
        )
        
        assert result['success'] is False
        assert result['error'] == "Workflow error"
        assert result['video_id'] == video_id
        assert result['url'] == url

    @pytest.mark.asyncio
    async def test_save_video_results_comprehensive(self, batch_processor, mock_video_service):
        """Test comprehensive video results saving."""
        video_id = "test123"
        url = "https://www.youtube.com/watch?v=test123"
        result = {
            'workflow_result': {
                'video_data': {
                    'title': 'Test Video',
                    'duration': 180,
                    'channel_name': 'Test Channel',
                    'description': 'Test description',
                    'published_date': '2023-01-01',
                    'view_count': 1000
                },
                'transcript_data': {
                    'content': 'Test transcript content',
                    'language': 'en'
                },
                'summary_data': {
                    'content': 'Test summary content',
                    'language': 'en',
                    'model_used': 'test_model'
                },
                'keywords': ['test', 'video'],
                'segments': [
                    {'start': 0, 'end': 60, 'text': 'First segment'},
                    {'start': 60, 'end': 120, 'text': 'Second segment'}
                ]
            }
        }
        
        # Mock video record
        mock_video_record = Mock()
        mock_video_record.id = 123
        mock_video_service.create_video_record.return_value = mock_video_record
        
        saved_record = await batch_processor._save_video_results(
            mock_video_service, video_id, result, url
        )
        
        assert saved_record == mock_video_record
        
        # Verify video record creation
        mock_video_service.create_video_record.assert_called_once_with(
            video_id=video_id,
            title='Test Video',
            duration=180,
            url=url,
            channel_name='Test Channel',
            description='Test description',
            published_date='2023-01-01',
            view_count=1000,
            transcript_content='Test transcript content',
            transcript_language='en'
        )
        
        # Verify summary saving
        mock_video_service.save_summary.assert_called_once_with(
            123, 'Test summary content', 'en', 'test_model'
        )
        
        # Verify keywords saving
        mock_video_service.save_keywords.assert_called_once_with(
            123, ['test', 'video']
        )
        
        # Verify segments saving
        mock_video_service.save_timestamped_segments.assert_called_once_with(
            123, [
                {'start': 0, 'end': 60, 'text': 'First segment'},
                {'start': 60, 'end': 120, 'text': 'Second segment'}
            ]
        )

    @pytest.mark.asyncio
    async def test_save_video_results_partial_data(self, batch_processor, mock_video_service):
        """Test saving video results with partial data."""
        video_id = "test123"
        url = "https://www.youtube.com/watch?v=test123"
        result = {
            'workflow_result': {
                'video_data': {
                    'title': 'Test Video',
                    'duration': None  # Missing duration
                },
                'transcript_data': {
                    'content': 'Test transcript'
                    # Missing language
                },
                'summary_data': None,  # No summary
                # Missing keywords and segments
            }
        }
        
        mock_video_record = Mock()
        mock_video_record.id = 123
        mock_video_service.create_video_record.return_value = mock_video_record
        
        saved_record = await batch_processor._save_video_results(
            mock_video_service, video_id, result, url
        )
        
        assert saved_record == mock_video_record
        
        # Verify video record creation with defaults
        mock_video_service.create_video_record.assert_called_once_with(
            video_id=video_id,
            title='Test Video',
            duration=None,
            url=url,
            channel_name=None,
            description=None,
            published_date=None,
            view_count=None,
            transcript_content='Test transcript',
            transcript_language=None
        )
        
        # Verify optional saves were not called
        mock_video_service.save_summary.assert_not_called()
        mock_video_service.save_keywords.assert_not_called()
        mock_video_service.save_timestamped_segments.assert_not_called()

    @pytest.mark.asyncio
    async def test_save_video_results_error_handling(self, batch_processor, mock_video_service):
        """Test error handling in video results saving."""
        video_id = "test123"
        url = "https://www.youtube.com/watch?v=test123"
        result = {
            'workflow_result': {
                'video_data': {'title': 'Test Video'}
            }
        }
        
        # Mock video service error
        mock_video_service.create_video_record.side_effect = Exception("Database error")
        
        saved_record = await batch_processor._save_video_results(
            mock_video_service, video_id, result, url
        )
        
        assert saved_record is None

    # Worker Management Tests

    @pytest.mark.asyncio
    async def test_start_batch_worker(self, batch_processor, mock_database_session):
        """Test starting a batch worker."""
        worker_id = "test_worker"
        queue_name = "test_queue"
        
        @asynccontextmanager
        async def mock_get_session():
            yield mock_database_session
        
        batch_processor._get_database_session = mock_get_session
        
        # Mock queue item
        mock_queue_item = Mock()
        mock_queue_item.batch_item_id = 1
        batch_processor.batch_service.get_next_queue_item.side_effect = [
            mock_queue_item, None  # First call returns item, second returns None
        ]
        
        # Mock processing result
        with patch.object(batch_processor, 'process_batch_item') as mock_process:
            mock_result = BatchItemResult(
                batch_item_id=1,
                status=BatchItemStatus.COMPLETED,
                video_id=123
            )
            mock_process.return_value = mock_result
            
            # Start worker (will run until no more items)
            await batch_processor.start_batch_worker(worker_id, queue_name)
            
            # Verify worker was registered
            assert worker_id in batch_processor._active_workers
            worker_info = batch_processor._active_workers[worker_id]
            assert worker_info['queue_name'] == queue_name
            assert worker_info['processed_items'] == 1
            assert worker_info['failed_items'] == 0

    @pytest.mark.asyncio
    async def test_start_batch_worker_with_failures(self, batch_processor, mock_database_session):
        """Test batch worker with processing failures."""
        worker_id = "test_worker"
        queue_name = "test_queue"
        
        @asynccontextmanager
        async def mock_get_session():
            yield mock_database_session
        
        batch_processor._get_database_session = mock_get_session
        
        # Mock queue items
        mock_queue_items = [Mock(batch_item_id=i) for i in range(3)]
        mock_queue_items.append(None)  # End processing
        batch_processor.batch_service.get_next_queue_item.side_effect = mock_queue_items
        
        # Mock mixed processing results
        with patch.object(batch_processor, 'process_batch_item') as mock_process:
            mock_results = [
                BatchItemResult(batch_item_id=0, status=BatchItemStatus.COMPLETED, video_id=100),
                BatchItemResult(batch_item_id=1, status=BatchItemStatus.FAILED, error_message="Error"),
                BatchItemResult(batch_item_id=2, status=BatchItemStatus.COMPLETED, video_id=102)
            ]
            mock_process.side_effect = mock_results
            
            await batch_processor.start_batch_worker(worker_id, queue_name)
            
            # Verify worker statistics
            worker_info = batch_processor._active_workers[worker_id]
            assert worker_info['processed_items'] == 2  # Two successes
            assert worker_info['failed_items'] == 1    # One failure

    @pytest.mark.asyncio
    async def test_stop_batch_worker(self, batch_processor):
        """Test stopping a batch worker."""
        worker_id = "test_worker"
        
        # Register worker
        batch_processor._active_workers[worker_id] = {
            'started_at': datetime.utcnow(),
            'queue_name': 'test_queue',
            'processed_items': 5,
            'failed_items': 1
        }
        
        # Stop worker
        await batch_processor.stop_batch_worker(worker_id)
        
        # Verify worker was removed
        assert worker_id not in batch_processor._active_workers

    def test_get_worker_stats(self, batch_processor):
        """Test getting worker statistics."""
        # Add some active workers
        workers = {
            'worker_1': {
                'started_at': datetime.utcnow() - timedelta(minutes=30),
                'queue_name': 'queue_1',
                'processed_items': 10,
                'failed_items': 2
            },
            'worker_2': {
                'started_at': datetime.utcnow() - timedelta(minutes=15),
                'queue_name': 'queue_2',
                'processed_items': 5,
                'failed_items': 0
            }
        }
        
        batch_processor._active_workers = workers
        
        stats = batch_processor.get_worker_stats()
        
        assert stats['active_workers'] == 2
        assert len(stats['workers']) == 2
        
        # Verify worker details
        for worker_id, worker_info in workers.items():
            worker_stats = stats['workers'][worker_id]
            assert worker_stats['queue_name'] == worker_info['queue_name']
            assert worker_stats['processed_items'] == worker_info['processed_items']
            assert worker_stats['failed_items'] == worker_info['failed_items']
            assert 'uptime_seconds' in worker_stats

    # Batch Processing Tests

    @pytest.mark.asyncio
    async def test_process_batch_by_id_success(self, batch_processor, mock_database_session):
        """Test processing entire batch by ID."""
        batch_id = "test_batch"
        num_workers = 2
        
        @asynccontextmanager
        async def mock_get_session():
            yield mock_database_session
        
        batch_processor._get_database_session = mock_get_session
        
        # Mock batch progress updates
        progress_updates = [
            Mock(status=BatchStatus.PROCESSING, progress_percentage=0.0),
            Mock(status=BatchStatus.PROCESSING, progress_percentage=50.0),
            Mock(status=BatchStatus.COMPLETED, progress_percentage=100.0)
        ]
        batch_processor.batch_service.get_batch_progress.side_effect = progress_updates
        
        # Mock worker processing (simulate quick completion)
        original_start_worker = batch_processor.start_batch_worker
        
        async def mock_start_worker(worker_id, queue_name):
            # Simulate worker processing briefly then stopping
            batch_processor._active_workers[worker_id] = {
                'started_at': datetime.utcnow(),
                'queue_name': queue_name,
                'processed_items': 1,
                'failed_items': 0
            }
            await asyncio.sleep(0.1)  # Brief processing simulation
        
        batch_processor.start_batch_worker = mock_start_worker
        
        result = await batch_processor.process_batch_by_id(batch_id, num_workers)
        
        assert result['batch_id'] == batch_id
        assert result['status'] == BatchStatus.COMPLETED.value
        assert result['progress_percentage'] == 100.0
        assert result['num_workers_used'] == num_workers
        
        # Verify batch service calls
        batch_processor.batch_service.get_batch.assert_called_with(batch_id)
        batch_processor.batch_service.start_batch_processing.assert_called_with(batch_id)

    @pytest.mark.asyncio
    async def test_process_batch_by_id_not_found(self, batch_processor, mock_database_session):
        """Test processing batch that doesn't exist."""
        batch_id = "non_existent_batch"
        
        @asynccontextmanager
        async def mock_get_session():
            yield mock_database_session
        
        batch_processor._get_database_session = mock_get_session
        batch_processor.batch_service.get_batch.return_value = None
        
        with pytest.raises(BatchProcessorError, match="not found"):
            await batch_processor.process_batch_by_id(batch_id)

    @pytest.mark.asyncio
    async def test_process_batch_by_id_with_error(self, batch_processor, mock_database_session):
        """Test batch processing with error."""
        batch_id = "error_batch"
        
        @asynccontextmanager
        async def mock_get_session():
            yield mock_database_session
        
        batch_processor._get_database_session = mock_get_session
        batch_processor.batch_service.start_batch_processing.side_effect = Exception("Processing error")
        
        with pytest.raises(BatchProcessorError, match="Failed to process batch"):
            await batch_processor.process_batch_by_id(batch_id)

    # Progress Tracking Tests

    @pytest.mark.asyncio
    async def test_progress_tracking_during_processing(self, batch_processor):
        """Test progress tracking during video processing."""
        url = "https://www.youtube.com/watch?v=test123"
        video_id = "test123"
        session_id = "test_session"
        
        # Mock workflow with progress updates
        mock_workflow = Mock(spec=YouTubeSummarizerFlow)
        mock_workflow.execute = AsyncMock(return_value={'test': 'result'})
        
        result = await batch_processor._process_video_workflow(
            mock_workflow, url, video_id, session_id
        )
        
        # Verify progress updates were called
        batch_processor.batch_service.update_processing_session.assert_has_calls([
            call(session_id, 5.0, "Initializing workflow"),
            call(session_id, 95.0, "Workflow completed, saving results")
        ])

    def test_session_management(self, batch_processor):
        """Test processing session management."""
        batch_item_id = 1
        worker_id = "test_worker"
        
        # Mock session creation
        mock_session = Mock()
        mock_session.session_id = "test_session_123"
        batch_processor.batch_service.create_processing_session.return_value = mock_session
        
        # Test session creation call
        session = batch_processor.batch_service.create_processing_session(batch_item_id, worker_id)
        
        assert session.session_id == "test_session_123"
        batch_processor.batch_service.create_processing_session.assert_called_with(batch_item_id, worker_id)

    # Error Handling Tests

    @pytest.mark.asyncio
    async def test_error_handling_database_errors(self, batch_processor):
        """Test error handling for database errors."""
        batch_item_id = 1
        worker_id = "test_worker"
        
        # Mock database session error
        @asynccontextmanager
        async def mock_get_session_error():
            raise Exception("Database connection failed")
        
        batch_processor._get_database_session = mock_get_session_error
        
        result = await batch_processor.process_batch_item(batch_item_id, worker_id)
        
        assert result.status == BatchItemStatus.FAILED
        assert "Database connection failed" in result.error_message

    @pytest.mark.asyncio
    async def test_error_handling_workflow_timeout(self, batch_processor, mock_database_session):
        """Test error handling for workflow timeout."""
        batch_item_id = 1
        worker_id = "test_worker"
        
        @asynccontextmanager
        async def mock_get_session():
            yield mock_database_session
        
        batch_processor._get_database_session = mock_get_session
        
        with patch('src.services.batch_processor.extract_video_id_from_url') as mock_extract:
            mock_extract.return_value = "test123"
            
            with patch.object(batch_processor, '_process_video_workflow') as mock_workflow:
                mock_workflow.side_effect = asyncio.TimeoutError("Workflow timeout")
                
                result = await batch_processor.process_batch_item(batch_item_id, worker_id)
                
                assert result.status == BatchItemStatus.FAILED
                assert "Workflow timeout" in result.error_message

    @pytest.mark.asyncio
    async def test_error_handling_worker_errors(self, batch_processor, mock_database_session):
        """Test error handling for worker errors."""
        worker_id = "error_worker"
        queue_name = "error_queue"
        
        @asynccontextmanager
        async def mock_get_session():
            yield mock_database_session
        
        batch_processor._get_database_session = mock_get_session
        
        # Mock queue service error
        batch_processor.batch_service.get_next_queue_item.side_effect = Exception("Queue error")
        
        # Worker should handle error gracefully
        await batch_processor.start_batch_worker(worker_id, queue_name)
        
        # Worker should be cleaned up after error
        assert worker_id not in batch_processor._active_workers

    # Performance and Concurrency Tests

    @pytest.mark.asyncio
    async def test_concurrent_batch_item_processing(self, batch_processor, mock_database_session):
        """Test concurrent processing of multiple batch items."""
        worker_ids = ["worker_1", "worker_2", "worker_3"]
        batch_item_ids = [1, 2, 3]
        
        @asynccontextmanager
        async def mock_get_session():
            yield mock_database_session
        
        batch_processor._get_database_session = mock_get_session
        
        # Mock successful processing for all items
        with patch('src.services.batch_processor.extract_video_id_from_url') as mock_extract:
            mock_extract.return_value = "test123"
            
            with patch.object(batch_processor, '_process_video_workflow') as mock_workflow:
                mock_workflow.return_value = {
                    'success': True,
                    'workflow_result': {'video_data': {'title': 'Test'}},
                    'video_id': 'test123',
                    'url': 'https://www.youtube.com/watch?v=test123'
                }
                
                with patch.object(batch_processor, '_save_video_results') as mock_save:
                    mock_save.return_value = Mock(id=123)
                    
                    # Process items concurrently
                    tasks = [
                        batch_processor.process_batch_item(batch_item_id, worker_id)
                        for batch_item_id, worker_id in zip(batch_item_ids, worker_ids)
                    ]
                    
                    results = await asyncio.gather(*tasks)
                    
                    # Verify all processed successfully
                    assert len(results) == 3
                    for result in results:
                        assert result.status == BatchItemStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_performance_large_batch_processing(self, batch_processor, mock_database_session):
        """Test performance with large batch processing."""
        import time
        
        @asynccontextmanager
        async def mock_get_session():
            yield mock_database_session
        
        batch_processor._get_database_session = mock_get_session
        
        # Mock fast processing
        with patch('src.services.batch_processor.extract_video_id_from_url') as mock_extract:
            mock_extract.return_value = "test123"
            
            with patch.object(batch_processor, '_process_video_workflow') as mock_workflow:
                async def fast_workflow(*args):
                    await asyncio.sleep(0.001)  # Simulate minimal processing
                    return {
                        'success': True,
                        'workflow_result': {'video_data': {'title': 'Test'}},
                        'video_id': 'test123',
                        'url': 'https://www.youtube.com/watch?v=test123'
                    }
                
                mock_workflow.side_effect = fast_workflow
                
                with patch.object(batch_processor, '_save_video_results') as mock_save:
                    mock_save.return_value = Mock(id=123)
                    
                    # Process 50 items
                    start_time = time.time()
                    
                    tasks = [
                        batch_processor.process_batch_item(i, f"worker_{i % 5}")
                        for i in range(50)
                    ]
                    
                    results = await asyncio.gather(*tasks)
                    
                    processing_time = time.time() - start_time
                    
                    # Verify performance
                    assert processing_time < 10.0  # Should complete within 10 seconds
                    assert len(results) == 50
                    assert all(r.status == BatchItemStatus.COMPLETED for r in results)

    def test_memory_usage_optimization(self, batch_processor):
        """Test memory usage optimization."""
        import psutil
        import os
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Create and cleanup many workers
        for i in range(100):
            worker_id = f"memory_worker_{i}"
            batch_processor._active_workers[worker_id] = {
                'started_at': datetime.utcnow(),
                'queue_name': f'queue_{i % 10}',
                'processed_items': i,
                'failed_items': i // 10
            }
        
        # Get statistics (this processes all worker data)
        stats = batch_processor.get_worker_stats()
        
        # Clear workers
        batch_processor._active_workers.clear()
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Check memory usage
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 20MB)
        assert memory_increase < 20 * 1024 * 1024
        assert stats['active_workers'] == 100

    # Integration Tests

    @pytest.mark.asyncio
    async def test_integration_with_batch_service(self, batch_processor, mock_database_session):
        """Test integration with BatchService."""
        batch_item_id = 1
        worker_id = "integration_worker"
        
        @asynccontextmanager
        async def mock_get_session():
            yield mock_database_session
        
        batch_processor._get_database_session = mock_get_session
        
        # Test all BatchService integration points
        with patch('src.services.batch_processor.extract_video_id_from_url') as mock_extract:
            mock_extract.return_value = "test123"
            
            with patch.object(batch_processor, '_process_video_workflow') as mock_workflow:
                mock_workflow.return_value = {
                    'success': True,
                    'workflow_result': {'video_data': {'title': 'Test'}},
                    'video_id': 'test123',
                    'url': 'https://www.youtube.com/watch?v=test123'
                }
                
                with patch.object(batch_processor, '_save_video_results') as mock_save:
                    mock_save.return_value = Mock(id=123)
                    
                    result = await batch_processor.process_batch_item(batch_item_id, worker_id)
                    
                    # Verify BatchService interactions
                    batch_processor.batch_service.create_processing_session.assert_called_once()
                    assert batch_processor.batch_service.update_processing_session.call_count >= 2

    @pytest.mark.asyncio
    async def test_integration_with_video_service(self, batch_processor, mock_database_session):
        """Test integration with VideoService."""
        video_id = "test123"
        url = "https://www.youtube.com/watch?v=test123"
        result = {
            'workflow_result': {
                'video_data': {'title': 'Integration Test'},
                'transcript_data': {'content': 'Test transcript'},
                'summary_data': {'content': 'Test summary'}
            }
        }
        
        # Test VideoService integration
        saved_record = await batch_processor._save_video_results(
            batch_processor.video_service, video_id, result, url
        )
        
        # Verify VideoService interactions
        batch_processor.video_service.create_video_record.assert_called_once()
        batch_processor.video_service.save_summary.assert_called_once()

    # Edge Cases and Special Scenarios

    @pytest.mark.asyncio
    async def test_edge_case_empty_workflow_result(self, batch_processor, mock_database_session):
        """Test handling of empty workflow results."""
        batch_item_id = 1
        worker_id = "test_worker"
        
        @asynccontextmanager
        async def mock_get_session():
            yield mock_database_session
        
        batch_processor._get_database_session = mock_get_session
        
        with patch('src.services.batch_processor.extract_video_id_from_url') as mock_extract:
            mock_extract.return_value = "test123"
            
            with patch.object(batch_processor, '_process_video_workflow') as mock_workflow:
                mock_workflow.return_value = {
                    'success': True,
                    'workflow_result': {},  # Empty result
                    'video_id': 'test123',
                    'url': 'https://www.youtube.com/watch?v=test123'
                }
                
                with patch.object(batch_processor, '_save_video_results') as mock_save:
                    mock_save.return_value = Mock(id=123)
                    
                    result = await batch_processor.process_batch_item(batch_item_id, worker_id)
                    
                    assert result.status == BatchItemStatus.COMPLETED
                    assert result.video_id == 123

    @pytest.mark.asyncio
    async def test_edge_case_malformed_url(self, batch_processor, mock_database_session):
        """Test handling of malformed URLs."""
        batch_item_id = 1
        worker_id = "test_worker"
        
        # Mock batch item with malformed URL
        mock_batch_item = Mock()
        mock_batch_item.url = "not-a-valid-youtube-url"
        mock_batch_item.batch_id = 1
        mock_database_session.get.return_value = mock_batch_item
        
        @asynccontextmanager
        async def mock_get_session():
            yield mock_database_session
        
        batch_processor._get_database_session = mock_get_session
        
        with patch('src.services.batch_processor.extract_video_id_from_url') as mock_extract:
            mock_extract.return_value = None  # Invalid URL
            
            result = await batch_processor.process_batch_item(batch_item_id, worker_id)
            
            assert result.status == BatchItemStatus.FAILED
            assert "Invalid YouTube URL" in result.error_message

    def test_edge_case_worker_cleanup_edge_cases(self, batch_processor):
        """Test worker cleanup edge cases."""
        # Test stopping non-existent worker
        asyncio.run(batch_processor.stop_batch_worker("non_existent_worker"))
        
        # Should not raise error
        assert True
        
        # Test getting stats with no workers
        stats = batch_processor.get_worker_stats()
        assert stats['active_workers'] == 0
        assert len(stats['workers']) == 0

    # Factory Function Tests

    def test_create_batch_processor_factory(self):
        """Test create_batch_processor factory function."""
        # Test with default config
        processor = create_batch_processor()
        assert isinstance(processor, BatchProcessor)
        assert processor.batch_service is None
        assert processor.video_service is None
        assert isinstance(processor.workflow_config, WorkflowConfig)
        
        # Test with custom config
        custom_config = WorkflowConfig()
        custom_config.max_retries = 10
        
        processor2 = create_batch_processor(custom_config)
        assert isinstance(processor2, BatchProcessor)
        assert processor2.workflow_config == custom_config
        assert processor2.workflow_config.max_retries == 10

    # Example Usage Tests

    @pytest.mark.asyncio
    async def test_example_batch_processing_workflow(self):
        """Test example batch processing workflow."""
        # This tests the example function structure without external dependencies
        
        with patch('src.services.batch_processor.BatchProcessor._get_database_session'):
            with patch('src.services.batch_processor.BatchService') as mock_batch_service_class:
                mock_batch_service = Mock()
                mock_batch_service_class.return_value = mock_batch_service
                
                mock_batch = Mock()
                mock_batch.batch_id = "example_batch"
                mock_batch_service.create_batch.return_value = mock_batch
                
                processor = create_batch_processor()
                
                # Test that the processor can be used in example scenarios
                assert processor is not None
                assert isinstance(processor.workflow_config, WorkflowConfig)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "--asyncio-mode=auto"])