"""
Configuration and setup for the status event system.

This module provides configuration classes and setup functions
to initialize the event system with appropriate handlers.
"""

import logging
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field

from .status_events import (
    StatusEventManager, EventHandler, LoggingEventHandler, 
    DatabaseEventHandler, WebhookEventHandler, EventType, EventPriority
)
from .status_event_integration import StatusEventIntegrator


logger = logging.getLogger(__name__)


@dataclass
class EventSystemConfig:
    """Configuration for the event system."""
    
    # Event manager settings
    max_workers: int = 5
    
    # Logging handler settings
    enable_logging: bool = True
    log_level: int = logging.INFO
    
    # Database handler settings
    enable_database: bool = False
    database_table_name: str = "status_events"
    
    # Webhook handler settings
    enable_webhooks: bool = False
    webhook_urls: List[str] = field(default_factory=list)
    webhook_event_types: Optional[Set[EventType]] = None
    
    # Custom handlers
    custom_handlers: List[EventHandler] = field(default_factory=list)
    
    # Event filtering
    global_event_types: Optional[Set[EventType]] = None
    global_priority_filter: Optional[Set[EventPriority]] = None


class EventSystemBuilder:
    """Builder class for setting up the event system."""
    
    def __init__(self):
        self.config = EventSystemConfig()
        self.event_manager: Optional[StatusEventManager] = None
        self.integrator: Optional[StatusEventIntegrator] = None
    
    def with_max_workers(self, max_workers: int) -> 'EventSystemBuilder':
        """Set the maximum number of event processing workers."""
        self.config.max_workers = max_workers
        return self
    
    def with_logging(self, enabled: bool = True, log_level: int = logging.INFO) -> 'EventSystemBuilder':
        """Configure logging event handler."""
        self.config.enable_logging = enabled
        self.config.log_level = log_level
        return self
    
    def with_database(self, enabled: bool = True, table_name: str = "status_events") -> 'EventSystemBuilder':
        """Configure database event handler."""
        self.config.enable_database = enabled
        self.config.database_table_name = table_name
        return self
    
    def with_webhooks(
        self, 
        webhook_urls: List[str], 
        event_types: Optional[Set[EventType]] = None
    ) -> 'EventSystemBuilder':
        """Configure webhook event handler."""
        self.config.enable_webhooks = True
        self.config.webhook_urls = webhook_urls
        self.config.webhook_event_types = event_types
        return self
    
    def with_custom_handler(self, handler: EventHandler) -> 'EventSystemBuilder':
        """Add a custom event handler."""
        self.config.custom_handlers.append(handler)
        return self
    
    def with_event_filter(
        self, 
        event_types: Optional[Set[EventType]] = None,
        priority_filter: Optional[Set[EventPriority]] = None
    ) -> 'EventSystemBuilder':
        """Configure global event filtering."""
        self.config.global_event_types = event_types
        self.config.global_priority_filter = priority_filter
        return self
    
    def build(self) -> StatusEventIntegrator:
        """Build and configure the event system."""
        # Create event manager
        self.event_manager = StatusEventManager(max_workers=self.config.max_workers)
        
        # Add handlers based on configuration
        self._add_configured_handlers()
        
        # Create integrator
        self.integrator = StatusEventIntegrator(event_manager=self.event_manager)
        
        logger.info(f"Built event system with {len(self.event_manager.handlers)} handlers")
        return self.integrator
    
    def _add_configured_handlers(self):
        """Add handlers based on configuration."""
        # Clear default handlers if logging is disabled
        if not self.config.enable_logging:
            self.event_manager.handlers.clear()
        
        # Add logging handler
        if self.config.enable_logging:
            # Remove default logging handler and add configured one
            self.event_manager.handlers = [
                h for h in self.event_manager.handlers 
                if not isinstance(h, LoggingEventHandler)
            ]
            logging_handler = LoggingEventHandler(log_level=self.config.log_level)
            self.event_manager.add_handler(logging_handler)
        
        # Add database handler
        if self.config.enable_database:
            database_handler = DatabaseEventHandler(table_name=self.config.database_table_name)
            self.event_manager.add_handler(database_handler)
        
        # Add webhook handler
        if self.config.enable_webhooks and self.config.webhook_urls:
            webhook_handler = WebhookEventHandler(
                webhook_urls=self.config.webhook_urls,
                event_types=self.config.webhook_event_types
            )
            self.event_manager.add_handler(webhook_handler)
        
        # Add custom handlers
        for handler in self.config.custom_handlers:
            self.event_manager.add_handler(handler)


def create_default_event_system() -> StatusEventIntegrator:
    """Create a default event system with logging enabled."""
    return (EventSystemBuilder()
            .with_logging(enabled=True, log_level=logging.INFO)
            .build())


def create_production_event_system(
    webhook_urls: Optional[List[str]] = None,
    enable_database: bool = True
) -> StatusEventIntegrator:
    """Create a production-ready event system."""
    builder = (EventSystemBuilder()
               .with_max_workers(10)
               .with_logging(enabled=True, log_level=logging.WARNING)
               .with_database(enabled=enable_database))
    
    if webhook_urls:
        # Configure webhooks for important events only
        important_events = {
            EventType.STATUS_FAILED,
            EventType.ERROR_OCCURRED,
            EventType.WORKFLOW_FAILED,
            EventType.NODE_FAILED,
            EventType.HEARTBEAT_MISSED
        }
        builder.with_webhooks(webhook_urls, important_events)
    
    return builder.build()


def create_development_event_system() -> StatusEventIntegrator:
    """Create a development event system with detailed logging."""
    return (EventSystemBuilder()
            .with_max_workers(3)
            .with_logging(enabled=True, log_level=logging.DEBUG)
            .build())


class CustomEventHandlers:
    """Collection of custom event handlers for specific use cases."""
    
    @staticmethod
    def create_metrics_handler() -> EventHandler:
        """Create a handler that collects metrics from events."""
        
        class MetricsEventHandler(EventHandler):
            def __init__(self):
                self.metrics = {
                    'status_updates': 0,
                    'errors': 0,
                    'completions': 0,
                    'workflows_started': 0,
                    'workflows_completed': 0
                }
                self.logger = logging.getLogger(f"{__name__}.MetricsEventHandler")
            
            async def handle_event(self, event) -> bool:
                try:
                    if event.event_type == EventType.STATUS_UPDATED:
                        self.metrics['status_updates'] += 1
                    elif event.event_type in [EventType.ERROR_OCCURRED, EventType.STATUS_FAILED]:
                        self.metrics['errors'] += 1
                    elif event.event_type == EventType.STATUS_COMPLETED:
                        self.metrics['completions'] += 1
                    elif event.event_type == EventType.WORKFLOW_STARTED:
                        self.metrics['workflows_started'] += 1
                    elif event.event_type == EventType.WORKFLOW_COMPLETED:
                        self.metrics['workflows_completed'] += 1
                    
                    return True
                except Exception as e:
                    self.logger.error(f"Error in metrics handler: {e}")
                    return False
            
            def get_handled_event_types(self) -> Set[EventType]:
                return {
                    EventType.STATUS_UPDATED,
                    EventType.ERROR_OCCURRED,
                    EventType.STATUS_FAILED,
                    EventType.STATUS_COMPLETED,
                    EventType.WORKFLOW_STARTED,
                    EventType.WORKFLOW_COMPLETED
                }
            
            def get_metrics(self) -> Dict[str, int]:
                return self.metrics.copy()
        
        return MetricsEventHandler()
    
    @staticmethod
    def create_alert_handler(alert_callback) -> EventHandler:
        """Create a handler that sends alerts for critical events."""
        
        class AlertEventHandler(EventHandler):
            def __init__(self, alert_callback):
                self.alert_callback = alert_callback
                self.logger = logging.getLogger(f"{__name__}.AlertEventHandler")
            
            async def handle_event(self, event) -> bool:
                try:
                    # Create alert message
                    if event.event_type == EventType.STATUS_FAILED:
                        message = f"Status failed: {event.status_id} - {event.error_info}"
                    elif event.event_type == EventType.WORKFLOW_FAILED:
                        message = f"Workflow failed: {event.workflow_name} - {event.error_info}"
                    elif event.event_type == EventType.HEARTBEAT_MISSED:
                        message = f"Heartbeat missed: {event.worker_id} for status {event.status_id}"
                    else:
                        message = f"Critical event: {event.event_type.value} for {event.status_id}"
                    
                    # Send alert
                    await self.alert_callback(message, event)
                    return True
                    
                except Exception as e:
                    self.logger.error(f"Error in alert handler: {e}")
                    return False
            
            def get_handled_event_types(self) -> Set[EventType]:
                return {
                    EventType.STATUS_FAILED,
                    EventType.WORKFLOW_FAILED,
                    EventType.NODE_FAILED,
                    EventType.HEARTBEAT_MISSED,
                    EventType.STALE_STATUS_DETECTED
                }
            
            def get_priority_filter(self) -> Optional[Set[EventPriority]]:
                return {EventPriority.HIGH, EventPriority.CRITICAL}
        
        return AlertEventHandler(alert_callback)
    
    @staticmethod
    def create_file_logger_handler(file_path: str) -> EventHandler:
        """Create a handler that logs events to a file."""
        
        class FileLoggerEventHandler(EventHandler):
            def __init__(self, file_path: str):
                self.file_path = file_path
                self.logger = logging.getLogger(f"{__name__}.FileLoggerEventHandler")
                
                # Setup file handler
                self.file_handler = logging.FileHandler(file_path)
                self.file_handler.setFormatter(
                    logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
                )
                
                # Create logger for file output
                self.file_logger = logging.getLogger(f"{__name__}.FileLogger")
                self.file_logger.addHandler(self.file_handler)
                self.file_logger.setLevel(logging.INFO)
            
            async def handle_event(self, event) -> bool:
                try:
                    # Format event for file logging
                    message = (
                        f"Event: {event.event_type.value} | "
                        f"Status: {event.status_id} | "
                        f"Progress: {event.progress_percentage}% | "
                        f"Step: {event.current_step} | "
                        f"Worker: {event.worker_id}"
                    )
                    
                    if event.error_info:
                        message += f" | Error: {event.error_info}"
                    
                    self.file_logger.info(message)
                    return True
                    
                except Exception as e:
                    self.logger.error(f"Error in file logger handler: {e}")
                    return False
            
            def get_handled_event_types(self) -> Set[EventType]:
                return set(EventType)  # Handle all events
        
        return FileLoggerEventHandler(file_path)


# Configuration examples
def get_example_configurations() -> Dict[str, EventSystemConfig]:
    """Get example configurations for different environments."""
    
    return {
        'minimal': EventSystemConfig(
            max_workers=2,
            enable_logging=True,
            log_level=logging.WARNING,
            enable_database=False,
            enable_webhooks=False
        ),
        
        'development': EventSystemConfig(
            max_workers=3,
            enable_logging=True,
            log_level=logging.DEBUG,
            enable_database=True,
            enable_webhooks=False
        ),
        
        'staging': EventSystemConfig(
            max_workers=5,
            enable_logging=True,
            log_level=logging.INFO,
            enable_database=True,
            enable_webhooks=True,
            webhook_urls=['http://staging-alerts.example.com/webhook']
        ),
        
        'production': EventSystemConfig(
            max_workers=10,
            enable_logging=True,
            log_level=logging.WARNING,
            enable_database=True,
            enable_webhooks=True,
            webhook_urls=[
                'http://alerts.example.com/webhook',
                'http://monitoring.example.com/webhook'
            ],
            webhook_event_types={
                EventType.STATUS_FAILED,
                EventType.ERROR_OCCURRED,
                EventType.WORKFLOW_FAILED,
                EventType.HEARTBEAT_MISSED
            }
        )
    }