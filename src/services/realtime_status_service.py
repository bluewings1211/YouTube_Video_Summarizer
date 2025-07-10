"""
Real-time status service for WebSocket-based status updates.
Provides real-time status broadcasting and subscription capabilities.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Set, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from weakref import WeakSet
from collections import defaultdict

from fastapi import WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session

from ..database.connection import get_db_session
from ..database.status_models import ProcessingStatusType, StatusChangeType
from .status_service import StatusService
from .status_updater import StatusUpdater, StatusUpdate, ProgressUpdate, ErrorUpdate


class SubscriptionType(Enum):
    """WebSocket subscription types."""
    ALL_STATUSES = "all_statuses"
    SPECIFIC_STATUS = "specific_status"
    VIDEO_STATUS = "video_status"
    BATCH_STATUS = "batch_status"
    WORKER_STATUS = "worker_status"
    STATUS_TYPE = "status_type"
    PERFORMANCE_METRICS = "performance_metrics"


@dataclass
class WebSocketMessage:
    """WebSocket message structure."""
    type: str
    event: str
    data: Dict[str, Any]
    timestamp: datetime
    subscription_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "type": self.type,
            "event": self.event,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "subscription_id": self.subscription_id
        }


@dataclass
class Subscription:
    """WebSocket subscription details."""
    websocket: WebSocket
    subscription_type: SubscriptionType
    filter_value: Optional[str] = None
    subscription_id: Optional[str] = None
    client_id: Optional[str] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
    
    def matches(self, event_data: Dict[str, Any]) -> bool:
        """Check if this subscription matches the event data."""
        if self.subscription_type == SubscriptionType.ALL_STATUSES:
            return True
        elif self.subscription_type == SubscriptionType.SPECIFIC_STATUS:
            return event_data.get("status_id") == self.filter_value
        elif self.subscription_type == SubscriptionType.VIDEO_STATUS:
            return str(event_data.get("video_id")) == self.filter_value
        elif self.subscription_type == SubscriptionType.BATCH_STATUS:
            return str(event_data.get("batch_item_id")) == self.filter_value
        elif self.subscription_type == SubscriptionType.WORKER_STATUS:
            return event_data.get("worker_id") == self.filter_value
        elif self.subscription_type == SubscriptionType.STATUS_TYPE:
            return event_data.get("status") == self.filter_value
        elif self.subscription_type == SubscriptionType.PERFORMANCE_METRICS:
            return True  # Performance metrics are global
        return False


class ConnectionManager:
    """WebSocket connection manager."""
    
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.subscriptions: Dict[WebSocket, List[Subscription]] = defaultdict(list)
        self.subscription_index: Dict[str, List[Subscription]] = defaultdict(list)
        self.logger = logging.getLogger(__name__)
    
    async def connect(self, websocket: WebSocket, client_id: Optional[str] = None):
        """Accept WebSocket connection."""
        await websocket.accept()
        self.active_connections.add(websocket)
        self.logger.info(f"WebSocket connected: {client_id or 'anonymous'}")
        
        # Send welcome message
        welcome_msg = WebSocketMessage(
            type="system",
            event="connected",
            data={
                "message": "Connected to status tracking service",
                "client_id": client_id,
                "server_time": datetime.utcnow().isoformat()
            },
            timestamp=datetime.utcnow()
        )
        await self.send_personal_message(websocket, welcome_msg)
    
    def disconnect(self, websocket: WebSocket):
        """Disconnect WebSocket and clean up subscriptions."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        
        # Clean up subscriptions
        if websocket in self.subscriptions:
            for subscription in self.subscriptions[websocket]:
                # Remove from subscription index
                for key, subs in self.subscription_index.items():
                    if subscription in subs:
                        subs.remove(subscription)
            del self.subscriptions[websocket]
        
        self.logger.info("WebSocket disconnected and cleaned up")
    
    async def send_personal_message(self, websocket: WebSocket, message: WebSocketMessage):
        """Send message to specific WebSocket connection."""
        try:
            await websocket.send_text(json.dumps(message.to_dict()))
        except Exception as e:
            self.logger.error(f"Error sending message to WebSocket: {e}")
            self.disconnect(websocket)
    
    async def broadcast_message(self, message: WebSocketMessage):
        """Broadcast message to all connected clients."""
        if not self.active_connections:
            return
        
        message_json = json.dumps(message.to_dict())
        disconnected = set()
        
        for websocket in self.active_connections.copy():
            try:
                await websocket.send_text(message_json)
            except Exception as e:
                self.logger.error(f"Error broadcasting to WebSocket: {e}")
                disconnected.add(websocket)
        
        # Clean up disconnected connections
        for websocket in disconnected:
            self.disconnect(websocket)
    
    async def send_to_subscribers(self, event_type: str, event_data: Dict[str, Any]):
        """Send message to subscribers based on their subscriptions."""
        if not self.subscriptions:
            return
        
        message = WebSocketMessage(
            type="status_update",
            event=event_type,
            data=event_data,
            timestamp=datetime.utcnow()
        )
        
        sent_to = set()
        
        for websocket, subscriptions in self.subscriptions.items():
            if websocket not in self.active_connections:
                continue
            
            for subscription in subscriptions:
                if subscription.matches(event_data):
                    if websocket not in sent_to:
                        await self.send_personal_message(websocket, message)
                        sent_to.add(websocket)
                    break
    
    def subscribe(
        self,
        websocket: WebSocket,
        subscription_type: SubscriptionType,
        filter_value: Optional[str] = None,
        subscription_id: Optional[str] = None,
        client_id: Optional[str] = None
    ) -> str:
        """Add subscription for WebSocket."""
        subscription = Subscription(
            websocket=websocket,
            subscription_type=subscription_type,
            filter_value=filter_value,
            subscription_id=subscription_id,
            client_id=client_id
        )
        
        self.subscriptions[websocket].append(subscription)
        
        # Add to subscription index for efficient lookup
        index_key = f"{subscription_type.value}:{filter_value or 'all'}"
        self.subscription_index[index_key].append(subscription)
        
        self.logger.info(f"Added subscription: {subscription_type.value} with filter: {filter_value}")
        return subscription_id or f"sub_{datetime.utcnow().timestamp()}"
    
    def unsubscribe(self, websocket: WebSocket, subscription_id: str) -> bool:
        """Remove specific subscription."""
        if websocket not in self.subscriptions:
            return False
        
        for i, subscription in enumerate(self.subscriptions[websocket]):
            if subscription.subscription_id == subscription_id:
                # Remove from main subscriptions
                del self.subscriptions[websocket][i]
                
                # Remove from subscription index
                for key, subs in self.subscription_index.items():
                    if subscription in subs:
                        subs.remove(subscription)
                        break
                
                self.logger.info(f"Removed subscription: {subscription_id}")
                return True
        
        return False
    
    def get_connection_count(self) -> int:
        """Get number of active connections."""
        return len(self.active_connections)
    
    def get_subscription_count(self) -> int:
        """Get total number of subscriptions."""
        return sum(len(subs) for subs in self.subscriptions.values())


class RealTimeStatusService:
    """
    Real-time status service for WebSocket-based status updates.
    
    This service provides real-time status broadcasting, subscription management,
    and WebSocket connection handling for the status tracking system.
    """
    
    def __init__(self, db_session: Optional[Session] = None):
        """
        Initialize the RealTimeStatusService.
        
        Args:
            db_session: Optional database session
        """
        self.db_session = db_session
        self._should_close_session = db_session is None
        self.connection_manager = ConnectionManager()
        self.status_service = StatusService(db_session)
        self.status_updater = StatusUpdater(db_session)
        self.logger = logging.getLogger(__name__)
        
        # Performance metrics cache
        self._metrics_cache = {}
        self._metrics_cache_expiry = datetime.utcnow()
        self._metrics_cache_ttl = 30  # seconds
        
        # Setup event handlers
        self._setup_event_handlers()
        
        # Background tasks
        self._background_tasks = set()
        self._shutdown_flag = False
    
    def __enter__(self):
        """Context manager entry."""
        if self.db_session is None:
            self.db_session = get_db_session()
        self.status_service.db_session = self.db_session
        self.status_updater.db_session = self.db_session
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self._should_close_session and self.db_session:
            self.db_session.close()
    
    def _setup_event_handlers(self):
        """Setup event handlers for status updates."""
        # Subscribe to status updater events
        self.status_updater.event_handler.subscribe(
            'status_changed', 
            self._handle_status_changed
        )
        self.status_updater.event_handler.subscribe(
            'progress_updated', 
            self._handle_progress_updated
        )
        self.status_updater.event_handler.subscribe(
            'error_occurred', 
            self._handle_error_occurred
        )
        self.status_updater.event_handler.subscribe(
            'processing_completed', 
            self._handle_processing_completed
        )
        self.status_updater.event_handler.subscribe(
            'processing_failed', 
            self._handle_processing_failed
        )
        self.status_updater.event_handler.subscribe(
            'heartbeat_missed', 
            self._handle_heartbeat_missed
        )
    
    async def _handle_status_changed(self, event_data: Dict[str, Any]):
        """Handle status changed events."""
        await self.connection_manager.send_to_subscribers(
            "status_changed", 
            event_data
        )
    
    async def _handle_progress_updated(self, event_data: Dict[str, Any]):
        """Handle progress updated events."""
        await self.connection_manager.send_to_subscribers(
            "progress_updated", 
            event_data
        )
    
    async def _handle_error_occurred(self, event_data: Dict[str, Any]):
        """Handle error occurred events."""
        await self.connection_manager.send_to_subscribers(
            "error_occurred", 
            event_data
        )
    
    async def _handle_processing_completed(self, event_data: Dict[str, Any]):
        """Handle processing completed events."""
        await self.connection_manager.send_to_subscribers(
            "processing_completed", 
            event_data
        )
    
    async def _handle_processing_failed(self, event_data: Dict[str, Any]):
        """Handle processing failed events."""
        await self.connection_manager.send_to_subscribers(
            "processing_failed", 
            event_data
        )
    
    async def _handle_heartbeat_missed(self, event_data: Dict[str, Any]):
        """Handle heartbeat missed events."""
        await self.connection_manager.send_to_subscribers(
            "heartbeat_missed", 
            event_data
        )
    
    async def connect_websocket(self, websocket: WebSocket, client_id: Optional[str] = None):
        """Connect WebSocket client."""
        await self.connection_manager.connect(websocket, client_id)
    
    def disconnect_websocket(self, websocket: WebSocket):
        """Disconnect WebSocket client."""
        self.connection_manager.disconnect(websocket)
    
    async def handle_websocket_message(self, websocket: WebSocket, message: Dict[str, Any]):
        """Handle incoming WebSocket message."""
        try:
            message_type = message.get("type")
            
            if message_type == "subscribe":
                await self._handle_subscribe_message(websocket, message)
            elif message_type == "unsubscribe":
                await self._handle_unsubscribe_message(websocket, message)
            elif message_type == "ping":
                await self._handle_ping_message(websocket, message)
            elif message_type == "get_status":
                await self._handle_get_status_message(websocket, message)
            elif message_type == "get_metrics":
                await self._handle_get_metrics_message(websocket, message)
            else:
                await self._send_error_message(
                    websocket, 
                    f"Unknown message type: {message_type}",
                    message.get("id")
                )
        
        except Exception as e:
            self.logger.error(f"Error handling WebSocket message: {e}")
            await self._send_error_message(websocket, str(e), message.get("id"))
    
    async def _handle_subscribe_message(self, websocket: WebSocket, message: Dict[str, Any]):
        """Handle subscription message."""
        try:
            subscription_type = SubscriptionType(message["subscription_type"])
            filter_value = message.get("filter_value")
            subscription_id = message.get("subscription_id")
            client_id = message.get("client_id")
            
            sub_id = self.connection_manager.subscribe(
                websocket=websocket,
                subscription_type=subscription_type,
                filter_value=filter_value,
                subscription_id=subscription_id,
                client_id=client_id
            )
            
            response = WebSocketMessage(
                type="system",
                event="subscribed",
                data={
                    "subscription_id": sub_id,
                    "subscription_type": subscription_type.value,
                    "filter_value": filter_value,
                    "message": "Successfully subscribed"
                },
                timestamp=datetime.utcnow(),
                subscription_id=message.get("id")
            )
            
            await self.connection_manager.send_personal_message(websocket, response)
        
        except Exception as e:
            await self._send_error_message(websocket, f"Subscription error: {e}", message.get("id"))
    
    async def _handle_unsubscribe_message(self, websocket: WebSocket, message: Dict[str, Any]):
        """Handle unsubscription message."""
        try:
            subscription_id = message["subscription_id"]
            success = self.connection_manager.unsubscribe(websocket, subscription_id)
            
            response = WebSocketMessage(
                type="system",
                event="unsubscribed" if success else "unsubscribe_failed",
                data={
                    "subscription_id": subscription_id,
                    "success": success,
                    "message": "Successfully unsubscribed" if success else "Subscription not found"
                },
                timestamp=datetime.utcnow(),
                subscription_id=message.get("id")
            )
            
            await self.connection_manager.send_personal_message(websocket, response)
        
        except Exception as e:
            await self._send_error_message(websocket, f"Unsubscription error: {e}", message.get("id"))
    
    async def _handle_ping_message(self, websocket: WebSocket, message: Dict[str, Any]):
        """Handle ping message."""
        response = WebSocketMessage(
            type="system",
            event="pong",
            data={
                "server_time": datetime.utcnow().isoformat(),
                "connections": self.connection_manager.get_connection_count(),
                "subscriptions": self.connection_manager.get_subscription_count()
            },
            timestamp=datetime.utcnow(),
            subscription_id=message.get("id")
        )
        
        await self.connection_manager.send_personal_message(websocket, response)
    
    async def _handle_get_status_message(self, websocket: WebSocket, message: Dict[str, Any]):
        """Handle get status message."""
        try:
            status_id = message["status_id"]
            status = self.status_service.get_processing_status(status_id)
            
            if status:
                # Convert status to dict (would need proper serialization)
                status_data = {
                    "status_id": status.status_id,
                    "status": status.status.value,
                    "progress_percentage": status.progress_percentage,
                    "current_step": status.current_step,
                    "worker_id": status.worker_id,
                    "updated_at": status.updated_at.isoformat() if status.updated_at else None
                }
            else:
                status_data = None
            
            response = WebSocketMessage(
                type="data",
                event="status_response",
                data={
                    "status_id": status_id,
                    "status": status_data,
                    "found": status is not None
                },
                timestamp=datetime.utcnow(),
                subscription_id=message.get("id")
            )
            
            await self.connection_manager.send_personal_message(websocket, response)
        
        except Exception as e:
            await self._send_error_message(websocket, f"Status query error: {e}", message.get("id"))
    
    async def _handle_get_metrics_message(self, websocket: WebSocket, message: Dict[str, Any]):
        """Handle get metrics message."""
        try:
            # Check cache
            now = datetime.utcnow()
            if now > self._metrics_cache_expiry:
                # Update cache
                from .status_metrics_service import StatusMetricsService
                with StatusMetricsService(self.db_session) as metrics_service:
                    self._metrics_cache = metrics_service.get_current_performance_summary()
                    self._metrics_cache_expiry = now + timedelta(seconds=self._metrics_cache_ttl)
            
            response = WebSocketMessage(
                type="data",
                event="metrics_response",
                data=self._metrics_cache,
                timestamp=datetime.utcnow(),
                subscription_id=message.get("id")
            )
            
            await self.connection_manager.send_personal_message(websocket, response)
        
        except Exception as e:
            await self._send_error_message(websocket, f"Metrics query error: {e}", message.get("id"))
    
    async def _send_error_message(self, websocket: WebSocket, error: str, message_id: Optional[str] = None):
        """Send error message to WebSocket client."""
        error_msg = WebSocketMessage(
            type="error",
            event="error",
            data={
                "error": error,
                "timestamp": datetime.utcnow().isoformat()
            },
            timestamp=datetime.utcnow(),
            subscription_id=message_id
        )
        
        await self.connection_manager.send_personal_message(websocket, error_msg)
    
    async def broadcast_system_message(self, message: str, event_type: str = "system_notification"):
        """Broadcast system message to all connected clients."""
        broadcast_msg = WebSocketMessage(
            type="system",
            event=event_type,
            data={
                "message": message,
                "timestamp": datetime.utcnow().isoformat()
            },
            timestamp=datetime.utcnow()
        )
        
        await self.connection_manager.broadcast_message(broadcast_msg)
    
    async def send_performance_metrics_update(self):
        """Send performance metrics update to subscribers."""
        try:
            from .status_metrics_service import StatusMetricsService
            with StatusMetricsService(self.db_session) as metrics_service:
                metrics = metrics_service.get_current_performance_summary()
            
            await self.connection_manager.send_to_subscribers(
                "performance_metrics_updated",
                metrics
            )
        except Exception as e:
            self.logger.error(f"Error sending performance metrics update: {e}")
    
    async def start_background_tasks(self):
        """Start background tasks for periodic updates."""
        self._shutdown_flag = False
        
        # Start performance metrics broadcaster
        task = asyncio.create_task(self._periodic_metrics_broadcast())
        self._background_tasks.add(task)
        
        # Start stale connection cleaner
        task = asyncio.create_task(self._periodic_connection_cleanup())
        self._background_tasks.add(task)
    
    async def stop_background_tasks(self):
        """Stop all background tasks."""
        self._shutdown_flag = True
        
        for task in self._background_tasks:
            if not task.done():
                task.cancel()
        
        await asyncio.gather(*self._background_tasks, return_exceptions=True)
        self._background_tasks.clear()
    
    async def _periodic_metrics_broadcast(self):
        """Periodically broadcast performance metrics."""
        while not self._shutdown_flag:
            try:
                await self.send_performance_metrics_update()
                await asyncio.sleep(60)  # Broadcast every minute
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in periodic metrics broadcast: {e}")
                await asyncio.sleep(60)
    
    async def _periodic_connection_cleanup(self):
        """Periodically clean up stale connections."""
        while not self._shutdown_flag:
            try:
                # Clean up stale connections
                stale_connections = []
                for websocket in self.connection_manager.active_connections.copy():
                    try:
                        await websocket.ping()
                    except Exception:
                        stale_connections.append(websocket)
                
                for websocket in stale_connections:
                    self.connection_manager.disconnect(websocket)
                
                if stale_connections:
                    self.logger.info(f"Cleaned up {len(stale_connections)} stale connections")
                
                await asyncio.sleep(300)  # Check every 5 minutes
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in connection cleanup: {e}")
                await asyncio.sleep(300)
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics."""
        return {
            "active_connections": self.connection_manager.get_connection_count(),
            "total_subscriptions": self.connection_manager.get_subscription_count(),
            "subscription_types": {
                sub_type.value: len([
                    sub for subs in self.connection_manager.subscriptions.values()
                    for sub in subs
                    if sub.subscription_type == sub_type
                ])
                for sub_type in SubscriptionType
            },
            "background_tasks": len(self._background_tasks),
            "metrics_cache_expires": self._metrics_cache_expiry.isoformat()
        }