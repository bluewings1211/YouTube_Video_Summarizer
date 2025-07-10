"""
Real-time status WebSocket API endpoints.
Provides WebSocket-based real-time status updates and monitoring.
"""

import json
import logging
from typing import Optional, Dict, Any
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, Query
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session

from ..database.connection import get_db_session
from ..services.realtime_status_service import RealTimeStatusService


# Global real-time service instance
realtime_service: Optional[RealTimeStatusService] = None

# Router for real-time endpoints
router = APIRouter(prefix="/api/realtime", tags=["Real-time Status"])

logger = logging.getLogger(__name__)


def get_realtime_service(db: Session = Depends(get_db_session)) -> RealTimeStatusService:
    """Dependency to get RealTimeStatusService instance."""
    global realtime_service
    if realtime_service is None:
        realtime_service = RealTimeStatusService(db_session=db)
    return realtime_service


@router.websocket("/status")
async def websocket_status_endpoint(
    websocket: WebSocket,
    client_id: Optional[str] = Query(None, description="Client identifier"),
    service: RealTimeStatusService = Depends(get_realtime_service)
):
    """
    WebSocket endpoint for real-time status updates.
    
    Supported message types:
    - subscribe: Subscribe to status updates
    - unsubscribe: Unsubscribe from updates
    - ping: Keep-alive ping
    - get_status: Get specific status
    - get_metrics: Get performance metrics
    """
    await service.connect_websocket(websocket, client_id)
    
    try:
        while True:
            # Receive message from client
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                await service.handle_websocket_message(websocket, message)
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON received: {e}")
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "event": "invalid_json",
                    "data": {"error": "Invalid JSON format"},
                    "timestamp": "now"
                }))
            except Exception as e:
                logger.error(f"Error processing WebSocket message: {e}")
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "event": "processing_error",
                    "data": {"error": str(e)},
                    "timestamp": "now"
                }))
    
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {client_id or 'anonymous'}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        service.disconnect_websocket(websocket)


@router.get("/connections/stats")
async def get_connection_stats(
    service: RealTimeStatusService = Depends(get_realtime_service)
):
    """Get real-time connection statistics."""
    return service.get_connection_stats()


@router.post("/broadcast")
async def broadcast_message(
    message: str,
    event_type: str = "system_notification",
    service: RealTimeStatusService = Depends(get_realtime_service)
):
    """Broadcast a system message to all connected clients."""
    await service.broadcast_system_message(message, event_type)
    return {"message": "Broadcast sent", "event_type": event_type}


@router.post("/metrics/broadcast")
async def broadcast_metrics_update(
    service: RealTimeStatusService = Depends(get_realtime_service)
):
    """Manually trigger performance metrics broadcast."""
    await service.send_performance_metrics_update()
    return {"message": "Metrics update broadcasted"}


@router.get("/demo")
async def get_demo_page():
    """Get demo HTML page for testing WebSocket connections."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Real-time Status Tracking Demo</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .container {
                background-color: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                margin-bottom: 20px;
            }
            .controls {
                display: flex;
                gap: 10px;
                align-items: center;
                flex-wrap: wrap;
            }
            button {
                padding: 8px 16px;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                font-size: 14px;
            }
            .primary { background-color: #007bff; color: white; }
            .success { background-color: #28a745; color: white; }
            .danger { background-color: #dc3545; color: white; }
            .warning { background-color: #ffc107; color: black; }
            input, select {
                padding: 8px;
                border: 1px solid #ddd;
                border-radius: 4px;
                font-size: 14px;
            }
            .status {
                padding: 10px;
                border-radius: 4px;
                margin: 10px 0;
                font-weight: bold;
            }
            .connected { background-color: #d4edda; color: #155724; }
            .disconnected { background-color: #f8d7da; color: #721c24; }
            .log {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 4px;
                padding: 15px;
                height: 400px;
                overflow-y: auto;
                font-family: monospace;
                font-size: 12px;
                white-space: pre-wrap;
            }
            .message {
                margin: 5px 0;
                padding: 5px;
                border-left: 3px solid #007bff;
                background-color: #e7f3ff;
            }
            .error {
                border-left-color: #dc3545;
                background-color: #f8d7da;
            }
            .success {
                border-left-color: #28a745;
                background-color: #d4edda;
            }
        </style>
    </head>
    <body>
        <h1>Real-time Status Tracking Demo</h1>
        
        <div class="container">
            <h2>Connection</h2>
            <div class="controls">
                <input type="text" id="clientId" placeholder="Client ID (optional)" value="demo-client">
                <button id="connectBtn" class="primary">Connect</button>
                <button id="disconnectBtn" class="danger" disabled>Disconnect</button>
                <button id="pingBtn" class="warning" disabled>Ping</button>
                <div id="connectionStatus" class="status disconnected">Disconnected</div>
            </div>
        </div>
        
        <div class="container">
            <h2>Subscriptions</h2>
            <div class="controls">
                <select id="subscriptionType">
                    <option value="all_statuses">All Statuses</option>
                    <option value="specific_status">Specific Status</option>
                    <option value="video_status">Video Status</option>
                    <option value="batch_status">Batch Status</option>
                    <option value="worker_status">Worker Status</option>
                    <option value="status_type">Status Type</option>
                    <option value="performance_metrics">Performance Metrics</option>
                </select>
                <input type="text" id="filterValue" placeholder="Filter value (if needed)">
                <button id="subscribeBtn" class="success" disabled>Subscribe</button>
                <button id="unsubscribeBtn" class="danger" disabled>Unsubscribe All</button>
            </div>
            <div id="subscriptions"></div>
        </div>
        
        <div class="container">
            <h2>Quick Actions</h2>
            <div class="controls">
                <input type="text" id="statusId" placeholder="Status ID">
                <button id="getStatusBtn" class="primary" disabled>Get Status</button>
                <button id="getMetricsBtn" class="primary" disabled>Get Metrics</button>
            </div>
        </div>
        
        <div class="container">
            <h2>Message Log</h2>
            <div class="controls">
                <button id="clearLogBtn" class="warning">Clear Log</button>
            </div>
            <div id="messageLog" class="log"></div>
        </div>
        
        <script>
            let ws = null;
            let subscriptions = new Map();
            let messageCounter = 0;
            
            const elements = {
                connectBtn: document.getElementById('connectBtn'),
                disconnectBtn: document.getElementById('disconnectBtn'),
                pingBtn: document.getElementById('pingBtn'),
                subscribeBtn: document.getElementById('subscribeBtn'),
                unsubscribeBtn: document.getElementById('unsubscribeBtn'),
                getStatusBtn: document.getElementById('getStatusBtn'),
                getMetricsBtn: document.getElementById('getMetricsBtn'),
                clearLogBtn: document.getElementById('clearLogBtn'),
                connectionStatus: document.getElementById('connectionStatus'),
                clientId: document.getElementById('clientId'),
                subscriptionType: document.getElementById('subscriptionType'),
                filterValue: document.getElementById('filterValue'),
                statusId: document.getElementById('statusId'),
                messageLog: document.getElementById('messageLog'),
                subscriptions: document.getElementById('subscriptions')
            };
            
            function log(message, type = 'info') {
                const timestamp = new Date().toISOString();
                const logEntry = document.createElement('div');
                logEntry.className = `message ${type}`;
                logEntry.textContent = `[${timestamp}] ${message}`;
                elements.messageLog.appendChild(logEntry);
                elements.messageLog.scrollTop = elements.messageLog.scrollHeight;
            }
            
            function updateConnectionStatus(connected) {
                elements.connectionStatus.textContent = connected ? 'Connected' : 'Disconnected';
                elements.connectionStatus.className = `status ${connected ? 'connected' : 'disconnected'}`;
                
                elements.connectBtn.disabled = connected;
                elements.disconnectBtn.disabled = !connected;
                elements.pingBtn.disabled = !connected;
                elements.subscribeBtn.disabled = !connected;
                elements.unsubscribeBtn.disabled = !connected;
                elements.getStatusBtn.disabled = !connected;
                elements.getMetricsBtn.disabled = !connected;
            }
            
            function connect() {
                const clientId = elements.clientId.value;
                const wsUrl = `ws://localhost:8000/api/realtime/status${clientId ? `?client_id=${clientId}` : ''}`;
                
                ws = new WebSocket(wsUrl);
                
                ws.onopen = function(event) {
                    log('WebSocket connected', 'success');
                    updateConnectionStatus(true);
                };
                
                ws.onmessage = function(event) {
                    const message = JSON.parse(event.data);
                    log(`Received: ${JSON.stringify(message, null, 2)}`);
                };
                
                ws.onclose = function(event) {
                    log(`WebSocket closed: ${event.code} ${event.reason}`, 'error');
                    updateConnectionStatus(false);
                };
                
                ws.onerror = function(error) {
                    log(`WebSocket error: ${error}`, 'error');
                };
            }
            
            function disconnect() {
                if (ws) {
                    ws.close();
                    ws = null;
                }
                subscriptions.clear();
                updateSubscriptionsDisplay();
            }
            
            function ping() {
                if (ws) {
                    const message = {
                        type: 'ping',
                        id: `ping_${++messageCounter}`
                    };
                    ws.send(JSON.stringify(message));
                    log(`Sent: ${JSON.stringify(message)}`);
                }
            }
            
            function subscribe() {
                if (ws) {
                    const subscriptionType = elements.subscriptionType.value;
                    const filterValue = elements.filterValue.value || null;
                    const subscriptionId = `sub_${++messageCounter}`;
                    
                    const message = {
                        type: 'subscribe',
                        subscription_type: subscriptionType,
                        filter_value: filterValue,
                        subscription_id: subscriptionId,
                        client_id: elements.clientId.value || null,
                        id: `subscribe_${messageCounter}`
                    };
                    
                    ws.send(JSON.stringify(message));
                    log(`Sent: ${JSON.stringify(message)}`);
                    
                    subscriptions.set(subscriptionId, {
                        type: subscriptionType,
                        filter: filterValue
                    });
                    updateSubscriptionsDisplay();
                }
            }
            
            function unsubscribeAll() {
                subscriptions.forEach((_, subscriptionId) => {
                    if (ws) {
                        const message = {
                            type: 'unsubscribe',
                            subscription_id: subscriptionId,
                            id: `unsubscribe_${++messageCounter}`
                        };
                        ws.send(JSON.stringify(message));
                        log(`Sent: ${JSON.stringify(message)}`);
                    }
                });
                subscriptions.clear();
                updateSubscriptionsDisplay();
            }
            
            function getStatus() {
                if (ws) {
                    const statusId = elements.statusId.value;
                    if (!statusId) {
                        alert('Please enter a status ID');
                        return;
                    }
                    
                    const message = {
                        type: 'get_status',
                        status_id: statusId,
                        id: `get_status_${++messageCounter}`
                    };
                    
                    ws.send(JSON.stringify(message));
                    log(`Sent: ${JSON.stringify(message)}`);
                }
            }
            
            function getMetrics() {
                if (ws) {
                    const message = {
                        type: 'get_metrics',
                        id: `get_metrics_${++messageCounter}`
                    };
                    
                    ws.send(JSON.stringify(message));
                    log(`Sent: ${JSON.stringify(message)}`);
                }
            }
            
            function updateSubscriptionsDisplay() {
                elements.subscriptions.innerHTML = '';
                subscriptions.forEach((sub, id) => {
                    const div = document.createElement('div');
                    div.textContent = `${id}: ${sub.type}${sub.filter ? ` (${sub.filter})` : ''}`;
                    elements.subscriptions.appendChild(div);
                });
            }
            
            function clearLog() {
                elements.messageLog.innerHTML = '';
            }
            
            // Event listeners
            elements.connectBtn.addEventListener('click', connect);
            elements.disconnectBtn.addEventListener('click', disconnect);
            elements.pingBtn.addEventListener('click', ping);
            elements.subscribeBtn.addEventListener('click', subscribe);
            elements.unsubscribeBtn.addEventListener('click', unsubscribeAll);
            elements.getStatusBtn.addEventListener('click', getStatus);
            elements.getMetricsBtn.addEventListener('click', getMetrics);
            elements.clearLogBtn.addEventListener('click', clearLog);
            
            // Initialize
            updateConnectionStatus(false);
            log('Demo page loaded');
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


# Startup and shutdown events for background tasks
async def startup_realtime_service():
    """Start background tasks for real-time service."""
    global realtime_service
    if realtime_service:
        await realtime_service.start_background_tasks()
        logger.info("Real-time status service background tasks started")


async def shutdown_realtime_service():
    """Stop background tasks for real-time service."""
    global realtime_service
    if realtime_service:
        await realtime_service.stop_background_tasks()
        logger.info("Real-time status service background tasks stopped")