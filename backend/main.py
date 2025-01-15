# backend/app/main.py
from fastapi import FastAPI, WebSocket, HTTPException, status, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
from typing import Optional, Dict
from datetime import datetime
import json
import asyncio
from bot import BotManager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Kryptos Trading API")

# Initialize bot manager
bot_manager = BotManager()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://kryptostrading.com",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class APIResponse(BaseModel):
    status: str
    message: str
    data: Optional[dict] = None

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket

    def disconnect(self, client_id: str):
        self.active_connections.pop(client_id, None)

    async def send_update(self, client_id: str, data: dict):
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_json(data)
            except Exception as e:
                logger.error(f"Error sending update to {client_id}: {str(e)}")
                await self.disconnect(client_id)

manager = ConnectionManager()

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Start the trading bots when the application starts"""
    try:
        asyncio.create_task(bot_manager.start_bots())
        logger.info("Trading bots started successfully")
    except Exception as e:
        logger.error(f"Error starting bots: {str(e)}")

@app.on_event("shutdown")
async def shutdown_event():
    """Stop the trading bots when the application shuts down"""
    try:
        await bot_manager.stop_bots()
        logger.info("Trading bots stopped successfully")
    except Exception as e:
        logger.error(f"Error stopping bots: {str(e)}")

# API Endpoints
@app.get("/api/bot-status/{user_id}")
async def get_bot_status(user_id: int):
    """Get status for both demo and live bots"""
    try:
        demo_status = bot_manager.get_demo_bot_status()
        live_status = bot_manager.get_live_bot_status()
        return APIResponse(
            status="success",
            message="Bot status retrieved successfully",
            data={
                "demo": demo_status,
                "live": live_status,
                "running": bot_manager.running
            }
        )
    except Exception as e:
        logger.error(f"Error getting bot status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.post("/api/start-bot/{user_id}")
async def start_bot(user_id: int):
    """Start both demo and live bots"""
    try:
        await bot_manager.start_bots()
        return APIResponse(
            status="success",
            message="Bots started successfully"
        )
    except Exception as e:
        logger.error(f"Error starting bots: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.post("/api/stop-bot/{user_id}")
async def stop_bot(user_id: int):
    """Stop both demo and live bots"""
    try:
        await bot_manager.stop_bots()
        return APIResponse(
            status="success",
            message="Bots stopped successfully"
        )
    except Exception as e:
        logger.error(f"Error stopping bots: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.get("/api/demo-status")
async def get_demo_status():
    """Get demo bot status"""
    try:
        status_data = bot_manager.get_demo_bot_status()
        return APIResponse(
            status="success",
            message="Demo bot status retrieved successfully",
            data=status_data
        )
    except Exception as e:
        logger.error(f"Error getting demo status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.get("/api/live-status")
async def get_live_status():
    """Get live bot status"""
    try:
        status_data = bot_manager.get_live_bot_status()
        return APIResponse(
            status="success",
            message="Live bot status retrieved successfully",
            data=status_data
        )
    except Exception as e:
        logger.error(f"Error getting live status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.get("/api/metrics")
async def get_combined_metrics():
    """Get combined metrics from both bots"""
    try:
        metrics = bot_manager.get_combined_metrics()
        return APIResponse(
            status="success",
            message="Metrics retrieved successfully",
            data=metrics
        )
    except Exception as e:
        logger.error(f"Error getting metrics: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.get("/api/trade-history/{bot_type}")
async def get_trade_history(bot_type: str):
    """Get trade history for specified bot"""
    try:
        history = bot_manager.get_trade_history(bot_type)
        return APIResponse(
            status="success",
            message=f"{bot_type.capitalize()} trade history retrieved successfully",
            data={"trades": history}
        )
    except Exception as e:
        logger.error(f"Error getting trade history: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.get("/api/portfolio-history/{bot_type}")
async def get_portfolio_history(bot_type: str):
    """Get portfolio history for specified bot"""
    try:
        history = bot_manager.get_portfolio_history(bot_type)
        return APIResponse(
            status="success",
            message=f"{bot_type.capitalize()} portfolio history retrieved successfully",
            data={"portfolio": history}
        )
    except Exception as e:
        logger.error(f"Error getting portfolio history: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

# WebSocket endpoint for real-time updates
@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket connection for real-time updates"""
    try:
        await manager.connect(websocket, client_id)
        try:
            while True:
                data = await websocket.receive_text()
                try:
                    message = json.loads(data)
                    if message.get("type") == "ping":
                        await websocket.send_json({"type": "pong"})
                except json.JSONDecodeError:
                    pass
        except WebSocketDisconnect:
            manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        manager.disconnect(client_id)

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "demo_bot": bot_manager.get_demo_bot_status().get("status", "unknown"),
        "live_bot": bot_manager.get_live_bot_status().get("status", "unknown")
    }

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return {
        "status": "error",
        "message": str(exc.detail),
        "code": exc.status_code
    }

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    return {
        "status": "error",
        "message": "Internal server error",
        "code": 500
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
