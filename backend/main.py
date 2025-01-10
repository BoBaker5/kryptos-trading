# backend/app/main.py
from fastapi import FastAPI, WebSocket, HTTPException, Depends, status, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
from typing import Optional, Dict
from datetime import datetime
import json
import asyncio

from .bot_manager import BotManager
from .database import get_db
from sqlalchemy.orm import Session

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
    allow_origins=["https://kryptostrading.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class BotCredentials(BaseModel):
    api_key: str
    secret_key: str

class APIResponse(BaseModel):
    status: str
    message: str
    data: Optional[dict] = None

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[int, WebSocket] = {}

    async def connect(self, user_id: int, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[user_id] = websocket
        await bot_manager.register_websocket(user_id, websocket)

    def disconnect(self, user_id: int):
        self.active_connections.pop(user_id, None)
        asyncio.create_task(bot_manager.remove_websocket(user_id))

    async def send_personal_message(self, message: str, user_id: int):
        if user_id in self.active_connections:
            await self.active_connections[user_id].send_text(message)

manager = ConnectionManager()

# API Endpoints
@app.post("/api/start-bot/{user_id}", response_model=APIResponse)
async def start_bot(user_id: int, credentials: BotCredentials):
    """Start trading bot for a user with provided API credentials"""
    try:
        # Validate credentials format
        if not credentials.api_key or not credentials.secret_key:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid API credentials"
            )

        # Create and start bot
        bot = await bot_manager.create_bot(
            user_id,
            credentials.api_key,
            credentials.secret_key
        )

        return APIResponse(
            status="success",
            message="Bot started successfully",
            data={"bot_status": "running"}
        )

    except Exception as e:
        logger.error(f"Error starting bot: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.post("/api/stop-bot/{user_id}", response_model=APIResponse)
async def stop_bot(user_id: int):
    """Stop trading bot for a user"""
    try:
        success = await bot_manager.stop_bot(user_id)
        if success:
            return APIResponse(
                status="success",
                message="Bot stopped successfully",
                data={"bot_status": "stopped"}
            )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Bot not found"
        )
    except Exception as e:
        logger.error(f"Error stopping bot: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.get("/api/bot-status/{user_id}", response_model=APIResponse)
async def get_bot_status(user_id: int):
    """Get current status and performance metrics of a user's bot"""
    try:
        status_data = bot_manager.get_bot_status(user_id)
        return APIResponse(
            status="success",
            message="Bot status retrieved successfully",
            data=status_data
        )
    except Exception as e:
        logger.error(f"Error getting bot status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.get("/api/performance/{user_id}", response_model=APIResponse)
async def get_performance_history(user_id: int):
    """Get historical performance data for a user's bot"""
    try:
        bot = bot_manager.get_bot(user_id)
        if not bot:
            return APIResponse(
                status="success",
                message="No active bot found",
                data={"performance": []}
            )

        status_data = bot_manager.get_bot_status(user_id)
        return APIResponse(
            status="success",
            message="Performance data retrieved successfully",
            data={"performance": status_data.get('performance', [])}
        )
    except Exception as e:
        logger.error(f"Error getting performance data: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.get("/api/positions/{user_id}", response_model=APIResponse)
async def get_active_positions(user_id: int):
    """Get current active positions for a user's bot"""
    try:
        status_data = bot_manager.get_bot_status(user_id)
        return APIResponse(
            status="success",
            message="Positions retrieved successfully",
            data={"positions": status_data.get('positions', [])}
        )
    except Exception as e:
        logger.error(f"Error getting positions: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

# WebSocket endpoint for real-time updates
@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: int):
    """WebSocket connection for real-time bot updates"""
    try:
        await manager.connect(user_id, websocket)
        try:
            while True:
                # Keep connection alive and handle incoming messages
                data = await websocket.receive_text()
                try:
                    message = json.loads(data)
                    if message.get("type") == "ping":
                        await websocket.send_json({"type": "pong"})
                except json.JSONDecodeError:
                    pass
        except WebSocketDisconnect:
            manager.disconnect(user_id)
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        manager.disconnect(user_id)

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

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
