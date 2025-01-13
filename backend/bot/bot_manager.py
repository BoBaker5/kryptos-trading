# backend/bot_manager.py
import asyncio
import logging
from typing import Dict, Optional
from datetime import datetime, timedelta
from pathlib import Path
import sys
from fastapi import WebSocket

from .demo_bot import DemoKrakenBot  # Note the '.' for relative import
from .kraken_crypto_bot_ai import EnhancedKrakenCryptoBott

logger = logging.getLogger(__name__)

class BotManager:
    def __init__(self):
        self.active_bots: Dict[int, 'EnhancedKrakenCryptoBot'] = {}
        self.websocket_connections: Dict[int, WebSocket] = {}
        self.performance_data: Dict[int, list] = {}
        self.update_tasks: Dict[int, asyncio.Task] = {}
        
        # Start both real and demo bots immediately
        self.demo_bot = DemoTradingBot(initial_balance=1000000.0)
        self.real_bot = EnhancedKrakenCryptoBot()  # Add your API keys here if needed
        
        # Initialize performance tracking
        self.demo_performance_data = []
        
        # Start both bots
        asyncio.create_task(self.initialize_bots())
        
    async def initialize_bots(self):
        """Start both real and demo bots"""
        try:
            # Start demo bot
            asyncio.create_task(self.demo_bot.run())
            logger.info("Demo bot started successfully")
            
            # Start real bot
            asyncio.create_task(self.real_bot.run())
            logger.info("Real bot started successfully")
            
            # Start update tasks
            asyncio.create_task(self._demo_periodic_updates())
            
        except Exception as e:
            logger.error(f"Error initializing bots: {str(e)}")

    def get_bot_status(self, user_id: int, is_demo: bool = False) -> dict:
        """Get status of either demo or real bot"""
        try:
            if is_demo:
                bot = self.demo_bot
            else:
                bot = self.real_bot
                
            if not bot:
                return {
                    "status": "stopped",
                    "portfolio_value": 0,
                    "positions": [],
                    "performance_data": [],
                    "daily_pnl": 0,
                    "signals": []
                }
            
            return {
                "status": "running",
                "portfolio_value": bot.portfolio_value if hasattr(bot, 'portfolio_value') else 0,
                "positions": bot.position_tracker.positions if hasattr(bot, 'position_tracker') else [],
                "performance_data": self.demo_performance_data if is_demo else self.performance_data.get(user_id, []),
                "daily_pnl": bot.daily_pnl if hasattr(bot, 'daily_pnl') else 0,
                "signals": bot.current_signals if hasattr(bot, 'current_signals') else []
            }
            
        except Exception as e:
            logger.error(f"Error getting bot status: {str(e)}")
            return {
                "status": "error",
                "portfolio_value": 0,
                "positions": [],
                "performance_data": [],
                "daily_pnl": 0,
                "signals": []
            }

    async def _demo_periodic_updates(self):
        """Update demo bot performance metrics"""
        try:
            while True:
                if self.demo_bot:
                    status = self.demo_bot.get_status()
                    self.demo_performance_data.append({
                        'timestamp': datetime.now().isoformat(),
                        'value': status.get('portfolio_value', 10000.0),
                        'pnl': status.get('daily_pnl', 0)
                    })
                    
                    # Keep last 24 hours
                    cutoff = datetime.now() - timedelta(hours=24)
                    self.demo_performance_data = [
                        d for d in self.demo_performance_data 
                        if datetime.fromisoformat(d['timestamp']) > cutoff
                    ]
                
                await asyncio.sleep(5)
                
        except Exception as e:
            logger.error(f"Error in demo periodic updates: {str(e)}")

    async def register_websocket(self, user_id: int, websocket: WebSocket):
        """Register websocket connection"""
        self.websocket_connections[user_id] = websocket

    async def remove_websocket(self, user_id: int):
        """Remove websocket connection"""
        self.websocket_connections.pop(user_id, None)
