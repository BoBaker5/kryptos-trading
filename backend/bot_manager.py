# backend/app/bot_manager.py
import asyncio
import logging
from typing import Dict, Optional
from datetime import datetime, timedelta
from pathlib import Path
import sys
from fastapi import WebSocket

# Add bot directory to Python path
CURRENT_DIR = Path(__file__).resolve().parent
BOT_DIR = CURRENT_DIR / "bot"
sys.path.append(str(BOT_DIR))

try:
    from kraken_crypto_bot_ai import EnhancedKrakenCryptoBot
    BOT_AVAILABLE = True
    print("Successfully imported trading bot")
except ImportError as e:
    print(f"Could not import trading bot: {e}")
    BOT_AVAILABLE = False
    EnhancedKrakenCryptoBot = None

logger = logging.getLogger(__name__)

class BotManager:
    def __init__(self):
        self.active_bots: Dict[int, 'EnhancedKrakenCryptoBot'] = {}
        self.websocket_connections: Dict[int, WebSocket] = {}
        self.performance_data: Dict[int, list] = {}
        self.update_tasks: Dict[int, asyncio.Task] = {}

    async def create_bot(self, user_id: int, api_key: str, secret_key: str):
        """Create and initialize a new trading bot"""
        if not BOT_AVAILABLE:
            raise Exception("Trading bot module not available")
            
        if user_id in self.active_bots:
            raise Exception("Bot already running for this user")
            
        try:
            # Create bot instance
            bot = EnhancedKrakenCryptoBot(api_key, secret_key)
            
            # Initialize performance tracking
            self.performance_data[user_id] = []
            
            # Store bot instance
            self.active_bots[user_id] = bot
            
            # Start update task
            self.update_tasks[user_id] = asyncio.create_task(
                self._periodic_updates(user_id)
            )
            
            return bot
            
        except Exception as e:
            logger.error(f"Error creating bot for user {user_id}: {str(e)}")
            raise

    def get_bot(self, user_id: int) -> Optional[EnhancedKrakenCryptoBot]:
        """Get bot instance for user"""
        return self.active_bots.get(user_id)

    async def stop_bot(self, user_id: int) -> bool:
        """Stop and cleanup bot instance"""
        try:
            if user_id in self.active_bots:
                # Stop bot
                bot = self.active_bots[user_id]
                if hasattr(bot, 'running'):
                    bot.running = False
                
                # Cancel update task
                if user_id in self.update_tasks:
                    self.update_tasks[user_id].cancel()
                    del self.update_tasks[user_id]
                
                # Store final performance data
                await self._update_performance(user_id)
                
                # Cleanup
                del self.active_bots[user_id]
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error stopping bot for user {user_id}: {str(e)}")
            return False

    async def register_websocket(self, user_id: int, websocket: WebSocket):
        """Register websocket connection for real-time updates"""
        self.websocket_connections[user_id] = websocket

    async def remove_websocket(self, user_id: int):
        """Remove websocket connection"""
        self.websocket_connections.pop(user_id, None)

    async def _periodic_updates(self, user_id: int):
        """Periodic updates for bot status and performance"""
        try:
            while True:
                await self._update_performance(user_id)
                await self._send_websocket_update(user_id)
                await asyncio.sleep(5)  # Update every 5 seconds
        except asyncio.CancelledError:
            logger.info(f"Update task cancelled for user {user_id}")
        except Exception as e:
            logger.error(f"Error in periodic updates for user {user_id}: {str(e)}")

    async def _update_performance(self, user_id: int):
        """Update bot performance metrics"""
        try:
            bot = self.active_bots.get(user_id)
            if not bot:
                return

            # Get current performance data
            portfolio_value = bot.portfolio_value if hasattr(bot, 'portfolio_value') else 0
            positions = bot.position_tracker.positions if hasattr(bot, 'position_tracker') else {}
            
            # Calculate daily PnL
            performance_data = self.performance_data.get(user_id, [])
            if performance_data:
                start_value = performance_data[0]['value']
                daily_pnl = ((portfolio_value - start_value) / start_value) * 100 if start_value else 0
            else:
                daily_pnl = 0

            # Add new data point
            new_data = {
                'timestamp': datetime.now().isoformat(),
                'value': portfolio_value,
                'pnl': daily_pnl
            }
            
            performance_data.append(new_data)
            
            # Keep only last 24 hours
            cutoff = datetime.now() - timedelta(hours=24)
            self.performance_data[user_id] = [
                d for d in performance_data 
                if datetime.fromisoformat(d['timestamp']) > cutoff
            ]
            
        except Exception as e:
            logger.error(f"Error updating performance for user {user_id}: {str(e)}")

    async def _send_websocket_update(self, user_id: int):
        """Send update via websocket"""
        try:
            websocket = self.websocket_connections.get(user_id)
            if not websocket:
                return

            bot = self.active_bots.get(user_id)
            if not bot:
                return

            # Prepare update data
            update_data = {
                'status': 'running' if getattr(bot, 'running', False) else 'stopped',
                'portfolio_value': getattr(bot, 'portfolio_value', 0),
                'positions': [
                    {
                        'symbol': symbol,
                        'quantity': pos['quantity'],
                        'entry_price': pos['entry_price'],
                        'current_price': bot.get_latest_price(symbol),
                        'pnl': ((bot.get_latest_price(symbol) - pos['entry_price']) / pos['entry_price']) * 100
                    }
                    for symbol, pos in getattr(bot, 'position_tracker', {}).positions.items()
                ],
                'performance': self.performance_data.get(user_id, [])
            }

            await websocket.send_json(update_data)
            
        except Exception as e:
            logger.error(f"Error sending websocket update for user {user_id}: {str(e)}")
            await self.remove_websocket(user_id)

    def get_bot_status(self, user_id: int) -> dict:
        """Get current bot status and performance"""
        try:
            bot = self.active_bots.get(user_id)
            if not bot:
                return {
                    'status': 'stopped',
                    'portfolio_value': 0,
                    'positions': [],
                    'performance': []
                }

            return {
                'status': 'running' if getattr(bot, 'running', False) else 'stopped',
                'portfolio_value': getattr(bot, 'portfolio_value', 0),
                'positions': [
                    {
                        'symbol': symbol,
                        'quantity': pos['quantity'],
                        'entry_price': pos['entry_price'],
                        'current_price': bot.get_latest_price(symbol),
                        'pnl': ((bot.get_latest_price(symbol) - pos['entry_price']) / pos['entry_price']) * 100
                    }
                    for symbol, pos in getattr(bot, 'position_tracker', {}).positions.items()
                ],
                'performance': self.performance_data.get(user_id, [])
            }
            
        except Exception as e:
            logger.error(f"Error getting bot status for user {user_id}: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }
