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
    from bot.demo_bot import DemoTradingBot
    BOT_AVAILABLE = True
    print("Successfully imported trading bots")
except ImportError as e:
    print(f"Could not import trading bots: {e}")
    BOT_AVAILABLE = False
    EnhancedKrakenCryptoBot = None
    DemoTradingBot = None

logger = logging.getLogger(__name__)

class BotManager:
    def __init__(self):
        # Real trading bots
        self.active_bots: Dict[int, 'EnhancedKrakenCryptoBot'] = {}
        self.websocket_connections: Dict[int, WebSocket] = {}
        self.performance_data: Dict[int, list] = {}
        self.update_tasks: Dict[int, asyncio.Task] = {}
        
        # Demo bot
        self.demo_bot = None
        self.demo_performance_data = []
        self.demo_websocket_connections: Dict[int, WebSocket] = {}
        
        # Start demo bot
        asyncio.create_task(self.initialize_demo_bot())

    async def initialize_demo_bot(self):
        """Initialize and start demo bot"""
        try:
            self.demo_bot = DemoTradingBot(initial_balance=10000.0)
            asyncio.create_task(self.demo_bot.run())
            asyncio.create_task(self._demo_periodic_updates())
            logger.info("Demo bot initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing demo bot: {str(e)}")

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

    async def register_websocket(self, user_id: int, websocket: WebSocket, is_demo: bool = False):
        """Register websocket connection for real-time updates"""
        if is_demo:
            self.demo_websocket_connections[user_id] = websocket
        else:
            self.websocket_connections[user_id] = websocket

    async def remove_websocket(self, user_id: int, is_demo: bool = False):
        """Remove websocket connection"""
        if is_demo:
            self.demo_websocket_connections.pop(user_id, None)
        else:
            self.websocket_connections.pop(user_id, None)

    async def _periodic_updates(self, user_id: int):
        """Periodic updates for real bot status and performance"""
        try:
            while True:
                await self._update_performance(user_id)
                await self._send_websocket_update(user_id)
                await asyncio.sleep(5)  # Update every 5 seconds
        except asyncio.CancelledError:
            logger.info(f"Update task cancelled for user {user_id}")
        except Exception as e:
            logger.error(f"Error in periodic updates for user {user_id}: {str(e)}")

    async def _demo_periodic_updates(self):
        """Periodic updates for demo bot"""
        try:
            while True:
                if self.demo_bot:
                    # Update demo performance data
                    status = self.demo_bot.get_status()
                    self.demo_performance_data.append({
                        'timestamp': datetime.now().isoformat(),
                        'value': status['portfolio_value'],
                        'pnl': status['returns']
                    })
                    
                    # Keep last 24 hours
                    cutoff = datetime.now() - timedelta(hours=24)
                    self.demo_performance_data = [
                        d for d in self.demo_performance_data 
                        if datetime.fromisoformat(d['timestamp']) > cutoff
                    ]
                    
                    # Send updates to all demo websocket connections
                    await self._send_demo_websocket_updates()
                
                await asyncio.sleep(5)
                
        except Exception as e:
            logger.error(f"Error in demo periodic updates: {str(e)}")

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
        """Send update via websocket for real bot"""
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

    async def _send_demo_websocket_updates(self):
        """Send updates to all demo websocket connections"""
        if not self.demo_bot:
            return
            
        status = self.demo_bot.get_status()
        update_data = {
            'status': status['status'],
            'portfolio_value': status['portfolio_value'],
            'positions': status['positions'],
            'performance': self.demo_performance_data,
            'trades': status['trades'],
            'signals': status.get('signals', {})
        }
        
        for user_id, websocket in list(self.demo_websocket_connections.items()):
            try:
                await websocket.send_json(update_data)
            except Exception as e:
                logger.error(f"Error sending demo update to user {user_id}: {str(e)}")
                await self.remove_websocket(user_id, is_demo=True)

    def get_bot_status(self, user_id: int, is_demo: bool = False) -> dict:
        """Get current bot status and performance"""
        try:
            if is_demo:
                if self.demo_bot:
                    status = self.demo_bot.get_status()
                    status['performance'] = self.demo_performance_data
                    return status
                return {
                    'status': 'stopped',
                    'portfolio_value': 0,
                    'positions': [],
                    'performance': []
                }
                
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
            logger.error(f"Error getting {'demo' if is_demo else ''} bot status: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }

    async def stop_all(self):
        """Stop all bots including demo"""
        # Stop all real bots
        for user_id in list(self.active_bots.keys()):
            await self.stop_bot(user_id)
            
        # Stop demo bot
        if self.demo_bot:
            self.demo_bot.stop()
            self.demo_bot = None
