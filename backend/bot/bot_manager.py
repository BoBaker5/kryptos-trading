import os
import logging
import asyncio
from typing import Dict, Optional, List, Any
import traceback
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv
from .demo_bot import DemoKrakenBot
from .kraken_crypto_bot import EnhancedKrakenCryptoBot

# Load environment variables
load_dotenv()

class BotManager:
    def __init__(self):
        # Set up logging
        self.logger = self._setup_logging()
        
        # Load API credentials
        self.api_key = os.getenv('KRAKEN_API_KEY')
        self.api_secret = os.getenv('KRAKEN_SECRET_KEY')
        
        if not self.api_key or not self.api_secret:
            self.logger.warning("API credentials not found in environment variables")
        
        # Initialize bot instances
        self.demo_bot: Optional[DemoKrakenBot] = None
        self.live_bot: Optional[EnhancedKrakenCryptoBot] = None
        self.running = False
        
        try:
            # Initialize demo bot with configuration
            self.demo_bot = DemoKrakenBot(initial_balance=1000000.0)
            self.logger.info("Demo bot initialized successfully")
            
            # Initialize live bot with API keys if available
            if self.api_key and self.api_secret:
                self.live_bot = EnhancedKrakenCryptoBot(
                    api_key=self.api_key,
                    secret_key=self.api_secret
                )
                self.logger.info("Live bot initialized successfully")
            else:
                self.logger.warning("Live bot not initialized - missing API credentials")
            
        except Exception as e:
            self.logger.error(f"Error initializing bots: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise

    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration"""
        logger = logging.getLogger("bot.bot_manager")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    async def start_bots(self):
        """Start both demo and live bots"""
        self.running = True
        try:
            # Start both bots concurrently using asyncio.gather
            await asyncio.gather(
                self._run_demo_bot() if self.demo_bot else asyncio.sleep(0),
                self._run_live_bot() if self.live_bot else asyncio.sleep(0)
            )
        except Exception as e:
            self.logger.error(f"Error starting bots: {str(e)}")
            self.logger.error(traceback.format_exc())
            self.running = False
            raise

    async def _run_demo_bot(self):
        """Run demo bot with error handling"""
        try:
            if self.demo_bot:
                await self.demo_bot.run()
        except Exception as e:
            self.logger.error(f"Demo bot error: {str(e)}")
            self.logger.error(traceback.format_exc())

    async def _run_live_bot(self):
        """Run live bot with error handling"""
        try:
            if self.live_bot:
                await self.live_bot.run()
        except Exception as e:
            self.logger.error(f"Live bot error: {str(e)}")
            self.logger.error(traceback.format_exc())

    async def stop_bots(self):
        """Stop both demo and live bots"""
        self.running = False
        try:
            if self.demo_bot:
                self.demo_bot.running = False
            if self.live_bot:
                self.live_bot.running = False
            self.logger.info("Bots stopped successfully")
        except Exception as e:
            self.logger.error(f"Error stopping bots: {str(e)}")
            self.logger.error(traceback.format_exc())

    def get_demo_bot_status(self) -> Dict:
        """Get current status of demo bot"""
        if not self.demo_bot:
            return {"status": "Not initialized"}
            
        try:
            return {
                "status": "Running" if self.running else "Stopped",
                "time": datetime.now().isoformat(),
                "balance": self.demo_bot.get_demo_balance(),
                "positions": self.demo_bot.get_demo_positions(),
                "metrics": self.demo_bot.get_portfolio_metrics()
            }
        except Exception as e:
            self.logger.error(f"Error getting demo bot status: {str(e)}")
            self.logger.error(traceback.format_exc())
            return {"status": "Error", "error": str(e)}

    def get_live_bot_status(self) -> Dict:
        """Get current status of live bot"""
        if not self.live_bot:
            return {"status": "Not initialized"}
            
        try:
            return {
                "status": "Running" if self.running else "Stopped",
                "time": datetime.now().isoformat(),
                "balance": self.live_bot.get_account_balance(),
                "positions": self.live_bot.get_positions(),
                "metrics": self.live_bot.get_portfolio_metrics()
            }
        except Exception as e:
            self.logger.error(f"Error getting live bot status: {str(e)}")
            self.logger.error(traceback.format_exc())
            return {"status": "Error", "error": str(e)}

    def get_combined_metrics(self) -> Dict:
        """Get combined metrics from both bots"""
        try:
            metrics = {
                "time": datetime.now().isoformat(),
                "demo_bot": self.get_demo_bot_status() if self.demo_bot else None,
                "live_bot": self.get_live_bot_status() if self.live_bot else None,
                "system_status": {
                    "running": self.running,
                    "last_update": datetime.now().isoformat()
                }
            }
            return metrics
        except Exception as e:
            self.logger.error(f"Error getting combined metrics: {str(e)}")
            self.logger.error(traceback.format_exc())
            return {
                "status": "Error",
                "error": str(e),
                "time": datetime.now().isoformat()
            }

    def get_trade_history(self, bot_type: str = "demo") -> List[Dict[str, Any]]:
        """Get trade history for specified bot"""
        try:
            if bot_type.lower() == "demo" and self.demo_bot:
                return self.demo_bot.trade_history
            elif bot_type.lower() == "live" and self.live_bot:
                return self.live_bot.trade_history
            return []
        except Exception as e:
            self.logger.error(f"Error getting trade history: {str(e)}")
            self.logger.error(traceback.format_exc())
            return []

    def get_portfolio_history(self, bot_type: str = "demo") -> List[Dict[str, Any]]:
        """Get portfolio history for specified bot"""
        try:
            if bot_type.lower() == "demo" and self.demo_bot:
                return self.demo_bot.portfolio_history
            elif bot_type.lower() == "live" and self.live_bot:
                return self.live_bot.portfolio_history
            return []
        except Exception as e:
            self.logger.error(f"Error getting portfolio history: {str(e)}")
            self.logger.error(traceback.format_exc())
            return []

    async def update_bot_settings(self, bot_type: str, settings: Dict) -> bool:
        """Update settings for specified bot"""
        try:
            if bot_type.lower() == "demo" and self.demo_bot:
                # Update demo bot settings
                for key, value in settings.items():
                    if hasattr(self.demo_bot, key):
                        setattr(self.demo_bot, key, value)
                return True
            elif bot_type.lower() == "live" and self.live_bot:
                # Update live bot settings
                for key, value in settings.items():
                    if hasattr(self.live_bot, key):
                        setattr(self.live_bot, key, value)
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error updating bot settings: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False

    async def restart_bot(self, bot_type: str) -> bool:
        """Restart specified bot"""
        try:
            if bot_type.lower() == "demo" and self.demo_bot:
                self.demo_bot.running = False
                await asyncio.sleep(1)  # Wait for current cycle to complete
                self.demo_bot.running = True
                asyncio.create_task(self._run_demo_bot())
                return True
            elif bot_type.lower() == "live" and self.live_bot:
                self.live_bot.running = False
                await asyncio.sleep(1)  # Wait for current cycle to complete
                self.live_bot.running = True
                asyncio.create_task(self._run_live_bot())
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error restarting bot: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False

    def get_bot_settings(self, bot_type: str) -> Dict:
        """Get current settings for specified bot"""
        try:
            if bot_type.lower() == "demo" and self.demo_bot:
                return {
                    "max_position_size": self.demo_bot.max_position_size,
                    "stop_loss_pct": self.demo_bot.stop_loss_pct,
                    "take_profit_pct": self.demo_bot.take_profit_pct,
                    "max_trades_per_hour": self.demo_bot.max_trades_per_hour,
                    "symbols": self.demo_bot.symbols
                }
            elif bot_type.lower() == "live" and self.live_bot:
                return {
                    "max_position_size": self.live_bot.max_position_size,
                    "stop_loss_pct": self.live_bot.stop_loss_pct,
                    "take_profit_pct": self.live_bot.take_profit_pct,
                    "max_trades_per_hour": self.live_bot.max_trades_per_hour,
                    "symbols": self.live_bot.symbols
                }
            return {}
        except Exception as e:
            self.logger.error(f"Error getting bot settings: {str(e)}")
            self.logger.error(traceback.format_exc())
            return {}
