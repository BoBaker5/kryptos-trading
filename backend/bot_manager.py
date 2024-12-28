import os
import sys
from pathlib import Path

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

class BotManager:
    def __init__(self):
        self.active_bots = {}
    
    def create_bot(self, user_id: int, api_key: str, secret_key: str):
        if not BOT_AVAILABLE:
            raise Exception("Trading bot module not available")
            
        if user_id in self.active_bots:
            raise Exception("Bot already running for this user")
            
        bot = EnhancedKrakenCryptoBot(api_key, secret_key)
        self.active_bots[user_id] = bot
        return bot
    
    def get_bot(self, user_id: int):
        return self.active_bots.get(user_id)
    
    def stop_bot(self, user_id: int):
        if user_id in self.active_bots:
            bot = self.active_bots[user_id]
            if hasattr(bot, 'running'):
                bot.running = False
            del self.active_bots[user_id]
            return True
        return False