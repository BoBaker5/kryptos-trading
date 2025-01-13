import asyncio
import sys
import os
from datetime import datetime
import logging
from demo.demo_bot import DemoKrakenBot
from live.live_bot import EnhancedKrakenCryptoBot

async def run_bots():
    # Set up logging
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure logging for each bot
    demo_logger = logging.getLogger('demo_bot')
    live_logger = logging.getLogger('live_bot')
    
    # Set up file handlers
    demo_handler = logging.FileHandler(f'{log_dir}/demo_bot_{datetime.now().strftime("%Y%m%d")}.log')
    live_handler = logging.FileHandler(f'{log_dir}/live_bot_{datetime.now().strftime("%Y%m%d")}.log')
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    demo_handler.setFormatter(formatter)
    live_handler.setFormatter(formatter)
    
    demo_logger.addHandler(demo_handler)
    live_logger.addHandler(live_handler)
    
    # Initialize bots
    demo_bot = DemoKrakenBot()
    live_bot = EnhancedKrakenCryptoBot(
        api_key="your_api_key",
        secret_key="your_secret_key"
    )
    
    # Run bots concurrently
    await asyncio.gather(
        demo_bot.run(),
        live_bot.run()
    )

if __name__ == "__main__":
    try:
        asyncio.run(run_bots())
    except KeyboardInterrupt:
        print("Shutting down bots...")
        sys.exit(0)
