import asyncio
import pandas as pd
import numpy as np
import krakenex
from pykrakenapi import KrakenAPI
import time
import logging
from datetime import datetime

class DemoTradingBot:
    def __init__(self):
        # Initialize Kraken API without keys for public data only
        self.kraken = krakenex.API()
        self.k = KrakenAPI(self.kraken)
        self.logger = logging.getLogger("DemoBot")
        
        # Portfolio settings
        self.initial_balance = 1000000  # $1M
        self.portfolio_value = self.initial_balance
        self.running = False
        self.position_tracker = {"positions": {}}
        self.daily_pnl = 0
        self.performance_metrics = []
        
        # Trading pairs with same allocations as real bot
        self.symbols = {
            "SOLUSD": 0.20,
            "AVAXUSD": 0.20,
            "XRPUSD": 0.20,
            "XDGUSD": 0.15,
            "SHIBUSD": 0.10,
            "PEPEUSD": 0.15
        }
        
        # Cache for prices
        self.price_cache = {}
        self.price_cache_time = {}
        self.cache_duration = 30  # 30 seconds

    def get_latest_price(self, symbol: str) -> float:
        """Get price with caching to avoid rate limits"""
        current_time = time.time()
        
        # Return cached price if fresh
        if symbol in self.price_cache:
            cache_age = current_time - self.price_cache_time.get(symbol, 0)
            if cache_age < self.cache_duration:
                return self.price_cache[symbol]
        
        try:
            ticker = self.k.get_ticker_information(symbol)
            if isinstance(ticker, pd.DataFrame):
                price = float(ticker['c'][0][0])
                self.price_cache[symbol] = price
                self.price_cache_time[symbol] = current_time
                return price
        except Exception as e:
            self.logger.error(f"Error getting price for {symbol}: {e}")
            return self.price_cache.get(symbol)  # Return last cached price

        return None

    def update_portfolio_value(self):
        """Update demo portfolio value"""
        old_value = self.portfolio_value
        total_value = self.initial_balance
        
        for symbol, pos in self.position_tracker["positions"].items():
            current_price = self.get_latest_price(symbol)
            if current_price:
                pos['current_price'] = current_price
                pnl = (current_price - pos['entry_price']) * pos['quantity']
                total_value += pnl
        
        self.portfolio_value = total_value
        self.daily_pnl = ((total_value - old_value) / old_value) * 100
        
        # Store performance
        self.performance_metrics.append({
            'timestamp': datetime.now().isoformat(),
            'value': self.portfolio_value,
            'daily_pnl': self.daily_pnl
        })
        
        # Keep last 24 hours
        if len(self.performance_metrics) > 17280:  # 24h * 60m * 12 (5-second intervals)
            self.performance_metrics = self.performance_metrics[-17280:]

    def simulate_trade(self, symbol: str, action: str, price: float, confidence: float = 0.5):
        """Simulate a trade with the demo portfolio"""
        try:
            if action == 'buy' and symbol not in self.position_tracker["positions"]:
                allocation = self.portfolio_value * self.symbols[symbol] * confidence
                quantity = allocation / price
                
                self.position_tracker["positions"][symbol] = {
                    'quantity': quantity,
                    'entry_price': price,
                    'current_price': price
                }
                
                self.logger.info(f"DEMO BOT: Opened {symbol} position - {quantity:.4f} @ ${price:.4f}")
                
            elif action == 'sell' and symbol in self.position_tracker["positions"]:
                pos = self.position_tracker["positions"][symbol]
                pnl = (price - pos['entry_price']) * pos['quantity']
                self.portfolio_value += pnl
                
                self.logger.info(f"DEMO BOT: Closed {symbol} position - PNL: ${pnl:.2f}")
                del self.position_tracker["positions"][symbol]
                
        except Exception as e:
            self.logger.error(f"Error simulating trade: {e}")

    async def run(self):
        """Run demo bot"""
        self.logger.info(f"Starting Demo Bot with ${self.initial_balance:,.2f}")
        self.running = True
        
        while self.running:
            try:
                self.update_portfolio_value()
                
                # Log portfolio status periodically
                if len(self.performance_metrics) % 12 == 0:  # Every minute
                    self.logger.info(f"\nDEMO BOT STATUS:")
                    self.logger.info(f"Portfolio Value: ${self.portfolio_value:,.2f}")
                    self.logger.info(f"24h P/L: {self.daily_pnl:.2f}%")
                    self.logger.info(f"Active Positions: {len(self.position_tracker['positions'])}")
                
                await asyncio.sleep(5)
                
            except Exception as e:
                self.logger.error(f"Error in demo bot loop: {e}")
                await asyncio.sleep(5)

    def get_account_balance(self):
        """Get demo account balance"""
        return {"ZUSD": self.portfolio_value}