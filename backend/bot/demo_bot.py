import asyncio
import pandas as pd
import numpy as np
from datetime import datetime
import krakenex
from pykrakenapi import KrakenAPI
from typing import Optional, Dict
import time
import logging

# Import from your real bot implementation
from .kraken_crypto_bot_ai import (
    AITradingEnhancer,
    MLModelManager,
    TransformerBlock
)

class DemoTradingBot:
    def __init__(self, initial_balance: float = 10000.0):
        # Initialize Kraken API for market data only
        self.kraken = krakenex.API()
        self.k = KrakenAPI(self.kraken)
        
        # Setup logging
        self.logger = logging.getLogger("DemoBot")
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
        # Trading pairs (same as real bot)
        self.symbols = {
            "SOLUSD": 0.20,    # SOL/USD
            "AVAXUSD": 0.20,   # AVAX/USD
            "XRPUSD": 0.20,    # XRP/USD
            "XDGUSD": 0.15,    # DOGE/USD
            "SHIBUSD": 0.10,   # SHIB/USD
            "PEPEUSD": 0.15    # PEPE/USD
        }
        
        # Initialize ML/AI components
        self.ai_enhancer = AITradingEnhancer()
        self.model_manager = MLModelManager()
        self.is_initially_trained = False
        self.ai_trained = False
        
        # Portfolio tracking
        self.initial_balance = initial_balance
        self.portfolio = {
            'cash': initial_balance,
            'positions': {}
        }
        self.portfolio_history = []
        self.trades_history = []
        self.performance_metrics = []
        self.running = False
        
        # Risk management (same as real bot)
        self.max_drawdown = 0.10
        self.trailing_stop_pct = 0.03
        self.max_trades_per_hour = 3
        self.trade_cooldown = 300
        self.last_trade_time = {}
        self.min_position_value = 2.0
        self.max_position_size = 0.3
        self.stop_loss_pct = 0.03
        self.take_profit_pct = 0.05
        
        # Market state tracking
        self.market_state = {symbol: {'trend': None, 'volatility': None} for symbol in self.symbols}
        
        # Cache for market data
        self.price_cache = {}
        self.price_cache_time = {}
        self.cache_duration = 30

    async def get_historical_data(self, symbol: str, lookback_days: int = 7) -> pd.DataFrame:
        """Fetch and preprocess historical data"""
        try:
            since = time.time() - (lookback_days * 24 * 60 * 60)
            ohlc, last = self.k.get_ohlc_data(symbol, interval=5, since=since)
            
            if ohlc is not None and not ohlc.empty:
                numeric_columns = ['open', 'high', 'low', 'close', 'volume']
                for col in numeric_columns:
                    if col in ohlc.columns:
                        ohlc[col] = pd.to_numeric(ohlc[col], errors='coerce')
                
                ohlc = ohlc.dropna(subset=['close', 'volume'])
                ohlc = ohlc[ohlc['close'] > 0]
                ohlc = ohlc[ohlc['volume'] > 0]
                
                return ohlc
                
            return pd.DataFrame()

        except Exception as e:
            self.logger.error(f"Error fetching historical data for {symbol}: {str(e)}")
            return pd.DataFrame()

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        return self.model_manager.prepare_features(df)

    def generate_signals(self, df: pd.DataFrame, symbol: str) -> dict:
        """Generate trading signals using AI/ML models"""
        if not self.is_initially_trained or not self.ai_trained:
            return {'action': 'hold', 'confidence': 0.5}
            
        try:
            # Get ML prediction (35% weight)
            features = self.model_manager.prepare_features(df)
            ml_signal = self.model_manager.predict(features)
            ml_confidence = ml_signal['confidence']

            # Get AI prediction (25% weight)
            ai_signal = self.ai_enhancer.predict_next_movement(df)
            ai_confidence = ai_signal['confidence']

            # Technical analysis (40% weight)
            tech_confidence = 0.5
            market_state = self.analyze_market_state(df)
            
            if market_state:
                if market_state['trend'] > 0:
                    tech_confidence += 0.02
                elif market_state['trend'] < 0:
                    tech_confidence -= 0.02

            # Combine signals
            final_confidence = 0.5 + (
                (ml_confidence - 0.5) * 0.35 +
                (ai_confidence - 0.5) * 0.25 +
                (tech_confidence - 0.5) * 0.40
            )

            final_confidence = max(0.45, min(0.55, final_confidence))
            
            if final_confidence > 0.52:
                action = 'buy'
            elif final_confidence < 0.47:
                action = 'sell'
            else:
                action = 'hold'

            return {'action': action, 'confidence': final_confidence}
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {str(e)}")
            return {'action': 'hold', 'confidence': 0.5}

    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get latest price with caching"""
        current_time = time.time()
        
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
            return self.price_cache.get(symbol)
            
        return None

    def calculate_portfolio_value(self) -> float:
        """Calculate total portfolio value"""
        total_value = self.portfolio['cash']
        
        for symbol, position in self.portfolio['positions'].items():
            current_price = self.get_latest_price(symbol)
            if current_price:
                position_value = position['quantity'] * current_price
                total_value += position_value
        
        return total_value

    def get_account_balance(self):
        """Get demo account balance"""
        return {
            "ZUSD": self.portfolio['cash'],
            "total_value": self.calculate_portfolio_value()
        }

    def analyze_market_state(self, df: pd.DataFrame) -> dict:
        """Analyze market state"""
        try:
            if len(df) < 50:
                return None

            latest = df.iloc[-1]
            trend = 1 if latest['sma_20'] > latest['sma_50'] else -1
            volatility = df['close'].pct_change().rolling(20).std().iloc[-1]
            volume_sma = df['volume'].rolling(window=20).mean().iloc[-1]
            volume_trend = 1 if latest['volume'] > volume_sma else -1

            return {
                'trend': trend,
                'volatility': volatility,
                'volume_trend': volume_trend
            }

        except Exception as e:
            self.logger.error(f"Error in market state analysis: {str(e)}")
            return None

    def simulate_trade(self, symbol: str, signal: dict) -> bool:
        """Execute a simulated trade"""
        try:
            current_price = self.get_latest_price(symbol)
            if not current_price:
                return False
                
            portfolio_value = self.calculate_portfolio_value()
            
            if signal['action'] == 'buy' and symbol not in self.portfolio['positions']:
                # Calculate position size
                max_position = portfolio_value * self.max_position_size
                allocation = portfolio_value * self.symbols[symbol] * signal['confidence']
                position_size = min(max_position, allocation)
                
                if position_size < self.min_position_value:
                    return False
                
                if position_size > self.portfolio['cash']:
                    return False
                
                # Execute buy
                quantity = position_size / current_price
                self.portfolio['cash'] -= position_size
                self.portfolio['positions'][symbol] = {
                    'quantity': quantity,
                    'entry_price': current_price,
                    'high_price': current_price,
                    'entry_time': datetime.now()
                }
                
                self.logger.info(f"DEMO: Bought {quantity:.4f} {symbol} @ ${current_price:.4f}")
                
            elif signal['action'] == 'sell' and symbol in self.portfolio['positions']:
                position = self.portfolio['positions'][symbol]
                position_value = position['quantity'] * current_price
                pnl = position_value - (position['quantity'] * position['entry_price'])
                pnl_pct = (pnl / (position['quantity'] * position['entry_price'])) * 100
                
                self.portfolio['cash'] += position_value
                del self.portfolio['positions'][symbol]
                
                self.logger.info(f"DEMO: Sold {symbol} - PNL: ${pnl:.2f} ({pnl_pct:.2f}%)")
            
            # Record trade
            self.trades_history.append({
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'action': signal['action'],
                'price': current_price,
                'confidence': signal['confidence'],
                'portfolio_value': self.calculate_portfolio_value()
            })
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error executing trade: {e}")
            return False

    def monitor_positions(self):
        """Monitor positions for exits"""
        try:
            for symbol in list(self.portfolio['positions'].keys()):
                current_price = self.get_latest_price(symbol)
                if not current_price:
                    continue
                    
                position = self.portfolio['positions'][symbol]
                entry_price = position['entry_price']
                high_price = position['high_price']
                
                # Update high price
                if current_price > high_price:
                    position['high_price'] = current_price
                
                # Calculate returns
                pnl_pct = (current_price - entry_price) / entry_price * 100
                
                # Check stop loss
                if pnl_pct <= -self.stop_loss_pct * 100:
                    self.logger.info(f"Stop loss triggered for {symbol}")
                    self.simulate_trade(symbol, {'action': 'sell', 'confidence': 1.0})
                    
                # Check take profit and trailing stop
                elif pnl_pct >= self.take_profit_pct * 100:
                    trailing_stop_price = position['high_price'] * (1 - self.trailing_stop_pct)
                    if current_price < trailing_stop_price:
                        self.logger.info(f"Trailing stop triggered for {symbol}")
                        self.simulate_trade(symbol, {'action': 'sell', 'confidence': 1.0})
                        
        except Exception as e:
            self.logger.error(f"Error monitoring positions: {e}")

    def update_performance_metrics(self):
        """Update performance tracking"""
        current_value = self.calculate_portfolio_value()
        returns = ((current_value - self.initial_balance) / self.initial_balance) * 100
        
        self.performance_metrics.append({
            'timestamp': datetime.now().isoformat(),
            'value': current_value,
            'returns': returns,
            'cash': self.portfolio['cash'],
            'positions_count': len(self.portfolio['positions'])
        })
        
        # Keep last 24 hours
        if len(self.performance_metrics) > 17280:
            self.performance_metrics = self.performance_metrics[-17280:]

    def get_status(self) -> dict:
        """Get current bot status"""
        current_value = self.calculate_portfolio_value()
        returns = ((current_value - self.initial_balance) / self.initial_balance) * 100
        
        positions = []
        for symbol, pos in self.portfolio['positions'].items():
            current_price = self.get_latest_price(symbol)
            if current_price:
                pnl = (current_price - pos['entry_price']) / pos['entry_price'] * 100
                positions.append({
                    'symbol': symbol,
                    'quantity': pos['quantity'],
                    'entry_price': pos['entry_price'],
                    'current_price': current_price,
                    'pnl_pct': pnl,
                    'high_price': pos['high_price']
                })
        
        return {
            'status': 'running' if self.running else 'stopped',
            'portfolio_value': current_value,
            'initial_balance': self.initial_balance,
            'cash': self.portfolio['cash'],
            'returns': returns,
            'positions': positions,
            'history': self.performance_metrics[-144:],  # Last 12 hours
            'trades': self.trades_history[-50:],  # Last 50 trades
            'signals': self.current_signals if hasattr(self, 'current_signals') else {}
        }

    async def run(self):
        """Run the demo bot"""
        self.logger.info(f"Starting Demo Bot with ${self.initial_balance:,.2f}")
        self.running = True
        self.current_signals = {}
        
        while self.running:
            try:
                # Monitor existing positions
                self.monitor_positions()
                
                # Process trading opportunities
                for symbol in self.symbols:
                    try:
                        # Get market data
                        df = await self.get_historical_data(symbol)
                        if df.empty:
                            continue
                            
                        # Calculate indicators and generate signals
                        df = self.calculate_indicators(df)
                        signal = self.generate_signals(df, symbol)
                        
                        # Store current signal
                        self.current_signals[symbol] = signal
                        
                        self.logger.info(f"\n--- Processing {symbol} ---")
                        self.logger.info(f"Signal: {signal['action'].upper()} (confidence: {signal['confidence']:.3f})")
                        
                        # Execute trade if conditions met
                        if signal['action'] != 'hold':
                            # Check trade cooldown
                            if symbol in self.last_trade_time:
                                time_since_last = time.time() - self.last_trade_time[symbol]
                                if time_since_last < self.trade_cooldown:
                                    continue
                            
                            trade_executed = self.simulate_trade(symbol, signal)
                            if trade_executed:
                                self.last_trade_time[symbol] = time.time()
                                
                    except Exception as e:
                        self.logger.error(f"Error processing {symbol}: {e}")
                        continue
                
                # Update performance tracking
                self.update_performance_metrics()
                
                # Log status periodically
                if len(self.performance_metrics) % 120 == 0:  # Every 10 minutes
                    status = self.get_status()
                    self.logger.info("\nDEMO BOT STATUS UPDATE:")
                    self.logger.info(f"Portfolio Value: ${status['portfolio_value']:,.2f}")
                    self.logger.info(f"Returns: {status['returns']:.2f}%")
                    self.logger.info(f"Active Positions: {len(status['positions'])}")
                    self.logger.info(f"Cash: ${status['cash']:,.2f}")
                
                await asyncio.sleep(5)
                
            except Exception as e:
                self.logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(5)
    
    def stop(self):
        """Stop the demo bot"""
        self.running = False
        self.logger.info("Demo bot stopped")
