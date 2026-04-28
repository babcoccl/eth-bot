#!/usr/bin/env python3
"""
eth_live_engine_v1.py — Event-Driven Live Trading Orchestrator
==============================================================
The "v33" production brain that coordinates real-time WebSocket events,
tactical bot evaluation, and order execution.

Core Responsibilities:
  1. WebSocket Data Flow: Consumes TickEvents and FillEvents.
  2. Tick Processing: Dispatches every price update to active bots for immediate reaction.
  3. Bar Aggregation: Consolidates ticks into 1m/1h bars for regime classification.
  4. Regime Governance: Every 1h bar, updates MacroSupervisor and manages bot lifecycle.
  5. Execution: Dispatches bot signals to the OrderExecutor (Live or Paper).
  6. Persistence: Saves bot state to disk on every fill for crash resilience.

Usage:
  python eth_live_engine_v1.py --paper --capital 10000
"""

import asyncio
import logging
import os
import signal
import sys
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional

import pandas as pd

# Project Imports
from eth_bot_interface import BotInterface, BotStatus
from eth_coinbase_ws import CoinbaseWSManager, TickEvent, FillEvent, OrderUpdateEvent
from eth_coinbase_executor import OrderExecutor, OrderResult
from eth_macrosupervisor_v31 import MacroSupervisor
from eth_trendbot_v1 import TrendBot
from eth_rangebot_v4 import RangeBot
from eth_correction_bot_v1 import CorrectionBot
from eth_recoverybot_v1 import RecoveryBot
from eth_hedgebot_v1 import HedgeBot

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("live_engine.log")
    ]
)
logger = logging.getLogger("live_engine")

# ─────────────────────────────────────────────────────────────────────────────
# TICK AGGREGATOR
# ─────────────────────────────────────────────────────────────────────────────

class TickAggregator:
    """Aggregates real-time TickEvents into OHLCV bars."""
    def __init__(self, timeframe_mins: int = 60):
        self.interval = timedelta(minutes=timeframe_mins)
        self.current_bar = None
        self.history = []

    def add_tick(self, tick: TickEvent) -> Optional[dict]:
        """Add a tick. Returns a completed bar (dict) if the interval has rolled."""
        ts = tick.ts.replace(second=0, microsecond=0)
        if timeframe_mins := self.interval.total_seconds() // 60:
             ts = ts - timedelta(minutes=ts.minute % timeframe_mins)

        if self.current_bar is None:
            self.current_bar = {
                "ts": ts, "open": tick.price, "high": tick.price,
                "low": tick.price, "close": tick.price, "volume": 0.0
            }
            return None

        if ts > self.current_bar["ts"]:
            # Interval rolled — return the old bar
            finished_bar = self.current_bar.copy()
            self.current_bar = {
                "ts": ts, "open": tick.price, "high": tick.price,
                "low": tick.price, "close": tick.price, "volume": 0.0
            }
            return finished_bar
        
        # Update current bar
        self.current_bar["high"]  = max(self.current_bar["high"], tick.price)
        self.current_bar["low"]   = min(self.current_bar["low"], tick.price)
        self.current_bar["close"] = tick.price
        # Note: accurate volume requires ticker.volume_24h diffing or trade messages
        return None

# ─────────────────────────────────────────────────────────────────────────────
# LIVE ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class LiveEngine:
    def __init__(self, args):
        self.paper_mode = args.paper
        self.capital    = args.capital
        self.symbol     = "ETH-USD"
        
        # 1. Initialize Components
        self.queue      = asyncio.Queue()
        self.supervisor = MacroSupervisor(total_capital=self.capital)
        self.executor   = OrderExecutor(
            api_key    = os.getenv("COINBASE_API_KEY", ""),
            api_secret = os.getenv("COINBASE_API_SECRET", ""),
            paper_mode = self.paper_mode
        )
        self.ws_manager = CoinbaseWSManager(
            api_key    = os.getenv("COINBASE_API_KEY", ""),
            api_secret = os.getenv("COINBASE_API_SECRET", ""),
            event_queue = self.queue,
            paper_mode  = self.paper_mode
        )
        
        # 2. Initialize Bot Fleet
        self.bots: Dict[str, BotInterface] = {
            "trendbot":      TrendBot(symbol=self.symbol),
            "rangebot":      RangeBot(symbol=self.symbol),
            "correctionbot": CorrectionBot(symbol=self.symbol),
            "recoverybot":   RecoveryBot(symbol=self.symbol),
            "hedgebot":      HedgeBot(symbol=self.symbol),
        }
        
        # 3. State Tracking
        self.agg_h1 = TickAggregator(60)
        self.agg_m5 = TickAggregator(5)
        self.h1_history = [] # To store enough bars for EMA/RSI
        self.m5_history = []
        
        self.running = False

    async def run(self):
        self.running = True
        logger.info(f"V33 ENGINE STARTING | MODE: {'PAPER' if self.paper_mode else 'LIVE'}")
        
        # Setup Signal Handlers
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda: asyncio.create_task(self.shutdown()))

        # 1. Start WebSockets
        await self.ws_manager.start([self.symbol])
        
        # 2. Main Event Loop
        try:
            while self.running:
                event = await self.queue.get()
                
                if isinstance(event, TickEvent):
                    await self._handle_tick(event)
                elif isinstance(event, FillEvent):
                    await self._handle_fill(event)
                elif isinstance(event, OrderUpdateEvent):
                    logger.debug(f"Order Update: {event.order_id} -> {event.status}")
                
                self.queue.task_done()
        except asyncio.CancelledError:
            pass
        finally:
            await self.shutdown()

    async def _handle_tick(self, tick: TickEvent):
        """Process real-time price updates."""
        # 1. Update Aggregators
        bar_h1 = self.agg_h1.add_tick(tick)
        if bar_h1:
            await self._on_h1_close(bar_h1)
            
        bar_m5 = self.agg_m5.add_tick(tick)
        if bar_m5:
            self.m5_history.append(bar_m5)
            if len(self.m5_history) > 100: self.m5_history.pop(0)

        # 2. Dispatch to Bots for Real-Time Reaction (Evaluate Tick)
        # Only dispatch to bots that support the current regime
        current_regime = self.supervisor.current_regime
        for bot_name, bot in self.bots.items():
            if current_regime in bot.supported_regimes:
                signals = bot.evaluate_tick(tick.price, tick.ts, supervisor=self.supervisor)
                if signals:
                    for sig in signals:
                        await self._execute_signal(bot_name, sig, tick.price)

    async def _on_h1_close(self, bar: dict):
        """Perform regime classification and lifecycle management."""
        self.h1_history.append(bar)
        if len(self.h1_history) > 100: self.h1_history.pop(0)
        
        # Build DF for MacroSupervisor
        df_h1 = pd.DataFrame(self.h1_history)
        if len(df_h1) < 50: # Need enough data for EMA_50
            return
            
        # Apply Regime Logic
        self.supervisor._compute_h1_signals(df_h1)
        logger.info(f"Regime Update: {self.supervisor.current_regime} (Conviction: {self.supervisor.conviction_score})")

    async def _execute_signal(self, bot_id: str, signal: dict, current_price: float):
        """Translate bot signals into exchange orders."""
        action = signal.get("action")
        logger.info(f"SIGNAL from {bot_id}: {action} at ${current_price:.2f}")
        
        res = None
        if action == "BUY":
            # Determine spend/qty
            spend = signal.get("spend")
            if not spend: # If bot didn't specify, use a default or signal qty
                qty = signal.get("qty", 0.0)
                spend = qty * current_price
            
            res = self.executor.buy_market(self.symbol, spend, current_price)
            
        elif action == "SELL":
            qty = signal.get("qty")
            res = self.executor.sell_market(self.symbol, qty, current_price)

        if res and not res.success:
            logger.error(f"Execution Failed for {bot_id}: {res.error}")

    async def _handle_fill(self, fill: FillEvent):
        """Reconcile internal state after an exchange fill."""
        # 1. Identify which bot this fill belongs to
        # (For now, we route by client_order_id prefix or just broadcast)
        for bot in self.bots.values():
            bot.process_fill(fill, supervisor=self.supervisor)
            
        # 2. Update Dashboard / Ledger
        logger.info(f"State Synchronized for fill {fill.order_id}")

    async def shutdown(self):
        if not self.running: return
        self.running = False
        logger.info("Engine shutting down...")
        await self.ws_manager.stop()
        # Save all bot states
        for bot in self.bots.values():
            if hasattr(bot, 'save_to_disk'):
                bot.save_to_disk()
        logger.info("Shutdown complete.")

# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper", action="store_true", help="Enable paper trading mode")
    parser.add_argument("--capital", type=float, default=10000.0, help="Initial virtual capital")
    args = parser.parse_args()

    engine = LiveEngine(args)
    try:
        asyncio.run(engine.run())
    except KeyboardInterrupt:
        pass
