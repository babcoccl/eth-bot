#!/usr/bin/env python3
"""
eth_bot_interface.py  —  Abstract bot contract + shared data types
==================================================================
FROZEN: Never version this file. If the interface must change,
        create BotInterfaceV2 and migrate bots incrementally.

This module defines the full contract between:
  - Tactical bots  (implement BotInterface)
  - MacroSupervisor (enables/disables bots, reads BotStatus)
  - LLM Orchestrator (reads BotStatus, writes capital allocation commands)

Intended bot implementations:
  DCAScalpBot  (v29 — current)  — uptrend_pb + range_dip + lv_scalp
  TrendBot     (v30 planned)    — uptrend_pb only, BULL/UPTREND gate
  RangeBot     (v31 planned)    — range_dip + lv_scalp, RANGE/RECOVERY gate
  VolBot       (future)         — ATR breakout, high-volatility gate
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# DATA TYPES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Lot:
    """Single buy fill. One or more lots make up an open Position."""
    qty:       float
    price:     float
    fee:       float
    ts:        Any
    row_idx:   int   # index into the trades list for SELL back-fill


@dataclass
class Position:
    """
    Full state of one open position owned by a tactical bot.
    Read by MacroSupervisor and LLM Orchestrator via BotStatus.
    """
    symbol:        str   = "ETH-USD"
    side:          str   = "LONG"    # LONG | SHORT
    qty:           float = 0.0
    avg_entry:     float = 0.0
    peak_price:    float = 0.0
    dca_count:     int   = 0
    entry_bar:     int   = 0
    entry_regime:  str   = "RANGE"
    bull_class:    str   = ""    # BULL depth class at entry: DEEP | SHALLOW_RECOV_LIGHT | SHALLOW_RECOV_DEEP | SHALLOW_CONT
    lots:          list  = field(default_factory=list)

    @property
    def is_open(self) -> bool:
        return self.qty > 1e-9

    @property
    def cost_basis(self) -> float:
        """Total capital at risk including all fill fees."""
        return sum(l.price * l.qty + l.fee for l in self.lots)

    def unrealized_pnl(self, current_price: float, fee_pct: float) -> float:
        """
        Mark-to-market PnL net of projected exit fee.

        Parameters
        ----------
        current_price : float
            The current market price of the asset.
        fee_pct : float
            The estimated exit fee percentage.

        Returns
        -------
        float
            The unrealized PnL.
        """
        if not self.is_open:
            return 0.0
        trade_value = self.qty * current_price
        exit_fee = trade_value * fee_pct
        if self.side == "LONG":
            return trade_value - exit_fee - self.cost_basis
        else:
            entry_value = sum(l.price * l.qty for l in self.lots)
            entry_fee = sum(l.fee for l in self.lots)
            return (entry_value - trade_value) - entry_fee - exit_fee

    def reset(self) -> None:
        """Clear position state after a full exit."""
        self.side         = "LONG"
        self.qty          = 0.0
        self.avg_entry    = 0.0
        self.peak_price   = 0.0
        self.dca_count    = 0
        self.entry_bar    = 0
        self.entry_regime = "RANGE"
        self.bull_class   = ""
        self.lots.clear()


@dataclass
class BotStatus:
    """
    Snapshot of bot state published to SQLite each h1 bar.
    Consumed by MacroSupervisor and LLM Orchestrator.

    LLM Orchestrator uses this to decide:
      - Whether to allocate more capital (DCA at network level)
      - Whether to rotate capital to another symbol
      - Whether to disable the bot due to regime mismatch
    """
    bot_id:             str
    symbol:             str
    capital_allocated:  float   # total capital authorised by supervisor
    capital_deployed:   float   # currently in open position (cost basis)
    capital_available:  float   # cash available for new entries
    open_side:          str     # LONG | SHORT
    open_qty:           float   # coin units held
    open_avg_entry:     float
    unrealized_pnl:     float
    realized_pnl:       float
    trade_count:        int     # completed round-trips
    active:             bool    # enabled by supervisor
    supported_regimes:  list    # regime5 values this bot trades in


# ─────────────────────────────────────────────────────────────────────────────
# ABSTRACT BASE CLASS
# ─────────────────────────────────────────────────────────────────────────────

class BotInterface(ABC):
    """
    Abstract base class defining the contract for all tactical bots.

    Tactical bots implement specific trading strategies and are managed by the
    MacroSupervisor. They must provide status reports and be capable of
    running backtests.

    Lifecycle
    ---------
    1.  `supervisor.enable(bot, capital)`: Bot starts accepting entries.
    2.  `supervisor.disable(bot)`: Bot stops new entries, holds open positions.
    3.  `bot.get_status()`: Called each h1 bar, results are persisted.

    Backtest Contract
    -----------------
    `bot.run_backtest(df, preset, capital, name) -> (trades_df, summary_dict)`
    The summary dictionary keys should align with project standards.
    """

    @property
    @abstractmethod
    def bot_id(self) -> str:
        """
        Unique bot identifier used as DB key and orchestrator reference.
        Convention: "<strategy>_<symbol>"  e.g. "dca_scalp_eth_usd"
        """
        ...

    @property
    @abstractmethod
    def supported_regimes(self) -> list:
        """
        regime5 values this bot is designed to trade in.
        MacroSupervisor disables the bot when current regime is not in this list.

        Examples:
          TrendBot  → ["BULL", "RECOVERY"]
          RangeBot  → ["RANGE", "RECOVERY"]
          VolBot    → ["BULL", "RANGE"]  (high ATR in any non-crash state)
        """
        ...

    @abstractmethod
    def get_status(self) -> BotStatus:
        """
        Return current bot state for orchestrator consumption.
        Called by MacroSupervisor on every h1 bar.
        """
        ...

    @abstractmethod
    def run_backtest(
        self,
        df: pd.DataFrame,
        preset: dict,
        capital: float,
        preset_name: str,
    ) -> tuple:
        """
        Simulate strategy over a prepared indicator DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            The input data containing price and indicators.
            Required columns: `macro_pause` (bool), `regime5` (str).
        preset : dict
            Parameter dictionary from the bot's PRESETS.
        capital : float
            Starting USDC capital for the backtest.
        preset_name : str
            Label for the preset, used in results.

        Returns
        -------
        trades_df : pd.DataFrame
            DataFrame where each row is a BUY or SELL event.
        summary : dict
            Dictionary containing backtest performance metrics.
        """
        ...
