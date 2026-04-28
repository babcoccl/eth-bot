#!/usr/bin/env python3
"""
src/eth_trading/bots/base.py  —  Base bot class with shared logic
==============================================================
"""

import numpy as np
import pandas as pd
from abc import abstractmethod
from typing import Dict, Tuple, List, Optional

from eth_trading.core.bot_interface import BotInterface, BotStatus, Position

class BaseBot(BotInterface):
    """
    Abstract base class for all tactical bots, providing shared state and boilerplate.

    Attributes
    ----------
    _symbol : str
        Trading symbol, e.g., "ETH-USD".
    _active : bool
        Whether the bot is currently enabled.
    _position : Position
        Current open position state.
    _cash : float
        Current available cash.
    _capital : float
        Initial capital allocated to the bot.
    _realized_pnl : float
        Total realized profit and loss.
    _trade_count : int
        Number of completed trades.
    _trades : List[Dict]
        History of all BUY and SELL events.
    _equity_curve : List[float]
        Time-series of total account equity.
    _cumulative : float
        Cumulative PnL.
    """

    def __init__(self, symbol: str = "ETH-USD"):
        self._symbol = symbol
        self._active = True
        self._position = Position(symbol=symbol)
        self._cash = 0.0
        self._capital = 0.0
        self._realized_pnl = 0.0
        self._trade_count = 0
        self._trades = []
        self._equity_curve = []
        self._cumulative = 0.0

    @property
    def bot_id(self) -> str:
        """Unique bot identifier."""
        return f"{self.__class__.__name__.lower()}_{self._symbol.lower().replace('-', '_')}"

    @property
    @abstractmethod
    def supported_regimes(self) -> List[str]:
        """Regime5 values this bot trades in."""
        ...

    def get_status(self) -> BotStatus:
        """
        Return current bot state for orchestrator consumption.

        Returns
        -------
        BotStatus
            Snapshot of the current bot status.
        """
        return BotStatus(
            bot_id=self.bot_id,
            symbol=self._symbol,
            capital_allocated=self._capital,
            capital_deployed=self._position.cost_basis,
            capital_available=self._cash,
            open_side=self._position.side,
            open_qty=self._position.qty,
            open_avg_entry=self._position.avg_entry,
            unrealized_pnl=0.0,  # Should be updated by subclass if needed
            realized_pnl=self._realized_pnl,
            trade_count=self._trade_count,
            active=self._active,
            supported_regimes=self.supported_regimes,
        )

    def _reset(self, capital: float) -> None:
        """
        Reset bot state for a new backtest run.

        Parameters
        ----------
        capital : float
            Starting capital.
        """
        self._cash = float(capital)
        self._capital = float(capital)
        self._position = Position(symbol=self._symbol)
        self._trades = []
        self._equity_curve = []
        self._cumulative = 0.0
        self._realized_pnl = 0.0
        self._trade_count = 0

    def _build_result(self, capital: float, preset_name: str) -> Tuple[pd.DataFrame, Dict]:
        """
        Compile backtest results into a DataFrame and summary dictionary.

        Parameters
        ----------
        capital : float
            Initial capital.
        preset_name : str
            Name of the parameter preset used.

        Returns
        -------
        Tuple[pd.DataFrame, Dict]
            A tuple containing the trades DataFrame and a summary dictionary.
        """
        if not self._trades:
            return pd.DataFrame(), {}

        tdf = pd.DataFrame(self._trades)
        eq = np.array(self._equity_curve)

        peak_eq = np.maximum.accumulate(eq)
        max_dd = float(((eq - peak_eq) / peak_eq).min()) * 100 if len(eq) > 0 else 0.0
        ret_s = np.diff(eq) / eq[:-1] if len(eq) > 1 else np.array([])
        sharpe = (ret_s.mean() / ret_s.std() * np.sqrt(105_120)
                   if len(ret_s) > 0 and ret_s.std() > 0 else 0.0)

        sells = tdf[tdf["side"].str.contains("SELL|COVER")]
        real = sells[~sells["reason"].isin(["end_of_period", "regime_exit"])]
        wins = real[real["pnl_after_fees"] > 0]

        summary = {
            "preset": preset_name,
            "trades": len(real),
            "win_rate": len(wins) / len(real) * 100 if len(real) > 0 else 0,
            "realized_pnl": float(tdf["pnl_after_fees"].sum()),
            "total_return": self._cumulative / capital * 100 if capital > 0 else 0.0,
            "final_equity": capital + self._cumulative,
            "max_drawdown": max_dd,
            "sharpe": sharpe,
            "fees": float(tdf["fee"].sum()),
            "avg_bars_held": float(real["bars_held"].mean()) if len(real) > 0 else 0,
        }

        return tdf, summary
