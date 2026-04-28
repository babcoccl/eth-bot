#!/usr/bin/env python3
"""
HedgeBot v1: CRASH Specialist
=============================
Protects the portfolio by opening leveraged short positions during the CRASH regime.

Strategy:
- Trigger: Entered when Regime == CRASH.
- Sizing: Uses a hedge_ratio and leverage to determine the short position size.
- Exit: Liquidated when Regime shifts out of CRASH.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from eth_bot_interface import BotInterface, BotStatus, Position, Lot

PRESETS = {
    "hedge_v1": {
        "hedge_ratio": 0.5,    # 50% of capital committed to margin
        "leverage":    2.0,    # 2x leverage on the committed capital (1x total portfolio delta)
        "fee_pct":     0.0006, 
        "exit_dwell_bars": 0,
    },
    "hedge_v2_optimized": {
        "hedge_ratio": 0.5,
        "leverage":    2.0,
        "fee_pct":     0.0006,
        "exit_dwell_bars": 24, # 2-hour hysteresis on 5m bars
    }
}

class HedgeBot(BotInterface):
    def __init__(self, symbol: str = "ETH-USD"):
        self._symbol   = symbol
        self._active   = True
        self._position = Position(symbol=symbol, side="SHORT")
        self._cash     = 0.0
        self._capital  = 0.0
        self._realized_pnl = 0.0
        self._trades       = []
        self._equity_curve = []
        
    @property
    def bot_id(self) -> str:
        return "hedgebot_v1"

    @property
    def supported_regimes(self) -> list:
        return ["CRASH"]

    def get_status(self) -> BotStatus:
        return BotStatus(
            bot_id            = "hedgebot_v1",
            symbol            = self._symbol,
            active            = self._active,
            supported_regimes = ["CRASH"],
            capital_allocated  = self._capital,
            capital_deployed   = self._position.cost_basis,
            capital_available  = self._cash,
            open_side          = self._position.side,
            open_qty           = self._position.qty,
            open_avg_entry     = self._position.avg_entry,
            unrealized_pnl     = 0.0, # Computed during run_backtest
            realized_pnl       = self._realized_pnl,
            trade_count        = len(self._trades)
        )

    def _reset(self, capital: float) -> None:
        self._cash         = float(capital)
        self._capital      = float(capital)
        self._position     = Position(symbol=self._symbol, side="SHORT")
        self._trades       = []
        self._equity_curve = []
        
    def run_backtest(self, df: pd.DataFrame, params: Dict[str, Any], 
                     capital: float, preset_name: str = "custom") -> tuple:
        self._reset(capital)
        
        hedge_ratio = params.get("hedge_ratio", 0.5)
        leverage    = params.get("leverage", 1.0)
        fee_pct     = params.get("fee_pct", 0.0006)
        exit_dwell_bars = params.get("exit_dwell_bars", 0)
        not_crash_count = 0
        
        results = []
        for idx_int, (idx, row) in enumerate(df.iterrows()):
            regime = row["regime5"]
            close  = float(row["close"])
            ts     = row["ts"]
            
            # 1. Update Equity Curve
            unreal = self._position.unrealized_pnl(close, fee_pct)
            self._equity_curve.append(self._cash + unreal)
            
            # 2. Logic: Enter SHORT if CRASH and not open
            if regime == "CRASH":
                not_crash_count = 0 # Reset dwell counter when back in CRASH
                if not self._position.is_open:
                    # Use hedge_ratio * capital to determine size, then apply leverage
                    target_value = self._capital * hedge_ratio * leverage
                    qty = target_value / close
                    fee = target_value * fee_pct
                    
                    # In futures, we don't "spend" the cash, we just lock up margin.
                    # For simplicity in this spot-oriented harness, we track cost_basis.
                    self._position.add_lot(Lot(qty=qty, price=close, fee=fee, ts=ts, row_idx=idx_int))
                
            # 3. Logic: Exit SHORT if NOT CRASH and open
            elif self._position.is_open:
                not_crash_count += 1
                if not_crash_count >= exit_dwell_bars:
                    self._close_position(close, fee_pct, ts, f"REGIME_SHIFT_DWELL_{exit_dwell_bars}")
                    not_crash_count = 0
                
            # Logging for trace
            results.append({
                "ts": ts,
                "close": close,
                "regime": regime,
                "pnl": self._cash + self._position.unrealized_pnl(close, fee_pct) - self._capital,
                "pos_qty": self._position.qty if self._position.is_open else 0
            })
            
        stats = {
            "trades":       len(self._trades),
            "realized_pnl": self._realized_pnl,
            "final_equity": self._equity_curve[-1] if self._equity_curve else self._capital,
            "win_rate":     sum(1 for t in self._trades if t['pnl'] > 0) / len(self._trades) if self._trades else 0
        }
        
        return pd.DataFrame(results), stats

    def _close_position(self, price: float, fee_pct: float, ts: Any, reason: str):
        if not self._position.is_open:
            return
            
        exit_val = self._position.qty * price
        exit_fee = exit_val * fee_pct
        
        # Short PnL = (EntryVal - ExitVal) - EntryFee - ExitFee
        entry_val = sum(l.price * l.qty for l in self._position.lots)
        entry_fee = sum(l.fee for l in self._position.lots)
        
        net_pnl = (entry_val - exit_val) - entry_fee - exit_fee
        self._realized_pnl += net_pnl
        self._cash += net_pnl # Add the gain/loss to our cash pool
        
        self._trades.append({
            "entry_ts":    self._position.lots[0].ts,
            "exit_ts":     ts,
            "entry_price": self._position.avg_entry,
            "exit_price":  price,
            "pnl":         net_pnl,
            "reason":      reason
        })
        self._position.reset()
        self._position.side = "SHORT" # Maintain side for next entry

if __name__ == "__main__":
    print("HedgeBot v1 loaded.")
