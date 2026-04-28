#!/usr/bin/env python3
"""
recovery_bot.py — RecoveryBot (Dead Cat Bounce short strategy)
==================================================================
"""

import warnings
import numpy as np
import pandas as pd

from eth_trading.bots.base import BaseBot
from eth_trading.core.bot_interface import BotStatus, Position, Lot

warnings.filterwarnings("ignore")

PRESETS = {
    "dcb_v2_optimized": {
        "base_qty":           0.05,
        "lookback_bars":      12 * 24 * 2,
        "min_drop_pct":       0.03,
        "vol_ratio_max":      0.85,
        "fib_entry_low":      0.382,
        "fib_entry_high":     0.500,
        "fib_stop":           0.786,
        "max_hold_bars":      12 * 24 * 2,
        "fee_pct":                0.00065,
    }
}

class RecoveryBot(BaseBot):
    """
    RECOVERY/DOWNTREND regime specialist — Dead Cat Bounce short strategy.
    """
    def __init__(self, symbol: str = "ETH-USD"):
        super().__init__(symbol=symbol)
        self._position = Position(symbol=symbol, side="SHORT")
        self._last_trade_ts = None

    @property
    def supported_regimes(self) -> list:
        return ["RECOVERY", "DOWNTREND"]

    def _reset(self, capital: float) -> None:
        super()._reset(capital)
        self._position = Position(symbol=self._symbol, side="SHORT")
        self._last_trade_ts = None

    def _sell_short(self, i, df, fill_price, reason, base_qty, fee_pct):
        row = df.iloc[i]
        sv = base_qty * fill_price
        if sv > self._cash: return False
        fee = sv * fee_pct
        self._cash -= fee
        self._position.qty = base_qty
        self._position.avg_entry = fill_price
        self._position.entry_bar = i
        self._position.lots = [Lot(qty=base_qty, price=fill_price, fee=fee, ts=row["ts"], row_idx=len(self._trades))]
        self._last_trade_ts = row["ts"]
        self._trades.append({
            "ts": row["ts"], "side": "SELL_SHORT", "reason": reason,
            "regime_h1": str(row.get("regime_h1", "")),
            "price": fill_price, "qty": base_qty, "fee": fee,
            "pnl": 0.0, "pnl_after_fees": 0.0, "win": float("nan"), "bars_held": float("nan"),
        })
        return True

    def _buy_to_cover(self, i, df, fill_price, reason, fee_pct):
        p = self._position
        if not p.is_open: return False
        row = df.iloc[i]
        lot = p.lots[0]
        gross_pnl = (lot.price - fill_price) * p.qty
        exit_fee = (p.qty * fill_price) * fee_pct
        net_pnl = gross_pnl - lot.fee - exit_fee
        self._cash += net_pnl
        self._cumulative += net_pnl
        if reason != "end_of_period":
            self._realized_pnl += net_pnl
            self._trade_count += 1
        bh = i - p.entry_bar
        entry_row = self._trades[lot.row_idx]
        entry_row.update({"pnl": net_pnl, "pnl_after_fees": net_pnl, "win": 1.0 if net_pnl > 0 else 0.0, "bars_held": bh})
        self._trades.append({
            "ts": row["ts"], "side": "BUY_COVER", "reason": reason,
            "regime_h1": str(row.get("regime_h1", "")),
            "price": fill_price, "qty": p.qty, "fee": exit_fee,
            "pnl": net_pnl, "pnl_after_fees": net_pnl, "win": 1.0 if net_pnl > 0 else 0.0, "bars_held": bh,
        })
        p.reset()
        return True

    def run_backtest(self, df: pd.DataFrame, preset: dict, capital: float, preset_name: str) -> tuple:
        self._reset(capital)
        p = preset
        fee_pct      = p["fee_pct"]
        base_qty     = p["base_qty"]
        lookback     = p["lookback_bars"]
        min_drop     = p["min_drop_pct"]
        vol_max      = p["vol_ratio_max"]
        f_low        = p["fib_entry_low"]
        f_high       = p["fib_entry_high"]
        f_stop       = p["fib_stop"]
        max_hold     = p["max_hold_bars"]

        df["rolling_high"] = df["high"].rolling(window=lookback, min_periods=1).max()

        for i in range(len(df)):
            row    = df.iloc[i]
            ts     = row["ts"]
            regime = str(row.get("regime_h1", "RANGE"))
            low    = float(row["low"])
            high   = float(row["high"])
            close  = float(row["close"])
            open_p = float(row["open"])

            unreal = self._position.unrealized_pnl(close, fee_pct)
            self._equity_curve.append(self._cash + unreal)

            if self._position.is_open:
                bh = i - self._position.entry_bar
                if high >= getattr(self, "_active_fib_stop", 999999):
                    self._buy_to_cover(i, df, getattr(self, "_active_fib_stop", high), "stop_loss_fib", fee_pct)
                    continue
                if low <= getattr(self, "_active_macro_low", 0):
                    self._buy_to_cover(i, df, getattr(self, "_active_macro_low", low), "target_macro_low", fee_pct)
                    continue
                if bh >= max_hold:
                    self._buy_to_cover(i, df, close, "time_stop", fee_pct)
                    continue
                continue

            if regime not in self.supported_regimes: continue
            if i < lookback: continue

            macro_peak = float(df["rolling_high"].iat[i])
            window_high = df["high"].iloc[i-lookback:i+1]
            peak_idx = window_high.values.argmax() + (i - lookback)
            if peak_idx >= i: continue

            macro_low = float(df["low"].iloc[peak_idx:i+1].min())
            drop_pct = (macro_peak - macro_low) / macro_peak
            if drop_pct < min_drop: continue

            drop_val = macro_peak - macro_low
            fib_382 = macro_low + drop_val * f_low
            fib_500 = macro_low + drop_val * f_high
            fib_786 = macro_low + drop_val * f_stop

            if high >= fib_382 and close <= fib_500:
                if close < open_p:
                    vol_r = float(row.get("vol_ratio", 1.0))
                    if vol_r <= vol_max:
                        if self._sell_short(i, df, close, "dcb_short", base_qty, fee_pct):
                            self._active_macro_low = macro_low
                            self._active_fib_stop  = fib_786

        if self._position.is_open:
            self._buy_to_cover(len(df)-1, df, float(df.iloc[-1]["close"]), "end_of_period", fee_pct)

        res_df, res_dict = self._build_result(capital, preset_name)
        if res_dict:
            tdf = pd.DataFrame(self._trades)
            real = tdf[(tdf["side"] == "BUY_COVER") & (tdf["reason"] != "end_of_period")]
            res_dict.update({
                "target_fires": len(real[real["reason"] == "target_macro_low"]),
                "target_pnl": float(real[real["reason"] == "target_macro_low"]["pnl_after_fees"].sum()),
                "psl_fires": len(real[real["reason"] == "stop_loss_fib"]),
                "psl_pnl": float(real[real["reason"] == "stop_loss_fib"]["pnl_after_fees"].sum()),
                "time_stop_fires": len(real[real["reason"] == "time_stop"]),
                "time_stop_pnl": float(real[real["reason"] == "time_stop"]["pnl_after_fees"].sum()),
                "time_stop_pct": len(real[real["reason"] == "time_stop"]) / len(real) * 100 if len(real) > 0 else 0.0,
            })
        return res_df, res_dict
