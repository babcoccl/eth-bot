#!/usr/bin/env python3
"""
range_bot.py — RangeBot (Grid Trading, RANGE/RECOVERY specialist)
==================================================================
"""

import warnings
import numpy as np
import pandas as pd

from eth_trading.bots.base import BaseBot
from eth_trading.core.bot_interface import BotStatus, Position, Lot

warnings.filterwarnings("ignore")

PRESETS = {
    "grid_v1": {
        "base_qty":           0.05,
        "grid_levels":        10,
        "grid_step_bps":      40,
        "pos_stop_loss_pct":  0.06,
        "fee_pct":            0.00065,
    },
}

class RangeBot(BaseBot):
    """
    RANGE/RECOVERY regime specialist — Grid Trading architecture.
    """
    def __init__(self, symbol: str = "ETH-USD"):
        super().__init__(symbol=symbol)
        self._grid_active = False
        self._base_price  = 0.0
        self._grid_step   = 0.0
        self._buy_levels  = []
        self._sell_levels = []

    @property
    def supported_regimes(self) -> list:
        return ["RANGE", "RECOVERY"]

    def _reset(self, capital: float) -> None:
        super()._reset(capital)
        self._grid_active  = False
        self._buy_levels   = []
        self._sell_levels  = []

    def run_backtest(self, df: pd.DataFrame, preset: dict, capital: float, preset_name: str) -> tuple:
        self._reset(capital)
        p = preset
        fee_pct      = p["fee_pct"]
        base_qty     = p["base_qty"]
        grid_levels  = p["grid_levels"]
        grid_bps     = p["grid_step_bps"]
        psl_pct      = p.get("pos_stop_loss_pct", 0.05)

        for i in range(len(df)):
            row    = df.iloc[i]
            ts     = row["ts"]
            regime = str(row.get("regime_h1", "RANGE"))
            low    = float(row["low"])
            high   = float(row["high"])
            close  = float(row["close"])

            self._equity_curve.append(self._cash + self._position.qty * close)

            if self._position.is_open:
                unreal_p = (close - self._position.avg_entry) / self._position.avg_entry if self._position.avg_entry > 0 else 0
                if unreal_p < -psl_pct:
                    self._sell_all(i, df, close, "pos_stop_loss", fee_pct)
                    self._grid_active = False
                    self._buy_levels = []
                    self._sell_levels = []
                    continue

            if regime not in self.supported_regimes:
                if self._grid_active:
                    if self._position.is_open: self._sell_all(i, df, close, "regime_exit", fee_pct)
                    self._grid_active = False
                    self._buy_levels = []
                    self._sell_levels = []
                continue

            if not self._grid_active:
                self._grid_active = True
                self._base_price  = close
                self._grid_step   = close * (grid_bps / 10000.0)
                self._buy_levels  = [self._base_price - self._grid_step * j for j in range(1, grid_levels + 1)]
                self._sell_levels = []
            elif not self._sell_levels and self._buy_levels:
                if close > self._buy_levels[0] + self._grid_step * 2:
                    self._base_price  = close
                    self._buy_levels  = [self._base_price - self._grid_step * j for j in range(1, grid_levels + 1)]

            for bl in sorted(self._buy_levels, reverse=True):
                if low <= bl:
                    if self._buy_lot(i, df, bl, "grid_buy", base_qty, fee_pct):
                        self._buy_levels.remove(bl)
                        self._sell_levels.append(bl + self._grid_step)

            for sl in sorted(self._sell_levels):
                if high >= sl:
                    if self._sell_lot(i, df, sl, "grid_sell", fee_pct):
                        self._sell_levels.remove(sl)
                        self._buy_levels.append(sl - self._grid_step)

        if self._position.is_open:
            self._sell_all(len(df)-1, df, float(df.iloc[-1]["close"]), "end_of_period", fee_pct)

        res_df, res_dict = self._build_result(capital, preset_name)
        if res_dict:
            tdf = pd.DataFrame(self._trades)
            sells = tdf[tdf["side"] == "SELL"]
            res_dict.update({
                "target_fires": len(sells[sells["reason"] == "grid_sell"]),
                "target_pnl": float(sells[sells["reason"] == "grid_sell"]["pnl_after_fees"].sum()),
                "psl_fires": len(sells[sells["reason"] == "pos_stop_loss"]),
                "psl_pnl": float(sells[sells["reason"] == "pos_stop_loss"]["pnl_after_fees"].sum()),
                "range_dip_fires": len(tdf[tdf["reason"] == "grid_buy"]),
                "eop_pnl": float(sells[sells["reason"].isin(["end_of_period", "regime_exit"])]["pnl_after_fees"].sum()),
            })
        return res_df, res_dict

    def _buy_lot(self, i, df, fill_price, reason, base_qty, fee_pct) -> bool:
        row = df.iloc[i]
        bv  = base_qty * fill_price
        if bv > self._cash: return False
        fee = bv * fee_pct
        self._cash -= (bv + fee)
        old_cost = self._position.qty * self._position.avg_entry
        self._position.qty += base_qty
        self._position.avg_entry = (old_cost + bv) / self._position.qty
        lot = Lot(qty=base_qty, price=fill_price, fee=fee, ts=row["ts"], row_idx=len(self._trades))
        self._position.lots.append(lot)
        self._trades.append({
            "ts": row["ts"], "side": "BUY", "reason": reason,
            "regime_h1": str(row.get("regime_h1", "")),
            "price": fill_price, "qty": base_qty, "fee": fee,
            "pnl": 0.0, "pnl_after_fees": 0.0, "win": float("nan"), "bars_held": float("nan"),
        })
        return True

    def _sell_lot(self, i, df, fill_price, reason, fee_pct) -> bool:
        if not self._position.lots: return False
        row = df.iloc[i]
        lot = self._position.lots.pop(0)
        sell_val = lot.qty * fill_price
        sell_fee = sell_val * fee_pct
        pnl = sell_val - sell_fee - (lot.qty * lot.price + lot.fee)
        self._cash += (sell_val - sell_fee)
        self._position.qty -= lot.qty
        if self._position.qty < 1e-9:
            self._position.avg_entry = 0.0
            self._position.qty = 0.0
        self._cumulative += pnl
        self._realized_pnl += pnl
        self._trade_count += 1
        bh = i - lot.row_idx
        buy_row = self._trades[lot.row_idx]
        buy_row.update({"pnl": pnl, "pnl_after_fees": pnl, "win": 1.0 if pnl > 0 else 0.0, "bars_held": bh})
        self._trades.append({
            "ts": row["ts"], "side": "SELL", "reason": reason,
            "regime_h1": str(row.get("regime_h1", "")),
            "price": fill_price, "qty": lot.qty, "fee": sell_fee,
            "pnl": pnl, "pnl_after_fees": pnl, "win": 1.0 if pnl > 0 else 0.0, "bars_held": bh,
        })
        return True

    def _sell_all(self, i, df, fill_price, reason, fee_pct):
        while self._position.lots: self._sell_lot(i, df, fill_price, reason, fee_pct)
