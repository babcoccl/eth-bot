#!/usr/bin/env python3
"""
crash_accumulator.py — CrashAccumulator (Patient Capital, CRASH specialist)
====================================================================================
"""

import warnings
import numpy as np
import pandas as pd

from eth_trading.bots.base import BaseBot
from eth_trading.utils.trading_utils import find_support_levels

warnings.filterwarnings("ignore")

PRESETS = {
    "accumulator_v3": {
        "drop_trigger_pct":           0.03,
        "deploy_pct":                 0.035,
        "max_deploy_pct":             0.60,
        "velocity_halt_pct":          0.08,
        "velocity_lookback_bars":     48,
        "velocity_soft_scale":        0.40,
        "support_levels":             3,
        "support_bonus_pct":          0.015,
        "emergency_drawdown_pct":     0.45,
        "emergency_action":           "HOLD",
        "catastrophic_depth_pct":     0.35,
        "max_accum_depth_pct":        0.35,
        "max_hold_days_shallow":      120,
        "max_hold_days_moderate":     180,
        "max_hold_days_major":        365,
        "max_hold_days_catastrophic": 730,
        "profit_target_pct":          0.05,
        "fee_pct":                    0.00065,
    },
}

class CrashAccumulator(BaseBot):
    """
    CRASH regime specialist — patient capital accumulator.
    """

    def __init__(self, symbol: str = "ETH-USD"):
        super().__init__(symbol=symbol)
        self._total_spent         = 0.0
        self._total_deployed_ever = 0.0
        self._last_buy_price      = None
        self._crash_start_price   = None
        self._support_levels      = []
        self._velocity_paused     = False
        self._hold_recalculated   = False
        self._hold_start_bar      = None
        self._state               = "IDLE"

    @property
    def supported_regimes(self) -> list:
        return ["CRASH", "CORRECTION", "RANGE", "RECOVERY", "BULL"]

    def _reset(self, capital: float) -> None:
        super()._reset(capital)
        self._total_spent         = 0.0
        self._total_deployed_ever = 0.0
        self._last_buy_price      = None
        self._crash_start_price   = None
        self._support_levels      = []
        self._velocity_paused     = False
        self._hold_recalculated   = False
        self._hold_start_bar      = None
        self._state               = "IDLE"

    def run_backtest(self, df: pd.DataFrame, preset: dict,
                     capital: float, preset_name: str,
                     df_warm: pd.DataFrame = None,
                     crash_end_ts=None) -> tuple:
        self._reset(capital)
        p = preset

        fee_pct          = p.get("fee_pct", 0.00065)
        drop_trigger     = p.get("drop_trigger_pct", 0.04)
        deploy_pct       = p.get("deploy_pct", 0.04)
        max_deploy       = capital * p.get("max_deploy_pct", 0.60)
        vel_halt         = p.get("velocity_halt_pct", 0.07)
        vel_bars         = int(p.get("velocity_lookback_bars", 48))
        vel_scale        = p.get("velocity_soft_scale", 0.0)
        support_bonus    = p.get("support_bonus_pct", 0.01)
        profit_target    = p.get("profit_target_pct", 0.05)
        emergency_dd     = p.get("emergency_drawdown_pct", 0.45)
        emergency_action = p.get("emergency_action", "SELL")
        catastro_depth   = p.get("catastrophic_depth_pct", 0.35)
        max_accum_depth  = p.get("max_accum_depth_pct", 1.0)
        max_hold_bars    = int(p.get("max_hold_days_moderate", 180) * 288)

        self._support_levels    = find_support_levels(df_warm, p.get("support_levels", 3))
        self._crash_start_price = float(df["close"].iloc[0])
        self._last_buy_price    = self._crash_start_price

        for i in range(len(df)):
            row   = df.iloc[i]
            close = float(row["close"])
            ts    = row["ts"]

            self._equity_curve.append(self._cash + self._position.qty * close)

            if self._position.is_open:
                if self._hold_start_bar is None: self._hold_start_bar = i
                if (crash_end_ts is not None and ts > crash_end_ts and not self._hold_recalculated):
                    self._hold_recalculated = True
                    if self._crash_start_price:
                        _depth = abs((close - self._crash_start_price) / self._crash_start_price)
                        if _depth >= 0.35: max_hold_bars = int(p.get("max_hold_days_catastrophic", 730) * 288)
                        elif _depth >= 0.15: max_hold_bars = int(p.get("max_hold_days_major", 365) * 288)
                        elif _depth >= 0.05: max_hold_bars = int(p.get("max_hold_days_moderate", 180) * 288)
                        else: max_hold_bars = int(p.get("max_hold_days_shallow", 120) * 288)

                if close >= self._position.avg_entry * (1 + profit_target):
                    self._sell(i, df, close, "profit_target", fee_pct)
                    self._state = "DONE"
                    continue
                if (i - self._hold_start_bar) > max_hold_bars:
                    self._sell(i, df, close, "time_stop", fee_pct)
                    self._state = "DONE"
                    continue

                dd = (close - self._position.avg_entry) / self._position.avg_entry
                crash_depth = (abs((close - self._crash_start_price) / self._crash_start_price) if self._crash_start_price else 0.0)
                if dd < -emergency_dd:
                    if emergency_action == "HOLD" and crash_depth >= catastro_depth: self._state = "FROZEN"
                    else:
                        self._sell(i, df, close, "emergency_exit", fee_pct)
                        self._state = "DONE"
                    continue

            if self._state in ("DONE", "FROZEN"): continue
            if self._total_spent >= max_deploy:
                self._state = "CAPPED"
                continue

            if crash_end_ts is not None and ts > crash_end_ts: continue
            if self._crash_start_price:
                depth = abs((close - self._crash_start_price) / self._crash_start_price)
                if depth > max_accum_depth: continue

            velocity_factor = 1.0
            if i >= vel_bars:
                price_vel_ago = float(df["close"].iat[i - vel_bars])
                velocity = (close - price_vel_ago) / price_vel_ago
                if velocity < -vel_halt:
                    self._velocity_paused = True
                    if vel_scale <= 0.0: continue
                    velocity_factor = vel_scale
                else:
                    self._velocity_paused = False
            if self._velocity_paused and vel_scale <= 0.0: continue

            drop = (close - self._last_buy_price) / self._last_buy_price
            if drop > -drop_trigger: continue

            near_support = any(abs(close - lvl) / lvl < 0.012 for lvl in self._support_levels)
            this_pct = (deploy_pct * (1 + support_bonus) if near_support else deploy_pct) * velocity_factor

            spend = min(self._cash * this_pct, max_deploy - self._total_spent, self._cash * 0.95)
            if spend < 1.0: continue

            qty = spend / close
            fee = spend * fee_pct
            self._cash        -= (spend + fee)
            self._total_spent += spend
            self._position.qty       += qty
            self._position.avg_entry  = self._total_spent / self._position.qty
            self._last_buy_price      = close
            self._state               = "ACTIVE"

            self._trades.append({
                "ts": ts, "side": "BUY", "reason": "drop_dca",
                "price": close, "qty": qty, "fee": fee, "spend": spend,
                "near_support": near_support, "velocity_paused": self._velocity_paused,
                "avg_entry": self._position.avg_entry, "total_spent": self._total_spent, "pnl": 0.0,
            })
            self._trade_count += 1

        res_df, res_dict = self._build_result(capital, preset_name)
        if res_dict:
            tdf = pd.DataFrame(self._trades)
            sells = tdf[tdf["side"] == "SELL"]
            res_dict.update({
                "buys": self._trade_count,
                "near_support_buys": len(tdf[(tdf["side"] == "BUY") & (tdf["near_support"])]),
                "vel_throttled_buys": len(tdf[(tdf["side"] == "BUY") & (tdf["velocity_paused"])]),
                "total_qty": self._position.qty,
                "avg_entry": self._position.avg_entry,
                "crash_start_price": self._crash_start_price,
                "total_deployed": self._total_spent,
                "deploy_pct": (self._total_spent + self._total_deployed_ever) / capital * 100,
                "profit_exits": len(sells[sells["reason"] == "profit_target"]),
                "emergency_exits": len(sells[sells["reason"] == "emergency_exit"]),
                "time_stops": len(sells[sells["reason"] == "time_stop"]),
                "position_open": self._position.qty > 0,
                "state": self._state,
                "frozen": self._state == "FROZEN",
                "fees_paid": float(tdf["fee"].sum()),
            })
        return res_df, res_dict

    def _sell(self, i, df, close, reason, fee_pct):
        sell_val = self._position.qty * close
        fee      = sell_val * fee_pct
        pnl      = sell_val - fee - self._total_spent
        self._cash          += sell_val - fee
        self._realized_pnl  += pnl
        self._cumulative    += pnl
        self._trades.append({
            "ts": df.iloc[i]["ts"], "side": "SELL", "reason": reason,
            "price": close, "qty": self._position.qty, "fee": fee,
            "avg_entry": self._position.avg_entry, "total_spent": self._total_spent,
            "pnl": pnl, "pnl_after_fees": pnl, "bars_held": i - self._hold_start_bar if self._hold_start_bar else 0,
        })
        self._position.reset()
        self._total_deployed_ever += self._total_spent
        self._total_spent = 0.0
        self._hold_start_bar = None
        self._trade_count += 1
