#!/usr/bin/env python3
"""
trend_bot.py — TrendBot (BULL / RECOVERY specialist)
==============================================================
Role : Trades uptrend pullbacks during BULL and RECOVERY regimes only.
"""

import warnings
import numpy as np
import pandas as pd

from eth_trading.bots.base import BaseBot
from eth_trading.core.bot_interface import BotStatus, Position, Lot
from eth_trading.core.bull_classifier import STOP_LOSS_BY_CLASS

warnings.filterwarnings("ignore")

PRESETS = {
    "trendbot_v1": {
        "base_qty":                   0.05,
        "pos_stop_loss_pct":          0.025,
        "uptrend_rsi_max":            38,
        "vol_mult_min":               1.30,
        "cooldown_secs":              14400,
        "psl_cooldown_secs":          28800,
        "min_profit_bps":             100,
        "zscore_max":                -1.5,
        "regime_stable_bars":         72,
        "macro_context_bars":         1440,
        "macro_bearish_max":          0.65,
        "time_stop_bars":             24,
        "time_stop_min_pct":          0.003,
        "rsi_lookback_bars":          24,
        "rsi_lookback_bars_recovery": 48,
        "qty_scale": {
            "STRONG":    1.0,
            "PARABOLIC": 1.0,
            "MODERATE":  0.5,
        },
        "trend_strength_allowed": {"STRONG", "PARABOLIC", "MODERATE"},
        "buy_fee_pct":        0.00065,
        "sell_fee_pct":       0.00025,
        "target_bps":        None,
        "target_atr_mult":   1.5,
        "target_bps_min":    120,
        "target_bps_max":    350,
        "psl_atr_max":       0.07,
        "manage_psl_mult":   3.0,
        "psl_atr_mult":      1.5,
        "macro_dd_skip":     -0.20,
        "entry_rsi_min":     30,
    },
}

_TREND_REGIMES = frozenset({"BULL", "RECOVERY"})

class TrendBot(BaseBot):
    """
    BULL/RECOVERY regime specialist — uptrend_pb signal only.
    """

    def __init__(self, symbol: str = "ETH-USD"):
        super().__init__(symbol=symbol)
        self._last_buy_ts  = None
        self._last_psl_ts  = None

    @property
    def supported_regimes(self) -> list:
        return ["BULL", "RECOVERY"]

    def _reset(self, capital: float) -> None:
        super()._reset(capital)
        self._last_buy_ts  = None
        self._last_psl_ts  = None

    def run_backtest(self, df: pd.DataFrame, preset: dict,
                    capital: float, preset_name: str) -> tuple:
        self._reset(capital)
        p = preset

        buy_fee_pct      = p.get("buy_fee_pct", 0.00065)
        sell_fee_pct     = p.get("sell_fee_pct", 0.00025)
        base_qty         = p["base_qty"]
        target_bps       = p["target_bps"]
        cooldown         = p["cooldown_secs"]
        psl_cooldown     = p.get("psl_cooldown_secs", cooldown)
        min_profit       = p.get("min_profit_bps", 0)
        zscore_max       = p.get("zscore_max", -0.3)
        qty_scale_map    = p.get("qty_scale", {})
        strength_allowed = set(p.get("trend_strength_allowed", {"STRONG", "PARABOLIC", "MODERATE"}))

        rsi_lb_bull     = p.get("rsi_lookback_bars", 24)
        rsi_lb_recovery = p.get("rsi_lookback_bars_recovery", 48)

        trend_streak = 0
        _g = {
            "open_position": 0, "regime": 0, "stable_bars": 0, "macro_context": 0,
            "macro_dd": 0, "strength": 0, "rsi_nan": 0, "cooldown": 0,
            "psl_cooldown": 0, "entry_rsi_min": 0, "rsi_lookback": 0,
            "sig_rsi": 0, "sig_zscore": 0, "sig_vol": 0, "entered": 0,
        }

        for i in range(len(df)):
            row    = df.iloc[i]
            close  = float(row["close"])
            ts     = row["ts"]
            regime5 = str(row.get("regime5", "RANGE"))

            if regime5 in _TREND_REGIMES:
                trend_streak += 1
            else:
                trend_streak = 0

            self._equity_curve.append(self._cash + self._position.qty * close)

            if self._position.is_open:
                _g["open_position"] += 1
                unreal = (close - self._position.avg_entry) / self._position.avg_entry
                bull_cls = self._position.bull_class
                manage_psl_mult = p.get("manage_psl_mult", 3.0)
                max_psl_pct     = p.get("psl_atr_max", 0.07)

                atr_pct_now   = (self._position.entry_atr_pct
                                if hasattr(self._position, "entry_atr_pct")
                                else float(df["atr_pct"].iat[i]))
                if atr_pct_now > 0:
                    atr_psl = atr_pct_now * manage_psl_mult
                    effective_psl = min(atr_psl, max_psl_pct)
                else:
                    effective_psl = p.get("pos_stop_loss_pct", 0.025)

                time_stop_bars = p.get("time_stop_bars", 0)
                if time_stop_bars > 0:
                    bars_in_trade = i - self._position.entry_bar
                    if bars_in_trade >= time_stop_bars:
                        min_progress = p.get("time_stop_min_pct", 0.003)
                        progress = (close - self._position.avg_entry) / self._position.avg_entry
                        if progress < min_progress:
                            self._sell(i, df, close, "time_stop", sell_fee_pct)
                            continue

                if bull_cls and bull_cls in STOP_LOSS_BY_CLASS:
                    effective_psl = min(effective_psl, STOP_LOSS_BY_CLASS[bull_cls])

                if unreal < -effective_psl:
                    self._sell(i, df, close, "pos_stop_loss", sell_fee_pct)
                    continue

                pos_target = self._position.target_bps if self._position.target_bps is not None else 180
                if close >= self._position.avg_entry * (1 + pos_target / 10_000):
                    self._sell(i, df, close, "target", sell_fee_pct)
                    continue
                continue

            if regime5 not in _TREND_REGIMES:
                _g["regime"] += 1
                continue

            regime_stable_bars = p.get("regime_stable_bars", 0)
            if regime_stable_bars > 0 and trend_streak < regime_stable_bars:
                _g["stable_bars"] += 1
                continue

            macro_lookback = p.get("macro_context_bars", 0)
            if macro_lookback > 0:
                recent = df["regime5"].iloc[max(0, i - macro_lookback):i]
                bearish_frac = recent.isin(["CRASH", "CORRECTION"]).sum() / len(recent)
                macro_bearish_max = p.get("macro_bearish_max", 0.60)
                if bearish_frac > macro_bearish_max:
                    _g["macro_context"] += 1
                    continue

            macro_dd_skip = p.get("macro_dd_skip", None)
            if macro_dd_skip is not None and regime5 != "RECOVERY":
                macro_dd = float(row.get("macro_dd_pct", 0.0))
                if macro_dd < macro_dd_skip:
                    _g["macro_dd"] += 1
                    continue

            strength = str(row.get("window_strength", "STRONG"))
            if strength not in strength_allowed:
                _g["strength"] += 1
                continue

            _rsi_raw = df["rsi"].iat[i]
            if pd.isna(_rsi_raw):
                _g["rsi_nan"] += 1
                continue
            rsi = float(_rsi_raw)

            rsi_prev  = float(row.get("rsi_prev", 50))
            zscore    = float(row.get("zscore", 0))
            vol_r     = float(row.get("vol_ratio", 1))
            rsi_prev2 = float(df["rsi"].iloc[i-2]) if i >= 2 and not pd.isna(df["rsi"].iloc[i-2]) else rsi_prev

            in_cooldown = (self._last_buy_ts is not None and
                           (ts - self._last_buy_ts).total_seconds() < cooldown)
            if in_cooldown:
                _g["cooldown"] += 1
                continue

            in_psl_cooldown = (self._last_psl_ts is not None and
                               (ts - self._last_psl_ts).total_seconds() < psl_cooldown)
            if in_psl_cooldown:
                _g["psl_cooldown"] += 1
                continue

            entry_rsi_min = p.get("entry_rsi_min", 0)
            if rsi < entry_rsi_min:
                _g["entry_rsi_min"] += 1
                continue

            rsi_lb = rsi_lb_recovery if regime5 == "RECOVERY" else rsi_lb_bull
            rsi_lookback_slice = df["rsi"].iloc[max(0, i - rsi_lb):i]
            if rsi_lookback_slice.empty or rsi_lookback_slice.max() < 55:
                _g["rsi_lookback"] += 1
                continue

            rsi_drop = rsi_prev2 - rsi_prev
            rsi_turning_up = (rsi > rsi_prev) and (
                (rsi_drop >= 2.0 and rsi_prev > rsi_prev2) or
                (rsi_drop < 2.0)
            )

            rsi_pass    = (rsi < p.get("uptrend_rsi_max", 38)) and rsi_turning_up
            zscore_pass = zscore < zscore_max
            vol_pass    = vol_r >= p.get("vol_mult_min", 1.30)

            if rsi_pass and zscore_pass and vol_pass:
                effective_qty = base_qty * qty_scale_map.get(strength, 1.0)
                if p.get("target_bps") is None:
                    atr_pct = float(row.get("atr_pct", 0.005))
                    raw_bps = int(atr_pct * p["target_atr_mult"] * 10_000)
                    resolved_target = max(p["target_bps_min"],
                                        min(p["target_bps_max"], raw_bps))
                else:
                    resolved_target = p["target_bps"]

                self._buy(i, df, close, "uptrend_pb", effective_qty,
                          buy_fee_pct, resolved_target, min_profit, sell_fee_pct)
                _g["entered"] += 1
            else:
                if not rsi_pass: _g["sig_rsi"] += 1
                elif not zscore_pass: _g["sig_zscore"] += 1
                else: _g["sig_vol"] += 1

        if self._position.is_open:
            self._sell(len(df) - 1, df, float(df.iloc[-1]["close"]), "end_of_period", sell_fee_pct)

        res_df, res_dict = self._build_result(capital, preset_name)
        # Add TrendBot specific stats to the summary
        if res_dict:
            tdf = pd.DataFrame(self._trades)
            sells = tdf[tdf["side"] == "SELL"]
            tgt_r = sells[sells["reason"] == "target"]
            psl_r = sells[sells["reason"] == "pos_stop_loss"]
            ts_r  = sells[sells["reason"] == "time_stop"]
            res_dict.update({
                "target_fires": len(tgt_r),
                "target_pnl": float(tgt_r["pnl_after_fees"].sum()),
                "psl_fires": len(psl_r),
                "psl_pnl": float(psl_r["pnl_after_fees"].sum()),
                "time_stop_fires": len(ts_r),
                "time_stop_pnl": float(ts_r["pnl_after_fees"].sum()),
                "eop_pnl": float(sells[sells["reason"] == "end_of_period"]["pnl_after_fees"].sum()),
            })
        return res_df, res_dict

    def _buy(self, i, df, close, reason, qty, buy_fee_pct,
             target_bps, min_profit, sell_fee_pct):
        row = df.iloc[i]
        bv  = qty * close
        if bv > self._cash: return
        round_trip_fee = bv * (buy_fee_pct + sell_fee_pct)
        if (bv * (target_bps / 10_000)) < round_trip_fee * (1 + min_profit / 10_000): return

        fee = bv * buy_fee_pct
        self._cash               -= bv + fee
        self._position.qty        = qty
        self._position.avg_entry  = close
        self._position.peak_price = close
        self._position.entry_bar  = i
        self._position.bull_class = str(row.get("bull_class_h1", ""))
        self._position.entry_atr_pct = float(row.get("atr_pct", 0.005))
        self._position.lots = [Lot(qty=qty, price=close, fee=fee, ts=row["ts"], row_idx=len(self._trades))]
        self._last_buy_ts = row["ts"]
        self._position.target_bps = target_bps

        self._trades.append({
            "ts": row["ts"], "side": "BUY", "reason": reason,
            "regime5": str(row.get("regime5", "")),
            "price": close, "qty": qty, "fee": fee,
            "rsi": float(df["rsi"].iat[i]) if not pd.isna(df["rsi"].iat[i]) else float("nan"),
            "zscore": float(row.get("zscore", float("nan"))),
            "vol_ratio": float(row.get("vol_ratio", float("nan"))),
            "pnl": 0.0, "pnl_after_fees": 0.0, "win": float("nan"), "bars_held": float("nan"),
            "exit_price": float("nan"),
        })

    def _sell(self, i, df, close, reason, sell_fee_pct):
        p = self._position
        row = df.iloc[i]
        sell_val = p.qty * close
        sell_fee = sell_val * sell_fee_pct
        pnl = sell_val - sell_fee - p.cost_basis
        self._cumulative += pnl
        bh = i - p.entry_bar

        if reason == "pos_stop_loss": self._last_psl_ts = row["ts"]

        buy_row = self._trades[p.lots[0].row_idx]
        buy_row.update({"pnl": pnl, "pnl_after_fees": pnl, "win": 1.0 if pnl > 0 else 0.0, "bars_held": bh})

        self._trades.append({
            "ts": row["ts"], "side": "SELL", "reason": reason,
            "regime5": str(row.get("regime5", "")),
            "price": close, "qty": p.qty, "fee": sell_fee,
            "rsi": float(df["rsi"].iat[i]) if not pd.isna(df["rsi"].iat[i]) else float("nan"),
            "zscore": float("nan"),
            "vol_ratio": float("nan"),
            "pnl": pnl, "pnl_after_fees": pnl, "win": 1.0 if pnl > 0 else 0.0, "bars_held": bh,
            "exit_price": float("nan"),
        })
        self._cash += sell_val - sell_fee
        if reason != "end_of_period":
            self._realized_pnl += pnl
            self._trade_count  += 1
        p.reset()
