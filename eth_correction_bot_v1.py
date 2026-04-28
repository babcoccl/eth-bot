#!/usr/bin/env python3
"""
eth_correction_bot_v1.py  —  CorrectionBot (Mean-Reversion, CORRECTION specialist)
====================================================================================
Design intent:
  CorrectionBot targets CORRECTION regime events: drops of 5–25% that develop
  over 5–35 days and tend to recover within 30–90 days. These are NOT crashes;
  they are healthy pullbacks in otherwise constructive markets.

  Key differences from CrashAccumulator:
    - Shallower drop trigger (-2% vs -3%) to catch earlier entry
    - Faster profit target (+4% vs +5%) — corrections recover quicker
    - No FREEZE mode — corrections don't go catastrophic; use stop-loss instead
    - Smaller per-buy deploy to spread entries across the shorter window
    - Hard stop-loss at -12% from avg_entry — cuts losses if correction extends
      to true crash territory (at which point CrashAccumulator takes over)
    - Max hold 90 days (corrections that last longer are regime misclassified)
    - No depth gate — corrections rarely exceed 25% so max_accum_depth unused

  Orchestrator hand-off:
    When CorrectionBot stop-losses out, it signals the MacroSupervisor.
    The supervisor may then activate CrashAccumulator on the same position
    if the drop has deepened past CRASH threshold.

Capital pool:
  Allocated by MacroSupervisor. Suggested: 15–20% of total portfolio.
  Operates independently of CrashAccumulator capital.

Exit logic (priority order):
  1. Profit target : price >= avg_entry * (1 + profit_target_pct)
  2. Stop-loss     : price <= avg_entry * (1 - stop_loss_pct)
  3. Time-stop     : bars held > max_hold_days * 288

Supported regimes: CORRECTION, RANGE (shallow dips in ranging markets)
"""

import warnings
import numpy as np
import pandas as pd
from typing import List

from eth_bot_interface import BotInterface, BotStatus, Position, Lot

warnings.filterwarnings("ignore")

PRESETS = {
    # ── v1 standard preset ───────────────────────────────────────────────────
    # Tuned for 5–25% corrections over 5–35 days.
    # Hypothesis targets: disc >= 8%, buys >= 3, stop-loss rate < 20%.
    "correction_v1": {
        # Entry
        "drop_trigger_pct":       0.02,    # buy each -2% drop from last buy
        "deploy_pct":             0.06,    # 6% of remaining pool per buy
        "max_deploy_pct":         0.70,    # deploy up to 70% of own pool
        # Velocity filter — softer than crash bot; corrections are gentler
        "velocity_halt_pct":      0.06,    # throttle if -6% in lookback
        "velocity_lookback_bars": 24,      # 24 x 5m = 2 hours
        "velocity_soft_scale":    0.50,    # throttle to 50% during fast moves
        # Volume-profile support
        "support_levels":         3,
        "support_bonus_pct":      0.015,   # +1.5x at VP support levels
        # Exit
        "profit_target_pct":      0.04,    # exit at +4% above avg_entry
        "stop_loss_pct":          0.15,    # hard stop at -12% from avg_entry
        "max_hold_days":          90,      # time-stop at 90 days
        "fee_pct":                0.00065,
    },
    # ── v1 aggressive preset ─────────────────────────────────────────────────
    # For deeper corrections (10–25%). Wider profit target, deeper stop.
    "correction_v1_aggressive": {
        "drop_trigger_pct":       0.025,
        "deploy_pct":             0.07,
        "max_deploy_pct":         0.75,
        "velocity_halt_pct":      0.07,
        "velocity_lookback_bars": 24,
        "velocity_soft_scale":    0.50,
        "support_levels":         3,
        "support_bonus_pct":      0.02,
        "profit_target_pct":      0.06,
        "stop_loss_pct":          0.15,
        "max_hold_days":          90,
        "fee_pct":                0.00065,
    },
    # ── v1 conservative preset ───────────────────────────────────────────────
    # For shallow 5–10% corrections. Quick in, quick out.
    "correction_v1_conservative": {
        "drop_trigger_pct":       0.015,
        "deploy_pct":             0.05,
        "max_deploy_pct":         0.60,
        "velocity_halt_pct":      0.05,
        "velocity_lookback_bars": 24,
        "velocity_soft_scale":    0.40,
        "support_levels":         3,
        "support_bonus_pct":      0.01,
        "profit_target_pct":      0.03,
        "stop_loss_pct":          0.10,
        "max_hold_days":          60,
        "fee_pct":                0.00065,
    },
}


def _find_support_levels(df_warm: pd.DataFrame, n_levels: int = 3) -> List[float]:
    """Volume-profile support from warmup window (identical to CrashAccumulator)."""
    if df_warm is None or len(df_warm) < 20:
        return []
    try:
        lo, hi = df_warm["low"].min(), df_warm["high"].max()
        if hi <= lo:
            return []
        buckets    = 20
        bucket_sz  = (hi - lo) / buckets
        vol_profile = {}
        for _, row in df_warm.iterrows():
            mid    = (row["high"] + row["low"]) / 2
            vol    = row.get("volume", 1.0)
            bucket = int((mid - lo) / bucket_sz)
            bucket = max(0, min(buckets - 1, bucket))
            node   = lo + (bucket + 0.5) * bucket_sz
            vol_profile[node] = vol_profile.get(node, 0) + vol
        mid_price  = (lo + hi) / 2
        candidates = {p: v for p, v in vol_profile.items() if p < mid_price}
        return sorted(candidates, key=lambda p: -candidates[p])[:n_levels]
    except Exception:
        return []


class CorrectionBot(BotInterface):
    """
    CORRECTION regime specialist — mean-reversion accumulator.
    Fast entry on shallow dips, exits at modest profit target or hard stop-loss.
    """

    def __init__(self, symbol: str = "ETH-USD"):
        self._symbol              = symbol
        self._active              = True
        self._position            = Position(symbol=symbol)
        self._cash                = 0.0
        self._capital             = 0.0
        self._total_spent         = 0.0
        self._total_deployed_ever = 0.0
        self._realized_pnl        = 0.0
        self._trade_count         = 0
        self._last_buy_price      = None
        self._correction_start_price = None
        self._support_levels      = []
        self._trades              = []
        self._equity_curve        = []
        self._buys                = []
        self._velocity_paused     = False
        self._hold_start_bar      = None
        self._state               = "IDLE"  # IDLE | ACTIVE | CAPPED | DONE | STOPPED

    @property
    def bot_id(self) -> str:
        return f"correctionbot_{self._symbol.lower().replace('-', '_')}"

    @property
    def supported_regimes(self) -> list:
        return ["CORRECTION", "RANGE"]

    def get_status(self) -> BotStatus:
        return BotStatus(
            bot_id=self.bot_id, symbol=self._symbol,
            capital_allocated=self._capital,
            capital_deployed=self._total_spent,
            capital_available=self._cash,
            open_side=self._position.side,
            open_qty=self._position.qty,
            open_avg_entry=self._position.avg_entry,
            unrealized_pnl=0.0,
            realized_pnl=self._realized_pnl,
            trade_count=self._trade_count,
            active=self._active,
            supported_regimes=self.supported_regimes,
        )

    def run_backtest(
        self,
        df: pd.DataFrame,
        preset: dict,
        capital: float,
        preset_name: str,
        df_warm: pd.DataFrame = None,
        correction_end_ts=None,
    ) -> tuple:
        """
        Run CorrectionBot over a correction window + recovery hold period.

        correction_end_ts : optional — stop accumulating after this timestamp;
                            only monitor exits beyond this point.
        """
        self._reset(capital)
        p = preset

        fee_pct       = p.get("fee_pct", 0.00065)
        drop_trigger  = p.get("drop_trigger_pct", 0.02)
        deploy_pct    = p.get("deploy_pct", 0.06)
        max_deploy    = capital * p.get("max_deploy_pct", 0.70)
        vel_halt      = p.get("velocity_halt_pct", 0.06)
        vel_bars      = int(p.get("velocity_lookback_bars", 24))
        vel_scale     = p.get("velocity_soft_scale", 0.50)
        support_bonus = p.get("support_bonus_pct", 0.015)
        profit_target = p.get("profit_target_pct", 0.04)
        stop_loss     = p.get("stop_loss_pct", 0.12)
        max_hold_bars = int(p.get("max_hold_days", 90) * 288)

        self._support_levels         = _find_support_levels(df_warm, p.get("support_levels", 3))
        self._correction_start_price = float(df["close"].iloc[0])
        self._last_buy_price         = self._correction_start_price

        for i in range(len(df)):
            row   = df.iloc[i]
            close = float(row["close"])
            ts    = row["ts"]

            self._equity_curve.append(
                self._cash + self._position.qty * close
            )

            # ── MANAGE OPEN POSITION ─────────────────────────────────────────
            if self._position.is_open:
                if self._hold_start_bar is None:
                    self._hold_start_bar = i

                dd = (close - self._position.avg_entry) / self._position.avg_entry

                # Profit target
                if dd >= profit_target:
                    self._sell(i, df, close, "profit_target", fee_pct)
                    self._state = "DONE"
                    continue

                # Hard stop-loss — correction has become a crash; hand off to CrashAccumulator
                if dd <= -stop_loss:
                    self._sell(i, df, close, "stop_loss", fee_pct)
                    self._state = "STOPPED"
                    continue

                # Time-stop
                if (i - self._hold_start_bar) > max_hold_bars:
                    self._sell(i, df, close, "time_stop", fee_pct)
                    self._state = "DONE"
                    continue

            # ── SKIP ACCUMULATION ─────────────────────────────────────────────
            if self._state in ("DONE", "STOPPED"):
                continue
            if self._total_spent >= max_deploy:
                self._state = "CAPPED"
                continue

            # ── CORRECTION-END GATE ───────────────────────────────────────────
            if correction_end_ts is not None and ts > correction_end_ts:
                continue

            # ── VELOCITY FILTER (soft-scale) ──────────────────────────────────
            velocity_factor = 1.0
            if i >= vel_bars:
                price_vel_ago = float(df["close"].iat[i - vel_bars])
                velocity = (close - price_vel_ago) / price_vel_ago
                if velocity < -vel_halt:
                    self._velocity_paused = True
                    if vel_scale <= 0.0:
                        continue
                    velocity_factor = vel_scale
                else:
                    self._velocity_paused = False
            if self._velocity_paused and vel_scale <= 0.0:
                continue

            # ── DROP TRIGGER ──────────────────────────────────────────────────
            drop = (close - self._last_buy_price) / self._last_buy_price
            if drop > -drop_trigger:
                continue

            # ── STAGED SUPPORT SIZING ─────────────────────────────────────────
            near_support = any(
                abs(close - lvl) / lvl < 0.012 for lvl in self._support_levels
            )
            this_pct  = deploy_pct * (1 + support_bonus) if near_support else deploy_pct
            this_pct *= velocity_factor

            spend = min(
                self._cash * this_pct,
                max_deploy - self._total_spent,
                self._cash * 0.95,
            )
            if spend < 1.0:
                continue

            qty = spend / close
            fee = spend * fee_pct
            self._cash        -= (spend + fee)
            self._total_spent += spend
            self._position.qty       += qty
            self._position.avg_entry  = self._total_spent / self._position.qty
            self._last_buy_price      = close
            self._state               = "ACTIVE"

            self._buys.append({
                "ts": ts, "price": close, "qty": qty, "spend": spend,
                "fee": fee, "near_support": near_support,
                "velocity_paused": self._velocity_paused,
                "avg_entry": self._position.avg_entry,
                "total_spent": self._total_spent,
            })
            self._trades.append({
                "ts": ts, "side": "BUY", "reason": "drop_dca",
                "price": close, "qty": qty, "fee": fee, "spend": spend,
                "near_support": near_support,
                "velocity_paused": self._velocity_paused,
                "avg_entry": self._position.avg_entry,
                "total_spent": self._total_spent, "pnl": 0.0,
            })
            self._trade_count += 1

        return self._build_result(capital, preset_name)

    def _sell(self, i, df, close, reason, fee_pct):
        if not self._position.is_open:
            return
        sell_val = self._position.qty * close
        fee      = sell_val * fee_pct
        pnl      = sell_val - fee - self._total_spent
        self._cash         += sell_val - fee
        self._realized_pnl += pnl
        self._trades.append({
            "ts": df.iloc[i]["ts"], "side": "SELL", "reason": reason,
            "price": close, "qty": self._position.qty, "fee": fee,
            "spend": 0, "near_support": False, "velocity_paused": False,
            "avg_entry": self._position.avg_entry,
            "total_spent": self._total_spent, "pnl": pnl,
        })
        self._position.reset()
        self._total_deployed_ever += self._total_spent
        self._total_spent    = 0.0
        self._hold_start_bar = None

    def _reset(self, capital: float) -> None:
        self._cash                   = float(capital)
        self._capital                = float(capital)
        self._position               = Position(symbol=self._symbol)
        self._trades                 = []
        self._buys                   = []
        self._equity_curve           = []
        self._total_spent            = 0.0
        self._realized_pnl           = 0.0
        self._trade_count            = 0
        self._last_buy_price         = None
        self._correction_start_price = None
        self._support_levels         = []
        self._velocity_paused        = False
        self._hold_start_bar         = None
        self._state                  = "IDLE"

    def _build_result(self, capital: float, preset_name: str) -> tuple:
        if not self._trades:
            return pd.DataFrame(), {}
        tdf = pd.DataFrame(self._trades)

        near_sup_buys  = len([b for b in self._buys if b["near_support"]])
        vel_throt_buys = len([b for b in self._buys if b.get("velocity_paused")])

        sells           = tdf[tdf["side"] == "SELL"]
        profit_exits    = len(sells[sells["reason"] == "profit_target"])
        stop_loss_exits = len(sells[sells["reason"] == "stop_loss"])
        time_stops      = len(sells[sells["reason"] == "time_stop"])

        discount_pct = 0.0
        if self._correction_start_price and self._position.avg_entry > 0:
            discount_pct = (
                (self._correction_start_price - self._position.avg_entry)
                / self._correction_start_price * 100
            )
        elif self._correction_start_price and (profit_exits > 0 or stop_loss_exits > 0):
            buys_df = tdf[tdf["side"] == "BUY"]
            if len(buys_df) > 0 and buys_df["qty"].sum() > 0:
                avg_buy = buys_df["spend"].sum() / buys_df["qty"].sum()
                discount_pct = (
                    (self._correction_start_price - avg_buy)
                    / self._correction_start_price * 100
                )

        pos_open = self._position.qty > 0
        if profit_exits > 0:    exit_str = "PROFIT_TARGET"
        elif stop_loss_exits > 0: exit_str = "STOP_LOSS"
        elif time_stops > 0:    exit_str = "TIME_STOP"
        elif pos_open:          exit_str = "STILL_OPEN"
        else:                   exit_str = self._state

        return tdf, {
            "preset":              preset_name,
            "buys":                len(self._buys),
            "near_support_buys":   near_sup_buys,
            "vel_throttled_buys":  vel_throt_buys,
            "total_qty":           self._position.qty,
            "avg_entry":           self._position.avg_entry,
            "correction_start_price": self._correction_start_price,
            "discount_pct":        discount_pct,
            "total_deployed":      self._total_spent,
            "deploy_pct":          (self._total_spent + self._total_deployed_ever) / capital * 100,
            "realized_pnl":        self._realized_pnl,
            "profit_exits":        profit_exits,
            "stop_loss_exits":     stop_loss_exits,
            "time_stops":          time_stops,
            "position_open":       pos_open,
            "exit_str":            exit_str,
            "state":               self._state,
            "stopped":             self._state == "STOPPED",
            "fees_paid":           float(tdf["fee"].sum()),
            "support_levels":      self._support_levels,
        }
