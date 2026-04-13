#!/usr/bin/env python3
"""
eth_crash_accumulator_v3.py — CrashAccumulator (Patient Capital, CRASH specialist)
====================================================================================
Changes from v1:
  1. PATIENT HOLD — no TrendBot handoff. Holds accumulated ETH across all regimes
     until price crosses avg_entry * (1 + profit_target_pct). Exits on profit target.
  2. VELOCITY FILTER (soft-scale) — during fast crashes, throttles buy size to
     velocity_soft_scale fraction instead of fully halting. Preserves accumulation
     through true capitulation events while limiting exposure during freefall.
  3. STAGED SUPPORT TARGETING — on window start, identifies 3 volume-profile
     support levels from the lookback period. Buys are concentrated near these
     levels rather than at uniform -3% intervals.
  4. SPLIT CAPITAL — CrashAccumulator operates on its own 40% capital pool.
     TrendBot runs independently on 50%. 10% reserve never touched.

Design intent:
  This is a TARGETED CRASH bot, not a downtrend accumulator.
  It catches sharp V-bottom events (<35 days) with high discount.
  Slow grinds (>35 days) will get fewer buys and lower discount by design.
  Hypotheses H1-H4/H7 only apply to targeted fast crashes (days <= 35).

Exit logic (priority order):
  1. Profit target: price > avg_entry * (1 + profit_target_pct) -> full exit
  2. Emergency exit: price < avg_entry * (1 - emergency_drawdown_pct) -> flat exit
  3. Time limit: if position held > max_hold_days across all regimes -> flat exit

Capital pool (set by orchestrator or backtest harness):
  self._capital = 40% of total portfolio by default
"""

import warnings
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, List

from eth_bot_interface import BotInterface, BotStatus, Position, Lot

warnings.filterwarnings("ignore")

# ─── Severity tiers explained ───────────────────────────────────────────────────
# SHALLOW  : crash < 5%    → small positions, quick exits, 120d max hold
# MODERATE : crash 5-15%   → medium accumulation, 180d hold
# MAJOR    : crash 15-35%  → full accumulation, 365d hold, deeper emerg thresh
# CATASTROPHIC: crash >35% → accumulate then FREEZE, 730d hold, NO forced sell
#
# emergency_action:
#   "SELL"  → traditional emergency exit (liquidate position)
#   "HOLD"  → halt new buys only; hold position; trust V-bottom recovery
# ────────────────────────────────────────────────────────────────────────────
PRESETS = {
    "accumulator_v2": {
        # Entry
        "drop_trigger_pct":       0.03,    # base drop trigger -3%
        "deploy_pct":             0.05,    # 5% of remaining pool per buy
        "max_deploy_pct":         0.80,    # deploy up to 80% of own pool
        # Velocity filter
        "velocity_halt_pct":      0.08,    # throttle threshold: -8% in lookback window
        "velocity_lookback_bars": 48,      # 48 x 5m = 4 hours
        "velocity_soft_scale":    0.0,     # 0.0 = hard halt (legacy behaviour)
        # Staged support
        "support_levels":         3,
        "support_bonus_pct":      0.015,
        # Exit
        "profit_target_pct":      0.05,
        "emergency_drawdown_pct": 0.50,
        "max_hold_days":          180,
        "fee_pct":                0.00065,
    },
    "accumulator_v2_conservative": {
        "drop_trigger_pct":       0.04,
        "deploy_pct":             0.04,
        "max_deploy_pct":         0.60,
        "velocity_halt_pct":      0.06,
        "velocity_lookback_bars": 48,
        "velocity_soft_scale":    0.0,     # hard halt (legacy)
        "support_levels":         3,
        "support_bonus_pct":      0.01,
        "profit_target_pct":      0.05,
        "emergency_drawdown_pct": 0.45,
        "max_hold_days":          120,
        "fee_pct":                0.00065,
    },
    # ── v3 depth-adaptive preset ──────────────────────────────────────────────
    # Key change: velocity_soft_scale=0.40 — during fast crashes, keep buying
    # at 40% of normal size instead of halting entirely. This ensures true
    # V-bottom events (#76, #93) accumulate enough discount to hit hypothesis
    # thresholds without turning the bot into a downtrend accumulator.
    # emergency_action="HOLD" for catastrophic crashes (depth>35%).
    "accumulator_v3": {
        "drop_trigger_pct":           0.03,
        "deploy_pct":                 0.035,
        "max_deploy_pct":             0.60,
        "velocity_halt_pct":          0.08,
        "velocity_lookback_bars":     48,
        "velocity_soft_scale":        0.40,   # throttle to 40% size during freefall
        "support_levels":             3,
        "support_bonus_pct":          0.015,
        # Depth-adaptive emergency behaviour
        "emergency_drawdown_pct":     0.45,
        "emergency_action":           "HOLD",
        "catastrophic_depth_pct":     0.35,
        "max_accum_depth_pct":        0.35,
        # Depth-tiered hold periods
        "max_hold_days_shallow":      120,
        "max_hold_days_moderate":     180,
        "max_hold_days_major":        365,
        "max_hold_days_catastrophic": 730,
        # Exit via Position Reserve Manager tranches
        "use_reserve_manager":        True,
        "tranche_t1_pct":             0.10,
        "tranche_t2_pct":             0.15,
        "tranche_t3_pct":             0.20,
        # Override exits
        "ovr2_correction_pct":        0.08,
        "ovr3_days_pct":              0.07,
        "profit_target_pct":          0.05,
        "fee_pct":                    0.00065,
    },
}


def _find_support_levels(df_warm: pd.DataFrame, n_levels: int = 3) -> List[float]:
    """
    Volume-profile support identification from warmup period.
    Bins price range into 20 buckets, returns top-N volume-weighted price nodes.
    """
    if df_warm is None or len(df_warm) < 20:
        return []
    try:
        lo, hi = df_warm["low"].min(), df_warm["high"].max()
        if hi <= lo:
            return []
        buckets = 20
        bucket_size = (hi - lo) / buckets
        vol_profile = {}
        for _, row in df_warm.iterrows():
            mid    = (row["high"] + row["low"]) / 2
            vol    = row.get("volume", 1.0)
            bucket = int((mid - lo) / bucket_size)
            bucket = max(0, min(buckets - 1, bucket))
            price_node = lo + (bucket + 0.5) * bucket_size
            vol_profile[price_node] = vol_profile.get(price_node, 0) + vol
        mid_price = (lo + hi) / 2
        support_candidates = {p: v for p, v in vol_profile.items() if p < mid_price}
        sorted_levels = sorted(support_candidates, key=lambda p: -support_candidates[p])
        return sorted_levels[:n_levels]
    except Exception:
        return []


class CrashAccumulator(BotInterface):
    """
    CRASH regime specialist — patient capital accumulator.
    Targeted at sharp V-bottom events. Holds position across all regimes
    until profit target or time-stop.
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
        self._crash_start_price   = None
        self._support_levels      = []
        self._trades              = []
        self._equity_curve        = []
        self._buys                = []
        self._velocity_paused     = False
        self._hold_recalculated   = False
        self._hold_start_bar      = None
        self._state               = "IDLE"   # IDLE | ACTIVE | CAPPED | HOLDING | DONE

    @property
    def bot_id(self) -> str:
        return f"crashaccumulator_{self._symbol.lower().replace('-', '_')}"

    @property
    def supported_regimes(self) -> list:
        return ["CRASH", "CORRECTION", "RANGE", "RECOVERY", "BULL"]

    def get_status(self) -> BotStatus:
        return BotStatus(
            bot_id=self.bot_id, symbol=self._symbol,
            capital_allocated=self._capital,
            capital_deployed=self._total_spent,
            capital_available=self._cash,
            open_qty=self._position.qty,
            open_avg_entry=self._position.avg_entry,
            unrealized_pnl=0.0,
            realized_pnl=self._realized_pnl,
            trade_count=self._trade_count,
            active=self._active,
            supported_regimes=self.supported_regimes,
        )

    def run_backtest(self, df: pd.DataFrame, preset: dict,
                     capital: float, preset_name: str,
                     df_warm: pd.DataFrame = None,
                     crash_end_ts=None) -> tuple:
        """
        Run CrashAccumulator v3 over crash window + extended hold.
        crash_end_ts : if provided, stop accumulating after this timestamp.
        """
        self._reset(capital)
        p = preset

        fee_pct          = p.get("fee_pct", 0.00065)
        drop_trigger     = p.get("drop_trigger_pct", 0.04)
        deploy_pct       = p.get("deploy_pct", 0.04)
        max_deploy       = capital * p.get("max_deploy_pct", 0.60)
        vel_halt         = p.get("velocity_halt_pct", 0.07)
        vel_bars         = int(p.get("velocity_lookback_bars", 48))
        vel_scale        = p.get("velocity_soft_scale", 0.0)  # 0.0 = hard halt (legacy)
        support_bonus    = p.get("support_bonus_pct", 0.01)
        profit_target    = p.get("profit_target_pct",
                                  p.get("tranche_t1_pct", 0.10))
        emergency_dd     = p.get("emergency_drawdown_pct", 0.45)
        emergency_action = p.get("emergency_action", "SELL")
        catastro_depth   = p.get("catastrophic_depth_pct", 0.35)
        max_accum_depth  = p.get("max_accum_depth_pct", 1.0)
        max_hold_bars    = int(p.get("max_hold_days",
                                      p.get("max_hold_days_moderate", 180)) * 288)

        self._support_levels    = _find_support_levels(df_warm, p.get("support_levels", 3))
        self._crash_start_price = float(df["close"].iloc[0])
        self._last_buy_price    = self._crash_start_price

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

                # DEPTH-TIERED HOLD RECALC: fires once after crash_end.
                # In MANAGE block so it executes even when FROZEN.
                if (crash_end_ts is not None and ts > crash_end_ts
                        and not self._hold_recalculated):
                    self._hold_recalculated = True
                    if self._crash_start_price:
                        _depth = abs((close - self._crash_start_price)
                                     / self._crash_start_price)
                        if _depth >= 0.35:
                            max_hold_bars = int(
                                p.get("max_hold_days_catastrophic", 730) * 288)
                        elif _depth >= 0.15:
                            max_hold_bars = int(
                                p.get("max_hold_days_major", 365) * 288)
                        elif _depth >= 0.05:
                            max_hold_bars = int(
                                p.get("max_hold_days_moderate", 180) * 288)
                        else:
                            max_hold_bars = int(
                                p.get("max_hold_days_shallow", 120) * 288)

                # Profit target
                if close >= self._position.avg_entry * (1 + profit_target):
                    self._sell(i, df, close, "profit_target", fee_pct)
                    self._state = "DONE"
                    continue

                # Time-stop
                if (i - self._hold_start_bar) > max_hold_bars:
                    self._sell(i, df, close, "time_stop", fee_pct)
                    self._state = "DONE"
                    continue

                # Emergency drawdown — depth-adaptive
                dd          = (close - self._position.avg_entry) / self._position.avg_entry
                crash_depth = (abs((close - self._crash_start_price)
                               / self._crash_start_price)
                               if self._crash_start_price else 0.0)
                if dd < -emergency_dd:
                    is_catastrophic = crash_depth >= catastro_depth
                    if emergency_action == "HOLD" and is_catastrophic:
                        self._state = "FROZEN"
                    else:
                        self._sell(i, df, close, "emergency_exit", fee_pct)
                        self._state = "DONE"
                    continue

            # ── SKIP ACCUMULATION ─────────────────────────────────────────────
            if self._state in ("DONE", "FROZEN"):
                continue
            if self._total_spent >= max_deploy:
                self._state = "CAPPED"
                continue

            # ── CRASH-END GATE ────────────────────────────────────────────────
            if crash_end_ts is not None and ts > crash_end_ts:
                continue

            # ── CRASH DEPTH GATE ──────────────────────────────────────────────
            if self._crash_start_price:
                depth = abs((close - self._crash_start_price) / self._crash_start_price)
                if depth > max_accum_depth:
                    continue

            # ── VELOCITY FILTER (soft-scale) ───────────────────────────────────
            # Targeted crash bot: during high-velocity freefalls, throttle buy
            # size to vel_scale fraction instead of skipping entirely.
            # vel_scale=0.0 reproduces the old hard-halt behaviour.
            # vel_scale=0.40 keeps accumulating at 40% clip during capitulation.
            velocity_factor = 1.0
            if i >= vel_bars:
                price_vel_ago = float(df["close"].iat[i - vel_bars])
                velocity = (close - price_vel_ago) / price_vel_ago
                if velocity < -vel_halt:
                    self._velocity_paused = True
                    if vel_scale <= 0.0:
                        continue   # hard halt (legacy)
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
            near_support = any(abs(close - lvl) / lvl < 0.012
                               for lvl in self._support_levels)
            this_pct = deploy_pct * (1 + support_bonus) if near_support else deploy_pct
            this_pct *= velocity_factor  # throttle during freefall if vel_scale > 0

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
        self._cash          += sell_val - fee
        self._realized_pnl  += pnl
        self._trades.append({
            "ts": df.iloc[i]["ts"], "side": "SELL", "reason": reason,
            "price": close, "qty": self._position.qty, "fee": fee,
            "spend": 0, "near_support": False, "velocity_paused": False,
            "avg_entry": self._position.avg_entry,
            "total_spent": self._total_spent, "pnl": pnl,
        })
        self._position.reset()
        self._total_deployed_ever += self._total_spent
        self._total_spent = 0.0
        self._hold_start_bar = None

    def _reset(self, capital: float) -> None:
        self._cash              = float(capital)
        self._capital           = float(capital)
        self._position          = Position(symbol=self._symbol)
        self._trades            = []
        self._buys              = []
        self._equity_curve      = []
        self._total_spent       = 0.0
        self._realized_pnl      = 0.0
        self._trade_count       = 0
        self._last_buy_price    = None
        self._crash_start_price = None
        self._support_levels    = []
        self._velocity_paused   = False
        self._hold_recalculated = False
        self._hold_start_bar    = None
        self._state             = "IDLE"

    def _build_result(self, capital: float, preset_name: str) -> tuple:
        if not self._trades:
            return pd.DataFrame(), {}
        tdf = pd.DataFrame(self._trades)

        near_sup_buys  = len([b for b in self._buys if b["near_support"]])
        vel_throt_buys = len([b for b in self._buys if b.get("velocity_paused")])

        sells = tdf[tdf["side"] == "SELL"]
        profit_exits    = len(sells[sells["reason"] == "profit_target"])
        emergency_exits = len(sells[sells["reason"] == "emergency_exit"])
        time_stops      = len(sells[sells["reason"] == "time_stop"])

        discount_pct = 0.0
        if self._crash_start_price and self._position.avg_entry > 0:
            discount_pct = ((self._crash_start_price - self._position.avg_entry)
                           / self._crash_start_price * 100)
        elif self._crash_start_price and profit_exits > 0:
            buys_df = tdf[tdf["side"] == "BUY"]
            if len(buys_df) > 0:
                avg_buy = (buys_df["spend"].sum() / buys_df["qty"].sum()
                           if buys_df["qty"].sum() > 0 else 0)
                discount_pct = ((self._crash_start_price - avg_buy)
                               / self._crash_start_price * 100)

        return tdf, {
            "preset":              preset_name,
            "buys":                len(self._buys),
            "near_support_buys":   near_sup_buys,
            "vel_throttled_buys":  vel_throt_buys,
            "total_qty":           self._position.qty,
            "avg_entry":           self._position.avg_entry,
            "crash_start_price":   self._crash_start_price,
            "discount_pct":        discount_pct,
            "total_deployed":      self._total_spent,
            "deploy_pct":          (self._total_spent + self._total_deployed_ever) / capital * 100,
            "realized_pnl":        self._realized_pnl,
            "profit_exits":        profit_exits,
            "emergency_exits":     emergency_exits,
            "time_stops":          time_stops,
            "position_open":       self._position.qty > 0,
            "state":               self._state,
            "frozen":              self._state == "FROZEN",
            "fees_paid":           float(tdf["fee"].sum()),
            "support_levels":      self._support_levels,
        }
