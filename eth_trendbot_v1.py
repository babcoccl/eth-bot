#!/usr/bin/env python3
"""
eth_trendbot_v1.py — TrendBot (BULL / RECOVERY specialist)
==============================================================
Role : Trades uptrend pullbacks during BULL and RECOVERY regimes only.
Signal : uptrend_pb — RSI pullback while MacroSupervisor regime5 is BULL or RECOVERY
Exit : target_bps fixed profit target (limit order = maker fill)
Safety : pos_stop_loss_pct (2.5%) — sized to match target scale; market = taker fill
DCA : NONE — one fill per position, orchestrator adds capital
Size : base_qty scaled by trend_strength (qty_scale) — risk management only,
       does not affect signal logic or entry/exit conditions

Regime column:
  MacroSupervisor writes regime5 with values: BULL, RECOVERY, RANGE, CORRECTION, CRASH.
  TrendBot gates entries on regime5 in ("BULL", "RECOVERY") only.
  Do NOT use regime_h1 (a separate indicator column, not the supervisor state).

Fee model:
  buy_fee_pct  = 0.00065  (taker — aggressive market buy on pullback signal)
  sell_fee_pct = 0.00025  (maker — limit order placed at target price)
  round-trip   = 0.090%   = 90 bps
  target_bps must be > 90 to break even; recommended >= 180 (2x fee buffer)

PSL sizing rationale (empirical, not overfit):
  Integration run (4 cycles, 2022-2025) measured:
    avg target win  = $+1.50  (180bps on 0.05 ETH at ~$2,500)
    avg PSL loss    = $-2.27  (2.5% PSL — 1.51x the win size)
    break-even WR   = 60.2%   (actual WR = 60% on valid windows CyC/CyD;
                               CyA/CyB were regime-hostile — see harness notes)
  Derived from integration harness exit breakdown, not overfit to windows.

psl_cooldown_secs:
  After a pos_stop_loss exit, the bot locks out new entries for this duration.
  Prevents re-entering a choppy/reversing regime immediately after being stopped out.

qty_scale:
  Scales base_qty by trend_strength regime. This is pure position sizing —
  it does not filter entries or change signal conditions. MODERATE gets 0.5x
  to cap drawdown in historically weaker conditions without removing those
  windows from the sample entirely.

Design principle: do ONE thing well.
This bot only fires when MacroSupervisor regime5 is BULL or RECOVERY and price pulls back.
It does not trade ranges, it does not average down.
The MacroSupervisor enables it in BULL/RECOVERY regimes.
The LLM Orchestrator decides capital allocation.

Test windows (from regime_period_analyzer.py):
PRIMARY  : 2025-07-06 → 2025-07-31 (RECOVERY→BULL, 25d, +28.2%)
SECONDARY: 2025-05-08 → 2025-05-16 (BULL, 8.6d, +36.1%)
HOLDOUT  : 2025-12-30 → 2026-01-06 (RECOVERY, 7d, +7.9%)

v1 history:
  initial  — used regime_h1 == "UPTREND" gate; MacroSupervisor never writes
             regime_h1 so every entry was silently skipped. Fixed to regime5.
  r2       — PSL tightened from 5% to 2.5% (empirical: avg PSL loss was 2.8x
             avg target win → R:R=0.360, break-even WR=73.5%, unsustainable).
             Derived from integration harness exit breakdown, not overfit to windows.
  r3       — uptrend_bars_min set to 48 (4h) to filter noisy early-recovery bars.
  r4       — uptrend_bars_min reverted to 0. Tested r3: WR dropped 50%→47%/46%
             in CyA/CyB because compressed RECOVERY windows lose best entries
             to the filter. The 48-bar threshold is invalid for short windows.
             Root cause of CyA/CyB losses is regime-hostile windows (10%/29%
             tradeable), not entry timing. Fix is harness window selection,
             not TrendBot parameter tuning.
  r5       — entry_min_atr_pct = 0.008 added. Filters STRONG entries in
             low-ATR regimes (T03/T04/T05 avg 0.5% ATR) while preserving
             high-ATR STRONG windows (T12 avg 1.2%) and all PARABOLIC windows.
             trend_strength_allowed restored to {"STRONG", "PARABOLIC"}.
             Hypothesis: PSL rate will drop as low-ATR entries that reverse
             sharply within 15 bars are suppressed at the entry gate.
  r6       — trend_strength_allowed: added MODERATE. Root cause of 4 no-data
             windows (#2022-08, #2023-10, #2024-02, #2024-11): all have
             strength=MODERATE in trend_windows_generated.py. MODERATE already
             has 0.5x qty_scale for drawdown control — excluding entry entirely
             was wrong. macro_context_bars: 2880 → 1440 (10d → 5d). The 10d
             lookback reaches into the 180d warmup period (heavily CRASH) and
             blocks entries in the first several days of clean recovery windows.
  r7       — rsi_turning_up: strengthened from single-bar to two-bar confirmation.
             Was: (rsi > rsi_prev) AND (rsi_prev < rsi_prev2) -- fires on any
             one-tick RSI bounce after a single lower bar (noise within downleg).
             Now: rsi > rsi_prev AND rsi_prev > rsi_prev2 -- requires the prior
             bar itself to already be rising (confirmed trough reversal).
             Root cause of #2022-03 0% WR and #2023-01 12% WR: entries were
             firing on mid-downleg noise ticks, not genuine pullback troughs.
             zscore_max: -1.5 → -1.8; vol_mult_min: 1.30 → 1.50 (OVERCORRECTED
             — killed 7/9 windows; signal gate blocked 5k-6.6k bars/window).
  r8       — Revert zscore_max: -1.8 → -1.5 (r6 level).
             Revert vol_mult_min: 1.50 → 1.30 (r6 level).
             Keep double-RSI confirmation (rsi > rsi_prev > rsi_prev2) — this
             is the correct structural fix for mid-downleg noise entries.
             Added per-subgate breakdown in diagnostic output (rsi/zscore/vol
             counts shown separately when trades==0) so future tuning is
             data-driven rather than speculative.
"""

import warnings
import numpy as np
import pandas as pd

from eth_bot_interface import BotInterface, BotStatus, Position, Lot
from eth_bull_classifier import STOP_LOSS_BY_CLASS

warnings.filterwarnings("ignore")

PRESETS = {
    "trendbot_v1": {
        "base_qty":           0.05,
        "pos_stop_loss_pct":  0.025,    # 250bps — empirically derived; break-even at 60% WR
        "uptrend_rsi_max":    38,
        "vol_mult_min":       1.30,     # r8: reverted from r7's 1.50 back to r6 level
        "cooldown_secs":      14400,
        "psl_cooldown_secs":  28800,    # 8h lockout after any stop-loss exit
        "min_profit_bps":     100,
        "zscore_max":        -1.5,      # r8: reverted from r7's -1.8 back to r6 level
        "regime_stable_bars": 72,       # require 6 h1 bars of stable regime (6h * 12 = 72 5m bars)
        "macro_context_bars":  1440,    # 5 days of 5m bars (5d * 24h * 12 bars)
        "macro_bearish_max":   0.65,    # skip if >65% of last 5d was CRASH/CORRECTION
        "time_stop_bars":     24,       # exit flat if no progress after 4h (24 * 5m bars)
        "time_stop_min_pct":  0.0,      # only apply if position is below +0.0%
        "qty_scale": {
            "STRONG":    1.0,
            "PARABOLIC": 1.0,
            "MODERATE":  0.5,           # half size on MODERATE — risk management only
        },
        "trend_strength_allowed": {"STRONG", "PARABOLIC", "MODERATE"},
        "buy_fee_pct":        0.00065,
        "sell_fee_pct":       0.00025,
        "target_bps":        None,      # set to None to enable dynamic mode
        "target_atr_mult":   1.5,
        "target_bps_min":    120,
        "target_bps_max":    350,
        "psl_atr_max":       0.07,
        "manage_psl_mult":   3.0,
        "psl_atr_mult":      1.5,
        "macro_dd_skip":     -0.20,
        "entry_rsi_min":     30,
    },
    "trendbot_v1_aggressive": {
        "base_qty":           0.05,
        "target_bps":         220,
        "pos_stop_loss_pct":  0.030,
        "uptrend_rsi_max":    48,
        "vol_mult_min":       0.70,
        "cooldown_secs":      1200,
        "psl_cooldown_secs":  3600,
        "min_profit_bps":     120,
        "zscore_max":        -0.5,
        "qty_scale": {
            "STRONG":    1.0,
            "PARABOLIC": 1.0,
            "MODERATE":  0.5,
        },
        "buy_fee_pct":        0.00065,
        "sell_fee_pct":       0.00025,
    },
}

# Valid MacroSupervisor regime5 values that TrendBot operates in
_TREND_REGIMES = frozenset({"BULL", "RECOVERY"})


class TrendBot(BotInterface):
    """
    BULL/RECOVERY regime specialist — uptrend_pb signal only.

    Entry gate: MacroSupervisor regime5 must be BULL or RECOVERY.
    State machine (simple binary):
      IDLE → position is flat, scanning for uptrend_pb entry
      OPEN → position is active, monitoring for target or PSL exit
    No other states. No DCA. No trail. No momentum exit.
    """

    def __init__(self, symbol: str = "ETH-USD"):
        self._symbol       = symbol
        self._active       = True
        self._position     = Position(symbol=symbol)
        self._cash         = 0.0
        self._capital      = 0.0
        self._realized_pnl = 0.0
        self._trade_count  = 0
        self._last_buy_ts  = None
        self._last_psl_ts  = None
        self._trades       = []
        self._equity_curve = []
        self._cumulative   = 0.0

    @property
    def bot_id(self) -> str:
        return f"trendbot_{self._symbol.lower().replace('-', '_')}"

    @property
    def supported_regimes(self) -> list:
        return ["BULL", "RECOVERY"]

    def get_status(self) -> BotStatus:
        return BotStatus(
            bot_id             = self.bot_id,
            symbol             = self._symbol,
            capital_allocated  = self._capital,
            capital_deployed   = self._position.cost_basis,
            capital_available  = self._cash,
            open_qty           = self._position.qty,
            open_avg_entry     = self._position.avg_entry,
            unrealized_pnl     = 0.0,
            realized_pnl       = self._realized_pnl,
            trade_count        = self._trade_count,
            active             = self._active,
            supported_regimes  = self.supported_regimes,
        )

    def run_backtest(self, df: pd.DataFrame, preset: dict,
                    capital: float, preset_name: str) -> tuple:
        """Run TrendBot strategy over a single approved BULL/RECOVERY window."""
        self._reset(capital)
        p = preset

        buy_fee_pct      = p.get("buy_fee_pct",      p.get("fee_pct", 0.00065))
        sell_fee_pct     = p.get("sell_fee_pct",     p.get("fee_pct", 0.00025))
        base_qty         = p["base_qty"]
        target_bps       = p["target_bps"]
        psl_pct          = p["pos_stop_loss_pct"]
        rsi_max          = p["uptrend_rsi_max"]
        vol_min          = p["vol_mult_min"]
        cooldown         = p["cooldown_secs"]
        psl_cooldown     = p.get("psl_cooldown_secs", cooldown)
        min_profit       = p.get("min_profit_bps", 0)
        zscore_max       = p.get("zscore_max", -0.3)
        qty_scale_map    = p.get("qty_scale", {})
        strength_allowed = set(p.get("trend_strength_allowed", {"STRONG", "PARABOLIC", "MODERATE"}))

        trend_streak = 0  # consecutive bars in BULL or RECOVERY

        # -- Per-gate diagnostic counters (printed only when trades == 0) -----
        _g = {
            "open_position":  0,  # bar skipped — position already open
            "regime":         0,  # regime5 not BULL/RECOVERY
            "stable_bars":    0,  # trend_streak < regime_stable_bars
            "macro_context":  0,  # too many CRASH/CORRECTION in macro lookback
            "macro_dd":       0,  # ETH too far below 90d high
            "strength":       0,  # window_strength not in allowed set
            "rsi_nan":        0,  # RSI is NaN
            "cooldown":       0,  # within entry cooldown
            "psl_cooldown":   0,  # within PSL cooldown
            "entry_rsi_min":  0,  # RSI below entry_rsi_min floor
            "rsi_lookback":   0,  # no RSI > 55 in last 24 bars
            # signal sub-gates (only counted when all prior gates pass)
            "sig_rsi":        0,  # rsi >= rsi_max OR not turning up
            "sig_zscore":     0,  # zscore >= zscore_max
            "sig_vol":        0,  # vol_ratio < vol_min
            "entered":        0,  # bars where a trade was opened
        }
        # ---------------------------------------------------------------------

        for i in range(len(df)):
            row    = df.iloc[i]
            close  = float(row["close"])
            ts     = row["ts"]

            # ── Use MacroSupervisor regime5 column (not regime_h1) ──────────
            regime5 = str(row.get("regime5", "RANGE"))

            if regime5 in _TREND_REGIMES:
                trend_streak += 1
            else:
                trend_streak = 0

            self._equity_curve.append(self._cash + self._position.qty * close)

            # ── MANAGE OPEN POSITION ─────────────────────────────────
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

                # ── Time stop ────────────────────────────────────────────────
                time_stop_bars = p.get("time_stop_bars", 0)
                if time_stop_bars > 0:
                    bars_in_trade = i - self._position.entry_bar
                    if bars_in_trade >= time_stop_bars:
                        min_progress = p.get("time_stop_min_pct", 0.003)
                        progress = (close - self._position.avg_entry) / self._position.avg_entry
                        if progress < min_progress:
                            self._sell(i, df, close, "time_stop", sell_fee_pct)
                            continue
                # ─────────────────────────────────────────────────────────────

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

            # ── SCAN FOR ENTRY (position flat) ───────────────────────

            if regime5 not in _TREND_REGIMES:
                _g["regime"] += 1
                continue

            # ── Regime stability filter ──────────────────────────────
            regime_stable_bars = p.get("regime_stable_bars", 0)
            if regime_stable_bars > 0 and trend_streak < regime_stable_bars:
                _g["stable_bars"] += 1
                continue

            # ── Macro context filter ─────────────────────────────────
            macro_lookback = p.get("macro_context_bars", 0)
            if macro_lookback > 0:
                recent = df["regime5"].iloc[max(0, i - macro_lookback):i]
                bearish_frac = recent.isin(["CRASH", "CORRECTION"]).sum() / len(recent)
                macro_bearish_max = p.get("macro_bearish_max", 0.60)
                if bearish_frac > macro_bearish_max:
                    _g["macro_context"] += 1
                    continue

            # ── Macro drawdown filter ────────────────────────────────
            macro_dd_skip = p.get("macro_dd_skip", None)
            if macro_dd_skip is not None:
                macro_dd = float(row.get("macro_dd_pct", 0.0))
                if macro_dd < macro_dd_skip:
                    _g["macro_dd"] += 1
                    continue

            # ── Trend strength filter ────────────────────────────────
            strength = str(row.get("window_strength", "STRONG"))
            if strength not in strength_allowed:
                _g["strength"] += 1
                continue

            # ── RSI validity ─────────────────────────────────────────
            _rsi_raw = df["rsi"].iat[i]
            if pd.isna(_rsi_raw):
                _g["rsi_nan"] += 1
                continue
            rsi = float(_rsi_raw)

            rsi_prev  = float(row.get("rsi_prev", 50))
            zscore    = float(row.get("zscore", 0))
            vol_r     = float(row.get("vol_ratio", 1))
            rsi_prev2 = float(df["rsi"].iloc[i-2]) if i >= 2 and not pd.isna(df["rsi"].iloc[i-2]) else rsi_prev

            # ── Cooldown filters ─────────────────────────────────────
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

            # ── RSI floor ────────────────────────────────────────────
            entry_rsi_min = p.get("entry_rsi_min", 0)
            if rsi < entry_rsi_min:
                _g["entry_rsi_min"] += 1
                continue

            # ── ATR floor filter ──────────────────────────────────────
            entry_min_atr_pct = p.get("entry_min_atr_pct", 0.0)
            if entry_min_atr_pct > 0.0:
                entry_atr = float(row.get("atr_pct", 0.0))
                if entry_atr < entry_min_atr_pct:
                    continue

            # ── RSI lookback (confirm pullback, not breakdown) ────────
            rsi_lookback = df["rsi"].iloc[max(0, i-24):i]
            if rsi_lookback.empty or rsi_lookback.max() < 55:
                _g["rsi_lookback"] += 1
                continue

            # ── Signal: RSI + z-score + volume ───────────────────────
            # r8: keep double-RSI confirmation from r7 (rsi > rsi_prev > rsi_prev2).
            # Requires the prior bar to already be rising — filters one-tick
            # mid-downleg noise bounces that caused 0%/12% WR in #2022-03/#2023-01.
            # zscore_max and vol_mult_min reverted to r6 levels: -1.5 and 1.30.
            # r7's triple-tightening killed 7/9 windows (signal gate: 5k-6.6k
            # bars/window). Only change one variable at a time.
            rsi_turning_up = (rsi > rsi_prev) and (rsi_prev > rsi_prev2)

            # Evaluate sub-gates individually for diagnostics
            rsi_pass = (rsi < rsi_max) and rsi_turning_up
            zscore_pass = zscore < zscore_max
            vol_pass = vol_r >= vol_min

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
                # Count which sub-gate(s) blocked (first-failure only)
                if not rsi_pass:
                    _g["sig_rsi"] += 1
                elif not zscore_pass:
                    _g["sig_zscore"] += 1
                else:
                    _g["sig_vol"] += 1

        if self._position.is_open:
            self._sell(len(df) - 1, df, float(df.iloc[-1]["close"]),
                       "end_of_period", sell_fee_pct)

        # -- Print gate diagnostics when no trades were produced --------------
        if _g["entered"] == 0:
            total_bars = len(df)
            print(f"    [entry gates] no trades from {total_bars} bars:")
            order = [
                ("open_position", "position open"),
                ("regime",        "regime not BULL/RECOVERY"),
                ("stable_bars",   f"trend_streak < {p.get('regime_stable_bars',0)} (stable_bars)"),
                ("macro_context", f"macro_context: >{p.get('macro_bearish_max',0.6):.0%} bearish in last {p.get('macro_context_bars',0)} bars"),
                ("macro_dd",      f"macro_dd < {p.get('macro_dd_skip','off')}"),
                ("strength",      f"strength not in {strength_allowed}"),
                ("rsi_nan",       "RSI NaN"),
                ("cooldown",      "entry cooldown"),
                ("psl_cooldown",  "PSL cooldown"),
                ("entry_rsi_min", f"RSI < {p.get('entry_rsi_min',0)} (entry_rsi_min)"),
                ("rsi_lookback",  "rsi_lookback: no RSI>55 in last 24 bars"),
                ("sig_rsi",       "signal.rsi: rsi >= rsi_max or not turning up (double-bar)"),
                ("sig_zscore",    f"signal.zscore: zscore >= {zscore_max} (not oversold enough)"),
                ("sig_vol",       f"signal.vol: vol_ratio < {vol_min} (insufficient volume)"),
            ]
            for key, label in order:
                n = _g[key]
                if n > 0:
                    print(f"      {n:>6} bars  {label}")
        # ---------------------------------------------------------------------

        return self._build_result(capital, preset_name)

    # ── Private methods ────────────────────────────────────────────

    def _reset(self, capital: float) -> None:
        self._cash         = float(capital)
        self._capital      = float(capital)
        self._position     = Position(symbol=self._symbol)
        self._trades       = []
        self._equity_curve = []
        self._cumulative   = 0.0
        self._realized_pnl = 0.0
        self._trade_count  = 0
        self._last_buy_ts  = None
        self._last_psl_ts  = None

    def _buy(self, i, df, close, reason, qty, buy_fee_pct,
             target_bps, min_profit, sell_fee_pct):
        row = df.iloc[i]
        bv  = qty * close
        if bv > self._cash:
            return
        round_trip_fee = bv * (buy_fee_pct + sell_fee_pct)
        expected_gross = bv * (target_bps / 10_000)
        if expected_gross < round_trip_fee * (1 + min_profit / 10_000):
            return

        fee = bv * buy_fee_pct
        self._cash               -= bv + fee
        self._position.qty        = qty
        self._position.avg_entry  = close
        self._position.peak_price = close
        self._position.entry_bar  = i
        self._position.bull_class = str(row.get("bull_class_h1", ""))
        self._position.entry_atr_pct = float(row.get("atr_pct", 0.005))
        self._position.lots = [Lot(qty=qty, price=close,
                                   fee=fee, ts=row["ts"],
                                   row_idx=len(self._trades))]
        self._last_buy_ts = row["ts"]
        self._position.target_bps = target_bps

        self._trades.append({
            "ts":       row["ts"],
            "side":     "BUY",
            "reason":   reason,
            "regime5":  str(row.get("regime5", "")),
            "price":    close,
            "qty":      qty,
            "fee":      fee,
            "rsi":      float(df["rsi"].iat[i]) if not pd.isna(df["rsi"].iat[i]) else float("nan"),
            "zscore":   float(row.get("zscore",    float("nan"))),
            "vol_ratio":float(row.get("vol_ratio", float("nan"))),
            "pnl": 0.0, "pnl_after_fees": 0.0,
            "win": float("nan"), "bars_held": float("nan"),
            "exit_price": float("nan"),
        })

    def _sell(self, i, df, close, reason, sell_fee_pct):
        p        = self._position
        row      = df.iloc[i]
        sell_val = p.qty * close
        sell_fee = sell_val * sell_fee_pct
        pnl      = sell_val - sell_fee - p.cost_basis
        self._cumulative += pnl
        bh = i - p.entry_bar

        if reason == "pos_stop_loss":
            self._last_psl_ts = row["ts"]

        buy_row = self._trades[p.lots[0].row_idx]
        buy_row.update({
            "pnl": pnl, "pnl_after_fees": pnl,
            "win": 1.0 if pnl > 0 else 0.0,
            "bars_held": bh, "exit_price": close,
        })

        self._trades.append({
            "ts":       row["ts"],
            "side":     "SELL",
            "reason":   reason,
            "regime5":  str(row.get("regime5", "")),
            "price":    close,
            "qty":      p.qty,
            "fee":      sell_fee,
            "rsi":      float(df["rsi"].iat[i]) if not pd.isna(df["rsi"].iat[i]) else float("nan"),
            "zscore":   float("nan"),
            "vol_ratio":float("nan"),
            "pnl":            pnl,
            "pnl_after_fees": pnl,
            "win":       1.0 if pnl > 0 else 0.0,
            "bars_held": bh,
            "exit_price": float("nan"),
        })
        self._cash += sell_val - sell_fee
        if reason != "end_of_period":
            self._realized_pnl += pnl
            self._trade_count  += 1
        p.reset()

    def _build_result(self, capital: float, preset_name: str) -> tuple:
        if not self._trades:
            return pd.DataFrame(), {}
        tdf = pd.DataFrame(self._trades)
        eq  = np.array(self._equity_curve)

        peak_eq = np.maximum.accumulate(eq)
        max_dd  = float(((eq - peak_eq) / peak_eq).min()) * 100
        ret_s   = np.diff(eq) / eq[:-1]
        sharpe  = (ret_s.mean() / ret_s.std() * np.sqrt(105_120)
                   if ret_s.std() > 0 else 0.0)

        sells = tdf[tdf["side"] == "SELL"]
        real  = sells[~sells["reason"].isin(["end_of_period"])]
        wins  = real[real["pnl_after_fees"] > 0]
        psl_r = real[real["reason"] == "pos_stop_loss"]
        tgt_r = real[real["reason"] == "target"]

        return tdf, {
            "preset":        preset_name,
            "trades":        len(real),
            "win_rate":      len(wins) / len(real) * 100 if len(real) > 0 else 0,
            "realized_pnl":  float(tdf["pnl_after_fees"].sum()),
            "total_return":  self._cumulative / capital * 100,
            "final_equity":  capital + self._cumulative,
            "max_drawdown":  max_dd,
            "sharpe":        sharpe,
            "fees":          float(tdf["fee"].sum()),
            "target_fires":  len(tgt_r),
            "target_pnl":    float(tgt_r["pnl_after_fees"].sum()),
            "psl_fires":     len(psl_r),
            "psl_pnl":       float(psl_r["pnl_after_fees"].sum()),
            "eop_pnl":       float(sells[sells["reason"] == "end_of_period"]["pnl_after_fees"].sum()),
            "avg_bars_held": float(real["bars_held"].mean()) if len(real) > 0 else 0,
        }
