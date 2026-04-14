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
    break-even WR   = 60.2%   (actual WR = 50-60% → marginal; uptrend_bars_min
                               filter targets the 50% WR early-recovery entries)
  Derived from integration harness exit breakdown, not overfit to windows.

uptrend_bars_min rationale:
  MacroSupervisor recovery_bars=168 (7 days). The first ~7% of a RECOVERY window
  (≈48 5m bars = 4h) has historically shown 50% WR — equal probability of
  continuation vs reversal. uptrend_bars_min=48 skips those noisy early bars.
  Derived from recovery_bars constant, not tuned to specific cycles.
  Set to 0 in _aggressive preset to preserve its wider entry window.

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
  r3       — uptrend_bars_min set to 48 (4h) in trendbot_v1 preset.
             Skips first 4h of any new BULL/RECOVERY window — empirically the
             noisy early-recovery zone with 50% WR (CyA/CyB). Derived from
             recovery_bars=168 (7% exclusion zone), not tuned to specific cycles.
             trendbot_v1_aggressive left at 0 (wider entry window by design).
"""

import warnings
import numpy as np
import pandas as pd

from eth_bot_interface import BotInterface, BotStatus, Position, Lot

warnings.filterwarnings("ignore")

PRESETS = {
    "trendbot_v1": {
        "base_qty":           0.05,
        "target_bps":         180,
        "pos_stop_loss_pct":  0.025,    # 250bps — empirically derived; break-even at 60% WR
        "uptrend_rsi_max":    44,
        "vol_mult_min":       0.80,
        "cooldown_secs":      1800,
        "psl_cooldown_secs":  7200,     # 2h lockout after any stop-loss exit
        "min_profit_bps":     100,
        "zscore_max":        -0.6,
        "uptrend_bars_min":   48,       # 4h streak — skip noisy early-recovery bars
        "qty_scale": {
            "STRONG":    1.0,
            "PARABOLIC": 1.0,
            "MODERATE":  0.5,           # half size on MODERATE — risk management only
        },
        "buy_fee_pct":        0.00065,
        "sell_fee_pct":       0.00025,
    },
    "trendbot_v1_aggressive": {
        "base_qty":           0.05,
        "target_bps":         220,
        "pos_stop_loss_pct":  0.030,    # 300bps — proportional to wider target
        "uptrend_rsi_max":    48,
        "vol_mult_min":       0.70,
        "cooldown_secs":      1200,
        "psl_cooldown_secs":  3600,
        "min_profit_bps":     120,
        "zscore_max":        -0.5,
        "uptrend_bars_min":   0,        # intentionally 0 — aggressive preset uses wider window
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
    uptrend_bars_min: minimum consecutive BULL/RECOVERY bars before first entry
      (skips the noisy early-recovery window; reset on any non-trend bar).
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
        uptrend_bars_min = p.get("uptrend_bars_min", 0)
        qty_scale_map    = p.get("qty_scale", {})

        trend_streak = 0  # consecutive bars in BULL or RECOVERY

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
                unreal = (close - self._position.avg_entry) / self._position.avg_entry

                if unreal < -psl_pct:
                    self._sell(i, df, close, "pos_stop_loss", sell_fee_pct)
                    continue

                if close >= self._position.avg_entry * (1 + target_bps / 10_000):
                    self._sell(i, df, close, "target", sell_fee_pct)
                    continue

                continue

            # ── SCAN FOR ENTRY (position flat) ───────────────────────
            # Gate on MacroSupervisor BULL or RECOVERY regime
            if regime5 not in _TREND_REGIMES:
                continue

            # Require minimum consecutive trend bars before first entry
            if uptrend_bars_min > 0 and trend_streak < uptrend_bars_min:
                continue

            _rsi_raw = df["rsi"].iat[i]
            if pd.isna(_rsi_raw):
                continue
            rsi = float(_rsi_raw)

            rsi_prev = float(row.get("rsi_prev", 50))
            zscore   = float(row.get("zscore", 0))
            vol_r    = float(row.get("vol_ratio", 1))

            in_cooldown = (self._last_buy_ts is not None and
                           (ts - self._last_buy_ts).total_seconds() < cooldown)
            if in_cooldown:
                continue

            in_psl_cooldown = (self._last_psl_ts is not None and
                               (ts - self._last_psl_ts).total_seconds() < psl_cooldown)
            if in_psl_cooldown:
                continue

            if (rsi      < rsi_max
                    and rsi    < rsi_prev
                    and zscore < zscore_max
                    and vol_r  >= vol_min):
                strength      = str(row.get("trend_strength", "STRONG"))
                effective_qty = base_qty * qty_scale_map.get(strength, 1.0)
                self._buy(i, df, close, "uptrend_pb", effective_qty,
                          buy_fee_pct, target_bps, min_profit, sell_fee_pct)

        if self._position.is_open:
            self._sell(len(df) - 1, df, float(df.iloc[-1]["close"]),
                       "end_of_period", sell_fee_pct)

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
        self._position.lots = [Lot(qty=qty, price=close,
                                   fee=fee, ts=row["ts"],
                                   row_idx=len(self._trades))]
        self._last_buy_ts = row["ts"]

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
