#!/usr/bin/env python3
"""
eth_trendbot_v1.py — TrendBot (BULL / RECOVERY specialist)
==============================================================
Role : Trades uptrend pullbacks during BULL and RECOVERY regimes only.
Signal : uptrend_pb — RSI pullback in h1 UPTREND
Exit : target_bps fixed profit target
Safety : pos_stop_loss_pct (5%) — tight, fail fast
DCA : NONE — one fill per position, orchestrator adds capital
Size : fixed base_qty ETH per trade

Design principle: do ONE thing well.
This bot only fires when the 1h trend is UP and price pulls back.
It does not trade ranges, it does not average down.
The MacroSupervisor enables it in BULL/RECOVERY regimes.
The LLM Orchestrator decides capital allocation.

Test windows (from regime_period_analyzer.py):
PRIMARY  : 2025-07-06 → 2025-07-31 (RECOVERY→BULL, 25d, +28.2%)
SECONDARY: 2025-05-08 → 2025-05-16 (BULL, 8.6d, +36.1%)
HOLDOUT  : 2025-12-30 → 2026-01-06 (RECOVERY, 7d, +7.9%)
"""

import warnings
import numpy as np
import pandas as pd

from eth_bot_interface import BotInterface, BotStatus, Position, Lot

warnings.filterwarnings("ignore")

PRESETS = {
    "trendbot_v1": {
        "base_qty":           0.05,
        "target_bps":         35,       # 0.35% target — small but frequent
        "pos_stop_loss_pct":  0.05,     # 5% tight PSL
        "uptrend_rsi_max":    44,       # RSI must be below 44 (pullback)
        "vol_mult_min":       0.80,     # volume confirmation
        "cooldown_secs":      600,      # 10 min between entries
        "min_profit_bps":     25,       # min expected profit after fees
        "fee_pct":            0.00065,
    },
    "trendbot_v1_aggressive": {
        "base_qty":           0.05,
        "target_bps":         50,
        "pos_stop_loss_pct":  0.06,
        "uptrend_rsi_max":    48,       # slightly looser RSI
        "vol_mult_min":       0.70,
        "cooldown_secs":      300,
        "min_profit_bps":     30,
        "fee_pct":            0.00065,
    },
}


class TrendBot(BotInterface):
    """
    BULL/RECOVERY regime specialist — uptrend_pb signal only.

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

        fee_pct    = p["fee_pct"]
        base_qty   = p["base_qty"]
        target_bps = p["target_bps"]
        psl_pct    = p["pos_stop_loss_pct"]
        rsi_max    = p["uptrend_rsi_max"]
        vol_min    = p["vol_mult_min"]
        cooldown   = p["cooldown_secs"]
        min_profit = p.get("min_profit_bps", 0)

        for i in range(len(df)):
            row    = df.iloc[i]
            close  = float(row["close"])
            ts     = row["ts"]
            regime = str(row.get("regime_h1", "RANGE"))

            self._equity_curve.append(self._cash + self._position.qty * close)

            # ── MANAGE OPEN POSITION ──────────────────────────────────
            if self._position.is_open:
                unreal = (close - self._position.avg_entry) / self._position.avg_entry

                if unreal < -psl_pct:
                    self._sell(i, df, close, "pos_stop_loss", fee_pct)
                    continue

                if close >= self._position.avg_entry * (1 + target_bps / 10_000):
                    self._sell(i, df, close, "target", fee_pct)
                    continue

                continue  # position open, no exit triggered — hold

            # ── SCAN FOR ENTRY (position flat) ────────────────────────
            if regime != "UPTREND":
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

            # uptrend_pb: RSI pullback with volume in h1 UPTREND
            if (rsi      < rsi_max
                    and rsi  < rsi_prev       # RSI falling — active pullback
                    and zscore < -0.3         # price below short-term mean
                    and vol_r  >= vol_min):
                self._buy(i, df, close, "uptrend_pb", base_qty,
                          fee_pct, target_bps, min_profit)

        # Close any open position at period end
        if self._position.is_open:
            self._sell(len(df) - 1, df, float(df.iloc[-1]["close"]),
                       "end_of_period", fee_pct)

        return self._build_result(capital, preset_name)

    # ── Private methods ──────────────────────────────────────────────────

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

    def _buy(self, i, df, close, reason, base_qty, fee_pct, target_bps, min_profit):
        row = df.iloc[i]
        bv  = base_qty * close
        if bv > self._cash:
            return
        expected = bv * (target_bps / 10_000)
        if expected < bv * fee_pct * 2 * (1 + min_profit / 10_000):
            return

        fee = bv * fee_pct
        self._cash             -= bv + fee
        self._position.qty      = base_qty
        self._position.avg_entry = close
        self._position.peak_price = close
        self._position.entry_bar  = i
        self._position.lots = [Lot(qty=base_qty, price=close,
                                   fee=fee, ts=row["ts"],
                                   row_idx=len(self._trades))]
        self._last_buy_ts = row["ts"]

        self._trades.append({
            "ts":        row["ts"],
            "side":      "BUY",
            "reason":    reason,
            "regime_h1": str(row.get("regime_h1", "")),
            "price":     close,
            "qty":       base_qty,
            "fee":       fee,
            "rsi":       float(df["rsi"].iat[i]) if not pd.isna(df["rsi"].iat[i]) else float("nan"),
            "zscore":    float(row.get("zscore",    float("nan"))),
            "vol_ratio": float(row.get("vol_ratio", float("nan"))),
            "pnl": 0.0, "pnl_after_fees": 0.0,
            "win": float("nan"), "bars_held": float("nan"),
            "exit_price": float("nan"),
        })

    def _sell(self, i, df, close, reason, fee_pct):
        p        = self._position
        row      = df.iloc[i]
        sell_val = p.qty * close
        sell_fee = sell_val * fee_pct
        pnl      = sell_val - sell_fee - p.cost_basis
        self._cumulative += pnl
        bh = i - p.entry_bar

        # Back-fill the BUY row
        buy_row = self._trades[p.lots[0].row_idx]
        buy_row.update({
            "pnl": pnl, "pnl_after_fees": pnl,
            "win": 1.0 if pnl > 0 else 0.0,
            "bars_held": bh, "exit_price": close,
        })

        self._trades.append({
            "ts":        row["ts"],
            "side":      "SELL",
            "reason":    reason,
            "regime_h1": str(row.get("regime_h1", "")),
            "price":     close,
            "qty":       p.qty,
            "fee":       sell_fee,
            "rsi":       float(df["rsi"].iat[i]) if not pd.isna(df["rsi"].iat[i]) else float("nan"),
            "zscore":    float("nan"),
            "vol_ratio": float("nan"),
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

        sells   = tdf[tdf["side"] == "SELL"]
        real    = sells[~sells["reason"].isin(["end_of_period"])]
        wins    = real[real["pnl_after_fees"] > 0]
        psl_r   = real[real["reason"] == "pos_stop_loss"]
        tgt_r   = real[real["reason"] == "target"]

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
