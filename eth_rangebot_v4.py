#!/usr/bin/env python3
"""
eth_rangebot_v4.py  —  RangeBot  (RANGE / RECOVERY specialist)
===============================================================
Role    : Trades mean-reversion signals during RANGE and RECOVERY regimes.
Signals : range_dip  — BB lower bounce (price at lower band, RSI oversold)
          lv_scalp   — low-volatility scalp (tight BB, low ATR, oversold RSI)
Exit    : target_bps
Safety  : pos_stop_loss (5%) — tight, fail fast
DCA     : NONE
Size    : fixed 0.05 ETH per trade

Design principle: do ONE thing well.
  This bot only fires when price touches the lower Bollinger Band
  or when volatility collapses into a low-vol squeeze.
  It does not trade trends, it does not average down.

Test windows (from regime_period_analyzer.py):
  PRIMARY  : 2026-01-07 → 2026-01-20  (RANGE, 13.4d, rng=12.1%)
  SECONDARY: 2026-03-12 → 2026-03-19  (RECOVERY, 7d, rng=15.1%)
  MARGINAL : 2025-04-25 → 2025-05-01  (RECOVERY, 7d, rng=6.7%)
"""

import warnings
import numpy as np
import pandas as pd

from eth_bot_interface import BotInterface, BotStatus, Position, Lot

warnings.filterwarnings("ignore")

PRESETS = {
    "rangebot_v4": {
        "base_qty":           0.05,
        "target_bps":        100,       # 1.0%  — v3: revert PSL to 5%, increase target
        "pos_stop_loss_pct":  0.05,     # 5% tight PSL
        "range_rsi_max":      42,       # RSI must be below 42 (oversold)
        "lv_scalp_rsi_max":   38,       # even more oversold for lv_scalp
        "lv_scalp_bw_max":    0.05,     # BB width below 5% (squeeze)
        "lv_scalp_atr_max":   0.004,    # ATR% below 0.4% (low vol)
        "vol_mult_min":       0.85,     # volume confirmation
        "cooldown_secs":      300,      # 5 min between entries
        "min_profit_bps":     70,
        "fee_pct":            0.00065,
        "max_hold_bars":      288,       # 24h time-stop: range scalp resolves in 1 day
    },
    "rangebot_v4_tight": {
        "base_qty":           0.05,
        "target_bps":         80,       # v3 tight
        "pos_stop_loss_pct":  0.04,
        "range_rsi_max":      38,
        "lv_scalp_rsi_max":   33,
        "lv_scalp_bw_max":    0.04,
        "lv_scalp_atr_max":   0.003,
        "vol_mult_min":       0.90,
        "cooldown_secs":      180,
        "min_profit_bps":     15,
        "fee_pct":            0.00065,
        "max_hold_bars":      192,       # 16h for tight preset
    },
}


class RangeBot(BotInterface):
    """
    RANGE/RECOVERY regime specialist — range_dip and lv_scalp signals only.

    State machine (binary):
      IDLE → scanning for BB lower touch or low-vol squeeze entry
      OPEN → holding, waiting for target or PSL
      No DCA. No trail. No momentum exit.
    """

    def __init__(self, symbol: str = "ETH-USD"):
        self._symbol   = symbol
        self._active   = True
        self._position = Position(symbol=symbol)
        self._cash     = 0.0
        self._capital  = 0.0
        self._realized_pnl = 0.0
        self._trade_count  = 0
        self._last_buy_ts  = None

    @property
    def bot_id(self) -> str:
        return f"rangebot_{self._symbol.lower().replace('-', '_')}"

    @property
    def supported_regimes(self) -> list:
        return ["RANGE", "RECOVERY"]

    def get_status(self) -> BotStatus:
        return BotStatus(
            bot_id            = self.bot_id,
            symbol            = self._symbol,
            capital_allocated = self._capital,
            capital_deployed  = self._position.cost_basis,
            capital_available = self._cash,
            open_qty          = self._position.qty,
            open_avg_entry    = self._position.avg_entry,
            unrealized_pnl    = 0.0,
            realized_pnl      = self._realized_pnl,
            trade_count       = self._trade_count,
            active            = self._active,
            supported_regimes = self.supported_regimes,
        )

    def run_backtest(self, df: pd.DataFrame, preset: dict,
                     capital: float, preset_name: str) -> tuple:
        """Run RangeBot strategy over a single approved RANGE/RECOVERY window."""
        self._reset(capital)
        p = preset

        fee_pct      = p["fee_pct"]
        base_qty     = p["base_qty"]
        target_bps   = p["target_bps"]
        psl_pct      = p["pos_stop_loss_pct"]
        rsi_max      = p["range_rsi_max"]
        lv_rsi_max   = p["lv_scalp_rsi_max"]
        lv_bw_max    = p["lv_scalp_bw_max"]
        lv_atr_max   = p["lv_scalp_atr_max"]
        vol_min      = p["vol_mult_min"]
        cooldown     = p["cooldown_secs"]
        min_profit   = p.get("min_profit_bps", 0)

        for i in range(len(df)):
            row    = df.iloc[i]
            close  = float(row["close"])
            ts     = row["ts"]
            regime = str(row.get("regime_h1", "RANGE"))

            self._equity_curve.append(self._cash + self._position.qty * close)

            # ── MANAGE OPEN POSITION ──────────────────────────────────────
            if self._position.is_open:
                unreal    = (close - self._position.avg_entry) / self._position.avg_entry
                bars_held = i - self._position.entry_bar

                if unreal < -psl_pct:
                    self._sell(i, df, close, "pos_stop_loss", fee_pct)
                    continue

                if close >= self._position.avg_entry * (1 + target_bps / 10_000):
                    self._sell(i, df, close, "target", fee_pct)
                    continue

                max_hold = p.get("max_hold_bars", 9999)
                if bars_held >= max_hold:
                    self._sell(i, df, close, "time_stop", fee_pct)
                    continue

                continue

            # ── SCAN FOR ENTRY (position flat) ────────────────────────────
            # RangeBot trades in RANGE or DOWNTREND at 1h level
            # (DOWNTREND within a RANGE macro period = oversold condition)
            if regime not in ("RANGE", "DOWNTREND"):
                continue

            _rsi_raw = df["rsi"].iat[i]
            if pd.isna(_rsi_raw):
                continue
            rsi = float(_rsi_raw)

            bb_lo   = float(row.get("bb_lower",   close))
            bw      = float(row.get("bw_pct",     0.05))
            atr_pct = float(row.get("atr_pct",    0))
            vol_r   = float(row.get("vol_ratio",   1))

            in_cooldown = (self._last_buy_ts is not None and
                           (ts - self._last_buy_ts).total_seconds() < cooldown)
            if in_cooldown:
                continue

            # Signal 1 — range_dip: BB lower band touch + oversold RSI
            if (rsi < rsi_max
                    and close < bb_lo * 1.002   # touching or below lower band
                    and vol_r >= vol_min):
                self._buy(i, df, close, "range_dip", base_qty,
                          fee_pct, target_bps, min_profit)
                continue

            # Signal 2 — lv_scalp: low-vol squeeze + deeply oversold
            if (rsi < lv_rsi_max
                    and bw < lv_bw_max          # BB squeeze (compressed range)
                    and atr_pct < lv_atr_max    # low absolute volatility
                    and vol_r >= vol_min * 0.80):
                self._buy(i, df, close, "lv_scalp", base_qty,
                          fee_pct, target_bps, min_profit)
                continue

        if self._position.is_open:
            self._sell(len(df)-1, df, float(df.iloc[-1]["close"]),
                       "end_of_period", fee_pct)

        return self._build_result(capital, preset_name)

    # ── Private methods (identical structure to TrendBot) ─────────────────────

    def _reset(self, capital: float) -> None:
        self._cash         = float(capital)
        self._capital      = float(capital)
        self._position     = Position(symbol=self._symbol)
        self._trades       = []
        self._equity_curve = []
        self._cumulative   = 0.0
        self._last_buy_ts  = None

    def _buy(self, i, df, close, reason, base_qty, fee_pct, target_bps, min_profit):
        row = df.iloc[i]
        bv  = base_qty * close
        if bv > self._cash:
            return
        expected = bv * (target_bps / 10_000)
        if expected < bv * fee_pct * 2 * (1 + min_profit / 10_000):
            return
        fee             = bv * fee_pct
        self._cash     -= bv + fee
        self._position.qty        = base_qty
        self._position.avg_entry  = close
        self._position.peak_price = close
        self._position.entry_bar  = i
        self._position.lots       = [Lot(qty=base_qty, price=close,
                                         fee=fee, ts=row["ts"], row_idx=len(self._trades))]
        self._last_buy_ts = row["ts"]
        self._trades.append({
            "ts": row["ts"], "side": "BUY", "reason": reason,
            "regime_h1": str(row.get("regime_h1", "")),
            "price": close, "qty": base_qty, "fee": fee,
            "rsi": float(df["rsi"].iat[i]) if not pd.isna(df["rsi"].iat[i]) else float("nan"),
            "bb_lower": float(row.get("bb_lower", float("nan"))),
            "bw_pct":   float(row.get("bw_pct",   float("nan"))),
            "pnl": 0.0, "pnl_after_fees": 0.0, "win": float("nan"),
            "bars_held": float("nan"), "exit_price": float("nan"),
        })

    def _sell(self, i, df, close, reason, fee_pct):
        p        = self._position
        row      = df.iloc[i]
        sell_val = p.qty * close
        sell_fee = sell_val * fee_pct
        pnl      = sell_val - sell_fee - p.cost_basis
        self._cumulative += pnl
        bh = i - p.entry_bar
        buy_row = self._trades[p.lots[0].row_idx]
        buy_row.update({
            "pnl": pnl, "pnl_after_fees": pnl,
            "win": 1.0 if pnl > 0 else 0.0,
            "bars_held": bh, "exit_price": close,
        })
        self._trades.append({
            "ts": row["ts"], "side": "SELL", "reason": reason,
            "regime_h1": str(row.get("regime_h1", "")),
            "price": close, "qty": p.qty, "fee": sell_fee,
            "rsi": float(df["rsi"].iat[i]) if not pd.isna(df["rsi"].iat[i]) else float("nan"),
            "bb_lower": float("nan"), "bw_pct": float("nan"),
            "pnl": pnl, "pnl_after_fees": pnl,
            "win": 1.0 if pnl > 0 else 0.0,
            "bars_held": bh, "exit_price": float("nan"),
        })
        self._cash += sell_val - sell_fee
        if reason != "end_of_period":
            self._realized_pnl += pnl
            self._trade_count  += 1
        p.reset()

    def _build_result(self, capital: float, preset_name: str) -> tuple:
        if not self._trades:
            return pd.DataFrame(), {}
        tdf     = pd.DataFrame(self._trades)
        eq      = np.array(self._equity_curve)
        peak_eq = np.maximum.accumulate(eq)
        max_dd  = float(((eq - peak_eq) / peak_eq).min()) * 100
        ret_s   = np.diff(eq) / eq[:-1]
        sharpe  = (ret_s.mean() / ret_s.std() * np.sqrt(105_120)
                   if ret_s.std() > 0 else 0.0)
        sells  = tdf[tdf["side"] == "SELL"]
        real   = sells[~sells["reason"].isin(["end_of_period"])]
        wins   = real[real["pnl_after_fees"] > 0]
        psl_r  = real[real["reason"] == "pos_stop_loss"]
        tgt_r  = real[real["reason"] == "target"]
        tst_r  = real[real["reason"] == "time_stop"]
        rd_r   = real[real["reason"] == "range_dip"]
        lv_r   = real[real["reason"] == "lv_scalp"]
        return tdf, {
            "preset":          preset_name,
            "trades":          len(real),
            "win_rate":        len(wins) / len(real) * 100 if len(real) > 0 else 0,
            "realized_pnl":    tdf["pnl_after_fees"].sum(),
            "total_return":    self._cumulative / capital * 100,
            "final_equity":    capital + self._cumulative,
            "max_drawdown":    max_dd,
            "sharpe":          sharpe,
            "fees":            tdf["fee"].sum(),
            "target_fires":    len(tgt_r),
            "target_pnl":      float(tgt_r["pnl_after_fees"].sum()),
            "psl_fires":       len(psl_r),
            "psl_pnl":         float(psl_r["pnl_after_fees"].sum()),
            "time_stop_fires": len(tst_r),
            "time_stop_pnl":   float(tst_r["pnl_after_fees"].sum()),
            "time_stop_pct":   len(tst_r) / len(real) * 100 if len(real) > 0 else 0,
            "range_dip_fires": len(rd_r),
            "lv_scalp_fires":  len(lv_r),
            "eop_pnl":         float(sells[sells["reason"]=="end_of_period"]["pnl_after_fees"].sum()),
            "avg_bars_held":   float(real["bars_held"].mean()) if len(real) > 0 else 0,
        }
