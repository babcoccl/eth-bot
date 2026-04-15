#!/usr/bin/env python3
"""
eth_macrosupervisor_v30_backtest.py
====================================
Backtest harness for MacroSupervisor v30.

Simulates a simple regime-driven strategy:
  - ENTER at the first h1 bar of each BULL segment
  - EXIT at the first subsequent CRASH signal (or end of data)
  - Hard stop: configurable max drawdown from entry (default -15%)

Outputs
-------
  eth_backtest_v30_trades.csv   -- one row per trade
  eth_backtest_v30_summary.csv  -- aggregate stats split by BULL class
                                   (DEEP vs SHALLOW)

BULL class definitions
----------------------
  DEEP    : drawdown_at_entry_pct <= -30%   (deep trough recoveries)
  SHALLOW : drawdown_at_entry_pct >  -15%   (noise / range breaks)
  MID     : everything in between

Usage
-----
  python eth_macrosupervisor_v30_backtest.py
  python eth_macrosupervisor_v30_backtest.py --start 2021-01-01 --end 2026-04-15
  python eth_macrosupervisor_v30_backtest.py --stop-loss 0.12   # 12% hard stop
  python eth_macrosupervisor_v30_backtest.py --out ./backtest_out/
"""

from __future__ import annotations
import argparse, os, sys, tempfile
from datetime import datetime, timezone, timedelta
from typing import List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Temp-DB helper (same as eth_regime_audit.py -- avoids :memory: bug)
# ---------------------------------------------------------------------------

class _TempDB:
    def __init__(self):
        self.path = ""
    def __enter__(self) -> str:
        fd, self.path = tempfile.mkstemp(suffix=".db", prefix="bt_sup_")
        os.close(fd)
        return self.path
    def __exit__(self, *_):
        try:
            os.unlink(self.path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# BULL depth classifier
# ---------------------------------------------------------------------------

DEEP_THRESHOLD    = -0.30   # drawdown at BULL entry <= -30%  -> DEEP
SHALLOW_THRESHOLD = -0.15   # drawdown at BULL entry >  -15%  -> SHALLOW


def classify_bull_depth(drawdown_at_entry_pct: float) -> str:
    """
    Classify a BULL segment by the drawdown present at its entry bar.

    Parameters
    ----------
    drawdown_at_entry_pct : float
        Drawdown as a signed percentage, e.g. -40.6 for a 40.6% trough.

    Returns
    -------
    str : 'DEEP' | 'MID' | 'SHALLOW'
    """
    dd = drawdown_at_entry_pct / 100.0  # normalise to fraction
    if dd <= DEEP_THRESHOLD:
        return "DEEP"
    if dd > SHALLOW_THRESHOLD:
        return "SHALLOW"
    return "MID"


# ---------------------------------------------------------------------------
# Core backtest engine
# ---------------------------------------------------------------------------

FEE_RATE = 0.001   # 0.1% per side (Coinbase Advanced)


def run_backtest(
    h1_df: pd.DataFrame,
    sup,
    stop_loss: float = 0.15,
) -> pd.DataFrame:
    """
    Walk h1_df bar by bar using the committed regime5 series from `sup`.
    Returns a DataFrame of all completed trades.

    Parameters
    ----------
    h1_df      : enriched h1 DataFrame (output from sup._compute_h1_signals)
    sup        : MacroSupervisor v30 instance (already run)
    stop_loss  : maximum drawdown from entry before hard-stop exit (default 0.15)
    """
    regime_series = sup._h1_r5_series.values
    close_series  = h1_df["close"].values
    ts_series     = h1_df["ts"].values
    dd_series     = h1_df["drawdown"].values
    n             = len(regime_series)

    trades: List[dict] = []

    in_trade         = False
    entry_bar        = None
    entry_price      = None
    entry_dd_pct     = None
    peak_since_entry = None

    prev_regime = str(regime_series[0])

    for i in range(n):
        cur_regime = str(regime_series[i])
        close      = float(close_series[i])
        ts         = ts_series[i]
        dd         = float(dd_series[i])

        # ---- ENTRY: first bar of a new BULL segment ----
        if not in_trade and cur_regime == "BULL" and prev_regime != "BULL":
            in_trade         = True
            entry_bar        = i
            entry_price      = close
            entry_dd_pct     = round(dd * 100, 2)
            peak_since_entry = close

        # ---- while in trade: track peak and check exits ----
        if in_trade:
            peak_since_entry = max(peak_since_entry, close)
            drawdown_from_peak = (close - peak_since_entry) / peak_since_entry

            exit_reason = None
            exit_bar    = i
            exit_price  = close

            # Hard stop: price has fallen stop_loss% from peak since entry
            if drawdown_from_peak <= -stop_loss:
                exit_reason = f"stop_loss_{stop_loss*100:.0f}pct"

            # Signal exit: regime flipped to CRASH
            elif cur_regime == "CRASH" and prev_regime != "CRASH":
                exit_reason = "crash_signal"

            # End of data
            elif i == n - 1:
                exit_reason = "end_of_data"

            if exit_reason:
                bars_held   = exit_bar - entry_bar
                gross_ret   = (exit_price - entry_price) / entry_price
                fee_cost    = FEE_RATE * 2
                net_ret     = gross_ret - fee_cost
                bull_class  = classify_bull_depth(entry_dd_pct)

                trades.append({
                    "entry_ts":           str(ts_series[entry_bar]),
                    "exit_ts":            str(ts),
                    "entry_price":        round(entry_price, 2),
                    "exit_price":         round(exit_price, 2),
                    "bars_held":          bars_held,
                    "days_held":          round(bars_held / 24, 2),
                    "drawdown_at_entry":  entry_dd_pct,
                    "bull_class":         bull_class,
                    "gross_return_pct":   round(gross_ret * 100, 2),
                    "net_return_pct":     round(net_ret * 100, 2),
                    "win":                int(net_ret > 0),
                    "exit_reason":        exit_reason,
                    "max_dd_from_peak":   round(drawdown_from_peak * 100, 2),
                })

                in_trade         = False
                entry_bar        = None
                entry_price      = None
                entry_dd_pct     = None
                peak_since_entry = None

        prev_regime = cur_regime

    return pd.DataFrame(trades)


# ---------------------------------------------------------------------------
# Summary builder
# ---------------------------------------------------------------------------

def build_summary(trades_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate trade stats by BULL class and overall."""
    if trades_df.empty:
        return pd.DataFrame()

    records = []
    groups  = [("ALL", trades_df)] + [
        (cls, trades_df[trades_df["bull_class"] == cls])
        for cls in ["DEEP", "MID", "SHALLOW"]
        if (trades_df["bull_class"] == cls).any()
    ]

    for label, grp in groups:
        wins = grp["win"].sum()
        n    = len(grp)
        records.append({
            "bull_class":          label,
            "trades":              n,
            "wins":                int(wins),
            "win_rate_pct":        round(wins / n * 100, 1) if n > 0 else 0,
            "mean_net_ret_pct":    round(grp["net_return_pct"].mean(), 2),
            "median_net_ret_pct":  round(grp["net_return_pct"].median(), 2),
            "total_net_ret_pct":   round(grp["net_return_pct"].sum(), 2),
            "mean_days_held":      round(grp["days_held"].mean(), 2),
            "median_days_held":    round(grp["days_held"].median(), 2),
            "mean_dd_at_entry":    round(grp["drawdown_at_entry"].mean(), 2),
            "best_trade_pct":      round(grp["net_return_pct"].max(), 2),
            "worst_trade_pct":     round(grp["net_return_pct"].min(), 2),
            "stop_loss_exits":     int((grp["exit_reason"].str.startswith("stop_loss")).sum()),
            "crash_signal_exits":  int((grp["exit_reason"] == "crash_signal").sum()),
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Console report
# ---------------------------------------------------------------------------

def print_report(trades_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    SEP = "=" * 70
    print(f"\n{SEP}")
    print("  ETH MacroSupervisor v30 Backtest Report")
    print(SEP)

    for _, row in summary_df.iterrows():
        label = row["bull_class"]
        print(f"\n  [{label}]  {int(row['trades'])} trades  "
              f"win={row['win_rate_pct']:.0f}%  "
              f"mean={row['mean_net_ret_pct']:+.1f}%  "
              f"median={row['median_net_ret_pct']:+.1f}%  "
              f"total={row['total_net_ret_pct']:+.1f}%")
        print(f"         days_held med={row['median_days_held']:.1f}  "
              f"dd_at_entry mean={row['mean_dd_at_entry']:.1f}%")
        print(f"         stops={int(row['stop_loss_exits'])}  "
              f"crash_exits={int(row['crash_signal_exits'])}  "
              f"best={row['best_trade_pct']:+.1f}%  "
              f"worst={row['worst_trade_pct']:+.1f}%")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="MacroSupervisor v30 backtest harness"
    )
    ap.add_argument("--start",     default="2021-01-01")
    ap.add_argument("--end",       default=None)
    ap.add_argument("--symbol",    default="ETH/USD")
    ap.add_argument("--stop-loss", type=float, default=0.15,
                    help="Hard stop as fraction of peak since entry (default 0.15)")
    ap.add_argument("--min-dwell", type=int, default=None,
                    help="Override v30 regime5_min_dwell_bars")
    ap.add_argument("--out",       default=".",
                    help="Output directory for CSV files")
    args = ap.parse_args()

    try:
        from eth_macrosupervisor_v30 import MacroSupervisor
        from eth_helpers import fetch_ohlcv
    except ImportError as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)

    start_dt = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_s    = args.end or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    end_dt   = (
        datetime.strptime(end_s, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        + timedelta(days=1)
    )

    print(f"Fetching data {args.start} -> {end_s} ...")
    df1h = fetch_ohlcv(args.symbol, "1h", start_dt, end_dt)
    df5  = fetch_ohlcv(args.symbol, "5m", start_dt, end_dt)
    print(f"  1h bars: {len(df1h):,}   5m bars: {len(df5):,}")

    kwargs: dict = {}
    if args.min_dwell is not None:
        kwargs["regime5_min_dwell_bars"] = args.min_dwell

    with _TempDB() as db_path:
        print("Running v30 supervisor ...")
        sup = MacroSupervisor(db_path=db_path, **kwargs)
        sup.apply_to_df(df5.iloc[:1].copy(), df1h.copy())

        # Rebuild enriched h1 frame for price series access
        h_ref = df1h.copy().sort_values("ts").reset_index(drop=True)
        h_ref["fast_ema"]     = h_ref["close"].ewm(span=sup.ema_fast,  adjust=False).mean()
        h_ref["slow_ema"]     = h_ref["close"].ewm(span=sup.ema_slow,  adjust=False).mean()
        h_ref["rolling_peak"] = h_ref["close"].rolling(sup.peak_window, min_periods=1).max()
        h_ref["drawdown"]     = (
            (h_ref["close"] - h_ref["rolling_peak"]) / h_ref["rolling_peak"]
        )

        print(f"Running backtest (stop_loss={args.stop_loss*100:.0f}%) ...")
        trades_df = run_backtest(h_ref, sup, stop_loss=args.stop_loss)

    summary_df = build_summary(trades_df)

    os.makedirs(args.out, exist_ok=True)
    p_trades  = os.path.join(args.out, "eth_backtest_v30_trades.csv")
    p_summary = os.path.join(args.out, "eth_backtest_v30_summary.csv")
    trades_df.to_csv(p_trades,   index=False)
    summary_df.to_csv(p_summary, index=False)

    print_report(trades_df, summary_df)

    out_abs = os.path.abspath(args.out)
    print(f"  Outputs written to: {out_abs}/")
    print(f"    eth_backtest_v30_trades.csv   ({len(trades_df)} trades)")
    print(f"    eth_backtest_v30_summary.csv  ({len(summary_df)} rows)")
    print()


if __name__ == "__main__":
    main()
