#!/usr/bin/env python3
"""
eth_macrosupervisor_v30_backtest.py
====================================
Backtest harness for MacroSupervisor v30.

Strategy
--------
  ENTER : first h1 bar of each BULL segment
  EXIT  : first CRASH signal OR hard stop from entry-peak (default -15%)

BULL depth class is determined by CYCLE TROUGH -- the actual percentage
drop from the last pre-crash peak (last BULL/RANGE bar before this BULL
entry) to the minimum close between that bar and this entry bar.

This correctly handles:
  - Long multi-month crashes with embedded RANGE/CORRECTION sub-segments
  - RANGE->BULL direct transitions (no crash between them)
  - First-bar-of-history edge cases

BULL class definitions
----------------------
  DEEP    : cycle_trough_pct <= -30%
  MID     : cycle_trough_pct in (-30%, -15%]
  SHALLOW : cycle_trough_pct >  -15%  (or no preceding crash found)

Outputs
-------
  eth_backtest_v30_trades.csv   -- one row per trade
  eth_backtest_v30_summary.csv  -- aggregate stats by BULL class

Usage
-----
  python eth_macrosupervisor_v30_backtest.py
  python eth_macrosupervisor_v30_backtest.py --start 2021-01-01 --end 2026-04-15
  python eth_macrosupervisor_v30_backtest.py --stop-loss 0.12
  python eth_macrosupervisor_v30_backtest.py --out ./backtest_out/
"""

from __future__ import annotations
import argparse, os, sys, tempfile
from datetime import datetime, timezone, timedelta
from typing import List

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Temp-DB helper
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
# BULL depth classification
# ---------------------------------------------------------------------------

DEEP_THRESHOLD    = -0.30   # cycle trough <= -30% -> DEEP
SHALLOW_THRESHOLD = -0.15   # cycle trough >  -15% -> SHALLOW

PAUSE_REGIMES     = {"CRASH", "CORRECTION", "RECOVERY"}
PEAK_REGIMES      = {"BULL", "RANGE"}       # regimes that represent price strength


def classify_bull_depth(cycle_trough_pct: float) -> str:
    dd = cycle_trough_pct / 100.0
    if dd <= DEEP_THRESHOLD:
        return "DEEP"
    if dd > SHALLOW_THRESHOLD:
        return "SHALLOW"
    return "MID"


def _cycle_trough_pct(regime_arr, close_arr, entry_idx: int) -> float:
    """
    Compute the true cycle trough preceding a BULL entry.

    Algorithm
    ---------
    1. Walk backwards from entry_idx-1 to find the most recent bar
       where the regime was BULL or RANGE (a "peak regime" bar).
       This is the reference point regardless of what came between
       (handles embedded RANGE bars within long crashes).

    2. ref_price = close at that bar.

    3. trough = (min(close[ref_bar+1 : entry_idx]) / ref_price) - 1

    Returns 0.0 if no prior BULL/RANGE bar exists (start of history).
    """
    # Step 1: find the most recent prior BULL or RANGE bar
    ref_bar = -1
    for j in range(entry_idx - 1, -1, -1):
        if str(regime_arr[j]) in PEAK_REGIMES:
            ref_bar = j
            break

    if ref_bar == -1:
        # No prior peak regime found -- start of history, treat as SHALLOW
        return 0.0

    ref_price = float(close_arr[ref_bar])
    if ref_price <= 0:
        return 0.0

    # Step 2: min close between ref_bar (exclusive) and entry_idx (exclusive)
    window = close_arr[ref_bar + 1 : entry_idx]
    if len(window) == 0:
        # Immediate BULL->BULL (shouldn't happen with dwell, but guard it)
        return 0.0

    min_close = float(np.min(window))
    trough    = (min_close / ref_price) - 1.0
    return round(trough * 100, 2)


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
    Walk h1_df bar by bar.
    Entry  : first bar of each new BULL segment.
    Exit   : first CRASH signal, hard stop from peak, or end of data.
    """
    regime_arr = sup._h1_r5_series.values
    close_arr  = h1_df["close"].values
    ts_arr     = h1_df["ts"].values
    n          = len(regime_arr)

    trades: List[dict] = []

    in_trade         = False
    entry_bar        = None
    entry_price      = None
    cycle_trough     = None
    peak_since_entry = None

    prev_regime = str(regime_arr[0])

    for i in range(n):
        cur_regime = str(regime_arr[i])
        close      = float(close_arr[i])
        ts         = ts_arr[i]

        # ---- ENTRY: first bar of a new BULL segment ----
        if not in_trade and cur_regime == "BULL" and prev_regime != "BULL":
            in_trade         = True
            entry_bar        = i
            entry_price      = close
            cycle_trough     = _cycle_trough_pct(regime_arr, close_arr, i)
            peak_since_entry = close

        # ---- while in trade: check exits ----
        if in_trade:
            peak_since_entry = max(peak_since_entry, close)
            dd_from_peak     = (close - peak_since_entry) / peak_since_entry

            exit_reason = None

            if dd_from_peak <= -stop_loss:
                exit_reason = f"stop_loss_{stop_loss*100:.0f}pct"
            elif cur_regime == "CRASH" and prev_regime != "CRASH":
                exit_reason = "crash_signal"
            elif i == n - 1:
                exit_reason = "end_of_data"

            if exit_reason:
                bars_held  = i - entry_bar
                gross_ret  = (close - entry_price) / entry_price
                net_ret    = gross_ret - FEE_RATE * 2
                bull_class = classify_bull_depth(cycle_trough)

                trades.append({
                    "entry_ts":             str(ts_arr[entry_bar]),
                    "exit_ts":              str(ts),
                    "entry_price":          round(entry_price, 2),
                    "exit_price":           round(close, 2),
                    "bars_held":            bars_held,
                    "days_held":            round(bars_held / 24, 2),
                    "cycle_trough_pct":     cycle_trough,
                    "bull_class":           bull_class,
                    "gross_return_pct":     round(gross_ret * 100, 2),
                    "net_return_pct":       round(net_ret * 100, 2),
                    "win":                  int(net_ret > 0),
                    "exit_reason":          exit_reason,
                    "max_dd_from_peak_pct": round(dd_from_peak * 100, 2),
                })

                in_trade         = False
                entry_bar        = None
                entry_price      = None
                cycle_trough     = None
                peak_since_entry = None

        prev_regime = cur_regime

    return pd.DataFrame(trades)


# ---------------------------------------------------------------------------
# Summary builder
# ---------------------------------------------------------------------------

def build_summary(trades_df: pd.DataFrame) -> pd.DataFrame:
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
            "bull_class":            label,
            "trades":                n,
            "wins":                  int(wins),
            "win_rate_pct":          round(wins / n * 100, 1) if n > 0 else 0,
            "mean_net_ret_pct":      round(grp["net_return_pct"].mean(), 2),
            "median_net_ret_pct":    round(grp["net_return_pct"].median(), 2),
            "total_net_ret_pct":     round(grp["net_return_pct"].sum(), 2),
            "mean_days_held":        round(grp["days_held"].mean(), 2),
            "median_days_held":      round(grp["days_held"].median(), 2),
            "mean_cycle_trough_pct": round(grp["cycle_trough_pct"].mean(), 2),
            "best_trade_pct":        round(grp["net_return_pct"].max(), 2),
            "worst_trade_pct":       round(grp["net_return_pct"].min(), 2),
            "stop_loss_exits":       int((grp["exit_reason"].str.startswith("stop_loss")).sum()),
            "crash_signal_exits":    int((grp["exit_reason"] == "crash_signal").sum()),
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
    print(f"  {'bull_class':<10} {'trades':>6} {'win%':>6} "
          f"{'mean%':>7} {'med%':>7} {'total%':>8} "
          f"{'med_days':>9} {'trough%':>8}")
    print("  " + "-" * 66)
    for _, row in summary_df.iterrows():
        print(f"  {row['bull_class']:<10} "
              f"{int(row['trades']):>6} "
              f"{row['win_rate_pct']:>5.0f}% "
              f"{row['mean_net_ret_pct']:>+7.1f}% "
              f"{row['median_net_ret_pct']:>+7.1f}% "
              f"{row['total_net_ret_pct']:>+8.1f}% "
              f"{row['median_days_held']:>9.1f} "
              f"{row['mean_cycle_trough_pct']:>+8.1f}%")
    r0 = summary_df.iloc[0]
    print(f"\n  stops={int(r0.stop_loss_exits)}  "
          f"crash_exits={int(r0.crash_signal_exits)}  "
          f"best={r0.best_trade_pct:+.1f}%  "
          f"worst={r0.worst_trade_pct:+.1f}%")
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
    ap.add_argument("--stop-loss", type=float, default=0.15)
    ap.add_argument("--min-dwell", type=int,   default=None)
    ap.add_argument("--out",       default=".")
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

    kwargs = {}
    if args.min_dwell is not None:
        kwargs["regime5_min_dwell_bars"] = args.min_dwell

    with _TempDB() as db_path:
        print("Running v30 supervisor ...")
        sup = MacroSupervisor(db_path=db_path, **kwargs)
        sup.apply_to_df(df5.iloc[:1].copy(), df1h.copy())

        h_ref = df1h.copy().sort_values("ts").reset_index(drop=True)

        print(f"Running backtest (stop_loss={args.stop_loss*100:.0f}%) ...")
        trades_df = run_backtest(h_ref, sup, stop_loss=args.stop_loss)

    if not trades_df.empty:
        print(f"  Trough distribution:")
        print(f"    DEEP    (<=-30%): {(trades_df.cycle_trough_pct <= -30).sum()}")
        print(f"    MID  (-30%,-15%]: {((trades_df.cycle_trough_pct > -30) & (trades_df.cycle_trough_pct <= -15)).sum()}")
        print(f"    SHALLOW  (>-15%): {(trades_df.cycle_trough_pct > -15).sum()}")
        print(f"    trough range: {trades_df.cycle_trough_pct.min():.1f}% to {trades_df.cycle_trough_pct.max():.1f}%")
        print()

    summary_df = build_summary(trades_df)

    os.makedirs(args.out, exist_ok=True)
    p_trades  = os.path.join(args.out, "eth_backtest_v30_trades.csv")
    p_summary = os.path.join(args.out, "eth_backtest_v30_summary.csv")
    trades_df.to_csv(p_trades,   index=False)
    summary_df.to_csv(p_summary, index=False)

    print_report(trades_df, summary_df)
    print(f"  Outputs: {os.path.abspath(args.out)}/")
    print(f"    eth_backtest_v30_trades.csv   ({len(trades_df)} trades)")
    print(f"    eth_backtest_v30_summary.csv  ({len(summary_df)} rows)")
    print()


if __name__ == "__main__":
    main()
