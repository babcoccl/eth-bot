#!/usr/bin/env python3
"""
export_annotated_history.py
============================
Generates a 6-year (2020-01-01 to present) regime-annotated 5m OHLCV CSV
for ETH/USD using MacroSupervisor v29.

Output: eth_annotated_history.csv

Columns:
  ts                  — UTC timestamp (5m bar open)
  open/high/low/close — OHLCV price
  volume
  regime5             — MacroSupervisor regime label (BULL/RECOVERY/RANGE/CORRECTION/CRASH)
  drawdown_pct        — rolling drawdown from local peak (negative float, e.g. -0.22)
  plus any additional indicator columns applied by prepare_indicators()

Usage:
  python export_annotated_history.py
  python export_annotated_history.py --start 2019-01-01 --end 2026-04-01
  python export_annotated_history.py --symbol ETH/USD --out my_output.csv

Requires:
  eth_helpers.py
  eth_macrosupervisor_v29.py
  (standard deps: pandas, numpy — same as harness)
"""

import argparse
import os
import sys
import tempfile
import warnings
from datetime import datetime, timezone

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

from eth_helpers             import fetch_ohlcv, prepare_indicators
from eth_macrosupervisor_v29 import MacroSupervisor


DEFAULT_START  = "2020-01-01"
DEFAULT_END    = datetime.now(timezone.utc).strftime("%Y-%m-%d")
DEFAULT_SYMBOL = "ETH/USD"
DEFAULT_OUT    = "eth_annotated_history.csv"

# Lookback warmup so indicators are valid from DEFAULT_START
WARMUP_DAYS = 60


def _parse_dt(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def main():
    ap = argparse.ArgumentParser(
        description="Export 6yr regime-annotated ETH/USD history to CSV."
    )
    ap.add_argument("--symbol", default=DEFAULT_SYMBOL,
                    help=f"Trading pair (default: {DEFAULT_SYMBOL})")
    ap.add_argument("--start",  default=DEFAULT_START,
                    help=f"Start date YYYY-MM-DD (default: {DEFAULT_START})")
    ap.add_argument("--end",    default=DEFAULT_END,
                    help=f"End date YYYY-MM-DD (default: today)")
    ap.add_argument("--out",    default=DEFAULT_OUT,
                    help=f"Output CSV filename (default: {DEFAULT_OUT})")
    ap.add_argument("--no-cache", action="store_true",
                    help="Clear OHLCV cache before fetching")
    args = ap.parse_args()

    if args.no_cache:
        from eth_helpers import clear_ohlcv_cache
        clear_ohlcv_cache()
        print("[cache cleared]")

    start_dt  = _parse_dt(args.start)
    end_dt    = _parse_dt(args.end)
    warm_dt   = start_dt - pd.Timedelta(days=WARMUP_DAYS)

    print(f"Exporting annotated history")
    print(f"  Symbol  : {args.symbol}")
    print(f"  Range   : {args.start} → {args.end}")
    print(f"  Warmup  : {WARMUP_DAYS}d before start (indicators)")
    print(f"  Output  : {args.out}")
    print()

    # ── Fetch 5m and 1h data ────────────────────────────────────────────────
    print("Fetching 5m OHLCV...")
    df5 = fetch_ohlcv(args.symbol, "5m", warm_dt, end_dt)
    if df5 is None or len(df5) < 100:
        print("ERROR: insufficient 5m data returned. Check symbol and date range.")
        sys.exit(1)
    print(f"  {len(df5):,} bars fetched")

    print("Fetching 1h OHLCV...")
    df1h = fetch_ohlcv(args.symbol, "1h", warm_dt, end_dt)
    if df1h is None or len(df1h) < 10:
        print("ERROR: insufficient 1h data returned.")
        sys.exit(1)
    print(f"  {len(df1h):,} bars fetched")

    # ── Apply indicators and supervisor ────────────────────────────────────
    print("Applying indicators...")
    df_ind = prepare_indicators(df5, df1h)

    print("Running MacroSupervisor v29...")
    fd, tmp_db = tempfile.mkstemp(suffix=".db", prefix="export_history_")
    os.close(fd)
    try:
        sup    = MacroSupervisor(db_path=tmp_db)
        df_ann = sup.apply_to_df(df_ind, df1h)
    finally:
        try:
            os.unlink(tmp_db)
        except OSError:
            pass

    # ── Trim warmup period ─────────────────────────────────────────────────
    start_ts = pd.Timestamp(start_dt)
    df_out   = df_ann[df_ann["ts"] >= start_ts].reset_index(drop=True)
    print(f"  {len(df_out):,} bars after trimming warmup")

    # ── Compute rolling drawdown_pct if not already present ───────────────
    if "drawdown_pct" not in df_out.columns:
        rolling_peak      = df_out["close"].cummax()
        df_out["drawdown_pct"] = (df_out["close"] - rolling_peak) / rolling_peak

    # ── Column ordering: core columns first, then extras ──────────────────
    core_cols = ["ts", "open", "high", "low", "close", "volume",
                 "regime5", "drawdown_pct"]
    extra_cols = [c for c in df_out.columns if c not in core_cols]
    ordered    = [c for c in core_cols if c in df_out.columns] + extra_cols
    df_out     = df_out[ordered]

    # ── Write CSV ──────────────────────────────────────────────────────────
    df_out.to_csv(args.out, index=False)
    size_mb = os.path.getsize(args.out) / (1024 * 1024)
    print(f"\nDone. Wrote {len(df_out):,} rows → {args.out}  ({size_mb:.1f} MB)")

    # ── Quick regime summary ───────────────────────────────────────────────
    if "regime5" in df_out.columns:
        print("\nRegime distribution (full export range):")
        total = len(df_out)
        for r5 in ["BULL", "RECOVERY", "RANGE", "CORRECTION", "CRASH"]:
            n   = int((df_out["regime5"] == r5).sum())
            pct = n / total * 100
            print(f"  {r5:<12} {n:>8,} bars  ({pct:>5.1f}%)")
        print(f"  {'TOTAL':<12} {total:>8,} bars")

    # ── Correction trough index (helper for window analysis) ───────────────
    print("\nDeep drawdown periods (dd < -10% for >= 3 consecutive days):")
    THRESH      = -0.10
    MIN_DAYS    = 3
    MIN_BARS    = MIN_DAYS * 288   # 288 5m bars per day
    in_dd       = df_out["drawdown_pct"] < THRESH
    cur_start   = None
    cur_len     = 0
    dd_periods  = []
    for i, val in enumerate(in_dd):
        if val:
            if cur_start is None:
                cur_start = i
            cur_len += 1
        else:
            if cur_start is not None and cur_len >= MIN_BARS:
                dd_periods.append((cur_start, i - 1, cur_len))
            cur_start = None
            cur_len   = 0
    if cur_start is not None and cur_len >= MIN_BARS:
        dd_periods.append((cur_start, len(df_out) - 1, cur_len))

    for s_idx, e_idx, n_bars in dd_periods:
        s_ts  = df_out.loc[s_idx, "ts"]
        e_ts  = df_out.loc[e_idx, "ts"]
        min_dd = df_out.loc[s_idx:e_idx, "drawdown_pct"].min()
        print(f"  {str(s_ts)[:10]} → {str(e_ts)[:10]}  "
              f"dd_min={min_dd*100:>6.1f}%  "
              f"({n_bars:,} bars / {n_bars/288:.1f}d)")


if __name__ == "__main__":
    main()
