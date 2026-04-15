#!/usr/bin/env python3
"""
eth_regime_audit.py
====================
Side-by-side diagnostic comparison of MacroSupervisor v29 vs v30.

Runs both supervisors over the same historical OHLCV data and writes
three output files:

  eth_audit_transitions.csv   -- full transition log with duration,
                                  trigger, dwell, cycle_id, and a
                                  v29/v30 diff flag on every row.

  eth_audit_summary.csv       -- per-regime aggregate stats for each
                                  version (count, mean/median duration,
                                  total days, pct of period).

  eth_audit_diff.csv          -- rows where v29 and v30 DISAGREE on
                                  regime label at the same h1 bar.
                                  Key file for confirming fixes worked.

Usage
-----
  python eth_regime_audit.py --start 2021-01-01 --end 2026-04-15
  python eth_regime_audit.py --start 2021-01-01              # end = today
  python eth_regime_audit.py --start 2021-01-01 --out ./audit_out/
  python eth_regime_audit.py --start 2021-01-01 --min-dwell 0  # v30 with dwell off

Outputs land in the current directory (or --out path).
"""

from __future__ import annotations
import argparse, os, sys
from datetime import datetime, timezone, timedelta
from typing import List, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_supervisors(min_dwell: Optional[int] = None):
    """Import both supervisor versions.  Raises ImportError with guidance."""
    try:
        from eth_macrosupervisor_v29 import MacroSupervisor as MS29
    except ImportError:
        raise ImportError("eth_macrosupervisor_v29.py not found in current directory.")
    try:
        from eth_macrosupervisor_v30 import MacroSupervisor as MS30
    except ImportError:
        raise ImportError("eth_macrosupervisor_v30.py not found in current directory.")
    try:
        from eth_helpers import fetch_ohlcv
    except ImportError:
        raise ImportError("eth_helpers.py not found in current directory.")

    v29_kwargs: dict = {}
    v30_kwargs: dict = {}
    if min_dwell is not None:
        v30_kwargs["regime5_min_dwell_bars"] = min_dwell

    sup29 = MS29(db_path=":memory:", **v29_kwargs)
    sup30 = MS30(db_path=":memory:", **v30_kwargs)
    return sup29, sup30, fetch_ohlcv


def _run_supervisor(sup, df5: pd.DataFrame, df1h: pd.DataFrame) -> pd.Series:
    """Apply supervisor; return the h1-aligned regime5 series indexed by ts."""
    # We only care about the h1 regime series -- use a minimal 5m df
    sup.apply_to_df(df5.head(1).copy(), df1h)
    return sup._h1_r5_series.values, sup._h1_ts_index


def _build_transition_df(
    regime_arr: np.ndarray,
    ts_index: pd.DatetimeIndex,
    h1_df: pd.DataFrame,
    version_label: str,
    pause_events: list,
    resume_events: list,
) -> pd.DataFrame:
    """
    Build a rich transition table from a regime5 array.

    Columns
    -------
    version, cycle_id, from_regime, to_regime, start_ts, end_ts,
    duration_bars, duration_days, entry_price, exit_price,
    drawdown_at_entry_pct, rsi_at_entry, trigger, dwell_at_exit
    """
    # build a lookup: ts -> pause trigger
    trigger_map: dict = {}
    for pe in pause_events:
        trigger_map[str(pe["ts"])] = pe.get("trigger", "")

    rows: List[dict] = []
    n = len(regime_arr)
    prev_regime = str(regime_arr[0])
    seg_start   = 0

    h1_close = h1_df["close"].values
    h1_dd    = h1_df["drawdown"].values
    h1_rsi   = h1_df["rsi"].values

    for i in range(1, n + 1):
        cur = str(regime_arr[i]) if i < n else "__END__"
        if cur != prev_regime:
            seg_end   = i - 1
            dur_bars  = seg_end - seg_start + 1
            start_ts  = ts_index[seg_start]
            end_ts    = ts_index[seg_end]
            ep        = float(h1_close[seg_start])
            xp        = float(h1_close[seg_end])
            dd_entry  = round(float(h1_dd[seg_start])  * 100, 2)
            rsi_entry = round(float(h1_rsi[seg_start]) if not np.isnan(h1_rsi[seg_start]) else 0.0, 1)
            trigger   = trigger_map.get(str(start_ts), "")

            rows.append({
                "version":            version_label,
                "from_regime":        prev_regime,
                "start_ts":           start_ts,
                "end_ts":             end_ts,
                "duration_bars":      dur_bars,
                "duration_days":      round(dur_bars / 24, 2),
                "entry_price":        round(ep, 2),
                "exit_price":         round(xp, 2),
                "drawdown_at_entry_pct": dd_entry,
                "rsi_at_entry":       rsi_entry,
                "trigger":            trigger,
            })
            prev_regime = cur
            seg_start   = i

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # cycle_id: each CRASH/CORRECTION start begins a new macro cycle
    cycle = 0
    cycle_ids = []
    for _, row in df.iterrows():
        if row["from_regime"] in ("CRASH", "CORRECTION"):
            cycle += 1
        cycle_ids.append(f"Cy{chr(64 + cycle)}" if cycle > 0 else "pre")
    df["cycle_id"] = cycle_ids

    # reorder columns
    cols = [
        "version", "cycle_id", "from_regime",
        "start_ts", "end_ts", "duration_bars", "duration_days",
        "entry_price", "exit_price",
        "drawdown_at_entry_pct", "rsi_at_entry", "trigger",
    ]
    return df[[c for c in cols if c in df.columns]]


def _build_summary(trans_df: pd.DataFrame) -> pd.DataFrame:
    """Per-version, per-regime aggregate stats."""
    records = []
    for (ver, regime), grp in trans_df.groupby(["version", "from_regime"]):
        records.append({
            "version":           ver,
            "regime":            regime,
            "segment_count":     len(grp),
            "total_days":        round(grp["duration_days"].sum(), 1),
            "mean_dur_days":     round(grp["duration_days"].mean(), 2),
            "median_dur_days":   round(grp["duration_days"].median(), 2),
            "min_dur_days":      round(grp["duration_days"].min(), 2),
            "max_dur_days":      round(grp["duration_days"].max(), 2),
            "mean_dd_entry_pct": round(grp["drawdown_at_entry_pct"].mean(), 2),
        })
    return pd.DataFrame(records).sort_values(["version", "total_days"], ascending=[True, False])


def _build_diff(
    reg29: np.ndarray,
    reg30: np.ndarray,
    ts_index: pd.DatetimeIndex,
    h1_df: pd.DataFrame,
) -> pd.DataFrame:
    """Bar-by-bar diff: only rows where v29 != v30."""
    mask = reg29 != reg30
    if not mask.any():
        return pd.DataFrame(columns=["ts", "v29_regime", "v30_regime",
                                     "price", "drawdown_pct", "rsi"])
    ts_arr  = ts_index[mask]
    v29_arr = reg29[mask]
    v30_arr = reg30[mask]
    close   = h1_df["close"].values[mask]
    dd      = (h1_df["drawdown"].values[mask] * 100).round(2)
    rsi     = h1_df["rsi"].values[mask]

    return pd.DataFrame({
        "ts":           ts_arr,
        "v29_regime":   v29_arr,
        "v30_regime":   v30_arr,
        "price":        close.round(2),
        "drawdown_pct": dd,
        "rsi":          np.where(np.isnan(rsi), 0.0, rsi).round(1),
    })


def _print_console_summary(
    trans29: pd.DataFrame,
    trans30: pd.DataFrame,
    diff_df: pd.DataFrame,
    sum_df: pd.DataFrame,
) -> None:
    sep = "-" * 66
    print(f"\n{'='*66}")
    print("  ETH Regime Audit -- v29 vs v30")
    print(f"{'='*66}")

    for ver, tdf in [("v29", trans29), ("v30", trans30)]:
        print(f"\n  [{ver}] {len(tdf)} regime segments")
        regime_order = ["CRASH", "CORRECTION", "RECOVERY", "BULL", "RANGE"]
        for regime in regime_order:
            sub = tdf[tdf["from_regime"] == regime]
            if sub.empty:
                continue
            print(f"    {regime:<12}  {len(sub):>3} segs  "
                  f"{sub['duration_days'].sum():>7.1f} days  "
                  f"med={sub['duration_days'].median():.1f}d  "
                  f"min={sub['duration_days'].min():.2f}d")

    print(f"\n  Bar-level diff (v29 != v30): {len(diff_df):,} h1 bars")
    if not diff_df.empty:
        pair_counts = diff_df.groupby(["v29_regime", "v30_regime"]).size() \
                             .reset_index(name="bars") \
                             .sort_values("bars", ascending=False)
        print(f"  {'v29':>12}  {'v30':>12}  {'bars':>8}")
        print(f"  {sep[:38]}")
        for _, row in pair_counts.iterrows():
            print(f"  {row['v29_regime']:>12}  {row['v30_regime']:>12}  {row['bars']:>8,}")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="eth_regime_audit: v29 vs v30 comparison")
    ap.add_argument("--start",     default="2021-01-01",
                    help="Backtest start date YYYY-MM-DD (default: 2021-01-01)")
    ap.add_argument("--end",       default=None,
                    help="Backtest end date YYYY-MM-DD (default: today)")
    ap.add_argument("--symbol",    default="ETH/USD")
    ap.add_argument("--out",       default=".",
                    help="Output directory for CSV files (default: current dir)")
    ap.add_argument("--min-dwell", type=int, default=None,
                    help="Override v30 regime5_min_dwell_bars (0 = disable)")
    args = ap.parse_args()

    try:
        sup29, sup30, fetch_ohlcv = _load_supervisors(args.min_dwell)
    except ImportError as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)

    start_dt = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_s    = args.end or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    end_dt   = (datetime.strptime(end_s, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                + timedelta(days=1))

    print(f"Fetching data {args.start} -> {end_s} ...")
    df1h = fetch_ohlcv(args.symbol, "1h", start_dt, end_dt)
    df5  = fetch_ohlcv(args.symbol, "5m", start_dt, end_dt)
    print(f"  1h bars: {len(df1h):,}   5m bars: {len(df5):,}")

    # --- compute h1 signals for both versions ---
    # We need the enriched h1 df (with drawdown, rsi) -- run v30 first to
    # get it, then reuse the same enriched frame for the diff.
    print("Running v29 ...")
    # Use a tiny 5m slice so apply_to_df doesn't choke on memory when 5m
    # is only needed to satisfy the signature.
    df5_stub = df5.iloc[:1].copy()
    sup29.apply_to_df(df5_stub, df1h.copy())
    reg29     = np.array(sup29._h1_r5_series.values, dtype=object)
    ts_index  = sup29._h1_ts_index

    print("Running v30 ...")
    sup30.apply_to_df(df5_stub, df1h.copy())
    reg30 = np.array(sup30._h1_r5_series.values, dtype=object)

    # rebuild enriched h1 frame from v30 supervisor internals
    # (_compute_h1_signals stores results on the supervisor but not publicly;
    # re-derive the enriched df by calling the private method directly)
    h1_enriched = sup30._MacroSupervisor__build_enriched_h1 \
        if hasattr(sup30, "_MacroSupervisor__build_enriched_h1") else None

    # Fallback: recompute signals manually using the same params as v30
    # so we have drawdown/rsi columns for the diff table.
    h_ref = df1h.copy().sort_values("ts").reset_index(drop=True)
    ema_fast = sup30.ema_fast
    ema_slow = sup30.ema_slow
    h_ref["fast_ema"]       = h_ref["close"].ewm(span=ema_fast, adjust=False).mean()
    h_ref["slow_ema"]       = h_ref["close"].ewm(span=ema_slow, adjust=False).mean()
    h_ref["rsi"]            = sup30._calc_rsi(h_ref["close"])
    h_ref["rolling_peak"]   = h_ref["close"].rolling(sup30.peak_window, min_periods=1).max()
    h_ref["drawdown"]       = (h_ref["close"] - h_ref["rolling_peak"]) / h_ref["rolling_peak"]

    # --- build output tables ---
    print("Building audit tables ...")
    trans29 = _build_transition_df(
        reg29, ts_index, h_ref, "v29",
        sup29.pause_events, sup29.resume_events)
    trans30 = _build_transition_df(
        reg30, ts_index, h_ref, "v30",
        sup30.pause_events, sup30.resume_events)
    trans_all = pd.concat([trans29, trans30], ignore_index=True)
    sum_df    = _build_summary(trans_all)
    diff_df   = _build_diff(reg29, reg30, ts_index, h_ref)

    # --- write CSVs ---
    os.makedirs(args.out, exist_ok=True)
    p_trans = os.path.join(args.out, "eth_audit_transitions.csv")
    p_sum   = os.path.join(args.out, "eth_audit_summary.csv")
    p_diff  = os.path.join(args.out, "eth_audit_diff.csv")

    trans_all.to_csv(p_trans, index=False)
    sum_df.to_csv(p_sum,     index=False)
    diff_df.to_csv(p_diff,   index=False)

    _print_console_summary(trans29, trans30, diff_df, sum_df)

    print(f"  Outputs written to: {os.path.abspath(args.out)}/")
    print(f"    eth_audit_transitions.csv  ({len(trans_all)} rows)")
    print(f"    eth_audit_summary.csv      ({len(sum_df)} rows)")
    print(f"    eth_audit_diff.csv         ({len(diff_df):,} rows -- bars where v29 != v30)")
    print()


if __name__ == "__main__":
    main()
