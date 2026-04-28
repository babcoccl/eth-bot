#!/usr/bin/env python3
"""
eth_regime_audit.py
====================
Side-by-side diagnostic comparison of MacroSupervisor v29 vs v30.

Runs both supervisors over the same historical OHLCV data and writes
three output files:

  eth_audit_transitions.csv   -- full transition log with duration,
                                  trigger, cycle_id on every row.

  eth_audit_summary.csv       -- per-regime aggregate stats for each
                                  version (count, mean/median duration,
                                  total days).

  eth_audit_diff.csv          -- rows where v29 and v30 DISAGREE on
                                  regime label at the same h1 bar.
                                  Key file for confirming fixes worked.

Usage
-----
  python eth_regime_audit.py --start 2021-01-01 --end 2026-04-15
  python eth_regime_audit.py --start 2021-01-01              # end = today
  python eth_regime_audit.py --start 2021-01-01 --out ./audit_out/
  python eth_regime_audit.py --start 2021-01-01 --min-dwell 0  # v30 dwell off

Outputs land in the current directory (or --out path).

Note: SQLite db files are written to temp files and deleted after the run.
"""

from __future__ import annotations
import argparse, os, sys, tempfile
from datetime import datetime, timezone, timedelta
from typing import List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Temp-DB context helper
# ---------------------------------------------------------------------------

class _TempDB:
    """
    Context manager that creates a real on-disk temp SQLite file and
    deletes it on exit.  Avoids the multi-connection schema-loss bug
    that occurs with db_path=':memory:'.
    """
    def __init__(self):
        self.path: str = ""

    def __enter__(self) -> str:
        fd, self.path = tempfile.mkstemp(suffix=".db", prefix="audit_sup_")
        os.close(fd)
        return self.path

    def __exit__(self, *_):
        try:
            os.unlink(self.path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_modules():
    """Import both supervisor versions and eth_helpers. Raises ImportError."""
    try:
        from eth_macrosupervisor_v30 import MacroSupervisor as MS29
    except ImportError:
        raise ImportError("eth_macrosupervisor_v30.py not found in current directory.")
    try:
        from eth_macrosupervisor_v30 import MacroSupervisor as MS30
    except ImportError:
        raise ImportError("eth_macrosupervisor_v30.py not found in current directory.")
    try:
        from eth_helpers import fetch_ohlcv
    except ImportError:
        raise ImportError("eth_helpers.py not found in current directory.")
    return MS29, MS30, fetch_ohlcv


def _build_transition_df(
    regime_arr: np.ndarray,
    ts_index: pd.DatetimeIndex,
    h1_df: pd.DataFrame,
    version_label: str,
    pause_events: list,
) -> pd.DataFrame:
    """
    Build a rich transition table from a committed regime5 array.

    Columns
    -------
    version, cycle_id, from_regime, start_ts, end_ts,
    duration_bars, duration_days, entry_price, exit_price,
    drawdown_at_entry_pct, rsi_at_entry, trigger
    """
    # ts -> pause trigger lookup
    trigger_map: dict = {}
    for pe in pause_events:
        trigger_map[str(pe["ts"])] = pe.get("trigger", "")

    rows: List[dict] = []
    n           = len(regime_arr)
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
            dd_entry  = round(float(h1_dd[seg_start]) * 100, 2)
            rsi_raw   = h1_rsi[seg_start]
            rsi_entry = round(float(rsi_raw) if not np.isnan(rsi_raw) else 0.0, 1)
            trigger   = trigger_map.get(str(start_ts), "")

            rows.append({
                "version":               version_label,
                "from_regime":           prev_regime,
                "start_ts":              start_ts,
                "end_ts":                end_ts,
                "duration_bars":         dur_bars,
                "duration_days":         round(dur_bars / 24, 2),
                "entry_price":           round(ep, 2),
                "exit_price":            round(xp, 2),
                "drawdown_at_entry_pct": dd_entry,
                "rsi_at_entry":          rsi_entry,
                "trigger":               trigger,
            })
            prev_regime = cur
            seg_start   = i

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # cycle_id: each CRASH/CORRECTION segment starts a new macro cycle
    cycle     = 0
    cycle_ids = []
    for _, row in df.iterrows():
        if row["from_regime"] in ("CRASH", "CORRECTION"):
            cycle += 1
        cycle_ids.append(f"Cy{chr(64 + cycle)}" if cycle > 0 else "pre")
    df["cycle_id"] = cycle_ids

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
    return pd.DataFrame(records).sort_values(
        ["version", "total_days"], ascending=[True, False]
    ).reset_index(drop=True)


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
    rsi_raw = h1_df["rsi"].values[mask]
    return pd.DataFrame({
        "ts":           ts_index[mask],
        "v29_regime":   reg29[mask],
        "v30_regime":   reg30[mask],
        "price":        h1_df["close"].values[mask].round(2),
        "drawdown_pct": (h1_df["drawdown"].values[mask] * 100).round(2),
        "rsi":          np.where(np.isnan(rsi_raw), 0.0, rsi_raw).round(1),
    })


def _print_console_summary(
    trans29: pd.DataFrame,
    trans30: pd.DataFrame,
    diff_df: pd.DataFrame,
) -> None:
    SEP = "=" * 66
    print(f"\n{SEP}")
    print("  ETH Regime Audit -- v29 vs v30")
    print(SEP)

    for ver, tdf in [("v29", trans29), ("v30", trans30)]:
        print(f"\n  [{ver}]  {len(tdf)} regime segments")
        for regime in ["CRASH", "CORRECTION", "RECOVERY", "BULL", "RANGE"]:
            sub = tdf[tdf["from_regime"] == regime]
            if sub.empty:
                continue
            print(
                f"    {regime:<12}  {len(sub):>3} segs  "
                f"{sub['duration_days'].sum():>8.1f} days  "
                f"med={sub['duration_days'].median():.2f}d  "
                f"min={sub['duration_days'].min():.2f}d"
            )

    print(f"\n  Bar-level diff (v29 != v30): {len(diff_df):,} h1 bars")
    if not diff_df.empty:
        pair_counts = (
            diff_df.groupby(["v29_regime", "v30_regime"])
            .size()
            .reset_index(name="bars")
            .sort_values("bars", ascending=False)
        )
        print(f"  {'v29_regime':>14}  {'v30_regime':>14}  {'bars':>8}")
        print("  " + "-" * 40)
        for _, row in pair_counts.iterrows():
            print(
                f"  {row['v29_regime']:>14}  {row['v30_regime']:>14}"
                f"  {row['bars']:>8,}"
            )
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="eth_regime_audit: v29 vs v30 side-by-side comparison"
    )
    ap.add_argument("--start",     default="2021-01-01",
                    help="Start date YYYY-MM-DD (default: 2021-01-01)")
    ap.add_argument("--end",       default=None,
                    help="End date YYYY-MM-DD (default: today)")
    ap.add_argument("--symbol",    default="ETH/USD")
    ap.add_argument("--out",       default=".",
                    help="Output directory for CSV files (default: current dir)")
    ap.add_argument("--min-dwell", type=int, default=None,
                    help="Override v30 regime5_min_dwell_bars (0 = disable)")
    args = ap.parse_args()

    try:
        MS29, MS30, fetch_ohlcv = _load_modules()
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

    # Use a single-row 5m stub -- apply_to_df requires df5m but we only
    # need the h1 regime series from each supervisor.
    df5_stub = df5.iloc[:1].copy()

    v30_kwargs: dict = {}
    if args.min_dwell is not None:
        v30_kwargs["regime5_min_dwell_bars"] = args.min_dwell

    # Run each supervisor with a real temp DB file to avoid the
    # multi-connection :memory: schema loss bug.
    with _TempDB() as db29, _TempDB() as db30:
        print("Running v29 ...")
        sup29 = MS29(db_path=db29)
        sup29.apply_to_df(df5_stub, df1h.copy())
        reg29    = np.array(sup29._h1_r5_series.values, dtype=object)
        ts_index = sup29._h1_ts_index

        print("Running v30 ...")
        sup30 = MS30(db_path=db30, **v30_kwargs)
        sup30.apply_to_df(df5_stub, df1h.copy())
        reg30 = np.array(sup30._h1_r5_series.values, dtype=object)

    # Recompute enriched h1 signals (drawdown, rsi) for the diff table.
    # Uses v30 params -- identical to v29 for these columns.
    print("Building reference h1 signals ...")
    h_ref = df1h.copy().sort_values("ts").reset_index(drop=True)
    h_ref["fast_ema"]     = h_ref["close"].ewm(span=sup30.ema_fast,  adjust=False).mean()
    h_ref["slow_ema"]     = h_ref["close"].ewm(span=sup30.ema_slow,  adjust=False).mean()
    h_ref["rsi"]          = sup30._calc_rsi(h_ref["close"])
    h_ref["rolling_peak"] = h_ref["close"].rolling(sup30.peak_window, min_periods=1).max()
    h_ref["drawdown"]     = (h_ref["close"] - h_ref["rolling_peak"]) / h_ref["rolling_peak"]

    # Build output tables
    print("Building audit tables ...")
    trans29   = _build_transition_df(reg29, ts_index, h_ref, "v29", sup29.pause_events)
    trans30   = _build_transition_df(reg30, ts_index, h_ref, "v30", sup30.pause_events)
    trans_all = pd.concat([trans29, trans30], ignore_index=True)
    sum_df    = _build_summary(trans_all)
    diff_df   = _build_diff(reg29, reg30, ts_index, h_ref)

    # Write CSVs
    os.makedirs(args.out, exist_ok=True)
    p_trans = os.path.join(args.out, "eth_audit_transitions.csv")
    p_sum   = os.path.join(args.out, "eth_audit_summary.csv")
    p_diff  = os.path.join(args.out, "eth_audit_diff.csv")

    trans_all.to_csv(p_trans, index=False)
    sum_df.to_csv(p_sum,     index=False)
    diff_df.to_csv(p_diff,   index=False)

    _print_console_summary(trans29, trans30, diff_df)

    out_abs = os.path.abspath(args.out)
    print(f"  Outputs written to: {out_abs}/")
    print(f"    eth_audit_transitions.csv  ({len(trans_all)} rows)")
    print(f"    eth_audit_summary.csv      ({len(sum_df)} rows)")
    print(f"    eth_audit_diff.csv         ({len(diff_df):,} rows -- bars where v29 != v30)")
    print()


if __name__ == "__main__":
    main()
