#!/usr/bin/env python3
"""
eth_bull_depth_classifier.py
=============================
Standalone BULL depth classifier.

Takes eth_audit_transitions.csv (or any transitions CSV with columns
  version, from_regime, start_ts, drawdown_at_entry_pct)
and tags every BULL segment with a depth class based on the PRECEDING
CRASH cycle trough (not the drawdown at the BULL entry bar itself,
which is always near-zero due to partial recovery by commit time).

Depth classes
-------------
  DEEP    : cycle_trough_pct <= -30%
  MID     : cycle_trough_pct in (-30%, -15%]
  SHALLOW : cycle_trough_pct >  -15%

Outputs
-------
  eth_bull_depth_v30.csv  -- BULL segments with bull_class + depth_score

Usage
-----
  python eth_bull_depth_classifier.py
  python eth_bull_depth_classifier.py --transitions eth_audit_transitions.csv
  python eth_bull_depth_classifier.py --deep-threshold -25
"""

from __future__ import annotations
import argparse, os, sys
import pandas as pd
import numpy as np


DEEP_DEFAULT    = -30.0
SHALLOW_DEFAULT = -15.0

PAUSE_REGIMES = {"CRASH", "CORRECTION", "RECOVERY"}


def classify_bull_depth(
    dd_pct: float,
    deep_threshold: float = DEEP_DEFAULT,
    shallow_threshold: float = SHALLOW_DEFAULT,
) -> str:
    if dd_pct <= deep_threshold:
        return "DEEP"
    if dd_pct > shallow_threshold:
        return "SHALLOW"
    return "MID"


def _find_cycle_trough(
    df: pd.DataFrame,
    bull_idx: int,
) -> float:
    """
    Walk backwards from bull_idx through consecutive CRASH/CORRECTION/RECOVERY
    rows and return the minimum drawdown_at_entry_pct encountered.

    This reconstructs the cycle trough from the transitions CSV, where each
    row represents a regime segment and drawdown_at_entry_pct is the
    rolling-peak drawdown at the START of that segment.

    Falls back to the BULL row's own drawdown if no pause segment found.
    """
    trough = 0.0
    j = bull_idx - 1
    found = False
    while j >= 0:
        r = str(df.iloc[j]["from_regime"])
        if r in PAUSE_REGIMES:
            found = True
            val = float(df.iloc[j]["drawdown_at_entry_pct"])
            if val < trough:
                trough = val
            j -= 1
        else:
            break
    if not found:
        # RANGE->BULL direct: look back up to 5 rows for any pause
        start = max(0, bull_idx - 5)
        for k in range(start, bull_idx):
            r = str(df.iloc[k]["from_regime"])
            if r in PAUSE_REGIMES:
                val = float(df.iloc[k]["drawdown_at_entry_pct"])
                if val < trough:
                    trough = val
    return round(trough, 2)


def build_bull_depth_table(
    transitions_path: str,
    version: str = "v30",
    deep_threshold: float = DEEP_DEFAULT,
    shallow_threshold: float = SHALLOW_DEFAULT,
) -> pd.DataFrame:
    if not os.path.exists(transitions_path):
        raise FileNotFoundError(
            f"Transitions file not found: {transitions_path}\n"
            "Run eth_regime_audit.py first."
        )

    df = pd.read_csv(transitions_path)
    if "version" in df.columns:
        df = df[df["version"] == version].copy()
    df = df.sort_values("start_ts").reset_index(drop=True)

    bull_rows = df[df["from_regime"] == "BULL"].copy()
    if bull_rows.empty:
        print(f"[WARN] No BULL segments found for version={version}")
        return bull_rows

    troughs = []
    for idx in bull_rows.index:
        trough = _find_cycle_trough(df, idx)
        troughs.append(trough)

    bull_rows = bull_rows.copy()
    bull_rows["cycle_trough_pct"] = troughs
    bull_rows["bull_class"] = bull_rows["cycle_trough_pct"].apply(
        lambda x: classify_bull_depth(float(x), deep_threshold, shallow_threshold)
    )
    # depth_score: 1.0 = at the DEEP threshold, >1.0 = deeper
    bull_rows["depth_score"] = (
        bull_rows["cycle_trough_pct"] / deep_threshold
    ).clip(upper=3.0).round(3)

    cols = [
        "version", "cycle_id", "bull_class", "depth_score",
        "start_ts", "end_ts", "duration_days",
        "entry_price", "exit_price",
        "drawdown_at_entry_pct", "cycle_trough_pct", "rsi_at_entry",
    ]
    available = [c for c in cols if c in bull_rows.columns]
    return bull_rows[available].sort_values("start_ts").reset_index(drop=True)


def print_depth_report(bull_df: pd.DataFrame, version: str) -> None:
    SEP = "=" * 66
    print(f"\n{SEP}")
    print(f"  BULL Depth Classification  [{version}]")
    print(SEP)
    total = len(bull_df)
    for cls in ["DEEP", "MID", "SHALLOW"]:
        sub = bull_df[bull_df["bull_class"] == cls]
        if sub.empty:
            continue
        pct = len(sub) / total * 100
        print(f"\n  {cls} ({len(sub)} segments, {pct:.0f}% of BULLs)")
        if "duration_days" in sub.columns:
            print(f"    dur       : mean={sub['duration_days'].mean():.2f}d  "
                  f"med={sub['duration_days'].median():.2f}d  "
                  f"max={sub['duration_days'].max():.2f}d")
        if "cycle_trough_pct" in sub.columns:
            print(f"    trough    : mean={sub['cycle_trough_pct'].mean():.1f}%  "
                  f"min={sub['cycle_trough_pct'].min():.1f}%")
        if "drawdown_at_entry_pct" in sub.columns:
            print(f"    dd@entry  : mean={sub['drawdown_at_entry_pct'].mean():.1f}%  "
                  f"(rolling-peak dd at BULL commit bar)")
        if "rsi_at_entry" in sub.columns:
            print(f"    rsi       : mean={sub['rsi_at_entry'].mean():.1f}")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transitions",       default="eth_audit_transitions.csv")
    ap.add_argument("--version",           default="v30")
    ap.add_argument("--deep-threshold",    type=float, default=DEEP_DEFAULT)
    ap.add_argument("--shallow-threshold", type=float, default=SHALLOW_DEFAULT)
    ap.add_argument("--out",               default=".")
    args = ap.parse_args()

    try:
        bull_df = build_bull_depth_table(
            args.transitions, args.version,
            args.deep_threshold, args.shallow_threshold,
        )
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)

    print_depth_report(bull_df, args.version)
    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "eth_bull_depth_v30.csv")
    bull_df.to_csv(out_path, index=False)
    print(f"  Written: {os.path.abspath(out_path)}  ({len(bull_df)} BULL segments)")
    print()


if __name__ == "__main__":
    main()
