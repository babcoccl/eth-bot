#!/usr/bin/env python3
"""
eth_bull_depth_classifier.py
=============================
Standalone BULL depth classifier.

Takes eth_audit_transitions.csv (or any transitions CSV with the columns
  version, from_regime, start_ts, drawdown_at_entry_pct)
and tags every BULL segment with a depth class:

  DEEP    : drawdown_at_entry_pct <= -30%   (post-crash deep trough recoveries)
  MID     : drawdown_at_entry_pct in (-30%, -15%]
  SHALLOW : drawdown_at_entry_pct >  -15%   (noise / range-break false BULLs)

Outputs
-------
  eth_bull_depth_v30.csv  -- v30 BULL segments only, with bull_class column
                             and transition-matrix confidence score

Usage
-----
  python eth_bull_depth_classifier.py
  python eth_bull_depth_classifier.py --transitions eth_audit_transitions.csv
  python eth_bull_depth_classifier.py --version v30  # default
  python eth_bull_depth_classifier.py --deep-threshold -25
"""

from __future__ import annotations
import argparse, os, sys
import pandas as pd
import numpy as np


DEEP_DEFAULT    = -30.0
SHALLOW_DEFAULT = -15.0


def classify_bull_depth(
    dd_pct: float,
    deep_threshold: float = DEEP_DEFAULT,
    shallow_threshold: float = SHALLOW_DEFAULT,
) -> str:
    """
    Classify a BULL segment by its drawdown at entry.

    Parameters
    ----------
    dd_pct            : drawdown as signed percentage, e.g. -40.6
    deep_threshold    : dd_pct <= this -> DEEP   (default -30)
    shallow_threshold : dd_pct >  this -> SHALLOW (default -15)
    """
    if dd_pct <= deep_threshold:
        return "DEEP"
    if dd_pct > shallow_threshold:
        return "SHALLOW"
    return "MID"


def build_bull_depth_table(
    transitions_path: str,
    version: str = "v30",
    deep_threshold: float = DEEP_DEFAULT,
    shallow_threshold: float = SHALLOW_DEFAULT,
) -> pd.DataFrame:
    """
    Load transition CSV and return a classified BULL table.
    """
    if not os.path.exists(transitions_path):
        raise FileNotFoundError(
            f"Transitions file not found: {transitions_path}\n"
            "Run eth_regime_audit.py first to generate eth_audit_transitions.csv"
        )

    df = pd.read_csv(transitions_path)

    if "version" in df.columns:
        df = df[df["version"] == version].copy()

    bulls = df[df["from_regime"] == "BULL"].copy().reset_index(drop=True)

    if bulls.empty:
        print(f"[WARN] No BULL segments found for version={version}")
        return bulls

    bulls["bull_class"] = bulls["drawdown_at_entry_pct"].apply(
        lambda x: classify_bull_depth(float(x), deep_threshold, shallow_threshold)
    )

    # Confidence score: how deep is this BULL relative to DEEP threshold?
    # Score 1.0 = exactly at deep_threshold, >1.0 = deeper, <1.0 = shallower
    bulls["depth_score"] = (bulls["drawdown_at_entry_pct"] / deep_threshold).clip(upper=3.0).round(3)

    cols = [
        "version", "cycle_id", "bull_class", "depth_score",
        "start_ts", "end_ts", "duration_days",
        "entry_price", "exit_price",
        "drawdown_at_entry_pct", "rsi_at_entry",
    ]
    available = [c for c in cols if c in bulls.columns]
    return bulls[available].sort_values("start_ts").reset_index(drop=True)


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
            print(f"    dur  : mean={sub['duration_days'].mean():.2f}d  "
                  f"med={sub['duration_days'].median():.2f}d  "
                  f"max={sub['duration_days'].max():.2f}d")
        if "drawdown_at_entry_pct" in sub.columns:
            print(f"    dd   : mean={sub['drawdown_at_entry_pct'].mean():.1f}%  "
                  f"min={sub['drawdown_at_entry_pct'].min():.1f}%")
        if "rsi_at_entry" in sub.columns:
            print(f"    rsi  : mean={sub['rsi_at_entry'].mean():.1f}")
        if "cycle_id" in sub.columns:
            print(f"    cycles: {', '.join(sub['cycle_id'].unique())}")

    print()


def main():
    ap = argparse.ArgumentParser(
        description="BULL depth classifier for ETH regime transitions"
    )
    ap.add_argument("--transitions",      default="eth_audit_transitions.csv",
                    help="Path to transitions CSV (default: eth_audit_transitions.csv)")
    ap.add_argument("--version",          default="v30",
                    help="Supervisor version to filter on (default: v30)")
    ap.add_argument("--deep-threshold",   type=float, default=DEEP_DEFAULT,
                    help=f"Drawdown pct threshold for DEEP class (default {DEEP_DEFAULT})")
    ap.add_argument("--shallow-threshold",type=float, default=SHALLOW_DEFAULT,
                    help=f"Drawdown pct threshold for SHALLOW class (default {SHALLOW_DEFAULT})")
    ap.add_argument("--out",              default=".",
                    help="Output directory (default: current dir)")
    args = ap.parse_args()

    try:
        bull_df = build_bull_depth_table(
            args.transitions,
            version=args.version,
            deep_threshold=args.deep_threshold,
            shallow_threshold=args.shallow_threshold,
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
