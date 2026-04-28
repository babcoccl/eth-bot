#!/usr/bin/env python3
"""
eth_regime_transition_matrix.py
================================
Builds a Markov-style regime transition probability matrix from
eth_audit_transitions.csv (v30 rows).

Produces three outputs:

  eth_transition_matrix.csv     -- 5x5 probability matrix (from->to)
  eth_transition_timing.csv     -- per (from->to) pair: mean/median days
                                   to transition, useful for timing signals
  eth_transition_confidence.csv -- for each live-regime, probability of
                                   reaching each target within N days
                                   (configurable, default 10d)

Also prints a console matrix for quick inspection.

Usage
-----
  python eth_regime_transition_matrix.py
  python eth_regime_transition_matrix.py --transitions eth_audit_transitions.csv
  python eth_regime_transition_matrix.py --window 7    # 7-day confidence window
  python eth_regime_transition_matrix.py --version v30
  python eth_regime_transition_matrix.py --out ./matrix_out/
"""

from __future__ import annotations
import argparse, os, sys
import pandas as pd
import numpy as np


REGIMES = ["CRASH", "CORRECTION", "RECOVERY", "BULL", "RANGE"]


def load_transitions(path: str, version: str = "v30") -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Transitions file not found: {path}\n"
            "Run eth_regime_audit.py first."
        )
    df = pd.read_csv(path, parse_dates=["start_ts", "end_ts"])
    if "version" in df.columns:
        df = df[df["version"] == version].copy()
    # Ensure we only have known regimes
    df = df[df["from_regime"].isin(REGIMES)].copy()
    return df.sort_values("start_ts").reset_index(drop=True)


def build_transition_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a normalised from->to probability matrix.
    Rows = current regime, Columns = next regime.
    """
    pairs = list(zip(df["from_regime"].values[:-1],
                     df["from_regime"].values[1:]))

    counts = pd.DataFrame(0, index=REGIMES, columns=REGIMES)
    for src, dst in pairs:
        if src in REGIMES and dst in REGIMES:
            counts.loc[src, dst] += 1

    # Normalise rows to probabilities
    row_sums = counts.sum(axis=1).replace(0, np.nan)
    prob_matrix = counts.div(row_sums, axis=0).fillna(0).round(4)
    prob_matrix.index.name   = "from_regime"
    prob_matrix.columns.name = "to_regime"
    return prob_matrix


def build_timing_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (from_regime -> to_regime) pair, compute the median and mean
    number of days the FROM segment lasted before transitioning.
    Useful for timing: 'RECOVERY typically lasts ~5d before BULL'.
    """
    if "duration_days" not in df.columns:
        return pd.DataFrame()

    df2 = df.copy()
    df2["next_regime"] = df2["from_regime"].shift(-1)
    df2 = df2.dropna(subset=["next_regime"])
    df2 = df2[df2["next_regime"].isin(REGIMES)]

    records = []
    for (src, dst), grp in df2.groupby(["from_regime", "next_regime"]):
        records.append({
            "from_regime":       src,
            "to_regime":         dst,
            "transitions":       len(grp),
            "mean_days_in_src":  round(grp["duration_days"].mean(), 2),
            "median_days_in_src":round(grp["duration_days"].median(), 2),
            "min_days_in_src":   round(grp["duration_days"].min(), 2),
            "max_days_in_src":   round(grp["duration_days"].max(), 2),
        })
    return (
        pd.DataFrame(records)
        .sort_values(["from_regime", "transitions"], ascending=[True, False])
        .reset_index(drop=True)
    )


def build_confidence_table(
    df: pd.DataFrame,
    window_days: float = 10.0,
) -> pd.DataFrame:
    """
    For each FROM regime: probability of reaching each TO regime
    within `window_days` days based on historical cumulative duration.

    Logic: count how many times a FROM->TO transition occurred where
    the FROM segment lasted <= window_days.
    """
    if "duration_days" not in df.columns:
        return pd.DataFrame()

    df2 = df.copy()
    df2["next_regime"] = df2["from_regime"].shift(-1)
    df2 = df2.dropna(subset=["next_regime"])

    records = []
    for src in REGIMES:
        src_df = df2[df2["from_regime"] == src]
        total  = len(src_df)
        if total == 0:
            continue
        within = src_df[src_df["duration_days"] <= window_days]
        for dst in REGIMES:
            n = int((within["next_regime"] == dst).sum())
            records.append({
                "from_regime":  src,
                "to_regime":    dst,
                "window_days":  window_days,
                "transitions":  n,
                "probability":  round(n / total, 4) if total > 0 else 0,
            })
    df_out = pd.DataFrame(records)
    df_out = df_out[df_out["transitions"] > 0].sort_values(
        ["from_regime", "probability"], ascending=[True, False]
    ).reset_index(drop=True)
    return df_out


def print_matrix(prob_matrix: pd.DataFrame, version: str) -> None:
    SEP = "=" * 70
    print(f"\n{SEP}")
    print(f"  Regime Transition Probability Matrix  [{version}]")
    print(f"  Rows=FROM, Columns=TO  (probability of next committed regime)")
    print(SEP)
    header = f"{'':>14}" + "".join(f"{c:>12}" for c in REGIMES)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for src in REGIMES:
        row_str = f"  {src:<12}  "
        for dst in REGIMES:
            val = prob_matrix.loc[src, dst] if src in prob_matrix.index else 0.0
            row_str += f"{val:>10.1%}  "
        print(row_str)
    print()


def print_timing_highlights(timing_df: pd.DataFrame) -> None:
    print("  Key timing pairs (most frequent transitions):")
    top = timing_df.sort_values("transitions", ascending=False).head(10)
    print(f"  {'FROM':>12}  {'TO':>12}  {'count':>6}  "
          f"{'med_days':>9}  {'mean_days':>9}")
    print("  " + "-" * 54)
    for _, row in top.iterrows():
        print(f"  {row['from_regime']:>12}  {row['to_regime']:>12}  "
              f"{int(row['transitions']):>6}  "
              f"{row['median_days_in_src']:>9.2f}  "
              f"{row['mean_days_in_src']:>9.2f}")
    print()


def main():
    ap = argparse.ArgumentParser(
        description="ETH regime transition probability matrix builder"
    )
    ap.add_argument("--transitions", default="eth_audit_transitions.csv")
    ap.add_argument("--version",     default="v30")
    ap.add_argument("--window",      type=float, default=10.0,
                    help="Confidence window in days (default 10)")
    ap.add_argument("--out",         default=".",
                    help="Output directory (default: current dir)")
    args = ap.parse_args()

    try:
        df = load_transitions(args.transitions, version=args.version)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)

    print(f"Loaded {len(df)} {args.version} transitions from {args.transitions}")

    prob_matrix = build_transition_matrix(df)
    timing_df   = build_timing_table(df)
    conf_df     = build_confidence_table(df, window_days=args.window)

    print_matrix(prob_matrix, args.version)
    if not timing_df.empty:
        print_timing_highlights(timing_df)

    os.makedirs(args.out, exist_ok=True)
    p_matrix  = os.path.join(args.out, "eth_transition_matrix.csv")
    p_timing  = os.path.join(args.out, "eth_transition_timing.csv")
    p_conf    = os.path.join(args.out, "eth_transition_confidence.csv")

    prob_matrix.to_csv(p_matrix)
    timing_df.to_csv(p_timing,   index=False)
    conf_df.to_csv(p_conf,       index=False)

    out_abs = os.path.abspath(args.out)
    print(f"  Outputs written to: {out_abs}/")
    print(f"    eth_transition_matrix.csv       ({len(REGIMES)}x{len(REGIMES)} matrix)")
    print(f"    eth_transition_timing.csv       ({len(timing_df)} pairs)")
    print(f"    eth_transition_confidence.csv   "
          f"({len(conf_df)} pairs within {args.window:.0f}d window)")
    print()


if __name__ == "__main__":
    main()
