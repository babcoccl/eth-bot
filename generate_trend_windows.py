#!/usr/bin/env python3
"""
generate_trend_windows.py — Auto-detect BULL/RECOVERY trend windows from ETH history
======================================================================================
Reads eth_annotated_history.csv.gz (which has regime5 already annotated by
MacroSupervisor v30) and algorithmically detects BULL/RECOVERY trend windows
using the same criteria MacroSupervisor uses to define those regimes.

Window detection logic (mirrors MacroSupervisor v30):
  - A window OPENS when regime5 enters BULL or RECOVERY (on h1 bars)
  - A window CLOSES when regime5 exits to CRASH, CORRECTION, or RANGE
    for >= min_close_bars consecutive h1 bars (debounced, matching
    regime5_min_dwell_bars logic in MacroSupervisor)
  - Windows are QUALIFIED using the same criteria the harness should enforce:
      * min_days         >= 7   (window must be long enough to be meaningful)
      * min_gain_pct     >= 8%  (price must actually trend up)
      * min_bull_recov_pct >= 0.40  (>=40% of bars must be BULL/RECOVERY — raised from harness 30%)
      * max_hostile_pct  <= 0.40  (CRASH+CORRECTION must be <40% of window bars)
  - Strength is classified from actual price gain (not hand-labeled):
      MODERATE : +8%  to +20%
      STRONG   : +20% to +40%
      PARABOLIC: >+40%

Output:
  Prints a new TREND_WINDOWS list ready to paste into trend_windows_4yr.py.
  Also writes trend_windows_generated.py with the full list.

Usage:
  python generate_trend_windows.py
  python generate_trend_windows.py --min-days 5 --min-gain 10 --min-bull-recov 0.45
"""

import argparse
import os
import sys

import pandas as pd


# ── Strength classification (matches trend_windows_4yr.py header definition) ──
def classify_strength(gain_pct: float) -> str:
    if gain_pct >= 40.0:
        return "PARABOLIC"
    if gain_pct >= 20.0:
        return "STRONG"
    return "MODERATE"


# ── Window qualifier: checks regime composition inside detected window ─────────
def qualify_window(df_window: pd.DataFrame, min_bull_recov_pct: float,
                   max_hostile_pct: float) -> tuple:
    """
    Returns (passes: bool, stats: dict).
    Uses h1-resampled regime5 to match MacroSupervisor's h1-level classification.
    """
    if df_window.empty:
        return False, {}

    # Resample to h1 to match MacroSupervisor's resolution
    h1 = (df_window.set_index("ts")["regime5"]
          .resample("1h").last()
          .dropna())

    if len(h1) == 0:
        return False, {}

    total      = len(h1)
    bull_recov = h1.isin(["BULL", "RECOVERY"]).sum() / total
    hostile    = h1.isin(["CRASH", "CORRECTION"]).sum() / total
    bull_pct   = (h1 == "BULL").sum() / total
    recov_pct  = (h1 == "RECOVERY").sum() / total
    crash_pct  = (h1 == "CRASH").sum() / total
    corr_pct   = (h1 == "CORRECTION").sum() / total

    stats = {
        "bull_recov_pct": bull_recov,
        "hostile_pct":    hostile,
        "bull_pct":       bull_pct,
        "recov_pct":      recov_pct,
        "crash_pct":      crash_pct,
        "corr_pct":       corr_pct,
        "h1_bars":        total,
    }

    passes = (bull_recov >= min_bull_recov_pct and hostile <= max_hostile_pct)
    return passes, stats


# ── Main detection logic ───────────────────────────────────────────────────────
def detect_windows(df: pd.DataFrame,
                   min_days: float,
                   min_gain_pct: float,
                   min_bull_recov_pct: float,
                   max_hostile_pct: float,
                   min_close_bars: int) -> list:
    """
    Detect BULL/RECOVERY windows from a 5m-resolution annotated DataFrame.

    Parameters
    ----------
    df                  : 5m OHLCV + regime5 DataFrame with 'ts', 'close', 'regime5'
    min_days            : minimum window duration in calendar days
    min_gain_pct        : minimum price gain % from window open to window close
    min_bull_recov_pct  : minimum fraction of h1 bars that must be BULL/RECOVERY
    max_hostile_pct     : maximum fraction of h1 bars allowed to be CRASH/CORRECTION
    min_close_bars      : h1 bars hostile regime must persist before closing window
                          (mirrors MacroSupervisor regime5_min_dwell_bars logic)
    """
    # Work on h1 bars — MacroSupervisor classifies at h1 resolution
    h1 = (df.set_index("ts")[["close", "regime5"]]
            .resample("1h").agg({"close": "last", "regime5": "last"})
            .dropna(subset=["regime5"])
            .reset_index())

    windows        = []
    in_window      = False
    win_start_idx  = None
    win_start_ts   = None
    win_open_price = None
    hostile_streak = 0   # consecutive h1 bars outside BULL/RECOVERY

    _TRADEABLE = {"BULL", "RECOVERY"}

    for i, row in h1.iterrows():
        regime = str(row["regime5"])
        ts     = row["ts"]
        close  = float(row["close"])

        if not in_window:
            if regime in _TRADEABLE:
                in_window      = True
                win_start_idx  = i
                win_start_ts   = ts
                win_open_price = close
                hostile_streak = 0
        else:
            if regime in _TRADEABLE:
                hostile_streak = 0
            else:
                hostile_streak += 1

            # Close window only after min_close_bars of non-tradeable bars
            # (mirrors regime5_min_dwell_bars debounce in MacroSupervisor)
            if hostile_streak >= min_close_bars:
                # Window closes min_close_bars bars ago (back-date the close)
                close_idx     = max(i - min_close_bars, win_start_idx)
                win_end_ts    = h1.loc[close_idx, "ts"]
                win_end_price = float(h1.loc[close_idx, "close"])

                days     = (win_end_ts - win_start_ts).total_seconds() / 86400
                gain_pct = (win_end_price - win_open_price) / win_open_price * 100

                if days >= min_days and gain_pct >= min_gain_pct:
                    mask      = (df["ts"] >= win_start_ts) & (df["ts"] <= win_end_ts)
                    df_window = df[mask].copy()
                    passes, stats = qualify_window(
                        df_window, min_bull_recov_pct, max_hostile_pct)

                    windows.append({
                        "start":          win_start_ts.strftime("%Y-%m-%d"),
                        "end":            win_end_ts.strftime("%Y-%m-%d"),
                        "days":           round(days, 1),
                        "gain_pct":       round(gain_pct, 1),
                        "strength":       classify_strength(gain_pct),
                        "passes":         passes,
                        "bull_recov_pct": round(stats.get("bull_recov_pct", 0) * 100, 1),
                        "hostile_pct":    round(stats.get("hostile_pct", 0) * 100, 1),
                        "bull_pct":       round(stats.get("bull_pct", 0) * 100, 1),
                        "recov_pct":      round(stats.get("recov_pct", 0) * 100, 1),
                        "crash_pct":      round(stats.get("crash_pct", 0) * 100, 1),
                        "corr_pct":       round(stats.get("corr_pct", 0) * 100, 1),
                        "h1_bars":        stats.get("h1_bars", 0),
                    })

                in_window      = False
                win_start_idx  = None
                win_start_ts   = None
                win_open_price = None
                hostile_streak = 0

                # If the current bar is already tradeable, open a new window immediately
                if regime in _TRADEABLE:
                    in_window      = True
                    win_start_idx  = i
                    win_start_ts   = ts
                    win_open_price = close
                    hostile_streak = 0

    # Handle open window at end of data
    if in_window and win_start_ts is not None:
        last_row      = h1.iloc[-1]
        win_end_ts    = last_row["ts"]
        win_end_price = float(last_row["close"])
        days          = (win_end_ts - win_start_ts).total_seconds() / 86400
        gain_pct      = (win_end_price - win_open_price) / win_open_price * 100

        if days >= min_days and gain_pct >= min_gain_pct:
            mask      = (df["ts"] >= win_start_ts) & (df["ts"] <= win_end_ts)
            df_window = df[mask].copy()
            passes, stats = qualify_window(
                df_window, min_bull_recov_pct, max_hostile_pct)
            windows.append({
                "start":          win_start_ts.strftime("%Y-%m-%d"),
                "end":            win_end_ts.strftime("%Y-%m-%d"),
                "days":           round(days, 1),
                "gain_pct":       round(gain_pct, 1),
                "strength":       classify_strength(gain_pct),
                "passes":         passes,
                "bull_recov_pct": round(stats.get("bull_recov_pct", 0) * 100, 1),
                "hostile_pct":    round(stats.get("hostile_pct", 0) * 100, 1),
                "bull_pct":       round(stats.get("bull_pct", 0) * 100, 1),
                "recov_pct":      round(stats.get("recov_pct", 0) * 100, 1),
                "crash_pct":      round(stats.get("crash_pct", 0) * 100, 1),
                "corr_pct":       round(stats.get("corr_pct", 0) * 100, 1),
                "h1_bars":        stats.get("h1_bars", 0),
            })

    return windows


# ── Output formatter ───────────────────────────────────────────────────────────
def format_output(windows: list) -> str:
    lines = [
        '#!/usr/bin/env python3',
        '"""',
        'trend_windows_generated.py — Auto-generated by generate_trend_windows.py',
        '',
        'DO NOT EDIT BY HAND. Re-run generate_trend_windows.py to regenerate.',
        '',
        'Qualification criteria applied at generation time:',
        '  - min_bull_recov_pct >= 40% of h1 bars are BULL/RECOVERY',
        '  - max_hostile_pct    <= 40% of h1 bars are CRASH/CORRECTION',
        '  - min_days           >= 7 calendar days',
        '  - min_gain_pct       >= 8% price gain over window',
        '  - Windows with passes=False are included as comments for audit trail.',
        '"""',
        '',
        'TREND_WINDOWS = [',
    ]

    year = None
    for w in windows:
        y = w["start"][:4]
        if y != year:
            year = y
            lines.append(f'    # ── {year} {"─" * 68}')

        regime_dist = (f"bull={w['bull_pct']:.0f}% recov={w['recov_pct']:.0f}% "
                       f"crash={w['crash_pct']:.0f}% corr={w['corr_pct']:.0f}%")
        qual_note   = (f"tradeable={w['bull_recov_pct']:.0f}% "
                       f"hostile={w['hostile_pct']:.0f}% | {regime_dist}")

        entry = (
            f'    {{\n'
            f'        "label":    "#{w["start"][:7]}",\n'
            f'        "start":    "{w["start"]}",\n'
            f'        "end":      "{w["end"]}",\n'
            f'        "days":     {w["days"]},\n'
            f'        "gain_pct": {w["gain_pct"]:+.1f},\n'
            f'        "strength": "{w["strength"]}",\n'
            f'        # {qual_note}\n'
            f'    }},'
        )

        if not w["passes"]:
            lines.append(f'    # EXCLUDED (failed qualifier):')
            for line in entry.splitlines():
                lines.append(f'    # {line.strip()}')
        else:
            lines.append(entry)

    lines.append(']')
    return "\n".join(lines)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Generate trend windows from ETH annotated history")
    ap.add_argument("--input",           default="eth_annotated_history.csv.gz",
                    help="Path to eth_annotated_history.csv.gz")
    ap.add_argument("--output",          default="trend_windows_generated.py",
                    help="Output file path")
    ap.add_argument("--min-days",        default=7.0,  type=float,
                    help="Minimum window duration in days (default: 7)")
    ap.add_argument("--min-gain",        default=8.0,  type=float,
                    help="Minimum price gain %% over window (default: 8)")
    ap.add_argument("--min-bull-recov",  default=0.40, type=float,
                    help="Min fraction of h1 bars that must be BULL/RECOVERY (default: 0.40)")
    ap.add_argument("--max-hostile",     default=0.40, type=float,
                    help="Max fraction of h1 bars allowed to be CRASH/CORRECTION (default: 0.40)")
    ap.add_argument("--min-close-bars",  default=3,    type=int,
                    help="H1 bars hostile regime must persist to close window — "
                         "mirrors MacroSupervisor regime5_min_dwell_bars (default: 3)")
    ap.add_argument("--show-excluded",   action="store_true",
                    help="Print excluded windows in the console summary")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        print(f"[ERROR] Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {args.input} ...")
    df = pd.read_csv(args.input, compression="gzip", parse_dates=["ts"])
    if df["ts"].dt.tz is None:
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.sort_values("ts").reset_index(drop=True)
    print(f"  Loaded {len(df):,} 5m bars  "
          f"({df['ts'].iloc[0].date()} -> {df['ts'].iloc[-1].date()})")

    if "regime5" not in df.columns:
        print("[ERROR] 'regime5' column not found. "
              "Run MacroSupervisor.apply_to_df() to annotate first.", file=sys.stderr)
        sys.exit(1)

    print(f"\nDetecting windows ...")
    print(f"  min_days={args.min_days}  min_gain={args.min_gain}%  "
          f"min_bull_recov={args.min_bull_recov:.0%}  "
          f"max_hostile={args.max_hostile:.0%}  "
          f"min_close_bars={args.min_close_bars}")

    windows = detect_windows(
        df,
        min_days           = args.min_days,
        min_gain_pct       = args.min_gain,
        min_bull_recov_pct = args.min_bull_recov,
        max_hostile_pct    = args.max_hostile,
        min_close_bars     = args.min_close_bars,
    )

    passed   = [w for w in windows if     w["passes"]]
    excluded = [w for w in windows if not w["passes"]]

    print(f"\n{'='*72}")
    print(f"  Detected {len(windows)} candidate windows  "
          f"->  {len(passed)} PASS  |  {len(excluded)} EXCLUDED")
    print(f"{'='*72}")
    print(f"  {'Start':<12} {'End':<12} {'Days':>5} {'Gain%':>7} "
          f"{'Strength':<10} {'Tradeable':>10} {'Hostile':>8}  Status")
    print(f"  {'-'*80}")
    for w in windows:
        if not w["passes"] and not args.show_excluded:
            continue
        status = ("PASS" if w["passes"]
                  else f"EXCLUDED (tradeable={w['bull_recov_pct']:.0f}% "
                       f"hostile={w['hostile_pct']:.0f}%)")
        print(f"  {w['start']:<12} {w['end']:<12} {w['days']:>5.1f} "
              f"{w['gain_pct']:>+6.1f}%  {w['strength']:<10} "
              f"{w['bull_recov_pct']:>8.0f}%  {w['hostile_pct']:>6.0f}%   {status}")

    print(f"\n  QUALIFYING windows:")
    for w in passed:
        print(f"    {w['start']} -> {w['end']}  {w['days']:.0f}d  "
              f"{w['gain_pct']:+.0f}%  {w['strength']}  "
              f"[tradeable={w['bull_recov_pct']:.0f}% hostile={w['hostile_pct']:.0f}%]")

    code = format_output(windows)
    with open(args.output, "w") as f:
        f.write(code)
    print(f"\n  Written -> {args.output}")
    print(f"  Replace trend_windows_4yr.py with this file "
          f"(or import TREND_WINDOWS from it).")


if __name__ == "__main__":
    main()
