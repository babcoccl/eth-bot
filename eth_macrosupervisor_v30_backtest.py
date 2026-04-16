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
drop from the last SUBSTANTIAL pre-crash peak to the minimum close
between that peak and this BULL entry.

"Substantial" means a contiguous BULL/RANGE block of >= MIN_PEAK_BARS bars
(default 48 = 2 days on 1h data). This filters out brief RANGE blips that
the regime classifier emits during bear-market consolidation.

BULL class definitions
----------------------
  DEEP    : cycle_trough_pct <= -30%
  MID     : cycle_trough_pct in (-30%, -15%]
  SHALLOW : cycle_trough_pct >  -15%  (or no qualifying peak found)

Per-class stop-loss (default behavior)
---------------------------------------
By default, each trade uses a class-specific hard stop:
  DEEP    : -20%  (large recovery swings; wider noise tolerance)
  MID     : -15%
  SHALLOW : -10%
Pass --no-stop-loss-by-class to use a single flat stop (--stop-loss).

History
-------
  v30.1 : Initial per-class stops (DEEP=20%, MID=12%, SHALLOW=10%)
  v30.2 : Raised SHALLOW stop from 10% -> 15% after backtest showed -37.3pp
           regression. SHALLOW trades need -10% to -14% drawdown room before
           recovering. Only 1 stop fired under flat 15%; 10 fired at 10%.
  v30.3 : Made stop-loss-by-class the default. MID reset to 15%, SHALLOW=10%.
           Signature: run_backtest(h1_df, sup, stop_loss=None,
           stop_loss_by_class=True, debug=False)

Outputs
-------
  eth_backtest_v30_trades.csv   -- one row per trade (includes stop_loss_used)
  eth_backtest_v30_summary.csv  -- aggregate stats by BULL class

Usage
-----
  python eth_macrosupervisor_v30_backtest.py
  python eth_macrosupervisor_v30_backtest.py --start 2021-01-01 --end 2026-04-16
  python eth_macrosupervisor_v30_backtest.py --stop-loss 0.12 --no-stop-loss-by-class
  python eth_macrosupervisor_v30_backtest.py --no-stop-loss-by-class --stop-loss 0.15
  python eth_macrosupervisor_v30_backtest.py --out ./backtest_out/
  python eth_macrosupervisor_v30_backtest.py --debug
"""

from __future__ import annotations
import argparse, os, sys, tempfile
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Dict

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

PAUSE_REGIMES = {"CRASH", "CORRECTION", "RECOVERY"}
PEAK_REGIMES  = {"BULL", "RANGE"}

# Minimum contiguous BULL/RANGE bars to qualify as a real peak segment.
# 48 bars = 2 days on 1h data. Filters out brief RANGE blips.
MIN_PEAK_BARS = 48

# Per-class stop-loss — default behavior when stop_loss_by_class=True.
STOP_LOSS_BY_CLASS: Dict[str, float] = {
    "DEEP":    0.20,   # 20% drawdown from peak
    "MID":     0.15,   # 15%
    "SHALLOW": 0.10,   # 10%
}


def classify_bull_depth(cycle_trough_pct: float) -> str:
    dd = cycle_trough_pct / 100.0
    if dd <= DEEP_THRESHOLD:
        return "DEEP"
    if dd > SHALLOW_THRESHOLD:
        return "SHALLOW"
    return "MID"


def _cycle_trough_pct(
    regime_arr,
    close_arr,
    entry_idx: int,
    min_peak_bars: int = MIN_PEAK_BARS,
    debug: bool = False,
) -> float:
    """
    Walk backwards to find the last CONTIGUOUS BULL/RANGE block with
    length >= min_peak_bars (default 48 = 2 days).

      ref_price     = max(close[block_start : block_end+1])   # peak of block
      trough_window = close[block_end+1   : entry_idx]        # gap after block

    If trough_window is empty (block ends at entry-1), this is a direct
    RANGE->BULL transition with no crash in between -> trough = 0.0 (SHALLOW).
    Short blip blocks are skipped because MIN_PEAK_BARS=48 filters them out.
    """
    j = entry_idx - 1

    while j >= 0:
        # Skip non-peak bars
        while j >= 0 and str(regime_arr[j]) not in PEAK_REGIMES:
            j -= 1
        if j < 0:
            break

        # Measure contiguous block
        block_end = j
        while j >= 0 and str(regime_arr[j]) in PEAK_REGIMES:
            j -= 1
        block_start = j + 1
        block_len   = block_end - block_start + 1

        if block_len >= min_peak_bars:
            ref_price     = float(np.max(close_arr[block_start : block_end + 1]))
            trough_window = close_arr[block_end + 1 : entry_idx]   # gap only

            if len(trough_window) == 0 or ref_price <= 0:
                trough = 0.0
            else:
                trough = round((float(np.min(trough_window)) / ref_price - 1.0) * 100, 2)

            if debug:
                print(
                    f"  [trough] entry_idx={entry_idx} "
                    f"block=[{block_start},{block_end}] block_len={block_len} "
                    f"ref_price={ref_price:.2f} "
                    f"trough={trough:.1f}%"
                )
            return trough

    if debug:
        print(f"  [trough] entry_idx={entry_idx}: no qualifying peak block -> 0.0")
    return 0.0


# ---------------------------------------------------------------------------
# Core backtest engine
# ---------------------------------------------------------------------------

FEE_RATE = 0.001   # 0.1% per side (Coinbase Advanced)


def run_backtest(
    h1_df: pd.DataFrame,
    sup,
    stop_loss: Optional[float] = None,
    stop_loss_by_class: bool = True,
    debug: bool = False,
) -> pd.DataFrame:
    """
    Walk h1_df bar by bar.
    Entry  : first bar of each new BULL segment.
    Exit   : first CRASH signal, hard stop from peak, or end of data.

    stop_loss_by_class : if True (default), each trade uses the class-specific
                         stop from STOP_LOSS_BY_CLASS (DEEP/MID/SHALLOW).
                         If False, uses stop_loss (flat); defaults to 0.15 if
                         stop_loss is None.
    """
    regime_arr = sup._h1_r5_series.values
    close_arr  = h1_df["close"].values
    ts_arr     = h1_df["ts"].values
    n          = len(regime_arr)

    assert len(regime_arr) == len(close_arr), (
        f"Alignment error: regime_arr len={len(regime_arr)} "
        f"close_arr len={len(close_arr)}. Ensure h1_df and "
        f"sup._h1_r5_series are built from the same sorted dataframe."
    )

    trades: List[dict] = []

    in_trade          = False
    entry_bar         = None
    entry_price       = None
    cycle_trough      = None
    peak_since_entry  = None
    active_stop_loss  = stop_loss if stop_loss is not None else 0.15

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
            cycle_trough     = _cycle_trough_pct(
                regime_arr, close_arr, i, debug=debug
            )
            peak_since_entry = close

            bull_class_now = classify_bull_depth(cycle_trough)

            # Determine stop threshold for this trade
            if stop_loss_by_class:
                active_stop_loss = STOP_LOSS_BY_CLASS[bull_class_now]
            else:
                active_stop_loss = stop_loss if stop_loss is not None else 0.15

            if debug:
                print(
                    f"  [entry] bar={i} ts={ts} "
                    f"price={close:.2f} trough={cycle_trough:.1f}% "
                    f"class={bull_class_now} "
                    f"stop={active_stop_loss*100:.0f}%"
                )

        # ---- while in trade: check exits ----
        if in_trade:
            peak_since_entry = max(peak_since_entry, close)
            dd_from_peak     = (close - peak_since_entry) / peak_since_entry

            exit_reason = None

            if dd_from_peak <= -active_stop_loss:
                exit_reason = f"stop_loss_{active_stop_loss*100:.0f}pct"
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
                    "stop_loss_used":       round(active_stop_loss * 100, 1),
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
                active_stop_loss = stop_loss if stop_loss is not None else 0.15

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
            "mean_stop_loss_used":   round(grp["stop_loss_used"].mean(), 1) if "stop_loss_used" in grp.columns else None,
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
          f"{'med_days':>9} {'trough%':>8} {'stop%':>6}")
    print("  " + "-" * 70)
    for _, row in summary_df.iterrows():
        stop_str = f"{row['mean_stop_loss_used']:>5.0f}%" if row.get('mean_stop_loss_used') is not None else "  n/a"
        print(f"  {row['bull_class']:<10} "
              f"{int(row['trades']):>6} "
              f"{row['win_rate_pct']:>5.0f}% "
              f"{row['mean_net_ret_pct']:>+7.1f}% "
              f"{row['median_net_ret_pct']:>+7.1f}% "
              f"{row['total_net_ret_pct']:>+8.1f}% "
              f"{row['median_days_held']:>9.1f} "
              f"{row['mean_cycle_trough_pct']:>+8.1f}% "
              f"{stop_str}")
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
    ap.add_argument("--stop-loss", type=float, default=None,
                    help="Single flat stop-loss (overrides --stop-loss-by-class). "
                         "e.g. 0.15 = 15%%. Only used when --no-stop-loss-by-class is set.")
    ap.add_argument("--no-stop-loss-by-class", action="store_true",
                    help="Disable per-class stop-loss; requires --stop-loss.")
    ap.add_argument("--min-dwell", type=int,   default=None)
    ap.add_argument("--min-peak-bars", type=int, default=MIN_PEAK_BARS,
                    help="Min contiguous BULL/RANGE bars to qualify as a real peak "
                         "(default 48 = 2 days on 1h data)")
    ap.add_argument("--out",       default=".")
    ap.add_argument("--debug",     action="store_true",
                    help="Print trough block details and stop-loss for every entry")
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

    stop_loss_by_class = not args.no_stop_loss_by_class

    print(f"Fetching data {args.start} -> {end_s} ...")
    df1h = fetch_ohlcv(args.symbol, "1h", start_dt, end_dt)
    df5  = fetch_ohlcv(args.symbol, "5m", start_dt, end_dt)
    print(f"  1h bars: {len(df1h):,}   5m bars: {len(df5):,}")

    if stop_loss_by_class:
        print(f"Running backtest (stop_loss=BY_CLASS DEEP={STOP_LOSS_BY_CLASS['DEEP']*100:.0f}% "
              f"MID={STOP_LOSS_BY_CLASS['MID']*100:.0f}% "
              f"SHALLOW={STOP_LOSS_BY_CLASS['SHALLOW']*100:.0f}%, "
              f"min_peak_bars={args.min_peak_bars}) ...")
    else:
        flat = args.stop_loss if args.stop_loss is not None else 0.15
        print(f"Running backtest (stop_loss={flat*100:.0f}%, "
              f"min_peak_bars={args.min_peak_bars}) ...")

    kwargs = {}
    if args.min_dwell is not None:
        kwargs["regime5_min_dwell_bars"] = args.min_dwell

    with _TempDB() as db_path:
        sup = MacroSupervisor(db_path=db_path, **kwargs)
        sup.apply_to_df(df5.iloc[:1].copy(), df1h.copy())

        h_ref = df1h.copy().sort_values("ts").reset_index(drop=True)

        import eth_macrosupervisor_v30_backtest as _self
        _self.MIN_PEAK_BARS = args.min_peak_bars

        trades_df = run_backtest(
            h_ref, sup,
            stop_loss=args.stop_loss,
            stop_loss_by_class=stop_loss_by_class,
            debug=args.debug,
        )

    if not trades_df.empty:
        print(f"  Trough distribution:")
        print(f"    DEEP    (<=-30%): {(trades_df.cycle_trough_pct <= -30).sum()}")
        print(f"    MID  (-30%,-15%]: "
              f"{((trades_df.cycle_trough_pct > -30) & (trades_df.cycle_trough_pct <= -15)).sum()}")
        print(f"    SHALLOW  (>-15%): {(trades_df.cycle_trough_pct > -15).sum()}")
        print(f"    trough range: {trades_df.cycle_trough_pct.min():.1f}% "
              f"to {trades_df.cycle_trough_pct.max():.1f}%")
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
