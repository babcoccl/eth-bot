#!/usr/bin/env python3
"""
eth_bull_classifier.py  —  BULL depth classification (shared constants & logic)
================================================================================
Single source of truth for BULL entry depth classes used across:
  - eth_macrosupervisor_v30_backtest.py  (historical backtest)
  - eth_helpers.py                        (live bull_class_h1 column)
  - eth_trendbot_v1.py                    (per-class PSL via STOP_LOSS_BY_CLASS)

BULL class definitions
----------------------
  DEEP                : cycle_trough_pct <= -30%
                        Large-cycle recovery. Highest conviction, widest stop.
  SHALLOW_RECOV_DEEP  : cycle_trough_pct in (-30%, -13%]
                        Mid-cycle recovery. Historically 0% win rate (5/5 losses).
                        SKIPPED at entry in backtest and live trading.
  SHALLOW_RECOV_LIGHT : cycle_trough_pct in (-13%, 0%)
                        Small dip recovery. Historically winners. Active.
  SHALLOW_CONT        : cycle_trough_pct == 0.0
                        Direct RANGE->BULL continuation, no crash preceded.
                        Active.

Threshold constants
-------------------
  DEEP_THRESHOLD         = -0.30   (-30%)
  SHALLOW_RECOV_CUTOFF   = -0.13   (-13%)  boundary between RECOV_DEEP and RECOV_LIGHT

Algorithm
---------
  _cycle_trough_pct() walks backwards from a BULL entry bar in an h1 regime
  array to find the last qualifying BULL/RANGE peak block (>= MIN_PEAK_BARS),
  takes the max close of that block as the reference price, then finds the
  minimum close between that block and the entry bar.
  Returns (trough_pct, ref_bar_idx, trough_block_start).
  trough_pct == 0.0 means no qualifying peak block was found — SHALLOW_CONT.

History
-------
  v1 : DEEP / MID / SHALLOW  (v30.1–v30.3 era)
  v2 : Split SHALLOW -> SHALLOW_CONT / SHALLOW_RECOV  (v30.4)
  v3 : Merged MID into SHALLOW_RECOV  (v30.5, n=2 historically, both losses)
  v4 : Split SHALLOW_RECOV -> SHALLOW_RECOV_LIGHT / SHALLOW_RECOV_DEEP (v30.7)
       Cutoff: -13%. RECOV_DEEP suppressed at entry — all 5 historical trades losses.
"""

from __future__ import annotations
import numpy as np

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

DEEP_THRESHOLD       = -0.30   # cycle_trough_pct <= -30% -> DEEP
SHALLOW_RECOV_CUTOFF = -0.13   # cycle_trough_pct <= -13% -> SHALLOW_RECOV_DEEP

MIN_PEAK_BARS = 48              # min contiguous BULL/RANGE bars to qualify as a real peak

PEAK_REGIMES  = {"BULL", "RANGE"}
PAUSE_REGIMES = {"CRASH", "CORRECTION", "RECOVERY"}

# ---------------------------------------------------------------------------
# Per-class stop-loss thresholds
# NOTE: These are calibrated for the MacroSupervisor h1 backtest (multi-week
# BULL segments). TrendBot (5m intraday) uses its own stop_loss_by_class
# dict inside its preset — do NOT use these values for TrendBot.
# ---------------------------------------------------------------------------

STOP_LOSS_BY_CLASS: dict[str, float] = {
    "DEEP":                0.20,   # 20% — large recovery, wide noise tolerance
    "SHALLOW_RECOV_LIGHT": 0.10,   # 10%
    "SHALLOW_RECOV_DEEP":  0.10,   # 10% — entry skipped anyway; kept for completeness
    "SHALLOW_CONT":        0.10,   # 10%
}

# ---------------------------------------------------------------------------
# classify_bull_depth()
# ---------------------------------------------------------------------------

def classify_bull_depth(cycle_trough_pct: float) -> str:
    """
    Map a cycle trough percentage to a BULL depth class.

    Parameters
    ----------
    cycle_trough_pct : float
        Percentage drop from the reference peak to the trough, e.g. -35.2.
        Value of 0.0 means no qualifying crash preceded this BULL entry
        (direct RANGE->BULL continuation).

    Returns
    -------
    str : one of DEEP | SHALLOW_RECOV_DEEP | SHALLOW_RECOV_LIGHT | SHALLOW_CONT
    """
    dd = cycle_trough_pct / 100.0  # convert to fraction

    if dd <= DEEP_THRESHOLD:
        return "DEEP"

    if dd == 0.0:
        return "SHALLOW_CONT"          # no qualifying crash — direct continuation

    if dd <= SHALLOW_RECOV_CUTOFF:
        return "SHALLOW_RECOV_DEEP"    # (-30%, -13%] — historically all losses, skipped

    return "SHALLOW_RECOV_LIGHT"       # (-13%, 0%) — small dip recovery, winners


# ---------------------------------------------------------------------------
# _cycle_trough_pct()
# ---------------------------------------------------------------------------

def _cycle_trough_pct(
    regime_arr,
    close_arr,
    entry_idx: int,
    min_peak_bars: int = MIN_PEAK_BARS,
    debug: bool = False,
) -> tuple[float, int, int]:
    """
    Walk backwards from entry_idx to find the last qualifying BULL/RANGE
    peak block (>= min_peak_bars bars), take its max close as the reference
    price, then find the minimum close between that block and entry_idx.

    Parameters
    ----------
    regime_arr  : array-like of str regime labels (aligned with close_arr)
    close_arr   : array-like of float close prices
    entry_idx   : int  — bar index of the BULL entry (first bar of new BULL segment)
    min_peak_bars : int — minimum contiguous BULL/RANGE bars to qualify as a real peak
    debug       : bool — print trough block details

    Returns
    -------
    (trough_pct, ref_bar_idx, trough_block_start)
      trough_pct        : float — percentage drop from ref_price to trough_price
                          0.0 if no qualifying block found (SHALLOW_CONT)
      ref_bar_idx       : int   — bar of max close within the peak block (-1 if none)
      trough_block_start: int   — start bar of the qualifying peak block (-1 if none)
    """
    n = entry_idx

    # Walk back to find end of last PEAK_REGIMES block
    peak_end = n - 1
    while peak_end >= 0 and str(regime_arr[peak_end]) not in PEAK_REGIMES:
        peak_end -= 1

    if peak_end < 0:
        return 0.0, -1, -1

    # Walk back to find start of that peak block
    peak_start = peak_end
    while peak_start > 0 and str(regime_arr[peak_start - 1]) in PEAK_REGIMES:
        peak_start -= 1

    block_len = peak_end - peak_start + 1
    if block_len < min_peak_bars:
        if debug:
            print(f"    [trough] block too short: {block_len} < {min_peak_bars} "
                  f"bars at [{peak_start},{peak_end}] -> SHALLOW_CONT")
        return 0.0, -1, -1

    ref_price = float(np.max(close_arr[peak_start:peak_end + 1]))
    ref_bar   = int(np.argmax(close_arr[peak_start:peak_end + 1])) + peak_start

    # Find trough between peak block end and entry bar
    search_start = peak_end + 1
    search_end   = n  # exclusive
    if search_start >= search_end:
        return 0.0, ref_bar, peak_start

    trough_price = float(np.min(close_arr[search_start:search_end]))
    trough_pct   = (trough_price - ref_price) / ref_price * 100.0

    if debug:
        print(f"    [trough] block=[{peak_start},{peak_end}] len={block_len} "
              f"ref={ref_price:.2f}@bar{ref_bar} "
              f"trough={trough_price:.2f} ({trough_pct:.1f}%)")

    return round(trough_pct, 4), ref_bar, peak_start
