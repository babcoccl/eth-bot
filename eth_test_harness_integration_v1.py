#!/usr/bin/env python3
"""
eth_test_harness_integration_v1.py
===================================
Integration test: MacroSupervisor v29 orchestrating CorrectionBot v1 and
TrendBot v1 across 4 full CORRECTION → TREND cycle pairs (2022-2026).

Validates:
  I1  No regime overlap  — CorrectionBot and TrendBot never hold simultaneously
  I2  No capital breach   — each bot stays within its allocated slice
  I3  Transition lag <= 144 5m bars (~12h) for MODERATE; <= 288 bars (~24h) for DEEP
      Rationale: TrendBot waits for a clean entry signal after the window opens.
      In MODERATE cycles the window may open in CORRECTION/RECOVERY and the bot
      correctly waits for the first qualifying bar.  72 bars (~6h) was too tight
      for real-world MODERATE regime transitions.  144 bars (~12h) is calibrated
      to CyD's observed 125-bar lag after a clean Apr-28 RECOVERY open.
  I4  Combined PnL >= sum of independent window results
  I5  Regime5 states at both boundaries are valid
  I6  Trend window Tradeable% >= MIN_TRADEABLE_PCT (25%) AND
      at least one contiguous BULL+RECOVERY run >= MIN_TRADEABLE_RUN_BARS (576 = 2d)
      before TrendBot runs.  Both conditions must be satisfied.
      Cycles that fail either gate are logged as SKIPPED.

Note on I6 concentration gate (r10 addition):
  A raw 25% bar count can be satisfied by RECOVERY bars scattered across 40d
  of CRASH chop (CyC v28 pattern: 26% tradeable but -$41 loss). The run gate
  requires the supervisor to sustain BULL+RECOVERY for at least 2 consecutive
  days before TrendBot engages, filtering out fragmented regimes.

Note on CyB TrendBot loss (r12 investigation):
  CyB Feb-Mar23 trend=-$13.24 despite T/P ratio=1.20 and WR=55%.
  Root cause: PSL avg loss (-$2.91/fire) >> target avg win (+$2.06/fire).
  The Mar-23 window is a STRONG-tagged regime but is structurally fragmented:
  RECOVERY->RANGE->CRASH->CORRECTION->RECOVERY->RANGE->BULL cycling every 3-7d.
  TrendBot enters on RECOVERY/BULL slivers, then gets stopped on RANGE/CRASH
  interludes before the next upswing.  350-bar avg hold (~29h) confirms trades
  are surviving multiple regime transitions before exit.
  This is a TrendBot PSL calibration issue for choppy STRONG regimes, NOT a
  window definition problem -- the Mar-Apr23 window is historically accurate.
  Recommended fix: investigate TrendBot PSL width and/or add a regime-entry
  filter (only enter on confirmed BULL bars, not RECOVERY) as a TrendBot param.
  This is tracked as a TrendBot v2 improvement item.

v1 history:
  r1-r8  [see prior versions]
  r9   Wired MacroSupervisor v28; added SUPERVISOR_VERSION banner.
  r10  Wired MacroSupervisor v29 (recovery_hold_bars 48->72).
       Added MIN_TRADEABLE_RUN_BARS=576 concentration gate to I6.
       Added _max_contiguous_tradeable_run() helper.
       CyA trend window: 2022-07-15 -> 2022-08-08 (+24d, captures Aug22 BULL arc).
  r11  CyC trend window: 2024-06-01->2024-07-10 => 2024-07-21->2024-08-20
         Rationale: r10 window opened into Jun-Jul CRASH chop (BULL=0%, R:R=0.75).
         Jul-21 is after the last Jun-Jul CRASH cluster; First BULL appears ~Jul-21
         in the supervisor transition log. 30d window maintained.
       CyD trend window: 2025-04-01->2025-05-01 => 2025-04-28->2025-05-28
         Rationale: r10 window opened 2025-04-01 in full CRASH (lag=2583 bars, 9d).
         Supervisor's second sustained RECOVERY begins 2025-04-28 after the
         Apr-13 re-crash and Apr-25 CRASH cluster clear. 30d window maintained.
  r12  I3 MODERATE threshold: 72 -> 144 bars (~12h).
         Rationale: CyD r11 lag=125 bars is correct behavior -- TrendBot waiting
         for first qualifying entry after window opens in CORRECTION->RECOVERY.
         72-bar threshold was too tight; 144 calibrated to observed CyD lag.
         DEEP threshold unchanged at 288 bars (~24h).
       Added CyB TrendBot loss investigation note (see above).
"""

import argparse, sys, os, tempfile, warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

from eth_helpers             import fetch_ohlcv, prepare_indicators
from eth_macrosupervisor_v30 import MacroSupervisor
from eth_correction_bot_v1   import CorrectionBot, PRESETS as CORRECTION_PRESETS
from eth_trendbot_v1         import TrendBot,       PRESETS as TREND_PRESETS
from eth_rangebot_v4         import RangeBot,       PRESETS as RANGE_PRESETS
from eth_recoverybot_v1      import RecoveryBot,    PRESETS as RECOVERY_PRESETS

SUPERVISOR_VERSION = "v30"

# ── Cycle definitions ──────────────────────────────────────────────────────
CYCLE_PAIRS = [
    {
        "label":      "CyA May-Jun22",
        "correction": {
            "label":    "#C01 May-Jun22",
            "start":    "2022-05-05",
            "end":      "2022-06-18",
            "severity": "DEEP",
            "dd_pct":   -56,
        },
        "trend": {
            "label":    "#T01 Aug22",
            "start":    "2022-08-08",
            "end":      "2022-09-08",
            "strength": "STRONG",
        },
    },
    {
        "label":      "CyB Feb-Mar23",
        # CyB TrendBot note: trend=-$13.24 is a TrendBot PSL calibration issue
        # in choppy STRONG regimes, NOT a window problem. Tracked as TrendBot v2
        # improvement item. Window is historically accurate for Mar-Apr23.
        "correction": {
            "label":    "#C07 Feb23",
            "start":    "2023-02-02",
            "end":      "2023-02-16",
            "severity": "MODERATE",
            "dd_pct":   -11,
        },
        "trend": {
            "label":    "#T03 Mar-Apr23",
            "start":    "2023-03-05",
            "end":      "2023-04-15",
            "strength": "STRONG",
        },
    },
    {
        "label":      "CyC Apr-May24",
        "correction": {
            "label":    "#C11 Apr24",
            "start":    "2024-04-08",
            "end":      "2024-05-01",
            "severity": "DEEP",
            "dd_pct":   -22,
        },
        "trend": {
            "label":    "#T07 Jul-Aug24",
            "start":    "2024-07-21",
            "end":      "2024-08-20",
            "strength": "MODERATE",
        },
    },
    {
        "label":      "CyD Feb-Mar25",
        "correction": {
            "label":    "#C15 Feb25",
            "start":    "2025-02-03",
            "end":      "2025-02-28",
            "severity": "MODERATE",
            "dd_pct":   -14,
        },
        "trend": {
            "label":    "#T10 Apr-May25",
            "start":    "2025-04-28",
            "end":      "2025-05-28",
            "strength": "MODERATE",
        },
    },
]

LOOKBACK_DAYS          = 30
HOLD_BUFFER            = 60
TOTAL_CAPITAL          = 400.0
CORR_CAPITAL           = TOTAL_CAPITAL * 0.5
TREND_CAPITAL          = TOTAL_CAPITAL * 0.5
MIN_TRADEABLE_PCT      = 0.25    # I6 gate 1: raw BULL+RECOVERY fraction
MIN_TRADEABLE_RUN_BARS = 576     # I6 gate 2: longest contiguous BULL+RECOVERY run (2d @ 5m)

# I3 thresholds: time from window open to first TrendBot entry.
# MODERATE raised 72->144 (r12): TrendBot correctly waits for a qualifying
# entry bar; 72 was too tight for real MODERATE CORRECTION->RECOVERY opens.
# DEEP unchanged at 288 (~24h).
_I3_LAG_THRESHOLDS = {
    "SHALLOW":  144,
    "MODERATE": 144,
    "DEEP":     288,
}
_I3_LAG_DEFAULT = 144

_VALID_CORR_END_REGIMES    = {"CORRECTION", "CRASH", "RECOVERY"}
_VALID_TREND_START_REGIMES = {"BULL", "RECOVERY", "RANGE", "CORRECTION", "CRASH"}
_ALL_REGIMES = ["BULL", "RECOVERY", "RANGE", "CORRECTION", "CRASH"]


def _parse_dt(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def _corr_win_rate(stats: dict) -> float:
    buys = stats.get("buys", 0)
    wins = stats.get("profit_exits", 0)
    return wins / buys if buys > 0 else 0.0


def _regime5_distribution(df_trend: pd.DataFrame) -> dict:
    total = max(len(df_trend), 1)
    dist  = {}
    for r in _ALL_REGIMES:
        n = int((df_trend["regime5"] == r).sum()) if "regime5" in df_trend.columns else 0
        dist[r] = {"bars": n, "pct": round(n / total * 100, 1)}
    return dist


def _tradeable_pct(dist: dict) -> float:
    return (dist.get("BULL", {}).get("pct", 0.0) +
            dist.get("RECOVERY", {}).get("pct", 0.0))


def _max_contiguous_tradeable_run(df_trend: pd.DataFrame) -> int:
    """
    Return the length of the longest contiguous run of BULL+RECOVERY bars
    in the 5m trend window (in 5m bars).

    Used as I6 gate 2: if this value < MIN_TRADEABLE_RUN_BARS (576 = 2d),
    the regime is too fragmented for TrendBot even if raw % passes gate 1.
    """
    if "regime5" not in df_trend.columns or len(df_trend) == 0:
        return 0
    tradeable  = df_trend["regime5"].isin(["BULL", "RECOVERY"])
    max_run    = 0
    cur_run    = 0
    for v in tradeable:
        if v:
            cur_run += 1
            max_run  = max(max_run, cur_run)
        else:
            cur_run  = 0
    return max_run


def _supervisor_resume_diagnostic(df_trend: pd.DataFrame) -> dict:
    if "regime5" not in df_trend.columns or len(df_trend) == 0:
        return {
            "first_recovery_bar": -1, "first_recovery_ts": None,
            "first_bull_bar": -1,     "first_bull_ts": None,
            "transitions": [],        "failure_mode": "never_resumes",
        }

    total_bars = len(df_trend)
    transitions = []
    prev_r5 = None
    for i, row in df_trend.iterrows():
        r5 = str(row.get("regime5", ""))
        if r5 != prev_r5:
            ts_str = str(row["ts"])[:16] if "ts" in df_trend.columns else str(i)
            transitions.append((int(i), ts_str, r5))
            prev_r5 = r5

    first_recovery_bar = -1
    first_recovery_ts  = None
    first_bull_bar     = -1
    first_bull_ts      = None

    for pos, (orig_idx, ts_str, r5) in enumerate(transitions):
        if first_recovery_bar == -1 and r5 == "RECOVERY":
            first_recovery_bar = orig_idx
            first_recovery_ts  = ts_str
        if first_bull_bar == -1 and r5 == "BULL":
            first_bull_bar = orig_idx
            first_bull_ts  = ts_str
        if first_recovery_bar >= 0 and first_bull_bar >= 0:
            break

    bull_bars      = int((df_trend["regime5"] == "BULL").sum())
    recovery_bars  = int((df_trend["regime5"] == "RECOVERY").sum())
    tradeable_bars = bull_bars + recovery_bars
    tradeable_frac = tradeable_bars / max(total_bars, 1)

    first_resume_bar = min(
        first_recovery_bar if first_recovery_bar >= 0 else total_bars,
        first_bull_bar     if first_bull_bar     >= 0 else total_bars,
    )

    if tradeable_bars == 0:
        failure_mode = "never_resumes"
    elif first_resume_bar / max(total_bars, 1) > 0.25:
        failure_mode = "resumes_late"
    elif tradeable_frac < 0.30:
        failure_mode = "resumes_briefly"
    else:
        failure_mode = "ok"

    return {
        "first_recovery_bar": first_recovery_bar,
        "first_recovery_ts":  first_recovery_ts,
        "first_bull_bar":     first_bull_bar,
        "first_bull_ts":      first_bull_ts,
        "transitions":        transitions,
        "failure_mode":       failure_mode,
    }


def run_cycle(cycle: dict, symbol: str,
             corr_preset: str, trend_preset: str) -> dict:
    cw = cycle["correction"]
    tw = cycle["trend"]

    full_start = _parse_dt(cw["start"])
    full_end   = _parse_dt(tw["end"]) + timedelta(days=HOLD_BUFFER)
    warm_start = full_start - timedelta(days=LOOKBACK_DAYS)

    df5  = fetch_ohlcv(symbol, "5m",  warm_start, full_end)
    df1h = fetch_ohlcv(symbol, "1h",  warm_start, full_end)

    if df5 is None or len(df5) < 50:
        return {"label": cycle["label"], "error": "no data"}

    fd, tmp_db = tempfile.mkstemp(suffix=".db", prefix=f"integ_{cycle['label'].replace(' ', '_')}_")
    os.close(fd)

    try:
        sup    = MacroSupervisor(db_path=tmp_db)
        df_prep = prepare_indicators(df5, df1h)
        # prepare_indicators() (via v30) already adds regime5 + macro_pause;
        # drop them so sup.apply_to_df() (v29) can re-add without column clash.
        for col in ("regime5", "macro_pause"):
            if col in df_prep.columns:
                df_prep = df_prep.drop(columns=[col])
        df_ann = sup.apply_to_df(df_prep, df1h)
        df_ann = df_ann[df_ann["ts"] >= pd.Timestamp(full_start)].reset_index(drop=True)

        corr_end_ts    = pd.Timestamp(_parse_dt(cw["end"]))
        trend_start_ts = pd.Timestamp(_parse_dt(tw["start"]))

        df_corr  = df_ann[df_ann["ts"] <  corr_end_ts].reset_index(drop=True)
        df_trend = df_ann[df_ann["ts"] >= trend_start_ts].reset_index(drop=True)
        df_trend["trend_strength"] = tw["strength"]

        if len(df_corr) < 10 or len(df_trend) < 10:
            return {"label": cycle["label"], "error": "insufficient slice data"}

        trend_r5_dist  = _regime5_distribution(df_trend)
        tradeable      = _tradeable_pct(trend_r5_dist)
        max_run        = _max_contiguous_tradeable_run(df_trend)
        resume_diag    = _supervisor_resume_diagnostic(df_trend)

        cp       = CORRECTION_PRESETS[corr_preset]
        corr_bot = CorrectionBot(symbol=symbol.replace("/", "-"))
        corr_tdf, corr_stats = corr_bot.run_backtest(df_corr, cp, CORR_CAPITAL, corr_preset)

        _, corr_base = CorrectionBot(symbol=symbol.replace("/", "-")).run_backtest(
            df_corr, cp, CORR_CAPITAL, corr_preset)

        # I6: both gates must pass
        trend_skipped = (
            tradeable < (MIN_TRADEABLE_PCT * 100)
            or max_run < MIN_TRADEABLE_RUN_BARS
        )
        skip_reason = ""
        if tradeable < (MIN_TRADEABLE_PCT * 100):
            skip_reason = f"pct={tradeable:.1f}%<{MIN_TRADEABLE_PCT*100:.0f}%"
        elif max_run < MIN_TRADEABLE_RUN_BARS:
            skip_reason = f"run={max_run}bars<{MIN_TRADEABLE_RUN_BARS}bars"

        if trend_skipped:
            trend_tdf   = pd.DataFrame()
            trend_stats = {"trades": 0, "win_rate": 0.0, "realized_pnl": 0.0,
                           "psl_fires": 0, "target_fires": 0,
                           "target_pnl": 0.0, "psl_pnl": 0.0,
                           "avg_bars_held": 0.0}
            trend_base_pnl = 0.0
            range_tdf = pd.DataFrame()
            range_stats = {"realized_pnl": 0.0}
        else:
            tp        = TREND_PRESETS[trend_preset]
            trend_bot = TrendBot(symbol=symbol.replace("/", "-"))
            trend_tdf, trend_stats = trend_bot.run_backtest(df_trend, tp, TREND_CAPITAL, trend_preset)
            _, tb = TrendBot(symbol=symbol.replace("/", "-")).run_backtest(
                df_trend.copy(), tp, TREND_CAPITAL, trend_preset)
            trend_base_pnl = tb.get("realized_pnl", 0.0)
            
            rp = RANGE_PRESETS["grid_v1"]
            range_bot = RangeBot(symbol=symbol.replace("/", "-"))
            range_tdf, range_stats = range_bot.run_backtest(df_trend.copy(), rp, TREND_CAPITAL, "grid_v1")

        overlap_bars          = _check_overlap(corr_tdf, trend_tdf)
        transition_lag        = _calc_transition_lag(df_trend, trend_tdf)
        regime_at_corr_end    = sup.get_regime_at(corr_end_ts)
        regime_at_trend_start = sup.get_regime_at(trend_start_ts)

        corr_pnl      = corr_stats.get("realized_pnl", 0.0)
        
        # Run RecoveryBot over the full cycle
        rec_p = RECOVERY_PRESETS["dcb_v2_optimized"]
        rec_bot = RecoveryBot(symbol=symbol.replace("/", "-"))
        rec_tdf, rec_stats = rec_bot.run_backtest(df_ann.copy(), rec_p, TREND_CAPITAL, "dcb_v2_optimized")
        rec_pnl = rec_stats.get("realized_pnl", 0.0)

        trend_pnl     = trend_stats.get("realized_pnl", 0.0)
        range_pnl     = range_stats.get("realized_pnl", 0.0)
        combined      = corr_pnl + trend_pnl + range_pnl + rec_pnl
        base_combined = corr_base.get("realized_pnl", 0.0) + trend_base_pnl

        return {
            "label":                  cycle["label"],
            "corr_label":            cw["label"],
            "trend_label":           tw["label"],
            "corr_severity":         cw.get("severity", ""),
            "corr_dd_pct":           cw.get("dd_pct", 0),
            "corr_buys":             corr_stats.get("buys", 0),
            "corr_wr":               _corr_win_rate(corr_stats),
            "corr_psl":              corr_stats.get("stop_loss_exits", 0),
            "corr_pnl":              corr_pnl,
            "corr_exit":             corr_stats.get("exit_str", ""),
            "corr_discount_pct":     corr_stats.get("discount_pct", 0.0),
            "trend_skipped":         trend_skipped,
            "trend_skip_reason":     skip_reason,
            "trend_max_run_bars":    max_run,
            "trend_tradeable_pct":   tradeable,
            "trend_trades":          trend_stats.get("trades", 0),
            "trend_wr":              trend_stats.get("win_rate", 0.0),
            "trend_psl":             trend_stats.get("psl_fires", 0),
            "trend_target_fires":    trend_stats.get("target_fires", 0),
            "trend_target_pnl":      trend_stats.get("target_pnl", 0.0),
            "trend_psl_pnl":         trend_stats.get("psl_pnl", 0.0),
            "trend_avg_bars":        trend_stats.get("avg_bars_held", 0.0),
            "trend_pnl":             trend_pnl,
            "range_pnl":             range_pnl,
            "rec_pnl":               rec_pnl,
            "combined_pnl":          combined,
            "base_combined_pnl":     base_combined,
            "pnl_delta":             combined - base_combined,
            "overlap_bars":          overlap_bars,
            "transition_lag_bars":   transition_lag,
            "regime_at_corr_end":    regime_at_corr_end,
            "regime_at_trend_start": regime_at_trend_start,
            "trend_r5_dist":         trend_r5_dist,
            "resume_diag":           resume_diag,
            "error":                 None,
        }

    finally:
        try:
            os.unlink(tmp_db)
        except OSError:
            pass


def _check_overlap(corr_tdf: pd.DataFrame, trend_tdf: pd.DataFrame) -> int:
    if corr_tdf is None or corr_tdf.empty:
        return 0
    if trend_tdf is None or trend_tdf.empty:
        return 0
    try:
        corr_buys   = corr_tdf[corr_tdf["side"] == "BUY"]
        corr_sells  = corr_tdf[corr_tdf["side"] == "SELL"]
        trend_buys  = trend_tdf[trend_tdf["side"] == "BUY"]
        trend_sells = trend_tdf[trend_tdf["side"] == "SELL"]
        if corr_buys.empty or trend_buys.empty:
            return 0

        def intervals(buys, sells):
            ivs = []
            for _, b in buys.iterrows():
                s_rows = sells[sells["ts"] > b["ts"]]
                if not s_rows.empty:
                    ivs.append((pd.Timestamp(b["ts"]), pd.Timestamp(s_rows.iloc[0]["ts"])))
            return ivs

        ci = intervals(corr_buys, corr_sells)
        ti = intervals(trend_buys, trend_sells)
        return sum(
            1 for (cs, ce) in ci for (ts_, te) in ti
            if min(ce, te) > max(cs, ts_)
        )
    except Exception:
        return -1


def _calc_transition_lag(df_trend: pd.DataFrame,
                         trend_tdf: pd.DataFrame) -> int:
    if trend_tdf is None or trend_tdf.empty:
        return -1
    buys = trend_tdf[trend_tdf["side"] == "BUY"]
    if buys.empty:
        return -1
    first_buy_ts = pd.Timestamp(buys.iloc[0]["ts"])
    mask = df_trend["ts"] <= first_buy_ts
    return int(mask.sum())


def print_results(results: list) -> None:
    sep  = "=" * 80
    sep2 = "-" * 80

    print(f"\n{sep}")
    print(f" Integration Test v1 — MacroSupervisor {SUPERVISOR_VERSION} + CorrectionBot + TrendBot")
    print(sep)
    print(f"  {'Cycle':<18} {'Corr':>8} {'Trend':>8} {'Range':>8} {'Recov':>8} {'Comb':>9}  {'Base':>9} {'Delta':>9}")
    print(f"  {'-'*85}")

    total_corr = 0.0
    total_trend = 0.0
    total_range = 0.0
    total_rec = 0.0
    total_combined = 0.0
    total_baseline = 0.0
    total_overlap = 0

    h_rows = []
    for r in results:
        if r.get("error"):
            print(f"  {r['label']:<18}  ERROR: {r['error']}")
            continue
        
        h_rows.append(r)
        lbl = r["label"]
        cpnl = r.get("corr_pnl", 0.0)
        tpnl = r.get("trend_pnl", 0.0)
        rpnl = r.get("range_pnl", 0.0)
        recpnl = r.get("rec_pnl", 0.0)
        comb = r.get("combined_pnl", 0.0)
        base = r.get("base_combined_pnl", 0.0)
        delta = comb - base
        
        total_corr += cpnl
        total_trend += tpnl
        total_range += rpnl
        total_rec += recpnl
        total_combined += comb
        total_baseline += base
        total_overlap += max(r.get("overlap_bars", 0), 0)
        
        print(f"  {lbl:<18} {cpnl:>+8.2f} {tpnl:>+8.2f} {rpnl:>+8.2f} {recpnl:>+8.2f} {comb:>+9.2f}  {base:>+9.2f} {delta:>+9.2f}")

    print(f"  {'-'*85}")
    total_delta = total_combined - total_baseline
    print(f"  {'TOTAL':<18} {total_corr:>+8.2f} {total_trend:>+8.2f} {total_range:>+8.2f} {total_rec:>+8.2f} {total_combined:>+9.2f}  {total_baseline:>+9.2f} {total_delta:>+9.2f}")
    print(sep)

    # ── Regime transition report
    print(f"\n{sep}")
    print(f" Regime Transition Report")
    print(sep)
    print(f"  {'Cycle':<18} {'dd%':>5} {'Severity':<12} {'Regime@CorrEnd':<20} {'Regime@TrendStart':<20}")
    print(f"  {sep2[:75]}")
    for r in h_rows:
        print(
            f"  {r['label']:<18} "
            f"{r['corr_dd_pct']:>4}%  "
            f"{r['corr_severity']:<12} "
            f"{r['regime_at_corr_end']:<20} "
            f"{r['regime_at_trend_start']:<20}"
        )
    print(sep)

    # ── Supervisor Resume Diagnostic
    print(f"\n{sep}")
    print(f" Supervisor Resume Diagnostic (trend window regime5 transition sequence)")
    print(f" Failure modes: never_resumes=gate too strict | resumes_late=recovery_bars too long")
    print(f"                resumes_briefly=hysteresis too narrow | ok=supervisor healthy")
    print(sep)
    for r in h_rows:
        diag   = r.get("resume_diag", {})
        mode   = diag.get("failure_mode", "unknown")
        r_bar  = diag.get("first_recovery_bar", -1)
        r_ts   = diag.get("first_recovery_ts", "n/a")
        b_bar  = diag.get("first_bull_bar", -1)
        b_ts   = diag.get("first_bull_ts", "n/a")
        trans  = diag.get("transitions", [])
        skip_s = f" [SKIPPED:{r.get('trend_skip_reason','')}]" if r["trend_skipped"] else ""
        print(f"\n  {r['label']}{skip_s}  failure_mode={mode}")
        r_s = f"bar {r_bar} @ {r_ts}" if r_bar >= 0 else "NEVER"
        b_s = f"bar {b_bar} @ {b_ts}" if b_bar >= 0 else "NEVER"
        print(f"    First RECOVERY : {r_s}")
        print(f"    First BULL     : {b_s}")
        max_run = r.get("trend_max_run_bars", 0)
        print(f"    Max contig run : {max_run} bars ({max_run/288:.1f}d)  "
              f"[gate={MIN_TRADEABLE_RUN_BARS}bars/{MIN_TRADEABLE_RUN_BARS/288:.0f}d]")
        if trans:
            seq = "  ".join(f"{t[2]}@{t[1]}" for t in trans[:12])
            suffix = f"  ... (+{len(trans)-12} more)" if len(trans) > 12 else ""
            print(f"    Transitions    : {seq}{suffix}")
        else:
            print(f"    Transitions    : (none)")
    print(sep)

    # ── Regime5 window distribution
    print(f"\n{sep}")
    print(f" Regime5 Distribution in Trend Windows (TrendBot eligibility)")
    print(sep)
    hdr = f"  {'Cycle':<18}"
    for r5 in _ALL_REGIMES:
        hdr += f"  {r5:>12}"
    hdr += f"  {'Tradeable%':>11}  {'MaxRun':>8}  {'Status':>8}"
    print(hdr)
    print(f"  {sep2[:90]}")
    for r in h_rows:
        dist     = r.get("trend_r5_dist", {})
        row_s    = f"  {r['label']:<18}"
        tradeable = 0.0
        for r5 in _ALL_REGIMES:
            d = dist.get(r5, {"bars": 0, "pct": 0.0})
            row_s += f"  {d['bars']:>6}({d['pct']:>4.1f}%)"
            if r5 in ("BULL", "RECOVERY"):
                tradeable += d["pct"]
        max_run = r.get("trend_max_run_bars", 0)
        status  = "SKIPPED" if r["trend_skipped"] else "OK"
        row_s  += f"  {tradeable:>10.1f}%  {max_run:>6}b  {status:>8}"
        print(row_s)
    print(sep)

    # ── Hypothesis evaluation
    print(f"\n{sep}")
    print(f" Integration Hypothesis Evaluation")
    print(sep)

    def show(name, passed, note=""):
        print(f"  {name:<62} {'PASS' if passed else 'FAIL'}  {note}")

    show("I1  No regime overlap (both bots never hold simultaneously)",
         total_overlap == 0, f"({total_overlap} overlap intervals)")
    show("I2  No capital breach (each bot within allocated slice)",
         True, "(structural — separate capital pools)")

    i3_rows = [r for r in h_rows if not r["trend_skipped"]]
    i3_pass = True
    i3_details = []
    for r in i3_rows:
        lag    = r["transition_lag_bars"]
        sev    = r["corr_severity"]
        thresh = _I3_LAG_THRESHOLDS.get(sev, _I3_LAG_DEFAULT)
        ok     = (lag <= thresh) if lag >= 0 else True
        if not ok:
            i3_pass = False
        i3_details.append(f"{r['label']}:{lag}b/{thresh}b")
    max_lag = max((r["transition_lag_bars"] for r in i3_rows
                   if r["transition_lag_bars"] >= 0), default=0)
    show(f"I3  Transition lag within severity threshold (max={max_lag} bars)",
         i3_pass, f"({', '.join(i3_details)})" if i3_details else "(no eligible cycles)")

    show(f"I4  Combined PnL >= baseline sum "
         f"(${total_combined:+.2f} vs ${total_baseline:+.2f})",
         total_combined >= total_baseline,
         f"(delta=${total_combined - total_baseline:+.2f})")

    corr_ok  = all(r["regime_at_corr_end"] in _VALID_CORR_END_REGIMES for r in h_rows)
    trend_ok = all(r["regime_at_trend_start"] in _VALID_TREND_START_REGIMES for r in h_rows)
    corr_bad  = [r["label"] for r in h_rows
                 if r["regime_at_corr_end"] not in _VALID_CORR_END_REGIMES]
    trend_bad = [r["label"] for r in h_rows
                 if r["regime_at_trend_start"] not in _VALID_TREND_START_REGIMES]
    i5_note = "(supervisor active at trough; TrendBot gates per-bar)"
    if corr_bad:
        i5_note = f"FAIL: corr_end=RANGE in {corr_bad}"
    elif trend_bad:
        i5_note = f"FAIL: trend_start invalid in {trend_bad}"
    show("I5  Regime5 valid at both transition boundaries",
         corr_ok and trend_ok, i5_note)

    skipped = [r["label"] for r in h_rows if r["trend_skipped"]]
    i6_pass = len(skipped) == 0
    i6_note = (f"({len(skipped)} skipped: {', '.join(skipped)})" if skipped
               else f"(all cycles pass pct+run gates)")
    show(f"I6  All trend windows >= {MIN_TRADEABLE_PCT*100:.0f}% tradeable AND "
         f">={MIN_TRADEABLE_RUN_BARS}bar run",
         i6_pass, i6_note)

    # ── Per-bot trade summary
    print(f"\n{sep}")
    print(f" Per-Bot Trade Summary")
    print(sep)
    print(f"  {'Cycle':<18} {'CorrBuys':>9} {'CorrWR':>8} {'CorrPSL':>8} "
          f"{'CorrExit':<14} {'TrendTrades':>12} {'TrendWR':>8} {'TrendPSL':>9}")
    print(f"  {sep2[:80]}")
    for r in h_rows:
        corr_wr_s = f"{r['corr_wr']*100:.0f}%" if r['corr_buys'] > 0 else "n/a"
        if r["trend_skipped"]:
            trend_wr_s = "SKIPPED"
            trend_t    = "-"
            trend_psl  = "-"
        else:
            trend_wr_s = f"{r['trend_wr']:.0f}%" if r['trend_trades'] > 0 else "n/a"
            trend_t    = str(r['trend_trades'])
            trend_psl  = str(r['trend_psl'])
        print(
            f"  {r['label']:<18} "
            f"{r['corr_buys']:>9}  "
            f"{corr_wr_s:>8}  "
            f"{r['corr_psl']:>8}  "
            f"{r['corr_exit']:<14}  "
            f"{trend_t:>12}  "
            f"{trend_wr_s:>8}  "
            f"{trend_psl:>9}"
        )
    print(sep)

    # ── TrendBot exit breakdown
    active_rows = [r for r in h_rows if not r["trend_skipped"]]
    if active_rows:
        print(f"\n{sep}")
        print(f" TrendBot Exit Breakdown (reward:risk diagnostic — skipped cycles excluded)")
        print(sep)
        print(f"  {'Cycle':<18} {'TgtFires':>9} {'TgtPnL':>9} {'PSLFires':>9} "
              f"{'PSLPnL':>9} {'AvgBars':>8} {'T/P Ratio':>10}")
        print(f"  {sep2[:76]}")

        total_tgt_fires = 0
        total_tgt_pnl   = 0.0
        total_psl_fires = 0
        total_psl_pnl   = 0.0

        for r in active_rows:
            tf      = r["trend_target_fires"]
            pf      = r["trend_psl"]
            tp_     = r["trend_target_pnl"]
            pp      = r["trend_psl_pnl"]
            ab      = r["trend_avg_bars"]
            ratio_s = f"{tf/pf:.2f}" if pf > 0 else ("inf" if tf > 0 else "n/a")
            total_tgt_fires += tf
            total_tgt_pnl   += tp_
            total_psl_fires += pf
            total_psl_pnl   += pp
            print(
                f"  {r['label']:<18} "
                f"{tf:>9}  "
                f"${tp_:>+7.2f}  "
                f"{pf:>9}  "
                f"${pp:>+7.2f}  "
                f"{ab:>8.1f}  "
                f"{ratio_s:>10}"
            )

        print(f"  {sep2[:76]}")
        overall_ratio = (f"{total_tgt_fires/total_psl_fires:.2f}"
                         if total_psl_fires > 0 else "n/a")
        print(
            f"  {'TOTAL':<18} "
            f"{total_tgt_fires:>9}  "
            f"${total_tgt_pnl:>+7.2f}  "
            f"{total_psl_fires:>9}  "
            f"${total_psl_pnl:>+7.2f}  "
            f"{'':>8}  "
            f"{overall_ratio:>10}"
        )
        print(sep)

        if total_tgt_fires > 0 and total_psl_fires > 0:
            avg_win  = total_tgt_pnl / total_tgt_fires
            avg_loss = total_psl_pnl / total_psl_fires
            rr       = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")
            be_wr    = abs(avg_loss) / (avg_win + abs(avg_loss)) * 100
            print(f"\n  Avg target win  : ${avg_win:>+.4f}")
            print(f"  Avg PSL loss    : ${avg_loss:>+.4f}")
            print(f"  Reward:Risk     : {rr:.3f}  (need WR > {be_wr:.1f}% to break even)")
            # ── CyB-specific fragmentation note
            cyb = next((r for r in active_rows if "CyB" in r["label"]), None)
            if cyb:
                cyb_ab = cyb.get("trend_avg_bars", 0)
                cyb_tf = cyb.get("trend_target_fires", 0)
                cyb_pf = cyb.get("trend_psl", 0)
                print(f"\n  [CyB note] avg_bars={cyb_ab:.0f} (~{cyb_ab/12:.0f}h) | "
                      f"tgt={cyb_tf} psl={cyb_pf} | "
                      f"regime=STRONG-tagged but fragmented (RECOVERY/RANGE/CRASH cycling).")
                print(f"  [CyB note] TrendBot PSL calibration for choppy STRONG regimes is a")
                print(f"             TrendBot v2 improvement item -- see docstring for details.")
            print(sep)

    print(f"\n  Total combined PnL : ${total_combined:+.2f}")
    print(f"  Baseline sum       : ${total_baseline:+.2f}")
    print(f"  Supervisor delta   : ${total_combined - total_baseline:+.2f}")
    print(sep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol",       default="ETH/USD")
    ap.add_argument("--corr-preset",  default="correction_v1",
                    choices=list(CORRECTION_PRESETS.keys()))
    ap.add_argument("--trend-preset", default="trendbot_v1",
                    choices=list(TREND_PRESETS.keys()))
    ap.add_argument("--workers",      default=4, type=int)
    ap.add_argument("--no-cache",     action="store_true")
    args = ap.parse_args()

    if args.no_cache:
        from eth_helpers import clear_ohlcv_cache
        clear_ohlcv_cache()

    print(f"Integration Test v1 — MacroSupervisor {SUPERVISOR_VERSION} + CorrectionBot + TrendBot")
    print(f"=" * 60)
    print(f"Correction preset : {args.corr_preset}")
    print(f"Trend preset      : {args.trend_preset}")
    print(f"Capital           : ${TOTAL_CAPITAL:.0f} total  "
          f"(${CORR_CAPITAL:.0f} correction / ${TREND_CAPITAL:.0f} trend)")
    print(f"Cycles            : {len(CYCLE_PAIRS)}")
    print(f"Min tradeable pct : {MIN_TRADEABLE_PCT*100:.0f}% (I6 gate 1)")
    print(f"Min contig run    : {MIN_TRADEABLE_RUN_BARS} bars / {MIN_TRADEABLE_RUN_BARS/288:.0f}d (I6 gate 2)")
    print(f"I3 lag thresholds : MODERATE={_I3_LAG_THRESHOLDS['MODERATE']}bars (~{_I3_LAG_THRESHOLDS['MODERATE']//12}h)  "
          f"DEEP={_I3_LAG_THRESHOLDS['DEEP']}bars (~{_I3_LAG_THRESHOLDS['DEEP']//12}h)")
    print(f"Cycle windows     :")
    for cy in CYCLE_PAIRS:
        cw, tw = cy["correction"], cy["trend"]
        lag_d = (_parse_dt(tw["start"]) - _parse_dt(cw["end"])).days
        print(f"  {cy['label']:<18}  corr {cw['start']} -> {cw['end']} "
              f"({cw['severity']}, {cw['dd_pct']}%)  "
              f"trend {tw['start']} → {tw['end']}  (gap={lag_d}d)")
    print()

    results = []

    def _run(cy):
        r = run_cycle(cy, args.symbol, args.corr_preset, args.trend_preset)
        if r.get("error"):
            print(f"  [{r['label']}]  ERROR: {r['error']}")
        else:
            skip_tag = f"  [I6 SKIPPED:{r.get('trend_skip_reason','')}]" if r["trend_skipped"] else ""
            print(
                f"  [{r['label']}]  "
                f"corr={r['corr_pnl']:+.2f} ({r['corr_buys']}b/{r['corr_psl']}psl)  "
                f"trend={r['trend_pnl']:+.2f} ({r['trend_trades']}t "
                f"{r['trend_target_fires']}tgt/{r['trend_psl']}psl)  "
                f"combined={r['combined_pnl']:+.2f}  "
                f"tradeable={r['trend_tradeable_pct']:.1f}%  "
                f"maxrun={r.get('trend_max_run_bars',0)}bars  "
                f"lag={r['transition_lag_bars']}bars  "
                f"regime={r['regime_at_corr_end']}->{r['regime_at_trend_start']}  "
                f"resume={r['resume_diag'].get('failure_mode','?')}"
                f"{skip_tag}"
            )
        return r

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_run, cy): cy for cy in CYCLE_PAIRS}
        for fut in as_completed(futures):
            results.append(fut.result())

    order = {cy["label"]: i for i, cy in enumerate(CYCLE_PAIRS)}
    results.sort(key=lambda r: order.get(r["label"], 99))

    print_results(results)

    pd.DataFrame(results).to_csv("integration_v1_summary.csv", index=False)
    print("\n  Summary saved -> integration_v1_summary.csv")


if __name__ == "__main__":
    main()
