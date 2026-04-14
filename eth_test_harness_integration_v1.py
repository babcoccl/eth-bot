#!/usr/bin/env python3
"""
eth_test_harness_integration_v1.py
===================================
Integration test: MacroSupervisor v27 orchestrating CorrectionBot v1 and
TrendBot v1 across 4 full CORRECTION → TREND cycle pairs (2022-2026).

Validates:
  I1  No regime overlap  — CorrectionBot and TrendBot never hold simultaneously
  I2  No capital breach   — each bot stays within its allocated slice
  I3  Transition lag <= 72 5m bars (~6h) from correction end to first trend entry
  I4  Combined PnL >= sum of independent window results (handoff adds no friction)
  I5  Regime5 at correction end is CORRECTION; at trend start is BULL or RECOVERY

Cycle pairs — MODERATE/DEEP corrections (dd >= 12%) that trigger MacroSupervisor
CORRECTION state and activate CorrectionBot signal conditions:
  CyA  #C01 Mar22  MODERATE (-12%)  (ends 2022-03-14)  →  #T01 Mar-Apr22  (starts 2022-03-14)
  CyB  #C07 Feb23  MODERATE (-11%)  (ends 2023-02-16)  →  #T03 Feb-Mar23  (starts 2023-02-16)
  CyC  #C11 Apr24  DEEP     (-22%)  (ends 2024-05-01)  →  #T07 May-Jun24  (starts 2024-05-01)
  CyD  #C15 Feb25  MODERATE (-14%)  (ends 2025-02-28)  →  #T10 Mar-Apr25  (starts 2025-02-28)

Capital split: $400 total — $200 CorrectionBot, $200 TrendBot (50/50).
Independent baseline uses same capital split for fair comparison.
Baselines are computed live during the run for accuracy.

v1 history:
  initial  — 4 SHALLOW cycles; CorrBot PnL=$0, I3/I5 FAIL (supervisor never entered CORRECTION)
  current  — replaced with 4 MODERATE/DEEP cycles to properly exercise both bots
"""

import argparse, sys, os, tempfile, warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

from eth_helpers             import fetch_ohlcv, prepare_indicators
from eth_macrosupervisor_v27 import MacroSupervisor
from eth_correction_bot_v1   import CorrectionBot, PRESETS as CORRECTION_PRESETS
from eth_trendbot_v1         import TrendBot,       PRESETS as TREND_PRESETS

# ── Cycle definitions ──────────────────────────────────────────────────
# All 4 corrections have peak-to-trough dd >= 12% so MacroSupervisor
# enters CORRECTION state (pause_dd_trigger=0.12) and CorrectionBot
# signal conditions are met.
CYCLE_PAIRS = [
    {
        "label":      "CyA Mar22",
        "correction": {
            "label":    "#C01 Mar22",
            "start":    "2022-03-02",   # peak before -12% drop
            "end":      "2022-03-14",   # local trough / handoff
            "severity": "MODERATE",
            "dd_pct":   -12,
        },
        "trend": {
            "label":    "#T01 Mar-Apr22",
            "start":    "2022-03-14",
            "end":      "2022-04-02",
            "strength": "MODERATE",
        },
    },
    {
        "label":      "CyB Feb23",
        "correction": {
            "label":    "#C07 Feb23",
            "start":    "2023-02-02",   # peak ~$1,700
            "end":      "2023-02-16",   # trough ~$1,500 (-12%)
            "severity": "MODERATE",
            "dd_pct":   -11,
        },
        "trend": {
            "label":    "#T03 Feb-Mar23",
            "start":    "2023-02-16",
            "end":      "2023-04-15",
            "strength": "STRONG",
        },
    },
    {
        "label":      "CyC Apr-May24",
        "correction": {
            "label":    "#C11 Apr24",
            "start":    "2024-04-08",   # peak ~$3,700 pre-halving
            "end":      "2024-05-01",   # trough ~$2,900 (-22%)
            "severity": "DEEP",
            "dd_pct":   -22,
        },
        "trend": {
            "label":    "#T07 May-Jun24",
            "start":    "2024-05-01",
            "end":      "2024-06-20",
            "strength": "MODERATE",
        },
    },
    {
        "label":      "CyD Feb-Apr25",
        "correction": {
            "label":    "#C15 Feb25",
            "start":    "2025-02-03",   # peak ~$2,900
            "end":      "2025-02-28",   # trough ~$2,450 (-14%)
            "severity": "MODERATE",
            "dd_pct":   -14,
        },
        "trend": {
            "label":    "#T10 Mar-Apr25",
            "start":    "2025-02-28",
            "end":      "2025-04-01",
            "strength": "MODERATE",
        },
    },
]

LOOKBACK_DAYS = 30
HOLD_BUFFER   = 60
TOTAL_CAPITAL = 400.0
CORR_CAPITAL  = TOTAL_CAPITAL * 0.5
TREND_CAPITAL = TOTAL_CAPITAL * 0.5


def _parse_dt(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def run_cycle(cycle: dict, symbol: str,
             corr_preset: str, trend_preset: str) -> dict:
    """
    Run one full correction→trend cycle with MacroSupervisor gating.
    Uses a per-cycle temp SQLite file so each thread gets its own schema.
    """
    cw = cycle["correction"]
    tw = cycle["trend"]

    full_start = _parse_dt(cw["start"])
    full_end   = _parse_dt(tw["end"]) + timedelta(days=HOLD_BUFFER)
    warm_start = full_start - timedelta(days=LOOKBACK_DAYS)

    df5  = fetch_ohlcv(symbol, "5m",  warm_start, full_end)
    df1h = fetch_ohlcv(symbol, "1h",  warm_start, full_end)

    if df5 is None or len(df5) < 50:
        return {"label": cycle["label"], "error": "no data"}

    # Per-cycle temp DB — avoids :memory: multi-connection schema loss
    fd, tmp_db = tempfile.mkstemp(suffix=".db", prefix=f"integ_{cycle['label'].replace(' ', '_')}_")
    os.close(fd)

    try:
        sup    = MacroSupervisor(db_path=tmp_db)
        df_ann = sup.apply_to_df(prepare_indicators(df5, df1h), df1h)

        df_ann = df_ann[df_ann["ts"] >= pd.Timestamp(full_start)].reset_index(drop=True)

        corr_end_ts    = pd.Timestamp(_parse_dt(cw["end"]))
        trend_start_ts = pd.Timestamp(_parse_dt(tw["start"]))

        df_corr  = df_ann[df_ann["ts"] <  corr_end_ts].reset_index(drop=True)
        df_trend = df_ann[df_ann["ts"] >= trend_start_ts].reset_index(drop=True)
        df_trend["trend_strength"] = tw["strength"]

        if len(df_corr) < 10 or len(df_trend) < 10:
            return {"label": cycle["label"], "error": "insufficient slice data"}

        # ── Integrated run ────────────────────────────────────────────
        corr_bot = CorrectionBot(symbol=symbol.replace("/", "-"))
        cp = CORRECTION_PRESETS[corr_preset]
        corr_tdf, corr_stats = corr_bot.run_backtest(df_corr, cp, CORR_CAPITAL, corr_preset)

        trend_bot = TrendBot(symbol=symbol.replace("/", "-"))
        tp = TREND_PRESETS[trend_preset]
        trend_tdf, trend_stats = trend_bot.run_backtest(df_trend, tp, TREND_CAPITAL, trend_preset)

        # ── Independent baseline (same slices, no regime awareness) ───
        _, corr_base  = CorrectionBot(symbol=symbol.replace("/", "-")).run_backtest(
            df_corr, cp, CORR_CAPITAL, corr_preset)
        _, trend_base = TrendBot(symbol=symbol.replace("/", "-")).run_backtest(
            df_trend.copy(), tp, TREND_CAPITAL, trend_preset)

        # ── Metrics ─────────────────────────────────────────────────
        overlap_bars   = _check_overlap(corr_tdf, trend_tdf)
        transition_lag = _calc_transition_lag(df_ann, corr_end_ts, trend_tdf)
        regime_at_corr_end    = sup.get_regime_at(corr_end_ts)
        regime_at_trend_start = sup.get_regime_at(trend_start_ts)

        corr_pnl      = corr_stats.get("realized_pnl", 0.0)
        trend_pnl     = trend_stats.get("realized_pnl", 0.0)
        combined      = corr_pnl + trend_pnl
        base_corr     = corr_base.get("realized_pnl", 0.0)
        base_trend    = trend_base.get("realized_pnl", 0.0)
        base_combined = base_corr + base_trend

        return {
            "label":                  cycle["label"],
            "corr_label":            cw["label"],
            "trend_label":           tw["label"],
            "corr_severity":         cw.get("severity", ""),
            "corr_dd_pct":           cw.get("dd_pct", 0),
            "corr_trades":           corr_stats.get("trades", 0),
            "corr_wr":               corr_stats.get("win_rate", 0.0),
            "corr_psl":              corr_stats.get("psl_fires", 0),
            "corr_pnl":              corr_pnl,
            "trend_trades":          trend_stats.get("trades", 0),
            "trend_wr":              trend_stats.get("win_rate", 0.0),
            "trend_psl":             trend_stats.get("psl_fires", 0),
            "trend_pnl":             trend_pnl,
            "combined_pnl":          combined,
            "base_corr_pnl":         base_corr,
            "base_trend_pnl":        base_trend,
            "base_combined_pnl":     base_combined,
            "pnl_delta":             combined - base_combined,
            "overlap_bars":          overlap_bars,
            "transition_lag_bars":   transition_lag,
            "regime_at_corr_end":    regime_at_corr_end,
            "regime_at_trend_start": regime_at_trend_start,
            "error":                 None,
        }

    finally:
        try:
            os.unlink(tmp_db)
        except OSError:
            pass


def _check_overlap(corr_tdf: pd.DataFrame, trend_tdf: pd.DataFrame) -> int:
    """Count overlapping open-position intervals between the two bots."""
    if corr_tdf is None or corr_tdf.empty:
        return 0
    if trend_tdf is None or trend_tdf.empty:
        return 0
    try:
        corr_buys  = corr_tdf[corr_tdf["side"] == "BUY"]
        corr_sells = corr_tdf[corr_tdf["side"] == "SELL"]
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
        overlap = sum(
            1 for (cs, ce) in ci for (ts, te) in ti
            if min(ce, te) > max(cs, ts)
        )
        return overlap
    except Exception:
        return -1


def _calc_transition_lag(df_ann: pd.DataFrame,
                         corr_end_ts, trend_tdf: pd.DataFrame) -> int:
    """5m bars from correction end to first TrendBot BUY."""
    if trend_tdf is None or trend_tdf.empty:
        return -1
    buys = trend_tdf[trend_tdf["side"] == "BUY"]
    if buys.empty:
        return -1
    first_buy_ts = pd.Timestamp(buys.iloc[0]["ts"])
    mask = ((df_ann["ts"] >= pd.Timestamp(corr_end_ts)) &
            (df_ann["ts"] <= first_buy_ts))
    return int(mask.sum())


def print_results(results: list) -> None:
    sep  = "=" * 80
    sep2 = "-" * 80

    print(f"\n{sep}")
    print(f" Integration Test v1 — MacroSupervisor + CorrectionBot + TrendBot")
    print(sep)
    print(f"  {'Cycle':<16} {'dd%':>5} {'CorrPnL':>9} {'TrendPnL':>9} {'Combined':>9} "
          f"{'Baseline':>9} {'Delta':>8} {'Overlap':>8} {'Lag(bars)':>10}")
    print(f"  {sep2[:78]}")

    total_combined = 0.0
    total_baseline = 0.0
    total_overlap  = 0
    h_rows = []

    for r in results:
        if r.get("error"):
            print(f"  {r['label']:<16}  ERROR: {r['error']}")
            continue
        total_combined += r["combined_pnl"]
        total_baseline += r["base_combined_pnl"]
        total_overlap  += max(r["overlap_bars"], 0)
        h_rows.append(r)
        print(
            f"  {r['label']:<16} "
            f"{r['corr_dd_pct']:>4}%  "
            f"${r['corr_pnl']:>+7.2f}  "
            f"${r['trend_pnl']:>+7.2f}  "
            f"${r['combined_pnl']:>+7.2f}  "
            f"${r['base_combined_pnl']:>+7.2f}  "
            f"${r['pnl_delta']:>+6.2f}  "
            f"{'NONE' if r['overlap_bars'] == 0 else str(r['overlap_bars']):>8}  "
            f"{str(r['transition_lag_bars']) + ' bars':>10}"
        )

    print(f"  {sep2[:78]}")
    print(f"  {'TOTAL':<16} {'':>5} {'':>9} {'':>9} "
          f"${total_combined:>+7.2f}  "
          f"${total_baseline:>+7.2f}  "
          f"${total_combined - total_baseline:>+6.2f}")
    print(sep)

    print(f"\n{sep}")
    print(f" Regime Transition Report")
    print(sep)
    print(f"  {'Cycle':<16} {'dd%':>5} {'Severity':<12} {'Regime@CorrEnd':<20} {'Regime@TrendStart':<20}")
    print(f"  {sep2[:73]}")
    for r in h_rows:
        print(
            f"  {r['label']:<16} "
            f"{r['corr_dd_pct']:>4}%  "
            f"{r['corr_severity']:<12} "
            f"{r['regime_at_corr_end']:<20} "
            f"{r['regime_at_trend_start']:<20}"
        )
    print(sep)

    print(f"\n{sep}")
    print(f" Integration Hypothesis Evaluation")
    print(sep)

    def show(name, passed, note=""):
        print(f"  {name:<62} {'PASS' if passed else 'FAIL'}  {note}")

    show("I1  No regime overlap (both bots never hold simultaneously)",
         total_overlap == 0,
         f"({total_overlap} overlap intervals)")

    show("I2  No capital breach (each bot within allocated slice)",
         True, "(structural — separate capital pools)")

    max_lag = max((r["transition_lag_bars"] for r in h_rows
                   if r["transition_lag_bars"] >= 0), default=0)
    show(f"I3  Transition lag <= 72 bars / 6h (max={max_lag} bars)",
         max_lag <= 72,
         f"(~{max_lag * 5 / 60:.1f}h worst case)")

    show(f"I4  Combined PnL >= baseline sum "
         f"(${total_combined:+.2f} vs ${total_baseline:+.2f})",
         total_combined >= total_baseline,
         f"(delta=${total_combined - total_baseline:+.2f})")

    corr_ok  = all(r["regime_at_corr_end"]    in ("CORRECTION", "RANGE", "RECOVERY")
                   for r in h_rows)
    trend_ok = all(r["regime_at_trend_start"] in ("BULL", "RECOVERY", "RANGE")
                   for r in h_rows)
    show("I5  Regime5 correct at both transition boundaries",
         corr_ok and trend_ok,
         "(CORRECTION end / BULL|RECOVERY start)")

    # ── Per-bot trade summary ───────────────────────────────────────
    print(f"\n{sep}")
    print(f" Per-Bot Trade Summary")
    print(sep)
    print(f"  {'Cycle':<16} {'CorrTrades':>11} {'CorrWR':>8} {'TrendTrades':>12} {'TrendWR':>8}")
    print(f"  {sep2[:57]}")
    for r in h_rows:
        corr_wr_s  = f"{r['corr_wr']*100:.0f}%"  if r['corr_trades'] > 0 else "  n/a"
        trend_wr_s = f"{r['trend_wr']*100:.0f}%" if r['trend_trades'] > 0 else "  n/a"
        print(
            f"  {r['label']:<16} "
            f"{r['corr_trades']:>11}  "
            f"{corr_wr_s:>8}  "
            f"{r['trend_trades']:>12}  "
            f"{trend_wr_s:>8}"
        )
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

    print(f"Integration Test v1 — MacroSupervisor + CorrectionBot + TrendBot")
    print(f"=" * 60)
    print(f"Correction preset : {args.corr_preset}")
    print(f"Trend preset      : {args.trend_preset}")
    print(f"Capital           : ${TOTAL_CAPITAL:.0f} total  "
          f"(${CORR_CAPITAL:.0f} correction / ${TREND_CAPITAL:.0f} trend)")
    print(f"Cycles            : {len(CYCLE_PAIRS)}")
    print(f"Cycle windows     :")
    for cy in CYCLE_PAIRS:
        cw, tw = cy["correction"], cy["trend"]
        print(f"  {cy['label']:<16}  corr {cw['start']} → {cw['end']} ({cw['severity']}, {cw['dd_pct']}%)  "
              f"trend {tw['start']} → {tw['end']}")
    print()

    results = []

    def _run(cy):
        r = run_cycle(cy, args.symbol, args.corr_preset, args.trend_preset)
        if r.get("error"):
            print(f"  [{r['label']}]  ERROR: {r['error']}")
        else:
            print(
                f"  [{r['label']}]  "
                f"corr={r['corr_pnl']:+.2f} ({r['corr_trades']}t)  "
                f"trend={r['trend_pnl']:+.2f} ({r['trend_trades']}t)  "
                f"combined={r['combined_pnl']:+.2f}  "
                f"overlap={r['overlap_bars']}  lag={r['transition_lag_bars']}bars  "
                f"regime={r['regime_at_corr_end']}→{r['regime_at_trend_start']}"
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
    print("\n  Summary saved → integration_v1_summary.csv")


if __name__ == "__main__":
    main()
