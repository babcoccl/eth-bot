#!/usr/bin/env python3
"""
eth_test_harness_recoverybot.py  —  RecoveryBot isolated test harness
================================================================
Tests RecoveryBot ONLY against approved RECOVERY windows.
"""

import argparse, sys, os, warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

from eth_trading.utils.helpers import fetch_ohlcv, prepare_indicators
from eth_trading.bots.recovery_bot import RecoveryBot, PRESETS

_LOOKBACK_DAYS = 180

# ── TIER 1 & TIER 2: RECOVERY windows ─────────────
APPROVED_WINDOWS = [
    {"label": "#14 T1 May23",    "start": "2023-05-23", "end": "2023-05-30", "days":  7.0, "regime": "RECOVERY", "quality": 9.0, "chg": +3.2},
    {"label": "#23 T1 Sep23",    "start": "2023-09-27", "end": "2023-10-04", "days":  7.0, "regime": "RECOVERY", "quality": 9.0, "chg": +2.2},
    {"label": "#28 T1 Dec23",    "start": "2023-12-27", "end": "2024-01-03", "days":  7.0, "regime": "RECOVERY", "quality": 9.0, "chg": +1.1},
    {"label": "#65 T1 Apr25",    "start": "2025-04-25", "end": "2025-05-01", "days":  7.0, "regime": "RECOVERY", "quality": 9.0, "chg": +3.6},
    {"label": "#100 T1 Mar26",   "start": "2026-03-12", "end": "2026-03-19", "days":  7.0, "regime": "RECOVERY", "quality": 9.0, "chg": +2.4},
    {"label": "#8 T2 Mar23",     "start": "2023-03-23", "end": "2023-03-30", "days":  7.0, "regime": "RECOVERY", "quality": 9.0, "chg": -2.0},
    {"label": "#19 T2 Jun23",    "start": "2023-06-30", "end": "2023-07-07", "days":  7.0, "regime": "RECOVERY", "quality": 9.0, "chg": -1.5},
    {"label": "#51 T2 Oct24",    "start": "2024-10-16", "end": "2024-10-23", "days":  7.0, "regime": "RECOVERY", "quality": 9.0, "chg": -1.9},
]

TIER3_WINDOWS = [
    {"label": "#20 T3 Jul-Aug23", "start": "2023-07-07", "end": "2023-08-17", "days": 41.7, "regime": "RANGE", "quality": 10.0, "chg": -7.5},
]

def run_window(symbol, window, capital, preset_name, lookback=_LOOKBACK_DAYS):
    p = PRESETS[preset_name]
    base_start = datetime.strptime(window["start"], "%Y-%m-%d").replace(tzinfo=timezone.utc)
    # Recoverybot tests extend by 10 days for exits
    trend_end = datetime.strptime(window["end"], "%Y-%m-%d").replace(tzinfo=timezone.utc)
    ext_end = trend_end + timedelta(days=10)
    warm_dt = base_start - timedelta(days=lookback)

    df5 = fetch_ohlcv(symbol, "5m", warm_dt, ext_end)
    df1h = fetch_ohlcv(symbol, "1h", warm_dt, ext_end)

    if df5 is None or len(df5) < 50:
        return window["label"], pd.DataFrame(), {}

    df_ind = prepare_indicators(df5, df1h, min_dwell=3)
    df_run = df_ind[df_ind["ts"] >= pd.Timestamp(base_start)].reset_index(drop=True)
    
    if len(df_run) < 10:
        return window["label"], pd.DataFrame(), {}

    bot = RecoveryBot(symbol=symbol.replace("/", "-"))
    tdf, s = bot.run_backtest(df_run, p, capital, preset_name)
    return window["label"], tdf, s

def print_results(results, capital, preset_name):
    sep = "=" * 80
    sep2 = "-" * 80
    print(f"\n{sep}")
    print(f" RecoveryBot v1 -- {preset_name} -- per-window results")
    print(sep)
    print(f"  {'Window':<18} {'Days':>5} {'Trades':>6} {'WR%':>6} "
          f"{'PSL':>4} {'PSL%':>5} {'TGT':>4} {'PnL':>9}")
    print(f"  {sep2[:78]}")

    valid_s = []
    for w, tdf, s in results:
        if not s:
            print(f"  {w['label']:<18}  -- no data --")
            continue
        valid_s.append(s)
        trades = s.get("trades", 0)
        wr = s.get("win_rate", 0)
        psl = s.get("psl_fires", 0)
        tgt = s.get("target_fires", 0)
        rpnl = s.get("realized_pnl", 0)
        psl_rate = psl / trades * 100 if trades > 0 else 0.0

        print(f"  {w['label']:<18} {w['days']:>5.1f} {trades:>6} "
              f"{wr:>5.1f}%  {psl:>3} {psl_rate:>4.0f}%  {tgt:>3}  ${rpnl:>+7.2f}")

    if not valid_s:
        return

    total_pnl = sum(s.get("realized_pnl", 0) for s in valid_s)
    total_trades = sum(s.get("trades", 0) for s in valid_s)
    total_psl = sum(s.get("psl_fires", 0) for s in valid_s)
    total_tgt = sum(s.get("target_fires", 0) for s in valid_s)
    total_psl_rate = total_psl / total_trades * 100 if total_trades > 0 else 0.0
    avg_wr = sum(s.get("win_rate", 0) * s.get("trades", 0) for s in valid_s) / total_trades if total_trades > 0 else 0
    tgt_ratio = total_tgt / total_trades if total_trades > 0 else 0
    psl_ratio = total_psl / total_trades if total_trades > 0 else 0

    print(f"  {sep2[:78]}")
    print(f"  {'COMBINED':<18} {'':>5} {total_trades:>6} "
          f"{'':>6}  {total_psl:>3} {total_psl_rate:>4.0f}%  {total_tgt:>3}  ${total_pnl:>+7.2f}")
    print(sep)
    
    print(f"\n{sep}")
    print(f" RecoveryBot v1 -- HYPOTHESIS EVALUATION")
    print(sep)
    def show(name, passed, note=""):
        print(f"  {name:<62} {'PASS' if passed else 'FAIL'}  {note}")

    show(f"H1  combined win_rate >= 60% ({avg_wr:.1f}%)", avg_wr >= 60.0, f"({total_trades} trades)")
    show(f"H2  combined PnL >= $0.00 (${total_pnl:+.2f})", total_pnl >= 0)
    print(f"\n  Total realized PnL: ${total_pnl:+.2f}")
    print(sep)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="ETH/USD")
    ap.add_argument("--capital", default=1000.0, type=float)
    ap.add_argument("--tier3", action="store_true", help="Run TIER3 long RANGE window instead")
    ap.add_argument("--preset", default="dcb_v1", choices=list(PRESETS.keys()))
    ap.add_argument("--workers", default=4, type=int)
    args = ap.parse_args()

    windows_to_run = TIER3_WINDOWS if args.tier3 else APPROVED_WINDOWS

    print(f"RecoveryBot Test Harness")
    print(f"=" * 60)
    print(f"Running {len(windows_to_run)} windows (preset={args.preset})...")

    results_map = {}
    def _run(w):
        try:
            return run_window(args.symbol, w, args.capital, args.preset)
        except Exception as exc:
            print(f"\n  [ERROR {w['label']}] {exc}")
            return w["label"], pd.DataFrame(), {}

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_run, w): w for w in windows_to_run}
        for fut in as_completed(futures):
            lbl, tdf, s = fut.result()
            results_map[lbl] = (lbl, tdf, s)

    # Reorder results to match input list
    results = []
    for w in windows_to_run:
        lbl = w["label"]
        if lbl in results_map:
            _, tdf, s = results_map[lbl]
            results.append((w, tdf, s))

    print_results(results, args.capital, args.preset)

if __name__ == "__main__":
    main()
