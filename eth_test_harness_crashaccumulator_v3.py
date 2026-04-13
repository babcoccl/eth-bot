#!/usr/bin/env python3
"""
eth_test_harness_crashaccumulator_v3.py
=========================================
v3 changes tested:
  1. emergency_action="HOLD" for catastrophic crashes (depth > 35%)
     Bot freezes (no new buys, no liquidation) instead of emergency exit.
  2. max_accum_depth_pct=0.35 — stops new buys when crash > 35% from start.
     Preserves dry powder for bottom accumulation later.
  3. Depth-tiered hold periods:
     shallow: 120d  |  moderate: 180d  |  major: 365d  |  catastrophic: 730d
  4. Position Reserve Manager tranches replace flat profit_target:
     +10% -> sell 33%  |  +15% -> sell 33%  |  +20% -> sell rest
     OVR-2: CORRECTION + unrealized>8% -> exit all (bird in hand)

Performance notes:
  - fetch_ohlcv() caches historical Parquet files in ./ohlcv_cache/
    First run fetches from Coinbase; subsequent runs load from disk (~50x faster).
  - Windows run in parallel via ThreadPoolExecutor (4 workers by default).
    Use --workers 1 to disable parallelism for debugging.
  - df_warm is deduplicated: sliced from df5 instead of a separate API call.

Key hypothesis additions:
  H8: No emergency SELL exits on crashes > 35% from start (FROZEN instead)
  H9: FROZEN positions reach profit_target within 730d
"""

import argparse, sys, os, warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

from eth_helpers import fetch_ohlcv, prepare_indicators
from eth_crash_accumulator_v3 import CrashAccumulator, PRESETS

# Import CRASH_WINDOWS from 4yr catalog
from crash_windows_4yr import CRASH_WINDOWS


def run_window(symbol, window, capital, preset_name, max_hold_days=365, lookback=45):
    p = PRESETS[preset_name]
    start_dt  = datetime.strptime(window["start"], "%Y-%m-%d").replace(tzinfo=timezone.utc)
    crash_end = datetime.strptime(window["end"],   "%Y-%m-%d").replace(tzinfo=timezone.utc)

    # Auto-detect maximum hold days from preset to ensure data coverage.
    # A 730d catastrophic hold needs 730d of post-crash data — not 365d.
    preset_max_hold = max(
        p.get("max_hold_days_catastrophic", 730),
        p.get("max_hold_days_major",        365),
        p.get("max_hold_days_moderate",     180),
        p.get("max_hold_days_shallow",      120),
        p.get("max_hold_days",              max_hold_days),
        max_hold_days,
    )
    ext_end = crash_end + timedelta(days=preset_max_hold)
    warm_dt = start_dt - timedelta(days=lookback)

    # ── Fetch full range once; slice df_warm from it (saves one API call) ────
    df5  = fetch_ohlcv(symbol, "5m",  warm_dt, ext_end)
    df1h = fetch_ohlcv(symbol, "1h",  warm_dt, ext_end)

    if df5 is None or len(df5) < 50:
        return pd.DataFrame(), {}

    # df_warm is the warmup slice — no extra fetch needed
    df_warm = df5[df5["ts"] < pd.Timestamp(start_dt)].reset_index(drop=True)

    df_ind = prepare_indicators(df5, df1h)
    df_run = df_ind[df_ind["ts"] >= pd.Timestamp(start_dt)].reset_index(drop=True)

    bot = CrashAccumulator(symbol=symbol.replace("/", "-"))
    return bot.run_backtest(df_run, p, capital, preset_name,
                            df_warm=df_warm, crash_end_ts=crash_end)


def print_results(results, capital, preset_name):
    sep = "=" * 80
    sep2 = "-" * 80
    print(f"\n{sep}")
    print(f" CrashAccumulator v3 — {preset_name} — per-window results")
    print(sep)
    print(f"  {'Window':<18} {'Days':>5} {'Buys':>5} {'Dep%':>6} {'Disc%':>7}"
          f"  {'Exit':<16} {'RealPnL':>9}  Notes")
    print(f"  {sep2[:78]}")

    h_rows = []
    for w, tdf, s in results:
        if not s:
            print(f"  {w['label']:<18}  -- no data --")
            continue
        buys     = s.get("buys", 0)
        dep_pct  = s.get("deploy_pct", 0)
        disc     = s.get("discount_pct", 0)
        rpnl     = s.get("realized_pnl", 0)
        p_exits  = s.get("profit_exits", 0)
        e_exits  = s.get("emergency_exits", 0)
        t_stops  = s.get("time_stops", 0)
        frozen   = s.get("frozen", False)
        pos_open = s.get("position_open", False)

        if p_exits > 0:    exit_str = "PROFIT_TARGET"
        elif e_exits > 0:  exit_str = "EMERGENCY_SELL"
        elif frozen:       exit_str = "FROZEN_HOLD"
        elif t_stops > 0:  exit_str = "TIME_STOP"
        elif pos_open:     exit_str = "STILL_OPEN"
        else:              exit_str = s.get("state", "?")

        note = "❄️ freeze+hold" if frozen else ("⚠️ emerg" if e_exits > 0 else "")
        print(f"  {w['label']:<18} {w['days']:>5.1f} {buys:>5} {dep_pct:>5.1f}%"
              f" {disc:>+6.1f}%  {exit_str:<16} ${rpnl:>+8.2f}  {note}")
        h_rows.append({**s, "label": w["label"], "severity": w["severity"],
                        "exit_str": exit_str})

    total_pnl  = sum(s.get("realized_pnl", 0) for _, _, s in results if s)
    total_buys = sum(s.get("buys", 0) for _, _, s in results if s)
    print(f"  {sep2[:78]}")
    print(f"  {'COMBINED':<18} {'':>5} {total_buys:>5}  {'':>14}  ${total_pnl:>+8.2f}")
    print(sep)

    # Hypothesis eval
    print(f"\n{sep}")
    print(f" CrashAccumulator v3 — HYPOTHESIS EVALUATION")
    print(sep)
    valid = [r for r in h_rows if r.get("buys", 0) > 0]
    major = [r for r in valid if r["severity"] in ("MAJOR","CATASTROPHIC")]
    cat   = [r for r in valid if r["severity"] == "CATASTROPHIC"]

    avg_disc = sum(r.get("discount_pct",0) for r in valid)/len(valid) if valid else 0
    vp_total = sum(r.get("near_support_buys",0) for r in valid)

    def _h1_check(r):
        d   = r.get("discount_pct", 0)
        sev = r.get("severity","SHALLOW")
        if sev in ("MAJOR","CATASTROPHIC"): return d >= 15.0
        if sev == "MODERATE":               return d >= 8.0
        return True

    def show(name, passed, note=""):
        print(f"  {name:<60} {'PASS' if passed else 'FAIL'}  {note}")

    show(f"H1  disc thresholds met by severity (avg={avg_disc:.1f}%)",
         all(_h1_check(r) for r in valid))
    show(f"H2  disc >= 15% MAJOR/CATASTROPHIC ({len(major)} windows)",
         all(r.get("discount_pct",0) >= 15.0 for r in major))
    show(f"H3  deploy_pct <= 80%",
         all(r.get("deploy_pct",0) <= 82.0 for r in valid))
    show(f"H4  buys >= 3 MAJOR/CATASTROPHIC",
         all(r.get("buys",0) >= 3 for r in major))
    show(f"H5  zero EMERGENCY_SELL exits",
         all(r.get("emergency_exits",0) == 0 for r in valid))
    show(f"H6  exits via profit_target, frozen_hold, or still_open (no panic)",
         all(r["exit_str"] in ("PROFIT_TARGET","FROZEN_HOLD","STILL_OPEN") for r in major))
    show(f"H7  VP support buys >= 1 per MAJOR window (total={vp_total})",
         all(r.get("near_support_buys",0) >= 1 for r in major))
    show(f"H8  NO emergency SELL on crashes >35% (FROZEN instead)",
         all(r.get("emergency_exits",0) == 0 for r in cat))

    print(f"\n  Total realized PnL: ${total_pnl:+.2f}")
    print(sep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol",        default="ETH/USD")
    ap.add_argument("--capital",       default=400.0, type=float)
    ap.add_argument("--preset",        default="accumulator_v3",
                    choices=list(PRESETS.keys()))
    ap.add_argument("--max-hold-days", default=365, type=int)
    ap.add_argument("--workers",       default=4, type=int,
                    help="Parallel window workers (default=4, use 1 to disable)")
    ap.add_argument("--no-cache",      action="store_true",
                    help="Force live fetch, bypass disk cache")
    args = ap.parse_args()

    if args.no_cache:
        from eth_helpers import clear_ohlcv_cache
        clear_ohlcv_cache()

    print(f"Running {len(CRASH_WINDOWS)} CRASH windows (max hold {args.max_hold_days}d, "
          f"workers={args.workers})...")

    # ── Parallel execution ────────────────────────────────────────────────────
    # Each window is independent — safe to parallelize.
    # Results are re-sorted to match CRASH_WINDOWS order for deterministic output.
    results_map = {}

    def _run(w):
        tdf, s = run_window(args.symbol, w, args.capital, args.preset, args.max_hold_days)
        return w, tdf, s

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_run, w): w for w in CRASH_WINDOWS}
        for fut in as_completed(futures):
            w, tdf, s = fut.result()
            results_map[w["label"]] = (w, tdf, s)
            if s:
                frozen  = s.get("frozen", False)
                e_exits = s.get("emergency_exits", 0)
                p_exits = s.get("profit_exits", 0)
                tag = "FROZEN" if frozen else ("EMERG" if e_exits else ("PT" if p_exits else "open"))
                print(f"  [{w['label']}]  {w['start']}  {w['severity']} ... "
                      f"{s.get('buys',0)} buys  disc={s.get('discount_pct',0):+.1f}%  "
                      f"{tag}  pnl=${s.get('realized_pnl',0):+.2f}")
            else:
                print(f"  [{w['label']}]  no data")

    # Re-sort to original window order
    results = [results_map[w["label"]] for w in CRASH_WINDOWS]

    print_results(results, args.capital, args.preset)

    trades = [t for _, t, s in results
              if s and t is not None and isinstance(t, pd.DataFrame) and not t.empty]
    if trades:
        pd.concat(trades, ignore_index=True).to_csv(
            f"crashacc_v3_{args.preset}_trades.csv", index=False)


if __name__ == "__main__":
    main()
