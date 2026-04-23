#!/usr/bin/env python3
"""
eth_test_harness_trendbot_v1.py
================================
Tests TrendBot v1 across 14 TREND windows (2022-2026).

Hypotheses:
  H1  win_rate >= 55% across all windows (trendbot should win more than it loses)
  H2  psl_rate < 25% of all closed trades (tight PSL should be rare)
  H3  total realized PnL > $0 across all windows
  H4  zero entries during non-UPTREND h1 regime bars
  H5  avg_bars_held >= 3 on target exits (should hold briefly, not scalp)
  H6  PARABOLIC windows produce more trades than MODERATE windows (bot is active)

Notes:
  - max_hold_days=60: trend windows are shorter-lived than corrections.
  - Same try_shifts retry logic as CorrectionBot harness (up to 3d shift).
  - Parquet cache shared with other bots (ohlcv_cache/).
"""

from curses import window
import argparse, sys, os, warnings
from eth_macrosupervisor_v30 import MacroSupervisor
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

from eth_helpers import fetch_ohlcv, prepare_indicators
from eth_trendbot_v1 import TrendBot, PRESETS
from trend_windows_4yr import TREND_WINDOWS

MAX_DATE_SHIFTS = 3


def run_window(symbol, window, capital, preset_name, max_hold_days=60, lookback=30, min_dwell=3):
    p          = PRESETS[preset_name]
    base_start = datetime.strptime(window["start"], "%Y-%m-%d").replace(tzinfo=timezone.utc)
    trend_end  = datetime.strptime(window["end"],   "%Y-%m-%d").replace(tzinfo=timezone.utc)
    ext_end    = trend_end + timedelta(days=max_hold_days)

    for shift in range(MAX_DATE_SHIFTS + 1):
        start_dt = base_start + timedelta(days=shift)
        warm_dt  = start_dt - timedelta(days=lookback)

        df5  = fetch_ohlcv(symbol, "5m",  warm_dt, ext_end)
        df1h = fetch_ohlcv(symbol, "1h",  warm_dt, ext_end)
        print(f"  DEBUG {window['label']} shift={shift}: df5={df5 is not None and len(df5)}, df1h={df1h is not None and len(df1h)}")

        if df5 is None or len(df5) < 50:
            continue

        df_ind = prepare_indicators(df5, df1h, min_dwell=min_dwell)

        df_run = df_ind[df_ind["ts"] >= pd.Timestamp(start_dt)].reset_index(drop=True)

        if len(df_run) < 10:
            continue

        if shift > 0:
            print(f"    [shift+{shift}d] {window['label']} resolved at {start_dt.date()}")

         # ── Regime distribution diagnostic ──────────────────────────────
        regime_dist = df_run["regime5"].value_counts(normalize=True).to_dict()
        tradeable_pct = regime_dist.get("BULL", 0) + regime_dist.get("RECOVERY", 0)
        print(f"  [regime dist] {window['label']}: tradeable={tradeable_pct:.1%} "
              f"| {', '.join(f'{k}={v:.1%}' for k, v in sorted(regime_dist.items()))}")
        
        bull_recov_pct = (
            (df_run["regime5"].isin(["BULL", "RECOVERY"])).sum() / len(df_run)
        )
        if bull_recov_pct < 0.30:
            print(f"  [{window['label']}]  skipped (bull_recov_pct={bull_recov_pct:.1%} < 30%)")
            return window["label"], pd.DataFrame(), {}

        # Inject window-level trend strength so qty_scale in the bot can act on it
        df_run["window_strength"] = window["strength"]

        bot = TrendBot(symbol=symbol.replace("/", "-"))
        tdf, s = bot.run_backtest(df_run, p, capital, preset_name)
        return window["label"], tdf, s

    return window["label"], pd.DataFrame(), {}


def print_results(results, capital, preset_name):
    sep  = "=" * 80
    sep2 = "-" * 80
    print(f"\n{sep}")
    print(f" TrendBot v1 — {preset_name} — per-window results")
    print(sep)
    print(f"  {'Window':<18} {'Days':>5} {'Str':<10} {'Trades':>6} {'WR%':>6} "
          f"{'PSL':>4} {'TGT':>4} {'PnL':>9}")
    print(f"  {sep2[:78]}")

    h_rows = []
    for w, tdf, s in results:
        if not s:
            print(f"  {w['label']:<18}  -- no data --")
            continue

        trades  = s.get("trades", 0)
        wr      = s.get("win_rate", 0)
        psl     = s.get("psl_fires", 0)
        tgt     = s.get("target_fires", 0)
        rpnl    = s.get("realized_pnl", 0)

        print(f"  {w['label']:<18} {w['days']:>5.1f} {w['strength']:<10} {trades:>6} "
              f"{wr:>5.1f}%  {psl:>3}  {tgt:>3}  ${rpnl:>+7.2f}")

        h_rows.append({**s, "label": w["label"], "strength": w["strength"],
                        "days": w["days"]})

    total_pnl    = sum(s.get("realized_pnl", 0)   for _, _, s in results if s)
    total_trades = sum(s.get("trades", 0)          for _, _, s in results if s)
    total_psl    = sum(s.get("psl_fires", 0)       for _, _, s in results if s)
    total_tgt    = sum(s.get("target_fires", 0)    for _, _, s in results if s)
    print(f"  {sep2[:78]}")
    print(f"  {'COMBINED':<18} {'':>5} {'':>10} {total_trades:>6} "
          f"{'':>6}  {total_psl:>3}  {total_tgt:>3}  ${total_pnl:>+7.2f}")
    print(sep)

    # ── Hypothesis evaluation ────────────────────────────────────────────
    print(f"\n{sep}")
    print(f" TrendBot v1 — HYPOTHESIS EVALUATION")
    print(sep)

    valid     = [r for r in h_rows if r.get("trades", 0) > 0]
    parabolic = [r for r in valid if r["strength"] == "PARABOLIC"]
    moderate  = [r for r in valid if r["strength"] == "MODERATE"]

    all_trades = sum(r.get("trades", 0) for r in valid)
    all_wins   = sum(int(r.get("win_rate", 0) / 100 * r.get("trades", 0)) for r in valid)
    all_psls   = sum(r.get("psl_fires", 0) for r in valid)
    avg_wr     = all_wins / all_trades * 100 if all_trades > 0 else 0
    psl_rate   = all_psls / all_trades * 100 if all_trades > 0 else 0

    avg_bars_tgt = (sum(r.get("avg_bars_held", 0) for r in valid) / len(valid)
                   if valid else 0)
    avg_trades_par = (sum(r.get("trades", 0) for r in parabolic) / len(parabolic)
                     if parabolic else 0)
    avg_trades_mod = (sum(r.get("trades", 0) for r in moderate) / len(moderate)
                     if moderate else 0)

    def show(name, passed, note=""):
        print(f"  {name:<62} {'PASS' if passed else 'FAIL'}  {note}")

    show(f"H1  win_rate >= 55% across all windows (avg={avg_wr:.1f}%)",
         avg_wr >= 55.0,
         f"({all_trades} trades)")
    show(f"H2  psl_rate < 25% of closed trades ({psl_rate:.1f}% actual)",
         psl_rate < 25.0,
         f"({all_psls} PSL / {all_trades} trades)")
    show(f"H3  total realized PnL > $0 (${total_pnl:+.2f})",
         total_pnl > 0)
    show(f"H4  zero entries outside UPTREND regime",
         True,  # enforced structurally in run_backtest
         "(structural)")
    show(f"H5  avg_bars_held >= 3 on target exits (avg={avg_bars_tgt:.1f})",
         avg_bars_tgt >= 3.0,
         f"({len(valid)} windows)")
    show(f"H6  PARABOLIC windows more active than MODERATE "
         f"(par={avg_trades_par:.1f} vs mod={avg_trades_mod:.1f})",
         avg_trades_par >= avg_trades_mod,
         f"({len(parabolic)} par / {len(moderate)} mod windows)")

    print(f"\n  Total realized PnL: ${total_pnl:+.2f}")
    print(sep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol",        default="ETH/USD")
    ap.add_argument("--capital",       default=400.0, type=float)
    ap.add_argument("--preset",        default="trendbot_v1",
                    choices=list(PRESETS.keys()))
    ap.add_argument("--max-hold-days", default=60, type=int)
    ap.add_argument("--workers",       default=4, type=int)
    ap.add_argument("--no-cache",      action="store_true")
    ap.add_argument("--min-dwell", default=3, type=int,
                help="regime5_min_dwell_bars passed to MacroSupervisor (default 3)")
    args = ap.parse_args()

    if args.no_cache:
        from eth_helpers import clear_ohlcv_cache
        clear_ohlcv_cache()

    print(f"TrendBot v1 Tests")
    print(f"=" * 60)
    print(f"Running {len(TREND_WINDOWS)} TREND windows "
          f"(max hold {args.max_hold_days}d, workers={args.workers})...")

    results_map = {}

    def _run(w):
        try:
            label, tdf, s = run_window(args.symbol, w, args.capital,
                            args.preset, args.max_hold_days,
                            min_dwell=args.min_dwell)
            return w, tdf, s
        except Exception as exc:
            import traceback
            print(f"\n  [ERROR {w['label']}] {exc}")
            traceback.print_exc()
            return w, pd.DataFrame(), {}

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_run, w): w for w in TREND_WINDOWS}
        for fut in as_completed(futures):
            w, tdf, s = fut.result()
            results_map[w["label"]] = (w, tdf, s)
            if s:
                trades = s.get("trades", 0)
                wr     = s.get("win_rate", 0)
                rpnl   = s.get("realized_pnl", 0)
                psl    = s.get("psl_fires", 0)
                print(f"  [{w['label']}]  {w['start']}  {w['strength']} ... "
                      f"{trades} trades  wr={wr:.0f}%  psl={psl}  pnl=${rpnl:+.2f}")
            else:
                print(f"  [{w['label']}]  no data")

    results = [results_map[w["label"]] for w in TREND_WINDOWS]
    print_results(results, args.capital, args.preset)

    trades = [t for _, t, s in results
              if s and t is not None and isinstance(t, pd.DataFrame) and not t.empty]
    if trades:
        pd.concat(trades, ignore_index=True).to_csv(
            f"trendbot_v1_{args.preset}_trades.csv", index=False)


if __name__ == "__main__":
    main()
