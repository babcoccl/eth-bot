#!/usr/bin/env python3
"""
eth_test_harness_rangebot.py  —  RangeBot isolated test harness
================================================================
Tests RangeBot ONLY against approved RANGE/RECOVERY windows.

Hypotheses:
  H1  combined win_rate  >= 75%    (may drop from 91% with higher target_bps)
  H2  combined PnL       >= $0.00
  H3  target / trades    >= 0.80
  H4  psl / trades       <= 0.12
  H5  total trades       >= 15     (range windows shorter than trend windows)

Usage:
  python eth_test_harness_rangebot.py                         # all windows
  python eth_test_harness_rangebot.py --start 2026-01-07 --end 2026-01-20
  python eth_test_harness_rangebot.py --preset rangebot_v1_tight
"""
import argparse, sys, warnings
warnings.filterwarnings("ignore")

from eth_rangebot_v4      import RangeBot, PRESETS
from eth_bot_harness_base import (run_single_window, run_all_windows,
                                   print_window_table, eval_hypotheses)

# ── TIER 1: Positive-drift RECOVERY windows (5 windows, 35 days) ─────────────
# These are the primary test set. All have chg > 0% — ideal for long-only bot.
APPROVED_WINDOWS = [
    {"label": "#14 T1 May23",    "start": "2023-05-23", "end": "2023-05-30",
     "days":  7.0, "regime": "RECOVERY", "quality": 9.0, "chg": +3.2},
    {"label": "#23 T1 Sep23",    "start": "2023-09-27", "end": "2023-10-04",
     "days":  7.0, "regime": "RECOVERY", "quality": 9.0, "chg": +2.2},
    {"label": "#28 T1 Dec23",    "start": "2023-12-27", "end": "2024-01-03",
     "days":  7.0, "regime": "RECOVERY", "quality": 9.0, "chg": +1.1},
    {"label": "#65 T1 Apr25",    "start": "2025-04-25", "end": "2025-05-01",
     "days":  7.0, "regime": "RECOVERY", "quality": 9.0, "chg": +3.6},  # known good
    {"label": "#100 T1 Mar26",   "start": "2026-03-12", "end": "2026-03-19",
     "days":  7.0, "regime": "RECOVERY", "quality": 9.0, "chg": +2.4},  # known near-BE
    # ── TIER 2: Mild negative-drift RECOVERY (<2.5%/window) ──────────────────
    {"label": "#8 T2 Mar23",     "start": "2023-03-23", "end": "2023-03-30",
     "days":  7.0, "regime": "RECOVERY", "quality": 9.0, "chg": -2.0},
    {"label": "#19 T2 Jun23",    "start": "2023-06-30", "end": "2023-07-07",
     "days":  7.0, "regime": "RECOVERY", "quality": 9.0, "chg": -1.5},
    {"label": "#51 T2 Oct24",    "start": "2024-10-16", "end": "2024-10-23",
     "days":  7.0, "regime": "RECOVERY", "quality": 9.0, "chg": -1.9},
]

# ── TIER 3: Long RANGE window — run separately with --tier3 flag ──────────────
# #20: 41.7d RANGE, chg=-7.5% but drift rate only 0.18%/day (slowest of all).
# With max_hold_bars=288 this should generate 50+ trades. Uncertain outcome.
TIER3_WINDOWS = [
    {"label": "#20 T3 Jul-Aug23", "start": "2023-07-07", "end": "2023-08-17",
     "days": 41.7, "regime": "RANGE", "quality": 10.0, "chg": -7.5},
]


H_WIN_RATE   = 75.0  # T1+T2 mix: T2 mild-neg windows may lower combined WR slightly
H_PNL        = 0.00
H_TGT_RATIO  = 0.80
H_PSL_RATIO  = 0.12  # PSL fires should stay ~4/46 = 8.7% with 5% PSL


def main():
    ap = argparse.ArgumentParser(description="RangeBot Test Harness")
    ap.add_argument("--start",   default=None)
    ap.add_argument("--end",     default=None)
    ap.add_argument("--symbol",  default="ETH/USD")
    ap.add_argument("--capital", default=1000.0, type=float)
    ap.add_argument("--tier3",   action="store_true",
                    help="Run TIER3 long RANGE window instead of T1+T2")
    ap.add_argument("--preset",  default="rangebot_v4",
                    choices=list(PRESETS.keys()))
    args = ap.parse_args()

    print("=" * 74)
    print(" RangeBot v1 — Isolated Regime Test (RANGE/RECOVERY windows only)")
    print(" Signals: range_dip + lv_scalp  |  Exit: target  |  PSL 5%  |  DCA: NONE")
    print("=" * 74)
    print(f" Preset : {args.preset}")
    print(f" Capital: ${args.capital:,.0f} per window")

    windows_to_run = TIER3_WINDOWS if args.tier3 else APPROVED_WINDOWS

    if args.tier3:
        print(" Mode: TIER3 long RANGE window (separate validation)")

    if args.start and args.end:
        print(f"\n Running single window: {args.start} → {args.end}")
        tdf, s = run_single_window(RangeBot, PRESETS, args.symbol,
                                   args.start, args.end, args.capital, args.preset)
        if not s:
            print("[WARN] No trades generated.")
            sys.exit(0)
        windows = [{"label": "custom", "start": args.start,
                    "end": args.end, "days": 0, "regime": "custom"}]
        results = [(windows[0], tdf, s)]
    else:
        print(f"\n Running all {len(APPROVED_WINDOWS)} approved windows...")
        results = run_all_windows(RangeBot, PRESETS, windows_to_run,
                                  args.symbol, args.capital, args.preset)

    all_t, all_pnl, all_wr, all_tgt, all_psl = print_window_table(
        results, "RangeBot v1", args.preset)

    eval_hypotheses(all_t, all_pnl, all_wr, all_tgt, all_psl,
                    H_WIN_RATE, H_PNL, H_TGT_RATIO, H_PSL_RATIO, "RangeBot v1")

    import pandas as pd
    all_trades = pd.concat([t for _,t,s in results if not t.empty], ignore_index=True)
    if not all_trades.empty:
        fname = f"rangebot_v1_{args.preset}_trades.csv"
        all_trades.to_csv(fname, index=False)
        print(f"\n Trades CSV: {fname}  ({len(all_trades)} rows)")


if __name__ == "__main__":
    main()
