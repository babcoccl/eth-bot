#!/usr/bin/env python3
"""
ETH Integration Test Harness v2 (Orchestrator Scaling Edition)
=============================================================
Simulates the full trading network (Correction + Trend + Range + Recovery + Hedge)
with Orchestrator-level scaling powered by MacroSupervisor v31.

New in v2:
- Supports MacroSupervisor v31 with Advisor Bridge.
- Scales capital allocation based on 'conviction' score.
- Dynamic reporting of 'Scaled Capital' vs 'Base Capital'.
"""

import os, sys, warnings, argparse
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# Suppress pandas warnings
warnings.filterwarnings("ignore")

# Ensure local imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eth_helpers             import fetch_ohlcv
from eth_macrosupervisor_v31 import MacroSupervisor
from eth_correction_bot_v1   import CorrectionBot, PRESETS as CORR_PRESETS
from eth_trendbot_v1         import TrendBot,       PRESETS as TREND_PRESETS
from eth_rangebot_v4         import RangeBot,       PRESETS as RANGE_PRESETS
from eth_recoverybot_v1      import RecoveryBot,    PRESETS as RECOVERY_PRESETS
from eth_hedgebot_v1         import HedgeBot,       PRESETS as HEDGE_PRESETS

SUPERVISOR_VERSION = "v31"

# --- Integration Test Specs (4 Primary Cycles) ---
CYCLE_PAIRS = [
    {"label": "CyA May-Jun22", "corr": "2022-05-01", "trend": "2022-08-01", "severity": "DEEP", "dd_pct": -56},
    {"label": "CyB Feb-Mar23", "corr": "2023-02-01", "trend": "2023-03-01", "severity": "MODERATE", "dd_pct": -11},
    {"label": "CyC Apr-May24", "corr": "2024-04-01", "trend": "2024-06-01", "severity": "DEEP", "dd_pct": -22},
    {"label": "CyD Feb-Mar25", "corr": "2025-02-01", "trend": "2025-04-20", "severity": "MODERATE", "dd_pct": -14},
]

# --- Global Config ---
BASE_CAPITAL = 1000.0
FEE_PCT      = 0.001
MIN_TRADEABLE_PCT = 0.25
MIN_TRADEABLE_RUN_BARS = 576 # 2 days of trend run

def run_cycle(cy, symbol, corr_preset, trend_preset, conviction_override=1.0):
    try:
        # 1. Fetch Data
        start_dt = datetime.strptime(cy["corr"], "%Y-%m-%d").replace(tzinfo=timezone.utc)
        # End window 90 days after trend start
        trend_dt = datetime.strptime(cy["trend"], "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end_dt   = trend_dt + timedelta(days=90)
        
        df1h = fetch_ohlcv(symbol, "1h", start_dt, end_dt)
        df5  = fetch_ohlcv(symbol, "5m", start_dt, end_dt)
        
        # 2. Setup Orchestrator (v31)
        from eth_helpers_v2 import prepare_indicators
        df_ann = prepare_indicators(df5, df1h, conviction=conviction_override)
        
        # Setup Supervisor instance for scale check
        sup = MacroSupervisor()
        sup.advisor_bridge_enabled = False # Disable live file read for backtest
        sup.conviction_score = conviction_override
        
        # Calculate Scaled Capital
        scaled_capital = BASE_CAPITAL * sup.get_capital_scale()
        
        # --- PHASE 1: Correction Windows ---
        # (Shortened for brevity: assuming standard split logic from v1)
        # ... [Logic to split df_ann into df_corr and df_trend based on pause events]
        # For v2, let's just use the full df_ann and let bots filter by regime
        
        # --- Tactical Bot Execution ---
        # Note: We now pass BASE_CAPITAL and the supervisor. 
        # The bots will call supervisor.request_allocation() to handle conviction-based scaling
        # and budget enforcement internally (v32 Orchestrator Logic).
        
        # 1. CorrectionBot
        cp = CORR_PRESETS[corr_preset]
        corr_bot = CorrectionBot(symbol=symbol.replace("/", "-"))
        _, corr_stats = corr_bot.run_backtest(df_ann.copy(), cp, BASE_CAPITAL, corr_preset, supervisor=sup)
        
        # 2. TrendBot (special handling for skip logic)
        tp = TREND_PRESETS[trend_preset]
        trend_bot = TrendBot(symbol=symbol.replace("/", "-"))
        _, trend_stats = trend_bot.run_backtest(df_ann.copy(), tp, BASE_CAPITAL, trend_preset, supervisor=sup)
        
        # 3. RangeBot (Grid)
        rp = RANGE_PRESETS["grid_v1"]
        range_bot = RangeBot(symbol=symbol.replace("/", "-"))
        _, range_stats = range_bot.run_backtest(df_ann.copy(), rp, BASE_CAPITAL, "grid_v1", supervisor=sup)
        
        # 4. RecoveryBot (DCB)
        rec_p = RECOVERY_PRESETS["dcb_v2_optimized"]
        rec_bot = RecoveryBot(symbol=symbol.replace("/", "-"))
        _, rec_stats = rec_bot.run_backtest(df_ann.copy(), rec_p, BASE_CAPITAL, "dcb_v2_optimized", supervisor=sup)
        
        # 5. HedgeBot (Futures)
        hp = HEDGE_PRESETS["hedge_v2_optimized"]
        hedge_bot = HedgeBot(symbol=symbol.replace("/", "-"))
        _, hedge_stats = hedge_bot.run_backtest(df_ann.copy(), hp, BASE_CAPITAL, "hedge_v2_optimized", supervisor=sup)
        
        # --- Totals ---
        cpnl = corr_stats.get("realized_pnl", 0.0)
        tpnl = trend_stats.get("realized_pnl", 0.0)
        rpnl = range_stats.get("realized_pnl", 0.0)
        recpnl = rec_stats.get("realized_pnl", 0.0)
        hpnl = hedge_stats.get("realized_pnl", 0.0)
        
        combined = cpnl + tpnl + rpnl + recpnl + hpnl
        
        return {
            "label": cy["label"],
            "conviction": sup.conviction_score,
            "scaled_capital": scaled_capital,
            "corr_pnl": cpnl,
            "trend_pnl": tpnl,
            "range_pnl": rpnl,
            "rec_pnl": recpnl,
            "hedge_pnl": hpnl,
            "combined_pnl": combined,
        }
    except Exception as e:
        return {"label": cy["label"], "error": str(e)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conviction", type=float, default=1.0)
    args = parser.parse_args()

    print(f"ETH Integration Test v2 — MacroSupervisor {SUPERVISOR_VERSION} (Orchestrator Edition)")
    print("=" * 100)
    print(f"{'Cycle':<18} {'Convict':>8} {'Capital':>8} {'Corr':>8} {'Trend':>8} {'Range':>8} {'Recov':>8} {'Hedge':>8} {'TOTAL':>10}")
    print("-" * 100)
    
    total_all = 0.0
    for cy in CYCLE_PAIRS:
        # Simulate different conviction levels for specific cycles to test scaling
        # In CyA (Luna), we simulate high conviction. In CyB (Choppy), we simulate caution.
        conv = args.conviction
        if "CyB" in cy["label"]: conv = 0.4 # Simulate extreme caution for choppy Mar23
        
        r = run_cycle(cy, "ETH/USD", "correction_v1", "trendbot_v1", conviction_override=conv)
        if "error" in r:
            print(f"{r['label']:<18} ERROR: {r['error']}")
            continue
            
        print(f"{r['label']:<18} {r['conviction']:>8.1f} ${r['scaled_capital']:>7.0f} "
              f"{r['corr_pnl']:>+8.2f} {r['trend_pnl']:>+8.2f} {r['range_pnl']:>+8.2f} "
              f"{r['rec_pnl']:>+8.2f} {r['hedge_pnl']:>+8.2f} ${r['combined_pnl']:>9.2f}")
        total_all += r["combined_pnl"]
        
    print("-" * 100)
    print(f"{'GRAND TOTAL':<18} {'':>8} {'':>8} {'':>8} {'':>8} {'':>8} {'':>8} {'':>8} ${total_all:>9.2f}")

if __name__ == "__main__":
    main()
