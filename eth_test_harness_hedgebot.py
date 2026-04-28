#!/usr/bin/env python3
import os, sys, warnings
import pandas as pd
from datetime import datetime, timezone

# Suppress pandas warnings
warnings.filterwarnings("ignore")

# Ensure local imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eth_helpers import fetch_ohlcv
from eth_macrosupervisor_v30 import MacroSupervisor
from eth_hedgebot_v1 import HedgeBot, PRESETS

# The 4 primary cycles from our integration test
TEST_WINDOWS = [
    {"label": "CyA May-Jun22", "start": "2022-05-01", "end": "2022-07-31"},
    {"label": "CyB Feb-Mar23", "start": "2023-02-01", "end": "2023-04-30"},
    {"label": "CyC Apr-May24", "start": "2024-04-01", "end": "2024-06-30"},
    {"label": "CyD Feb-Mar25", "start": "2025-02-01", "end": "2025-05-30"},
]

def run_window(symbol, win, capital, preset_name):
    start_dt = datetime.strptime(win["start"], "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt   = datetime.strptime(win["end"], "%Y-%m-%d").replace(tzinfo=timezone.utc)
    
    # 1. Fetch Data
    df1h = fetch_ohlcv(symbol, "1h", start_dt, end_dt)
    df5  = fetch_ohlcv(symbol, "5m", start_dt, end_dt)
    
    # 2. Apply MacroSupervisor
    sup = MacroSupervisor()
    df_ann = sup.apply_to_df(df5, df1h)
    
    # 3. Run HedgeBot
    bot = HedgeBot(symbol=symbol.replace("/", "-"))
    params = PRESETS[preset_name]
    tdf, stats = bot.run_backtest(df_ann, params, capital, preset_name)
    
    return tdf, stats

def main():
    symbol = "ETH/USD"
    capital = 1000.0
    preset = "hedge_v1"
    
    print(f"HedgeBot Regression Test (Preset: {preset})")
    print("=" * 60)
    print(f"{'Window':<18} {'Trades':>8} {'WR':>8} {'PnL':>10}")
    print("-" * 60)
    
    total_pnl = 0.0
    for win in TEST_WINDOWS:
        tdf, stats = run_window(symbol, win, capital, preset)
        pnl = stats["realized_pnl"]
        wr  = stats["win_rate"]
        trd = stats["trades"]
        total_pnl += pnl
        
        print(f"{win['label']:<18} {trd:>8} {wr:>8.1%} ${pnl:>9.2f}")
        
    print("-" * 60)
    print(f"{'TOTAL':<18} {'':>8} {'':>8} ${total_pnl:>9.2f}")

if __name__ == "__main__":
    main()
