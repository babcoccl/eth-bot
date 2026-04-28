#!/usr/bin/env python3
import os, sys, warnings
import pandas as pd
from datetime import datetime, timezone
import itertools

# Suppress pandas warnings
warnings.filterwarnings("ignore")

# Ensure local imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from eth_helpers import fetch_ohlcv
from eth_macrosupervisor_v30 import MacroSupervisor
from eth_hedgebot_v1 import HedgeBot

TEST_WINDOWS = [
    {"label": "CyA May-Jun22", "start": "2022-05-01", "end": "2022-07-31"},
    {"label": "CyB Feb-Mar23", "start": "2023-02-01", "end": "2023-04-30"},
    {"label": "CyC Apr-May24", "start": "2024-04-01", "end": "2024-06-30"},
    {"label": "CyD Feb-Mar25", "start": "2025-02-01", "end": "2025-05-30"},
]

def main():
    symbol = "ETH/USD"
    capital = 1000.0
    
    # 1. Pre-fetch all data to speed up sweep
    print("Pre-fetching data for all cycles...")
    sup = MacroSupervisor()
    cycle_data = []
    for win in TEST_WINDOWS:
        start_dt = datetime.strptime(win["start"], "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end_dt   = datetime.strptime(win["end"], "%Y-%m-%d").replace(tzinfo=timezone.utc)
        df1h = fetch_ohlcv(symbol, "1h", start_dt, end_dt)
        df5  = fetch_ohlcv(symbol, "5m", start_dt, end_dt)
        df_ann = sup.apply_to_df(df5, df1h)
        cycle_data.append({"label": win["label"], "df": df_ann})

    # 2. Define Sweep Space
    hedge_ratios = [0.3, 0.5, 0.8, 1.0]
    leverages    = [1.0, 1.5, 2.0, 3.0]
    dwell_bars   = [0, 6, 12, 18, 24] # 5m bars: 0h, 30m, 1h, 1.5h, 2h
    
    combinations = list(itertools.product(hedge_ratios, leverages, dwell_bars))
    print(f"Starting sweep of {len(combinations)} combinations...")
    
    results = []
    
    for hr, lev, db in combinations:
        params = {
            "hedge_ratio": hr,
            "leverage": lev,
            "exit_dwell_bars": db,
            "fee_pct": 0.0006
        }
        
        total_pnl = 0.0
        cycle_pnls = {}
        
        bot = HedgeBot(symbol=symbol.replace("/", "-"))
        for cy in cycle_data:
            _, stats = bot.run_backtest(cy["df"], params, capital)
            pnl = stats["realized_pnl"]
            total_pnl += pnl
            cycle_pnls[cy["label"]] = pnl
            
        results.append({
            "hedge_ratio": hr,
            "leverage": lev,
            "exit_dwell_bars": db,
            "total_pnl": total_pnl,
            **cycle_pnls
        })
        
    # 3. Analyze Results
    res_df = pd.DataFrame(results)
    res_df = res_df.sort_values("total_pnl", ascending=False)
    
    print("\nTop 10 Parameter Combinations:")
    print(res_df.head(10).to_string(index=False))
    
    best = res_df.iloc[0]
    print(f"\nRecommended Preset: hedge_v2_optimized")
    print(f"hedge_ratio: {best['hedge_ratio']}")
    print(f"leverage:    {best['leverage']}")
    print(f"exit_dwell_bars: {best['exit_dwell_bars']}")
    print(f"Total PnL:   ${best['total_pnl']:.2f}")
    
    res_df.to_csv("sweep_hedgebot_results.csv", index=False)
    print("\nFull results saved to sweep_hedgebot_results.csv")

if __name__ == "__main__":
    main()
