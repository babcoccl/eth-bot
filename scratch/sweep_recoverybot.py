import os, sys, warnings
import pandas as pd
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

from eth_test_harness_recoverybot import APPROVED_WINDOWS, TIER3_WINDOWS, run_window

def sweep():
    print("RecoveryBot Parameter Sweep")
    print("===========================")
    
    # Sweep Grid
    lookbacks = [12*24*1, 12*24*2] # 1, 2 days
    min_drops = [0.03, 0.05]
    vol_maxes = [0.85, 0.90]
    fib_highs = [0.500]
    
    combos = list(product(lookbacks, min_drops, vol_maxes, fib_highs))
    print(f"Total combinations: {len(combos)}")
    
    windows = APPROVED_WINDOWS
    capital = 1000.0
    symbol = "ETH/USD"
    
    results = []
    
    for l, d, v, f in combos:
        preset_name = f"L{l//288}d_D{int(d*100)}_V{int(v*100)}_F{int(f*100)}"
        preset = {
            "base_qty":           0.05,
            "lookback_bars":      l,
            "min_drop_pct":       d,
            "vol_ratio_max":      v,
            "fib_entry_low":      0.382,
            "fib_entry_high":     f,
            "fib_stop":           0.786,
            "max_hold_bars":      12 * 24 * 2, # 2 days
            "fee_pct":            0.00065,
        }
        
        # Inject preset into module dynamically
        import eth_recoverybot_v1
        eth_recoverybot_v1.PRESETS["sweep"] = preset
        
        # Run across all windows
        win_results = []
        for w in windows:
            try:
                _, tdf, s = run_window(symbol, w, capital, "sweep")
                if s:
                    win_results.append(s)
            except Exception as e:
                pass
                
        if not win_results:
            continue
            
        total_pnl = sum(s.get("realized_pnl", 0) for s in win_results)
        total_trades = sum(s.get("trades", 0) for s in win_results)
        total_psl = sum(s.get("psl_fires", 0) for s in win_results)
        total_tgt = sum(s.get("target_fires", 0) for s in win_results)
        avg_wr = sum(s.get("win_rate", 0) * s.get("trades", 0) for s in win_results) / total_trades if total_trades > 0 else 0
        
        results.append({
            "preset": preset_name,
            "lookback_d": l//288,
            "drop_pct": d,
            "vol_max": v,
            "fib_high": f,
            "trades": total_trades,
            "win_rate": avg_wr,
            "tgt": total_tgt,
            "psl": total_psl,
            "pnl": total_pnl
        })
        
    df_res = pd.DataFrame(results)
    df_res = df_res.sort_values(by="pnl", ascending=False)
    
    print("\nTop 10 Combinations by PnL:")
    print(df_res.head(10).to_string(index=False))
    
    # Save to csv
    df_res.to_csv("sweep_recoverybot_results.csv", index=False)
    print("\nResults saved to sweep_recoverybot_results.csv")

if __name__ == "__main__":
    sweep()
