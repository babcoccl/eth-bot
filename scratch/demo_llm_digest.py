#!/usr/bin/env python3
import sys
import os
import pandas as pd
from datetime import datetime

# Add root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eth_macrosupervisor_v31 import MacroSupervisor
from eth_trendbot_v1 import TrendBot
from eth_rangebot_v4 import RangeBot

def main():
    print("Generating Sample LLM Performance Digest...")
    
    # 1. Setup Mock State
    sup = MacroSupervisor()
    sup.current_regime = "BULL"
    sup.conviction_score = 0.8
    sup.advisor_notes = "Market structure looks strong; seeing RSI divergence on H1. Bullish bias remains."
    
    # 2. Setup Mock Bots
    tbot = TrendBot(symbol="ETH-USD")
    rbot = RangeBot(symbol="ETH-USD")
    
    # Inject some mock trades for the digest
    tbot._trades = [
        {"side": "BUY", "price": 2400.0, "reason": "uptrend_pb", "pnl_after_fees": 0.0},
        {"side": "SELL", "price": 2500.0, "reason": "target", "pnl_after_fees": 100.0},
    ]
    tbot._realized_pnl = 100.0
    tbot._cash = 5000.0
    
    rbot._trades = [
        {"side": "BUY", "price": 2350.0, "reason": "grid_buy", "pnl_after_fees": 0.0},
        {"side": "SELL", "price": 2360.0, "reason": "grid_sell", "pnl_after_fees": 10.0},
    ]
    rbot._realized_pnl = 10.0
    rbot._cash = 2000.0
    
    # 3. Simulate Risk Events
    sup._risk_events.append({
        "ts": "2026-04-28T07:10:00",
        "bot_id": "trendbot_eth_usd",
        "requested": 1000.0,
        "allowed": 800.0,
        "reason": "Low Conviction"
    })
    
    # 4. Generate Digest
    digest = sup.generate_llm_digest([tbot, rbot])
    
    # Save to a file for the user to see
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_llm_digest.md")
    with open(output_path, "w") as f:
        f.write(digest)
    
    print(f"Digest generated at: {output_path}")
    print("-" * 50)
    print(digest)

if __name__ == "__main__":
    main()
