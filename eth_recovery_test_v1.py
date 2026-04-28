#!/usr/bin/env python3
import os
import shutil
from eth_trendbot_v1 import TrendBot
from eth_bot_interface import Position, Lot

def test_recovery():
    print("ETH Bot Recovery Test")
    print("=====================")
    
    # 1. Setup a bot with some mock state
    bot = TrendBot(symbol="ETH-USD")
    bot._cash = 500.0
    bot._capital = 1000.0
    bot._position.qty = 0.2
    bot._position.avg_entry = 2500.0
    bot._position.lots.append(Lot(qty=0.2, price=2500.0, fee=1.5, ts="2026-04-27 12:00:00", row_idx=0))
    
    print(f"Initial State: Cash=${bot._cash}, Qty={bot._position.qty}")
    
    # 2. Save state
    bot.save_to_disk()
    print("State saved to disk.")
    
    # 3. Create a NEW bot instance (simulating restart)
    new_bot = TrendBot(symbol="ETH-USD")
    print(f"New Bot (Before Load): Cash=${new_bot._cash}, Qty={new_bot._position.qty}")
    
    # 4. Load state
    if new_bot.load_from_disk():
        print(f"New Bot (After Load): Cash=${new_bot._cash}, Qty={new_bot._position.qty}")
        
        # Verify
        assert new_bot._cash == 500.0
        assert new_bot._position.qty == 0.2
        assert len(new_bot._position.lots) == 1
        print("\nSUCCESS: Bot re-hydrated state correctly!")
    else:
        print("\nFAILURE: Bot failed to load state.")

if __name__ == "__main__":
    test_recovery()
